"""双目联合 BA:内参 + 公共外参 + 逐对位姿联合精化(03-05 gap-closure 引擎)。

移植声明:参数布局与「内参+外参联合」思想借鉴参考工程 2
core/multibundle.py 的 IntrinsicJointBA(L208 起,只读引用,绝不修改);
本工程 2 相机特化 + Tikhonov 先验为先验正则增量(参考工程没有先验防线)。
为什么必须联合:03-02/03-04 已定性——两路独立标定的内参在双目精度下互不
兼容,刚性外参 + 固定内参的 SC3 距离残差 16.93mm(阈 5mm);而全内参无约束
放开始联合 descent 不收敛、内参持续漂移,故走结构化变体(fc / fc_prior /
full_prior)+ Tikhonov 先验防漂移,先裁决后落地。

θ 布局(与 stereo/solve.py 不同,不共用装箱/行号函数,两处各自测试锁定):
    [K1 (fx,fy,cx,cy) 4 | D1 5 | K2 4 | D2 5 | 公共外参 6 | 逐对位姿 6×N]
变体决定 D 块是否进 θ:fc / fc_prior 冻结 D(不进 θ,解包原样回填初值),
full_prior 全放开。cam1 恒为单位位姿 (I, 0),是参考系本身,不进 θ。
先验行(乘 prior_weight 追加在重投影残差之后):fc_prior 8 行 = 两路 f/c
相对偏差 (θ−θ0)/θ0;full_prior 18 行 = 上述 8 行 + 两路 D 绝对偏差 (θ−θ0);
fc 无先验行。物点来自 build_object_points 的公制方格(25mm),平移尺度
毫米由数据本身保证,不引入任何尺度参数。

单一数学源纪律:残差与雅可比只经 intrinsic.models(project /
project_jacobian / project_point_jacobian / transform_to_camera)与
cv2.Rodrigues 第二返回值,本文件不出现第二份投影/畸变/旋转公式(源码
防线测试锁定)。WR-02 声明:jac 为 callable 时 scipy 会静默忽略稀疏性
结构参数(死计算)——稀疏性只经「解析 jac 返回 csr + tr_solver=lsmr」
这一条真实机制,绝不传稀疏性结构矩阵参数(源码防线测试锁定)。

畸变模型边界(2026-10-24 增补):联合 BA 当前仅支持 M2——θ 的 D 块
布局(D1 5 | D2 5)按五参定死,12 参 M8 传进来会被 [4:9] 切片静默截断,
故 refine_joint 入口对 M8 内参 fail-closed 拒收(中文报错);M8 接线
(θ 布局扩展 + 24 键契约)待后续任务落地后才放开。joint.enabled 默认
false,run_stereo 入口另有硬门,此处是第二道防线。

dtype 纪律(Pitfall 7):全函数 float64 进出;OpenCV 调用点装包规则不进
本文件(初值 PnP 住 solve.py)。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix, csr_matrix

from stereo_calib.intrinsic import models
from stereo_calib.intrinsic.solve import _is_practically_converged, _tail_improvement
from stereo_calib.io.config import (
    StereoCalibConfig,
    StereoJointConfig,
    StereoRefineConfig,
)
from stereo_calib.stereo.observation import IntrinsicsView, Pairs
from stereo_calib.stereo.solve import (
    StereoInit,
    StereoResult,
    disambiguate_pairs,
    pnp_median,
)

# 结构化变体白名单(与配置层 _ALLOWED_JOINT_VARIANTS 同集,函数独立调用时自证)
_VARIANTS = ("fc", "fc_prior", "full_prior")


@dataclass(frozen=True)
class _Layout:
    """θ 布局速查(按变体定死;全部偏移按参数个数计,下标即切片边界)。

    fc / fc_prior:θ = [K1 4 | K2 4 | 外参 6 | 位姿 6N](n_intr=8,无 D 块);
    full_prior:   θ = [K1 4 | D1 5 | K2 4 | D2 5 | 外参 6 | 位姿 6N]
                  (n_intr=18)。n_prior = 先验行数(0 / 8 / 18)。
    """

    variant: str
    n_intr: int
    k1_off: int
    d1_off: int | None
    k2_off: int
    d2_off: int | None
    ext_off: int
    pose_off: int
    n_prior: int


def _layout(variant: str) -> _Layout:
    """变体 → θ 布局;白名单外当场中文报错(装箱/解包/工厂共用入口防线)。"""
    if variant == "fc" or variant == "fc_prior":
        return _Layout(variant=variant, n_intr=8, k1_off=0, d1_off=None,
                       k2_off=4, d2_off=None, ext_off=8, pose_off=14,
                       n_prior=0 if variant == "fc" else 8)
    if variant == "full_prior":
        return _Layout(variant=variant, n_intr=18, k1_off=0, d1_off=4,
                       k2_off=9, d2_off=13, ext_off=18, pose_off=24, n_prior=18)
    raise ValueError(
        f"联合 BA 变体不合法:{variant!r}。只支持 "
        f"{' / '.join(_VARIANTS)}(结构化约束防内参无界漂移,全放开无先验"
        f"已被实测证明不收敛)。请检查 stereo.joint.variant 配置。")


def row_index_joint(pair_idx: int, cam_idx: int, point_idx: int,
                    coord: int, n_points: int) -> int:
    """重投影残差行号契约(与 solve.py 同式但独立实现,两处各自测试锁定):
    对 → 路(cam1 先 cam2 后)→ 点 → (u,v) 两行交错。先验行追加在全部
    重投影行之后,行号 = 4·N·P 起连续排列(不进本函数)。"""
    if cam_idx not in (0, 1):
        raise AssertionError(
            f"cam_idx 只能是 0(cam1)或 1(cam2),实际是 {cam_idx!r}。")
    if coord not in (0, 1):
        raise AssertionError(
            f"coord 只能是 0(u 分量)或 1(v 分量),实际是 {coord!r}。")
    return 2 * ((pair_idx * 2 + cam_idx) * n_points + point_idx) + coord


# ---------------------------------------------------------------------------
# θ 装箱 / 解箱(布局不同,绝不与 solve.py 共用;往返一致性测试锁定)
# ---------------------------------------------------------------------------


def _check_pose_args(rvec_rel, t_rel, rvecs_j, tvecs_j) -> tuple[np.ndarray, ...]:
    """位姿参数形状防线:外参各 (3,),逐对位姿各 (N,3) 且 N >= 1。"""
    r = np.asarray(rvec_rel, dtype=np.float64).ravel()
    t = np.asarray(t_rel, dtype=np.float64).ravel()
    rv = np.asarray(rvecs_j, dtype=np.float64)
    tv = np.asarray(tvecs_j, dtype=np.float64)
    if r.shape != (3,) or t.shape != (3,):
        raise AssertionError(
            f"外参形状不合:期望 rvec_rel/t_rel 各 (3,),实际是 {r.shape} / {t.shape}。")
    if rv.ndim != 2 or rv.shape[1] != 3 or rv.shape[0] < 1 or tv.shape != rv.shape:
        raise AssertionError(
            f"rvecs_j/tvecs_j 形状不合:期望各 (对数,3) 且对数 >= 1,实际是 "
            f"{rv.shape} / {tv.shape}。请检查位姿是否按对逐行给出。")
    return r, t, rv, tv


def _check_params9(params, what: str) -> np.ndarray:
    p = np.asarray(params, dtype=np.float64).ravel()
    if p.shape != (9,):
        raise AssertionError(
            f"{what} 形状不合:期望 (9,)(布局 models.INTRINSIC_NAMES),"
            f"实际拍平后是 {p.shape}。")
    return p


def pack_theta_joint(params1: np.ndarray, params2: np.ndarray,
                     rvec_rel: np.ndarray, t_rel: np.ndarray,
                     rvecs_j: np.ndarray, tvecs_j: np.ndarray,
                     variant: str) -> np.ndarray:
    """两路内参 + 外参 + 逐对位姿 → 联合单向量 θ(布局见模块 docstring)。

    fc / fc_prior 变体只装两路 K 块(D 冻结不进 θ);full_prior 装全 18 个
    内参数。cam1 恒为单位位姿,不进 θ。"""
    lay = _layout(variant)
    p1 = _check_params9(params1, "params1")
    p2 = _check_params9(params2, "params2")
    r, t, rv, tv = _check_pose_args(rvec_rel, t_rel, rvecs_j, tvecs_j)
    if lay.d1_off is None:
        intr = np.concatenate([p1[:4], p2[:4]])
    else:
        intr = np.concatenate([p1[:4], p1[4:9], p2[:4], p2[4:9]])
    pieces = [intr, r, t]
    for j in range(rv.shape[0]):
        pieces.append(np.concatenate([rv[j], tv[j]]))
    return np.concatenate(pieces)


def unpack_theta_joint(theta: np.ndarray, params1_init: np.ndarray,
                       params2_init: np.ndarray, variant: str
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                  np.ndarray, np.ndarray]:
    """θ → (params1 (9,), params2 (9,), rvec_rel, t_rel, rvecs_j, tvecs_j)。

    fc / fc_prior 变体的 D 块不进 θ,解包时原样回填初值(冻结语义);
    full_prior 从 θ 取全 18 参。长度对不上任何对数时当场中文报错——装箱与
    解箱必须同一布局。"""
    lay = _layout(variant)
    p1_init = _check_params9(params1_init, "params1_init")
    p2_init = _check_params9(params2_init, "params2_init")
    th = np.asarray(theta, dtype=np.float64).ravel()
    n_intr_total = th.shape[0]
    if n_intr_total < lay.pose_off + 6 or (n_intr_total - lay.pose_off) % 6 != 0:
        raise AssertionError(
            f"θ 向量长度不合:应为 内参{lay.n_intr} + 外参6 + 6×对数"
            f"(变体 {variant}),实际 {n_intr_total}(对不上任何对数)。"
            f"请检查 θ 是否由 pack_theta_joint 按同一变体装出。")
    n_pairs = (n_intr_total - lay.pose_off) // 6
    if lay.d1_off is None:
        p1 = np.concatenate([th[lay.k1_off:lay.k1_off + 4], p1_init[4:9]])
        p2 = np.concatenate([th[lay.k2_off:lay.k2_off + 4], p2_init[4:9]])
    else:
        p1 = np.concatenate([th[lay.k1_off:lay.k1_off + 4],
                             th[lay.d1_off:lay.d1_off + 5]])
        p2 = np.concatenate([th[lay.k2_off:lay.k2_off + 4],
                             th[lay.d2_off:lay.d2_off + 5]])
    rvec_rel = th[lay.ext_off:lay.ext_off + 3].copy()
    t_rel = th[lay.ext_off + 3:lay.ext_off + 6].copy()
    rvecs = np.empty((n_pairs, 3), dtype=np.float64)
    tvecs = np.empty((n_pairs, 3), dtype=np.float64)
    for j in range(n_pairs):
        chunk = th[lay.pose_off + 6 * j: lay.pose_off + 6 * (j + 1)]
        rvecs[j] = chunk[:3]
        tvecs[j] = chunk[3:]
    return p1, p2, rvec_rel, t_rel, rvecs, tvecs


# ---------------------------------------------------------------------------
# 残差 / 解析稀疏雅可比工厂(单一数学源:只经 models 与 Rodrigues 第二返回值)
# ---------------------------------------------------------------------------


def _check_obs(uv1: np.ndarray, uv2: np.ndarray, objp: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u1 = np.asarray(uv1, dtype=np.float64)
    u2 = np.asarray(uv2, dtype=np.float64)
    ob = np.asarray(objp, dtype=np.float64)
    if u1.ndim != 3 or u1.shape[2] != 2 or u2.shape != u1.shape:
        raise ValueError(
            f"联合 BA 观测形状不合:期望 uv1/uv2 各 (对数,角点数,2),实际是 "
            f"{u1.shape} / {u2.shape}。请检查调用方的数据排布。")
    if ob.shape != (u1.shape[1], 3):
        raise ValueError(
            f"联合 BA 物点形状与角点不一致:objp {ob.shape} 对不上角点数 "
            f"{u1.shape[1]}。物点网格必须与角点逐点对齐。")
    return u1, u2, ob


def _prior_spec(variant: str, prior_weight: float, p1_init: np.ndarray,
                p2_init: np.ndarray, lay: _Layout
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """先验行规格:(θ 列号, 中心 θ0 值, 行缩放)——行值 = 缩放×(θ−θ0)。

    fc_prior 8 行:两路 f/c 相对偏差,缩放 = prior_weight / θ0(θ0 是
    Phase 2 初值,分母零会除炸——f/c 恒正,防线断言拦);full_prior 再加
    10 行 D 绝对偏差,缩放 = prior_weight。行序:cam1 f/c → cam2 f/c →
    (full_prior)cam1 D → cam2 D。fc 无先验行,返回空数组。"""
    if lay.n_prior == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty.astype(np.int64), empty, empty
    cols: list[int] = []
    centers: list[float] = []
    scales: list[float] = []
    for off, block_init in ((lay.k1_off, p1_init), (lay.k2_off, p2_init)):
        for k in range(4):
            cols.append(off + k)
            centers.append(float(block_init[k]))
            scales.append(prior_weight / float(block_init[k]))
    if lay.d1_off is not None:
        for d_off, d_init in ((lay.d1_off, p1_init[4:9]),
                              (lay.d2_off, p2_init[4:9])):
            for k in range(5):
                cols.append(d_off + k)
                centers.append(float(d_init[k]))
                scales.append(prior_weight)
    if min(centers[:4]) <= 0 or min(centers[4:8]) <= 0:
        raise AssertionError(
            "先验行中心值出现非正 f/c:焦距/主点必须为正数,否则相对偏差"
            "无定义。请检查 Phase 2 内参初值。")
    return (np.asarray(cols, dtype=np.int64),
            np.asarray(centers, dtype=np.float64),
            np.asarray(scales, dtype=np.float64))


def make_residual_joint_fn(objp: np.ndarray, uv1: np.ndarray, uv2: np.ndarray,
                           params1_init: np.ndarray, params2_init: np.ndarray,
                           variant: str, prior_weight: float):
    """残差函数工厂:callable(θ) → (4·N·P + 先验行数,) 残差向量,float64。

    残差 = 两路重投影(models.transform_to_camera + models.project,内参
    从 θ 现拼 9 参向量)+ 先验行(乘 prior_weight,追加在重投影之后)。
    行序:重投影按 row_index_joint 契约,先验行按 _prior_spec 的固定次序。
    """
    u1, u2, ob = _check_obs(uv1, uv2, objp)
    p1_init = _check_params9(params1_init, "params1_init")
    p2_init = _check_params9(params2_init, "params2_init")
    lay = _layout(variant)
    if lay.n_prior > 0 and not (isinstance(prior_weight, int | float)
                                and prior_weight > 0):
        raise ValueError(
            f"先验权重不合法:期望 > 0 的数(带先验变体),实际是 "
            f"{prior_weight!r}。请检查 stereo.joint.prior_weight 配置。")
    n_pairs, n_points = u1.shape[0], u1.shape[1]
    prior_cols, prior_centers, prior_scales = _prior_spec(
        variant, float(prior_weight), p1_init, p2_init, lay)
    n_reproj = 4 * n_pairs * n_points

    def residual_fn(theta: np.ndarray) -> np.ndarray:
        p1, p2, rvec_rel, t_rel, rvecs_j, tvecs_j = unpack_theta_joint(
            theta, p1_init, p2_init, variant)
        R_rel, _ = cv2.Rodrigues(rvec_rel)
        th = np.asarray(theta, dtype=np.float64).ravel()
        out = np.empty(n_reproj + lay.n_prior, dtype=np.float64)
        for j in range(n_pairs):
            x1, _ = models.transform_to_camera(ob, rvecs_j[j], tvecs_j[j])
            x2 = x1 @ R_rel.T + t_rel  # X_cam2 = R_rel @ X_cam1 + t_rel(行向量式)
            out[j * 4 * n_points: j * 4 * n_points + 2 * n_points] = (
                models.project(x1, p1) - u1[j]).ravel()
            out[j * 4 * n_points + 2 * n_points:(j + 1) * 4 * n_points] = (
                models.project(x2, p2) - u2[j]).ravel()
        if lay.n_prior:
            out[n_reproj:] = prior_scales * (th[prior_cols] - prior_centers)
        return out

    return residual_fn


def _structure_positions_joint(n_pairs: int, n_points: int, variant: str
                               ) -> tuple[np.ndarray, np.ndarray]:
    """全部重投影 (行, 列) 结构位置,vals 展平顺序与之一致(工厂内部契约)。

    每对 5 块,依次:cam1 内参块(2P × n_col1)→ cam1 位姿块(2P × 6)→
    cam2 内参块(2P × n_col2)→ cam2 外参块(2P × 6)→ cam2 位姿块
    (2P × 6)。结构事实:cam1 行碰本路内参列 + 本对位姿列;cam2 行碰
    本路内参列 + 外参 6 列 + 本对位姿列;先验行是对角块,由工厂单独拼。"""
    lay = _layout(variant)
    if lay.d1_off is None:
        cols1_theta = np.arange(lay.k1_off, lay.k1_off + 4)
        cols2_theta = np.arange(lay.k2_off, lay.k2_off + 4)
    else:
        cols1_theta = np.arange(lay.k1_off, lay.k1_off + 9)
        cols2_theta = np.concatenate([np.arange(lay.k2_off, lay.k2_off + 4),
                                      np.arange(lay.d2_off, lay.d2_off + 5)])
    row_cam1 = np.arange(2 * n_points)
    row_cam2 = np.arange(2 * n_points, 4 * n_points)
    col_ext = np.arange(lay.ext_off, lay.ext_off + 6)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    for j in range(n_pairs):
        base = j * 4 * n_points
        col_pose = np.arange(lay.pose_off + 6 * j, lay.pose_off + 6 * (j + 1))
        for row_block, col_block in (
                (row_cam1, np.concatenate([cols1_theta, col_pose])),
                (row_cam2, np.concatenate([cols2_theta, col_ext, col_pose]))):
            k = col_block.shape[0]
            rows.append(np.broadcast_to((base + row_block)[:, None], (2 * n_points, k)))
            cols.append(np.broadcast_to(col_block, (2 * n_points, k)))
    return (np.concatenate([r.ravel() for r in rows]),
            np.concatenate([c.ravel() for c in cols]))


def make_jacobian_joint_fn(objp: np.ndarray, uv1: np.ndarray, uv2: np.ndarray,
                           params1_init: np.ndarray, params2_init: np.ndarray,
                           variant: str, prior_weight: float):
    """解析稀疏雅可比工厂:callable(θ) → scipy.sparse.csr_matrix,
    形状 (4NP + 先验行数, n_intr + 6 + 6N)。

    逐对组装(链式法则,构件只来自 models 与 cv2.Rodrigues 第二返回值,
    严禁手写旋转导数):
    - 内参列:models.project_jacobian (P,2,9) 按变体列切片
      (fc 只取 f/c 四列,full_prior 取全 9 列);
    - cam1 位姿列:project_point_jacobian ∘ transform_to_camera 链式;
    - cam2 外参旋转列:rot_jac 一次 einsum、平移列单位阵,再链 cam2 的
      投影点雅可比;
    - 先验行:对角块,fc_prior 行值导数 = prior_weight/θ0、D 行 =
      prior_weight(常数,工厂期一次算好)。
    WR-02:返回 csr 供 tr_solver=lsmr 消费稀疏性,绝不传稀疏性结构参数。
    数值正确性由 θ 全维 approx_derivative 差分基准锁定(相对差 < 1e-6)。
    """
    u1, _u2, ob = _check_obs(uv1, uv2, objp)
    p1_init = _check_params9(params1_init, "params1_init")
    p2_init = _check_params9(params2_init, "params2_init")
    lay = _layout(variant)
    n_pairs, n_points = u1.shape[0], u1.shape[1]
    prior_cols, _prior_centers, prior_scales = _prior_spec(
        variant, float(prior_weight), p1_init, p2_init, lay)
    n_reproj = 4 * n_pairs * n_points
    shape = (n_reproj + lay.n_prior, lay.pose_off + 6 * n_pairs)
    rows, cols = _structure_positions_joint(n_pairs, n_points, variant)
    if lay.n_prior:
        # 先验行是对角块:行号接在全部重投影行之后,列号即对应 θ 内参列
        rows = np.concatenate([rows, np.arange(n_reproj, n_reproj + lay.n_prior)])
        cols = np.concatenate([cols, prior_cols])
    eye3 = np.eye(3, dtype=np.float64)

    def jacobian_fn(theta: np.ndarray) -> csr_matrix:
        p1, p2, rvec_rel, t_rel, rvecs_j, tvecs_j = unpack_theta_joint(
            theta, p1_init, p2_init, variant)
        R_rel, rot_jac_rel = cv2.Rodrigues(rvec_rel)  # 第二返回值 = 现成雅可比
        vals: list[np.ndarray] = []
        for j in range(n_pairs):
            x1, dxc1 = models.transform_to_camera(
                ob, rvecs_j[j], tvecs_j[j], want_jacobian=True)
            x2 = x1 @ R_rel.T + t_rel
            pj1 = models.project_jacobian(x1, p1)        # (P,2,9)
            pj2 = models.project_jacobian(x2, p2)
            jp1 = models.project_point_jacobian(x1, p1)  # (P,2,3)
            jp2 = models.project_point_jacobian(x2, p2)
            jac1_pose = np.einsum("nij,njk->nik", jp1, dxc1)          # (P,2,6)
            dxc2_pose = np.einsum("ij,njk->nik", R_rel, dxc1)
            jac2_pose = np.einsum("nij,njk->nik", jp2, dxc2_pose)
            dxc2_ext = np.empty((x1.shape[0], 3, 6), dtype=np.float64)
            dxc2_ext[:, :, 0:3] = np.einsum("kij,nj->nik",
                                            rot_jac_rel.reshape(3, 3, 3), x1)
            dxc2_ext[:, :, 3:6] = eye3
            jac2_ext = np.einsum("nij,njk->nik", jp2, dxc2_ext)
            # 展平顺序与 _structure_positions_joint 的行主序三元组一致:
            # 相机块内列是「行内交错」的,必须沿 axis=2 拼成 (P,2,块宽) 再
            # ravel(直接逐块 ravel 会把不同行的值错接到同一行——COO 不报错、
            # 只是梯度接错,数值差分基准当场抓住的坑):
            # cam1 块 = [内参列 | 本对位姿列],cam2 块 = [内参列 | 外参列 | 本对位姿列]
            if lay.d1_off is None:
                block1 = np.concatenate([pj1[:, :, :4], jac1_pose], axis=2)
                block2 = np.concatenate([pj2[:, :, :4], jac2_ext, jac2_pose],
                                        axis=2)
            else:
                block1 = np.concatenate([pj1, jac1_pose], axis=2)
                block2 = np.concatenate([pj2, jac2_ext, jac2_pose], axis=2)
            vals.append(block1.ravel())
            vals.append(block2.ravel())
        if lay.n_prior:
            vals.append(np.broadcast_to(prior_scales, (lay.n_prior,)))
        return coo_matrix((np.concatenate(vals), (rows, cols)),
                          shape=shape).tocsr()

    return jacobian_fn


# ---------------------------------------------------------------------------
# 精化与编排入口
# ---------------------------------------------------------------------------


def _intr_drift(params: np.ndarray, params_init: np.ndarray) -> dict:
    """一路内参的漂移量(fx/fy 相对漂移、cx/cy 绝对像素、D 绝对偏差)。"""
    p = np.asarray(params, dtype=np.float64).ravel()
    p0 = np.asarray(params_init, dtype=np.float64).ravel()
    return {
        "fx_rel": float((p[0] - p0[0]) / p0[0]),
        "fy_rel": float((p[1] - p0[1]) / p0[1]),
        "cx_abs": float(p[2] - p0[2]),
        "cy_abs": float(p[3] - p0[3]),
        "D_abs": [float(v) for v in (p[4:9] - p0[4:9])],
    }


@dataclass(frozen=True)
class JointRefine:
    """联合精化终解(不可变):精化内参 + 外参 + 逐对位姿 + 全部诊断留痕。

    params1/params2 (9,):精化后两路内参(models.INTRINSIC_NAMES 布局);
    R (3,3) / t (3,) / rvec / rotation_angle_deg / baseline_mm:语义同
    StereoResult(X_cam2 = R @ X_cam1 + t,t 单位 mm);
    rvecs1/tvecs1 (N,3):精化后逐对 board→cam1 位姿(行序与输入 pairs
    一一对应);per_pair_rms (N,) 为逐对两路 pooled 重投影 RMS(重投影
    行口径,不含先验行);rms_reproj 为 pooled 口径;intr_drift 是内参
    漂移 dict(两路明细 + max 汇总);solver_meta 留收敛判定,不静默。
    """

    params1: np.ndarray
    params2: np.ndarray
    R: np.ndarray
    t: np.ndarray
    rvec: np.ndarray
    rotation_angle_deg: float
    baseline_mm: float
    rvecs1: np.ndarray
    tvecs1: np.ndarray
    per_pair_rms: np.ndarray
    rms_reproj: float
    intr_drift: dict
    solver_meta: dict


def refine_joint(pairs: Pairs, intr1: IntrinsicsView, intr2: IntrinsicsView,
                 init: StereoInit, joint_cfg: StereoJointConfig,
                 refine_cfg: StereoRefineConfig) -> JointRefine:
    """联合 BA 精化:θ = [内参 | 公共外参 6 | 逐对位姿 6×N] 最小二乘。

    初值 = Phase 2 两路 params + PnP-median pass-2 的逐对位姿与中位外参
    (refine_stereo 同源);least_squares(method 照 refine_cfg、
    tr_solver="lsmr"、x_scale="jac"、loss="linear"、容差照 refine_cfg);
    收敛判定经 cost_trace + _is_practically_converged/_tail_improvement,
    判定依据全进 solver_meta 不静默(T-03-06)。
    畸变模型边界:仅支持 M2;M8(D 8 长/params 12 参)入口 fail-closed
    拒收——θ 的 D 块按五参定死,12 参会被静默截断,绝不带病截断。"""
    uv1 = np.asarray(pairs.uv1, dtype=np.float64)
    uv2 = np.asarray(pairs.uv2, dtype=np.float64)
    objp = np.asarray(pairs.objp, dtype=np.float64)
    n_pairs, n_points = uv1.shape[0], uv1.shape[1]
    # 畸变模型边界第一道检查(先于形状防线):M8 fail-closed 拒收
    for intr, what in ((intr1, "intr1"), (intr2, "intr2")):
        if getattr(intr, "model", "M2") != "M2":
            raise ValueError(
                f"联合 BA 不支持畸变模型 {getattr(intr, 'model', 'M2')}({what}.model):"
                f"θ 的 D 块布局按 M2 五参定死,12 参 M8 会被静默截断(fail-closed "
                f"拦截)。联合 BA 的 M8 接线尚未落地;joint.enabled 默认 false,"
                f"请检查 stereo.joint.enabled 与产线 distortion_model 配置。")
    if (init.rvecs1.shape != (n_pairs, 3) or init.tvecs1.shape != (n_pairs, 3)
            or not all(init.per_pair_ok)):
        raise ValueError(
            f"联合精化初值与观测不一致:初值 {init.rvecs1.shape} 对不上 "
            f"{n_pairs} 个对,或存在 PnP 失败对(位姿初值缺失)。精化必须来自"
            f"同一批对、全部对均有解的 pass-2 结果;失败对应先进剔除通道。")
    p1_init = _check_params9(intr1.params, "intr1.params")
    p2_init = _check_params9(intr2.params, "intr2.params")
    variant = joint_cfg.variant
    _layout(variant)  # 变体白名单防线自证(独立调用时不带病运行)

    theta0 = pack_theta_joint(p1_init, p2_init, init.rvec, init.t,
                              init.rvecs1, init.tvecs1, variant)
    residual_fn = make_residual_joint_fn(objp, uv1, uv2, p1_init, p2_init,
                                         variant, joint_cfg.prior_weight)

    # 每次评估把 0.5·r·r 追加进代价轨迹(先验行一并计入,与 scipy 的代价
    # 定义一致),供实用收敛判定的尾部窗口使用。
    cost_trace: list[float] = []

    def traced_residual(theta: np.ndarray) -> np.ndarray:
        r = residual_fn(theta)
        cost_trace.append(0.5 * float(r @ r))
        return r

    sol = least_squares(
        traced_residual, theta0,
        jac=make_jacobian_joint_fn(objp, uv1, uv2, p1_init, p2_init,
                                   variant, joint_cfg.prior_weight),
        method=refine_cfg.method, tr_solver="lsmr", x_scale="jac",
        loss="linear",
        max_nfev=refine_cfg.max_nfev,
        ftol=refine_cfg.ftol, xtol=refine_cfg.xtol, gtol=refine_cfg.gtol)

    p1, p2, rvec_rel, t_rel, rvecs_j, tvecs_j = unpack_theta_joint(
        sol.x, p1_init, p2_init, variant)
    R_rel, _ = cv2.Rodrigues(rvec_rel)

    # 逐对 RMS 与 pooled 指标:重投影行口径(float64 回代,只经 models),
    # 先验行不计入(它是正则项,不是观测误差)。
    per_pair_rms = np.empty(n_pairs, dtype=np.float64)
    sq_total = 0.0
    for j in range(n_pairs):
        x1, _ = models.transform_to_camera(objp, rvecs_j[j], tvecs_j[j])
        x2 = x1 @ R_rel.T + t_rel
        d1 = models.project(x1, p1) - uv1[j]
        d2 = models.project(x2, p2) - uv2[j]
        sq_points = np.sum(d1 * d1, axis=1) + np.sum(d2 * d2, axis=1)
        per_pair_rms[j] = float(np.sqrt(sq_points.mean()))
        sq_total += float(sq_points.sum())
    rms_reproj = float(np.sqrt(sq_total / (2 * n_pairs * n_points)))

    drift = {"cam1": _intr_drift(p1, p1_init), "cam2": _intr_drift(p2, p2_init)}
    drift["max_f_rel"] = max(abs(drift["cam1"]["fx_rel"]),
                             abs(drift["cam1"]["fy_rel"]),
                             abs(drift["cam2"]["fx_rel"]),
                             abs(drift["cam2"]["fy_rel"]))
    drift["max_c_abs"] = max(abs(drift["cam1"]["cx_abs"]),
                             abs(drift["cam1"]["cy_abs"]),
                             abs(drift["cam2"]["cx_abs"]),
                             abs(drift["cam2"]["cy_abs"]))
    drift["max_D_abs"] = max(max(abs(v) for v in drift["cam1"]["D_abs"]),
                             max(abs(v) for v in drift["cam2"]["D_abs"]))

    tail = _tail_improvement(cost_trace)
    meta = {
        "path": "stereo_joint_refine", "variant": variant,
        "prior_weight": float(joint_cfg.prior_weight),
        "method": refine_cfg.method,
        "tr_solver": "lsmr", "x_scale": "jac", "loss": "linear",
        "status": int(sol.status), "nfev": int(sol.nfev), "njev": int(sol.njev),
        "final_cost": float(sol.cost), "tail_improvement": float(tail),
        "n_cost_evals": len(cost_trace),
        "converged": _is_practically_converged(int(sol.status), tail),
        "jacobian": "analytic-jac, no sparsity",
        "n_theta": int(sol.x.shape[0]),
    }
    return JointRefine(
        params1=p1, params2=p2,
        R=np.asarray(R_rel, dtype=np.float64), t=np.asarray(t_rel, dtype=np.float64),
        rvec=np.asarray(rvec_rel, dtype=np.float64),
        rotation_angle_deg=float(np.degrees(np.linalg.norm(rvec_rel))),
        baseline_mm=float(np.linalg.norm(t_rel)),
        rvecs1=rvecs_j, tvecs1=tvecs_j,
        per_pair_rms=per_pair_rms, rms_reproj=rms_reproj,
        intr_drift=drift, solver_meta=meta)


@dataclass(frozen=True)
class JointSolveOutput:
    """联合求解终解(不可变):rigid 契约的 StereoResult + 精化内参 + 元数据。

    result:solver 如实标 "scipy_joint",flip_log 结构与 solve_stereo 一致
    (顶层固定含 "train" 键);params1/params2 是精化后两路内参(评估链
    与 JSON 快照的精化 K/D 来源);joint_meta 含 variant/prior_weight/
    converged/intr_drift/spread_ratio 等裁决与对账信息,不静默(T-03-06)。"""

    result: StereoResult
    params1: np.ndarray
    params2: np.ndarray
    joint_meta: dict


def solve_stereo_joint(pairs: Pairs, intr1: IntrinsicsView,
                       intr2: IntrinsicsView,
                       cfg: StereoCalibConfig) -> JointSolveOutput:
    """双目联合求解编排入口(03-03 编排在 joint.enabled 时消费)。

    两遍法照 solve_stereo(pass-1 关防线 → 消歧 → pass-2),但 pass-2 离散度
    防线一律不启用——逐对位姿只作联合精化的初值、将被重估,防线语义由
    精化后的 joint 散布防线接替(逐对残差 max/median 比值,fail-closed)。
    精化后执行散布防线:max(per_pair_rms)/median(per_pair_rms) 超过
    joint_cfg.spread_ratio_max → 中文 ValueError(防翻转帧/孪生解污染的
    联合解带病出链)。"""
    stereo = cfg.stereo
    joint_cfg = stereo.joint
    pass1 = pnp_median(pairs, intr1, intr2, None)
    pairs2, flip = disambiguate_pairs(pairs, intr1, intr2, pass1, stereo.flip_ratio)
    pass2 = pnp_median(pairs2, intr1, intr2, None)  # 防线移交 joint 散布防线
    flip_log = {"train": flip}

    if not all(pass2.per_pair_ok):
        failed = [gid for gid, ok in zip(pairs.image_ids, pass2.per_pair_ok) if not ok]
        raise ValueError(
            f"双目联合求解失败:{len(failed)} 个对在消歧后仍解不出 PnP 位姿"
            f"(group_id:{failed})——精化没有位姿初值,硬算必错。请把它们"
            f"加入 stereo.exclude_pairs 手动剔除清单,或检查角点数据质量。")

    refined = refine_joint(pairs2, intr1, intr2, pass2, joint_cfg, stereo.refine)

    finite = refined.per_pair_rms[np.isfinite(refined.per_pair_rms)]
    if finite.size != refined.per_pair_rms.size or finite.size == 0:
        raise ValueError(
            "联合精化后逐对 RMS 出现非数值:优化过程产生异常解(位姿跑到"
            "板点 Z<=0 的盆地里),请检查角点数据质量与剔除配置。")
    spread_ratio = float(finite.max() / max(float(np.median(finite)), 1e-30))
    if spread_ratio > joint_cfg.spread_ratio_max:
        raise ValueError(
            f"联合解散布防线触发:精化后逐对重投影 RMS 的 max/median 实测 "
            f"{spread_ratio:.4f},超过配置阈值 {joint_cfg.spread_ratio_max}"
            f"——可能存在翻转帧/孪生解或大偏移坏对污染联合解(逐对位姿已把"
            f"坏对吸收进解里,必须 fail-closed 拦下)。请检查逐对 RMS 诊断,"
            f"把坏对加入 stereo.exclude_pairs。")

    meta = dict(refined.solver_meta, spread_ratio=spread_ratio)
    result = StereoResult(
        R=refined.R, t=refined.t, rvec=refined.rvec,
        rotation_angle_deg=refined.rotation_angle_deg,
        baseline_mm=refined.baseline_mm,
        per_pair_rms=refined.per_pair_rms, rms_reproj=refined.rms_reproj,
        flip_log=flip_log, solver="scipy_joint", solver_meta=meta)
    joint_meta = {
        "variant": joint_cfg.variant,
        "prior_weight": float(joint_cfg.prior_weight),
        "converged": bool(refined.solver_meta["converged"]),
        "intr_drift": refined.intr_drift,
        "spread_ratio": spread_ratio,
        "status": int(refined.solver_meta["status"]),
        "nfev": int(refined.solver_meta["nfev"]),
        "n_pairs": int(refined.per_pair_rms.shape[0]),
    }
    return JointSolveOutput(result=result, params1=refined.params1,
                            params2=refined.params2, joint_meta=joint_meta)
