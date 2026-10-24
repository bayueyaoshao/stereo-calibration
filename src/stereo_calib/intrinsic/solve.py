"""标定求解引擎:OpenCV 初值 + scipy 精化(模块2 引擎切片,INTR-01)。

移植自参考工程 calib/solve/opencv.py、calib/solve/scipy_solver.py、
calib/solve/residuals.py(只读引用,绝不修改)。裁剪声明:本工程不做
参考工程的多模型分派/注册机制,畸变模型按配置走 M2(九参,flags=0)
或 M8(rational 8,flags=CALIB_RATIONAL_MODEL)双布局,模型名只从
config 传入(2026-10-24 增补);算法与调用形态照搬参考工程已验证版本。

结构(单一数学源纪律,RESEARCH Pattern 1/3):
- OpenCV 路径 solve_opencv:标定只当黑盒初始化器,float32 装包规则隔离
  在本文件调用处(绑定层只收 32 位,进 64 位会当场拒收);单遍拟合
  (fresh,M2/M8 同构,flags 按模型分派);
- scipy 路径 refine_scipy:theta 联合装箱 + 解析稀疏雅可比 + trf/lsmr;
  残差与雅可比只准 import intrinsic.models 的投影/变换函数,本文件不
  出现第二份投影或畸变公式(源码防线测试锁定);
- 编排入口 solve_calibration:初值 →(配置开关)精化 →(仅 M8)洗盆,
  02-03 唯一消费它。

M8 洗盆(2026-10-24 追加轮,真实数据逐级探针实证):fresh rational 初值
在本数据上常落垃圾盆地(cam1 114 帧 fx≈5892 / cam2 152 帧 fx≈8908,D 爆
炸~500),scipy 精化把它拉回 ~2040 邻域,但 cam2 恒落 fx-D 简并谷(D 系数
爆炸互相抵消 [k2≈75,k5≈77,k6≈71],fx 2034.88 偏离锚点 2037,残差带结构
A1=0.028px——局部极小、RMS 与好盆地相当,标量指标看不出)。修法:精化
之后再做一遍 cv2 拟合(flags=RATIONAL|USE_INTRINSIC_GUESS,guess=精化解)
——从谷解 guess 确定性洗回好盆地(实测 cam2 fx 2034.88→2037.53、体检
A1 0.028→0.0011;好盆地再拟合不变坏)。

为什么洗盆在精化之后而非 fresh 之后(与追加任务书字面「solve_opencv 做
两遍拟合」的位置偏差,生产实测归因):fresh 解是远离真解的垃圾(fx 5892/
8908),以它作 guess 的第二遍 cv2 拟合洗不动(cam2 实测仍 fx 8256 垃圾,
逐帧中位 4.24px);已验证可洗的输入「2034.88+爆炸系数」恰是精化后的谷解。
故洗盆消费精化解,放在 solve_calibration 编排位;solve_opencv 保持单遍
(剔除循环评分口径随之与第一轮完全一致)。

指标口径(Pitfall 12):OpenCV 返回的整体 RMS 基于 float32 装包角点,
与 float64 回代差约 1e-4 像素——只进 solver_meta["opencv_rms"] 留对账
参考;一切报告指标只准走 intrinsic.metrics 的 float64 回代尺。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix, csr_matrix

from stereo_calib.intrinsic import models
from stereo_calib.io.config import RefineConfig

# 求解器自身的视图数底线:两三个视角定不了内参(装载层 min_views 默认 10,
# 是编排层的配置下限;这里是硬算底线,低于它直接中文报错不硬算)
_MIN_SOLVE_VIEWS = 2

# 实用收敛判定的尾部窗口(约 10 次残差评估,滤掉被信赖域拒绝的尝试步回弹)
_TAIL_WINDOW = 10
# 尾部改进的"走平"判定线:相对改进小于它就认为代价不再下降
_PLATEAU_TOL = 1e-6


@dataclass(frozen=True)
class SolveResult:
    """一次求解的参数与逐视图位姿(不可变):全部数组 float64。

    params (9,)/(12,) 布局见 models.intrinsic_names(M2/M8);rvecs/tvecs
    (V,3) 与输入视图逐行对齐(行序契约,装载层定死,下游绝不重排);
    solver 如实标注解的来源("opencv" 初值 | "scipy" 精化 |
    "opencv+wash" 精化后洗盆,M8 专属);solver_meta
    留对账与收敛判定依据(status/nfev/final_cost/tail_improvement/
    opencv_rms/distortion_model),不静默(T-02-05)。
    """

    params: np.ndarray
    rvecs: np.ndarray
    tvecs: np.ndarray
    solver: str
    solver_meta: dict


def _as_view_array(uvs) -> np.ndarray:
    """把 (V,P,2) 数组或逐视图 (P,2) 列表统一成 (V,P,2) float64 并断言形状。"""
    arr = np.asarray(uvs, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] != 2 or arr.shape[0] < 1:
        raise ValueError(
            f"角点观测形状不合:期望 (视图数,角点数,2),实际是 {arr.shape}。"
            f"请检查调用方传入的是不是逐视图逐角点的像素坐标(或这类数组的列表)。"
        )
    return arr


def _check_common_inputs(uvs, objp, image_size) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """求解入口的公共形状/取值防线(T-02-04:装载层三重防线之后的再断言)。"""
    uvs_arr = _as_view_array(uvs)
    objp_arr = np.asarray(objp, dtype=np.float64)
    if objp_arr.ndim != 2 or objp_arr.shape[1] != 3:
        raise ValueError(
            f"物点形状不合:期望 (角点数,3),实际是 {objp_arr.shape}。"
            f"板系物点必须逐点给 (x,y,0) 三个数,请检查棋盘物点网格的排布。"
        )
    if uvs_arr.shape[1] != objp_arr.shape[0]:
        raise ValueError(
            f"角点与物点数不一致:角点 {uvs_arr.shape[1]} 个,物点 {objp_arr.shape[0]} 个。"
            f"两者必须逐点对齐,请检查角点文件与棋盘规格是否同一套。"
        )
    size = tuple(image_size)
    if (len(size) != 2 or not all(isinstance(v, (int, np.integer)) or float(v).is_integer()
                                  for v in size) or any(v <= 0 for v in size)):
        raise ValueError(
            f"image_size 不合法:期望 (宽, 高) 两个正整数,实际是 {image_size!r}。"
            f"注意 OpenCV 口径是 (宽, 高),与 numpy 图像形状的 (高, 宽) 相反。"
        )
    return uvs_arr, objp_arr, (int(size[0]), int(size[1]))


def _unpack_opencv_solution(model: str, names, K, D, rvecs, tvecs,
                            n_views: int) -> np.ndarray:
    """OpenCV 解 → params (9,)/(12,) + 位姿 (V,3)×2,解包断言按模型。

    OpenCV 把 D 装成 (1,n) 矩阵返回,先拉平:M2 恰 5 长
    [k1,k2,p1,p2,k3];M8 实测(位置探针,cv2 4.14.0)在 rational flag 下
    返回 14 长(s1..s4/τx,τy 槽位仅在对应 flag 下非零)——断言 len ≥ 8、
    余位全零后取前 8([k1,k2,p1,p2,k3,k4,k5,k6])。两遍拟合复用同一解包
    (单一解包纪律:断言只此一份,两遍解防不了布局陷阱等于没防)。
    """
    D = np.asarray(D, dtype=np.float64).ravel()
    if model == "M2":
        assert D.shape == (5,), (
            f"OpenCV 解包的畸变向量长度是 {D.shape},期望 (5,)(M2 布局 {names})。"
            f"出现此错说明求解调用被改动,请对照参考工程核查。")
    else:
        assert D.shape[0] >= 8, (
            f"OpenCV 解包的畸变向量长度是 {D.shape},M8 布局 {names} 至少要 8 长。"
            f"出现此错说明 rational flag 未生效或求解调用被改动,请核查。")
        if D.shape[0] > 8:
            assert bool(np.allclose(D[8:], 0.0)), (
                f"OpenCV 解包的畸变向量是 {D.shape[0]} 长,但第 8 位之后非全零"
                f"({D[8:]})——出现了薄棱镜/倾斜项(s1..s4/τx,τy),超出 M8 布局。"
                f"请核查 flags 是否混入了 THIN_PRISM/TILTED 类选项。")
        D = D[:8]
    params = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2], *D], dtype=np.float64)
    rvecs_arr = np.asarray([np.asarray(r, dtype=np.float64).ravel() for r in rvecs],
                           dtype=np.float64).reshape(n_views, 3)
    tvecs_arr = np.asarray([np.asarray(t, dtype=np.float64).ravel() for t in tvecs],
                           dtype=np.float64).reshape(n_views, 3)
    return params, rvecs_arr, tvecs_arr


def _solve_opencv_m8_pass2(objs32, imgs32, size, K1: np.ndarray, D1: np.ndarray,
                           rational_flags: int) -> tuple:
    """M8 第二遍拟合:从既有解出发的 guess 再拟合(退化谷洗盆的机器)。

    flags = rational flags | CALIB_USE_INTRINSIC_GUESS,guess = 输入 K/D
    (产线 D 本就 8 长;OpenCV 返回多于 8 位时余位恒零,截前 8 传入)。
    独立成函数:退化谷单测直接驱动它(构造爆炸系数 guess → 洗回好盆地),
    不必让编排层暴露 guess 注入口。失败契约与第一遍同:cv2.error 收口为
    中文 ValueError。注意:洗盆输入须是近真解(精化后)的解——从远离真解
    的垃圾 guess 出发洗不动(真实数据探针实证,见模块 docstring)。
    """
    flags2 = int(rational_flags) | int(cv2.CALIB_USE_INTRINSIC_GUESS)
    try:
        # 第 8 个返回值 per_view 同第一遍:float32 口径逐帧表,不读。
        rms2, K2, D2, rvecs2, tvecs2, _std_int, _std_ext, _per_view = \
            cv2.calibrateCameraExtended(
                objs32, imgs32, size,
                np.array(K1, dtype=np.float64),
                np.asarray(D1, dtype=np.float64).ravel().reshape(-1, 1),
                flags=flags2)
    except cv2.error as e:
        raise ValueError(
            f"OpenCV 标定第二遍拟合失败:{e}。"
            f"第二遍吃既有解作初值,理论上不因初值炸——出现此错说明该解"
            f"病态或数据退化,请检查该批视图的分布。") from e
    return rms2, K2, D2, rvecs2, tvecs2, flags2


def solve_opencv(uvs, objp, image_size, model: str = "M2") -> SolveResult:
    """OpenCV 初值路径:按配置的畸变模型一次估全畸变,返回"初值版"标定结果。

    白话:把角点观测交标定黑盒拿一份能用的起点(内参 + 每视图位姿),
    精度到初值口径为止——后续要么直接消费(精化开关关),要么交给
    refine_scipy 精化。flags 按模型分派:M2 全 5 畸变 flags=0;M8 加
    CALIB_RATIONAL_MODEL(不给该 flag 时 OpenCV 对 8 长布局会静默截断)。
    本函数恒单遍拟合(M8 的洗盆第二遍在精化之后,见 solve_calibration;
    剔除循环的评分口径因此与初值解严格一致)。

    参数:
        uvs:        (V,P,2) 或逐视图 (P,2) 列表,角点像素坐标(行序契约);
        objp:       (P,3) 板系物点(Z=0 平面);
        image_size: (宽, 高) 元组——OpenCV 口径,与 numpy 图像形状的
                    (高, 宽) 相反(Pitfall 2,写反会让 RMS 高得离谱);
        model:      "M2" | "M8"(布局见 models.intrinsic_names,默认 M2)。
    """
    names = models.intrinsic_names(model)
    flags = cv2.CALIB_RATIONAL_MODEL if model == "M8" else 0
    uvs_arr, objp_arr, size = _check_common_inputs(uvs, objp, image_size)
    n_views = uvs_arr.shape[0]
    n_points = uvs_arr.shape[1]
    if n_views < _MIN_SOLVE_VIEWS:
        raise ValueError(
            f"视图数不足以标定:只给了 {n_views} 个视图(下限 {_MIN_SOLVE_VIEWS}),"
            f"image_size={size}。两三个视角定不了内参,硬算必错——请检查角点"
            f"检出率与 exclude_images 清单。"
        )

    # OpenCV 绑定只收 32 位浮点(Point2f/Point3f),传 64 位报"Unsupported
    # format"——显式降精度装包:像素坐标 float32 表达精度 ~1e-4 像素,远
    # 小于 0.5px 断言线,可接受(参考工程同款纪律);回代指标走 float64。
    objs32 = [objp_arr.astype(np.float32).reshape(n_points, 1, 3) for _ in range(n_views)]
    imgs32 = [uv.astype(np.float32).reshape(n_points, 1, 2) for uv in uvs_arr]

    try:
        # 第 8 个返回值 per_view 是 float32 口径的逐帧表,剔除判据与报告
        # 一律不用它(Pitfall 12),下划线接收不读。
        rms, K, D, rvecs, tvecs, _std_int, _std_ext, _per_view = cv2.calibrateCameraExtended(
            objs32, imgs32, size, None, None, flags=flags)
    except cv2.error as e:
        raise ValueError(
            f"OpenCV 标定求解失败:{n_views} 个视图,image_size={size},"
            f"OpenCV 报错——{e}。多半是数据退化(角点近共线/位姿过近),"
            f"请检查该批视图的分布。"
        ) from e

    params, rvecs_arr, tvecs_arr = _unpack_opencv_solution(
        model, names, K, D, rvecs, tvecs, n_views)
    return SolveResult(
        params=params, rvecs=rvecs_arr, tvecs=tvecs_arr, solver="opencv",
        solver_meta={"path": "calibrateCameraExtended", "flags_used": int(flags),
                     "distortion_model": model, "opencv_rms": float(rms)})


# ---------------------------------------------------------------------------
# scipy 精化路径:theta 联合装箱 + row_index 行序契约 + 解析稀疏雅可比
# (移植自参考工程 calib/solve/residuals.py 与 scipy_solver.py,只读引用)
# ---------------------------------------------------------------------------

def row_index(view: int, point: int, uv_axis: int, n_points: int) -> int:
    """残差行号契约:视图 → 点 → (u,v) 两行交错,残差与雅可比共用的唯一行号函数。

    白话:第 view 个视图第 point 个角点的 u 残差住偶数行,v 残差住紧随的
    奇数行。行号算错 = 梯度接错行(Pitfall 4),不报异常、只是收敛奇慢
    或原地不动——所以两侧只准调本函数,不准各自手算下标。
    """
    if uv_axis not in (0, 1):
        raise AssertionError(
            f"uv_axis 只能是 0(u 分量)或 1(v 分量),实际是 {uv_axis!r}。"
            f"残差按 (u,v) 两行交错排列,行号函数只认识这两个分量。"
        )
    return 2 * (view * n_points + point) + uv_axis


def pack_theta(params: np.ndarray, rvecs: np.ndarray, tvecs: np.ndarray) -> np.ndarray:
    """内参 + 逐视图位姿 → 联合单向量 θ = [内参 | 视图 0 的 (ω,t) 6 个 | …]。

    内参块长度按布局分派(M2 九参 / M8 十二参);θ 总长对 6 取模的余数
    即布局指纹(M2 ≡3、M8 ≡0),unpack_theta 按它无损解箱。不做"内参/
    外参分块交替优化"——那等于自研优化器,收敛性质更差且实测收益为零
    (Don't Hand-Roll)。V 视图 → 9/12 + 6V 维 float64。
    """
    arr = np.asarray(params, dtype=np.float64).ravel()
    if arr.shape[0] not in (9, 12):
        raise AssertionError(
            f"参数向量形状不合:期望 (9,)(布局 {models.INTRINSIC_NAMES_M2})或 "
            f"(12,)(布局 {models.INTRINSIC_NAMES_M8}),实际拍平后是 {arr.shape}。"
            f"请按 intrinsic_names(模型名) 的顺序传参,不要按位置猜。"
        )
    rv = np.asarray(rvecs, dtype=np.float64)
    tv = np.asarray(tvecs, dtype=np.float64)
    if rv.ndim != 2 or rv.shape[1] != 3 or rv.shape[0] < 1 or tv.shape != rv.shape:
        raise AssertionError(
            f"rvecs/tvecs 形状不合:期望各 (V,3) 且 V>=1(逐视图位姿),"
            f"实际是 {rv.shape} / {tv.shape}。请检查位姿是否按视图逐行给出。"
        )
    pieces = [arr]
    for i in range(rv.shape[0]):
        pieces.append(np.concatenate([rv[i], tv[i]]))  # 视图 i 的 (ω,t) 6 个连排
    return np.concatenate(pieces)


def unpack_theta(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """θ (9/12+6V,) → (params (9,)/(12,), rvecs (V,3), tvecs (V,3));与 pack_theta 互逆。

    内参块长度按 θ 对 6 取模的余数无损分派(9+6V ≡ 3、12+6V ≡ 0 (mod 6));
    长度对不上任何布局/视图数时直接中文报错——装箱与解箱必须同一布局,
    错位在这里点名,不带病进优化器。
    """
    th = np.asarray(theta, dtype=np.float64).ravel()
    remainder = th.shape[0] % 6
    if remainder == 3:
        n_params = 9    # M2 布局:9 + 6×视图数
    elif remainder == 0:
        n_params = 12   # M8 布局:12 + 6×视图数
    else:
        raise AssertionError(
            f"θ 向量长度不合:长度 {th.shape[0]} 对不上任何布局"
            f"(M2:内参 9+6×视图数;M8:内参 12+6×视图数)。"
            f"请检查 θ 是否由 pack_theta 按同一布局装出。"
        )
    if th.shape[0] < n_params + 6:
        raise AssertionError(
            f"θ 向量长度不合:长度 {th.shape[0]} 对不上任何布局"
            f"(M2:内参 9+6×视图数;M8:内参 12+6×视图数)。"
            f"请检查 θ 是否由 pack_theta 按同一布局装出。"
        )
    n_views = (th.shape[0] - n_params) // 6
    params = th[:n_params].copy()
    rvecs = np.empty((n_views, 3), dtype=np.float64)
    tvecs = np.empty((n_views, 3), dtype=np.float64)
    for i in range(n_views):
        chunk = th[n_params + 6 * i: n_params + 6 * (i + 1)]
        rvecs[i] = chunk[:3]   # 前 3 个 = 旋转向量 ω(Rodrigues 语义)
        tvecs[i] = chunk[3:]   # 后 3 个 = 平移向量 t
    return params, rvecs, tvecs


def make_residual_fn(objp: np.ndarray, uvs: np.ndarray):
    """残差函数工厂:callable(θ) → (2·V·P,) 残差向量,float64。

    残差 = models.project(物点经 models.transform_to_camera 变换到相机
    坐标系) − 实测角点——投影只经模型层(单一数学源,源码防线锁定),
    本文件不出现第二份投影或畸变公式。按视图循环、视图内全 numpy 向量化
    (Pitfall 5:逐点 Python 循环慢一个数量级以上);(n,2) 拉平后天然是
    (u,v) 两行交错,与 row_index 的约定一致。
    """
    objp_arr = np.asarray(objp, dtype=np.float64)
    uvs_arr = _as_view_array(uvs)
    n_views, n_points = uvs_arr.shape[0], uvs_arr.shape[1]

    def residual_fn(theta: np.ndarray) -> np.ndarray:
        params, rvecs, tvecs = unpack_theta(theta)
        out = np.empty(2 * n_views * n_points, dtype=np.float64)
        for i in range(n_views):
            pts_cam, _pose_block = models.transform_to_camera(objp_arr, rvecs[i], tvecs[i])
            out[i * 2 * n_points:(i + 1) * 2 * n_points] = (
                models.project(pts_cam, params) - uvs_arr[i]).ravel()
        return out

    return residual_fn


def _structure_positions(n_views: int, n_points: int,
                         n_params: int) -> tuple[np.ndarray, np.ndarray]:
    """全部 (行, 列) 结构位置,按 [视图0 内参块, 视图0 外参块, 视图1 …] 顺序展平。

    结构事实:每残差行恰碰 n_params 个内参列(M2 九列/M8 十二列)+ 本视图
    6 个外参列,别视图的外参列结构为零。雅可比组装的行列下标由本函数
    唯一给出(残差行序同 row_index 契约,Pitfall 4 的结构性防线)。
    """
    row_template = np.arange(2 * n_points)
    col_int = np.arange(n_params)
    col_pose = np.arange(n_params, n_params + 6)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    for i in range(n_views):
        base = i * 2 * n_points
        rows.append(np.broadcast_to(base + row_template[:, None], (2 * n_points, n_params)))
        cols.append(np.broadcast_to(col_int, (2 * n_points, n_params)))
        rows.append(np.broadcast_to(base + row_template[:, None], (2 * n_points, 6)))
        cols.append(np.broadcast_to(col_pose + 6 * i, (2 * n_points, 6)))
    return (np.concatenate([r.ravel() for r in rows]),
            np.concatenate([c.ravel() for c in cols]))


def make_jacobian_fn(objp: np.ndarray, uvs: np.ndarray, model: str = "M2"):
    """解析稀疏雅可比工厂:callable(θ) → scipy.sparse.csr_matrix,
    形状 (2VP, 9/12+6V)(内参块按 model 布局,M2 九列/M8 十二列)。

    逐视图组装 (2P, n_params+6) 块:
    - 内参列 = models.project_jacobian 的 (n,2,n_params) 直接 reshape——数组
      内存布局天然就是 (u,v) 交错行序,不做任何重排;
    - 外参 6 列 = 链式乘积 models.project_point_jacobian (n,2,3)
      · ∂Xc/∂(ω,t) (n,3,6),einsum 一次算完整个视图。dω 部分由
      models.transform_to_camera 内部经 cv2.Rodrigues 的第二返回值
      (现成雅可比)给出,严禁在本文件手写 ∂Xc/∂ω = −[Xc]× 形式——那只在
      ω=0 成立(Pitfall 7);∂t 部分为单位阵(平移逐分量直接搬)。
    COO 三元组向量化拼接后转 csr(参考实测 0.15s/次 @ 640 视图);数值
    正确性由 θ 全维数值差分基准测试逐项锁定(相对差 < 1e-6,M2/M8 各锁)。
    """
    names = models.intrinsic_names(model)
    objp_arr = np.asarray(objp, dtype=np.float64)
    uvs_arr = _as_view_array(uvs)
    n_views, n_points = uvs_arr.shape[0], uvs_arr.shape[1]
    n_params = len(names)
    shape = (2 * n_views * n_points, n_params + 6 * n_views)
    rows, cols = _structure_positions(n_views, n_points, n_params)

    def jacobian_fn(theta: np.ndarray) -> csr_matrix:
        params, rvecs, tvecs = unpack_theta(theta)
        vals: list[np.ndarray] = []
        for i in range(n_views):
            pts_cam, dxc_d_pose = models.transform_to_camera(
                objp_arr, rvecs[i], tvecs[i], want_jacobian=True)
            jac_int = models.project_jacobian(pts_cam, params).reshape(2 * n_points, n_params)
            jac_pose = np.einsum(
                "nij,njk->nik",                    # (n,2,3)·(n,3,6) → (n,2,6)
                models.project_point_jacobian(pts_cam, params),
                dxc_d_pose).reshape(2 * n_points, 6)
            vals.append(jac_int.ravel())   # 展平顺序与 _structure_positions 一致
            vals.append(jac_pose.ravel())
        return coo_matrix((np.concatenate(vals), (rows, cols)), shape=shape).tocsr()

    return jacobian_fn


def _tail_improvement(cost_trace: list[float]) -> float:
    """代价尾部的相对改进:取"运行最小值包络"(只降不升)后看最后一段降了多少。

    为什么要包络:trf 的尝试步会被信赖域拒绝(那次残差白算、代价回弹),
    原始曲线锯齿状;包络后单调不增,尾部改进才真实反映"优化器还能不能
    压低代价"。评估次数不足 2 次时无从谈趋势,按 0(走平)处理。
    """
    if len(cost_trace) < 2:
        return 0.0
    envelope = np.minimum.accumulate(np.asarray(cost_trace, dtype=np.float64))
    tail_n = min(_TAIL_WINDOW, envelope.shape[0] - 1)
    prev = float(envelope[-1 - tail_n])
    last = float(envelope[-1])
    return (prev - last) / max(abs(prev), 1e-30)


def _is_practically_converged(status: int, tail_improvement: float) -> bool:
    """实用收敛判定(纯函数,判定依据显式传入,真值表由测试锁定)。

    status > 0:优化器自己确认停机条件,直接算收敛;
    status <= -1:优化器报错,绝不判收敛;
    status == 0(到达评估上限或无法改进):代价尾部(最后 10 次评估)相对
    改进 < 1e-6 视为走平收敛。本工程 loss 固定 linear(无鲁棒核把解压歪
    的问题),比参考工程少一条"与 linear 参照比较"的支路。
    """
    if status > 0:
        return True
    if status <= -1:
        return False
    return tail_improvement < _PLATEAU_TOL


def refine_scipy(uvs, objp, image_size, init: SolveResult, refine_cfg: RefineConfig,
                 model: str = "M2") -> SolveResult:
    """scipy 精化路径:OpenCV 初值 → 稀疏最小二乘联合精化(INTR-01)。

    白话:θ = [内参(按 model 布局 9/12)| 每视图 6 个位姿参数] 装成
    **一个**向量,交给 least_squares 的 trf(信赖域反射法)+ lsmr(解稀疏
    线性子问题)+ 解析稀疏雅可比联合优化;不做内参/外参分块交替优化
    (Don't Hand-Roll)。

    失败契约:非收敛**不抛异常**——判定依据全进 solver_meta(T-02-05
    不静默),由编排层消费 converged 标志决定后续;数值路径异常
    (ValueError/ArithmeticError/RuntimeError/AssertionError,含线性代数
    炸掉与 models.project 的 Z>0 数据防卫)如实上抛,由编排层 run_intrinsic
    收口为该路失败记录、CLI 收口为中文一句(两层收口真实存在),不吞成
    静默垃圾解。

    参数:
        uvs / objp / image_size:与 solve_opencv 同一口径;
        init:       OpenCV 初值解(必须与 uvs 视图数一致);
        refine_cfg: 精化开关 + method + max_nfev + 容差,全部配置控制;
        model:      "M2" | "M8"(布局见 models.intrinsic_names,默认 M2)。
    """
    names = models.intrinsic_names(model)
    uvs_arr, objp_arr, size = _check_common_inputs(uvs, objp, image_size)
    n_views = uvs_arr.shape[0]
    if init.params.shape != (len(names),) or init.rvecs.shape != (n_views, 3) \
            or init.tvecs.shape != (n_views, 3):
        raise ValueError(
            f"精化初值与观测不一致:params 形状 {init.params.shape}(期望 "
            f"{len(names)},布局 {names}),rvecs/tvecs "
            f"{init.rvecs.shape}/{init.tvecs.shape},而观测是 {n_views} 个视图"
            f"(image_size={size})。初值必须来自同一批视图、同一模型的 "
            f"solve_opencv 结果。"
        )

    theta0 = pack_theta(init.params, init.rvecs, init.tvecs)
    residual_fn = make_residual_fn(objp_arr, uvs_arr)

    # 每次评估把 0.5·r·r 追加进代价轨迹(与 scipy 的代价定义一致),
    # 供实用收敛判定的尾部窗口使用。
    cost_trace: list[float] = []

    def traced_residual(theta: np.ndarray) -> np.ndarray:
        r = residual_fn(theta)
        cost_trace.append(0.5 * float(r @ r))
        return r

    sol = least_squares(
        traced_residual, theta0,
        jac=make_jacobian_fn(objp_arr, uvs_arr, model),
        # 不传稀疏结构矩阵参数:jac 为 callable 时 scipy 对它静默忽略(传入
        # 即死计算);稀疏性利用只经"callable jac 返回 csr + tr_solver='lsmr'"
        # 这一条真实机制(见 make_jacobian_fn),不靠结构矩阵。
        method=refine_cfg.method, tr_solver="lsmr", x_scale="jac",
        # 离群帧由 02-03 的剔除机制处理,不引入鲁棒核(loss 写死 linear,
        # RESEARCH OQ2 裁决;若未来需要鲁棒核走配置化扩展,不在这里塞分支)
        loss="linear",
        max_nfev=refine_cfg.max_nfev,
        ftol=refine_cfg.ftol, xtol=refine_cfg.xtol, gtol=refine_cfg.gtol)

    params, rvecs, tvecs = unpack_theta(sol.x)
    tail = _tail_improvement(cost_trace)
    meta = {
        "path": "scipy_refine", "method": refine_cfg.method,
        "tr_solver": "lsmr", "x_scale": "jac", "loss": "linear",
        "status": int(sol.status), "nfev": int(sol.nfev), "njev": int(sol.njev),
        "final_cost": float(sol.cost), "tail_improvement": float(tail),
        "n_cost_evals": len(cost_trace),
        "converged": _is_practically_converged(int(sol.status), tail),
        # OpenCV 初值的内部 RMS 随解传递,精化前后对账口径不断链(Pitfall 12)
        "opencv_rms": init.solver_meta.get("opencv_rms"),
    }
    return SolveResult(params=params, rvecs=rvecs, tvecs=tvecs,
                       solver="scipy", solver_meta=meta)


def _wash_m8_valley(uvs, objp, image_size, refined: SolveResult,
                    model: str) -> SolveResult:
    """M8 洗盆:精化解作 guess 的第二遍 cv2 拟合(退化谷确定性洗出)。

    为什么(真实数据逐级探针实证,模块 docstring 有全记录):cam2 的 scipy
    精化恒落 fx-D 简并谷(fx 2034.88、D 爆炸 [k2≈75,k5≈77,k6≈71]、A1 残差
    带 0.028px)——谷是 RMS 相当的局部极小,精化洗不动自己;而以「精化后的
    谷解」作 guess 的 cv2 GUESS 再拟合确定性洗回好盆地(实测 cam2 fx
    2037.53、体检 A1 0.0011,好盆地再拟合不变坏)。fresh 初值是远离真解的
    垃圾(fx 5892/8908),以它作 guess 洗不动——故本遍只吃精化解。

    solver 如实标 "opencv+wash"(最终参数出自 cv2 优化器,非精化器本身);
    精化链对账不静默:精化前 fx、solver 标签与完整精化 solver_meta 都随行。
    """
    uvs_arr, objp_arr, size = _check_common_inputs(uvs, objp, image_size)
    n_views, n_points = uvs_arr.shape[0], uvs_arr.shape[1]
    names = models.intrinsic_names(model)
    objs32 = [objp_arr.astype(np.float32).reshape(n_points, 1, 3) for _ in range(n_views)]
    imgs32 = [uv.astype(np.float32).reshape(n_points, 1, 2) for uv in uvs_arr]
    rms_w, K_w, D_w, rvecs_w, tvecs_w, flags_w = _solve_opencv_m8_pass2(
        objs32, imgs32, size, models.intrinsics_matrix(refined.params),
        refined.params[4:], int(cv2.CALIB_RATIONAL_MODEL))
    params_w, rvecs_arr, tvecs_arr = _unpack_opencv_solution(
        model, names, K_w, D_w, rvecs_w, tvecs_w, n_views)
    meta = {
        "path": "calibrateCameraExtended",
        "flags_used": int(flags_w),
        "distortion_model": model,
        "opencv_rms": float(rms_w),
        "wash_pass": ("M8 洗盆:scipy 精化解作 guess 的第二遍 cv2 拟合"
                      "(RATIONAL|USE_INTRINSIC_GUESS,防 fx-D 简并谷)"),
        "pre_wash_solver": refined.solver,
        "pre_wash_fx": float(refined.params[0]),
        "pre_wash_meta": refined.solver_meta,
    }
    return SolveResult(params=params_w, rvecs=rvecs_arr, tvecs=tvecs_arr,
                       solver="opencv+wash", solver_meta=meta)


def solve_calibration(uvs, objp, image_size, refine_cfg: RefineConfig,
                      model: str = "M2") -> SolveResult:
    """编排入口(02-03 唯一消费它):OpenCV 初值 →(配置开关)scipy 精化
    →(仅 M8)洗盆。

    refine_cfg.enabled=False 时原样返回初值解,solver 如实标 "opencv";
    True 时精化,solver 标 "scipy" 且 solver_meta 保留 opencv_rms 对账;
    M8 再加洗盆第二遍(solver 标 "opencv+wash",meta 带精化链对账)——
    M2 路径零改动(无洗盆,已验证契约)。model("M2"|"M8")透传各路径。
    """
    init = solve_opencv(uvs, objp, image_size, model)
    if not refine_cfg.enabled:
        return init
    refined = refine_scipy(uvs, objp, image_size, init, refine_cfg, model)
    if model != "M8":
        return refined
    return _wash_m8_valley(uvs, objp, image_size, refined, model)
