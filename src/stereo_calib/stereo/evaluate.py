"""双目评估链:消歧 → per-frame 2 视图位姿 BA → 去畸变 → DLT 三角化 → 板系
反变换 → 对角距离残差(STER-03 的全部数学住址)。

移植自参考工程 2 cli/06_evaluate.py(消歧 L180-203、去畸变+三角化装配+
板系反变换 L245-263)与 core/triangulation.py(projection_matrix L24-29、
undist_pix L32-40、DLT 保号除法 L59-85、最远两角点距离 L88-100)(只读
引用,绝不修改)。版本教训(参考 v1→v2 实测,必须带 per-frame BA 这
半步):v1「独立 PnP + DLT」在 1m 处误差 24mm,v2 加 per-frame 2 视图位
姿 BA 后 0.23mm——百倍差距,本模块按 v2 全链实现。

三角化唯一住址(源码防线):SVD 齐次方程求解只准出现在本文件,工程里
任何其他文件不得自行三角化(测试锁定)。分轴对齐诊断(参考
triangulation.py 122-174)不移植——决策 6:MVP 最小指标 = 板系对角距离
残差,参考工程保留可随时补。

两条链路的畸变契约(Pitfall 3,搞反即静默错):
- PnP/BA 链路:喂带畸变原始像素,畸变由内参 params 内部处理(经 models);
- DLT 链路:喂 undist_pix(models.unproject 精确牛顿逆,回乘 K 保持
  【像素】口径,Pitfall 4)去畸变后的观测,配无畸变投影矩阵 P = K[R|t]
  ——线性方程在针孔模型下严格成立;直接喂带畸变点会有板边缘系统性
  偏差(参考注释原话记录)。

失败契约(照 intrinsic/scoring.py 体例):单帧任何数值路径失败返回
NaN + 中文备注,绝不抛异常中断整批;形状/参数契约在入口防线拦;
evaluate_split 全帧失败属环境异常,中文 ValueError(考卷全糊不静默产出
NaN 报告)。

真值纪律:对角距离真值只准 np.linalg.norm(objp[-1] − objp[0]) 从
build_object_points 的物点网格导出(11×8/25mm 即 87 号与 0 号角点,
305.1639mm),任何地方绝不写死该常数——板规格变了真值自动跟随。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares

from stereo_calib.intrinsic import models
from stereo_calib.io.config import BoardSpec
from stereo_calib.preprocess.artifacts import build_object_points
from stereo_calib.stereo.observation import IntrinsicsView, Pairs, solve_view_pose

# 单帧数值异常收口家族(scoring.py 同款):np.linalg.LinAlgError 是
# ValueError 子类一并覆盖;AssertionError 覆盖 models.project 的 Z>0 数据
# 防卫(精化期位姿由优化器控制,试验步可能把板点挪到相机身后)。
_FRAME_EXC = (ValueError, ArithmeticError, RuntimeError, AssertionError, cv2.error)

# 位姿 BA 的迭代上限(6 参数 176 残差的小问题,参考 multiview_pose_ba
# 默认 100,取 200 留裕量;收敛失败不是异常——RMS 如实返回供上游判读)。
_POSE_BA_MAX_NFEV = 200

# solvePnP 迭代解法要求的最少点数(与 scoring.py 同一防线)
_MIN_PNP_POINTS = 4


# ---------------------------------------------------------------------------
# DLT 三角化(唯一住址)+ 去畸变 + 投影矩阵(参考 triangulation.py 移植)
# ---------------------------------------------------------------------------


def projection_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """针孔投影矩阵 P = K @ [R | t],(3, 4) float64(参考 triangulation.py:24-29)。

    只配无畸变观测:DLT 的线性方程在针孔模型下严格成立,带畸变像素必须
    先经 undist_pix 去畸变再进本矩阵(Pitfall 3)。
    """
    Rt = np.zeros((3, 4), dtype=np.float64)
    Rt[:3, :3] = np.asarray(R, dtype=np.float64)
    Rt[:3, 3] = np.asarray(t, dtype=np.float64).ravel()
    return np.asarray(K, dtype=np.float64) @ Rt


def undist_pix(pts2d: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """检测点去畸变到理想针孔像素坐标(供 DLT 三角化;参考 triangulation.py:32-40)。

    白话:把带畸变的检测像素映到「移除镜头畸变后,该点在原始图像像素格
    上的位置」——与无畸变投影矩阵 P = K[R|t] 同一口径,配错口径三角化
    点尺度全错(Pitfall 4),故函数名锁死像素语义,全链路统一像素口径。

    反解走 models.unproject(精确牛顿逆,1e-12px)而不是 cv2 的迭代逆:
    cv2.undistortPoints 是定点迭代,畸变越强收敛越残——本工程畸变域实测
    r²=0.5 处逆误差已达 0.28px(r²=1.0 处 8px),进 DLT 会变成 mm 级
    距离假残差(Rule 1 修复,合成数据当场抓住);models.unproject 是
    scoring.py 同款的精确逆(解析雅可比牛顿迭代到 1e-15),去畸变公式
    只住 models 单一数学源。返回口径不变:归一化坐标乘 K 回【像素】框
    (P=K 契约,不是归一化坐标)。
    """
    k_arr = np.asarray(K, dtype=np.float64)
    d_arr = np.asarray(dist, dtype=np.float64).ravel()
    params = np.concatenate([np.diag(k_arr)[:2], k_arr[:2, 2], d_arr])
    rays = models.unproject(np.asarray(pts2d, dtype=np.float64), params)
    norm = rays[:, :2] / rays[:, 2:3]
    return norm @ k_arr[:2, :2].T + k_arr[:2, 2]


def triangulate_dlt(projections, image_points) -> np.ndarray:
    """多视图 DLT 三角化(2 视图特化;参考 triangulation.py:59-85 原样移植)。

    参数:projections 为 [(3,4)] 无畸变投影矩阵;image_points 为 [(N,2)]
    去畸变针孔像素(undist_pix 产物),逐相机顺序对齐。返回 (N, 3) 三维点,
    坐标系 = projections[0] 的世界系。

    保号除法是命门(Pitfall 5,参考注释原话):SVD 零空间向量符号任意,
    必须除以齐次分量本身;若按绝对值除,X_h[3]<0 时整点取反,三角化点
    跑到相机后方、深度变负,距离/深度全错——本函数与其测试共同锁定该
    纪律(源文不出现绝对值调用,防线测试断言)。
    """
    projs = [np.asarray(p, dtype=np.float64) for p in projections]
    pts = [np.asarray(ip, dtype=np.float64) for ip in image_points]
    if not projs or len(projs) != len(pts):
        raise ValueError(
            f"三角化输入不合法:投影矩阵 {len(projs)} 份对不上观测 {len(pts)} 路,"
            f"每路相机必须一一配对。请检查调用方的数据排布。")
    if projs[0].shape != (3, 4):
        raise AssertionError(
            f"投影矩阵形状不合:期望 (3,4),实际是 {projs[0].shape}。"
            f"请经 projection_matrix 构造,不要手工拼矩阵。")
    n_cams = len(projs)
    n_pts = pts[0].shape[0]
    for i in range(n_cams):
        if pts[i].ndim != 2 or pts[i].shape[1] != 2 or pts[i].shape[0] != n_pts:
            raise AssertionError(
                f"第 {i} 路观测形状不合:期望 ({n_pts},2) 与其余路逐点对齐,"
                f"实际是 {pts[i].shape}。三角化的多路观测必须同一批角点。")
    out = np.zeros((n_pts, 3), dtype=np.float64)
    for k in range(n_pts):
        A = np.zeros((2 * n_cams, 4), dtype=np.float64)
        for i in range(n_cams):
            x, y = pts[i][k]
            A[2 * i] = x * projs[i][2] - projs[i][0]
            A[2 * i + 1] = y * projs[i][2] - projs[i][1]
        _, _, Vt = np.linalg.svd(A)
        X_h = Vt[-1]
        # 保号:阈值只判量级、除法用带符号原值(整点取反坑见 docstring)
        w = X_h[3] if X_h[3] > 1e-12 or X_h[3] < -1e-12 else 1e-12
        out[k] = X_h[:3] / w
    return out


def farthest_corners_distance(points_3d: np.ndarray,
                              objp: np.ndarray) -> tuple[float, int, int]:
    """板系角点里首尾两角点(11×8 即 0 号与 87 号)的三维距离(mm)+ 索引。

    参考 triangulation.py:88-100,索引改为从 objp 形状导出(尾角点 =
    len(objp) − 1,行主序下即 (rows−1, cols−1) 角),棋盘规格变了自动跟随。
    """
    pts = np.asarray(points_3d, dtype=np.float64)
    objp_arr = np.asarray(objp, dtype=np.float64)
    idx_a = 0
    idx_b = int(objp_arr.shape[0]) - 1
    return float(np.linalg.norm(pts[idx_a] - pts[idx_b])), idx_a, idx_b


def true_board_distance(objp: np.ndarray) -> float:
    """对角距离真值 = ‖objp[87] − objp[0]‖(11×8/25mm 即 305.1639mm)。

    真值数字纪律:只准从 build_object_points 产物导出,绝不写死常数——
    板规格变了真值自动跟随(配置单一来源)。
    """
    objp_arr = np.asarray(objp, dtype=np.float64)
    return float(np.linalg.norm(objp_arr[-1] - objp_arr[0]))


# ---------------------------------------------------------------------------
# 180° 消歧(参考 cli/06_evaluate.py:180-203,锚 = 当帧 cam1 PnP)
# ---------------------------------------------------------------------------


def disambiguate_frame(uv1: np.ndarray, uv2: np.ndarray, objp: np.ndarray,
                       intr1: IntrinsicsView, intr2: IntrinsicsView,
                       R_rel: np.ndarray, t_rel: np.ndarray,
                       flip_ratio: float) -> tuple[np.ndarray, np.ndarray, tuple[bool, bool]]:
    """单帧 180° 角点序消歧:证据充分才翻,返回纠正后副本 + 翻转标记。

    白话:棋盘有 180° 旋转对称,两相机视角不同,Phase 1 的单图规范序不
    保证跨相机物理同一性。拿当帧 cam1 的 PnP 位姿当锚(solvePnP 吃带畸变
    原始像素——PnP 链路契约),把物点经锚位姿 + 当前外参预测投到两路,
    比较正序/反序的投影误差;反序误差 < flip_ratio × 正序误差才翻(证据
    充分才翻,Pitfall 11)。cam1 的锚即自身解、数学上恒不翻,照判保对称
    (参考 06 对含锚在内的所有路都消歧,自洽时是无害的恒等操作)。

    锚自洽性(整体翻转帧为何不用翻):若某帧两路角点序整体翻转,cam1 的
    PnP 锚解出的是 180° 转板位姿,与翻转序自洽——消歧判定不再翻,而下游
    指标(板系对角距离)对整体翻转天然不变(角点 0↔87 互换,距离相同),
    自洽性论证由测试锁定。

    返回:(uv1 纠正副本, uv2 纠正副本, (cam1 是否翻转, cam2 是否翻转));
    输入数组绝不被原地修改。PnP 锚失败抛 RuntimeError(由调用方的单帧
    失败收口接住,不中断整批)。
    """
    if not 0 < flip_ratio <= 1.0:
        raise ValueError(
            f"消歧判定阈值不合法:期望 0 到 1.0 之间(含上端点)的数,实际是 "
            f"{flip_ratio!r}。超过 1.0 即证据不足也翻(Pitfall 11)。")
    uv1_arr = np.asarray(uv1, dtype=np.float64)
    uv2_arr = np.asarray(uv2, dtype=np.float64)
    objp_arr = np.asarray(objp, dtype=np.float64)
    if uv1_arr.ndim != 2 or uv1_arr.shape[1] != 2 or uv2_arr.shape != uv1_arr.shape:
        raise AssertionError(
            f"单帧观测形状不合:期望 uv1/uv2 各 (角点数,2),实际是 "
            f"{uv1_arr.shape} / {uv2_arr.shape}。评估链一次处理一个双目对。")

    k1 = np.asarray(intr1.K, dtype=np.float64)
    d1 = np.asarray(intr1.D, dtype=np.float64).ravel()
    R_rel_arr = np.asarray(R_rel, dtype=np.float64)
    t_rel_arr = np.asarray(t_rel, dtype=np.float64).ravel()

    # 锚 = 当帧 cam1 位姿(参考 06 L186-190)。PnP 一律经 observation.solve_view_pose
    # (按模型分派的唯一住址):M2/M8 喂带畸变原始像素,M4 走自家精确反投影 + 切平面。
    ok, rvec1, tvec1 = solve_view_pose(objp_arr, uv1_arr, intr1)
    if not ok:
        raise RuntimeError("消歧锚求解失败:当帧 cam1 PnP 无解,该帧几何可疑")
    R1, _ = cv2.Rodrigues(np.asarray(rvec1, dtype=np.float64).ravel())
    t1 = np.asarray(tvec1, dtype=np.float64).ravel()
    xw = (R1 @ objp_arr.T).T + t1  # 板 → cam1(锚位姿)

    params1 = np.asarray(intr1.params, dtype=np.float64).ravel()
    params2 = np.asarray(intr2.params, dtype=np.float64).ravel()
    out1 = uv1_arr.copy()
    out2 = uv2_arr.copy()
    flipped: list[bool] = []
    # cam2 观测点的 cam2 系坐标 = 外参组合(X_cam2 = R_rel @ X_cam1 + t_rel,
    # 行向量式写法);两路同判,误差别序即翻(参考 06 L191-202)
    for x_cam, params, obs, out in (
            (xw, params1, uv1_arr, out1),
            (xw @ R_rel_arr.T + t_rel_arr, params2, uv2_arr, out2)):
        proj = models.project(x_cam, params)
        err_fwd = float(np.linalg.norm(proj - obs))
        err_rev = float(np.linalg.norm(proj - obs[::-1]))
        if err_rev < flip_ratio * err_fwd:
            out[:] = obs[::-1]
            flipped.append(True)
        else:
            flipped.append(False)
    return out1, out2, (flipped[0], flipped[1])


# ---------------------------------------------------------------------------
# per-frame 2 视图位姿 BA(参考 multibundle.py multiview_pose_ba 的 2 路特化,
# 初值路线照 intrinsic/scoring.py 的反投影模式)
# ---------------------------------------------------------------------------


def frame_pose_ba(uv1: np.ndarray, uv2: np.ndarray, objp: np.ndarray,
                  intr1: IntrinsicsView, intr2: IntrinsicsView,
                  R_rel: np.ndarray, t_rel: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray, float]:
    """单帧 2 视图联合位姿 BA:固定外参与内参,只优化 board→cam1 的 6 参数。

    白话:v1 的「两路各自独立 PnP」各自带噪声与畸变误差,DLT 拿这两份
    噪声位姿当投影矩阵,1m 处实测 24mm;v2 先用两路联合残差把这一帧的
    板位姿精修一遍(外参是刚架约束,两路观测互相拉紧),再拿精修位姿
    配 DLT,实测 0.23mm——这半步是评估链精度的来源。

    初值路线(照 scoring.py,强畸变防错误盆地):models.unproject 反投影
    成单位射线(牛顿迭代天然完成去畸变)→ 单位阵 K + 零畸变 solvePnP 解
    位姿初值;直接对带畸变像素零畸变解会进错误盆地(Phase 2 实测教训)。

    残差只经 models(单一数学源):cam1 残差 + 经外参组合的 cam2 残差,
    least_squares(method="trf", loss="linear"(决策 7:消歧后无离群核需求,
    与全工程一致), x_scale="jac", max_nfev 上限 200)。返回
    (rvec_pose, tvec_pose, pooled_rms),RMS 为两路全部 (u,v) 残差的平方
    均值开根(float64,pooled 口径)。
    """
    uv1_arr = np.asarray(uv1, dtype=np.float64)
    uv2_arr = np.asarray(uv2, dtype=np.float64)
    objp_arr = np.asarray(objp, dtype=np.float64)
    if uv1_arr.ndim != 2 or uv1_arr.shape[1] != 2 or uv2_arr.shape != uv1_arr.shape:
        raise AssertionError(
            f"单帧观测形状不合:期望 uv1/uv2 各 (角点数,2),实际是 "
            f"{uv1_arr.shape} / {uv2_arr.shape}。评估链一次处理一个双目对。")
    if uv1_arr.shape[0] < _MIN_PNP_POINTS:
        raise AssertionError(
            f"角点数 {uv1_arr.shape[0]} 少于位姿求解下限 {_MIN_PNP_POINTS} 个:"
            f"方程组欠定,位姿解不出来。")
    params1 = np.asarray(intr1.params, dtype=np.float64).ravel()
    params2 = np.asarray(intr2.params, dtype=np.float64).ravel()
    R_rel_arr = np.asarray(R_rel, dtype=np.float64)
    t_rel_arr = np.asarray(t_rel, dtype=np.float64).ravel()

    # ---- 初值:cam1 反投影到归一化平面 + 单位阵 K 零畸变 solvePnP ----
    rays = models.unproject(uv1_arr, params1)
    if not bool((rays[:, 2] > 0).all()):
        raise RuntimeError("反投影射线出现 Z 分量非正的角点,归一化平面坐标不存在,"
                           "该帧几何可疑")
    ok, rvec0, tvec0 = cv2.solvePnP(
        objp_arr.astype(np.float32).reshape(-1, 1, 3),
        (rays[:, :2] / rays[:, 2:3]).astype(np.float32).reshape(-1, 1, 2),
        np.eye(3), np.zeros(5))
    if not ok:
        raise RuntimeError("位姿初值求解失败:该帧角点可能退化(近共线/分布过窄)")

    def residual(theta: np.ndarray) -> np.ndarray:
        x1, _ = models.transform_to_camera(objp_arr, theta[:3], theta[3:])
        x2 = x1 @ R_rel_arr.T + t_rel_arr  # X_cam2 = R_rel @ X_cam1 + t_rel
        return np.concatenate([(models.project(x1, params1) - uv1_arr).ravel(),
                               (models.project(x2, params2) - uv2_arr).ravel()])

    theta0 = np.concatenate([np.asarray(rvec0, dtype=np.float64).ravel(),
                             np.asarray(tvec0, dtype=np.float64).ravel()])
    sol = least_squares(residual, theta0, method="trf", loss="linear",
                        x_scale="jac", max_nfev=_POSE_BA_MAX_NFEV)
    rms = float(np.sqrt(np.mean(sol.fun ** 2)))
    return sol.x[:3].copy(), sol.x[3:].copy(), rms


# ---------------------------------------------------------------------------
# 单帧评估 + split 汇总(失败契约照 scoring.py 的 _failure 体例)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameEval:
    """单对评估结果(不可变):NaN 表示该帧失败,note 记中文病因。

    dist_err_mm:    三角化板系对角距离 − 真值(mm,带符号);
    dist_measured_mm: 三角化出的对角距离本身(mm);
    reproj_rms:     per-frame BA 的两路 pooled 重投影 RMS(px);
    flipped:        (cam1 是否被消歧翻转, cam2 是否被消歧翻转);
    note:           成功为空串,失败为中文病因(绝不抛异常中断整批)。
    """

    group_id: str
    dist_err_mm: float
    dist_measured_mm: float
    reproj_rms: float
    flipped: tuple[bool, bool]
    note: str


def _failure_frame(group_id: str, note: str) -> FrameEval:
    """失败帧统一出口:全 NaN + 中文备注,绝不抛异常中断整批。"""
    nan = float("nan")
    return FrameEval(group_id=group_id, dist_err_mm=nan, dist_measured_mm=nan,
                     reproj_rms=nan, flipped=(False, False), note=note)


def _evaluate_frame_full(group_id: str, uv1: np.ndarray, uv2: np.ndarray,
                         objp: np.ndarray, intr1: IntrinsicsView,
                         intr2: IntrinsicsView, R_rel: np.ndarray,
                         t_rel: np.ndarray, flip_ratio: float
                         ) -> tuple[FrameEval, np.ndarray, np.ndarray]:
    """单对全链(内部全量版):消歧 → BA → 去畸变 → DLT → 板系 → 对角残差。

    额外返回该帧两路的逐角点重投影残差 (P,2)(BA 位姿回代,float64,
    只经 models),供 split 层组装诊断 npz;单帧失败时残差为 None。

    链路装配(参考 06 L245-263):DLT 输出在 cam1 系(projections[0] 的
    世界系),用 BA 位姿反变换回板系:pts_board = R_poseᵀ @ (pts_ref − t_pose),
    R_pose/t_pose 与三角化输出系一致(board→cam1)。
    """
    uv1_arr = np.asarray(uv1, dtype=np.float64)
    uv2_arr = np.asarray(uv2, dtype=np.float64)
    objp_arr = np.asarray(objp, dtype=np.float64)
    if uv1_arr.ndim != 2 or uv1_arr.shape[1] != 2 or uv2_arr.shape != uv1_arr.shape:
        return (_failure_frame(
            group_id,
            f"观测形状不合:期望 (角点数,2),实际是 {uv1_arr.shape} / {uv2_arr.shape}。"
            f"评估链一次处理一个双目对,请检查调用方的数据排布。"), None, None)
    if not (np.isfinite(uv1_arr).all() and np.isfinite(uv2_arr).all()):
        bad = int((~np.isfinite(uv1_arr)).sum() + (~np.isfinite(uv2_arr)).sum())
        return (_failure_frame(
            group_id,
            f"角点里混入 {bad} 个非数值(NaN/无穷),该帧几何可疑:按单帧失败"
            f"记录,不影响其余帧。"), None, None)

    try:
        uv1_fix, uv2_fix, flipped = disambiguate_frame(
            uv1_arr, uv2_arr, objp_arr, intr1, intr2, R_rel, t_rel, flip_ratio)
        rvec_pose, tvec_pose, rms = frame_pose_ba(
            uv1_fix, uv2_fix, objp_arr, intr1, intr2, R_rel, t_rel)

        # 去畸变(P=K 像素口径)+ 无畸变投影矩阵:cam1 参考系 [I|0],
        # cam2 用外参 [R_rel|t_rel](参考 06 L245-254)
        k1 = np.asarray(intr1.K, dtype=np.float64)
        d1 = np.asarray(intr1.D, dtype=np.float64).ravel()
        k2 = np.asarray(intr2.K, dtype=np.float64)
        d2 = np.asarray(intr2.D, dtype=np.float64).ravel()
        p1 = projection_matrix(k1, np.eye(3), np.zeros(3))
        p2 = projection_matrix(k2, R_rel, t_rel)
        pts_ref = triangulate_dlt(
            [p1, p2],
            [undist_pix(uv1_fix, k1, d1), undist_pix(uv2_fix, k2, d2)])

        # 板系反变换(参考 06 L263)+ 对角距离残差(真值从 objp 导出)
        R_pose, _ = cv2.Rodrigues(np.asarray(rvec_pose, dtype=np.float64))
        t_pose = np.asarray(tvec_pose, dtype=np.float64).ravel()
        pts_board = (R_pose.T @ (pts_ref - t_pose).T).T
        measured, _idx_a, _idx_b = farthest_corners_distance(pts_board, objp_arr)
        dist_err = measured - true_board_distance(objp_arr)

        # 该帧两路逐角点残差(BA 位姿回代,float64,只经 models;与
        # frame_pose_ba 的残差同构,供诊断 npz)
        x1, _ = models.transform_to_camera(objp_arr, rvec_pose, tvec_pose)
        x2 = x1 @ np.asarray(R_rel, dtype=np.float64).T + np.asarray(t_rel, dtype=np.float64).ravel()
        res1 = models.project(x1, np.asarray(intr1.params, dtype=np.float64).ravel()) - uv1_fix
        res2 = models.project(x2, np.asarray(intr2.params, dtype=np.float64).ravel()) - uv2_fix

        frame = FrameEval(group_id=group_id, dist_err_mm=float(dist_err),
                          dist_measured_mm=float(measured), reproj_rms=float(rms),
                          flipped=flipped, note="")
        return frame, res1, res2
    except _FRAME_EXC as e:
        # 数值路径失败收口成单帧失败;形状/参数契约已在入口防线拦下,
        # 不吞代码契约(scoring L114-120 AssertionError-safe 体例)
        return (_failure_frame(
            group_id, f"评估链失败:{type(e).__name__}——{e}"), None, None)


def evaluate_frame(group_id: str, uv1: np.ndarray, uv2: np.ndarray,
                   objp: np.ndarray, intr1: IntrinsicsView, intr2: IntrinsicsView,
                   R_rel: np.ndarray, t_rel: np.ndarray,
                   flip_ratio: float) -> FrameEval:
    """单对评估(公开入口):全链跑完给 FrameEval,失败返回 NaN + 中文备注。

    参数与链路见 _evaluate_frame_full;本入口只剥掉诊断残差,供逐对
    调试与测试使用。
    """
    frame, _res1, _res2 = _evaluate_frame_full(
        group_id, uv1, uv2, objp, intr1, intr2, R_rel, t_rel, flip_ratio)
    return frame


@dataclass(frozen=True)
class SplitEval:
    """一个 split(test 或 train 幸存集)的评估汇总(不可变)。

    evals:           逐对 FrameEval(行序与输入 pairs 一一对应,行序锁);
    distance_rms_mm: 成功帧对角距离残差的 pooled RMS(sqrt(平方均值),mm);
    residuals_cam1 / residuals_cam2: (V,P,2) 逐角点重投影残差,失败行
                     整行 NaN(诊断 npz 的数据源);
    true_distance_mm: 真值 = norm(objp[-1] − objp[0])(从配置棋盘导出)。
    """

    evals: tuple[FrameEval, ...]
    distance_rms_mm: float
    residuals_cam1: np.ndarray
    residuals_cam2: np.ndarray
    true_distance_mm: float


def evaluate_split(pairs: Pairs, intr1: IntrinsicsView, intr2: IntrinsicsView,
                   R: np.ndarray, t: np.ndarray, board: BoardSpec,
                   flip_ratio: float) -> SplitEval:
    """逐对评估一个 split,汇成 pooled 距离 RMS(STER-03 的整条链)。

    真值经 build_object_points(board) 导出(配置单一来源);单帧失败
    NaN + 中文备注不中断;全部失败 → 中文 ValueError(考卷全糊属环境
    异常而非数据波动,不静默产出 NaN 报告)。本函数不剔除任何对——
    test 永不剔除只标记,train 的剔除决策在编排层(距离残差绝不回流
    剔除判据,决策 2/决策 9)。
    """
    uv1 = np.asarray(pairs.uv1, dtype=np.float64)
    uv2 = np.asarray(pairs.uv2, dtype=np.float64)
    objp = np.asarray(pairs.objp, dtype=np.float64)
    if uv1.ndim != 3 or uv1.shape[2] != 2 or uv2.shape != uv1.shape:
        raise AssertionError(
            f"对集形状不合:期望 uv1/uv2 各 (对数,角点数,2),实际是 "
            f"{uv1.shape} / {uv2.shape}。请传入装载层的 Pairs。")
    n_pairs, n_points = uv1.shape[0], uv1.shape[1]
    if objp.shape != (n_points, 3):
        raise AssertionError(
            f"物点形状与角点不一致:objp {objp.shape} 对不上角点数 {n_points}。")

    true_d = true_board_distance(build_object_points(board))
    evals: list[FrameEval] = []
    residuals1 = np.full((n_pairs, n_points, 2), np.nan)
    residuals2 = np.full((n_pairs, n_points, 2), np.nan)
    err_sq: list[float] = []
    for j, gid in enumerate(pairs.image_ids):
        frame, res1, res2 = _evaluate_frame_full(
            str(gid), uv1[j], uv2[j], objp, intr1, intr2, R, t, flip_ratio)
        evals.append(frame)
        if res1 is not None:
            residuals1[j] = res1
            residuals2[j] = res2
            err_sq.append(frame.dist_err_mm * frame.dist_err_mm)

    if not err_sq:
        raise ValueError(
            f"评估失败:{n_pairs} 个对全部算不出距离残差(考卷全糊,属环境异常"
            f"而非数据波动):外参/内参与角点数据多半不同源,或棋盘规格错位。"
            f"逐帧备注:{[e.note for e in evals]}。请检查产物来源与 config.yaml。")
    distance_rms = float(np.sqrt(np.mean(np.asarray(err_sq, dtype=np.float64))))
    return SplitEval(evals=tuple(evals), distance_rms_mm=distance_rms,
                     residuals_cam1=residuals1, residuals_cam2=residuals2,
                     true_distance_mm=true_d)
