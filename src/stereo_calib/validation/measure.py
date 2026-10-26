"""模块4 测量核心:冻结外参/逐对 PnP 双模式测棋盘对角距离(全链数学零自研)。

单一数学源纪律(照 stereo.evaluate 的声明,此处只装配不发明):三角化只经
evaluate.triangulate_dlt(源码防线:全工程 SVD 齐次求解唯一住址)、去畸变只经
evaluate.undist_pix(内部 models.unproject 牛顿逆)、真值只经
evaluate.true_board_distance / farthest_corners_distance、消歧只经
evaluate.disambiguate_frame(与训练评估同一 180° 角点序口径)、投影/位姿
只经 intrinsic.models。本文件出现的每一行几何计算都是对上述函数的调用装配,
没有任何第二份公式。

「绝不重拟合」边界(模块4 的宪法):本文件消费的 K/D/R/t 全部来自冻结产物,
绝不被任何求解器更新。唯一出现的求解是逐对模式的 cv2.solvePnP——它是 PLAN
规格明确要求的**归因诊断量**(每对自成一体测自洽性,不回流任何标定参数,
与 stereo.solve 的外参求解不同物);scipy 精化/solve_stereo 在本包零调用。

失败契约(照 stereo.evaluate 体例):单对任何数值路径失败返回 NaN + 中文
备注,绝不抛异常中断整批;形状/参数契约在入口防线拦。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from stereo_calib.intrinsic import models
from stereo_calib.stereo.evaluate import (
    disambiguate_frame,
    farthest_corners_distance,
    projection_matrix,
    triangulate_dlt,
    true_board_distance,
    undist_pix,
)
from stereo_calib.stereo.observation import IntrinsicsView

# 单对数值异常收口家族(evaluate.py 同款):消歧锚 PnP / 牛顿逆 / DLT 的
# 数值失败都收成单对失败,不中断整批
_FRAME_EXC = (ValueError, ArithmeticError, RuntimeError, AssertionError, cv2.error)

# solvePnP 迭代解法要求的最少点数(evaluate.py 同一防线)
_MIN_PNP_POINTS = 4


@dataclass(frozen=True)
class FrozenExtrinsics:
    """冻结双目外参快照(只读):语义 X_cam2 = R @ X_cam1 + t(cam1 参考系,
    t 单位 mm),与 extrinsics_stereo.json 的写侧契约逐字一致。"""

    R: np.ndarray            # (3, 3) 旋转(Rodrigues(rvec) 还原)
    t: np.ndarray            # (3,)   平移(mm)
    rvec: np.ndarray         # (3,)   存档旋转向量(留痕原样)
    baseline_mm: float       # ‖t‖


def load_frozen_extrinsics(json_path: Path | str) -> FrozenExtrinsics:
    """按名读取 extrinsics_stereo.json 的 rvec/t → 冻结外参(绝不重解)。

    只读消费契约键(rvec/t;baseline_mm 存在时做一致性核对,防存档被手改
    出现两处来源漂移);R 一律由 cv2.Rodrigues(rvec) 现场还原——旋转在
    存档里的唯一权威形态是 rvec。缺键/形状不合均中文报错。
    """
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(  # noqa: TRY004  CLI 异常网只收 ValueError 家族
            f"双目外参文件 {Path(json_path).name} 不合法:顶层应当是 JSON 对象。")
    missing = [key for key in ("rvec", "t") if key not in data]
    if missing:
        raise ValueError(
            f"双目外参文件 {Path(json_path).name} 不合法:缺字段 {missing}——"
            f"它不是 extrinsics_stereo.json 契约的产物,请先运行 stereo-calib stereo。")
    rvec = np.asarray(data["rvec"], dtype=np.float64).ravel()
    t = np.asarray(data["t"], dtype=np.float64).ravel()
    if rvec.shape != (3,) or t.shape != (3,):
        raise ValueError(
            f"双目外参文件 {Path(json_path).name} 不合法:rvec/t 应当各是 3 长"
            f"(Rodrigues 旋转向量与 mm 平移),实际是 {rvec.shape} / {t.shape}。"
            f"请重跑 stereo-calib stereo。")
    R, _ = cv2.Rodrigues(rvec)
    baseline = float(np.linalg.norm(t))
    if "baseline_mm" in data and abs(baseline - float(data["baseline_mm"])) > 1.0e-6:
        raise ValueError(
            f"双目外参文件 {Path(json_path).name} 不合法:存档 baseline_mm="
            f"{float(data['baseline_mm'])!r} 与 ‖t‖={baseline!r} 不一致——"
            f"存档被改动或写侧契约漂移,请重跑 stereo-calib stereo。")
    return FrozenExtrinsics(R=R, t=t, rvec=rvec, baseline_mm=baseline)


@dataclass(frozen=True)
class PairMeasurement:
    """单对双模式测量结果(不可变):NaN 表示该对失败,note 记中文病因。

    err_frozen_mm / err_pair_mm: 冻结/逐对模式的对角距离偏差(mm,带符号,
                      测得 − 真值);
    measured_frozen_mm / measured_pair_mm: 两种模式测得的对角距离本身(mm);
    center_u_px / center_v_px:   板中心(cam1 角点均值,像素;区域分桶用);
    reproj_rms_px:               逐对 PnP 位姿回代的两路 pooled 重投影 RMS
                                 (该对数据质量参考,只经 models);
    flipped:                     (cam1, cam2) 是否被 180° 消歧翻转;
    note:                        成功为空串,失败为中文病因。
    """

    group_id: str
    err_frozen_mm: float
    err_pair_mm: float
    measured_frozen_mm: float
    measured_pair_mm: float
    center_u_px: float
    center_v_px: float
    reproj_rms_px: float
    flipped: tuple[bool, bool]
    note: str


def _failure_pair(group_id: str, note: str) -> PairMeasurement:
    """失败对统一出口:全 NaN + 中文备注,绝不抛异常中断整批。"""
    nan = float("nan")
    return PairMeasurement(group_id=group_id, err_frozen_mm=nan, err_pair_mm=nan,
                           measured_frozen_mm=nan, measured_pair_mm=nan,
                           center_u_px=nan, center_v_px=nan, reproj_rms_px=nan,
                           flipped=(False, False), note=note)


def _pnp_pose(objp: np.ndarray, uv: np.ndarray, K: np.ndarray,
              D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """单路 PnP 位姿(带畸变原始像素 + 全套 K/D,evaluate 消歧锚同口径)。

    只服务逐对归因诊断:位姿用完即弃,绝不回流任何标定参数(模块4 宪法)。
    """
    ok, rvec, tvec = cv2.solvePnP(
        objp.astype(np.float32).reshape(-1, 1, 3),
        uv.astype(np.float32).reshape(-1, 1, 2),
        K, D.reshape(-1, 1), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("PnP 无解,该对几何可疑")
    return (np.asarray(rvec, dtype=np.float64).ravel(),
            np.asarray(tvec, dtype=np.float64).ravel())


def measure_pair(group_id: str, uv1: np.ndarray, uv2: np.ndarray,
                 objp: np.ndarray, intr1: IntrinsicsView, intr2: IntrinsicsView,
                 ext: FrozenExtrinsics, flip_ratio: float) -> PairMeasurement:
    """单对双模式测量(公开入口):失败返回 NaN + 中文备注,不中断整批。

    链路装配(数学全部 import 复用,见模块 docstring):
      1. 消歧:disambiguate_frame(冻结外参当锚,与训练评估同一 180° 口径);
      2. 冻结模式:P1=K1[I|0]、P2=K2[R|t](存档 R/t)→ 去畸变(undist_pix)
         → DLT → 对角距离 → err_frozen = 测得 − 真值;
      3. 逐对模式:两路各自 PnP → 相对位姿 R21=R2·R1ᵀ、t21=t2−R21·t1
         (solve.py relative_pose 同一组合语义)→ DLT → err_pair;
      4. 重投影参考:两路 PnP 位姿回代(只经 models)的 pooled RMS。
    """
    uv1_arr = np.asarray(uv1, dtype=np.float64)
    uv2_arr = np.asarray(uv2, dtype=np.float64)
    objp_arr = np.asarray(objp, dtype=np.float64)
    if uv1_arr.ndim != 2 or uv1_arr.shape[1] != 2 or uv2_arr.shape != uv1_arr.shape:
        raise AssertionError(
            f"观测形状不合:期望 uv1/uv2 各 (角点数,2),实际是 "
            f"{uv1_arr.shape} / {uv2_arr.shape}。验证链一次处理一个双目对。")
    if uv1_arr.shape[0] < _MIN_PNP_POINTS:
        raise AssertionError(
            f"角点数 {uv1_arr.shape[0]} 少于求解下限 {_MIN_PNP_POINTS} 个:方程组欠定。")

    try:
        uv1_fix, uv2_fix, flipped = disambiguate_frame(
            uv1_arr, uv2_arr, objp_arr, intr1, intr2, ext.R, ext.t, flip_ratio)

        k1 = np.asarray(intr1.K, dtype=np.float64)
        d1 = np.asarray(intr1.D, dtype=np.float64).ravel()
        k2 = np.asarray(intr2.K, dtype=np.float64)
        d2 = np.asarray(intr2.D, dtype=np.float64).ravel()
        params1 = np.asarray(intr1.params, dtype=np.float64).ravel()
        params2 = np.asarray(intr2.params, dtype=np.float64).ravel()

        # 去畸变(P=K 像素口径;undist_pix 内部走 models.unproject 牛顿逆)
        und1 = undist_pix(uv1_fix, k1, d1)
        und2 = undist_pix(uv2_fix, k2, d2)
        # cam1 参考系的针孔投影矩阵:冻结与逐对模式共用 P1
        p1 = projection_matrix(k1, np.eye(3), np.zeros(3))

        # ---- 冻结模式(健康判据):P2 = K2[R|t],R/t 是存档原值 ----
        p2_frozen = projection_matrix(k2, ext.R, ext.t)
        pts_frozen = triangulate_dlt([p1, p2_frozen], [und1, und2])
        measured_frozen, _a, _b = farthest_corners_distance(pts_frozen, objp_arr)
        theory = true_board_distance(objp_arr)
        err_frozen = measured_frozen - theory

        # ---- 逐对模式(归因诊断):每对自成一体,吸收外参漂移 ----
        rv1, tv1 = _pnp_pose(objp_arr, uv1_fix, k1, d1)
        rv2, tv2 = _pnp_pose(objp_arr, uv2_fix, k2, d2)
        R1, _ = cv2.Rodrigues(rv1)
        R2, _ = cv2.Rodrigues(rv2)
        R21 = R2 @ R1.T
        t21 = tv2 - R21 @ tv1
        p2_pair = projection_matrix(k2, R21, t21)
        pts_pair = triangulate_dlt([p1, p2_pair], [und1, und2])
        measured_pair, _c, _d = farthest_corners_distance(pts_pair, objp_arr)
        err_pair = measured_pair - theory

        # ---- 重投影参考:两路 PnP 位姿回代(只经 models,单一数学源)----
        x1, _ = models.transform_to_camera(objp_arr, rv1, tv1)
        x2, _ = models.transform_to_camera(objp_arr, rv2, tv2)
        res = np.concatenate([
            (models.project(x1, params1) - uv1_fix).ravel(),
            (models.project(x2, params2) - uv2_fix).ravel()])
        reproj_rms = float(np.sqrt(np.mean(res ** 2)))

        center = uv1_fix.mean(axis=0)
        return PairMeasurement(
            group_id=group_id, err_frozen_mm=float(err_frozen),
            err_pair_mm=float(err_pair), measured_frozen_mm=float(measured_frozen),
            measured_pair_mm=float(measured_pair), center_u_px=float(center[0]),
            center_v_px=float(center[1]), reproj_rms_px=reproj_rms,
            flipped=flipped, note="")
    except _FRAME_EXC as e:
        # 数值路径失败收口成单对失败;形状/参数契约已在入口防线拦下
        return _failure_pair(group_id, f"测量链失败:{type(e).__name__}——{e}")


def evaluate_gates(errs_frozen_mm: list[float], bias_max_mm: float,
                   rms_max_mm: float) -> dict:
    """冻结模式判定闸门:|平均偏差| 与 RMS 双红线(全配置化,fail-closed)。

    只吃有限值(失败对的 NaN 由调用方过滤);逐对模式不设闸门(诊断量,
    不判生死)。返回 dict 直接进 validation.json 的 gates 键。
    """
    arr = np.asarray(errs_frozen_mm, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError(
            "判定闸门无输入:冻结模式没有任何一个有效偏差值(全部对测量失败)。"
            "属环境异常而非数据波动,不静默产出空判定——请检查内参/外参产物与验证图像是否同源。")
    mean = float(np.mean(arr))
    rms = float(np.sqrt(np.mean(arr ** 2)))
    return {
        "n_pairs_used": int(arr.size),
        "mean_err_frozen_mm": mean,
        "rms_err_frozen_mm": rms,
        "distance_bias_max_mm": float(bias_max_mm),
        "distance_rms_max_mm": float(rms_max_mm),
        "bias_pass": bool(abs(mean) <= float(bias_max_mm)),
        "rms_pass": bool(rms <= float(rms_max_mm)),
    }


def analyze_regions(pairs: list[PairMeasurement], region_grid: tuple[int, int],
                    image_size: tuple[int, int]) -> list[dict]:
    """按板中心(cam1 角点均值)在 cam1 视场分桶,逐桶冻结模式 |偏差| 统计。

    region_grid = (行桶数, 列桶数);[0, 0] 表示关闭(返回空清单)。桶序
    行主序(行在外、列在内);无样本桶如实给 n=0 + 「无样本」,绝不拿空桶
    编造统计。统计对象是 |err_frozen|(漂移体检关心的是量级)。
    """
    n_rows, n_cols = int(region_grid[0]), int(region_grid[1])
    if n_rows <= 0 or n_cols <= 0:
        return []
    width, height = int(image_size[0]), int(image_size[1])
    buckets: dict[tuple[int, int], list[float]] = {
        (r, c): [] for r in range(n_rows) for c in range(n_cols)}
    for pm in pairs:
        if not np.isfinite(pm.err_frozen_mm) or not np.isfinite(pm.center_u_px):
            continue  # 失败对(或无中心)不进任何桶,如实缺席
        col = min(int(pm.center_u_px / width * n_cols), n_cols - 1)
        row = min(int(pm.center_v_px / height * n_rows), n_rows - 1)
        buckets[(row, col)].append(abs(pm.err_frozen_mm))
    regions: list[dict] = []
    for r in range(n_rows):
        for c in range(n_cols):
            vals = buckets[(r, c)]
            if vals:
                arr = np.asarray(vals, dtype=np.float64)
                regions.append({
                    "region": f"r{r}c{c}", "n": int(arr.size),
                    "median_abs_err_mm": float(np.median(arr)),
                    "max_abs_err_mm": float(arr.max()), "note": ""})
            else:
                regions.append({
                    "region": f"r{r}c{c}", "n": 0,
                    "median_abs_err_mm": None, "max_abs_err_mm": None,
                    "note": "无样本"})
    return regions
