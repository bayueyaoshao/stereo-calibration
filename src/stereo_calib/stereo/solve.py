"""双目求解引擎:PnP-median 初值 + 两遍法消歧 + scipy 外参精化(模块3 引擎切片)。

移植自参考工程 2 core/stereo.py(PnP-median 全套)、core/multibundle.py(外参
联合 BA 的 2 相机特化)与 cli/05_extrinsic_multi.py 的排序规范化消歧(只读
引用,绝不修改)。双目外参禁用 OpenCV 内建双目标定例程——参考工程在该机型
实测存在缺陷(无鲁棒损失 LM 被坏对应拖变形、平移模长被标到 2500mm 量级),
本模块全程走「逐对 PnP → 相对位姿 → 中位聚合 → scipy 联合精化」,禁用由
源码防线测试锁定。

结构(单一数学源纪律,与 intrinsic/solve.py 同构):
- relative_pose:cam1→cam2 相对位姿的唯一实现(方向语义在 docstring 锁定);
- pnp_median:逐对两路 solvePnP(带畸变原始像素 + 迭代法)→ 消去世界系 →
  Rodrigues 向量逐分量中位;失败对跳过不入堆叠(防 NaN 传播);离散度
  防线防 Rodrigues 中位退化;
- disambiguate_pairs / refine_stereo / solve_stereo:两遍法消歧与解析稀疏
  雅可比联合精化(θ = [6 外参 | 6×N 位姿],cam1 恒为单位位姿不进 θ)。

残差与指标只准 import intrinsic.models 的投影/变换函数,本文件不出现第二份
投影或畸变公式(源码防线测试锁定)。dtype 纪律(Pitfall 7):OpenCV 调用点
显式装包 float32 点位、K/D float64;models 回代路径全 float64,报告指标绝不
消费 OpenCV 的返回值。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from stereo_calib.intrinsic import models
from stereo_calib.stereo.observation import IntrinsicsView, Pairs

# solvePnP 逐对调用的数值异常家族收口(scoring.py 同款体例):失败对按单对
# 失败记录(跳过不入堆叠),绝不抛异常中断整批、更绝不填 NaN 进聚合。
_PNP_EXC = (ValueError, ArithmeticError, RuntimeError, AssertionError, cv2.error)


@dataclass(frozen=True)
class StereoInit:
    """PnP-median 初值解(不可变):cam1 参考系下的双目外参 + 逐对诊断。

    R (3,3) / t (3,) float64:cam1→cam2 外参(语义 X_cam2 = R @ X_cam1 + t,
    t 单位 mm——objp 是 mm);baseline_mm = ‖t‖;rvec (3,) 为 R 的 Rodrigues
    向量,rotation_angle_deg = ‖rvec‖ 的度数。rvecs1/tvecs1 与 rvecs2/tvecs2
    (N,3) 是每对 board→各相机的 PnP 位姿(消歧锚要用,行序与输入 pairs
    一一对应;该对求解失败的行为 NaN);per_pair_rms (N,) 为逐对两路 pooled
    重投影 RMS(float64 回代,失败对为 NaN);per_pair_ok 逐对标记成败;
    spread_deg 是逐对相对位姿相对中位的最大夹角(离散度诊断);n_valid 是
    成功对数。
    """

    R: np.ndarray
    t: np.ndarray
    baseline_mm: float
    rvec: np.ndarray
    rotation_angle_deg: float
    rvecs1: np.ndarray
    tvecs1: np.ndarray
    rvecs2: np.ndarray
    tvecs2: np.ndarray
    per_pair_rms: np.ndarray
    per_pair_ok: tuple[bool, ...]
    spread_deg: float
    n_valid: int


def relative_pose(R_a: np.ndarray, t_a: np.ndarray,
                  R_b: np.ndarray, t_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """从两路各自的世界系位姿算 cam_a → cam_b 的相对位姿(方向唯一实现)。

    世界系: X_world
    cam_a 系: X_a = R_a @ X_world + t_a
    cam_b 系: X_b = R_b @ X_world + t_b

    消去 X_world: X_b = R_b @ R_a.T @ (X_a - t_a) + t_b
                   = (R_b @ R_a.T) @ X_a + (t_b - R_b @ R_a.T @ t_a)
    所以 R_rel = R_b @ R_a.T,  t_rel = t_b - R_rel @ t_a.

    语义(Pitfall 6,方向唯一):X_cam2 = R_rel @ X_cam1 + t_rel——外参把
    cam1 系的点搬到 cam2 系;搞反(差个转置)时重投影 RMS 可能看不出灾难
    性差异,合成真值测试在点变换层面锁定方向。
    """
    R_rel = R_b @ R_a.T
    t_rel = t_b.flatten() - R_rel @ t_a.flatten()
    return R_rel, t_rel


def pnp_median(pairs: Pairs, intr1: IntrinsicsView, intr2: IntrinsicsView,
               median_spread_max_deg: float | None) -> StereoInit:
    """PnP-median 初值:逐对两路 PnP → 相对位姿 → 中位聚合(参考工程原样)。

    白话:每个双目对里两台相机各自解棋盘位姿(PnP,输入是带畸变的原始
    像素——畸变由 OpenCV 在 PnP 内部处理,这是 PnP 链路与 DLT 链路的
    契约分界),消去世界系得到一个「cam1→cam2 相对位姿」样本;刚架的
    相对位姿跨对不变,所以全部样本取中位数就是对它最稳健的估计。旋转的
    中位数走 Rodrigues 向量逐分量中位再转回(参考工程实测做法)。

    参数:
        pairs: 双目对集(行序契约:一切输出按 image_ids 顺序对齐);
        intr1/intr2: 两路固定内参(Phase 2 产物,只读);
        median_spread_max_deg: 离散度防线上限(度)。逐对相对位姿相对中位
            的最大夹角超过它 → 中文 ValueError(翻转帧/孪生解污染,中位
            数对多峰数据失效,Pitfall 1);传 None 关闭防线(仅供消歧前
            的 pass-1 使用——那一步的离散度高恰恰是消歧要修的东西,
            两遍法消歧后的 pass-2 必须传配置阈值)。
    """
    uv1 = np.asarray(pairs.uv1, dtype=np.float64)
    uv2 = np.asarray(pairs.uv2, dtype=np.float64)
    objp = np.asarray(pairs.objp, dtype=np.float64)
    if uv1.ndim != 3 or uv1.shape[2] != 2 or uv2.shape != uv1.shape:
        raise ValueError(
            f"双目对集形状不合:期望 uv1/uv2 各 (对数,角点数,2),实际是 "
            f"{uv1.shape} / {uv2.shape}。请检查调用方传入的是不是装载层的 "
            f"Pairs(逐对逐角点像素坐标)。"
        )
    n_pairs, n_points = uv1.shape[0], uv1.shape[1]
    if objp.shape != (n_points, 3):
        raise ValueError(
            f"物点形状与角点不一致:objp {objp.shape} 对不上角点数 {n_points}。"
            f"物点网格必须与角点逐点对齐,请检查 Pairs 的装载来源。"
        )
    if median_spread_max_deg is not None and not median_spread_max_deg > 0:
        raise ValueError(
            f"离散度防线上限不合法:期望正数(度)或 None(关防线),实际是 "
            f"{median_spread_max_deg!r}。请检查 stereo.median_spread_max_deg 配置。"
        )

    # dtype 纪律(Pitfall 7):点位 float32、K/D float64 reshape(-1,1)——
    # 参考工程实测的合法组合,装包规则集中在调用处显式完成。
    objp32 = objp.astype(np.float32).reshape(-1, 1, 3)
    k1 = np.asarray(intr1.K, dtype=np.float64)
    d1 = np.asarray(intr1.D, dtype=np.float64).ravel().reshape(-1, 1)
    k2 = np.asarray(intr2.K, dtype=np.float64)
    d2 = np.asarray(intr2.D, dtype=np.float64).ravel().reshape(-1, 1)
    params1 = np.asarray(intr1.params, dtype=np.float64).ravel()
    params2 = np.asarray(intr2.params, dtype=np.float64).ravel()

    R_rels: list[np.ndarray] = []
    t_rels: list[np.ndarray] = []
    rvecs1 = np.full((n_pairs, 3), np.nan)
    tvecs1 = np.full((n_pairs, 3), np.nan)
    rvecs2 = np.full((n_pairs, 3), np.nan)
    tvecs2 = np.full((n_pairs, 3), np.nan)
    per_pair_rms = np.full(n_pairs, np.nan)
    per_pair_ok: list[bool] = []

    for j in range(n_pairs):
        ok_pair = True
        rv_a = tv_a = rv_b = tv_b = None
        try:
            ok_a, rv_a, tv_a = cv2.solvePnP(
                objp32, uv1[j].astype(np.float32).reshape(-1, 1, 2),
                k1, d1, flags=cv2.SOLVEPNP_ITERATIVE)
            ok_b, rv_b, tv_b = cv2.solvePnP(
                objp32, uv2[j].astype(np.float32).reshape(-1, 1, 2),
                k2, d2, flags=cv2.SOLVEPNP_ITERATIVE)
            if not (ok_a and ok_b):
                ok_pair = False
        except _PNP_EXC:
            ok_pair = False
        if ok_pair:
            rv_a = np.asarray(rv_a, dtype=np.float64).ravel()
            tv_a = np.asarray(tv_a, dtype=np.float64).ravel()
            rv_b = np.asarray(rv_b, dtype=np.float64).ravel()
            tv_b = np.asarray(tv_b, dtype=np.float64).ravel()
            if not all(np.isfinite(v).all() for v in (rv_a, tv_a, rv_b, tv_b)):
                ok_pair = False
        if not ok_pair:
            # 失败对跳过不入堆叠(Pitfall 9:NaN 塞进堆叠会污染整个中位数),
            # 逐对诊断位照常占位(NaN/False),供剔除循环与报告解释 n_valid。
            per_pair_ok.append(False)
            continue

        R_a, _ = cv2.Rodrigues(rv_a)
        R_b, _ = cv2.Rodrigues(rv_b)
        R_rel, t_rel = relative_pose(R_a, tv_a, R_b, tv_b)

        # 逐对重投影 RMS(判据尺,float64 回代,绝不用 OpenCV 返回值):
        # cam1 残差经 models, cam2 残差经 models,两路 pooled 成单值。
        x1, _ = models.transform_to_camera(objp, rv_a, tv_a)
        x2, _ = models.transform_to_camera(objp, rv_b, tv_b)
        d1_res = models.project(x1, params1) - uv1[j]
        d2_res = models.project(x2, params2) - uv2[j]
        sq_points = (np.sum(d1_res * d1_res, axis=1) + np.sum(d2_res * d2_res, axis=1))
        per_pair_rms[j] = float(np.sqrt(sq_points.mean()))

        rvecs1[j], tvecs1[j], rvecs2[j], tvecs2[j] = rv_a, tv_a, rv_b, tv_b
        R_rels.append(R_rel)
        t_rels.append(t_rel)
        per_pair_ok.append(True)

    n_valid = len(R_rels)
    if n_valid < 2:
        raise ValueError(
            f"双目初值求解失败:只有 {n_valid} 个有效对(PnP 全部成功且有限的"
            f"对数,下限 2)——一对定不了相对位姿,硬算必错。请检查角点数据"
            f"质量、min_pairs 配置与 exclude_pairs 清单。"
        )

    # 中位聚合(参考工程 L113-121 原样):t 逐分量中位;旋转经 Rodrigues
    # 向量逐分量中位再转回(旋转矩阵没有天然的逐元素中位)。
    t_rels_arr = np.stack(t_rels, axis=0)
    t_med = np.median(t_rels_arr, axis=0)
    rv_rels = np.array([cv2.Rodrigues(R)[0].flatten() for R in R_rels])
    rv_med = np.median(rv_rels, axis=0)
    R_med, _ = cv2.Rodrigues(rv_med.reshape(3, 1))

    # 离散度诊断(参考没有、本工程新增的防线):每对相对旋转与中位旋转的
    # 测地夹角(R_j @ R_med.T 的转角,度)取最大。刚架的相对位姿跨对应
    # 紧密聚集;离散度过大 = 翻转帧/孪生解混入,中位数将被拖偏——当场
    # 报错而不是静默出坏外参(Pitfall 1;消歧后的 pass-2 必须传配置阈值)。
    # 注:不经「Rodrigues 向量方向夹角」度量——小转角场景下转轴方向对
    # 噪声的放大是 1/θ 倍(θ≈2° 时 0.05px 检测噪声就能让方向摆动 1°+,
    # 干净数据会被误杀);测地转角直接量「这帧的相对旋转离中位旋转多远」,
    # 对翻转帧/孪生解同样灵敏(污染帧的偏离是数十度量级)。
    dev_deg = np.array(
        [np.degrees(np.linalg.norm(cv2.Rodrigues(R_j @ R_med.T)[0])) for R_j in R_rels])
    spread_deg = float(dev_deg.max())
    if median_spread_max_deg is not None and spread_deg > median_spread_max_deg:
        raise ValueError(
            f"双目初值离散度防线触发:逐对相对位姿相对中位的最大夹角实测 "
            f"{spread_deg:.4f}°,超过配置阈值 {median_spread_max_deg}°——可能存在"
            f"180° 角点序翻转帧或共面孪生解污染(Rodrigues 向量中位数对多峰"
            f"数据失效)。请检查消歧流程是否执行、以及逐对 RMS 诊断。"
        )

    return StereoInit(
        R=np.asarray(R_med, dtype=np.float64), t=np.asarray(t_med, dtype=np.float64),
        baseline_mm=float(np.linalg.norm(t_med)),
        rvec=np.asarray(rv_med, dtype=np.float64),
        rotation_angle_deg=float(np.degrees(np.linalg.norm(rv_med))),
        rvecs1=rvecs1, tvecs1=tvecs1, rvecs2=rvecs2, tvecs2=tvecs2,
        per_pair_rms=per_pair_rms, per_pair_ok=tuple(per_pair_ok),
        spread_deg=spread_deg, n_valid=n_valid)
