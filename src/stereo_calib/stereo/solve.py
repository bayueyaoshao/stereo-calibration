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

from dataclasses import dataclass, replace

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix, csr_matrix

from stereo_calib.intrinsic import models
from stereo_calib.intrinsic.solve import _is_practically_converged, _tail_improvement
from stereo_calib.io.config import StereoCalibConfig, StereoRefineConfig
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


# ---------------------------------------------------------------------------
# 180° 角点序消歧:跨对相对位姿 t+R 联合聚类(主) + 单对投影自洽(fallback)
# 升级自原两遍法单对投影自洽;借鉴 four_eyed_vision/lib/run_all.py stage5
# ---------------------------------------------------------------------------

# 消歧聚类参数(取自 four_eyed_vision/lib/config.py 的实测值)
_DISAMB_DELTA_MM = 30.0      # 相对位姿平移聚类阈值(rig 刚性,t 跨对应近常数)
_DISAMB_DEG = 3.0            # 相对旋转聚类阈值(正/反序旋转差→联合 t+R 分离镜像)
_DISAMB_ROT_W = 50.0         # 选序时旋转角相对平移的权重(dn=‖t−t_true‖+W·∠R)
_DISAMB_MIN_CLUSTER = 4      # 最大簇内点数下限,低于此回落 fallback


def _rangle(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """相对旋转角(度),消歧 t+R 联合聚类的旋转度量。"""
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


def disambiguate_pairs(pairs: Pairs, intr1: IntrinsicsView, intr2: IntrinsicsView,
                       init: StereoInit, flip_ratio: float) -> tuple[Pairs, dict]:
    """180° 角点序消歧:跨对相对位姿 t+R 联合聚类(主) + 单对投影自洽(fallback)。

    升级说明(2026-10-26):原实现是「两遍法单对投影自洽」——用 pass-1 中位
    外参当锚,对每对独立比较「原始序/翻转序」重投影误差,`e_rev < flip_ratio
    ·e_fwd` 才翻。该判据对接近对称的棋盘会漏判(两种序误差差别不够大),本次
    新数据 20261026_175009_264608 即因此漏翻,触发离散度防线。

    现改为借鉴 four_eyed_vision 的「相对位姿跨对应为常数(刚体)」联合聚类:
      1) cam1 锁原序(world 系约定,杜绝跨对 180° 世界系歧义;原实现 cam1
         锚即自身解数学上恒不翻,此处显式锁死,行为等价);
      2) 对每对 cam2 枚举正/反序 PnP → 相对位姿候选(relative_pose);
      3) 全部候选做 t+R 联合聚类(平移<30mm 且旋转<3°),取最大一致簇为真值
         (R 取簇内 Rodrigues 向量中位数、t 取均值,复用本工程中位数聚合习惯);
      4) 每对选与真值(‖t−t_true‖ + 50·∠R)更近的序翻。
    此法用全体对投票,单对边界模糊时不再漏判。

    fallback:有效对过少致最大簇 < 4 时,回落到旧单对投影自洽逻辑(仍受
    flip_ratio 约束),保证小数据/异常场景行为不变(向后兼容)。

    纯函数:同输入同输出,无随机性。返回(消歧后的新 Pairs, flip_log
    {"pass2_flipped","pair_ids","method"})。绝不原地改输入 Pairs(T-03-06)。
    """
    if not 0 < flip_ratio <= 1.0:
        # 配置层已校验;此处防线自证(函数独立调用时不带病运行)
        raise ValueError(
            f"消歧判定阈值不合法:期望 0 到 1.0 之间(含上端点)的数,实际是 "
            f"{flip_ratio!r}。超过 1.0 即证据不足也翻(Pitfall 11)。"
        )
    params2 = np.asarray(intr2.params, dtype=np.float64).ravel()
    objp = np.asarray(pairs.objp, dtype=np.float64)
    objp32 = objp.astype(np.float32).reshape(-1, 1, 3)
    k2 = np.asarray(intr2.K, dtype=np.float64)
    d2 = np.asarray(intr2.D, dtype=np.float64).ravel().reshape(-1, 1)
    # cam1 锁原序:uv1 永不改;uv2 拷贝后按需翻转
    uv2 = np.array(pairs.uv2, dtype=np.float64)

    # ---- 构造相对位姿候选(每对 cam2 正/反序;cam1 锁正序) ----
    candidates: list[tuple[np.ndarray, np.ndarray, int, bool]] = []
    valid: list[int] = []
    for j, _gid in enumerate(pairs.image_ids):
        if not init.per_pair_ok[j]:
            continue  # 该对 pass-1 PnP 失败、无锚:不翻,留给剔除机制处置
        valid.append(j)
        R1, _ = cv2.Rodrigues(np.asarray(init.rvecs1[j], dtype=np.float64).ravel())
        t1 = np.asarray(init.tvecs1[j], dtype=np.float64).ravel()
        R2, _ = cv2.Rodrigues(np.asarray(init.rvecs2[j], dtype=np.float64).ravel())
        t2 = np.asarray(init.tvecs2[j], dtype=np.float64).ravel()
        Rn, tn = relative_pose(R1, t1, R2, t2)        # cam2 正序候选
        candidates.append((Rn, tn, j, False))
        try:                                          # cam2 反序候选:翻转 uv2 重 PnP
            ok_f, rv2f, tv2f = cv2.solvePnP(
                objp32, uv2[j][::-1].astype(np.float32).reshape(-1, 1, 2),
                k2, d2, flags=cv2.SOLVEPNP_ITERATIVE)
            if ok_f:
                R2f, _ = cv2.Rodrigues(np.asarray(rv2f, dtype=np.float64).ravel())
                t2f = np.asarray(tv2f, dtype=np.float64).ravel()
                Rf, tf = relative_pose(R1, t1, R2f, t2f)
                candidates.append((Rf, tf, j, True))
        except _PNP_EXC:
            pass

    # ---- 主路径:中位数共识消歧(对少数翻转离群稳健,无对称平局) ----
    # 默认"多数对的 cam2 正序正确",故取全部 cam2 正序相对位姿的中位数作共识;
    # 逐对比较其正/反序相对位姿与共识的(平移+旋转)距离,反序更近则翻。
    # 中位数对单峰数据中的少数翻转离群稳健(翻转对<半数时共识始终正确),
    # 不似 t+R 联合聚类在"恰好半数翻转"时因两簇对称而平局退化(本数据即此例)。
    flipped_cam2: set[int] = set()
    use_cluster = False
    fwd = [c for c in candidates if not c[3]]   # 全部 cam2 正序候选
    if len(fwd) >= _DISAMB_MIN_CLUSTER:
        Rs_fwd = np.array([cv2.Rodrigues(c[0])[0].ravel() for c in fwd])
        ts_fwd = np.array([c[1] for c in fwd])
        R_cons = cv2.Rodrigues(np.median(Rs_fwd, axis=0).reshape(3, 1))[0]
        t_cons = np.median(ts_fwd, axis=0)
        use_cluster = True
        for j in valid:
            cn = next(c for c in candidates if c[2] == j and not c[3])
            cf = next((c for c in candidates if c[2] == j and c[3]), None)
            if cf is None:
                continue
            dn = np.linalg.norm(cn[1] - t_cons) + _DISAMB_ROT_W * _rangle(cn[0], R_cons)
            df = np.linalg.norm(cf[1] - t_cons) + _DISAMB_ROT_W * _rangle(cf[0], R_cons)
            if df < dn:
                flipped_cam2.add(j)

    if not use_cluster:
        # ---- fallback:旧单对投影自洽(向后兼容) ----
        for j in valid:
            R1, _ = cv2.Rodrigues(np.asarray(init.rvecs1[j], dtype=np.float64).ravel())
            t1 = np.asarray(init.tvecs1[j], dtype=np.float64).ravel()
            xw = (R1 @ objp.T).T + t1
            x2 = xw @ init.R.T + init.t          # cam2 锚 = pass-1 中位外参
            proj2 = models.project(x2, params2)
            e_fwd = float(np.linalg.norm(proj2 - uv2[j]))
            e_rev = float(np.linalg.norm(proj2 - uv2[j][::-1]))
            if e_rev < flip_ratio * e_fwd:
                flipped_cam2.add(j)

    # cam1 锁原序:不翻 uv1;仅翻转标记过的 cam2 对
    for j in flipped_cam2:
        uv2[j] = uv2[j][::-1].copy()

    flipped_ids = [pairs.image_ids[j] for j in sorted(flipped_cam2)]
    pairs_out = Pairs(image_ids=pairs.image_ids, source1=pairs.source1,
                      source2=pairs.source2, routes=pairs.routes,
                      uv1=np.array(pairs.uv1, dtype=np.float64), uv2=uv2,
                      objp=pairs.objp)
    return pairs_out, {"pass2_flipped": len(flipped_ids), "pair_ids": flipped_ids,
                       "method": "median_consensus" if use_cluster else "fallback_proj"}


# ---------------------------------------------------------------------------
# scipy 外参精化:θ 装箱 + row_index 行序契约 + 解析稀疏雅可比
# (移植自参考 core/multibundle.py JointBundleAdjustment 的 2 相机特化;
# 仓库体例照 intrinsic/solve.py)
# ---------------------------------------------------------------------------


def row_index(pair_idx: int, cam_idx: int, point_idx: int,
              coord: int, n_points: int) -> int:
    """残差行号契约:对 → 路(cam1 先 cam2 后)→ 点 → (u,v) 两行交错。

    残差与雅可比共用的唯一行号函数,两侧只准调本函数、不准各自手算下标
    ——行号算错 = 梯度接错行(Pitfall 14 的姊妹坑),不报异常、只是收敛
    奇慢或原地不动。cam_idx 0 = cam1(该块只碰本对位姿列),1 = cam2
    (碰外参列 + 本对位姿列)。
    """
    if cam_idx not in (0, 1):
        raise AssertionError(
            f"cam_idx 只能是 0(cam1)或 1(cam2),实际是 {cam_idx!r}。"
            f"双目残差按「每对先 cam1 块后 cam2 块」排列,行号函数只认识两路。"
        )
    if coord not in (0, 1):
        raise AssertionError(
            f"coord 只能是 0(u 分量)或 1(v 分量),实际是 {coord!r}。"
            f"残差按 (u,v) 两行交错排列,行号函数只认识这两个分量。"
        )
    return 2 * ((pair_idx * 2 + cam_idx) * n_points + point_idx) + coord


def pack_theta(rvec_rel: np.ndarray, t_rel: np.ndarray,
               rvecs_j: np.ndarray, tvecs_j: np.ndarray) -> np.ndarray:
    """外参 + 逐对位姿 → 联合单向量 θ = [外参 (ω_rel,t_rel) 6 个 | 对 j 的 6 个 | …]。

    cam1 恒为单位位姿 (I, 0),是参考系本身、不是被估对象,不进 θ——
    N 对 → 6 + 6N 维 float64。不做「外参/位姿分块交替优化」(Don't
    Hand-Roll,与 Phase 2 同一裁决)。
    """
    r = np.asarray(rvec_rel, dtype=np.float64).ravel()
    t = np.asarray(t_rel, dtype=np.float64).ravel()
    rv = np.asarray(rvecs_j, dtype=np.float64)
    tv = np.asarray(tvecs_j, dtype=np.float64)
    if r.shape != (3,) or t.shape != (3,):
        raise AssertionError(
            f"外参形状不合:期望 rvec_rel/t_rel 各 (3,),实际是 {r.shape} / {t.shape}。"
            f"旋转向量与平移向量必须各 3 个数(Rodrigues 语义)。"
        )
    if rv.ndim != 2 or rv.shape[1] != 3 or rv.shape[0] < 1 or tv.shape != rv.shape:
        raise AssertionError(
            f"rvecs_j/tvecs_j 形状不合:期望各 (对数,3) 且对数 >= 1,实际是 "
            f"{rv.shape} / {tv.shape}。请检查位姿是否按对逐行给出。"
        )
    pieces = [r, t]
    for j in range(rv.shape[0]):
        pieces.append(np.concatenate([rv[j], tv[j]]))  # 对 j 的 (ω,t) 6 个连排
    return np.concatenate(pieces)


def unpack_theta(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """θ (6+6N,) → (rvec_rel, t_rel, rvecs_j, tvecs_j);与 pack_theta 互逆。

    长度对不上任何对数时直接中文报错——装箱与解箱必须同一布局,错位在
    这里点名,不带病进优化器。
    """
    th = np.asarray(theta, dtype=np.float64).ravel()
    if th.shape[0] < 12 or (th.shape[0] - 6) % 6 != 0:
        raise AssertionError(
            f"θ 向量长度不合:应为 外参6 + 6×对数,实际 {th.shape[0]}"
            f"(对不上任何对数)。请检查 θ 是否由 pack_theta 按同一布局装出。"
        )
    n_pairs = (th.shape[0] - 6) // 6
    rvec_rel = th[:3].copy()
    t_rel = th[3:6].copy()
    rvecs = np.empty((n_pairs, 3), dtype=np.float64)
    tvecs = np.empty((n_pairs, 3), dtype=np.float64)
    for j in range(n_pairs):
        chunk = th[6 + 6 * j: 6 + 6 * (j + 1)]
        rvecs[j] = chunk[:3]   # 前 3 个 = 该对旋转向量 ω(Rodrigues 语义)
        tvecs[j] = chunk[3:]   # 后 3 个 = 该对平移向量 t
    return rvec_rel, t_rel, rvecs, tvecs


def make_residual_fn(objp: np.ndarray, uv1: np.ndarray, uv2: np.ndarray,
                     params1: np.ndarray, params2: np.ndarray):
    """残差函数工厂:callable(θ) → (4·N·P,) 残差向量,float64。

    残差 = 两路重投影误差,只由 models.transform_to_camera + models.project
    构成(单一数学源,源码防线锁定):
      cam1:X_cam1 = transform_to_camera(objp, ω_j, t_j),project(·, params1) − uv1_j
      cam2:X_cam2 = R_rel @ X_cam1 + t_rel,project(·, params2) − uv2_j
    行序 = row_index 契约(每对先 cam1 块 2P 行、后 cam2 块 2P 行;块内
    (n,2) ravel 天然是 (u,v) 交错)。
    """
    objp_arr = np.asarray(objp, dtype=np.float64)
    u1 = np.asarray(uv1, dtype=np.float64)
    u2 = np.asarray(uv2, dtype=np.float64)
    if u1.ndim != 3 or u1.shape[2] != 2 or u2.shape != u1.shape:
        raise ValueError(
            f"双目观测形状不合:期望 uv1/uv2 各 (对数,角点数,2),实际是 "
            f"{u1.shape} / {u2.shape}。请检查调用方的数据排布。"
        )
    n_pairs, n_points = u1.shape[0], u1.shape[1]
    p1 = np.asarray(params1, dtype=np.float64).ravel()
    p2 = np.asarray(params2, dtype=np.float64).ravel()

    def residual_fn(theta: np.ndarray) -> np.ndarray:
        rvec_rel, t_rel, rvecs_j, tvecs_j = unpack_theta(theta)
        R_rel, _ = cv2.Rodrigues(rvec_rel)
        out = np.empty(4 * n_pairs * n_points, dtype=np.float64)
        for j in range(n_pairs):
            x1, _ = models.transform_to_camera(objp_arr, rvecs_j[j], tvecs_j[j])
            x2 = x1 @ R_rel.T + t_rel  # X_cam2 = R_rel @ X_cam1 + t_rel(行向量式)
            out[j * 4 * n_points: j * 4 * n_points + 2 * n_points] = (
                models.project(x1, p1) - u1[j]).ravel()
            out[j * 4 * n_points + 2 * n_points:(j + 1) * 4 * n_points] = (
                models.project(x2, p2) - u2[j]).ravel()
        return out

    return residual_fn


def _structure_positions(n_pairs: int, n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """全部 (行, 列) 结构位置,按 [对0 cam1 块、对0 cam2 块、对1 …] 顺序展平。

    结构事实:cam1 行恰碰本对 6 个位姿列(nnz = 6);cam2 行碰外参 6 列 +
    本对 6 个位姿列(nnz = 12);别对位姿列与 cam1 行的外参列结构为零。
    行序同 row_index 契约(Pitfall 14:残差与雅可比共用行序,错序不报错
    只收敛奇慢)。
    """
    row_cam1 = np.arange(2 * n_points)
    row_cam2 = np.arange(2 * n_points, 4 * n_points)
    col_ext = np.arange(6)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    for j in range(n_pairs):
        base = j * 4 * n_points
        col_pose = np.arange(6 + 6 * j, 6 + 6 * (j + 1))
        rows.append(np.broadcast_to((base + row_cam1)[:, None], (2 * n_points, 6)))
        cols.append(np.broadcast_to(col_pose, (2 * n_points, 6)))
        rows.append(np.broadcast_to((base + row_cam2)[:, None], (2 * n_points, 12)))
        cols.append(np.broadcast_to(np.concatenate([col_ext, col_pose]),
                                    (2 * n_points, 12)))
    return (np.concatenate([r.ravel() for r in rows]),
            np.concatenate([c.ravel() for c in cols]))


def make_jacobian_fn(objp: np.ndarray, uv1: np.ndarray, uv2: np.ndarray,
                     params1: np.ndarray, params2: np.ndarray):
    """解析稀疏雅可比工厂:callable(θ) → scipy.sparse.csr_matrix,形状 (4NP, 6+6N)。

    逐对组装(链式法则,全部构件来自 models 与 cv2.Rodrigues 第二返回值,
    严禁手写 ∂Xc/∂ω = −[Xc]× 形式——那只在 ω=0 成立):
    - cam1 位姿 6 列 = project_point_jacobian (P,2,3) · ∂X1/∂(ω,t) (P,3,6);
    - cam2 位姿 6 列:∂X2/∂pose_j = R_rel @ ∂X1/∂pose_j(外参组合是线性
      变换),再链 cam2 的投影点雅可比;
    - cam2 外参 6 刢:∂X2/∂ω_rel[k] = Σ_j' rot_jac_rel[k,i,j']·X1_j'
      (einsum 一次算完,与 models.transform_to_camera 同一模式)、
      ∂X2/∂t_rel = 单位阵,再链投影点雅可比;内参固定,无内参列。
    COO 三元组向量化拼接后转 csr;数值正确性由 θ 全维差分基准测试锁定
    (相对差 < 1e-6)。决策 1:解析雅可比路线,绝不传稀疏结构矩阵参数
    (jac 为 callable 时 scipy 对它静默忽略——传了是死代码,WR-02 教训;
    源码防线测试锁定)。
    """
    objp_arr = np.asarray(objp, dtype=np.float64)
    u1 = np.asarray(uv1, dtype=np.float64)
    u2 = np.asarray(uv2, dtype=np.float64)
    if u1.ndim != 3 or u1.shape[2] != 2 or u2.shape != u1.shape:
        raise ValueError(
            f"双目观测形状不合:期望 uv1/uv2 各 (对数,角点数,2),实际是 "
            f"{u1.shape} / {u2.shape}。请检查调用方的数据排布。"
        )
    n_pairs, n_points = u1.shape[0], u1.shape[1]
    p1 = np.asarray(params1, dtype=np.float64).ravel()
    p2 = np.asarray(params2, dtype=np.float64).ravel()
    shape = (4 * n_pairs * n_points, 6 + 6 * n_pairs)
    rows, cols = _structure_positions(n_pairs, n_points)
    eye3 = np.eye(3, dtype=np.float64)

    def jacobian_fn(theta: np.ndarray) -> csr_matrix:
        rvec_rel, _t_rel, rvecs_j, tvecs_j = unpack_theta(theta)
        R_rel, rot_jac_rel = cv2.Rodrigues(rvec_rel)  # 第二返回值 = 现成雅可比
        vals: list[np.ndarray] = []
        for j in range(n_pairs):
            x1, dxc1 = models.transform_to_camera(
                objp_arr, rvecs_j[j], tvecs_j[j], want_jacobian=True)
            x2 = x1 @ R_rel.T + _t_rel
            jp1 = models.project_point_jacobian(x1, p1)   # (P,2,3)
            jp2 = models.project_point_jacobian(x2, p2)   # (P,2,3)
            jac1_pose = np.einsum("nij,njk->nik", jp1, dxc1)          # (P,2,6)
            dxc2_pose = np.einsum("ij,njk->nik", R_rel, dxc1)         # (P,3,6)
            jac2_pose = np.einsum("nij,njk->nik", jp2, dxc2_pose)     # (P,2,6)
            dxc2_ext = np.empty((x1.shape[0], 3, 6), dtype=np.float64)
            dxc2_ext[:, :, 0:3] = np.einsum("kij,nj->nik", rot_jac_rel.reshape(3, 3, 3), x1)
            dxc2_ext[:, :, 3:6] = eye3
            jac2_ext = np.einsum("nij,njk->nik", jp2, dxc2_ext)       # (P,2,6)
            vals.append(jac1_pose.ravel())  # 展平顺序与 _structure_positions 一致
            vals.append(np.concatenate([jac2_ext, jac2_pose], axis=2).ravel())
        return coo_matrix((np.concatenate(vals), (rows, cols)), shape=shape).tocsr()

    return jacobian_fn


@dataclass(frozen=True)
class StereoResult:
    """双目求解终解(不可变):cam1 参考系外参 + 全部诊断留痕。

    R (3,3) / t (3,) / rvec (3,) / rotation_angle_deg / baseline_mm:语义同
    StereoInit(X_cam2 = R @ X_cam1 + t,t 单位 mm,baseline = ‖t‖)。
    per_pair_rms (N,) 与输入 pairs 行序一一对应(float64 回代);rms_reproj
    为 pooled 口径(全部对两路平方欧氏残差均值的平方根,非逐对简写平均)。
    flip_log 顶层固定含 "train" 键(本引擎即 train 侧语义;test 侧不调它)。
    solver 如实标注解的来源("pnp_median" 初值 | "scipy" 精化);solver_meta
    留收敛判定与雅可比组合依据,不静默(T-03-06)。
    """

    R: np.ndarray
    t: np.ndarray
    rvec: np.ndarray
    rotation_angle_deg: float
    baseline_mm: float
    per_pair_rms: np.ndarray
    rms_reproj: float
    flip_log: dict
    solver: str
    solver_meta: dict


def refine_stereo(pairs: Pairs, intr1: IntrinsicsView, intr2: IntrinsicsView,
                  init: StereoInit, refine_cfg: StereoRefineConfig) -> StereoResult:
    """scipy 外参精化:θ = [6 外参 | 6×N 位姿] 联合最小二乘(STER-01)。

    白话:固定两路内参,把「cam1→cam2 外参 6 个 + 每对棋盘位姿 6 个」装成
    一个向量,交给 least_squares 的 trf(信赖域反射)+ lsmr(解稀疏线性
    子问题)+ 解析稀疏雅可比联合优化;初值来自 PnP-median(pass-2)。

    失败契约与 Phase 2 一致:数值路径异常如实上抛(中文),收敛判定依据
    全进 solver_meta 不静默;loss 写死 linear(离群对由 03-03 剔除机制
    处理,不引入鲁棒核把解压歪)。
    """
    uv1 = np.asarray(pairs.uv1, dtype=np.float64)
    uv2 = np.asarray(pairs.uv2, dtype=np.float64)
    objp = np.asarray(pairs.objp, dtype=np.float64)
    n_pairs, n_points = uv1.shape[0], uv1.shape[1]
    if (init.rvecs1.shape != (n_pairs, 3) or init.tvecs1.shape != (n_pairs, 3)
            or not all(init.per_pair_ok)):
        raise ValueError(
            f"精化初值与观测不一致:初值 {init.rvecs1.shape} 对不上 {n_pairs} 个对,"
            f"或存在 PnP 失败对(位姿初值缺失)。精化必须来自同一批对、全部对均有解"
            f"的 pass-2 结果;失败对应先进剔除通道。"
        )

    theta0 = pack_theta(init.rvec, init.t, init.rvecs1, init.tvecs1)
    residual_fn = make_residual_fn(objp, uv1, uv2, intr1.params, intr2.params)

    # 每次评估把 0.5·r·r 追加进代价轨迹(与 scipy 的代价定义一致),
    # 供实用收敛判定的尾部窗口使用(照 intrinsic/solve.py 体例)。
    cost_trace: list[float] = []

    def traced_residual(theta: np.ndarray) -> np.ndarray:
        r = residual_fn(theta)
        cost_trace.append(0.5 * float(r @ r))
        return r

    sol = least_squares(
        traced_residual, theta0,
        jac=make_jacobian_fn(objp, uv1, uv2, intr1.params, intr2.params),
        # 不传稀疏结构矩阵参数:jac 为 callable 时 scipy 对它静默忽略(死计算);
        # 稀疏性利用只经「callable jac 返回 csr + tr_solver 解稀疏子问题」
        # 这一条真实机制(见 make_jacobian_fn)。
        method=refine_cfg.method, tr_solver="lsmr", x_scale="jac",
        # 离群对由 03-03 的剔除机制处理,不引入鲁棒核(loss 写死 linear;
        # 若未来需要鲁棒核走配置化扩展,不在这里塞分支)
        loss="linear",
        max_nfev=refine_cfg.max_nfev,
        ftol=refine_cfg.ftol, xtol=refine_cfg.xtol, gtol=refine_cfg.gtol)

    rvec_rel, t_rel, rvecs_j, tvecs_j = unpack_theta(sol.x)
    R_rel, _ = cv2.Rodrigues(rvec_rel)

    # 最终位姿逐对 RMS 重算与 pooled 指标(float64 回代,只经 models;
    # 绝不用 OpenCV 返回值当指标)。
    p1 = np.asarray(intr1.params, dtype=np.float64).ravel()
    p2 = np.asarray(intr2.params, dtype=np.float64).ravel()
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

    tail = _tail_improvement(cost_trace)
    meta = {
        "path": "stereo_refine", "method": refine_cfg.method,
        "tr_solver": "lsmr", "x_scale": "jac", "loss": "linear",
        "status": int(sol.status), "nfev": int(sol.nfev), "njev": int(sol.njev),
        "final_cost": float(sol.cost), "tail_improvement": float(tail),
        "n_cost_evals": len(cost_trace),
        "converged": _is_practically_converged(int(sol.status), tail),
        # 雅可比组合留痕(T-03-06 不静默;决策 1 的路线声明)
        "jacobian": "analytic-jac, no sparsity",
    }
    return StereoResult(
        R=np.asarray(R_rel, dtype=np.float64), t=np.asarray(t_rel, dtype=np.float64),
        rvec=np.asarray(rvec_rel, dtype=np.float64),
        rotation_angle_deg=float(np.degrees(np.linalg.norm(rvec_rel))),
        baseline_mm=float(np.linalg.norm(t_rel)),
        per_pair_rms=per_pair_rms, rms_reproj=rms_reproj,
        flip_log={}, solver="scipy", solver_meta=meta)


def solve_stereo(pairs: Pairs, intr1: IntrinsicsView, intr2: IntrinsicsView,
                 cfg: StereoCalibConfig) -> StereoResult:
    """双目求解编排入口(03-03 唯一消费它):全配置进、终解出。

    内部两遍法管线:pass-1 pnp_median(原始角点,离散度防线关——翻转对
    未消歧时离散度高恰恰是消歧要修的东西)→ disambiguate_pairs(配置
    flip_ratio)→ pass-2 pnp_median(消歧后角点,离散度防线按配置)→
    cfg.stereo.refine.enabled 时 refine_stereo,否则直接组装初值版
    StereoResult(solver 如实标 "pnp_median")。剔除循环内的 fast 路径由
    03-03 先 deepcopy 配置置 refine.enabled=False 再入(精化不进循环)。
    flip_log 顶层固定含 "train" 键(本函数即 train 侧语义;test 侧不调它),
    结构 {"train": {"pass2_flipped": int, "pair_ids": [...]}}。
    """
    stereo = cfg.stereo
    pass1 = pnp_median(pairs, intr1, intr2, None)
    pairs2, flip = disambiguate_pairs(pairs, intr1, intr2, pass1, stereo.flip_ratio)
    pass2 = pnp_median(pairs2, intr1, intr2, stereo.median_spread_max_deg)
    flip_log = {"train": flip}

    if not all(pass2.per_pair_ok):
        failed = [gid for gid, ok in zip(pairs.image_ids, pass2.per_pair_ok) if not ok]
        raise ValueError(
            f"双目求解失败:{len(failed)} 个对在消歧后仍解不出 PnP 位姿"
            f"(group_id:{failed})——精化没有位姿初值,硬算必错。请把它们"
            f"加入 stereo.exclude_pairs 手动剔除清单,或检查角点数据质量。"
        )

    if stereo.refine.enabled:
        return replace(refine_stereo(pairs2, intr1, intr2, pass2, stereo.refine),
                       flip_log=flip_log)
    # 初值出口(精化开关关闭):pass-2 结果原样组装,pooled RMS = 逐对
    # RMS 的平方均值开根(等角点数下与 pooled 公式一致)。
    pooled = float(np.sqrt(np.mean(pass2.per_pair_rms ** 2)))
    return StereoResult(
        R=pass2.R, t=pass2.t, rvec=pass2.rvec,
        rotation_angle_deg=pass2.rotation_angle_deg, baseline_mm=pass2.baseline_mm,
        per_pair_rms=pass2.per_pair_rms, rms_reproj=pooled,
        flip_log=flip_log, solver="pnp_median",
        solver_meta={"path": "pnp_median", "spread_deg": pass2.spread_deg,
                     "n_valid": pass2.n_valid})
