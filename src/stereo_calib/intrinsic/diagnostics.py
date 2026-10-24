"""内参模型充分性体检(model-adequacy battery):仅凭自家数据 + 自家结果,
判定畸变模型是否充分、内参是否可靠(2026-10-24 quick 任务落地)。

根因背景(2026-10-24 晨):M2 五参畸变模型对本相机欠拟合时,每帧自由位姿
+ fx 会吸收模型误差,标量重投影 RMS 完全看不出(cam1 fx 曾被推高 11%,
RMS 依旧 0.14px,直到双目阶段才以相对位姿假散布暴露)。本模块不依赖任何
参考工程,只对「某相机的 train 角点 + 产线解出的 K/D」做四个检查,任意
两项红灯即判「不可靠」(fail-closed,禁止消费该内参进入双目标定):

  A0 残差能量:PnP 逐点池化 RMS ÷ 中心桶(r_norm<0.2)噪声底。结构性
              欠拟合的残差集中在画面边缘,比值显著大于 1;模型充分时 ≈1。
  A1 残差径向结构:残差按归一化半径 5 桶取径向分量均值(带符号)。模型
              充分时各桶均值与噪声底同量级;欠拟合时边缘桶出现系统性均值。
  A2 去畸变直线度:去畸变后棋盘行/列的 SVD 共线最大正交偏差(射影保直
              线)。实测该统计量对本类欠拟合无分辨力(M2 0.20 vs R8 0.15),
              仅作信息项报告,不判红。
  A3 切片 fx 稳定性:fx 若在吸收畸变,会随板位漂移——按每帧板中心切 6
              子集(左/右/上/下/内圈/外圈)各自重标 fx,极差 % 超限判红
              (M8 产线解下切片重标自动带 rational flag,与产线同阶)。
  A4 模型阶次探针:全量数据同协议升阶 rational8 重标,fx 跳变 % 超限判红
              (跳变远超模型内 σ = 低阶模型欠拟合实锤);产线解已是 M8 时,
              探针语义为「cv2 独立重标一致性」——fx 跳变仍检验 fx 吸收
              畸变型欠拟合,只是不再有更高阶可升。

判定规则(阈值全部来自 config.yaml 的 intrinsic.diagnostics 节):
A0/A1/A3/A4 红灯数 ≥ min_red_count(默认 2)→ verdict「不可靠」。
A3/A4 的探针拟合失败、数据不足时如实记 note、不计红——红灯必须来自真实
的判定量,绝不许由故障间接撑出来(fail-closed 的可信度前提)。

口径纪律:全 float64(给 cv2 的 float32 降精度只发生在绑定调用边界,与
solve.py 同款);体检只读数据与产线结果,绝不改写;不 import models.py
的 M2 契约——A3/A4 探针拟合直接用 cv2,与产线求解器/模型层解耦。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise

import cv2
import numpy as np

# A1 的归一化半径分桶:r_norm ∈ [0,0.2,…,1.0) 共 5 桶(口径与 PLAN 一致)
_A1_BIN_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
# A0 中心噪声底桶的半径上界
_A0_CENTER_RADIUS = 0.2
# A3 有效切片数下限:少于它判「数据不足」不计红(极差无统计意义)
_A3_MIN_SUBSETS = 4
# A3 的 6 个切片名(按每帧板中心相对画面中心的方位)
_A3_SPLIT_NAMES = ("左半", "右半", "上半", "下半", "内圈", "外圈")


@dataclass(frozen=True)
class CheckResult:
    """单项检查结果:判定量、阈值、红灯布尔与报告附加数值(detail)。"""

    name: str                 # "A0" / "A1" / "A2" / "A3" / "A4"
    label: str                # 中文名(报告/报错用)
    value: float | None       # 判定量;None = 数据不足或探针失败(note 说明)
    threshold: float | None   # 红线阈值(A2 无阈值 → None)
    red: bool                 # 是否红灯(A2 恒 False)
    note: str = ""            # 数据不足/失败等说明(空串 = 无)
    detail: dict = field(default_factory=dict)  # 附加数值(JSON 可序列化)


@dataclass(frozen=True)
class DiagnosticsReport:
    """一次体检的完整报告:逐检查结果 + verdict(fail-closed 判定结论)。"""

    enabled: bool                       # 体检开关(false 时只此一项有效)
    checks: tuple[CheckResult, ...]     # A0/A1/A2/A3/A4 逐项
    red_count: int                      # 红灯数(A0/A1/A3/A4)
    red_names: tuple[str, ...]          # 红灯检查名清单(报错/报告用)
    reliable: bool | None               # True=可靠 / False=不可靠 / None=未体检
    verdict: str                        # "可靠" / "不可靠" / "未体检"
    notes: tuple[str, ...]              # 全局说明(开关关闭、数据不足等)


def _as_float64(arr, name: str, shape: tuple | None = None) -> np.ndarray:
    """入口归一:统一 float64 并按需断言形状(中文报错,绝不返回半成品)。"""
    a = np.asarray(arr, dtype=np.float64)
    if shape is not None and a.shape != shape:
        raise ValueError(
            f"体检输入 {name} 形状不合:期望 {shape},实际是 {a.shape}。"
            f"请检查调用方传入的数组排布(uv=(帧数,角点数,2)、objp=(角点数,3)、"
            f"K=3×3、D=5 长 M2 / 8 长 M8 布局)。"
        )
    return a


def _validate_inputs(uv, objp, K, D, image_size) -> tuple[np.ndarray, np.ndarray,
                                                          np.ndarray, np.ndarray,
                                                          tuple[int, int]]:
    """体检入口的形状/取值防线(与 solve._check_common_inputs 同风格的轻量版)。"""
    uv_arr = _as_float64(uv, "uv")
    if uv_arr.ndim != 3 or uv_arr.shape[2] != 2 or uv_arr.shape[0] < 1:
        raise ValueError(
            f"体检输入 uv 形状不合:期望 (帧数,角点数,2) 且帧数 >= 1,实际是 "
            f"{uv_arr.shape}。体检消费的是 train 幸存视图角点,请检查调用方。"
        )
    objp_arr = _as_float64(objp, "objp")
    if objp_arr.ndim != 2 or objp_arr.shape[1] != 3:
        raise ValueError(
            f"体检输入 objp 形状不合:期望 (角点数,3),实际是 {objp_arr.shape}。")
    if uv_arr.shape[1] != objp_arr.shape[0]:
        raise ValueError(
            f"体检输入角点与物点数不一致:uv 每帧 {uv_arr.shape[1]} 点,"
            f"objp {objp_arr.shape[0]} 点。两者必须逐点对齐。")
    K_arr = _as_float64(K, "K", shape=(3, 3))
    D_arr = _as_float64(D, "D")
    if D_arr.shape not in ((5,), (8,)):
        # 布局分派:M2 五参 / M8 rational 8(2026-10-24 增补),其余长度一律拒收
        raise ValueError(
            f"体检输入 D 形状不合:期望 (5,)(M2 五参)或 (8,)(M8 rational 8),"
            f"实际是 {D_arr.shape}。请检查调用方传入的畸变布局与产线 "
            f"distortion_model 配置是否一致。")
    size = tuple(image_size)
    if len(size) != 2 or any(int(v) != v or v <= 0 for v in size):
        raise ValueError(
            f"体检输入 image_size 不合法:期望 (宽, 高) 两个正整数,实际是 "
            f"{image_size!r}(OpenCV 口径,与 numpy 形状的 (高, 宽) 相反)。")
    if not (np.isfinite(uv_arr).all() and np.isfinite(objp_arr).all()
            and np.isfinite(K_arr).all() and np.isfinite(D_arr).all()):
        raise ValueError(
            "体检输入含 NaN/Inf:体检只接受有限数值(train 幸存视图在装载层已做"
            "防线,K/D 来自定稿求解,出现非数值说明上游防线被击穿,请反馈维护者)。")
    return uv_arr, objp_arr, K_arr, D_arr, (int(size[0]), int(size[1]))


def _per_point_residuals(uv: np.ndarray, objp: np.ndarray, K: np.ndarray,
                         D: np.ndarray) -> np.ndarray:
    """逐点残差(公共前置):每帧 PnP 估位姿 → projectPoints 回代 → uv − 投影。

    协议(与已验证体检脚本一致):solvePnP 用 SOLVEPNP_ITERATIVE,cv2 绑定
    边界降 float32 装包,projectPoints 回代 float64;返回 (F,P,2) float64。
    某帧 PnP 失败属数值路径异常(数据已过装载与求解防线,此处失败说明
    K/D 病态到 PnP 都不稳),中文 ValueError 上抛由编排层单路收口。
    """
    objp32 = np.ascontiguousarray(objp, dtype=np.float32).reshape(-1, 1, 3)
    dist = D.ravel().reshape(-1, 1)
    residuals: list[np.ndarray] = []
    for i, pts in enumerate(uv):
        pts32 = np.ascontiguousarray(pts, dtype=np.float32).reshape(-1, 1, 2)
        ok, rvec, tvec = cv2.solvePnP(objp32, pts32, K, dist,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            raise ValueError(
                f"体检第 {i} 帧 solvePnP 求位姿失败(OpenCV 返回 ok=False):"
                f"产线 K/D 对该帧角点连位姿都解不稳,内参可靠性无从谈起。"
                f"请检查该帧角点与标定结果。")
        proj, _ = cv2.projectPoints(objp.astype(np.float64), rvec, tvec, K, dist)
        residuals.append(pts - proj.reshape(-1, 2))
    return np.stack(residuals)


def _refit_fx(uv_sub: np.ndarray, objp: np.ndarray, K: np.ndarray, D: np.ndarray,
              image_size: tuple[int, int], rational: bool) -> tuple[float, float]:
    """同协议重标一份子集/全量数据,返回 (fx, cv2 整体 RMS)。

    flags=CALIB_USE_INTRINSIC_GUESS(初值吃产线 K/D);rational=True 或
    产线 D 为 8 长(M8)时加 CALIB_RATIONAL_MODEL——不给该 flag 时 OpenCV
    对 8 长布局会静默截断,故 M8 产线解下探针恒为同阶 rational8;M2 五长
    产线值按 rational 开关决定升阶与否(A4 探针,D0 仍传 5 长产线值,
    OpenCV 内部扩到 8 系)。拟合失败以 cv2.error 上抛,由调用方决定记
    note 或计数。
    """
    n = uv_sub.shape[0]
    use_rational = bool(rational) or np.asarray(D).size == 8
    flags = cv2.CALIB_USE_INTRINSIC_GUESS | (cv2.CALIB_RATIONAL_MODEL if use_rational else 0)
    objs32 = [np.ascontiguousarray(objp, dtype=np.float32).reshape(-1, 1, 3)
              for _ in range(n)]
    imgs32 = [np.ascontiguousarray(p, dtype=np.float32).reshape(-1, 1, 2)
              for p in uv_sub]
    K0 = np.array(K, dtype=np.float64, copy=True)
    D0 = np.array(D, dtype=np.float64, copy=True).ravel().reshape(-1, 1)
    rms, K_fit, _D_fit, *_ = cv2.calibrateCameraExtended(
        objs32, imgs32, image_size, K0, D0, flags=flags)
    fx = float(np.asarray(K_fit, dtype=np.float64)[0, 0])
    return fx, float(rms)


def _check_a0(res: np.ndarray, r_norm: np.ndarray, cfg) -> CheckResult:
    """A0 残差能量:pooled RMS ÷ 中心桶噪声底。红:> a0_max_ratio。"""
    sq = np.sum(res * res, axis=-1)                       # (F,P) 平方欧氏残差
    pooled = float(np.sqrt(sq.mean()))
    center = r_norm < _A0_CENTER_RADIUS
    if not center.any():
        return CheckResult("A0", "残差能量比", None, float(cfg.a0_max_ratio), False,
                           note=f"中心桶(r_norm<{_A0_CENTER_RADIUS})没有角点,"
                                f"噪声底无从取,本检查不计红")
    floor = float(np.sqrt(sq[center].mean()))
    ratio = pooled / floor
    return CheckResult("A0", "残差能量比", ratio, float(cfg.a0_max_ratio),
                       bool(ratio > cfg.a0_max_ratio),
                       detail={"pooled_rms_px": pooled, "center_floor_px": floor})


def _check_a1(res: np.ndarray, r_norm: np.ndarray, uv: np.ndarray,
              center: np.ndarray, cfg) -> CheckResult:
    """A1 残差径向结构:5 桶径向分量均值取最大绝对值。红:> a1_max_bin_mean_px。"""
    diff = uv - center                                    # (F,P,2)
    norm = np.linalg.norm(diff, axis=-1, keepdims=True)
    uhat = np.where(norm > 0.0, diff / np.where(norm > 0.0, norm, 1.0), 0.0)
    radial = np.sum(res * uhat, axis=-1)                  # (F,P) 径向分量
    bin_means: list[float] = []
    bin_counts: list[int] = []
    for lo, hi in pairwise(_A1_BIN_EDGES):
        m = (r_norm >= lo) & (r_norm < hi)
        if not m.any():
            bin_means.append(float("nan"))
            bin_counts.append(0)
            continue
        bin_means.append(float(radial[m].mean()))
        bin_counts.append(int(m.sum()))
    finite = [abs(v) for v in bin_means if np.isfinite(v)]
    worst = max(finite) if finite else float("nan")
    return CheckResult("A1", "残差径向结构", worst, float(cfg.a1_max_bin_mean_px),
                       bool(np.isfinite(worst) and worst > cfg.a1_max_bin_mean_px),
                       detail={"bin_means_px": bin_means, "bin_counts": bin_counts,
                               "bin_edges": list(_A1_BIN_EDGES)})


def _check_a2(uv: np.ndarray, objp: np.ndarray, K: np.ndarray, D: np.ndarray) -> CheckResult:
    """A2 去畸变直线度(仅报告不判红):去畸变后行/列 SVD 共线最大偏差中位数。"""
    dist = D.ravel().reshape(-1, 1)
    ys = np.unique(objp[:, 1])
    xs = np.unique(objp[:, 0])
    groups = [np.where(np.isclose(objp[:, 1], y))[0] for y in ys]
    groups += [np.where(np.isclose(objp[:, 0], x))[0] for x in xs]
    worst_per_frame: list[float] = []
    for pts in uv:
        und = cv2.undistortPoints(
            np.ascontiguousarray(pts, dtype=np.float64).reshape(-1, 1, 2),
            K, dist, None, K).reshape(-1, 2)
        devs = []
        for g in groups:
            q = und[g] - und[g].mean(axis=0)
            _u, _s, vt = np.linalg.svd(q, full_matrices=False)
            devs.append(float(np.abs(q @ vt[1]).max()))
        worst_per_frame.append(max(devs))
    median_dev = float(np.median(np.asarray(worst_per_frame, dtype=np.float64)))
    return CheckResult("A2", "去畸变直线度", median_dev, None, False,
                       note="仅报告不判红(实测对 M2 欠拟合无分辨力)",
                       detail={"worst_line_px_median": median_dev,
                               "worst_line_px_max": float(max(worst_per_frame))})


def _check_a3(uv: np.ndarray, objp: np.ndarray, K: np.ndarray, D: np.ndarray,
              image_size: tuple[int, int], center: np.ndarray, cfg) -> CheckResult:
    """A3 切片 fx 稳定性:6 切片各重标 fx,极差 %。红:> a3_max_fx_range_pct。

    单切片帧数 < a3_min_frames 跳过;有效切片 < 4 → 「数据不足」不计红
    (红灯必须来自真实判定量)。单切片拟合失败(cv2.error)记 note 跳过。
    产线 D 为 8 长(M8)时切片重标自动带 rational flag(与产线同阶,
    见 _refit_fx)——切片间 fx 极差度量的是「fx 吸收畸变随板位漂移」,
    与模型阶次无关。
    """
    u_c = uv.mean(axis=1)[:, 0]
    v_c = uv.mean(axis=1)[:, 1]
    r_c = np.linalg.norm(uv.mean(axis=1) - center, axis=1)
    med = float(np.median(r_c))
    masks = {
        "左半": u_c < center[0], "右半": u_c >= center[0],
        "上半": v_c < center[1], "下半": v_c >= center[1],
        "内圈": r_c < med, "外圈": r_c >= med,
    }
    notes: list[str] = []
    detail: dict = {}
    fxs: list[float] = []
    for name in _A3_SPLIT_NAMES:
        m = masks[name]
        n = int(m.sum())
        if n < cfg.a3_min_frames:
            notes.append(f"{name}切片仅 {n} 帧(< a3_min_frames={cfg.a3_min_frames}),跳过")
            continue
        try:
            fx, _rms = _refit_fx(uv[m], objp, K, D, image_size, rational=False)
        except cv2.error as e:
            notes.append(f"{name}切片重标失败(OpenCV 报错),跳过:{e}")
            continue
        fxs.append(fx)
        detail[name] = {"fx": fx, "n_frames": n}
    if len(fxs) < _A3_MIN_SUBSETS:
        notes.append(f"有效切片仅 {len(fxs)} 个(<{_A3_MIN_SUBSETS}),数据不足,不计红")
        return CheckResult("A3", "切片 fx 稳定性", None, float(cfg.a3_max_fx_range_pct),
                           False, note=";".join(notes), detail=detail)
    fxs_arr = np.asarray(fxs, dtype=np.float64)
    range_pct = float((fxs_arr.max() - fxs_arr.min()) / fxs_arr.mean() * 100.0)
    return CheckResult("A3", "切片 fx 稳定性", range_pct, float(cfg.a3_max_fx_range_pct),
                       bool(range_pct > cfg.a3_max_fx_range_pct),
                       note=";".join(notes), detail=detail)


def _check_a4(uv: np.ndarray, objp: np.ndarray, K: np.ndarray, D: np.ndarray,
              image_size: tuple[int, int], cfg) -> CheckResult:
    """A4 模型阶次探针:升阶 rational8 全量重标,fx 跳变 %。红:> a4_max_fx_jump_pct。

    M2 产线解:升阶探针(rational8 vs 产线五参);M8 产线解(D 8 长):
    同阶独立重标一致性探针(cv2 rational8 vs 产线 scipy M8——fx 跳变仍
    检验 fx 吸收畸变型欠拟合,不再有更高阶可升)。
    探针拟合失败(cv2 异常/非有限解)→ 记 note 不计红(探针故障不许撑红灯)。
    """
    fx_prod = float(K[0, 0])
    same_order = np.asarray(D).size == 8
    probe_label = "同阶 rational8 独立重标探针" if same_order else "升阶 rational8 探针"
    try:
        fx8, _rms = _refit_fx(uv, objp, K, D, image_size, rational=True)
    except cv2.error as e:
        return CheckResult("A4", "模型阶次探针", None, float(cfg.a4_max_fx_jump_pct),
                           False, note=f"{probe_label}拟合失败(OpenCV 报错),不计红:{e}")
    if not np.isfinite(fx8) or fx8 <= 0.0:
        return CheckResult("A4", "模型阶次探针", None, float(cfg.a4_max_fx_jump_pct),
                           False, note=f"{probe_label}解非有限(fx8={fx8}),不计红")
    jump_pct = float(abs(fx_prod - fx8) / fx8 * 100.0)
    return CheckResult("A4", "模型阶次探针", jump_pct, float(cfg.a4_max_fx_jump_pct),
                       bool(jump_pct > cfg.a4_max_fx_jump_pct),
                       note="产线 M8:探针与产线同阶(独立重标一致性)" if same_order else "",
                       detail={"fx_production": fx_prod, "fx_rational8": fx8,
                               "probe": "same_order_rational8" if same_order
                                        else "upgrade_to_rational8"})


def _finalize(checks: tuple[CheckResult, ...], notes: tuple[str, ...],
              min_red_count: int, enabled: bool) -> DiagnosticsReport:
    """verdict 判定(唯一裁决点):红检 A0/A1/A3/A4 计数,≥ min_red_count 判不可靠。"""
    red_names = tuple(c.name for c in checks if c.red)
    red_count = len(red_names)
    if not enabled:
        return DiagnosticsReport(enabled=False, checks=checks, red_count=0,
                                 red_names=(), reliable=None, verdict="未体检",
                                 notes=notes)
    reliable = red_count < min_red_count
    verdict = "可靠" if reliable else "不可靠"
    extra = (f"红灯 {','.join(red_names)} 共 {red_count} 项,"
             f"达到判定线 min_red_count={min_red_count}" if not reliable else
             f"红灯 {red_count} 项,低于判定线 min_red_count={min_red_count}")
    return DiagnosticsReport(enabled=True, checks=checks, red_count=red_count,
                             red_names=red_names, reliable=reliable, verdict=verdict,
                             notes=notes + (extra,))


def run_battery(uv, objp, K, D, image_size, cfg) -> DiagnosticsReport:
    """体检唯一入口:train 幸存角点 + 产线 K/D → 五检查 + verdict。

    参数:
        uv:          (F,P,2) float64,某相机 train 幸存视图角点(像素);
        objp:        (P,3) float64,与角点同序的板系物点;
        K / D:       产线定稿解出的 3×3 内参矩阵与畸变向量(M2 5 长 /
                     M8 8 长,按长度分派);
        image_size:  (宽, 高);
        cfg:         IntrinsicDiagnosticsConfig(阈值与判定规则全部来自配置)。
    返回:
        DiagnosticsReport(检查数值/阈值/red/note/verdict/red_count);
        cfg.enabled=False 时返回「未体检」报告,不做任何计算(跳过路径)。
    """
    if not cfg.enabled:
        disabled_note = ("体检开关关闭(intrinsic.diagnostics.enabled: false),"
                         "本轮未体检,内参可靠性未判定")
        return _finalize((), (disabled_note,), cfg.min_red_count, False)
    uv_arr, objp_arr, K_arr, D_arr, size = _validate_inputs(uv, objp, K, D, image_size)
    width, height = size
    center = np.array([width / 2.0, height / 2.0], dtype=np.float64)
    half_diag = float(np.hypot(center[0], center[1]))

    res = _per_point_residuals(uv_arr, objp_arr, K_arr, D_arr)
    diff = uv_arr - center
    r_norm = np.linalg.norm(diff, axis=-1) / half_diag     # (F,P)

    checks = (
        _check_a0(res, r_norm, cfg),
        _check_a1(res, r_norm, uv_arr, center, cfg),
        _check_a2(uv_arr, objp_arr, K_arr, D_arr),
        _check_a3(uv_arr, objp_arr, K_arr, D_arr, size, center, cfg),
        _check_a4(uv_arr, objp_arr, K_arr, D_arr, size, cfg),
    )
    return _finalize(checks, (), cfg.min_red_count, True)


def _jsonify_value(v):
    """单个值转 Python 原生类型:np 整数→int、数值→float、NaN→None、
    字符串原样(A4 探针语义标注等文字型 detail)。"""
    if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
        return int(v)
    if isinstance(v, float | np.floating):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, str):
        return v
    return float(v)


def _jsonify_detail(detail: dict) -> dict:
    """detail 统一转 JSON 原生形态:list 项保留 int、NaN→null、切片 dict 原样
    递归(A3 的每切片 {"fx":…, "n_frames":…})。"""
    out: dict = {}
    for k, v in detail.items():
        if isinstance(v, dict):
            out[k] = _jsonify_detail(v)
        elif isinstance(v, list):
            out[k] = [_jsonify_value(x) for x in v]
        else:
            out[k] = _jsonify_value(v)
    return out


def diagnostics_to_dict(report: DiagnosticsReport) -> dict:
    """报告 → JSON 可序列化 dict(intrinsics.json 的 diagnostics 键契约)。"""
    return {
        "enabled": bool(report.enabled),
        "verdict": str(report.verdict),
        "reliable": None if report.reliable is None else bool(report.reliable),
        "red_count": int(report.red_count),
        "red_names": [str(n) for n in report.red_names],
        "notes": [str(n) for n in report.notes],
        "checks": [
            {
                "name": c.name,
                "label": c.label,
                "value": None if c.value is None else float(c.value),
                "threshold": None if c.threshold is None else float(c.threshold),
                "red": bool(c.red),
                "note": str(c.note),
                "detail": _jsonify_detail(c.detail),
            }
            for c in report.checks
        ],
    }
