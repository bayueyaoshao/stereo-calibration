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
              (M8 产线解下切片重标自动带 rational flag,与产线同阶;且
              冻结全部畸变系数只放 fx,fy+位姿——切片全自由重拟合在
              rational 近简并谷下 fx 弱可辨识、会沿谷漂移放大极差,
              2026-10-24 追加轮实证与冻结协议见 _refit_fx)。
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

# 模型名 → OpenCV dist 向量长度(2026-10-26 增补 M4 鱼眼)。
# 三个长度互不相同,但语义完全不同:4 长是鱼眼 θ 多项式 [k1,k2,k3,k4],
# 5/8 长是针孔 [k1,k2,p1,p2,k3(,k4,k5,k6)]——因此按长度反推模型是安全的,
# 而「拿鱼眼 D 去喂针孔 cv2 调用」必须当场拦下(它会被安静地当成
# [k1,k2,p1,p2],结果看着像正常数、实则错模型)。
_EXPECTED_D_LEN = {"M2": 5, "M8": 8, "M4": 4}
# 长度 → 模型:只登记**针孔族**的两条(M2/M8)。4 长一律不许反推——它是鱼眼
# 族的长度,而同一长度在针孔 cv2 眼里会被当成 [k1,k2,p1,p2];要声明"这是 M4"
# 必须显式传 model="M4"(旧测试与产线调用形态都不靠猜语义)。
_D_LEN_TO_MODEL = {n: m for m, n in _EXPECTED_D_LEN.items() if m != "M4"}

# 鱼眼系迭代停止条件:容差压到 1e-12(比 solve.py 的 1e-6 更紧)。
# 理由:这里的 cv2 调用要的是**精确逆**(PnP 前的去畸变、去畸变直线度),
# 逆算误差会直接冒成「残差结构」,把 A0/A1 该判的模型误差信号淹没掉。
_FISHEYE_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 1e-12)


def _expected_d_len(model: str) -> int:
    """模型名 → OpenCV dist 长度;未知模型中文 ValueError(布局必须在入口点名)。"""
    try:
        return _EXPECTED_D_LEN[model]
    except KeyError:
        raise ValueError(
            f"畸变模型名不合法:{model!r}(体检只支持 "
            f"{'/'.join(sorted(_EXPECTED_D_LEN))})。请传入来自配置文件或内参"
            f"产物的模型名,不要按位置猜布局。") from None


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
            f"K=3×3、D=5 长 M2 / 8 长 M8 / 4 长 M4 布局)。"
        )
    return a


def _validate_inputs(uv, objp, K, D, image_size, model: str | None = None,
                     ) -> tuple[np.ndarray, np.ndarray,
                                np.ndarray, np.ndarray,
                                tuple[int, int], str]:
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
    # model 缺省(None,旧调用形态)时按 D 长度反推,但**只认针孔族**的 5→M2、
    # 8→M8;4 长(鱼眼族)不许猜,必须显式点名 model="M4"。
    if model is None:
        model = _D_LEN_TO_MODEL.get(D_arr.shape[0])
        if model is None:
            raise ValueError(
                f"体检输入 D 形状不合:期望 (5,)(M2 五参)、(8,)(M8 rational 8) "
                f"或 (4,)(M4 鱼眼),实际是 {D_arr.shape}。请检查调用方传入的"
                f"畸变布局,或显式传入 model 参数点名布局。")
    expected = _expected_d_len(model)      # 模型名非法在这里就炸(入口点名)
    if D_arr.shape != (expected,):
        # 布局派位按模型:M2 五参 / M8 rational 8 / M4 鱼眼 4。长度与模型对不上
        # = 调用方把另一族的 D 递了进来(如把 M4 的 4 长喂给针孔路径),必须拦。
        raise ValueError(
            f"体检输入 D 形状不合:模型 {model} 期望 ({expected},)("
            f"{'鱼眼 θ 多项式 [k1,k2,k3,k4]' if model == 'M4' else '针孔 dist 布局'})"
            f",实际是 {D_arr.shape}。请检查调用方传入的畸变布局与产线 "
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
    return uv_arr, objp_arr, K_arr, D_arr, (int(size[0]), int(size[1])), model


def _per_point_residuals_fisheye(uv: np.ndarray, objp: np.ndarray,
                                 K: np.ndarray, D: np.ndarray) -> np.ndarray:
    """鱼眼版逐点残差:cv2.fisheye.solvePnP + cv2.fisheye.projectPoints。

    为什么不能用针孔那条路solvePnP(K, D):4 长 D 在针孔 cv2 眼里是
    [k1,k2,p1,p2],代入后一个数也不报错,只是整个残差场都错模型。这里
    显式走 fisheye 命名空间;内部去畸变用 _FISHEYE_CRITERIA 收紧到 1e-12,
    避免逆算误差冒成「残差结构」混淆 A0/A1 的模型误差信号。
    """
    objp64 = np.ascontiguousarray(objp, dtype=np.float64).reshape(-1, 1, 3)
    dist = D.ravel().reshape(-1, 1)
    residuals: list[np.ndarray] = []
    for i, pts in enumerate(uv):
        pts64 = np.ascontiguousarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        ok, rvec, tvec = cv2.fisheye.solvePnP(
            objp64, pts64, K, dist, None, None, False, cv2.SOLVEPNP_ITERATIVE,
            _FISHEYE_CRITERIA)
        if not ok:
            raise ValueError(
                f"体检第 {i} 帧鱼眼 solvePnP 求位姿失败(OpenCV 返回 ok=False):"
                f"产线 K/D 对该帧角点连位姿都解不稳,内参可靠性无从谈起。"
                f"请检查该帧角点与标定结果。")
        proj, _jac = cv2.fisheye.projectPoints(objp64, rvec, tvec, K, dist)
        residuals.append(pts - np.asarray(proj, dtype=np.float64).reshape(-1, 2))
    return np.stack(residuals)


def _per_point_residuals(uv: np.ndarray, objp: np.ndarray, K: np.ndarray,
                         D: np.ndarray, model: str = "M2") -> np.ndarray:
    """逐点残差(公共前置):每帧 PnP 估位姿 → 投影回代 → uv − 投影。

    协议(与已验证体检脚本一致):PnP 用迭代法,cv2 绑定边界降精度装包/鱼眼
    侧全 float64,回代 float64;返回 (F,P,2) float64。某帧 PnP 失败属数值
    路径异常(数据已过装载与求解防线,此处失败说明 K/D 病态到 PnP 都不稳),
    中文 ValueError 上抛由编排层单路收口。模型分支:M2/M8 走针孔套件,
    M4 走鱼眼套件(_per_point_residuals_fisheye)。
    """
    if model == "M4":
        return _per_point_residuals_fisheye(uv, objp, K, D)
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


def _refit_fx_fisheye(uv_sub: np.ndarray, objp: np.ndarray, K: np.ndarray,
                      D: np.ndarray, image_size: tuple[int, int],
                      freeze_distortion: bool = False) -> tuple[float, float]:
    """鱼眼版同协议重标(返回重标后的 fx 与 cv2 整体 RMS)。

    flags = RECOMPUTE_EXTRINSIC | FIX_SKEW | USE_INTRINSIC_GUESS(初值吃产线
    K/D),冻结时再加 FIX_K1..K4(与 M8 的「冻结全部畸变只放 fx」同款语义);
    形状 (N,1,x)、dtype float64——鱼眼系的规矩,与针孔族的 float32 不同。
    """
    n = uv_sub.shape[0]
    objs = [np.ascontiguousarray(objp, dtype=np.float64).reshape(-1, 1, 3)
            for _ in range(n)]
    imgs = [np.ascontiguousarray(p, dtype=np.float64).reshape(-1, 1, 2)
            for p in uv_sub]
    flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
             | cv2.fisheye.CALIB_USE_INTRINSIC_GUESS)
    if freeze_distortion:
        flags |= (cv2.fisheye.CALIB_FIX_K1 | cv2.fisheye.CALIB_FIX_K2
                  | cv2.fisheye.CALIB_FIX_K3 | cv2.fisheye.CALIB_FIX_K4)
    rms, K_fit, _D_fit, _rv, _tv = cv2.fisheye.calibrate(
        objs, imgs, tuple(image_size),
        np.array(K, dtype=np.float64, copy=True),
        np.array(D, dtype=np.float64).ravel().reshape(-1, 1),
        flags=flags, criteria=_FISHEYE_CRITERIA)
    return float(np.asarray(K_fit, dtype=np.float64)[0, 0]), float(rms)


def _refit_fx(uv_sub: np.ndarray, objp: np.ndarray, K: np.ndarray, D: np.ndarray,
              image_size: tuple[int, int], rational: bool,
              freeze_distortion: bool = False,
              model: str = "M2") -> tuple[float, float]:
    """同协议重标一份子集/全量数据,返回 (fx, cv2 整体 RMS)。

    flags=CALIB_USE_INTRINSIC_GUESS(初值吃产线 K/D);rational=True 或
    产线 D 为 8 长(M8)时加 CALIB_RATIONAL_MODEL——不给该 flag 时 OpenCV
    对 8 长布局会静默截断,故 M8 产线解下探针恒为同阶 rational8;M2 五长
    产线值按 rational 开关决定升阶与否(A4 探针,D0 仍传 5 长产线值,
    OpenCV 内部扩到 8 系)。

    freeze_distortion=True(A3 切片在 M8 产线解下的冻结协议,2026-10-24
    追加轮):加 FIX_K1..K6 | FIX_TANGENT_DIST——8 个畸变系数全部冻结,
    只放 fx,fy + 每帧位姿。为什么:M8 下切片「全自由」重拟合的 fx 弱可
    辨识——num/den 近简并谷里 fx 可被畸变系数的谷内漂移抵偿,切片间 fx
    极差被简并放大(cam2 外圈切片曾读出 fx 2021 的假信号);冻结后 fx 是
    唯一自由的场依赖信号,极差回归「fx 吸收畸变」的本义(真实数据实测:
    全自由 cam2 0.94% → 冻结 0.199%,cam1 0.157%,锚点 fx≈2037 附近全绿;
    M2 协议零改动)。拟合失败以 cv2.error 上抛,由调用方决定记 note 或计数。

    M4(鱼眼):转调 _refit_fx_fisheye(flags/形状/dtype 都是另一套规矩);
    语义不变——冻结与否仍是「只放 fx 还是全放开」。
    """
    if model == "M4":
        return _refit_fx_fisheye(uv_sub, objp, K, D, image_size, freeze_distortion)
    n = uv_sub.shape[0]
    use_rational = bool(rational) or np.asarray(D).size == 8
    flags = cv2.CALIB_USE_INTRINSIC_GUESS | (cv2.CALIB_RATIONAL_MODEL if use_rational else 0)
    if freeze_distortion:
        # 只收产线 D 为 8 长(M8)的冻结:5 长 M2 的冻结语义走「A3 全自由
        # 不变」契约(调用方保证),这里不按长度分派、由调用方显式传入。
        flags |= (cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
                  | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6
                  | cv2.CALIB_FIX_TANGENT_DIST)
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


def _check_a2(uv: np.ndarray, objp: np.ndarray, K: np.ndarray, D: np.ndarray,
              model: str = "M2") -> CheckResult:
    """A2 去畸变直线度(仅报告不判红):去畸变后行/列 SVD 共线最大偏差中位数。

    M4 走 cv2.fisheye.undistortPoints(与针孔同名、不同义):同为「把观测点还原
    到理想针孔像素平面」,但去畸变的逆是另一条公式。
    """
    dist = D.ravel().reshape(-1, 1)
    ys = np.unique(objp[:, 1])
    xs = np.unique(objp[:, 0])
    groups = [np.where(np.isclose(objp[:, 1], y))[0] for y in ys]
    groups += [np.where(np.isclose(objp[:, 0], x))[0] for x in xs]
    worst_per_frame: list[float] = []
    for pts in uv:
        pts_c = np.ascontiguousarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        if model == "M4":
            und = cv2.fisheye.undistortPoints(
                pts_c, K, dist, None, K, criteria=_FISHEYE_CRITERIA).reshape(-1, 2)
        else:
            und = cv2.undistortPoints(pts_c, K, dist, None, K).reshape(-1, 2)
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
              image_size: tuple[int, int], center: np.ndarray, cfg,
              model: str = "M2") -> CheckResult:
    """A3 切片 fx 稳定性:6 切片各重标 fx,极差 %。红:> a3_max_fx_range_pct。

    单切片帧数 < a3_min_frames 跳过;有效切片 < 4 → 「数据不足」不计红
    (红灯必须来自真实判定量)。单切片拟合失败(cv2.error)记 note 跳过。
    产线 D 为 8 长(M8)时切片重标自动带 rational flag(与产线同阶)并
    冻结全部畸变系数(只放 fx,fy+位姿,见 _refit_fx:全自由重拟合在
    rational 近简并谷下 fx 沿谷漂移,极差被简并放大)——切片间 fx 极差
    度量的是「fx 吸收畸变随板位漂移」,与模型阶次无关。M2(5 长)协议
    零改动。
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
    # 冻结协议一次派定:M8(D 8 长)冻结 / M4(鱼眼,KB4 同样存在 fx-k 的补偿阱)
    # 冻结 K1..K4 / M2(5 长)全自由(契约零改动)。
    freeze_d = model in ("M8", "M4")
    for name in _A3_SPLIT_NAMES:
        m = masks[name]
        n = int(m.sum())
        if n < cfg.a3_min_frames:
            notes.append(f"{name}切片仅 {n} 帧(< a3_min_frames={cfg.a3_min_frames}),跳过")
            continue
        try:
            fx, _rms = _refit_fx(uv[m], objp, K, D, image_size, rational=False,
                                 freeze_distortion=freeze_d, model=model)
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
              image_size: tuple[int, int], cfg, model: str = "M2") -> CheckResult:
    """A4 模型阶次探针:全量重标,fx 跳变 %。红:> a4_max_fx_jump_pct。

    M2 产线解:升阶探针(rational8 vs 产线五参);M8 产线解(D 8 长):
    同阶独立重标一致性探针(cv2 rational8 vs 产线 scipy M8——fx 跳变仍
    检验 fx 吸收畸变型欠拟合,不再有更高阶可升);M4 产线解(D 4 长,鱼眼):
    cv2 没有更高阶的鱼眼模型可供升阶,探针语义退化为「同阶 cv2 鱼眼独立
    重标一致性」——与 M8 同为「看 cv2 能否从产线解出发独立再现同一个 fx」
    ( fx 仍有吸收畸变的自由度,跳变仍照舊检验该类欠拟合)。
    探针拟合失败(cv2 异常/非有限解)→ 记 note 不计红(探针故障不许撑红灯)。
    """
    fx_prod = float(K[0, 0])
    if model == "M4":
        try:
            fx_probe, _rms = _refit_fx(uv, objp, K, D, image_size, rational=False,
                                       freeze_distortion=False, model="M4")
        except cv2.error as e:
            return CheckResult("A4", "模型阶次探针", None, float(cfg.a4_max_fx_jump_pct),
                               False, note=f"同阶鱼眼独立重标探针拟合失败(OpenCV 报错),不计红:{e}")
        if not np.isfinite(fx_probe) or fx_probe <= 0.0:
            return CheckResult("A4", "模型阶次探针", None, float(cfg.a4_max_fx_jump_pct),
                               False, note=f"同阶鱼眼独立重标探针解非有限(fx={fx_probe}),不计红")
        jump = float(abs(fx_prod - fx_probe) / fx_probe * 100.0)
        return CheckResult("A4", "模型阶次探针", jump, float(cfg.a4_max_fx_jump_pct),
                           bool(jump > cfg.a4_max_fx_jump_pct),
                           note="产线 M4(鱼眼 KB4):探针为同阶 cv2 鱼眼独立重标一致性"
                                "(鱼眼侧无更高阶可升)",
                           detail={"fx_production": fx_prod, "fx_re_refit": fx_probe,
                                   "probe": "same_order_fisheye_kb4"})
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


def run_battery(uv, objp, K, D, image_size, cfg, model: str | None = None,
                ) -> DiagnosticsReport:
    """体检唯一入口:train 幸存角点 + 产线 K/D → 五检查 + verdict。

    参数:
        uv:          (F,P,2) float64,某相机 train 幸存视图角点(像素);
        objp:        (P,3) float64,与角点同序的板系物点;
        K / D:       产线定稿解出的 3×3 内参矩阵与畸变向量(M2 5 长 /
                     M8 8 长 / M4 4 长,长度与模型必须一致);
        image_size:  (宽, 高);
        cfg:         IntrinsicDiagnosticsConfig(阈值与判定规则全部来自配置);
        model:       产线畸变模型名("M2" | "M8" | "M4"),决定每路调用走针孔
                     还是鱼眼的 cv2 套件(鱼眼系的形状/dtype/flags 是另一套);
                     传 None(旧调用形态)时按 D 长度反推,语义仍一一对应。
    返回:
        DiagnosticsReport(检查数值/阈值/red/note/verdict/red_count);
        cfg.enabled=False 时返回「未体检」报告,不做任何计算(跳过路径)。
    """
    if not cfg.enabled:
        disabled_note = ("体检开关关闭(intrinsic.diagnostics.enabled: false),"
                         "本轮未体检,内参可靠性未判定")
        return _finalize((), (disabled_note,), cfg.min_red_count, False)
    uv_arr, objp_arr, K_arr, D_arr, size, model = _validate_inputs(
        uv, objp, K, D, image_size, model)
    width, height = size
    center = np.array([width / 2.0, height / 2.0], dtype=np.float64)
    half_diag = float(np.hypot(center[0], center[1]))

    res = _per_point_residuals(uv_arr, objp_arr, K_arr, D_arr, model)
    diff = uv_arr - center
    r_norm = np.linalg.norm(diff, axis=-1) / half_diag     # (F,P)

    checks = (
        _check_a0(res, r_norm, cfg),
        _check_a1(res, r_norm, uv_arr, center, cfg),
        _check_a2(uv_arr, objp_arr, K_arr, D_arr, model),
        _check_a3(uv_arr, objp_arr, K_arr, D_arr, size, center, cfg, model),
        _check_a4(uv_arr, objp_arr, K_arr, D_arr, size, cfg, model),
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
