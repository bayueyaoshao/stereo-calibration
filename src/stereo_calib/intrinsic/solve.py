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
from stereo_calib.intrinsic.metrics import compute_metrics
from stereo_calib.io.config import RefineConfig

# 求解器自身的视图数底线:两三个视角定不了内参(装载层 min_views 默认 10,
# 是编排层的配置下限;这里是硬算底线,低于它直接中文报错不硬算)
_MIN_SOLVE_VIEWS = 2

# 实用收敛判定的尾部窗口(约 10 次残差评估,滤掉被信赖域拒绝的尝试步回弹)
_TAIL_WINDOW = 10
# 尾部改进的"走平"判定线:相对改进小于它就认为代价不再下降
_PLATEAU_TOL = 1e-6

# ---- M4(cv2.fisheye.calibrate)专用常量 --------------------------------------
# flags:CALIB_RECOMPUTE_EXTRINSIC(=2)每轮迭代重算位姿(不带它有假收敛);
#        CALIB_FIX_SKEW(=8)把 skew 锁 0——M4 参数表里本就没有 skew,数学层
#        投影也恒为 0,放开它等于让 OpenCV 白调一个不影响结果的自由度
#        (与参考工程 kb4 适配器的 flags 逐位一致)。
_FISHEYE_FLAGS = int(cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW)
# 停止条件( errno 实测默认:迭代上限 100、收敛容差 1e-6)
_FISHEYE_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
# 多起步触发门限:单起步(float64 回代 RMS)高于它就说明可能卡进了坏盆地,
# 才升级到多起步——好数据一次就过、零额外算力(与参考工程 opencv 后端的
# 同类门控同思路)。
_FISHEYE_MULTISTART_TRIGGER = 0.5


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
    if model == "M4":
        # 鱼眼解包:D 恰 4 长 [k1,k2,k3,k4](θ 多项式系数),语义与针孔族
        # 完全不同,长度不对必须当场拦——喂给针孔路径会被解释成
        # [k1,k2,p1,p2] 且不报错(静默错是这里唯一的可怕结局)。
        assert D.shape == (4,), (
            f"OpenCV 解包的畸变向量长度是 {D.shape},期望 (4,)(M4 鱼眼布局 "
            f"{names})。出现此错说明走的不是 cv2.fisheye.calibrate,请核查。")
    elif model == "M2":
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


# ---------------------------------------------------------------------------
# M4(鱼眼 KB4)的 OpenCV 初值:多起步自运用 PINHOLE 自举(2026-10-26 增补)
# ---------------------------------------------------------------------------

def _fisheye_tan_seed(theta_max: float) -> np.ndarray:
    """把 θd(θ) 拟成 tanθ 的最小二乘 k 系数(让 M4 先站在「近似针孔」处)。

    白话:同一台相机若其实是近针孔,k 全 0 起步离真相很远( pinhole 模型的
    像素半径 ∝ tanθ,鱼眼模型的 ∝ θd)。这里我们只要解线性方程
    θ + k1θ³ + k2θ⁵ + k3θ⁷ + k4θ⁹ ≈ tanθ 就能拿到一组「站在针孔附近」的
    种子,纯解析、无任何外部标定结果输入。
    """
    th = np.linspace(1e-3, float(theta_max), 200)
    basis = np.stack([th**3, th**5, th**7, th**9], axis=1)
    target = np.tan(th) - th
    coef, *_ = np.linalg.lstsq(basis, target, rcond=None)
    return np.asarray(coef, dtype=np.float64)


def _fisheye_k_seeds() -> list[np.ndarray]:
    """M4 的畸变初值候选(第 0 项零初值 = 原单起步行为)。

    三档:零 → tan 拟合族(覆盖不同视场上限,θmax 越大越广角)→ 鱼眼式大
    k1 扫(正负两向)。全部由本文件的解析式直接产生,既不读外部标定结果
    也不塞魔数 patch(工程自包含纪律)。
    """
    seeds: list[np.ndarray] = [np.zeros(4, dtype=np.float64)]
    for theta_max in (0.35, 0.6, 0.9, 1.2):
        seeds.append(_fisheye_tan_seed(theta_max))
    for m in (-0.5, -0.3, 0.3, 0.6, 1.0):
        seeds.append(np.array([m, 0.0, 0.0, 0.0], dtype=np.float64))
    return seeds


def _fisheye_focal_guesses(image_size: tuple[int, int],
                           pinhole_fx: float | None) -> list[np.ndarray]:
    """M4 的 K 初值候选清单:等距约定 + 若干典型视场 + (可选)自家针孔 f。

    pinhole_fx 来自**本工程自己**在同一批观测上的针孔解(solve_opencv 的
    pinhole 路径),不是外部产物——正因为是「自家解」的自举,才既能拿到靠谱
    的焦距量级,又不违反自包含纪律(外部 K 作初值只是藏住脆弱性,换相机仍崩)。
    """
    width, height = int(image_size[0]), int(image_size[1])
    guesses: list[np.ndarray] = []
    seen: list[float] = []

    def _push(fx: float) -> None:
        if fx <= 0.0 or any(abs(fx - got) < 1e-6 * max(got, 1.0) for got in seen):
            return
        seen.append(float(fx))
        g = np.eye(3, dtype=np.float64)
        g[0, 0] = g[1, 1] = float(fx)
        g[0, 2] = width / 2.0
        g[1, 2] = height / 2.0
        guesses.append(g)

    # 等距鱼眼约定:180° 视场正好铺满最长边
    _push(max(width, height) / np.pi)
    # 若干典型水平视场(60°/90°/120°/150°):f = (w/2)/tan(hfov/2)
    for hfov_deg in (150.0, 120.0, 90.0, 60.0):
        _push((width / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0))
    if pinhole_fx is not None:     # 自家针孔解自举(本工程产物,非外部结果)
        _push(float(pinhole_fx))
    return guesses


def _fisheye_bootstrap_focal_theta(uvs_arr: np.ndarray, objp_arr: np.ndarray,
                                   image_size: tuple[int, int],
                                   image_size_note: str) -> tuple[float | None, float | None]:
    """用**自家**针孔解先估一次焦距与视场量级,给鱼眼多起步当种子。

    返回 (自家针孔 fx, 画面最大入射角估计 θmax);任一步失败返回 (None, None)
    ——自举只是「有则用」,绝不因它失败而中断标定。
    """
    try:
        pin = solve_opencv(uvs_arr, objp_arr, image_size, "M8")
    except (ValueError, ArithmeticError, RuntimeError, AssertionError, cv2.error):
        return None, None
    fx = float(pin.params[0])
    width, height = int(image_size[0]), int(image_size[1])
    corners = np.array([[0.0, 0.0], [width, 0.0], [0.0, height], [width, height]],
                       dtype=np.float64)
    try:
        rays = models.unproject(corners, pin.params)
    except (ValueError, ArithmeticError, AssertionError):
        return fx, None
    theta = np.arctan2(np.hypot(rays[:, 0], rays[:, 1]), rays[:, 2])
    return fx, float(theta.max())


def _run_fisheye_once(objs, imgs, size: tuple[int, int], K_guess: np.ndarray,
                      D_seed: np.ndarray):
    """单次 cv2.fisheye.calibrate(K/D 初值 copy 后传入,防 OpenCV 就地改写)。"""
    return cv2.fisheye.calibrate(
        objs, imgs, tuple(size), K_guess.copy(), D_seed.copy().reshape(4, 1),
        flags=_FISHEYE_FLAGS, criteria=_FISHEYE_CRITERIA)


def _solve_opencv_m4(uvs_arr: np.ndarray, objp_arr: np.ndarray,
                     size: tuple[int, int], n_views: int) -> SolveResult:
    """M4 初值路径:cv2.fisheye.calibrate(float64 + (N,1,x) 形状) + 多起步择优。

    为什么必须多起步(用户 2026-10-26 裁决 + 本工程的既有教训):cv2 鱼眼
    求解器对「非鱼眼数据」或「强畸变数据」极易落进 fx 被撑大/缩小的退化盆
    地,单起步拼运气。**多起步**是让本工程自行落到好盆地的正道;引入外部
    K/dist 作初值只是藏住脆弱性。

    择优用**自家 float64 回代 RMS**(compute_metrics,Pitfall 12:OpenCV 的
    rms 是 float32 口径,只留 solver_meta 对账),不是 OpenCV 返回值。

    初值候选 = K 猜值清单 × 畸变种子清单。**每次拟合失败都不是全局失败**:
    强畸变数据下 OpenCV 鱼眼求解器常在某些种子上报内部断言( InitExtrinsics
    类),正是要靠别的种子救回来;只有**全部候选都失败**才允许抛错。数据本
    就好时第一次起步即过闸门,只拟合一次(与 pinhole 路径一样便宜)。
    """
    n_points = objp_arr.shape[0]
    # fisheye 系函数既不收 float32(与 calibrateCamera 不同)也不收 (N,x):
    # 形状固定 (N,1,3)/(N,1,2)、dtype 固定 float64(参考实测行为)。
    objs = [objp_arr.astype(np.float64).reshape(n_points, 1, 3) for _ in range(n_views)]
    imgs = [uv.astype(np.float64).reshape(n_points, 1, 2) for uv in uvs_arr]

    pinhole_fx, theta_max = _fisheye_bootstrap_focal_theta(uvs_arr, objp_arr, size, size)
    k_guesses = _fisheye_focal_guesses(size, pinhole_fx)
    seeds = _fisheye_k_seeds()
    if theta_max is not None and theta_max > 0.0:      # 自家解给出的视场 → 多一份量身种子
        seeds = seeds + [_fisheye_tan_seed(theta_max)]

    names = models.intrinsic_names("M4")

    def _pack(rms, K_out, D_out, rvecs, tvecs):
        params, rvecs_arr, tvecs_arr = _unpack_opencv_solution(
            "M4", names, K_out, D_out, rvecs, tvecs, n_views)
        scored = compute_metrics(objp_arr, uvs_arr, params, rvecs_arr, tvecs_arr)
        return (scored.rms_overall, params, rvecs_arr, tvecs_arr, float(rms))

    # 第 1 步:与原 pinhole 路径同价的那一次起步(零畸变 + 等距约定焦距)。
    best = None
    n_starts = 0
    last_error: Exception | None = None
    try:
        best = _pack(*_run_fisheye_once(objs, imgs, size, k_guesses[0], seeds[0]))
        n_starts = 1
    except (cv2.error, ValueError, ArithmeticError, AssertionError) as e:
        last_error = e   # 单起步炸 ≠ 全局失败:正是多起步要救的那一类数据

    # 第 2 步:首次起步没出结果,或出了但疑似坏盆地(回代 RMS 超门限)→
    # 铺 K×D 网格逐一起步,单次失败照许(记录最后一个错备用);同一把
    # float64 尺比对择优。
    multistart = best is None or best[0] > _FISHEYE_MULTISTART_TRIGGER
    if multistart:
        first_tried = n_starts >= 1                 # 首步成功过就不再重跑同一对
        for k_idx, k_guess in enumerate(k_guesses):
            for s_idx, seed in enumerate(seeds):
                if first_tried and k_idx == 0 and s_idx == 0:
                    continue
                try:
                    cand = _pack(*_run_fisheye_once(objs, imgs, size, k_guess, seed))
                except (cv2.error, ValueError, ArithmeticError, AssertionError) as e:
                    last_error = e                 # 个别种子病态是常态,继续
                    continue
                n_starts += 1
                if best is None or cand[0] < best[0]:
                    best = cand

    if best is None:
        raise ValueError(
            f"OpenCV 鱼眼标定求解失败:{n_views} 个视图,image_size={size},"
            f"{n_starts} 次起步全部失败,最后一次 OpenCV 报错——{last_error}。"
            f"鱼眼模型对数据分布很挑(需要跨视场的大角度覆盖),请检查是否为"
            f"鱼眼数据、视图朝向是否足够分散。"
        ) from last_error

    rms_scored, params, rvecs_arr, tvecs_arr, opencv_rms = best
    if not np.isfinite(params).all():
        raise ValueError(
            f"OpenCV 鱼眼标定未产出有限数值:{n_views} 个视图,image_size={size}。"
            f"全部 {n_starts} 个起步都落进了病态解,多半是数据不是鱼眼/视场覆盖不足。")
    return SolveResult(
        params=params, rvecs=rvecs_arr, tvecs=tvecs_arr, solver="opencv",
        solver_meta={"path": "fisheye.calibrate", "flags_used": int(_FISHEYE_FLAGS),
                     "distortion_model": "M4", "opencv_rms": opencv_rms,
                     "n_starts": n_starts, "multistart": multistart,
                     "rms_float64": float(rms_scored)})


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
    uvs_arr, objp_arr, size = _check_common_inputs(uvs, objp, image_size)
    n_views = uvs_arr.shape[0]
    n_points = uvs_arr.shape[1]
    if model == "M4":      # 鱼眼走另一条初值路线(fisheye.calibrate + 多起步)
        return _solve_opencv_m4(uvs_arr, objp_arr, size, n_views)
    flags = cv2.CALIB_RATIONAL_MODEL if model == "M8" else 0
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

    内参块长度按布局分派(M4 八参 / M2 九参 / M8 十二参);θ 总长对 6 取模的
    余数即布局指纹(M2 ≡3、M4 ≡2、M8 ≡0,三者互不相同),unpack_theta 按它
    无损解箱。不做"内参/外参分块交替优化"——那等于自研优化器,收敛性质更差
    且实测收益为零(Don't Hand-Roll)。V 视图 → 8/9/12 + 6V 维 float64。
    """
    arr = np.asarray(params, dtype=np.float64).ravel()
    if arr.shape[0] not in (8, 9, 12):
        raise AssertionError(
            f"参数向量形状不合:期望 (8,)(布局 {models.INTRINSIC_NAMES_M4})、"
            f"(9,)(布局 {models.INTRINSIC_NAMES_M2})或 "
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

    内参块长度按 θ 对 6 取模的余数无损分派(8+6V ≡ 2、9+6V ≡ 3、
    12+6V ≡ 0 (mod 6)——三种布局余数互不相同);
    长度对不上任何布局/视图数时直接中文报错——装箱与解箱必须同一布局,
    错位在这里点名,不带病进优化器。
    """
    th = np.asarray(theta, dtype=np.float64).ravel()
    remainder = th.shape[0] % 6
    if remainder == 2:
        n_params = 8    # M4 布局:8 + 6×视图数
    elif remainder == 3:
        n_params = 9    # M2 布局:9 + 6×视图数
    elif remainder == 0:
        n_params = 12   # M8 布局:12 + 6×视图数
    else:
        raise AssertionError(
            f"θ 向量长度不合:长度 {th.shape[0]} 对不上任何布局"
            f"(M4:内参 8+6×视图数;M2:内参 9+6×视图数;M8:内参 12+6×视图数)。"
            f"请检查 θ 是否由 pack_theta 按同一布局装出。"
        )
    if th.shape[0] < n_params + 6:
        raise AssertionError(
            f"θ 向量长度不合:长度 {th.shape[0]} 对不上任何布局"
            f"(M4:内参 8+6×视图数;M2:内参 9+6×视图数;M8:内参 12+6×视图数)。"
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

    白话:θ = [内参(按 model 布局 8/9/12)| 每视图 6 个位姿参数] 装成
    **一个**向量,交给 least_squares 的 trf(信赖域反射法)+ lsmr(解稀疏
    线性子问题)+ 解析稀疏雅可比联合优化;不做内参/外参分块交替优化
    (Don't Hand-Roll)。三个模型走同一段代码,差异只在雅可比列数。

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
        model:      "M2" | "M8" | "M4"(布局见 models.intrinsic_names,默认 M2)。
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
