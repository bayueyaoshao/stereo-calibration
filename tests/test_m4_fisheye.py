"""M4(鱼眼等距 KB4)全链路测试:数学层 / 初值求解 / 体检 / 装载 / 双目外参。

体例照 test_intrinsic_models.py 与 test_stereo_solve.py:零渲染、零读图(装载
契约那两条例外),真值由 models.project 合成,OpenCV 只当**考官**不作枪手
——M4 的投影必须与 cv2.fisheye.projectPoints 对齐;而求解必须靠本工程自己的
多起步落到好盆地(不依赖任何外部标定结果,自包含纪律)。

覆盖口径:
- 数学层:与 cv2.fisheye.projectPoints 对齐 < 1e-9 px、两份雅可比 < 1e-6、
  往返 < 1e-9、鱼眼放宽 Z>0 而针孔那道防线不许动、按参数长度不串模型;
- 初值层:合成鱼眼数据能恢复(fx 误差 < 1%、无噪声回代 RMS < 1e-5)、含噪
  数据落到噪声底、多起步网格跑得通且如实记录起步次数;
- θ 装箱:8+6V ≡ 2 (mod 6),与 M2/M8 不串箱;
- 体检层:点名 model="M4" 时五道电池都出数;4 长 D 不许按长度反推;
- 装载层:distortion_model="M4" 的 intrinsics.json 可写可读、D 长度错位
  中文报错;
- 双目层:solve_view_pose 用自家精确逆、joint 对 M4 fail-closed、端到端
  恢复 R/t(转角 < 0.1°、平移 < 1mm)。
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

from stereo_calib.intrinsic import models
from stereo_calib.intrinsic.artifacts import write_intrinsics_json
from stereo_calib.intrinsic.diagnostics import run_battery
from stereo_calib.intrinsic.metrics import compute_metrics
from stereo_calib.intrinsic.solve import pack_theta, solve_opencv, unpack_theta
from stereo_calib.io.config import (
    BoardSpec,
    IntrinsicDiagnosticsConfig,
    StereoCalibConfig,
    StereoJointConfig,
    StereoRefineConfig,
    load_config,
)
from stereo_calib.preprocess.artifacts import build_object_points
from stereo_calib.stereo.joint import refine_joint
from stereo_calib.stereo.observation import (
    IntrinsicsView,
    Pairs,
    load_intrinsics,
    solve_view_pose,
)
from stereo_calib.stereo.solve import solve_stereo
from test_intrinsic_config import MINIMAL_YAML, _mutate, _write_config

# M4 真值:~180° 对角视场的鱼眼(fx = 画面半对角 /(π/2)),畸变取常见量级
_PARAMS_M4_TRUE = np.array([467.0, 466.0, 640.0, 360.0,
                            -0.02, 0.008, -0.002, 0.0004])
_PARAMS_M2_TRUE = np.array([1850.0, 1840.0, 1280.0, 720.0,
                            -0.40, 0.23, 1.1e-4, -3.8e-5, -0.08])
_IMAGE_SIZE = (1280, 720)
_BOARD = BoardSpec(cols=11, rows=8, square_size_mm=25.0)


def _max_rel_diff(ana: np.ndarray, num: np.ndarray) -> float:
    """解析式 vs 数值差分的最大相对差(按数值基准的最大绝对值归一)。"""
    scale = float(np.abs(num).max())
    if scale < 1e-30:
        return float(np.abs(ana - num).max())
    return float(np.abs(ana - num).max() / scale)


def _random_front_points(rng: np.random.Generator, n: int = 200) -> np.ndarray:
    """一批「镜头前方」的相机系点(Z>0,覆盖常规棋盘的距离与横向分布)。"""
    return np.column_stack([
        rng.uniform(-160.0, 160.0, n),
        rng.uniform(-120.0, 120.0, n),
        rng.uniform(300.0, 900.0, n),
    ])


def _synth_fisheye_views(params: np.ndarray, n_views: int = 24, seed: int = 11,
                         noise_px: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """用鱼眼真值合成一批标定视图,返回 (objp, uvs)。"""
    rng = np.random.default_rng(seed)
    objp = build_object_points(_BOARD)
    uvs = []
    for _ in range(n_views):
        rvec = rng.uniform(-0.6, 0.6, 3)
        tvec = np.array([rng.uniform(-100.0, 100.0),
                         rng.uniform(-70.0, 70.0),
                         rng.uniform(300.0, 600.0)])
        pts, _ = models.transform_to_camera(objp, rvec, tvec)
        uv = models.project(pts, params)
        uvs.append(uv + rng.normal(0.0, noise_px, uv.shape))
    return objp, np.stack(uvs)


# ---------- 1. 数学层:与 cv2.fisheye 对齐 / 雅可比 / 往返 ----------

def test_m4_project_matches_cv2_fisheye_project_points():
    """M4 投影必须逐像素对上 cv2.fisheye.projectPoints(< 1e-9 px)。

    这是移植正确性的唯一外源证据:cv2 是考官不是枪手——本工程产线调用绝不
    依赖 cv2 的投影结果,这条只在测试里用。
    """
    rng = np.random.default_rng(7)
    pts = _random_front_points(rng)
    ours = models.project(pts, _PARAMS_M4_TRUE)
    K = models.intrinsics_matrix(_PARAMS_M4_TRUE)
    ref, _ = cv2.fisheye.projectPoints(
        pts.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K,
        _PARAMS_M4_TRUE[4:].reshape(4, 1))
    err = float(np.abs(ours - np.asarray(ref, dtype=np.float64).reshape(-1, 2)).max())
    assert err < 1e-9, f"M4 投影与 cv2.fisheye.projectPoints 偏差 {err:.3e}px(须 < 1e-9)"


def test_m4_project_jacobian_matches_numeric():
    """project_jacobian 对 8 个内参列逐列对上数值差分(最大相对差 < 1e-6)。"""
    rng = np.random.default_rng(13)
    pts = _random_front_points(rng, n=64)
    num = approx_derivative(lambda p: models.project(pts, p).ravel(),
                            _PARAMS_M4_TRUE.copy(), method="3-point")
    ana = models.project_jacobian(pts, _PARAMS_M4_TRUE).reshape(-1, 8)
    rel = _max_rel_diff(ana, num)
    assert rel < 1e-6, f"M4 投影对内参的解析雅可比相对差 {rel:.3e}(须 < 1e-6)"


def test_m4_project_point_jacobian_matches_numeric():
    """project_point_jacobian 对相机点 3 列逐列对上数值差分(< 1e-6)。"""
    rng = np.random.default_rng(17)
    pts = _random_front_points(rng, n=40)
    num = approx_derivative(
        lambda x: models.project(x.reshape(-1, 3), _PARAMS_M4_TRUE).ravel(),
        pts.copy().ravel(), method="3-point")
    blocks = models.project_point_jacobian(pts, _PARAMS_M4_TRUE)
    n = len(pts)
    ana = np.zeros((2 * n, 3 * n), dtype=np.float64)
    for i in range(n):
        ana[2 * i:2 * i + 2, 3 * i:3 * i + 3] = blocks[i]
    rel = _max_rel_diff(ana, num)
    assert rel < 1e-6, f"M4 投影对相机点的解析雅可比相对差 {rel:.3e}(须 < 1e-6)"


def test_m4_project_unproject_roundtrip():
    """M4 往返闭合 < 1e-9 px;unproject 返回单位射线(不是 pinhole 的 z≡1 口径)。"""
    rng = np.random.default_rng(19)
    pts = _random_front_points(rng, n=48)
    uv = models.project(pts, _PARAMS_M4_TRUE)
    rays = models.unproject(uv, _PARAMS_M4_TRUE)
    assert np.allclose(np.linalg.norm(rays, axis=1), 1.0, atol=1e-15), "M4 射线必须单位化"
    back = models.project(rays, _PARAMS_M4_TRUE)
    err = float(np.abs(back - uv).max())
    assert err < 1e-9, f"M4 往返偏差 {err:.3e}px(须 < 1e-9)"


def test_m4_project_allows_nonpositive_z_but_m2_keeps_guard():
    """鱼眼放宽 Z>0(Z<=0 在 θ=arctan2 下仍可成像);针孔那道防线不许跟着松。"""
    pts = np.array([[0.0, 0.0, 500.0], [40.0, -20.0, -300.0]])
    uv = models.project(pts, _PARAMS_M4_TRUE)
    assert np.isfinite(uv).all(), "M4 对 Z<=0 的点必须给出有限像点"
    with pytest.raises(AssertionError, match="Z"):
        models.project(pts, _PARAMS_M2_TRUE)


def test_m4_layout_dispatch_by_params_length():
    """8/9/12 长互不串模型:model_of_params / is_fisheye / intrinsic_names 一致。"""
    assert models.model_of_params(_PARAMS_M4_TRUE) == "M4"
    assert models.model_of_params(_PARAMS_M2_TRUE) == "M2"
    assert models.is_fisheye(_PARAMS_M4_TRUE) is True
    assert models.is_fisheye(_PARAMS_M2_TRUE) is False
    assert list(models.intrinsic_names("M4")) == ["fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4"]
    with pytest.raises(ValueError, match="畸变模型名不合法"):
        models.intrinsic_names("M9")
    with pytest.raises(AssertionError, match="参数向量形状不合"):
        models.project(np.array([[0.0, 0.0, 1.0]]), np.zeros(7))


# ---------- 2. 初值求解:合成鱼眼数据能恢复 ----------

def test_m4_solve_opencv_recovers_synthetic_fisheye():
    """无噪声鱼眼数据:初值解即回到真值附近(fx 误差 < 1%、回代 RMS < 1e-5)。"""
    objp, uvs = _synth_fisheye_views(_PARAMS_M4_TRUE, n_views=26, seed=3)
    sol = solve_opencv(uvs, objp, _IMAGE_SIZE, "M4")
    assert sol.params.shape == (8,), "M4 初值解必须恰 8 参"
    assert sol.solver_meta["path"] == "fisheye.calibrate", "必须走 cv2.fisheye 路线"
    assert sol.solver_meta["distortion_model"] == "M4"
    fx_err_pct = abs(sol.params[0] - _PARAMS_M4_TRUE[0]) / _PARAMS_M4_TRUE[0] * 100.0
    rms = compute_metrics(objp, uvs, sol.params, sol.rvecs, sol.tvecs).rms_overall
    assert fx_err_pct < 1.0, f"fx 恢复误差 {fx_err_pct:.3f}%(须 < 1%)"
    assert rms < 1e-5, f"无噪声数据回代 RMS {rms:.3e}px(须 < 1e-5)"


def test_m4_solve_opencv_noisy_data_hits_noise_floor():
    """含噪(σ=0.3px)数据:回代 RMS 落在噪声底附近(0.3·√2≈0.424 的 1.5 倍内)。"""
    objp, uvs = _synth_fisheye_views(_PARAMS_M4_TRUE, n_views=30, seed=5, noise_px=0.3)
    sol = solve_opencv(uvs, objp, _IMAGE_SIZE, "M4")
    rms = compute_metrics(objp, uvs, sol.params, sol.rvecs, sol.tvecs).rms_overall
    assert 0.3 <= rms <= 0.3 * np.sqrt(2.0) * 1.5, f"含噪数据 RMS {rms:.4f}px 不在噪声底附近"


def test_m4_multistart_grid_runs_and_records_starts():
    """强制铺 K×D 网格时:允许单个种子失败、必须真的多次起步并给出有限解。"""
    objp, uvs = _synth_fisheye_views(_PARAMS_M4_TRUE, n_views=18, seed=9, noise_px=0.2)
    import stereo_calib.intrinsic.solve as S   # 只在本用例里改门限,跑完必须还原
    old = S._FISHEYE_MULTISTART_TRIGGER
    S._FISHEYE_MULTISTART_TRIGGER = 0.0
    try:
        sol = solve_opencv(uvs, objp, _IMAGE_SIZE, "M4")
    finally:
        S._FISHEYE_MULTISTART_TRIGGER = old
    assert S._FISHEYE_MULTISTART_TRIGGER == old, "本用例必须还原全局状态"
    assert sol.solver_meta["multistart"] is True, "门限置 0 时必须走多起步"
    assert sol.solver_meta["n_starts"] > 1, "多起步必须真的多次起步"
    assert np.isfinite(sol.params).all(), "多起步择优结果必须有限"


def test_m4_pack_unpack_theta_roundtrip():
    """θ 装箱/解箱对 M4(8 长)无损:8 + 6×视图数 ≡ 2 (mod 6)。"""
    objp, uvs = _synth_fisheye_views(_PARAMS_M4_TRUE, n_views=6, seed=23)
    sol = solve_opencv(uvs, objp, _IMAGE_SIZE, "M4")
    th = pack_theta(sol.params, sol.rvecs, sol.tvecs)
    assert th.shape[0] == 8 + 6 * len(sol.rvecs)
    p2, rv2, tv2 = unpack_theta(th)
    assert np.array_equal(p2, sol.params), "M4 内参箱必须无损"
    assert np.array_equal(rv2, sol.rvecs) and np.array_equal(tv2, sol.tvecs)


# ---------- 3. 体检:M4 五道电池可跑 ----------

def test_m4_diagnostics_battery_runs_with_model_named():
    """显式点名 model="M4" 时五道电池都出数(A2 为仅报告项,允许 None)。"""
    objp, uvs = _synth_fisheye_views(_PARAMS_M4_TRUE, n_views=40, seed=21, noise_px=0.12)
    sol = solve_opencv(uvs, objp, _IMAGE_SIZE, "M4")
    cfg = IntrinsicDiagnosticsConfig(
        enabled=True, a0_max_ratio=2.0, a1_max_bin_mean_px=0.02,
        a3_max_fx_range_pct=5.0, a3_min_frames=20, a4_max_fx_jump_pct=2.0,
        min_red_count=2)
    report = run_battery(uvs, objp, models.intrinsics_matrix(sol.params),
                         sol.params[4:], _IMAGE_SIZE, cfg, model="M4")
    assert len(report.checks) == 5, "体检电池必须有五道"
    a0 = next(c for c in report.checks if c.name == "A0")
    assert a0.value is not None and np.isfinite(a0.value), "A0 残差中边比必须出数"
    a4 = next(c for c in report.checks if c.name == "A4")
    assert a4.value is not None, "A4 鱼眼重标探针必须出结果(鱼眼侧无更高阶可升)"


def test_m4_diagnostics_refuses_to_guess_fisheye_by_length():
    """4 长 D 不许按长度反推成 M4(语义错位,必须显式点名)。"""
    objp, uvs = _synth_fisheye_views(_PARAMS_M4_TRUE, n_views=6, seed=31)
    cfg = IntrinsicDiagnosticsConfig(enabled=True, min_red_count=2)
    with pytest.raises(ValueError, match="D 形状"):
        run_battery(uvs, objp, models.intrinsics_matrix(_PARAMS_M4_TRUE),
                    _PARAMS_M4_TRUE[4:], _IMAGE_SIZE, cfg)


# ---------- 4. 装载与落盘契约 ----------

def test_m4_intrinsics_json_roundtrip(tmp_path):
    """落盘 distortion_model="M4" → 参数八名、D 恰 4 长;读回 model 自述一致。"""
    out = write_intrinsics_json(
        tmp_path / "cam1" / "intrinsics.json", cam=1, image_size=_IMAGE_SIZE,
        params=_PARAMS_M4_TRUE, K=models.intrinsics_matrix(_PARAMS_M4_TRUE),
        D=_PARAMS_M4_TRUE[4:], solver="opencv", refine_enabled=False,
        rms_train_overall=0.2128, rms_test_overall=0.2201,
        n_views_train=30, n_views_test=6, dropped_frames=[],
        solver_meta={"path": "fisheye.calibrate", "n_starts": 1},
        board=_BOARD, config_snapshot={"intrinsic": {"distortion_model": "M4"}},
        diagnostics={"enabled": False}, distortion_model="M4")
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["distortion_model"] == "M4"
    assert list(payload["params"]) == list(models.intrinsic_names("M4"))
    assert payload["D"] == pytest.approx(list(_PARAMS_M4_TRUE[4:]))

    view = load_intrinsics(out)
    assert view.model == "M4", "装载必须按名记住布局,不许按长度猜"
    assert view.params.shape == (8,) and view.D.shape == (4,)


def test_load_intrinsics_rejects_m4_d_length_mismatch(tmp_path):
    """M4 但 D 长错位(塞了 5 长针孔布局)→ 中文 ValueError,绝不静默错读。"""
    bad = {"cam": 1, "distortion_model": "M4", "image_size": [1280, 720],
           "params": {n: float(v) for n, v in
                      zip(models.intrinsic_names("M4"), _PARAMS_M4_TRUE)},
           "K": models.intrinsics_matrix(_PARAMS_M4_TRUE).tolist(),
           "D": [0.0] * 5}
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="D 的形状"):
        load_intrinsics(path)


def test_config_accepts_m4_and_rejects_unknown(tmp_path):
    """配置白名单:"M4" 合法解析,白名单外照旧中文报错。"""
    cfg = load_config(_write_config(tmp_path, _mutate("distortion_model", '"M4"')))
    assert cfg.intrinsic.distortion_model == "M4"
    with pytest.raises(ValueError, match="不合法"):
        load_config(_write_config(tmp_path, _mutate("distortion_model", '"M5"')))


# ---------- 5. 双目:M4 能走完整条外参链路 ----------

def _m4_stereo_case(n_pairs: int = 24, seed: int = 31, noise_px: float = 0.05):
    """合成一对 M4 鱼眼双目:观测 + 两路 IntrinsicsView + 相对位姿真值。"""
    rng = np.random.default_rng(seed)
    p1 = np.array([467.0, 466.0, 640.0, 360.0, -0.02, 0.008, -0.002, 0.0004])
    p2 = np.array([464.0, 465.0, 642.0, 362.0, -0.021, 0.009, -0.0019, 0.00035])
    objp = build_object_points(_BOARD)
    axis = np.array([0.3, -0.5, 0.8])
    axis = axis / np.linalg.norm(axis)
    R_rel, _ = cv2.Rodrigues(axis * np.deg2rad(2.0))
    t_rel = np.array([300.0, 15.0, 8.0])
    uv1_rows, uv2_rows = [], []
    for _ in range(n_pairs):
        rvec1 = rng.uniform(-0.3, 0.3, 3)
        tvec1 = np.array([rng.normal(0.0, 40.0), rng.normal(0.0, 40.0),
                          rng.uniform(400.0, 600.0)])
        R2 = R_rel @ cv2.Rodrigues(rvec1)[0]
        tvec2 = R_rel @ tvec1 + t_rel
        x1, _ = models.transform_to_camera(objp, rvec1, tvec1)
        x2, _ = models.transform_to_camera(objp, cv2.Rodrigues(R2)[0].ravel().astype(float),
                                           tvec2)
        uv1_rows.append(models.project(x1, p1) + rng.normal(0.0, noise_px, (len(objp), 2)))
        uv2_rows.append(models.project(x2, p2) + rng.normal(0.0, noise_px, (len(objp), 2)))
    pairs = Pairs(image_ids=tuple(f"g{i:02d}" for i in range(n_pairs)),
                  source1=("sb",) * n_pairs, source2=("sb",) * n_pairs,
                  routes=(("sb", "sb"),) * n_pairs,
                  uv1=np.stack(uv1_rows), uv2=np.stack(uv2_rows), objp=objp)

    def _k(p):
        return np.array([[p[0], 0.0, p[2]], [0.0, p[1], p[3]], [0.0, 0.0, 1.0]])

    i1 = IntrinsicsView(cam=1, image_size=_IMAGE_SIZE, params=p1, K=_k(p1),
                        D=p1[4:].copy(), model="M4")
    i2 = IntrinsicsView(cam=2, image_size=_IMAGE_SIZE, params=p2, K=_k(p2),
                        D=p2[4:].copy(), model="M4")
    return pairs, i1, i2, R_rel, t_rel


def test_solve_view_pose_m4_uses_precise_inverse():
    """M4 逐帧 PnP 用自家精确牛顿逆:回代 RMS 应落在噪声量级(< 0.2px)。"""
    pairs, i1, _i2, _R, _t = _m4_stereo_case(n_pairs=1)
    objp = np.asarray(pairs.objp, dtype=np.float64)
    uv = np.asarray(pairs.uv1[0], dtype=np.float64)
    ok, rvec, tvec = solve_view_pose(objp, uv, i1)
    assert ok, "M4 位姿估计必须成功"
    pts, _ = models.transform_to_camera(objp, rvec, tvec)
    resid = models.project(pts, i1.params) - uv
    rms = float(np.sqrt(np.mean(np.sum(resid ** 2, axis=1))))
    assert rms < 0.2, f"M4 位姿回代 RMS {rms:.4f}px 偏大(怀疑用了松弛的迭代逆)"


def test_joint_refuse_m4_fail_closed():
    """联合 BA 对 M4 fail-closed:θ 的 D 块按五参定死,鱼眼会被静默截断。"""
    pairs, i1, i2, _R, _t = _m4_stereo_case(n_pairs=4)
    with pytest.raises(ValueError, match="联合 BA 不支持畸变模型"):
        refine_joint(pairs, i1, i2, None, StereoJointConfig(), StereoRefineConfig())


def test_m4_stereo_end_to_end_recovers_relative_pose():
    """M4 端到端:PnP-median → 消歧 → 精化,恢复 R/t(转角 < 0.1°、平移 < 1mm)。"""
    pairs, i1, i2, R_rel, t_rel = _m4_stereo_case(n_pairs=30, seed=41)
    result = solve_stereo(pairs, i1, i2, StereoCalibConfig())
    ang = float(np.degrees(np.linalg.norm(
        cv2.Rodrigues(result.R @ R_rel.T)[0].ravel())))
    t_err = float(np.linalg.norm(result.t - t_rel))
    assert result.rms_reproj < 0.2, f"重投影 RMS {result.rms_reproj:.4f}px 偏大"
    assert ang < 0.1, f"相对旋转误差 {ang:.4f}°(须 < 0.1°)"
    assert t_err < 1.0, f"平移误差 {t_err:.4f}mm(须 < 1mm)"
