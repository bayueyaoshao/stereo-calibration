"""标定求解引擎测试:合成真值恢复、float64 出包、初值口径、失败契约。

标定级测试走直接合成路线(RESEARCH Code Examples 第 1 节):真值内参 +
随机位姿经 cv2 的投影工具合成 (V, 88, 2) 角点 + 0.05px 高斯检测噪声,
不经图像渲染,秒级完成。数值阈值 plan 定死,无 skip 无放宽。

注意(源码防线自伤规避体例):本文件后续会对求解器源文本做子串计数
防线;凡被计数的调用串原文,只准出现在断言所在文件,不得在求解器
docstring/注释里复述——本文件的注释措辞刻意避开那些串。
"""

from __future__ import annotations

import dataclasses
import pathlib

import cv2
import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

from stereo_calib.intrinsic import models
from stereo_calib.intrinsic.metrics import compute_metrics
from stereo_calib.intrinsic.solve import (
    SolveResult,
    make_jacobian_fn,
    make_residual_fn,
    pack_theta,
    row_index,
    solve_calibration,
    solve_opencv,
    unpack_theta,
)
from stereo_calib.io.config import RefineConfig

# 合成真值参数(量级取自本数据实测锚点:焦距 ~1850、主点画面中心、强径向畸变)
_PARAMS_TRUE = np.array([1850.0, 1840.0, 1280.0, 720.0, -0.40, 0.23, 1.1e-4, -3.8e-5, -0.08])
_IMAGE_SIZE = (2560, 1440)

# 切向对(p1,p2)的绝对判别下限:真值量级 ~1e-4,对像素的全部影响在千分之几
# 像素,比 0.05px 检测噪声低一个量级以上——12 视图下"相对误差"对它们没有
# 判别力(即使无噪估计器也定不到 5%),所以只对它们用绝对阈值兜底,
# 其余 7 项一律走 5e-2 相对阈值(plan 初值口径)。
_TANGENT_ABS_TOL = 5.0e-5


def _object_points() -> np.ndarray:
    """88 个内角点的板系物点(行主序布局,与 preprocess.build_object_points 同构)。"""
    xs = np.arange(11, dtype=np.float64) * 25.0
    ys = np.arange(8, dtype=np.float64) * 25.0
    objp = np.zeros((11 * 8, 3), dtype=np.float64)
    objp[:, 0] = np.tile(xs, 8)
    objp[:, 1] = np.repeat(ys, 11)
    return objp


def _synthetic_views(n_views: int = 12, noise_px: float = 0.05, seed: int = 42) -> np.ndarray:
    """合成真值视图:真值内参 + 随机位姿投影 + 高斯噪声(不需要渲染图像)。

    位姿采样:转角 ~0.3rad、横向 ±50mm 量级、深度 400–900mm(实测工作距离)。
    """
    rng = np.random.default_rng(seed)
    objp = _object_points()
    k = models.intrinsics_matrix(_PARAMS_TRUE)
    dist = _PARAMS_TRUE[4:]
    views = []
    for _ in range(n_views):
        rvec = rng.normal(0.0, 0.3, 3)
        tvec = np.array([rng.normal(0.0, 50.0),
                         rng.normal(0.0, 50.0),
                         rng.uniform(400.0, 900.0)])
        proj, _ = cv2.projectPoints(objp.reshape(-1, 1, 3), rvec, tvec, k, dist)
        views.append(proj.reshape(-1, 2) + rng.normal(0.0, noise_px, (88, 2)))
    return np.asarray(views)


# ---------- 1. OpenCV 初值路径:合成真值恢复(初值口径 5e-2) ----------

def test_solve_opencv_recovers_synthetic_truth_within_initial_tolerance():
    """12 视图合成真值 → 初值逐项相对误差 < 5e-2(切向对按绝对判别下限)。"""
    objp = _object_points()
    uvs = _synthetic_views()
    result = solve_opencv(uvs, objp, _IMAGE_SIZE)

    assert result.solver == "opencv"
    assert result.params.shape == (9,)
    # 非切向 7 项:逐项相对误差 < 5e-2(初值口径)
    rel = np.abs(result.params - _PARAMS_TRUE) / np.abs(_PARAMS_TRUE)
    non_tangent = [0, 1, 2, 3, 4, 5, 8]  # fx,fy,cx,cy,k1,k2,k3
    worst = float(rel[non_tangent].max())
    assert worst < 5e-2, f"初值最大相对误差 {worst:.4f}(须 < 5e-2):{rel}"
    # 切向对 p1,p2:像素效应低于检测噪声,相对误差无判别力,用绝对下限
    assert abs(result.params[6] - _PARAMS_TRUE[6]) < _TANGENT_ABS_TOL, (
        f"p1 初值偏差 {result.params[6] - _PARAMS_TRUE[6]:.3e}(须 < {_TANGENT_ABS_TOL:.0e})")
    assert abs(result.params[7] - _PARAMS_TRUE[7]) < _TANGENT_ABS_TOL, (
        f"p2 初值偏差 {result.params[7] - _PARAMS_TRUE[7]:.3e}(须 < {_TANGENT_ABS_TOL:.0e})")

    assert result.rvecs.shape == (12, 3)
    assert result.tvecs.shape == (12, 3)
    assert float(result.solver_meta["opencv_rms"]) > 0.0, (
        "solver_meta 必须带正的 OpenCV 内部 RMS 作对账参考")


def test_solve_opencv_outputs_all_float64():
    """绑定层 32 位进、64 位出:params/rvecs/tvecs dtype 全 float64。"""
    objp = _object_points()
    uvs = _synthetic_views(n_views=6)
    result = solve_opencv(uvs, objp, _IMAGE_SIZE)
    assert result.params.dtype == np.float64
    assert result.rvecs.dtype == np.float64
    assert result.tvecs.dtype == np.float64


def test_solve_result_has_no_rms_metric_field():
    """OpenCV 返回的 RMS 只准进 solver_meta 对账,SolveResult 无 rms 指标字段。"""
    names = {f.name for f in dataclasses.fields(SolveResult)}
    assert "rms" not in names, f"SolveResult 出现 rms 指标字段:{names}(指标只准走 metrics 回代尺)"


# ---------- 2. 失败契约:入口防线中文报错 ----------

def test_solve_opencv_rejects_bad_shapes_and_too_few_views():
    """入口防线:形状不合/视图数不足 → 中文 ValueError(带视图数与 image_size)。"""
    objp = _object_points()
    uvs = _synthetic_views()
    with pytest.raises(ValueError, match="形状"):
        solve_opencv(uvs[:, :, :1], objp, _IMAGE_SIZE)
    with pytest.raises(ValueError, match="视图"):
        solve_opencv(uvs[:1], objp, _IMAGE_SIZE)


# ---------- 3. scipy 精化:真值恢复(1e-3 口径)/ 不劣化 / 实用收敛 ----------

def test_solve_calibration_refine_recovers_truth_to_sub_pixel():
    """12 视图合成真值 → 精化后 fx 相对误差 < 1e-3,回代 RMS < 0.2px。

    同一把尺对账:scipy 解的 float64 回代 RMS 不得超过 OpenCV 初值解
    + 1e-6(精化不劣化;残差同构,代价单调不增保证这一点)。
    """
    objp = _object_points()
    uvs = _synthetic_views()
    result = solve_calibration(uvs, objp, _IMAGE_SIZE, RefineConfig())

    assert result.solver == "scipy"
    fx_rel = abs(result.params[0] - _PARAMS_TRUE[0]) / _PARAMS_TRUE[0]
    assert fx_rel < 1e-3, f"精化后 fx 相对误差 {fx_rel:.3e}(须 < 1e-3)"

    m = compute_metrics(objp, uvs, result.params, result.rvecs, result.tvecs)
    assert m.rms_overall < 0.2, f"精化后回代 RMS {m.rms_overall:.4f}px(须 < 0.2)"

    init = solve_opencv(uvs, objp, _IMAGE_SIZE)
    m_init = compute_metrics(objp, uvs, init.params, init.rvecs, init.tvecs)
    assert m.rms_overall <= m_init.rms_overall + 1e-6, (
        f"精化 RMS {m.rms_overall:.6f} 劣于初值 {m_init.rms_overall:.6f}(精化不劣化)")

    for key in ("status", "nfev", "final_cost", "tail_improvement", "converged", "opencv_rms"):
        assert key in result.solver_meta, f"solver_meta 缺 {key}(判定依据不静默)"
    assert result.solver_meta["converged"] is True


def test_refine_disabled_returns_initial_solution():
    """精化开关(配置控制):enabled=False → solver 如实标 opencv,结果等于初值解。"""
    objp = _object_points()
    uvs = _synthetic_views(n_views=6)
    result = solve_calibration(uvs, objp, _IMAGE_SIZE, RefineConfig(enabled=False))
    assert result.solver == "opencv"
    init = solve_opencv(uvs, objp, _IMAGE_SIZE)
    assert np.array_equal(result.params, init.params)
    assert np.array_equal(result.rvecs, init.rvecs)
    assert np.array_equal(result.tvecs, init.tvecs)


def test_row_index_contract_interleaves_uv():
    """行序契约:第 view 视图第 point 角点的 u 残差住偶数行、v 住紧随奇数行。

    残差与雅可比只准调这一个函数定行号,各自手算下标 = 梯度接错行,
    不报错、只是收敛奇慢(静默失败防线)。
    """
    n_points = 88
    assert row_index(0, 0, 0, n_points) == 0
    assert row_index(0, 0, 1, n_points) == 1
    assert row_index(0, 1, 0, n_points) == 2
    assert row_index(0, 1, 1, n_points) == 3
    assert row_index(1, 0, 0, n_points) == 2 * n_points
    with pytest.raises(AssertionError, match="uv_axis"):
        row_index(0, 0, 2, n_points)


def test_pack_unpack_theta_roundtrip():
    """θ 联合装箱 [9 内参 | 视图 i 的 (ω,t) 6 个] 与解箱互逆,坏长度当场报错。"""
    objp = _object_points()
    uvs = _synthetic_views(n_views=3)
    init = solve_opencv(uvs, objp, _IMAGE_SIZE)
    theta = pack_theta(init.params, init.rvecs, init.tvecs)
    assert theta.shape == (9 + 6 * 3,)
    params, rvecs, tvecs = unpack_theta(theta)
    assert np.array_equal(params, init.params)
    assert np.array_equal(rvecs, init.rvecs)
    assert np.array_equal(tvecs, init.tvecs)
    with pytest.raises(AssertionError, match="θ 向量长度"):
        unpack_theta(np.zeros(13))


def test_analytic_sparse_jacobian_matches_numeric_full_theta():
    """解析稀疏雅可比对 θ 全维数值差分逐项一致(相对差 < 1e-6,2 视图小场)。

    这是行序错位的数值防线:残差按 row_index 契约交错排列,雅可比行序
    一旦接错,数值差分比对立刻爆炸。
    """
    objp = _object_points()
    uvs = _synthetic_views(n_views=2, seed=13)
    init = solve_opencv(uvs, objp, _IMAGE_SIZE)
    theta0 = pack_theta(init.params, init.rvecs, init.tvecs)

    jac_num = approx_derivative(make_residual_fn(objp, uvs), theta0.copy(), method="3-point")
    jac_ana = make_jacobian_fn(objp, uvs)(theta0).toarray()
    scale = max(float(np.abs(jac_num).max()), 1e-30)
    rel = float(np.abs(jac_ana - jac_num).max()) / scale
    assert rel < 1e-6, f"解析雅可比对 θ 全维数值差分相对差 {rel:.3e}(须 < 1e-6)"


# ---------- 4. 源码防线:残差只经 models / Rodrigues 雅可比路线 / trf+lsmr 配置 ----------

def test_solve_source_guard_models_only_math():
    """solve.py 源文防线:无第二份投影/畸变公式;domega 走 Rodrigues 现成雅可比。

    注意:被计数的串原文只在本文件与断言处出现,solve.py 的注释措辞
    刻意避开它们(自伤规避体例)。
    """
    source = pathlib.Path("src/stereo_calib/intrinsic/solve.py").read_text(encoding="utf-8")
    assert source.count("_distort") == 0, (
        "solve.py 出现畸变核符号:投影/畸变公式只准经 intrinsic.models(单一数学源)")
    assert "cv2.Rodrigues" in source, (
        "solve.py 须声明 domega 雅可比来自 Rodrigues 第二返回值路线(严禁手写)")
    assert 'tr_solver="lsmr"' in source and 'x_scale="jac"' in source, (
        "scipy 精化必须走 trf + lsmr + x_scale=jac 的稀疏配置")
    assert 'loss="linear"' in source, "loss 固定 linear(离群由 02-03 剔除机制处理)"
