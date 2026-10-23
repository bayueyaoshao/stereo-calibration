"""M2 模型层数值锁定测试:cv2 parity、解析雅可比对数值差分、往返投影、Z 防线。

四层数值锁定(plan 定死阈值,无 skip 无放宽):
1. parity:project 对 cv2.projectPoints(float64 输入,同一畸变口径)< 1e-9 px;
2. 雅可比:project_jacobian / project_point_jacobian / transform_to_camera
   的解析式对 scipy 数值差分基准逐列校验 < 1e-6(专防手写旋转导数回潮);
3. 往返:project → unproject → project < 1e-9 px;
4. 防线:Z<=0 抛 AssertionError;强畸变边缘点 unproject 收敛且射线单位范数。
"""

import pathlib

import cv2
import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

from stereo_calib.intrinsic import models

# 物理合理的参数(量级取自本数据实测锚点:焦距 ~1850、主点画面中心、强径向畸变)
_PARAMS = np.array([1850.0, 1840.0, 1280.0, 720.0, -0.40, 0.23, 1.1e-4, -3.8e-5, -0.08])


def _random_cam_points(rng: np.random.Generator, n: int = 64) -> np.ndarray:
    """随机相机系点:横向 ±150、纵向 ±100、深度 400–900(棋盘实测工作距离量级)。"""
    return np.column_stack([
        rng.uniform(-150.0, 150.0, n),
        rng.uniform(-100.0, 100.0, n),
        rng.uniform(400.0, 900.0, n),
    ])


def _max_rel_diff(ana: np.ndarray, num: np.ndarray) -> float:
    """两雅可比的最大相对差:最大绝对差 / 数值基准的最大绝对值(尺度归一)。"""
    scale = max(float(np.abs(num).max()), 1e-30)
    return float(np.abs(ana - num).max()) / scale


# ---------- 1. parity:对 cv2.projectPoints(同畸变口径,float64 输入)----------

def test_project_parity_with_cv2_project_points():
    """M2 投影公式与 cv2.projectPoints 在 float64 输入下 parity < 1e-9 px。"""
    rng = np.random.default_rng(42)
    pts = _random_cam_points(rng)
    ours = models.project(pts, _PARAMS)
    k = models.intrinsics_matrix(_PARAMS)
    ref, _ = cv2.projectPoints(pts.reshape(-1, 1, 3),
                               np.zeros(3, dtype=np.float64),
                               np.zeros(3, dtype=np.float64),
                               k, _PARAMS[4:])
    ref = ref.reshape(-1, 2)
    max_abs = float(np.abs(ours - ref).max())
    assert max_abs < 1e-9, f"M2 投影对 cv2.projectPoints parity 偏差 {max_abs:.3e}px(须 < 1e-9)"


# ---------- 2. 解析雅可比对数值差分基准 ----------

def test_project_jacobian_matches_numeric():
    """project_jacobian 对 9 个内参列逐列对上数值差分(最大相对差 < 1e-6)。"""
    rng = np.random.default_rng(7)
    pts = _random_cam_points(rng)

    def f(p: np.ndarray) -> np.ndarray:
        return models.project(pts, p).ravel()

    jac_num = approx_derivative(f, _PARAMS.copy(), method="3-point")
    jac_ana = models.project_jacobian(pts, _PARAMS).reshape(2 * len(pts), 9)
    rel = _max_rel_diff(jac_ana, jac_num)
    assert rel < 1e-6, f"投影对内参的解析雅可比对数值差分相对差 {rel:.3e}(须 < 1e-6)"


def test_project_point_jacobian_matches_numeric():
    """project_point_jacobian 对相机点 3 列逐列对上数值差分(最大相对差 < 1e-6)。"""
    rng = np.random.default_rng(13)
    pts = _random_cam_points(rng, n=32)

    def f(x: np.ndarray) -> np.ndarray:
        return models.project(x.reshape(-1, 3), _PARAMS).ravel()

    x0 = pts.ravel().copy()
    jac_num = approx_derivative(f, x0, method="3-point")
    blocks = models.project_point_jacobian(pts, _PARAMS)      # (N,2,3)
    n = len(pts)
    jac_ana = np.zeros((2 * n, 3 * n), dtype=np.float64)
    for i in range(n):
        jac_ana[2 * i:2 * i + 2, 3 * i:3 * i + 3] = blocks[i]
    rel = _max_rel_diff(jac_ana, jac_num)
    assert rel < 1e-6, f"投影对相机点的解析雅可比对数值差分相对差 {rel:.3e}(须 < 1e-6)"


def test_transform_jacobian_matches_numeric():
    """transform_to_camera 的外参导数块对 rvec/tvec 数值差分 < 1e-6。

    专防手写 ∂Xc/∂ω = −[Xc]× 回潮:那只在 ω=0 成立,一般转角下 0.1 rad
    就偏 70%——本用例的 rvec 各分量 ~0.3 rad,手写形式必炸。
    """
    rng = np.random.default_rng(11)
    objp = np.column_stack([
        rng.uniform(-250.0, 250.0, 40),
        rng.uniform(-175.0, 175.0, 40),
        np.zeros(40),
    ])
    rvec = rng.normal(0.0, 0.3, 3)
    tvec = np.array([15.0, -10.0, 620.0])

    _pts_cam, dxc_dpose = models.transform_to_camera(objp, rvec, tvec, want_jacobian=True)
    assert dxc_dpose is not None
    n = len(objp)
    for label, vec, cols, pose_fn in (
        ("旋转向量", rvec, slice(0, 3),
         lambda v: models.transform_to_camera(objp, v, tvec)[0].ravel()),
        ("平移向量", tvec, slice(3, 6),
         lambda v: models.transform_to_camera(objp, rvec, v)[0].ravel()),
    ):
        jac_num = approx_derivative(pose_fn, vec.copy(), method="3-point")
        # 解析块 (N,3,3) → (3N,3):行 3i+a ↔ ∂Xc_{i,a}/∂vec_k
        jac_ana = dxc_dpose[:, :, cols].reshape(3 * n, 3)
        rel = _max_rel_diff(jac_ana, jac_num)
        assert rel < 1e-6, f"transform_to_camera 对{label}的解析雅可比相对差 {rel:.3e}(须 < 1e-6)"


def test_transform_without_jacobian_returns_none():
    """want_jacobian=False(默认)只回位姿、导数块为 None(回代/评分零开销)。"""
    pts_cam, dxc = models.transform_to_camera(
        np.array([[0.0, 0.0, 0.0]]), np.zeros(3), np.array([0.0, 0.0, 5.0]))
    assert dxc is None
    assert pts_cam.shape == (1, 3)


# ---------- 3. 往返投影 ----------

def test_project_unproject_roundtrip():
    """project → unproject → project 往返最大偏差 < 1e-9 px。"""
    rng = np.random.default_rng(19)
    pts = _random_cam_points(rng, n=48)
    uv = models.project(pts, _PARAMS)
    rays = models.unproject(uv, _PARAMS)
    uv2 = models.project(rays, _PARAMS)
    max_abs = float(np.abs(uv - uv2).max())
    assert max_abs < 1e-9, f"投影-反投影往返偏差 {max_abs:.3e}px(须 < 1e-9)"


# ---------- 4. 防线:Z<=0 拒收 / unproject 强畸变收敛 ----------

def test_project_rejects_nonpositive_z():
    """project 对含 Z<=0 的点抛 AssertionError(数学层断言,绝不带病进公式)。"""
    bad = np.array([[0.0, 0.0, -1.0], [1.0, 1.0, 2.0]])
    with pytest.raises(AssertionError, match="Z"):
        models.project(bad, _PARAMS)


def test_unproject_converges_on_strong_distortion_edges():
    """强畸变边缘点 unproject 收敛:射线单位范数、往返闭合 < 1e-9 px。

    选点口径:取画面四角的强畸变边缘点(归一化 |xd|≈0.5+,r²≈0.3 量级,
    远超真实棋盘角点的工作范围 r²≈0.15)。刻意不用图像最极端的 (0,0)/
    (2560,1440) 角像素:k1=-0.40 的强畸变下,那些像素落在畸变映射可逆
    区的折返边界附近(雅可比近奇异),是模型物理边界而非实现缺陷——
    真实棋盘角点永远到不了那里(参考工程的往返契约同样限定在可逆区内)。
    """
    uv_edge = np.array([[300.0, 200.0], [2260.0, 200.0], [2260.0, 1240.0], [300.0, 1240.0]])
    rays = models.unproject(uv_edge, _PARAMS)
    norms = np.linalg.norm(rays, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-15), f"射线未归一化:{norms}"
    uv2 = models.project(rays, _PARAMS)
    max_abs = float(np.abs(uv2 - uv_edge).max())
    assert max_abs < 1e-9, f"强畸变边缘点往返偏差 {max_abs:.3e}px(牛顿迭代未收敛?)"


# ---------- 5. 源码防线:唯一数学源 / 裁剪声明 ----------

def test_models_source_guard_single_math_source():
    """源码防线:参数布局常量与 Rodrigues 路线在位;多模型分支/注册机制不回潮。

    注意:本用例检查的是 models.py 的源文本,断言里出现的被禁词只在本文件。
    """
    source = pathlib.Path("src/stereo_calib/intrinsic/models.py").read_text(encoding="utf-8")
    assert "INTRINSIC_NAMES" in source, "参数布局常量缺失(下游按名取参依赖它)"
    assert "def project(" in source and "def unproject(" in source
    assert source.count("cv2.Rodrigues(") == 1, "旋转必须全走 cv2 现成雅可比这一条路(调用恰 1 处)"
    # 裁剪声明:参考工程的多模型分支与模型注册机制不得回潮(本工程固定九参数模型)
    for token in ("M1", "M3", "register_model"):
        assert token not in source, f"models.py 出现被裁剪的参考工程痕迹:{token}"
