"""只估位姿评分测试:已知位姿帧、大倾角错误盆地回归、失败契约、源码防线。

错误盆地回归的背景(Pitfall 6 / T-02-06):test 帧评分若走"零畸变直接
按像素解 PnP"的老路线,大倾角 + 强畸变 + 棋盘偏离主点时初值本身带畸变
偏差,会把 6 参数精化带进错误盆地——参考工程真实案例:40° 倾角帧被误评
4.82px(正确 0.07px)。本文件的回归用例在合成层锁死同场景,防线是
unproject 反投影初值路线(反投影天然完成去畸变)。

注意(源码防线自伤规避体例):被计数的调用串原文只出现在断言处,
scoring.py 的注释措辞刻意避开。
"""

from __future__ import annotations

import pathlib

import numpy as np

from stereo_calib.intrinsic import models
from stereo_calib.intrinsic.scoring import score_pose_only

# 合成真值参数(强畸变量级同本数据实测锚点;布局见 models.INTRINSIC_NAMES)
_PARAMS_TRUE = np.array([1850.0, 1840.0, 1280.0, 720.0, -0.40, 0.23, 1.1e-4, -3.8e-5, -0.08])
_PRINCIPAL = np.array([1280.0, 720.0])


def _object_points() -> np.ndarray:
    """88 个内角点的板系物点(行主序布局,与 preprocess.build_object_points 同构)。"""
    xs = np.arange(11, dtype=np.float64) * 25.0
    ys = np.arange(8, dtype=np.float64) * 25.0
    objp = np.zeros((11 * 8, 3), dtype=np.float64)
    objp[:, 0] = np.tile(xs, 8)
    objp[:, 1] = np.repeat(ys, 11)
    return objp


def _synthetic_frame(rvec: np.ndarray, tvec: np.ndarray,
                     noise_px: float = 0.05, seed: int = 42) -> np.ndarray:
    """真值参数投影的单帧合成角点 + 高斯检测噪声(真值位姿已知)。"""
    rng = np.random.default_rng(seed)
    objp = _object_points()
    pts_cam, _ = models.transform_to_camera(objp, rvec, tvec)
    return models.project(pts_cam, _PARAMS_TRUE) + rng.normal(0.0, noise_px, (88, 2))


# ---------- 1. 已知位姿合成帧:评分 ≈ 噪声底 ----------

def test_score_pose_only_known_pose_frame_rms_near_noise():
    """真值位姿合成帧(0.05px 噪声)→ 评分 < 0.2px,残差形状 (88,2),备注为空。"""
    rvec = np.array([0.25, -0.15, 0.10])
    tvec = np.array([-30.0, 20.0, 650.0])
    uv = _synthetic_frame(rvec, tvec)

    score = score_pose_only(uv, _object_points(), _PARAMS_TRUE)
    assert score.rms < 0.2, f"已知位姿帧评分 {score.rms:.4f}px(须 < 0.2,即 ≈ 噪声底)"
    assert score.residuals is not None and score.residuals.shape == (88, 2)
    assert score.note == "", f"成功路径不应有备注,实际:{score.note!r}"


# ---------- 2. 大倾角错误盆地回归(T-02-06 防线) ----------

def test_score_pose_only_large_tilt_strong_distortion_no_wrong_basin():
    """40° 大倾角 + 强畸变 + 板面偏离主点 ≥ 500px → 评分 < 1.0px。

    场景复刻参考工程错误盆地真实案例的三个要素(倾角/强畸变/偏心);
    老路线在该场景虚高数 px,反投影初值路线必须把它拉回噪声量级。
    """
    rvec = np.array([np.deg2rad(40.0), 0.0, 0.0])  # 绕 x 轴倾 40°
    tvec = np.array([60.0, -30.0, 600.0])
    uv = _synthetic_frame(rvec, tvec, seed=7)

    # 场景自证:合成帧确实满足"板面偏离主点 >= 500px"的预设
    offset = np.linalg.norm(uv - _PRINCIPAL, axis=1)
    assert offset.mean() >= 500.0, f"场景预设不成立:平均偏离主点 {offset.mean():.0f}px"

    score = score_pose_only(uv, _object_points(), _PARAMS_TRUE)
    assert score.rms < 1.0, (
        f"大倾角帧评分 {score.rms:.4f}px(须 < 1.0;虚高数 px = 错误盆地回潮)")


# ---------- 3. 失败契约:单帧失败绝不中断整批 ----------

def test_score_pose_only_nan_input_returns_failure_not_raise():
    """uv 含 NaN → 返回 rms=NaN + 非空中文备注,绝不 raise(单帧失败不中断整批)。"""
    uv = _synthetic_frame(np.array([0.2, 0.1, 0.05]), np.array([0.0, 0.0, 600.0]))
    uv[3, 1] = np.nan
    score = score_pose_only(uv, _object_points(), _PARAMS_TRUE)
    assert np.isnan(score.rms), "失败帧 rms 必须是 NaN"
    assert score.residuals is None
    assert score.note, "失败帧必须带非空中文备注"
    assert "NaN" in score.note or "非数值" in score.note


# ---------- 4. 源码防线:反投影初值路线 + 数学只经 models ----------

def test_scoring_source_guard_unproject_route():
    """scoring.py 源文防线:反投影初值路线在位;无第二份投影/畸变公式。"""
    source = pathlib.Path("src/stereo_calib/intrinsic/scoring.py").read_text(encoding="utf-8")
    assert "unproject" in source, "评分初值必须走反投影路线(天然去畸变,防错误盆地)"
    assert source.count("_distort") == 0, (
        "scoring.py 出现畸变核符号:投影/畸变公式只准经 intrinsic.models")
    assert "solvePnP" in source, "位姿初值解必须走归一化平面 solvePnP"
    assert "np.eye(3)" in source and "np.zeros(5)" in source, (
        "solvePnP 必须配单位阵 K 与零畸变(归一化平面口径)")
