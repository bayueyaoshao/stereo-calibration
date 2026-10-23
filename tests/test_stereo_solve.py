"""双目求解引擎测试:PnP-median 初值恢复、消歧翻转、精化全链、雅可比数值基准。

标定级测试走直接合成路线(照 test_intrinsic_solve.py 体例):conftest 的
make_stereo_synth 真值双目几何(R_rel/t_rel 已知)+ 0.05px 高斯检测噪声,
零渲染、零读图,秒级完成。数值阈值 plan 定死,无 skip 无放宽:
- Task 1 初值口径:转角误差 < 0.5°、t 误差 < 5mm;
- Task 2 全链口径:转角误差 < 0.1°、t 误差 < 1mm(消歧 + 精化)。

注意(源码防线自伤规避体例):本文件会对求解器源文本做子串计数防线;
凡被计数的调用串原文,只准出现在本断言文件,不得在 solve.py 的
docstring/注释里复述——solve.py 的注释措辞刻意避开那些串。
"""

from __future__ import annotations

import pathlib

import cv2
import numpy as np
import pytest

# pytest rootdir 模式(tests/ 无 __init__.py)把 tests 目录插进 sys.path
from conftest import make_stereo_synth

from stereo_calib.io.config import StereoCalibConfig
from stereo_calib.stereo.observation import IntrinsicsView, Pairs
from stereo_calib.stereo.solve import pnp_median, relative_pose

_SOLVE_SOURCE = pathlib.Path("src/stereo_calib/stereo/solve.py")


def _pairs_from_synth(synth) -> Pairs:
    """合成真值 → Pairs(不经落盘:求解级测试直接消费数组,装载契约另测)。"""
    n = len(synth.group_ids)
    return Pairs(image_ids=synth.group_ids,
                 source1=("sb",) * n, source2=("sb",) * n,
                 routes=(("sb", "sb"),) * n,
                 uv1=synth.uv1, uv2=synth.uv2, objp=synth.objp)


def _intr_views(synth) -> tuple[IntrinsicsView, IntrinsicsView]:
    """合成真值内参 → 两路 IntrinsicsView(D 取 params 后五项,dist 布局)。"""
    v1 = IntrinsicsView(cam=1, image_size=(2560, 1440), params=synth.params1,
                        K=synth.K1, D=synth.params1[4:9].copy())
    v2 = IntrinsicsView(cam=2, image_size=(2560, 1440), params=synth.params2,
                        K=synth.K2, D=synth.params2[4:9].copy())
    return v1, v2


def _rotation_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """两旋转矩阵的夹角(度):相对旋转经 Rodrigues 取模长。"""
    rvec, _ = cv2.Rodrigues(R_a.T @ R_b)
    return float(np.degrees(np.linalg.norm(rvec)))


# ---------- Task 1: relative_pose 方向语义(Pitfall 6 方向唯一实现) ----------


def test_relative_pose_direction_semantics():
    """方向语义 X_cam2 = R_rel @ X_cam1 + t_rel:用已知两路位姿逐点验证。

    搞反方向(差个转置)时重投影 RMS 可能看不出灾难性差异(Pitfall 6),
    这里直接在点变换层面锁死:cam1 系的点经 (R_rel, t_rel) 必须落到
    cam2 系的真值位置。
    """
    rng = np.random.default_rng(11)
    rvec_a = rng.normal(0.0, 0.3, 3)
    t_a = np.array([10.0, -20.0, 700.0])
    rvec_b = rng.normal(0.0, 0.3, 3)
    t_b = np.array([-15.0, 5.0, 690.0])
    R_a, _ = cv2.Rodrigues(rvec_a)
    R_b, _ = cv2.Rodrigues(rvec_b)

    R_rel, t_rel = relative_pose(R_a, t_a, R_b, t_b)
    assert R_rel.shape == (3, 3) and t_rel.shape == (3,)

    # 真值语义推导:同一世界点 X_w,X_a = R_a@X_w + t_a,X_b = R_b@X_w + t_b
    # 消去 X_w 得 X_b = R_rel@X_a + t_rel——逐点比对。
    x_w = rng.normal(0.0, 50.0, (20, 3))
    x_a = (R_a @ x_w.T).T + t_a
    x_b_true = (R_b @ x_w.T).T + t_b
    x_b_rel = (R_rel @ x_a.T).T + t_rel
    assert np.allclose(x_b_rel, x_b_true, atol=1e-9), (
        "relative_pose 方向语义被破坏:X_cam2 = R_rel @ X_cam1 + t_rel 不成立"
        "(cam1→cam2 方向唯一实现,Pitfall 6)")


# ---------- Task 1: pnp_median 合成真值初值恢复 ----------


def test_pnp_median_recovers_synthetic_truth_initial():
    """合成真值 8 对(无翻转注入)→ 初值口径恢复:转角 < 0.5°、t < 5mm。

    同锁:baseline_mm == ‖t_est‖(单位 mm,objp 是 mm);n_valid == 8;
    干净合成下离散度 spread_deg < 0.5(刚架相对位姿应紧密聚集)。
    """
    synth = make_stereo_synth(n_pairs=8, seed=7)
    pairs = _pairs_from_synth(synth)
    intr1, intr2 = _intr_views(synth)
    init = pnp_median(pairs, intr1, intr2, 1.0)

    assert init.n_valid == 8
    assert _rotation_angle_deg(init.R, synth.R_rel_true) < 0.5, (
        f"初值转角误差 {_rotation_angle_deg(init.R, synth.R_rel_true):.4f}° 应 < 0.5°")
    t_err = float(np.linalg.norm(init.t - synth.t_rel_true))
    assert t_err < 5.0, f"初值平移误差 {t_err:.3f}mm 应 < 5mm"
    assert init.baseline_mm == pytest.approx(float(np.linalg.norm(init.t)), rel=0.0, abs=0.0), (
        "baseline_mm 必须恰为 ‖t‖(cam1 参考系,t 单位 mm)")
    assert init.rvec.shape == (3,) and init.R.shape == (3, 3) and init.t.shape == (3,)
    assert init.rotation_angle_deg < 90.0, "刚架两相机近同向,转角应远离 90°(A2)"
    for arr in (init.rvecs1, init.tvecs1, init.rvecs2, init.tvecs2):
        assert arr.shape == (8, 3) and np.isfinite(arr).all()
    assert init.per_pair_ok == (True,) * 8
    assert init.spread_deg < 0.5, f"干净合成离散度 {init.spread_deg:.4f}° 应 < 0.5°"


def test_pnp_median_per_pair_rms_finite_and_scaled():
    """逐对 RMS 判据尺:形状 (8,)、全有限正数、量级跟随噪声(0.05px → < 0.5px)。

    全 float64 回代(经 models),绝不用 OpenCV 返回值当指标(Pitfall 12
    的双目版):这是 03-03 剔除循环的判据尺,口径必须从这里锁死。
    """
    synth = make_stereo_synth(n_pairs=8, seed=7)
    pairs = _pairs_from_synth(synth)
    intr1, intr2 = _intr_views(synth)
    init = pnp_median(pairs, intr1, intr2, 1.0)

    assert init.per_pair_rms.shape == (8,)
    assert init.per_pair_rms.dtype == np.float64
    assert np.isfinite(init.per_pair_rms).all() and (init.per_pair_rms > 0).all()
    assert (init.per_pair_rms < 0.5).all(), (
        f"0.05px 噪声下逐对 RMS 应 < 0.5px,实测最大 {init.per_pair_rms.max():.4f}")


def test_pnp_median_spread_guard_raises_with_measured_value():
    """离散度防线:阈值压到 1e-4°(干净合成也超)→ 中文 ValueError 带实测数值。

    防 Rodrigues 中位退化(Pitfall 1):离散度过大说明有翻转帧/孪生解
    污染,中位数被拖偏时必须当场报错而不是静默出坏外参。
    """
    synth = make_stereo_synth(n_pairs=8, seed=7)
    pairs = _pairs_from_synth(synth)
    intr1, intr2 = _intr_views(synth)
    with pytest.raises(ValueError, match="离散度"):
        pnp_median(pairs, intr1, intr2, 1.0e-4)


def test_pnp_median_valid_underflow_raises():
    """有效对数 < 2 → 中文报错不硬算(一对定不了相对位姿)。"""
    synth = make_stereo_synth(n_pairs=1, seed=7)
    pairs = _pairs_from_synth(synth)
    intr1, intr2 = _intr_views(synth)
    with pytest.raises(ValueError, match="有效"):
        pnp_median(pairs, intr1, intr2, 1.0)


# ---------- 源码防线(Task 1 雏形:单一数学源 + dtype 纪律) ----------


def test_solve_source_guards_task1():
    """solve.py 源文防线:无畸变核符号、PnP 调用形态与中位聚合齐全。

    被计数的调用串原文只在本断言文件出现;solve.py 的注释措辞避开它们
    (自伤规避体例,照 test_intrinsic_solve.py)。
    """
    source = _SOLVE_SOURCE.read_text(encoding="utf-8")
    assert source.count("_distort") == 0, (
        "solve.py 出现畸变核符号:投影/畸变公式只准经 intrinsic.models(单一数学源)")
    assert "relative_pose" in source, "相对位姿必须唯一实现(cam1→cam2 方向语义)"
    assert "SOLVEPNP_ITERATIVE" in source, "solvePnP 必须走迭代法(带畸变原始像素输入)"
    assert "np.median" in source, "初值聚合必须是中位数(对翻转帧/孪生解鲁棒)"
    assert "astype(np.float32)" in source, (
        "solvePnP 装包必须显式降 float32(绑定层 dtype 陷阱,Pitfall 7)")
    assert "median_spread" in source, "离散度防线阈值必须来自配置参数"
