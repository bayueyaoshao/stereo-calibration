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
from scipy.optimize._numdiff import approx_derivative

# pytest rootdir 模式(tests/ 无 __init__.py)把 tests 目录插进 sys.path
from conftest import make_stereo_synth

from stereo_calib.io.config import StereoCalibConfig, StereoConfig, StereoRefineConfig
from stereo_calib.stereo.observation import IntrinsicsView, Pairs
from stereo_calib.stereo.solve import (
    disambiguate_pairs,
    make_jacobian_fn,
    make_residual_fn,
    pack_theta,
    pnp_median,
    relative_pose,
    row_index,
    solve_stereo,
    unpack_theta,
)

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


# ---------- Task 2: θ 装箱/解箱 + 行号契约 ----------


def test_pack_unpack_theta_roundtrip():
    """θ = [6 外参 | 每对 (ω,t) 6 个] 装箱与解箱互逆(随机 θ 逐位还原)。

    cam1 恒 (I, 0) 不进 θ(参考锚定系,N 对 → 6 + 6N 维);坏长度当场
    中文报错,不带病进优化器。
    """
    rng = np.random.default_rng(3)
    n = 3
    rvec_rel = rng.normal(0.0, 0.05, 3)
    t_rel = rng.normal(0.0, 30.0, 3)
    rvecs = rng.normal(0.0, 0.3, (n, 3))
    tvecs = np.column_stack([rng.normal(0.0, 40.0, n),
                             rng.normal(0.0, 40.0, n),
                             rng.uniform(500.0, 900.0, n)])
    theta = pack_theta(rvec_rel, t_rel, rvecs, tvecs)
    assert theta.shape == (6 + 6 * n,)
    assert np.array_equal(theta[:3], rvec_rel) and np.array_equal(theta[3:6], t_rel)
    rvec_rel2, t_rel2, rvecs2, tvecs2 = unpack_theta(theta)
    assert np.array_equal(rvec_rel2, rvec_rel), "外参旋转向必须逐位还原"
    assert np.array_equal(t_rel2, t_rel)
    assert np.array_equal(rvecs2, rvecs) and np.array_equal(tvecs2, tvecs)
    with pytest.raises(AssertionError, match="θ 向量长度"):
        unpack_theta(np.zeros(11))  # 对不上任何对数(6+6N)


def test_stereo_row_index_contract():
    """残差行号契约:对 → 路(cam1 先 cam2 后)→ 点 → (u,v) 交错。

    残差与雅可比只准调这一个函数定行号,各自手算下标 = 梯度接错行,
    不报错、只是收敛奇慢(Pitfall 14 的姊妹坑)。
    """
    n_points = 88
    assert row_index(0, 0, 0, 0, n_points) == 0
    assert row_index(0, 0, 0, 1, n_points) == 1
    assert row_index(0, 0, 1, 0, n_points) == 2
    assert row_index(0, 1, 0, 0, n_points) == 2 * n_points, "cam2 行紧随该对 cam1 行"
    assert row_index(1, 0, 0, 0, n_points) == 4 * n_points, "下一对从 4P 起"
    with pytest.raises(AssertionError):
        row_index(0, 2, 0, 0, n_points)  # cam_idx 只认 0/1
    with pytest.raises(AssertionError):
        row_index(0, 0, 0, 2, n_points)  # coord 只认 0/1


# ---------- Task 2: 消歧 + 精化全链(solve_stereo 编排) ----------


def test_solve_stereo_full_chain_with_flips():
    """10 对注入 2 对 uv2 翻转 → 消歧检出翻回 + 精化全链恢复 0.1°/1mm。

    这是全阶段数学风险最集中的一条:翻转对若漏网,相对位姿双峰、中位
    被拖偏;消歧后精化必须把真值恢复到转角 < 0.1°、t < 1mm(plan 全链
    口径),flip_log 留痕命中对号。
    """
    synth = make_stereo_synth(n_pairs=10, seed=7, flip_pair_indices=(1, 4))
    pairs = _pairs_from_synth(synth)
    intr1, intr2 = _intr_views(synth)
    cfg = StereoCalibConfig()  # 默认:refine 开、flip_ratio 0.5、spread 1.0°
    result = solve_stereo(pairs, intr1, intr2, cfg)

    assert result.solver == "scipy"
    assert result.flip_log["train"]["pass2_flipped"] == 2, (
        f"应检出 2 个翻转对,实测 {result.flip_log['train']['pass2_flipped']}")
    assert set(result.flip_log["train"]["pair_ids"]) == {
        synth.group_ids[1], synth.group_ids[4]}, "翻转留痕必须命中注入对号"

    ang_err = _rotation_angle_deg(result.R, synth.R_rel_true)
    t_err = float(np.linalg.norm(result.t - synth.t_rel_true))
    assert ang_err < 0.1, f"全链转角误差 {ang_err:.4f}° 应 < 0.1°"
    assert t_err < 1.0, f"全链平移误差 {t_err:.4f}mm 应 < 1mm"
    assert result.baseline_mm == pytest.approx(float(np.linalg.norm(result.t)),
                                               rel=0.0, abs=0.0)
    assert result.per_pair_rms.shape == (10,) and np.isfinite(result.per_pair_rms).all()
    assert result.rms_reproj < 0.5, f"全链 pooled RMS {result.rms_reproj:.4f}px 应 < 0.5"


def test_solve_stereo_deterministic_flip_log():
    """消歧确定性:同输入重复求解,flip_log 与 R/t 逐位一致(纯函数无随机性)。"""
    synth = make_stereo_synth(n_pairs=10, seed=7, flip_pair_indices=(1, 4))
    pairs = _pairs_from_synth(synth)
    intr1, intr2 = _intr_views(synth)
    cfg = StereoCalibConfig()
    r1 = solve_stereo(pairs, intr1, intr2, cfg)
    r2 = solve_stereo(pairs, intr1, intr2, cfg)
    assert r1.flip_log == r2.flip_log, "flip_log 必须逐位一致"
    assert np.array_equal(r1.R, r2.R) and np.array_equal(r1.t, r2.t), (
        "求解必须确定(逐位一致)")


def test_refine_disabled_returns_pass2_init_bitwise():
    """精化开关(配置控制):enabled=False → solver 如实标 pnp_median,
    结果与「消歧后 pass-2 初值」逐位相等(R/t/rvec/转角/baseline/逐对 RMS)。"""
    synth = make_stereo_synth(n_pairs=10, seed=7, flip_pair_indices=(1, 4))
    pairs = _pairs_from_synth(synth)
    intr1, intr2 = _intr_views(synth)
    cfg_off = StereoCalibConfig(
        stereo=StereoConfig(refine=StereoRefineConfig(enabled=False)))
    result = solve_stereo(pairs, intr1, intr2, cfg_off)
    assert result.solver == "pnp_median"
    assert "converged" not in result.solver_meta, "关闭精化不应出现精化收敛键"

    # 复现 solve_stereo 内部管线到 pass-2,逐位比对
    pass1 = pnp_median(pairs, intr1, intr2, None)
    pairs2, _flip = disambiguate_pairs(pairs, intr1, intr2, pass1,
                                       cfg_off.stereo.flip_ratio)
    pass2 = pnp_median(pairs2, intr1, intr2, cfg_off.stereo.median_spread_max_deg)
    assert np.array_equal(result.R, pass2.R)
    assert np.array_equal(result.t, pass2.t)
    assert np.array_equal(result.rvec, pass2.rvec)
    assert result.rotation_angle_deg == pass2.rotation_angle_deg
    assert result.baseline_mm == pass2.baseline_mm
    assert np.array_equal(result.per_pair_rms, pass2.per_pair_rms)


# ---------- Task 2: 解析稀疏雅可比(数值基准 + 稀疏结构) ----------


def test_analytic_jacobian_matches_numeric_full_theta():
    """解析稀疏雅可比对 θ 全维数值差分逐项一致(相对差 < 1e-6,6 对小场)。

    行序错位的数值防线:残差按 row_index 契约排列,雅可比行序一旦接错
    (外参列/位姿列串行),数值差分比对立刻爆炸。
    """
    synth = make_stereo_synth(n_pairs=6, seed=13)
    pairs = _pairs_from_synth(synth)
    intr1, intr2 = _intr_views(synth)
    init = pnp_median(pairs, intr1, intr2, None)
    theta0 = pack_theta(init.rvec, init.t, init.rvecs1, init.tvecs1)

    res_fn = make_residual_fn(synth.objp, pairs.uv1, pairs.uv2,
                              intr1.params, intr2.params)
    jac_num = approx_derivative(res_fn, theta0.copy(), method="3-point")
    jac_ana = make_jacobian_fn(synth.objp, pairs.uv1, pairs.uv2,
                               intr1.params, intr2.params)(theta0).toarray()
    scale = max(float(np.abs(jac_num).max()), 1e-30)
    rel = float(np.abs(jac_ana - jac_num).max()) / scale
    assert rel < 1e-6, f"解析雅可比对 θ 全维数值差分相对差 {rel:.3e}(须 < 1e-6)"


def test_jacobian_row_nnz_bounded():
    """稀疏结构:每行 nnz ≤ 12(cam2 行 = 外参 6 + 本对位姿 6);cam1 行恰 6。

    别对位姿列与 cam1 行的外参列结构为零;行-列配对错了不报错、只是
    收敛奇慢,所以结构事实单独锁定。
    """
    synth = make_stereo_synth(n_pairs=3, seed=13)
    pairs = _pairs_from_synth(synth)
    intr1, intr2 = _intr_views(synth)
    init = pnp_median(pairs, intr1, intr2, None)
    theta0 = pack_theta(init.rvec, init.t, init.rvecs1, init.tvecs1)
    jac = make_jacobian_fn(synth.objp, pairs.uv1, pairs.uv2,
                           intr1.params, intr2.params)(theta0)
    n_points = pairs.objp.shape[0]
    nnz_per_row = np.diff(jac.indptr)
    assert nnz_per_row.max() <= 12, f"每行 nnz 应 ≤ 12,实测最大 {nnz_per_row.max()}"
    cam1_rows = np.arange(jac.shape[0])
    cam1_rows = cam1_rows[(cam1_rows % (4 * n_points)) < 2 * n_points]
    assert (nnz_per_row[cam1_rows] == 6).all(), "cam1 行不碰外参列,nnz 应恰 6"


def test_refine_solver_meta_records_evidence():
    """solver_meta 判定依据不静默:status/nfev/代价/尾部改进/收敛/雅可比组合。"""
    synth = make_stereo_synth(n_pairs=10, seed=7, flip_pair_indices=(1, 4))
    pairs = _pairs_from_synth(synth)
    intr1, intr2 = _intr_views(synth)
    result = solve_stereo(pairs, intr1, intr2, StereoCalibConfig())
    for key in ("status", "nfev", "final_cost", "tail_improvement", "converged"):
        assert key in result.solver_meta, f"solver_meta 缺 {key}(判定依据不静默)"
    assert result.solver_meta["converged"] is True
    assert result.solver_meta["jacobian"] == "analytic-jac, no sparsity", (
        "雅可比组合必须留痕(决策 1:解析雅可比绝不传稀疏结构矩阵参数)")


# ---------- 源码防线(Task 2:死代码回归 + 禁用例程) ----------


def test_solve_source_guards_task2():
    """solve.py 源文防线:无稀疏结构矩阵参数死代码、无禁用双目例程调用串、
    Rodrigues 雅可比路线与 trf/lsmr/linear 配置字面量齐全(WR-02 / 决策 1)。
    """
    source = _SOLVE_SOURCE.read_text(encoding="utf-8")
    assert source.count("_distort") == 0, "畸变公式只准经 intrinsic.models"
    assert "jac_sparsity" not in source, (
        "jac 为 callable 时稀疏结构矩阵参数被 scipy 静默忽略(死计算,WR-02),"
        "不得传入")
    assert "stereoCalibrate" not in source, (
        "内建双目例程在本工程禁用(参考实测该机型 |t| 失真到 2500mm 量级)")
    assert "cv2.Rodrigues" in source, "domega 雅可比必须走 Rodrigues 第二返回值"
    assert 'tr_solver="lsmr"' in source and 'x_scale="jac"' in source, (
        "scipy 精化必须走 trf + lsmr + x_scale=jac 的稀疏配置")
    assert 'loss="linear"' in source, "loss 固定 linear(离群由剔除机制处理)"
