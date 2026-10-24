"""联合 BA 求解器测试:内参扰动恢复、雅可比全维数值基准、变体布局、
先验拉回、散布防线(03-05 gap-closure 的全部引擎级行为)。

体例照 tests/test_stereo_solve.py:conftest 的 make_stereo_synth 真值双目
几何 + 0.05px 高斯检测噪声,零渲染、零读图,秒级。核心场景(计划 behavior
定死,无 skip 无放宽):

- 内参扰动恢复:两路 params 各扰动(fx/fy +1.5%、cx/cy +6px、k1 +0.03)
  模拟「Phase 2 互不兼容初值」→ solve_stereo_joint(variant=full_prior、
  prior_weight=1e-2)必须把内参与外参同时拉回真值(内参 f/c 相对 < 0.3%、
  k1 绝对 < 0.05;t < 2mm、转角 < 0.05°);同扰动下 rigid 路径
  (solve_stereo,内参固定)达不到该精度——联合路线必要性的引擎侧反证。

注意(源码防线自伤规避体例):guards 对 joint.py 源文做子串计数;被计数的
调用串原文只准出现在断言文件,不得在 joint.py 的 docstring/注释里复述。
"""

from __future__ import annotations

import dataclasses

import cv2
import numpy as np
import pytest

# pytest rootdir 模式(tests/ 无 __init__.py)把 tests 目录插进 sys.path
from conftest import make_stereo_synth
from scipy.optimize._numdiff import approx_derivative

from stereo_calib.io.config import (
    StereoCalibConfig,
    StereoConfig,
    StereoJointConfig,
    StereoRefineConfig,
)
from stereo_calib.stereo.joint import (
    JointSolveOutput,
    make_jacobian_joint_fn,
    make_residual_joint_fn,
    pack_theta_joint,
    refine_joint,
    solve_stereo_joint,
    unpack_theta_joint,
)
from stereo_calib.stereo.observation import IntrinsicsView, Pairs
from stereo_calib.stereo.solve import pnp_median, solve_stereo


def _pairs_from_synth(synth) -> Pairs:
    """合成真值 → Pairs(不经落盘:引擎级测试直接消费数组)。"""
    n = len(synth.group_ids)
    return Pairs(image_ids=synth.group_ids,
                 source1=("sb",) * n, source2=("sb",) * n,
                 routes=(("sb", "sb"),) * n,
                 uv1=synth.uv1, uv2=synth.uv2, objp=synth.objp)


def _perturb_params(params: np.ndarray) -> np.ndarray:
    """模拟「Phase 2 互不兼容初值」的内参扰动:fx/fy +1.5%、cx/cy +6px、k1 +0.03。"""
    p = np.asarray(params, dtype=np.float64).copy()
    p[0] *= 1.015
    p[1] *= 1.015
    p[2] += 6.0
    p[3] += 6.0
    p[4] += 0.03
    return p


def _intr_views(params1: np.ndarray, params2: np.ndarray
                ) -> tuple[IntrinsicsView, IntrinsicsView]:
    """params 九名 → 两路 IntrinsicsView(D 取 params 后五项,dist 布局)。"""

    def _kd(p: np.ndarray):
        k = np.array([[p[0], 0.0, p[2]], [0.0, p[1], p[3]], [0.0, 0.0, 1.0]])
        return IntrinsicsView(cam=0, image_size=(2560, 1440), params=p,
                              K=k, D=p[4:9].copy())

    return _kd(params1), _kd(params2)


def _rotation_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """两旋转矩阵的夹角(度):相对旋转经 Rodrigues 取模长。"""
    rvec, _ = cv2.Rodrigues(R_a.T @ R_b)
    return float(np.degrees(np.linalg.norm(rvec)))


def _joint_cfg(variant: str = "full_prior", prior_weight: float = 1e-2,
               spread_ratio_max: float = 5.0) -> StereoCalibConfig:
    return StereoCalibConfig(stereo=StereoConfig(joint=StereoJointConfig(
        enabled=True, variant=variant, prior_weight=prior_weight,
        spread_ratio_max=spread_ratio_max)))


# ---------- 行为 1:合成内参扰动恢复(联合路线必要性的直接证据) ----------


def test_joint_recovers_truth_under_intrinsic_perturbation():
    """两路内参互不兼容扰动下 solve_stereo_joint(full_prior)恢复真值。

    精化 params 对真值:f/c 相对误差 < 0.3%、k1 绝对误差 < 0.05(两路);
    外参 t 对 t_rel_true 误差 < 2mm、转角误差 < 0.05°;同扰动 rigid 路径
    达不到该精度(内参固定在扰动值,误差被逐对位姿吸收)。
    """
    synth = make_stereo_synth(n_pairs=10, seed=7)
    pairs = _pairs_from_synth(synth)
    intr1p, intr2p = _intr_views(_perturb_params(synth.params1),
                                 _perturb_params(synth.params2))
    out = solve_stereo_joint(pairs, intr1p, intr2p, _joint_cfg())
    assert isinstance(out, JointSolveOutput)
    assert out.result.solver == "scipy_joint"
    assert "train" in out.result.flip_log

    for cam, refined, truth in ((1, out.params1, synth.params1),
                                (2, out.params2, synth.params2)):
        rel_fc = np.abs(refined[:4] - truth[:4]) / truth[:4]
        assert rel_fc.max() < 0.003, (
            f"cam{cam} 精化 f/c 相对误差 {rel_fc.max():.4f} 应 < 0.3%")
        assert abs(refined[4] - truth[4]) < 0.05, (
            f"cam{cam} 精化 k1 绝对误差 {abs(refined[4] - truth[4]):.4f} 应 < 0.05")

    t_err = float(np.linalg.norm(out.result.t - synth.t_rel_true))
    assert t_err < 2.0, f"联合精化 t 误差 {t_err:.3f}mm 应 < 2mm"
    ang_err = _rotation_angle_deg(out.result.R, synth.R_rel_true)
    assert ang_err < 0.05, f"联合精化转角误差 {ang_err:.4f}° 应 < 0.05°"

    # joint_meta 契约:五键在位、converged 如实为真
    assert set(out.joint_meta) >= {"variant", "prior_weight", "converged",
                                   "intr_drift", "spread_ratio"}
    assert out.joint_meta["variant"] == "full_prior"
    assert out.joint_meta["converged"] is True

    # 接线反证的引擎侧基线:同扰动 rigid 路径(内参固定)达不到该精度
    rigid = solve_stereo(pairs, intr1p, intr2p, _joint_cfg())
    rigid_t_err = float(np.linalg.norm(rigid.t - synth.t_rel_true))
    assert rigid_t_err > t_err, (
        f"同扰动下 rigid t 误差 {rigid_t_err:.3f}mm 应劣于联合路线 {t_err:.3f}mm"
        f"(否则联合 BA 的必要性证据不成立)")


# ---------- 行为 2:雅可比 θ 全维数值差分基准 ----------


def test_joint_jacobian_matches_numeric_full_dim():
    """解析稀疏雅可比(含先验行)对 scipy approx_derivative 全维差分 < 1e-6。

    full_prior 变体覆盖全部列块:K1/D1/K2/D2 内参列 + 外参列 + 逐对位姿列,
    以及 18 行先验的对角块——任何一块接错行都不报错、只是收敛奇慢,必须
    数值基准当场抓住。
    """
    synth = make_stereo_synth(n_pairs=3, seed=13)
    pairs = _pairs_from_synth(synth)
    p1p, p2p = _perturb_params(synth.params1), _perturb_params(synth.params2)
    intr1p, intr2p = _intr_views(p1p, p2p)
    init = pnp_median(pairs, intr1p, intr2p, None)
    theta0 = pack_theta_joint(p1p, p2p, init.rvec, init.t,
                              init.rvecs1, init.tvecs1, "full_prior")

    res_fn = make_residual_joint_fn(synth.objp, pairs.uv1, pairs.uv2,
                                    p1p, p2p, "full_prior", 1e-2)
    jac_num = approx_derivative(res_fn, theta0.copy(), method="3-point")
    jac_ana = make_jacobian_joint_fn(
        synth.objp, pairs.uv1, pairs.uv2, p1p, p2p, "full_prior", 1e-2
    )(theta0).toarray()

    n_pairs, n_points = pairs.uv1.shape[0], pairs.uv1.shape[1]
    assert jac_ana.shape == (4 * n_pairs * n_points + 18, theta0.shape[0])
    scale = max(float(np.abs(jac_num).max()), 1e-30)
    rel = float(np.abs(jac_ana - jac_num).max()) / scale
    assert rel < 1e-6, f"联合雅可比全维数值差分相对差 {rel:.3e}(须 < 1e-6)"


# ---------- 行为 3:变体布局(冻结块不进 θ,解包原样回填) ----------


def test_variant_layout_theta_and_roundtrip():
    """fc / fc_prior 的 θ 不含 D 块(解包后 D 原样回填初值);full_prior 含;
    pack/unpack 往返一致;白名单外变体拒收。"""
    synth = make_stereo_synth(n_pairs=4, seed=11)
    pairs = _pairs_from_synth(synth)
    p1p, p2p = _perturb_params(synth.params1), _perturb_params(synth.params2)
    intr1p, intr2p = _intr_views(p1p, p2p)
    init = pnp_median(pairs, intr1p, intr2p, None)
    n_pairs = pairs.uv1.shape[0]

    for variant in ("fc", "fc_prior"):
        theta = pack_theta_joint(p1p, p2p, init.rvec, init.t,
                                 init.rvecs1, init.tvecs1, variant)
        assert theta.shape == (8 + 6 + 6 * n_pairs,), (
            f"{variant} 的 θ 维度应 = 8 内参 + 6 外参 + 6×对数,实际 {theta.shape}")
        p1, p2, rvec, t, rvecs, tvecs = unpack_theta_joint(
            theta, p1p, p2p, variant)
        assert p1.shape == (9,) and p2.shape == (9,)
        assert np.array_equal(p1[4:], p1p[4:]), "fc 变体 D1 必须原样回填初值"
        assert np.array_equal(p2[4:], p2p[4:]), "fc 变体 D2 必须原样回填初值"
        assert np.allclose(p1[:4], theta[:4])
        assert np.allclose(rvec, init.rvec) and np.allclose(t, init.t)
        assert np.allclose(rvecs, init.rvecs1) and np.allclose(tvecs, init.tvecs1)

    theta_full = pack_theta_joint(p1p, p2p, init.rvec, init.t,
                                  init.rvecs1, init.tvecs1, "full_prior")
    assert theta_full.shape == (18 + 6 + 6 * n_pairs,)
    p1, p2, rvec, t, rvecs, tvecs = unpack_theta_joint(
        theta_full, p1p, p2p, "full_prior")
    assert np.allclose(p1, p1p) and np.allclose(p2, p2p)
    assert np.allclose(rvecs, init.rvecs1) and np.allclose(tvecs, init.tvecs1)
    # 往返:再装回去必须逐位一致
    theta_back = pack_theta_joint(p1, p2, rvec, t, rvecs, tvecs, "full_prior")
    assert np.array_equal(theta_back, theta_full)

    with pytest.raises(ValueError, match="不合法"):
        pack_theta_joint(p1p, p2p, init.rvec, init.t,
                         init.rvecs1, init.tvecs1, "bogus_variant")


# ---------- 行为 4:先验拉回(强先验锁死内参;无先验自由漂移) ----------


def test_prior_pullback_and_free_drift():
    """full_prior 且 prior_weight=1e6 → 精化后内参 ≈ 初值(相对漂移 < 1e-6);
    fc(无先验)内参漂移不受约束(漂移量显著大于强先验)。"""
    synth = make_stereo_synth(n_pairs=6, seed=21)
    pairs = _pairs_from_synth(synth)
    p1p, p2p = _perturb_params(synth.params1), _perturb_params(synth.params2)
    intr1p, intr2p = _intr_views(p1p, p2p)
    init = pnp_median(pairs, intr1p, intr2p, None)

    strong = refine_joint(pairs, intr1p, intr2p, init,
                          StereoJointConfig(enabled=True, variant="full_prior",
                                            prior_weight=1e6),
                          StereoRefineConfig())
    for refined, init_p in ((strong.params1, p1p), (strong.params2, p2p)):
        rel = np.abs(refined[:4] - init_p[:4]) / init_p[:4]
        assert rel.max() < 1e-6, f"强先验下 f/c 相对漂移 {rel.max():.2e} 应 < 1e-6"
        assert np.abs(refined[4:] - init_p[4:]).max() < 1e-6, (
            "强先验下 D 绝对漂移应 < 1e-6")

    free = refine_joint(pairs, intr1p, intr2p, init,
                        StereoJointConfig(enabled=True, variant="fc",
                                          prior_weight=1.0),
                        StereoRefineConfig())
    drift_free = float((np.abs(free.params1[:4] - p1p[:4]) / p1p[:4]).max())
    drift_strong = float((np.abs(strong.params1[:4] - p1p[:4]) / p1p[:4]).max())
    assert drift_free > drift_strong, (
        f"fc 无先验漂移 {drift_free:.2e} 应大于强先验 {drift_strong:.2e}"
        f"(先验拉回机制失效)")


# ---------- 行为 5:joint 散布防线(fail-closed) ----------


def test_joint_spread_defense_fail_closed():
    """注入一对 uv 大偏移 → 精化后 max/median 超阈值 → 中文 ValueError;
    阈值放宽时不触发(防线判别力在案)。"""
    synth = make_stereo_synth(n_pairs=8, seed=7)
    pairs = _pairs_from_synth(synth)
    p1p, p2p = _perturb_params(synth.params1), _perturb_params(synth.params2)
    intr1p, intr2p = _intr_views(p1p, p2p)

    uv1_bad = np.array(pairs.uv1, dtype=np.float64, copy=True)
    uv1_bad[3, :20] += 40.0  # 单对大偏移:该对残差必然显著高于其余
    pairs_bad = Pairs(image_ids=pairs.image_ids, source1=pairs.source1,
                      source2=pairs.source2, routes=pairs.routes,
                      uv1=uv1_bad, uv2=pairs.uv2, objp=pairs.objp)

    with pytest.raises(ValueError, match="散布"):
        solve_stereo_joint(pairs_bad, intr1p, intr2p,
                           _joint_cfg(variant="fc", spread_ratio_max=5.0))

    out = solve_stereo_joint(pairs_bad, intr1p, intr2p,
                             _joint_cfg(variant="fc", spread_ratio_max=1e9))
    assert out.joint_meta["spread_ratio"] > 5.0, (
        "注入坏对后 spread_ratio 应 > 5(防线判别力的数值证据)")


# ---------- 畸变模型边界:M8 fail-closed 拒收(2026-10-24 增补) ----------

def test_refine_joint_rejects_m8_fail_closed():
    """联合 BA 当前仅支持 M2:M8 内参入口当场拒收(θ 的 D 块按五参定死,
    12 参 M8 走 [4:9] 切片会被静默截断——绝不带病截断,中文报错点名)。"""
    synth = make_stereo_synth(n_pairs=4, seed=7)
    pairs = _pairs_from_synth(synth)
    intr1, intr2 = _intr_views(synth.params1, synth.params2)
    init = pnp_median(pairs, intr1, intr2, None)
    intr1_m8 = dataclasses.replace(intr1, model="M8")
    with pytest.raises(ValueError, match="不支持"):
        refine_joint(pairs, intr1_m8, intr2, init,
                     _joint_cfg(), StereoRefineConfig())
    # cam2 侧同样拦(两路任一 M8 都不许进)
    intr2_m8 = dataclasses.replace(intr2, model="M8")
    with pytest.raises(ValueError, match="不支持"):
        refine_joint(pairs, intr1, intr2_m8, init,
                     _joint_cfg(), StereoRefineConfig())
