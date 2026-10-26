"""模块4 测量核心单元测试:合成场景数学正确性 + 漂移分离性 + 闸门/区域/装载守卫。

测试分层(零渲染、零读图,纯 numpy/cv2 投影合成,秒级):
- M8 合成零误差:已知 K/D8/R/t 真值投影角点 → 冻结/逐对双模式测得对角
  距离与理论值之差 < 1e-6 mm(全链数学 import 复用的正确性证据);
- 180° 翻转注入:uv2 反序进 measure_pair,消歧(与训练同一 disambiguate_frame)
  应修复,偏差仍应 ≈ 0;
- 漂移分离性(判读矩阵的数学根据):存档 t 注入 +1.5mm 漂移 → 冻结偏差
  显著超 bias 红线(≥1mm),逐对偏差仍 ≈ 0(逐对模式吸收外参漂移);
- evaluate_gates / analyze_regions / load_frozen_extrinsics 的契约与守卫;
- 真实数据冒烟(有 output/ 产物才跑,CI 缺产物自动跳过):真实冻结参数 +
  真实 train 角点,冻结偏差必须落红线内。
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from stereo_calib.io.config import BoardSpec
from stereo_calib.preprocess.artifacts import build_object_points
from stereo_calib.stereo.observation import IntrinsicsView
from stereo_calib.validation.measure import (
    FrozenExtrinsics,
    PairMeasurement,
    analyze_regions,
    evaluate_gates,
    load_frozen_extrinsics,
    measure_pair,
)

_FLIP_RATIO = 0.5  # 与生产 config.yaml 的 stereo.flip_ratio 同值


def _m8_scene():
    """已知真值的 M8 双目合成场景:返回 (objp, intr1, intr2, ext, uv1, uv2)。

    内参取 M8 十二参布局(与生产 M8 同构:fx/fy/cx/cy/k1..k6),畸变量级
    与 make_stereo_synth 的强畸变域同源;外参 R_rel/t_rel 与 03 系列夹具
    同量级(~120mm baseline、~2° 夹角)。观测由 cv2.projectPoints 从真值
    位姿零噪声合成——任何非零偏差都只能来自被测代码本身。
    """
    params1 = np.array([800.0, 796.0, 320.0, 240.0,
                        -0.30, 0.15, 1.0e-4, -5.0e-5, -0.05, 0.10, -0.05, 0.01])
    params2 = np.array([802.0, 798.0, 318.0, 242.0,
                        -0.30, 0.15, 1.0e-4, -5.0e-5, -0.05, 0.10, -0.05, 0.01])

    def _view(cam: int, params: np.ndarray) -> IntrinsicsView:
        k = np.array([[params[0], 0.0, params[2]],
                      [0.0, params[1], params[3]], [0.0, 0.0, 1.0]])
        return IntrinsicsView(cam=cam, image_size=(640, 480), params=params,
                              K=k, D=params[4:12].copy(), model="M8")

    intr1 = _view(1, params1)
    intr2 = _view(2, params2)

    objp = build_object_points(BoardSpec())  # 11×8 / 25mm,对角真值随之确定
    rvec1 = np.array([0.10, -0.05, 0.03])
    tvec1 = np.array([20.0, -15.0, 700.0])
    axis = np.array([0.3, -0.5, 0.8])
    axis = axis / np.linalg.norm(axis)
    r_rel, _ = cv2.Rodrigues(axis * np.deg2rad(2.0))
    t_rel = np.array([120.0, 15.0, 8.0])
    r1, _ = cv2.Rodrigues(rvec1)
    r2 = r_rel @ r1
    tvec2 = r_rel @ tvec1 + t_rel
    rvec2, _ = cv2.Rodrigues(r2)

    # 投影用 float64 物点(不照抄 conftest 的 float32 惯例):零噪声场景的
    # 误差地板必须只由被测数学决定,float32 量化会带来 ~1e-4 mm 的假地板。
    uv1, _ = cv2.projectPoints(objp, rvec1, tvec1, intr1.K,
                               intr1.D.reshape(-1, 1))
    uv2, _ = cv2.projectPoints(objp, rvec2.astype(np.float64), tvec2,
                               intr2.K, intr2.D.reshape(-1, 1))
    ext = FrozenExtrinsics(R=r_rel, t=t_rel, rvec=axis * np.deg2rad(2.0),
                           baseline_mm=float(np.linalg.norm(t_rel)))
    return objp, intr1, intr2, ext, uv1.reshape(-1, 2), uv2.reshape(-1, 2)


def _pm(**overrides) -> PairMeasurement:
    """造 PairMeasurement 的便捷工厂(区域统计单元测试用)。"""
    base = {"group_id": "g", "err_frozen_mm": 0.1, "err_pair_mm": 0.1,
            "measured_frozen_mm": 646.0, "measured_pair_mm": 646.0,
            "center_u_px": 50.0, "center_v_px": 50.0, "reproj_rms_px": 0.1,
            "flipped": (False, False), "note": ""}
    base.update(overrides)
    return PairMeasurement(**base)


def test_measure_pair_m8_synthetic_zero_error_both_modes():
    """零噪声 M8 合成:冻结模式偏差 < 1e-6 mm;逐对模式地板由生产代码
    _pnp_pose 的 float32 入参决定(< 1e-3 mm,量级仍与漂移信号差 5 个数量级)。"""
    objp, intr1, intr2, ext, uv1, uv2 = _m8_scene()
    pm = measure_pair("synth", uv1, uv2, objp, intr1, intr2, ext, _FLIP_RATIO)
    assert pm.note == "", f"零噪声合成不允许测量链失败:{pm.note}"
    assert abs(pm.err_frozen_mm) < 1.0e-6, f"冻结模式偏差 {pm.err_frozen_mm!r}"
    # 逐对模式的地板在生产代码 _pnp_pose 的 float32 入参转换(solvePnP 的
    # 合法 dtype 组合,与 evaluate 消歧锚同口径):像素值 float32 量化
    # ≈3e-5px → 距离地板 ~4e-5 mm,阈值给到 1e-3 留足量级余裕。
    assert abs(pm.err_pair_mm) < 1.0e-3, f"逐对模式偏差 {pm.err_pair_mm!r}"
    assert pm.flipped == (False, False), "正序输入不允许消歧翻转"


def test_measure_pair_disambiguates_flipped_input():
    """uv2 注入 180° 反序:消歧(训练同一 disambiguate_frame)应修复并留痕。"""
    objp, intr1, intr2, ext, uv1, uv2 = _m8_scene()
    pm = measure_pair("synth", uv1, uv2[::-1].copy(), objp, intr1, intr2, ext,
                      _FLIP_RATIO)
    assert pm.note == "", f"消歧修复路径不允许失败:{pm.note}"
    assert pm.flipped == (False, True), "反序输入必须被消歧翻转并留痕"
    assert abs(pm.err_frozen_mm) < 1.0e-6
    assert abs(pm.err_pair_mm) < 1.0e-3  # 地板 = _pnp_pose 的 float32 入参(见上)


def test_measure_pair_drift_separation_frozen_vs_pair():
    """漂移分离性(判读矩阵的数学根据):存档 t 注入 +1.5mm 漂移后,
    冻结偏差必须显著超 bias 红线(≥1mm),逐对偏差仍 ≈ 0——「冻结大 +
    逐对小 = 外参漂移」这一归因才成立。"""
    objp, intr1, intr2, ext, uv1, uv2 = _m8_scene()
    t_drift = ext.t + np.array([1.5, 0.0, 0.0])
    ext_drift = FrozenExtrinsics(R=ext.R, t=t_drift,
                                 rvec=ext.rvec,
                                 baseline_mm=float(np.linalg.norm(t_drift)))
    pm = measure_pair("synth", uv1, uv2, objp, intr1, intr2, ext_drift,
                      _FLIP_RATIO)
    assert pm.note == "", f"注入漂移不允许测量链失败:{pm.note}"
    assert abs(pm.err_frozen_mm) >= 1.0, (
        f"+1.5mm 外参漂移必须把冻结偏差顶过 1mm 红线,实测 {pm.err_frozen_mm!r}")
    assert abs(pm.err_pair_mm) < 1.0e-3, (
        f"逐对模式必须吸收外参漂移(地板 = float32 PnP 入参,见上),"
        f"实测 {pm.err_pair_mm!r}")


def test_measure_pair_contract_guards():
    """入口防线:形状不合/角点数不足在进数值链之前中文 AssertionError 拦下。"""
    objp, intr1, intr2, ext, uv1, uv2 = _m8_scene()
    with pytest.raises(AssertionError, match="观测形状不合"):
        measure_pair("bad", uv1.reshape(-1), uv2, objp, intr1, intr2, ext,
                     _FLIP_RATIO)
    with pytest.raises(AssertionError, match="少于求解下限"):
        measure_pair("bad", uv1[:3], uv2[:3], objp[:3], intr1, intr2, ext,
                     _FLIP_RATIO)


def test_evaluate_gates_boundaries_and_nan_filter():
    gates = evaluate_gates([0.5, -0.5, 0.2], 1.0, 2.0)
    assert gates["n_pairs_used"] == 3
    assert gates["bias_pass"] and gates["rms_pass"]

    gates = evaluate_gates([1.5], 1.0, 2.0)  # bias 超红线、rms 未超
    assert not gates["bias_pass"] and gates["rms_pass"]

    gates = evaluate_gates([float("nan"), 0.5], 1.0, 2.0)  # NaN 如实剔除
    assert gates["n_pairs_used"] == 1

    with pytest.raises(ValueError, match="判定闸门无输入"):
        evaluate_gates([float("nan")], 1.0, 2.0)


def test_analyze_regions_buckets_and_empty_cells():
    pairs = [
        _pm(group_id="a", err_frozen_mm=0.2, center_u_px=25.0, center_v_px=25.0),
        _pm(group_id="b", err_frozen_mm=-0.6, center_u_px=75.0, center_v_px=25.0),
        _pm(group_id="c", err_frozen_mm=0.4, center_u_px=25.0, center_v_px=75.0),
        _pm(group_id="nan", err_frozen_mm=float("nan"),
            center_u_px=float("nan")),  # 失败对不进任何桶
    ]
    regions = analyze_regions(pairs, (2, 2), (100, 100))
    assert [r["region"] for r in regions] == ["r0c0", "r0c1", "r1c0", "r1c1"]
    assert regions[0]["n"] == 1 and regions[0]["median_abs_err_mm"] == pytest.approx(0.2)
    assert regions[1]["max_abs_err_mm"] == pytest.approx(0.6)
    assert regions[3] == {"region": "r1c1", "n": 0, "median_abs_err_mm": None,
                          "max_abs_err_mm": None, "note": "无样本"}


def test_analyze_regions_disabled_grid_returns_empty():
    assert analyze_regions([_pm()], (0, 0), (100, 100)) == []


def test_load_frozen_extrinsics_contract(tmp_path: Path):
    """读侧契约:正常读回 + 缺字段/形状/baseline 漂移/顶层非对象全部中文报错。"""
    rvec = np.array([0.01, 0.02, -0.03])
    t = np.array([120.0, 15.0, 8.0])
    good = tmp_path / "good.json"
    good.write_text(json.dumps({
        "rvec": rvec.tolist(), "t": t.tolist(),
        "baseline_mm": float(np.linalg.norm(t))}), encoding="utf-8")
    ext = load_frozen_extrinsics(good)
    r_expect, _ = cv2.Rodrigues(rvec)
    assert np.allclose(ext.R, r_expect) and np.allclose(ext.t, t)
    assert ext.baseline_mm == pytest.approx(float(np.linalg.norm(t)))

    bad = tmp_path / "missing.json"
    bad.write_text(json.dumps({"t": t.tolist()}), encoding="utf-8")
    with pytest.raises(ValueError, match="缺字段"):
        load_frozen_extrinsics(bad)

    bad = tmp_path / "shape.json"
    bad.write_text(json.dumps({"rvec": [0.0, 0.0], "t": t.tolist()}),
                   encoding="utf-8")
    with pytest.raises(ValueError, match="3 长"):
        load_frozen_extrinsics(bad)

    bad = tmp_path / "baseline.json"
    bad.write_text(json.dumps({"rvec": rvec.tolist(), "t": t.tolist(),
                               "baseline_mm": 1.0}), encoding="utf-8")
    with pytest.raises(ValueError, match="不一致"):
        load_frozen_extrinsics(bad)

    bad = tmp_path / "toplevel.json"
    bad.write_text(json.dumps([1, 2]), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON 对象"):
        load_frozen_extrinsics(bad)


# ---------------------------------------------------------------------------
# 真实数据冒烟:冻结参数 + 真实 train 角点(零检测、零拟合,只读 npz)
# ---------------------------------------------------------------------------

def _real_artifacts_ready(repo_root: Path) -> bool:
    return (repo_root / "output/intrinsic/cam1/intrinsics.json").is_file() \
        and (repo_root / "output/stereo/extrinsics_stereo.json").is_file() \
        and (repo_root / "output/train/cam1_train.npz").is_file()


def test_real_frozen_artifacts_measure_within_gate():
    """真实冻结产物 + 真实 train 角点:冻结偏差必须落 bias 红线内
    (v1.0 定稿参数的健康证据;产物缺失则跳过,不伪造环境)。"""
    repo_root = Path(__file__).resolve().parents[1]
    if not _real_artifacts_ready(repo_root):
        pytest.skip("output/ 冻结产物不在位(干净检出环境),跳过真实数据冒烟")

    from stereo_calib.stereo.observation import load_intrinsics

    intr1 = load_intrinsics(repo_root / "output/intrinsic/cam1/intrinsics.json")
    intr2 = load_intrinsics(repo_root / "output/intrinsic/cam2/intrinsics.json")
    ext = load_frozen_extrinsics(
        repo_root / "output/stereo/extrinsics_stereo.json")
    with np.load(repo_root / "output/train/cam1_train.npz",
                 allow_pickle=False) as d1, \
         np.load(repo_root / "output/train/cam2_train.npz",
                 allow_pickle=False) as d2:
        both = d1["detected"] & d2["detected"]
        hit = int(np.flatnonzero(both)[0])
        uv1 = d1["uv"][hit]
        uv2 = d2["uv"][hit]
        objp = d1["objp"]
    pm = measure_pair("real", uv1, uv2, objp, intr1, intr2, ext, _FLIP_RATIO)
    assert pm.note == "", f"真实数据测量链不允许失败:{pm.note}"
    assert abs(pm.err_frozen_mm) <= 1.0, (
        f"真实冻结偏差 {pm.err_frozen_mm!r} mm 超 bias 红线,v1.0 参数健康证据被打破")
    assert abs(pm.err_pair_mm) <= 1.0
