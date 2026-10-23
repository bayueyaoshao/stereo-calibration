"""stereo 评估链测试:DLT 往返/保号除法/undist P=K 契约/消歧翻转/位姿 BA/失败契约。

体例照 tests/test_intrinsic_scoring.py(合成真值 + 失败不抛)+
tests/test_intrinsic_models.py(数值锁定层)。全部用例吃 conftest 的
make_stereo_synth 真值双目几何(强畸变域,零渲染零读图)。

注意(源码防线自伤规避):被计数的调用串原文只出现在本测试文件,
evaluate.py 源文侧的措辞刻意避开(防线锁定的是 src/,本文件不在其列)。
"""

import pathlib

import cv2
import numpy as np
import pytest

from conftest import SynthStereo, make_stereo_synth

from stereo_calib.intrinsic import models
from stereo_calib.io.config import BoardSpec
from stereo_calib.preprocess.artifacts import build_object_points
from stereo_calib.stereo.evaluate import (
    disambiguate_frame,
    evaluate_frame,
    evaluate_split,
    farthest_corners_distance,
    frame_pose_ba,
    projection_matrix,
    triangulate_dlt,
    true_board_distance,
    undist_pix,
)
from stereo_calib.stereo.observation import IntrinsicsView, Pairs

_EVAL_SOURCE = pathlib.Path("src/stereo_calib/stereo/evaluate.py")


def _pairs_from_synth(synth: SynthStereo) -> Pairs:
    """合成观测 → Pairs(行序契约与夹具 group_ids 一致)。"""
    n = len(synth.group_ids)
    return Pairs(image_ids=synth.group_ids,
                 source1=("sb",) * n, source2=("sb",) * n,
                 routes=(("sb", "sb"),) * n,
                 uv1=synth.uv1, uv2=synth.uv2, objp=synth.objp)


def _intr_views(synth: SynthStereo) -> tuple[IntrinsicsView, IntrinsicsView]:
    """合成内参 → IntrinsicsView 两路(与夹具 K/D/params 同源)。"""
    return (IntrinsicsView(cam=1, image_size=(2560, 1440),
                           params=synth.params1, K=synth.K1, D=synth.D1),
            IntrinsicsView(cam=2, image_size=(2560, 1440),
                           params=synth.params2, K=synth.K2, D=synth.D2))


def _pinhole_project(x_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    """针孔投影(测试专用,无畸变):DLT 侧的合成观测口径。"""
    return (x_cam[:, :2] / x_cam[:, 2:3]) @ K[:2, :2].T + K[:2, 2]


# ---------- 合成板全链恢复(STER-03 的自动化证据) ----------


def test_evaluate_split_recovers_synthetic_board():
    """合成真值 6 对 + 已知外参 → 逐对 |dist_err_mm| < 0.5、pooled < 0.5mm;
    真值 = norm(objp[87]-objp[0]) = 305.1639(从 build_object_points 导出)。"""
    synth = make_stereo_synth(n_pairs=6, seed=7)
    report = evaluate_split(_pairs_from_synth(synth), *_intr_views(synth),
                            synth.R_rel_true, synth.t_rel_true,
                            BoardSpec(), flip_ratio=0.5)
    assert len(report.evals) == 6
    true_d = float(np.linalg.norm(synth.objp[87] - synth.objp[0]))
    assert report.true_distance_mm == pytest.approx(true_d, abs=1e-9)
    assert report.true_distance_mm == pytest.approx(305.16389039334257, abs=1e-6)
    for ev in report.evals:
        assert abs(ev.dist_err_mm) < 0.5, f"对角距离恢复失败:{ev.group_id} {ev.dist_err_mm:.4f}mm"
        assert ev.note == "", f"干净合成帧不该有失败备注:{ev.note}"
        assert ev.flipped == (False, False)
        assert ev.reproj_rms < 0.5
        assert ev.dist_measured_mm == pytest.approx(
            true_d + ev.dist_err_mm, abs=1e-9)
    assert report.distance_rms_mm < 0.5


def test_true_board_distance_derived_from_objp_not_hardcoded():
    """真值必须从 objp 导出:换棋盘规格(9×7/20mm)真值自动跟随,绝不写死数字。"""
    objp_small = build_object_points(BoardSpec(cols=9, rows=7, square_size_mm=20.0))
    expected = float(np.linalg.norm(objp_small[-1] - objp_small[0]))
    assert true_board_distance(objp_small) == pytest.approx(expected)
    assert expected == pytest.approx(np.sqrt(8.0 ** 2 + 6.0 ** 2) * 20.0)
    assert farthest_corners_distance(objp_small, objp_small)[0] == pytest.approx(expected)


# ---------- DLT 保号除法(Pitfall 5:负齐次分量不整点取反) ----------


def test_triangulate_dlt_signed_homogeneous_division():
    """把两份投影矩阵整体取负:SVD 零空间向量随之变号(X_h[3] < 0)→
    三角化结果必须与正号完全一致且深度为正——按绝对值除齐次分量会在
    负号情形把整点取反(参考工程注释记录的坑),本用例当场抓住。"""
    synth = make_stereo_synth(n_pairs=1, seed=7)
    x1, _ = models.transform_to_camera(synth.objp, synth.rvecs1[0], synth.tvecs1[0])
    uv1 = _pinhole_project(x1, synth.K1)
    x2 = x1 @ synth.R_rel_true.T + synth.t_rel_true
    uv2 = _pinhole_project(x2, synth.K2)
    p1 = projection_matrix(synth.K1, np.eye(3), np.zeros(3))
    p2 = projection_matrix(synth.K2, synth.R_rel_true, synth.t_rel_true)

    pts_pos = triangulate_dlt([p1, p2], [uv1, uv2])
    assert pts_pos.shape == (88, 3)
    assert (pts_pos[:, 2] > 0).all(), "三角化点深度必须为正(板在相机前方)"
    assert np.abs(pts_pos - x1).max() < 1e-6, "针孔像素 + 无畸变 P 的 DLT 应精确恢复"

    pts_neg = triangulate_dlt([-p1, -p2], [uv1, uv2])
    assert (pts_neg[:, 2] > 0).all(), "负齐次分量情形绝不允许整点取反(深度变负)"
    assert np.abs(pts_neg - pts_pos).max() < 1e-9, "保号除法:投影矩阵整体变号不改结果"


# ---------- undistortPoints 的 P=K 像素口径(Pitfall 4) ----------


def test_undist_pix_pixel_frame_contract():
    """undist_pix 必须返回针孔【像素】坐标(第 5 参传 K),不是归一化坐标:
    与 K @ undistortPoints(..., None) 的解析关系一致到浮点噪声级,cv2 内部
    float32 装包决定逐点差在 1e-4 px 量级(锁 1e-3 px);归一化泄漏(值
    掉到 1e-3 量级)当场抓住。"""
    synth = make_stereo_synth(n_pairs=1, seed=7)
    pts = synth.uv1[0]
    got = undist_pix(pts, synth.K1, synth.D1)
    assert got.shape == (88, 2)
    norm = cv2.undistortPoints(
        pts.astype(np.float32).reshape(-1, 1, 2),
        synth.K1, synth.D1.reshape(-1, 1), None, None).reshape(-1, 2)
    expected = (np.column_stack([norm, np.ones(len(norm))]) @ synth.K1.T)[:, :2]
    assert np.abs(got.astype(np.float64) - expected).max() < 1e-3, (
        "undist_pix 与「归一化坐标再乘 K」的像素口径关系被破坏")
    assert np.abs(got).max() > 100.0, "返回值掉到归一化量级 = P=K 口径丢失(Pitfall 4)"


# ---------- 180° 消歧(锚 = 当帧 cam1 PnP) ----------


def test_disambiguate_frame_single_route_flip_detected_and_restored():
    """仅 cam2 角点序翻转 → 检出并翻回(flags=(False, True)),cam1 原样;
    输入数组绝不被原地修改。"""
    synth = make_stereo_synth(n_pairs=2, seed=7)
    intr1, intr2 = _intr_views(synth)
    uv1 = synth.uv1[0]
    uv2 = synth.uv2[0]
    uv2_input = uv2[::-1].copy()
    out1, out2, flags = disambiguate_frame(
        uv1, uv2_input, synth.objp, intr1, intr2,
        synth.R_rel_true, synth.t_rel_true, 0.5)
    assert flags == (False, True), "单路翻转必须被检出(证据充分才翻)"
    assert np.array_equal(out1, uv1), "cam1 未翻转、必须原样返回"
    assert np.allclose(out2, uv2), "cam2 必须翻回物理一致序"
    assert np.array_equal(uv2_input, uv2[::-1]), "消歧绝不原地改输入"


def test_disambiguate_frame_full_flip_metric_invariant():
    """uv1+uv2 双路整体翻转 → 锚(cam1 PnP)解出 180° 转板位姿、与翻转序
    自洽,消歧不再翻(锚自洽性,研究文档 Pattern 3);评估链对整体翻转
    不变——距离残差与原帧一致(自洽性论证的自动化)。"""
    synth = make_stereo_synth(n_pairs=2, seed=7)
    intr1, intr2 = _intr_views(synth)
    uv1 = synth.uv1[0]
    uv2 = synth.uv2[0]
    ev_orig = evaluate_frame("orig", uv1, uv2, synth.objp, intr1, intr2,
                             synth.R_rel_true, synth.t_rel_true, 0.5)
    ev_flip = evaluate_frame("flip", uv1[::-1].copy(), uv2[::-1].copy(),
                             synth.objp, intr1, intr2,
                             synth.R_rel_true, synth.t_rel_true, 0.5)
    assert np.isfinite(ev_orig.dist_err_mm) and np.isfinite(ev_flip.dist_err_mm)
    assert abs(ev_flip.dist_err_mm - ev_orig.dist_err_mm) < 0.1, (
        "整体翻转帧的板系对角距离必须不变(角点 0↔87 互换,距离相同)")
    assert ev_flip.note == "" and ev_orig.note == ""


# ---------- per-frame 2 视图位姿 BA(v2 的必须半步) ----------


def test_frame_pose_ba_recovers_known_pose_strong_tilt():
    """强倾角已知位姿帧 → BA 恢复位姿 pooled RMS < 0.5px,与真值位姿的
    投影一致性 < 0.2px(unproject 反投影初值路线,不进错误盆地)。"""
    synth = make_stereo_synth(n_pairs=1, seed=7)
    intr1, intr2 = _intr_views(synth)
    objp = synth.objp
    rvec_true = np.array([0.45, -0.35, 0.25])  # 强倾角(约 36°)
    tvec_true = np.array([30.0, -20.0, 600.0])
    x1_true, _ = models.transform_to_camera(objp, rvec_true, tvec_true)
    uv1 = _pinhole_project(x1_true, synth.K1) + np.random.default_rng(3).normal(
        0.0, 0.02, (88, 2))
    x2_true = x1_true @ synth.R_rel_true.T + synth.t_rel_true
    uv2 = _pinhole_project(x2_true, synth.K2) + np.random.default_rng(4).normal(
        0.0, 0.02, (88, 2))

    rvec, tvec, rms = frame_pose_ba(uv1, uv2, objp, intr1, intr2,
                                    synth.R_rel_true, synth.t_rel_true)
    assert rms < 0.5, f"位姿 BA 残差应亚像素,实测 {rms:.4f}px"
    x1_rec, _ = models.transform_to_camera(objp, rvec, tvec)
    proj_true = models.project(x1_true, synth.params1)
    proj_rec = models.project(x1_rec, synth.params1)
    pose_rms = float(np.sqrt(np.mean(np.sum((proj_rec - proj_true) ** 2, axis=1))))
    assert pose_rms < 0.2, f"恢复位姿与真值位姿的投影一致性应优于 0.2px,实测 {pose_rms:.4f}px"
    assert np.allclose(tvec, tvec_true, atol=2.0), "平移恢复应在 mm 级"


# ---------- 失败契约(单帧 NaN + 中文备注,绝不中断整批) ----------


def test_evaluate_frame_failure_contract_no_raise():
    """含 NaN 观测 / 退化观测(全部角点重合)→ 该帧 dist_err_mm=NaN +
    note 非空中文,绝不 raise 中断整批(scoring _failure 体例)。"""
    synth = make_stereo_synth(n_pairs=1, seed=7)
    intr1, intr2 = _intr_views(synth)

    uv1_nan = synth.uv1[0].copy()
    uv1_nan[5, 0] = np.nan
    ev_nan = evaluate_frame("nan帧", uv1_nan, synth.uv2[0], synth.objp,
                            intr1, intr2, synth.R_rel_true, synth.t_rel_true, 0.5)
    assert np.isnan(ev_nan.dist_err_mm) and np.isnan(ev_nan.reproj_rms)
    assert ev_nan.note, "失败帧必须有非空中文备注"

    uv_deg = np.tile(synth.uv1[0][0], (88, 1))
    ev_deg = evaluate_frame("退化帧", uv_deg, synth.uv2[0], synth.objp,
                            intr1, intr2, synth.R_rel_true, synth.t_rel_true, 0.5)
    assert np.isnan(ev_deg.dist_err_mm) and ev_deg.note


def test_evaluate_split_all_frames_failed_raises():
    """全帧失败 → 中文 ValueError(考卷全糊属环境异常,不静默产出 NaN 报告)。"""
    synth = make_stereo_synth(n_pairs=2, seed=7)
    pairs = _pairs_from_synth(synth)
    broken = Pairs(image_ids=pairs.image_ids, source1=pairs.source1,
                   source2=pairs.source2, routes=pairs.routes,
                   uv1=np.full_like(pairs.uv1, np.nan), uv2=pairs.uv2,
                   objp=pairs.objp)
    with pytest.raises(ValueError, match="全部"):
        evaluate_split(broken, *_intr_views(synth), synth.R_rel_true,
                       synth.t_rel_true, BoardSpec(), flip_ratio=0.5)


# ---------- 源码防线(三角化唯一住址 / 单一数学源 / 保号纪律) ----------


def test_evaluate_source_guards():
    """evaluate.py 源文防线:无第二份畸变公式、无 abs 出现(保号除法纪律)、
    SVD 三角化只住本文件、无分轴对齐诊断(决策 6)、关键调用在位。"""
    source = _EVAL_SOURCE.read_text(encoding="utf-8")
    assert source.count("_distort") == 0, "畸变公式只准经 intrinsic.models(单一数学源)"
    assert source.count("abs(") == 0, "保号除法纪律:齐次分量除法严禁绝对值(Pitfall 5)"
    assert "np.linalg.svd" in source, "DLT 的 SVD 必须在本文件(三角化唯一住址)"
    assert "undistortPoints" in source and '"trf"' in source
    assert "kabsch" not in source.lower(), "分轴对齐诊断不移植(决策 6:MVP 指标 = 对角距离残差)"
    stereo_dir = pathlib.Path("src/stereo_calib/stereo")
    for path in stereo_dir.glob("*.py"):
        if path.name != "evaluate.py":
            assert "np.linalg.svd" not in path.read_text(encoding="utf-8"), (
                f"{path.name} 出现 SVD 三角化:三角化只准住 evaluate.py(源码防线)")
