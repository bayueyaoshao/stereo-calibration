"""模块4 编排层 e2e 测试:渲染棋盘 → 既有检测链 → 双模式测量 → 闸门 → 落盘。

夹具策略(与 conftest 渲染夹具同源,物理方格恒 25mm 对齐 BoardSpec):
conftest 反推物理方格 = square_px × depth / focal,故各路 square_px 都按
25 × focal / depth 显式计算。几何取「收敛式双目」:两机各自正视棋盘
(检测最稳,入画完整),相对位姿自然带 ~19° 转角、基线 ~207mm、深度
~600mm——比正对平移构型(基线/深度比弱)冻结偏差小一个量级:干净渲染
实测冻结 ~0.10mm、逐对 ~0.01mm,远低于 1.0/2.0mm 红线。内参/外参契约
文件用真实写侧落盘(契约零漂移)。

覆盖:PASS 全契约(JSON 键集/verdict/区域桶/数据只读/D-11 清残留)、
FAIL verdict(存档 t 注入漂移)→ CLI 退出码 1 且产物先落盘、min_pairs
fail-closed、--help 暴露 validate。
"""

from __future__ import annotations

import json
import pathlib

import cv2
import numpy as np
import pytest

from stereo_calib.intrinsic.artifacts import write_intrinsics_json
from stereo_calib.io.config import BoardSpec, load_config
from stereo_calib.preprocess.artifacts import build_object_points
from stereo_calib.stereo.artifacts import write_extrinsics_stereo_json
from stereo_calib.validation.artifacts import _VALIDATION_JSON_KEYS
from stereo_calib.validation.pipeline import run_validation

# 全键显式 config.yaml(preprocess/intrinsic/stereo 与生产同构;validation
# 是被测段;红线 1.0/2.0 与生产一致)
_VAL_YAML = """\
board:
  board_type: "chessboard"
  cols: 11
  rows: 8
  square_size_mm: 25.0
  unit: "mm"
data:
  train: "data/train"
  test: "data/test"
output_root: "output"
cv2_threads: 0
preprocess:
  max_workers: 2
  corners:
    classic_flags: [adaptive_thresh, normalize_image]
    sb_flags: 0
    sb_fallback_classic: true
    subpix: {window_granularity: frame, min_win: 3, max_iter: 30, epsilon: 0.01}
  copy_fail_originals: true
  visualization: {jpeg_quality: 90}
intrinsic:
  refine:
    enabled: false
    method: "trf"
    max_nfev: 300
    ftol: 1.0e-10
    xtol: 1.0e-12
    gtol: 1.0e-10
  outlier:
    enabled: false
    rms_multiplier: 5.0
    max_drop_fraction: 0.2
    max_iterations: 3
    min_views: 10
  rms_test_max: 1.0
  exclude_images: []
stereo:
  refine:
    enabled: false
    method: "trf"
    max_nfev: 200
    ftol: 1.0e-10
    xtol: 1.0e-12
    gtol: 1.0e-10
  outlier:
    enabled: false
    rms_multiplier: 5.0
    max_drop_fraction: 0.2
    max_iterations: 3
    min_pairs: 2
  flip_ratio: 0.5
  median_spread_max_deg: 3.0
  distance_rms_max_mm: 5.0
  baseline_band_mm: [20, 2000]
  exclude_pairs: []
validation:
  data_dir: "data/val"
  distance_bias_max_mm: 1.0
  distance_rms_max_mm: 2.0
  region_grid: [3, 3]
  min_pairs: 1
"""


def _tree_snapshot(root: pathlib.Path) -> dict:
    """相对路径 → sha256 快照(数据树只读防线的自动化证据,03 体例)。"""
    import hashlib

    snap: dict[str, str] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            snap[p.relative_to(root).as_posix()] = hashlib.sha256(
                p.read_bytes()).hexdigest()
    return snap


def _val_e2e_env(tmp_path: pathlib.Path, make_board) -> pathlib.Path:
    """e2e 环境:1 组渲染验证对 + 契约内参/外参产物 + 全键 config.yaml。"""
    # 收敛式双目真值几何:两机各自正视棋盘(各自小滚转),cam2 绕 Y 轴
    # 偏航 ~19° 看向棋盘;深度 ~600mm、基线 ~207mm(强几何,渲染实测
    # 冻结偏差 ~0.1mm)
    rvec1 = np.array([0.0, 0.03, 0.0])
    tvec1 = np.array([30.0, -10.0, 600.0])
    r1, _ = cv2.Rodrigues(rvec1)
    ry, _ = cv2.Rodrigues(np.array([0.0, np.deg2rad(19.0), 0.0]))
    r2_small, _ = cv2.Rodrigues(np.array([0.0, -0.02, 0.0]))
    r2 = ry @ r2_small
    rvec2, _ = cv2.Rodrigues(r2)
    tvec2 = np.array([-10.0, 10.0, 580.0])

    # 两路内参同焦距(物理方格渲染对齐 25mm 的前提)、主点微调;零畸变
    params1 = np.array([800.0, 800.0, 320.0, 240.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    params2 = np.array([800.0, 800.0, 320.0, 242.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def _k_of(params: np.ndarray) -> np.ndarray:
        return np.array([[params[0], 0.0, params[2]],
                         [0.0, params[1], params[3]], [0.0, 0.0, 1.0]])

    # 物理方格恒 25mm:square_px = 25 × focal / depth(fixture 反推 world =
    # square_px × depth / focal)
    img1, _truth1, _sq1 = make_board(
        K=_k_of(params1), dist_coeffs=np.zeros(5), rvec=rvec1, tvec=tvec1,
        image_size=(480, 640), square_px=25.0 * 800.0 / 600.0)
    img2, _truth2, _sq2 = make_board(
        K=_k_of(params2), dist_coeffs=np.zeros(5),
        rvec=rvec2.astype(np.float64), tvec=tvec2,
        image_size=(480, 640), square_px=25.0 * 800.0 / 580.0)
    val_dir = tmp_path / "data" / "val" / "g1"
    val_dir.mkdir(parents=True)
    assert cv2.imwrite(str(val_dir / "cam1.jpg"), img1)
    assert cv2.imwrite(str(val_dir / "cam2.jpg"), img2)

    diag = {"enabled": True, "verdict": "可靠", "reliable": True,
            "red_count": 0, "red_names": [], "notes": [], "checks": []}
    for cam, params in ((1, params1), (2, params2)):
        write_intrinsics_json(
            tmp_path / "output" / "intrinsic" / f"cam{cam}" / "intrinsics.json",
            cam=cam, image_size=(640, 480), params=params,
            K=_k_of(params), D=params[4:9], solver="opencv",
            refine_enabled=False, rms_train_overall=0.1, rms_test_overall=0.1,
            n_views_train=1, n_views_test=1, dropped_frames=[],
            solver_meta={"status": 0}, board=BoardSpec(), config_snapshot={},
            diagnostics=diag)

    # 冻结外参 = 两机真值位姿组合(X_cam2 = R2 R1ᵀ X_cam1 + (t2 − R2 R1ᵀ t1))
    r_rel = r2 @ r1.T
    t_rel = tvec2 - r_rel @ tvec1
    rvec_rel, _ = cv2.Rodrigues(r_rel)
    objp = build_object_points(BoardSpec())
    write_extrinsics_stereo_json(
        tmp_path / "output" / "stereo" / "extrinsics_stereo.json",
        ref_cam=1, R=r_rel, t=t_rel, rvec=rvec_rel.ravel(),
        rotation_angle_deg=float(np.degrees(np.linalg.norm(rvec_rel))),
        baseline_mm=float(np.linalg.norm(t_rel)), K1=_k_of(params1),
        D1=params1[4:9], K2=_k_of(params2), D2=params2[4:9],
        n_pairs_train=1, n_pairs_test=1, rms_reproj_train=0.1,
        rms_reproj_test=0.1, distance_rms_train_mm=0.1,
        distance_rms_test_mm=0.1,
        true_distance_mm=float(np.linalg.norm(objp[-1] - objp[0])),
        dropped_pairs=[], flip_log={}, solver_meta={"status": 0},
        board=BoardSpec(), config_snapshot={})

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(_VAL_YAML, encoding="utf-8")
    return cfg_path


def test_run_validation_pass_full_contract(tmp_path, monkeypatch,
                                           make_chessboard_image):
    """干净渲染对 → run_validation 全契约:PASS、JSON 键集、逐对行、区域桶、
    report 节全、数据只读、新产物全在 output/validation 之下。"""
    cfg_path = _val_e2e_env(tmp_path, make_chessboard_image)
    before = _tree_snapshot(tmp_path)
    monkeypatch.chdir(tmp_path)
    rep = run_validation(load_config(cfg_path))
    after = _tree_snapshot(tmp_path)

    # 数据只读:既有文件逐字节不变,新增文件全在 output/validation 之下
    for key, digest in before.items():
        assert after.get(key) == digest, f"既有文件被改动:{key}"
    new_keys = set(after) - set(before)
    assert new_keys, "run_validation 没有写任何产物"
    assert all(k.startswith("output/validation/") for k in new_keys), (
        f"产物写到约定目录之外:"
        f"{sorted(k for k in new_keys if not k.startswith('output/validation/'))}")

    assert set(rep) == set(_VALIDATION_JSON_KEYS), "顶层字段集漂移"
    assert rep["verdict"] == "PASS"
    assert rep["n_pairs_valid"] == 1 and rep["n_groups_found"] == 1
    assert rep["n_pairs_failed"] == 0 and rep["pair_fail_notes"] == {}
    assert rep["gates"]["bias_pass"] and rep["gates"]["rms_pass"]
    assert rep["gates"]["n_pairs_used"] == 1
    assert abs(rep["true_distance_mm"] - 305.16389039334257) < 1e-3
    # 干净渲染实测冻结 ~0.10mm / 逐对 ~0.01mm,须远低于 1.0mm 红线
    row = rep["pairs"][0]
    assert row["group_id"] == "g1" and row["note"] == ""
    assert abs(row["err_frozen_mm"]) < 0.5, f"冻结偏差 {row['err_frozen_mm']!r}"
    assert abs(row["err_pair_mm"]) < 0.5, f"逐对偏差 {row['err_pair_mm']!r}"

    payload = json.loads((tmp_path / "output/validation/validation.json")
                         .read_text(encoding="utf-8"))
    assert set(payload) == set(_VALIDATION_JSON_KEYS)
    assert payload["verdict"] == "PASS"
    assert len(payload["regions"]) == 9  # 3×3 网格全桶如实罗列
    assert sum(r["n"] for r in payload["regions"]) == 1
    # 板中心 ≈ (360, 227) → 行列桶 r1c1(行主序下标 4)
    assert payload["regions"][4]["n"] == 1, (
        f"板中心应落 r1c1 桶,实际分桶:{[r['n'] for r in payload['regions']]}")
    assert payload["fingerprint"]["baseline_mm"] == pytest.approx(
        float(np.linalg.norm(_frozen_t_of(tmp_path))), rel=1e-6)

    report = (tmp_path / "output/validation/report.md").read_text(encoding="utf-8")
    for token in ("verdict:PASS", "判读矩阵", "逐对结果", "区域精度分布",
                  "参数指纹", "能力边界", "配置快照", "样本不足"):
        assert token in report, f"report.md 缺少「{token}」"


def _frozen_t_of(tmp_path: pathlib.Path) -> np.ndarray:
    """读回夹具写的存档 t(基线指纹断言用)。"""
    payload = json.loads((tmp_path / "output" / "stereo"
                          / "extrinsics_stereo.json").read_text(encoding="utf-8"))
    return np.asarray(payload["t"], dtype=np.float64)


def test_run_validation_d11_cleans_stale_artifacts(tmp_path, monkeypatch,
                                                   make_chessboard_image):
    """重跑清残留(D-11):陈旧文件不跨运行存活(照 stereo 体例)。"""
    cfg_path = _val_e2e_env(tmp_path, make_chessboard_image)
    monkeypatch.chdir(tmp_path)
    stale = tmp_path / "output" / "validation" / "stale_from_last_run.txt"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text("陈旧残留", encoding="utf-8")
    run_validation(load_config(cfg_path))
    assert not stale.exists(), "重跑必须清掉上次残留"


def test_cli_validate_exit_codes(tmp_path, monkeypatch,
                                 make_chessboard_image):
    """CLI 退出码契约:PASS → 0;存档 t 注入漂移 → FAIL 且退出码 1
    (fail-closed),且 validation.json/report.md 在退出前已落盘留证;
    min_pairs 不足 → 退出码 1 中文收口。"""
    from typer.testing import CliRunner

    from stereo_calib.cli.main import app

    monkeypatch.chdir(tmp_path)
    cfg_path = _val_e2e_env(tmp_path, make_chessboard_image)

    result = CliRunner().invoke(app, ["validate"])
    assert result.exit_code == 0, result.output
    assert "PASS" in result.output

    # 存档外参注入 +40mm 漂移(baseline_mm 同步改,保装载一致性核对通过)
    ext_path = tmp_path / "output" / "stereo" / "extrinsics_stereo.json"
    payload = json.loads(ext_path.read_text(encoding="utf-8"))
    payload["t"] = [v + 40.0 for v in payload["t"]]
    payload["baseline_mm"] = float(np.linalg.norm(
        np.asarray(payload["t"], dtype=np.float64)))
    ext_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    result = CliRunner().invoke(app, ["validate"])
    assert result.exit_code == 1, result.output
    assert "FAIL" in result.output
    val_json = json.loads((tmp_path / "output/validation/validation.json")
                          .read_text(encoding="utf-8"))
    assert val_json["verdict"] == "FAIL", "FAIL 证据必须先落盘再退 1"
    report = (tmp_path / "output/validation/report.md").read_text(encoding="utf-8")
    assert "verdict:FAIL" in report

    # min_pairs 不足:验证目录只有 1 组 → fail-closed 中文一句 + 退出码 1
    #(\n + 两空格定位 validation 段的 min_pairs,不误伤 stereo.outlier 的)
    cfg_path.write_text(
        _VAL_YAML.replace("\n  min_pairs: 1", "\n  min_pairs: 5"),
        encoding="utf-8")
    result = CliRunner().invoke(app, ["validate"])
    assert result.exit_code == 1
    assert "有效验证对数不足" in result.output


def test_cli_help_exposes_validate():
    """--help 必须暴露 validate(模块4 入口在位)。"""
    from typer.testing import CliRunner

    from stereo_calib.cli.main import app

    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "validate" in result.output
