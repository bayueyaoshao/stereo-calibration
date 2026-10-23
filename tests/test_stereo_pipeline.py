"""stereo 装载契约测试:合成 npz/pairs 对装载三重防线 + 内参按名读取。

03-01 只放装载契约与 load_intrinsics 用例;03-03 扩展端到端(合成 npz →
run_stereo → 产物契约)。合成路线:conftest 的 make_stereo_synth 真值双目
几何 + write_stereo_synth_files 落盘(零渲染、零读图,秒级)——连接键与
行序契约在合成数据上锁死,真实产物只做冒烟。

注意(源码防线自伤规避体例):被计数的调用串原文只出现在断言处,
observation.py 的注释措辞刻意避开。
"""

import csv
import json
import pathlib

import cv2
import numpy as np
import pytest

# pytest rootdir 模式(tests/ 无 __init__.py)把 tests 目录插进 sys.path
from conftest import make_stereo_synth, write_stereo_synth_files

from stereo_calib.io.config import BoardSpec
from stereo_calib.preprocess.artifacts import build_object_points
from stereo_calib.stereo.observation import (
    IntrinsicsView,
    load_intrinsics,
    load_pairs,
    select_pairs,
)

_CFG = BoardSpec()


def _build_tree(tmp_path: pathlib.Path, n_pairs: int = 8, split: str = "train"):
    """合成一份 train 树:两路 npz + pairs CSV(含 detected=False 额外组)。"""
    synth = make_stereo_synth(n_pairs=n_pairs, seed=7)
    csv_path = write_stereo_synth_files(tmp_path, synth, split)
    return synth, csv_path, tmp_path / "output" / split


# ---------- 装载契约:计数 / 行序 / 形状 / 未检出组过滤 ----------


def test_load_pairs_contract_counts_order_shapes(tmp_path):
    """对清单行序 = 幸存 image_ids 顺序(行序锁);额外未检出组不进装载。"""
    synth, csv_path, split_dir = _build_tree(tmp_path, n_pairs=8)
    p = load_pairs(_CFG, csv_path,
                   split_dir / "cam1_train.npz", split_dir / "cam2_train.npz",
                   min_pairs=2)
    assert p.image_ids == synth.group_ids, "幸存对必须按 pairs CSV 行序一一对应(行序锁)"
    assert len(p.image_ids) == 8, "detected=False 的额外组不得进装载"
    assert p.uv1.shape == (8, 88, 2) and p.uv2.shape == (8, 88, 2)
    assert p.uv1.dtype == np.float64 and p.uv2.dtype == np.float64
    assert p.objp.shape == (88, 3)
    assert np.allclose(p.objp, build_object_points(_CFG))
    assert p.source1 == ("sb",) * 8 and p.source2 == ("sb",) * 8
    assert p.routes == (("sb", "sb"),) * 8
    assert not np.isnan(p.uv1).any() and not np.isnan(p.uv2).any(), "幸存对不得残留 NaN"
    # 两路观测与合成真值一致:夹具消费的 uv 原样进了 Pairs(行序对齐的自证)
    assert np.allclose(p.uv1, synth.uv1)
    assert np.allclose(p.uv2, synth.uv2)


def test_load_pairs_exclude_pairs_skips_keeps_order(tmp_path):
    """exclude_pairs 命中的组被跳过(手动剔除通道),幸存顺序保持 CSV 行序。"""
    synth, csv_path, split_dir = _build_tree(tmp_path, n_pairs=8)
    p = load_pairs(_CFG, csv_path,
                   split_dir / "cam1_train.npz", split_dir / "cam2_train.npz",
                   exclude_pairs=(synth.group_ids[0], synth.group_ids[5], "999_bogus"),
                   min_pairs=2)
    expected = tuple(g for i, g in enumerate(synth.group_ids) if i not in (0, 5))
    assert p.image_ids == expected
    assert p.uv1.shape == (6, 88, 2)
    # 清单里不存在的 group_id 不报错(逻辑剔除:命中才算)


# ---------- 三重防线:连接键 / detected / 路径交叉核对 / objp / finite ----------


def test_load_pairs_connection_key_miss_raises(tmp_path):
    """连接键不命中(CSV 多一行 npz 里不存在的 group_id)→ 中文报错。"""
    _synth, csv_path, split_dir = _build_tree(tmp_path, n_pairs=8)
    with open(csv_path, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["20991231_235959_999999",
                         "data/train/x/cam1.jpg", "data/train/x/cam2.jpg", "sb", "sb"])
    with pytest.raises(ValueError, match="不合法"):
        load_pairs(_CFG, csv_path,
                   split_dir / "cam1_train.npz", split_dir / "cam2_train.npz",
                   min_pairs=2)


def test_load_pairs_path_cross_check_raises(tmp_path):
    """csv 的 cam2_path 与 npz paths 对应行不一致 → 中文报错(错位防线)。"""
    _synth, csv_path, split_dir = _build_tree(tmp_path, n_pairs=8)
    lines = csv_path.read_text(encoding="utf-8-sig").splitlines()
    parts = lines[1].split(",")
    parts[2] = "data/train/tampered/cam2.jpg"
    lines[1] = ",".join(parts)
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    with pytest.raises(ValueError, match="不合法"):
        load_pairs(_CFG, csv_path,
                   split_dir / "cam1_train.npz", split_dir / "cam2_train.npz",
                   min_pairs=2)


def test_load_pairs_objp_tampered_raises_rerun_preprocess(tmp_path):
    """单路 objp 被篡改/规格漂移 → 中文报错提示重跑 preprocess(Pitfall 11)。"""
    _synth, csv_path, split_dir = _build_tree(tmp_path, n_pairs=8)
    npz_path = split_dir / "cam2_train.npz"
    data = dict(np.load(npz_path, allow_pickle=False))
    objp = data["objp"].copy()
    objp[0, 0] += 5.0
    np.savez(npz_path, objp=objp, **{k: v for k, v in data.items() if k != "objp"})
    with pytest.raises(ValueError, match="重跑 preprocess"):
        load_pairs(_CFG, csv_path,
                   split_dir / "cam1_train.npz", npz_path, min_pairs=2)


def test_load_pairs_detected_row_with_nan_raises(tmp_path):
    """检出但角点含 NaN/Inf → 中文报错,绝不带毒进求解(三重防线第 3 道)。"""
    _synth, csv_path, split_dir = _build_tree(tmp_path, n_pairs=8)
    npz_path = split_dir / "cam1_train.npz"
    data = dict(np.load(npz_path, allow_pickle=False))
    uv = data["uv"].copy()
    uv[3, 7, 0] = np.nan  # 检出行埋一个 NaN
    np.savez(npz_path, uv=uv, **{k: v for k, v in data.items() if k != "uv"})
    with pytest.raises(ValueError, match="不合法"):
        load_pairs(_CFG, csv_path,
                   npz_path, split_dir / "cam2_train.npz", min_pairs=2)


def test_load_pairs_min_pairs_underflow_raises(tmp_path):
    """幸存对数低于下限 → 中文报错不硬算(一对定不了相对位姿)。"""
    _synth, csv_path, split_dir = _build_tree(tmp_path, n_pairs=8)
    with pytest.raises(ValueError, match="可用双目对"):
        load_pairs(_CFG, csv_path,
                   split_dir / "cam1_train.npz", split_dir / "cam2_train.npz",
                   min_pairs=100)


# ---------- select_pairs:相对行序子集重建 ----------


def test_select_pairs_keeps_relative_order(tmp_path):
    """select_pairs([2, 0]) → image_ids 保持 [2, 0] 的相对序,uv 行一一对应。"""
    _synth, csv_path, split_dir = _build_tree(tmp_path, n_pairs=8)
    p = load_pairs(_CFG, csv_path,
                   split_dir / "cam1_train.npz", split_dir / "cam2_train.npz",
                   min_pairs=2)
    sub = select_pairs(p, [2, 0])
    assert sub.image_ids == (p.image_ids[2], p.image_ids[0])
    assert np.array_equal(sub.uv1[0], p.uv1[2]) and np.array_equal(sub.uv1[1], p.uv1[0])
    assert np.array_equal(sub.uv2[0], p.uv2[2]) and np.array_equal(sub.uv2[1], p.uv2[0])
    assert sub.routes == (p.routes[2], p.routes[0])
    assert np.array_equal(sub.objp, p.objp), "objp 是板系常量,不随子集重排"


# ---------- load_intrinsics:按名读取 + D 布局防护(Pitfall 13) ----------


def _write_intrinsics_json(tmp_path: pathlib.Path, **overrides) -> pathlib.Path:
    """合成一份 Phase 2 契约的 intrinsics.json(15 字段,参数量级同生产)。"""
    params = {"fx": 1850.0, "fy": 1840.0, "cx": 1280.0, "cy": 720.0,
              "k1": -0.40, "k2": 0.23, "p1": 1.1e-4, "p2": -3.8e-5, "k3": -0.08}
    payload = {
        "cam": 1,
        "image_size": [2560, 1440],
        "params": params,
        "K": [[1850.0, 0.0, 1280.0], [0.0, 1840.0, 720.0], [0.0, 0.0, 1.0]],
        "D": [params["k1"], params["k2"], params["p1"], params["p2"], params["k3"]],
        "solver": "scipy",
        "refine_enabled": True,
        "rms_train_overall": 0.33,
        "rms_test_overall": 0.24,
        "n_views_train": 141,
        "n_views_test": 15,
        "dropped_frames": [],
        "solver_meta": {"status": 0},
        "board": {"cols": 11, "rows": 8, "square_size_mm": 25.0},
        "config_snapshot": {},
    }
    payload.update(overrides)
    path = tmp_path / "intrinsics.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_load_intrinsics_contract_by_name(tmp_path):
    """按名读取:K (3,3)、D (5,)、params 九名齐(INTRINSIC_NAMES 布局)。"""
    from stereo_calib.intrinsic.models import INTRINSIC_NAMES

    path = _write_intrinsics_json(tmp_path)
    view = load_intrinsics(path)
    assert isinstance(view, IntrinsicsView)
    assert view.cam == 1
    assert view.image_size == (2560, 1440), "image_size 必须是 (宽, 高) 口径"
    assert view.K.shape == (3, 3) and view.K[0, 0] == pytest.approx(1850.0)
    assert view.D.shape == (5,)
    assert view.params.shape == (9,)
    for name, expected in zip(INTRINSIC_NAMES,
                              [1850.0, 1840.0, 1280.0, 720.0, -0.40, 0.23, 1.1e-4, -3.8e-5, -0.08]):
        assert view.params[INTRINSIC_NAMES.index(name)] == pytest.approx(expected)


def test_load_intrinsics_d_layout_tamper_raises(tmp_path):
    """D 被写成 4 元素(params 九名 → dist 五名的布局错位前兆)→ 中文报错。"""
    path = _write_intrinsics_json(tmp_path, D=[-0.40, 0.23, 1.1e-4, -3.8e-5])
    with pytest.raises(ValueError, match="不合法"):
        load_intrinsics(path)


def test_load_intrinsics_missing_k_field_raises(tmp_path):
    """缺 K 字段(不是 Phase 2 契约产物)→ 中文报错。"""
    path = _write_intrinsics_json(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["K"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="不合法"):
        load_intrinsics(path)


# ---------- 源码防线 ----------


def test_observation_source_guards():
    """observation.py 源文防线:零读图、无第二份畸变公式、装载必经 load_corners。"""
    source = pathlib.Path("src/stereo_calib/stereo/observation.py").read_text(encoding="utf-8")
    assert source.count("imread") == 0, "双目链路零读图:paths 只作交叉核对,绝不打开图像"
    assert source.count("_distort") == 0, (
        "observation.py 出现畸变核符号:投影/畸变公式只准经 intrinsic.models")
    assert "np.load(" not in source, "装载必须经 load_corners(allow_pickle=False 防线)"
    assert "load_corners" in source and "build_object_points" in source, (
        "必须复用 Phase 1 读取入口与物点重建")
    assert "INTRINSIC_NAMES" in source, "params 必须按名导航(禁魔数下标,Pitfall 13)"


# ---------- Pattern 7 契约落盘:writer roundtrip(体例照 test_intrinsic_pipeline) ----------


def _write_all_artifacts(tmp_path: pathlib.Path) -> dict:
    """写齐五件产物(真实 writer 路径),返回各文件路径 + 写入样本。"""
    from stereo_calib.stereo.artifacts import (
        _EXTRINSICS_STEREO_JSON_KEYS,
        write_extrinsics_stereo_json,
        write_per_frame_csv,
        write_report_md,
        write_residuals_npz,
    )

    rng = np.random.default_rng(7)
    t = np.array([120.0, 15.0, 8.0])
    rvec = np.array([0.01, -0.02, 0.0349])
    R = cv2.Rodrigues(rvec)[0]  # 真实正交阵(手写近似阵不满足刚架)
    angle = float(np.linalg.norm(rvec))
    K1 = np.array([[1850.0, 0.0, 1280.0], [0.0, 1840.0, 720.0], [0.0, 0.0, 1.0]])
    K2 = np.array([[1850.0, 0.0, 1280.0], [0.0, 1836.0, 724.0], [0.0, 0.0, 1.0]])
    D1 = np.array([-0.40, 0.23, 1.1e-4, -3.8e-5, -0.08])
    D2 = D1.copy()
    board = BoardSpec()
    out = tmp_path / "output" / "stereo"

    json_path = write_extrinsics_stereo_json(
        out / "extrinsics_stereo.json", ref_cam=1, R=R, t=t, rvec=rvec,
        rotation_angle_deg=angle, baseline_mm=float(np.linalg.norm(t)),
        K1=K1, D1=D1, K2=K2, D2=D2, n_pairs_train=9, n_pairs_test=4,
        rms_reproj_train=0.31, rms_reproj_test=0.27,
        distance_rms_train_mm=0.42, distance_rms_test_mm=0.66,
        true_distance_mm=305.16389039334257,
        dropped_pairs=[{"group_id": "g_bad", "rms": 3.2, "round": 1, "threshold": 1.5}],
        flip_log={"train": {"pass2_flipped": 1, "pairs": ["g_flip"]},
                  "test": {"pass2_flipped": 0, "pairs": []}},
        solver_meta={"status": 0, "nfev": 42}, board=board,
        config_snapshot={"flip_ratio": 0.5})

    rows = [
        {"group_id": "g_flip", "split": "train", "route_cam1": "sb", "route_cam2": "sb",
         "reproj_rms": f"{0.29:.4f}", "dist_err_mm": f"{0.51:.4f}",
         "dropped": "False", "note": "pass2 翻转"},
        {"group_id": "g_bad", "split": "train", "route_cam1": "sb", "route_cam2": "sb",
         "reproj_rms": f"{3.2:.4f}", "dist_err_mm": f"{9.9:.4f}",
         "dropped": "True", "note": "超阈值剔除"},
        {"group_id": "g_t1", "split": "test", "route_cam1": "sb", "route_cam2": "sb",
         "reproj_rms": f"{0.27:.4f}", "dist_err_mm": f"{0.66:.4f}",
         "dropped": "False", "note": ""},
    ]
    csv_path = write_per_frame_csv(out / "per_frame_train.csv", rows)

    group_ids = ["g_a", "g_b"]
    res1 = rng.normal(0.0, 0.1, (2, 88, 2))
    res2 = rng.normal(0.0, 0.1, (2, 88, 2))
    npz_path = write_residuals_npz(
        out / "train_residuals.npz", group_ids, res1, res2,
        np.array([0.29, 0.31]), np.array([0.51, 0.42]))

    summary = {
        "ref_cam": 1, "R": R, "t": t, "rvec": rvec,
        "rotation_angle_deg": angle, "baseline_mm": float(np.linalg.norm(t)),
        "rms_reproj_train": 0.31, "rms_reproj_test": 0.27,
        "distance_rms_train_mm": 0.42, "distance_rms_test_mm": 0.66,
        "true_distance_mm": 305.16389039334257,
        "n_pairs_train": 9, "n_pairs_train_initial": 10, "n_pairs_test": 4,
        "train_evals": [
            {"group_id": "g_flip", "reproj_rms": 0.29, "dist_err_mm": 0.51,
             "flipped": "(True, False)", "dropped": False, "note": "pass2 翻转"},
            {"group_id": "g_bad", "reproj_rms": 3.2, "dist_err_mm": 9.9,
             "flipped": "(False, False)", "dropped": True, "note": "超阈值剔除"},
        ],
        "test_evals": [
            {"group_id": "g_t1", "reproj_rms": 0.27, "dist_err_mm": 0.66,
             "flipped": "(False, False)", "dropped": False, "note": ""},
        ],
        "dropped_pairs": [{"group_id": "g_bad", "rms": 3.2, "round": 1, "threshold": 1.5}],
        "flip_log": {"train": {"pass2_flipped": 1, "pairs": ["g_flip"]},
                     "test": {"pass2_flipped": 0, "pairs": []}},
        "outlier_loop": "第 1 轮后无新离群对,定稿",
    }
    md_path = write_report_md(out / "report.md", summary,
                              distance_rms_max_mm=5.0,
                              baseline_band_mm=(20.0, 2000.0),
                              config_snapshot={"flip_ratio": 0.5})
    return {"json": json_path, "csv": csv_path, "npz": npz_path, "md": md_path,
            "keys": _EXTRINSICS_STEREO_JSON_KEYS, "rows": rows}


def test_extrinsics_stereo_json_roundtrip_full_contract(tmp_path):
    """JSON 写入→读回:键全集 == 契约常量、R 3×3 嵌套、baseline_mm == ‖t‖、
    剔除/翻转留痕逐字段保真(Phase 2 契约体例,Pattern 7 + 决策 8)。"""
    art = _write_all_artifacts(tmp_path)
    payload = json.loads(art["json"].read_text(encoding="utf-8"))
    assert set(payload) == set(art["keys"]), "JSON 字段集漂移(契约 = Pattern 7 全集 + train 距离 RMS)"
    R = np.asarray(payload["R"], dtype=np.float64)
    t = np.asarray(payload["t"], dtype=np.float64)
    assert R.shape == (3, 3) and t.shape == (3,)
    assert abs(np.linalg.norm(R @ R.T) - np.sqrt(3.0)) < 1e-12, "R 必须是正交阵(刚架)"
    assert payload["baseline_mm"] == pytest.approx(float(np.linalg.norm(t))), (
        "契约:baseline_mm = ‖t‖(两处来源漂移当场抓住)")
    assert payload["ref_cam"] == 1
    assert payload["n_pairs_train"] == 9 and payload["n_pairs_test"] == 4
    assert payload["true_distance_mm"] == pytest.approx(305.16389039334257, abs=1e-9)
    assert payload["dropped_pairs"] == [
        {"group_id": "g_bad", "rms": 3.2, "round": 1, "threshold": 1.5}]
    assert payload["flip_log"]["train"] == {"pass2_flipped": 1, "pairs": ["g_flip"]}
    assert payload["flip_log"]["test"] == {"pass2_flipped": 0, "pairs": []}
    assert np.asarray(payload["D1"]).shape == (5,), "D 快照必须保持 OpenCV 5 长布局"


def test_extrinsics_json_baseline_tamper_raises(tmp_path):
    """baseline_mm 与 ‖t‖ 两张皮 → 写前断言中文报错(契约一致性防线)。"""
    from stereo_calib.stereo.artifacts import write_extrinsics_stereo_json

    R = np.eye(3)
    t = np.array([120.0, 15.0, 8.0])
    with pytest.raises(AssertionError, match="不一致"):
        write_extrinsics_stereo_json(
            tmp_path / "x.json", ref_cam=1, R=R, t=t, rvec=np.zeros(3),
            rotation_angle_deg=0.0, baseline_mm=999.0,
            K1=np.eye(3), D1=np.zeros(5), K2=np.eye(3), D2=np.zeros(5),
            n_pairs_train=1, n_pairs_test=1, rms_reproj_train=0.0,
            rms_reproj_test=0.0, distance_rms_train_mm=0.0,
            distance_rms_test_mm=0.0, true_distance_mm=1.0,
            dropped_pairs=[], flip_log={}, solver_meta={}, board=BoardSpec(),
            config_snapshot={})


def test_residuals_npz_roundtrip_no_pickle(tmp_path):
    """npz 写入→allow_pickle=False 读回:unicode 对账列 + float64 形状契约。"""
    art = _write_all_artifacts(tmp_path)
    with np.load(art["npz"], allow_pickle=False) as data:
        assert data["group_ids"].dtype.kind == "U", "对账列必须定长 unicode"
        assert data["group_ids"].tolist() == ["g_a", "g_b"]
        assert data["residuals_cam1"].dtype == np.float64
        assert data["residuals_cam1"].shape == (2, 88, 2)
        assert data["residuals_cam2"].shape == (2, 88, 2)
        assert data["per_pair_rms"].shape == (2,)
        assert data["dist_err_mm"].shape == (2,)


def test_per_frame_csv_header_and_order(tmp_path):
    """CSV utf-8-sig 表头 + 列序契约:8 列依 _PER_FRAME_FIELDS,行序保真。"""
    from stereo_calib.stereo.artifacts import _PER_FRAME_FIELDS

    art = _write_all_artifacts(tmp_path)
    raw = art["csv"].read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf"), "必须 utf-8-sig(Excel 双击直开)"
    text = art["csv"].read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    assert lines[0].split(",") == _PER_FRAME_FIELDS, "列序漂移"
    assert len(lines) == 4, "表头 + 3 数据行"
    assert lines[1].startswith("g_flip,train,sb,sb,"), "行序与内容保真"


def test_report_md_sections_and_verdict(tmp_path):
    """report.md:距离残差表 / PASS 判定 / 基线量级核对 / 剔除与翻转留痕 /
    配置快照附录逐节在位(人工核对出口:量级核对行 + 报告结论非退出条件)。"""
    art = _write_all_artifacts(tmp_path)
    text = art["md"].read_text(encoding="utf-8")
    for token in ("距离残差", "PASS", "基线", "量级核对", "剔除留痕",
                  "翻转留痕", "配置快照", "只作报告结论,不作退出条件",
                  "305.1639"):
        assert token in text, f"report.md 缺少「{token}」节/字样"
    assert "g_bad" in text and "超阈值剔除" in text, "逐帧表须含剔除对留痕"

