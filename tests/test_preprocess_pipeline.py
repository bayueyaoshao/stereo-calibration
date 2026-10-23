"""preprocess 端到端行为测试:合成双目树 → cli preprocess() → 产物契约断言。

模式照参考工程 tests/test_cli_main.py(只读引用):进程内直接调命令函数 +
monkeypatch.chdir 临时目录,产物落 tmp,真实 output/ 不受影响。

本测试是 Walking Skeleton 的契约锁定:npz 八数组 schema、NaN 掩码行、
usable/fail 行数一致、objp 顺序、全部产物在 output/ 下、以及 INFRA-03
(data/ 只读——运行前后对合成数据树逐文件 st_mtime_ns 快照比对不变)。
"""

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from stereo_calib.cli.main import preprocess

# 最小合法配置(与 test_io_config 同一体例;数据只有 train,test 缺目录记警告)
_CONFIG_YAML = """\
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
  max_workers: 12
  corners:
    classic_flags: [adaptive_thresh, normalize_image]
    sb_flags: 0
    sb_fallback_classic: true
    subpix: {window_granularity: frame, min_win: 3, max_iter: 30, epsilon: 0.01}
  copy_fail_originals: true
  visualization: {jpeg_quality: 90}
"""

# 两个组 × 两路相机的稍不同视角(模拟真实双目位姿差)
_VIEWS = {
    ("g1", 1): dict(rvec=np.array([0.0, 0.05, 0.0]), tvec=np.array([0.0, 0.0, 800.0])),
    ("g1", 2): dict(rvec=np.array([0.05, 0.0, 0.0]), tvec=np.array([20.0, 0.0, 800.0])),
    ("g2", 1): dict(rvec=np.array([0.0, -0.10, 0.03]), tvec=np.array([-15.0, 10.0, 850.0])),
    ("g2", 2): dict(rvec=np.array([0.10, 0.05, -0.03]), tvec=np.array([10.0, -5.0, 900.0])),
}
_GROUPS = {
    "g1": ("20261023_154025", "20261023_154025_519359"),
    "g2": ("20261023_154043", "20261023_154043_321789"),
    "g3": ("20261023_154100", "20261023_154100_000001"),  # 纯噪声组:只有 cam1
}
NOISE_ID = "20261023_154100_000001_cam1"

_NPZ_KEYS = {"image_ids", "paths", "uv", "objp", "source", "detected", "sq_px", "win_size"}


def _tree_snapshot(root: Path) -> dict:
    """INFRA-03:data/ 树逐文件 (mtime_ns, size) 快照,运行前后必须逐字节不变。"""
    return {
        str(p.relative_to(root)): (p.stat().st_mtime_ns, p.stat().st_size)
        for p in sorted(root.rglob("*")) if p.is_file()
    }


def _read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _build_synthetic_dataset(tmp_path, make_chessboard_image,
                             config_yaml: str = _CONFIG_YAML) -> dict:
    """造合成双目树 + 写配置文件(多个测试夹具共用的树工厂),返回 data 快照。

    树内容:g1/g2 两组两路合成棋盘(检出),g3 组纯噪声 cam1(检测失败);
    详见 _VIEWS / _GROUPS 表。
    """
    for (g, cam), view in _VIEWS.items():
        ts, leaf = _GROUPS[g]
        img, _corners, _sq = make_chessboard_image(degrade="blur", **view)
        target = tmp_path / "data" / "train" / ts / leaf / f"cam{cam}.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(target), img), f"合成图写盘失败:{target}"

    # 纯噪声 cam1:制造一条 fail 记录(SB 必不检出)
    ts3, leaf3 = _GROUPS["g3"]
    rng = np.random.default_rng(3)
    noise = rng.integers(0, 256, size=(480, 640), dtype=np.uint8)
    noise_target = tmp_path / "data" / "train" / ts3 / leaf3 / "cam1.jpg"
    noise_target.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(noise_target), noise)

    (tmp_path / "config.yaml").write_text(config_yaml, encoding="utf-8")
    return _tree_snapshot(tmp_path / "data")


@pytest.fixture
def synthetic_run(tmp_path, monkeypatch, make_chessboard_image):
    """造合成双目树 + 最小配置,进程内跑 preprocess(),返回产物根 output/ 与 data 快照。"""
    before = _build_synthetic_dataset(tmp_path, make_chessboard_image)
    monkeypatch.chdir(tmp_path)
    preprocess()  # 进程内直调命令函数;坏输入才会 typer.Exit(1),这里应成功
    return tmp_path / "output", before


def test_npz_schema_and_masked_rows(synthetic_run):
    """四个 npz 齐且 allow_pickle=False 可读;八数组 schema、dtype、形状正确;
    噪声帧行 detected=False、uv 整行 NaN、source 为空串。"""
    out, _before = synthetic_run
    data1 = np.load(out / "train" / "cam1_train.npz", allow_pickle=False)
    data2 = np.load(out / "train" / "cam2_train.npz", allow_pickle=False)
    for data, expected_n in ((data1, 3), (data2, 2)):
        assert set(data.files) == _NPZ_KEYS
        assert data["image_ids"].shape == (expected_n,)
        assert data["uv"].shape == (expected_n, 88, 2)
        assert data["uv"].dtype == np.float64
        assert data["objp"].shape == (88, 3)
        assert data["detected"].dtype == bool

    ids1 = [str(s) for s in data1["image_ids"]]
    j = ids1.index(NOISE_ID)
    assert data1["detected"][j] == np.False_
    assert np.isnan(data1["uv"][j]).all(), "未检出帧 uv 必须整行 NaN(带掩码稠密契约)"
    assert data1["source"][j] == ""
    assert np.isnan(data1["sq_px"][j]) and np.isnan(data1["win_size"][j])
    # 检出帧 source 必是两颗检测器之一(合成图全部 SB 检出;classic 兜底
    # 接入后该列取值 ∈ {"sb", "classic"},绝不为空)
    ok = data1["detected"]
    assert np.isin(data1["source"][ok], ("sb", "classic")).all()
    assert not np.isnan(data1["uv"][ok]).any()


def test_objp_order_contract(synthetic_run):
    """objp 契约:objp[row*cols+col] = (col*25, row*25, 0),与 canonicalize 后 uv 同序。"""
    out, _before = synthetic_run
    data = np.load(out / "train" / "cam1_train.npz", allow_pickle=False)
    assert data["objp"][3].tolist() == [75.0, 0.0, 0.0]
    assert data["objp"][87].tolist() == [250.0, 175.0, 0.0]


def test_usable_csv_rows_match_detected(synthetic_run):
    """usable CSV 行数(去表头)== 对应 npz detected 计数(D-01:检出即可用)。"""
    out, _before = synthetic_run
    for cam in (1, 2):
        data = np.load(out / "train" / f"cam{cam}_train.npz", allow_pickle=False)
        rows = _read_csv(out / "train" / f"usable_cam{cam}_train.csv")
        assert len(rows) == int(data["detected"].sum())
        for row in rows:  # 每行 route 都标了来源(SB 或 classic 兜底)
            assert row["route"] in ("sb", "classic")
            assert row["image_id"]


def test_fail_csv_contains_noise_frame_with_reason(synthetic_run):
    """fail CSV 含噪声帧且 reason 非空(人工复核入口 D-02)。"""
    out, _before = synthetic_run
    rows = _read_csv(out / "train" / "fail_train.csv")
    ids = [r["image_id"] for r in rows]
    assert NOISE_ID in ids
    assert all(r["reason"] for r in rows)


def test_meta_json_sidecar(synthetic_run):
    """meta.json 旁车:board 规格与配置快照在,detected 计数与 CSV 行数一致。"""
    out, _before = synthetic_run
    meta = json.loads(
        (out / "train" / "cam1_train.meta.json").read_text(encoding="utf-8"))
    assert meta["board"]["cols"] == 11
    assert meta["board"]["rows"] == 8
    assert meta["board"]["square_size_mm"] == 25.0
    assert "preprocess" in meta["config_snapshot"]
    assert meta["canonical_rule"].startswith("canonical-v1")
    data = np.load(out / "train" / "cam1_train.npz", allow_pickle=False)
    rows = _read_csv(out / "train" / "usable_cam1_train.csv")
    assert meta["n_detected"] == int(data["detected"].sum()) == len(rows)
    assert NOISE_ID in meta["fail_reasons"]


def test_artifacts_under_output_and_data_untouched(synthetic_run):
    """全部产物都在 output/ 下;合成 data/ 树 mtime/size 运行前后不变(INFRA-03)。"""
    out, before = synthetic_run
    tmp_root = out.parent
    produced = [p for p in sorted(out.rglob("*")) if p.is_file()]
    assert produced, "output/ 下必须有产物"
    for p in produced:
        assert out.resolve() in p.resolve().parents or p.resolve() == out.resolve()
    after = _tree_snapshot(tmp_root / "data")
    assert after == before, "preprocess 绝不允许改动 data/ 里任何文件"


# ---- 01-03 任务 1:全量角点标注图 + 失败帧原图副本(D-07/D-08/D-02)----

def test_visualization_full_coverage_no_sampling(synthetic_run):
    """D-07:visualization JPEG 数 == 该 split 两路 detected 计数和(全量无抽样);
    每张检出帧都有同名标注图,且 JPEG 非空(size > 10KB)。"""
    out, _before = synthetic_run
    vis = sorted((out / "train" / "visualization").glob("*.jpg"))
    n_detected = 0
    for cam in (1, 2):
        data = np.load(out / "train" / f"cam{cam}_train.npz", allow_pickle=False)
        det = data["detected"]
        n_detected += int(det.sum())
        for i, iid in enumerate(data["image_ids"]):
            if det[i]:
                marked = out / "train" / "visualization" / f"{iid}.jpg"
                assert marked.is_file(), f"检出帧 {iid} 缺角点标注图(D-07 全量覆盖被破坏)"
    assert len(vis) == n_detected, "标注图张数必须恰等于两路检出数之和(无抽样、无残留)"
    for p in vis:
        assert p.stat().st_size > 10_000, f"{p.name} 疑似空图(JPEG q90 应远大于 10KB)"


def test_fail_originals_byte_identical_copies(synthetic_run):
    """D-02:每个 fail 帧都有 fail_{image_id}.jpg 副本且字节数与源一致(copy2);
    副本数恰等于 fail 清单行数(不多不少)。"""
    out, _before = synthetic_run
    fail_dir = out / "train" / "fail_originals"
    rows = _read_csv(out / "train" / "fail_train.csv")
    assert rows, "合成树含噪声帧,fail 清单不应为空"
    for row in rows:
        dst = fail_dir / f"fail_{row['image_id']}.jpg"
        assert dst.is_file(), f"失败帧 {row['image_id']} 缺原图副本"
        assert dst.stat().st_size == Path(row["path"]).stat().st_size, \
            f"{dst.name} 与源文件字节数不一致(copy2 应逐字节复制)"
    assert len(list(fail_dir.glob("fail_*.jpg"))) == len(rows)


def test_copy_fail_originals_switch_off(tmp_path, monkeypatch, make_tiny_jpg, capsys):
    """开关关闭:fail_originals 不产出(目录不存在)且中文汇总含「已关闭」。"""
    make_tiny_jpg("data/train/20261023_154100/20261023_154100_000001/cam1.jpg", 64, 48)
    (tmp_path / "config.yaml").write_text(
        _CONFIG_YAML.replace("copy_fail_originals: true", "copy_fail_originals: false"),
        encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    preprocess()
    out = tmp_path / "output"
    rows = _read_csv(out / "train" / "fail_train.csv")
    assert rows, "纯噪声小图必然检出失败,fail 清单不应为空"
    assert not (out / "train" / "fail_originals").exists(), \
        "copy_fail_originals 关闭时绝不能产出 fail_originals 目录"
    assert "已关闭" in capsys.readouterr().out, "开关关闭时汇总必须打印「已关闭」提示"


# ---- 01-03 任务 2:公共视场配对 + 运行前清理 + 只读锁定(PREP-04/D-05/D-11)----

def test_common_fov_pairs_pairing_rules(tmp_path, monkeypatch, make_chessboard_image):
    """配对规则:一组两路均检出 → 1 对;一组仅 cam1 检出 → 不成对且绝不把
    cam1 误标为失败(cam1 仍在 usable、cam2 在 fail);列序恰为契约五列。"""
    # 组 ok:两路棋盘 → 期望 1 对
    ts, leaf_ok = _GROUPS["g1"]
    for cam in (1, 2):
        img, _, _ = make_chessboard_image(degrade="blur", **_VIEWS[("g1", cam)])
        target = tmp_path / "data" / "train" / ts / leaf_ok / f"cam{cam}.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(target), img)
    # 组 half:cam1 棋盘 + cam2 纯噪声 → cam1 检出、cam2 失败,期望 0 对
    ts_h, leaf_half = _GROUPS["g2"]
    img1, _, _ = make_chessboard_image(degrade="blur", **_VIEWS[("g2", 1)])
    half_dir = tmp_path / "data" / "train" / ts_h / leaf_half
    half_dir.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(half_dir / "cam1.jpg"), img1)
    rng = np.random.default_rng(7)
    noise = rng.integers(0, 256, size=(480, 640), dtype=np.uint8)
    assert cv2.imwrite(str(half_dir / "cam2.jpg"), noise)

    (tmp_path / "config.yaml").write_text(_CONFIG_YAML, encoding="utf-8")
    before = _tree_snapshot(tmp_path / "data")
    monkeypatch.chdir(tmp_path)
    preprocess()
    out = tmp_path / "output"

    rows = _read_csv(out / "train" / "common_fov_pairs_train.csv")
    assert len(rows) == 1, "只有两路均检出的组才成对"
    assert rows[0]["group_id"] == leaf_ok
    assert list(rows[0].keys()) == ["group_id", "cam1_path", "cam2_path",
                                    "route_cam1", "route_cam2"]
    assert rows[0]["cam1_path"].endswith("cam1.jpg")
    assert rows[0]["cam2_path"].endswith("cam2.jpg")
    assert rows[0]["route_cam1"] in ("sb", "classic")

    # 单边检出:cam1 仍在 usable、cam2 在 fail;cam1 绝不因配不上对被记失败
    usable1 = _read_csv(out / "train" / "usable_cam1_train.csv")
    assert f"{leaf_half}_cam1" in [r["image_id"] for r in usable1]
    fail_ids = [r["image_id"] for r in _read_csv(out / "train" / "fail_train.csv")]
    assert f"{leaf_half}_cam2" in fail_ids
    assert f"{leaf_half}_cam1" not in fail_ids, "fail 清单只记检测失败,绝不把单边检出误标"

    # Pitfall 8 警告信号永不触发:公共对数 <= min(两路检出数)
    d1 = np.load(out / "train" / "cam1_train.npz", allow_pickle=False)
    d2 = np.load(out / "train" / "cam2_train.npz", allow_pickle=False)
    assert len(rows) <= min(int(d1["detected"].sum()), int(d2["detected"].sum()))
    assert _tree_snapshot(tmp_path / "data") == before, "配对流程同样绝不允许改动 data/"


def test_common_pairs_count_and_meta(synthetic_run):
    """主夹具:g1/g2 两路均检出 → 恒 2 对(单边噪声组不入对);meta.json 记
    common_pairs;公共对数 <= min(cam1, cam2 检出)(Pitfall 8 全量断言)。"""
    out, _before = synthetic_run
    rows = _read_csv(out / "train" / "common_fov_pairs_train.csv")
    d1 = np.load(out / "train" / "cam1_train.npz", allow_pickle=False)
    d2 = np.load(out / "train" / "cam2_train.npz", allow_pickle=False)
    assert len(rows) == 2  # g1、g2;g3 只有 cam1(噪声)不成对
    assert len(rows) <= min(int(d1["detected"].sum()), int(d2["detected"].sum()))
    for cam in (1, 2):
        meta = json.loads(
            (out / "train" / f"cam{cam}_train.meta.json").read_text(encoding="utf-8"))
        assert meta["common_pairs"] == 2


def test_rerun_clears_stale_artifacts(tmp_path, monkeypatch, make_chessboard_image):
    """Pitfall 7:运行前清理——重跑后孤儿文件(含手工塞进 visualization/ 的
    stale.jpg)消失;两次运行 data/ 快照均不变;目录重建后产物仍齐全。"""
    _build_synthetic_dataset(tmp_path, make_chessboard_image)
    monkeypatch.chdir(tmp_path)
    before = _tree_snapshot(tmp_path / "data")
    preprocess()
    out = tmp_path / "output"
    vis = out / "train" / "visualization"
    (vis / "stale.jpg").write_bytes(b"stale artifact from a previous run")
    (out / "train" / "orphan.txt").write_text("orphan inside train split dir")

    preprocess()  # 重跑:开头 rmtree 应把 train/test 产物目录整个重建
    assert not (vis / "stale.jpg").exists(), "孤儿标注图必须随重跑消失(D-11 无残留)"
    assert not (out / "train" / "orphan.txt").exists(), "产物子目录内的任何残留都要清掉"
    assert _tree_snapshot(tmp_path / "data") == before, "重跑同样绝不允许改动 data/"
    n_vis = len(list(vis.glob("*.jpg")))
    n_detected = sum(int(np.load(out / "train" / f"cam{c}_train.npz",
                                 allow_pickle=False)["detected"].sum())
                     for c in (1, 2))
    assert n_vis == n_detected == 4, "重建后标注图数仍须恰等于两路检出数之和"
