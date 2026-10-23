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


@pytest.fixture
def synthetic_run(tmp_path, monkeypatch, make_chessboard_image):
    """造合成双目树 + 最小配置,进程内跑 preprocess(),返回产物根 output/。"""
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

    (tmp_path / "config.yaml").write_text(_CONFIG_YAML, encoding="utf-8")

    before = _tree_snapshot(tmp_path / "data")
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
    # 检出帧 source 全部记 sb(SB 主路线)
    ok = data1["detected"]
    assert (data1["source"][ok] == "sb").all()
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
        for row in rows:  # 每行 route 都标了来源
            assert row["route"] == "sb"
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
