"""preprocess/parallel 的批完整性测试:异常信封不杀批、串行并行同果、进度计数。

信封的必要性(joblib 在任一任务抛异常时会关掉整条结果流,参考工程
2026-10-19 真实事故 700+ 任务全没)在这里用"坏 JPEG 混进任务集"实测锁定:
结果数必须等于任务数,坏任务交回 TaskFailed 信封,好任务照常出结果。
另有一条端到端用例:max_workers=1 串行与 =2 并行跑完整 preprocess,
npz 逐帧一致,且 fail 清单同时覆盖「读图失败」与「SB 与 classic 均未检出」
两类 reason。
"""

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from stereo_calib.io.config import CornersConfig
from stereo_calib.preprocess.parallel import TaskFailed, run_parallel
from stereo_calib.preprocess.pipeline import detect_one

# 检测 worker 用的配置:直接用数据类默认值(与仓库 config.yaml 一致)
_CFG = CornersConfig()
_CV2_THREADS = 0


def _make_task(path: Path, image_id: str) -> tuple:
    """按 detect_one 的任务元组契约造一个最小任务(单路 cam1/train)。"""
    return (str(path), image_id, 1, image_id.removesuffix("_cam1"),
            "train", 11, 8, _CFG, _CV2_THREADS)


@pytest.fixture
def task_set(tmp_path: Path, make_chessboard_image):
    """三个正常合成棋盘任务 + 一个坏 JPEG 任务(读图当场抛 ValueError)。"""
    tasks = []
    for i, rvec in enumerate([np.array([0.0, 0.05, 0.0]),
                              np.array([0.0, -0.10, 0.03]),
                              np.array([0.05, 0.0, 0.0])]):
        img, _, _ = make_chessboard_image(degrade="blur", rvec=rvec)
        p = tmp_path / f"g{i}" / "cam1.jpg"
        p.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(p), img), f"合成图写盘失败:{p}"
        tasks.append(_make_task(p, f"g{i}_cam1"))
    bad = tmp_path / "g9" / "cam1.jpg"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_bytes(b"not a jpeg at all")
    tasks.append(_make_task(bad, "g9_cam1"))
    return tasks


def test_taskfailed_envelope_keeps_batch_intact(task_set):
    """坏 JPEG 只废它自己:结果数 == 任务数(T-01-06 批完整性),坏任务
    交回 TaskFailed 信封且装着读图 ValueError,好任务照常出检测 dict。"""
    results = list(run_parallel(2, task_set, worker=detect_one))
    assert len(results) == len(task_set), "单帧失败绝不允许杀掉整批结果流"
    envelopes = [r for r in results if isinstance(r, TaskFailed)]
    assert len(envelopes) == 1
    assert isinstance(envelopes[0].error, ValueError)  # imread_checked 的读图报错
    good = [r for r in results if not isinstance(r, TaskFailed)]
    assert len(good) == 3
    for r in good:
        assert r["result"] is not None
        assert r["result"]["uv"].shape == (88, 2)


def test_serial_and_parallel_consistent(task_set):
    """max_workers=1 串行回退与 max_workers=2 并行:逐帧 uv np.allclose、
    detector 一致(检测无随机性,串行是调试入口,结果必须与并行同序同果)。"""
    good_tasks = task_set[:3]
    serial = list(run_parallel(1, good_tasks, worker=detect_one))
    par = list(run_parallel(2, good_tasks, worker=detect_one))
    assert len(serial) == len(par) == 3
    for s, p in zip(serial, par):
        assert not isinstance(s, TaskFailed) and not isinstance(p, TaskFailed)
        assert np.allclose(s["result"]["uv"], p["result"]["uv"], atol=1e-9)
        assert s["result"]["detector"] == p["result"]["detector"]


def test_progress_callback_counts_all(task_set):
    """进度回调按 1..N 逐个触发,最后一个 done == total(批完整性的进度侧证据)。"""
    seen: list[dict] = []
    list(run_parallel(2, task_set, worker=detect_one, progress=seen.append))
    assert seen, "进度回调必须被调用"
    assert [d["done"] for d in seen] == list(range(1, len(task_set) + 1))
    assert seen[-1]["done"] == seen[-1]["total"] == len(task_set)


# ---------- 端到端:串行与并行跑完整 preprocess,产物逐帧一致 ----------

# 最小合法配置(与 test_preprocess_pipeline 同一体例);@MAX@/@OUT@ 两个槽位
# 分别注入 max_workers 与 output_root,同树两跑、产物分目录互不覆盖
_CONFIG_TEMPLATE = """\
board:
  board_type: "chessboard"
  cols: 11
  rows: 8
  square_size_mm: 25.0
  unit: "mm"
data:
  train: "data/train"
  test: "data/test"
output_root: "@OUT@"
cv2_threads: 0
preprocess:
  max_workers: @MAX@
  corners:
    classic_flags: [adaptive_thresh, normalize_image]
    sb_flags: 0
    sb_fallback_classic: true
    subpix: {window_granularity: frame, min_win: 3, max_iter: 30, epsilon: 0.01}
  copy_fail_originals: true
  visualization: {jpeg_quality: 90}
"""

_VIEWS = {
    ("g1", 1): {"rvec": np.array([0.0, 0.05, 0.0]), "tvec": np.array([0.0, 0.0, 800.0])},
    ("g1", 2): {"rvec": np.array([0.05, 0.0, 0.0]), "tvec": np.array([20.0, 0.0, 800.0])},
    ("g2", 1): {"rvec": np.array([0.0, -0.10, 0.03]), "tvec": np.array([-15.0, 10.0, 850.0])},
    ("g2", 2): {"rvec": np.array([0.10, 0.05, -0.03]), "tvec": np.array([10.0, -5.0, 900.0])},
}
_GROUPS = {
    "g1": ("20261023_154025", "20261023_154025_519359"),
    "g2": ("20261023_154043", "20261023_154043_321789"),
}
_NOISE_ID = "20261023_154100_000001_cam1"   # 纯噪声组:双路线均未检出
_CORRUPT_ID = "20261023_154111_000002_cam1"  # 坏 JPEG 组:读图失败


def _build_data_tree(tmp_path: Path, make_chessboard_image) -> None:
    """两个干净双目组 + 一个纯噪声 cam1 + 一个坏 JPEG cam1 的合成数据树。"""
    for (g, cam), view in _VIEWS.items():
        ts, leaf = _GROUPS[g]
        img, _, _ = make_chessboard_image(degrade="blur", **view)
        target = tmp_path / "data" / "train" / ts / leaf / f"cam{cam}.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(target), img), f"合成图写盘失败:{target}"
    noise = np.random.default_rng(3).integers(0, 256, size=(480, 640), dtype=np.uint8)
    noise_t = tmp_path / "data" / "train" / "20261023_154100" / "20261023_154100_000001" / "cam1.jpg"
    noise_t.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(noise_t), noise)
    corrupt_t = tmp_path / "data" / "train" / "20261023_154111" / "20261023_154111_000002" / "cam1.jpg"
    corrupt_t.parent.mkdir(parents=True, exist_ok=True)
    corrupt_t.write_bytes(b"not a jpeg at all")


def _read_fail_csv(out_root: Path) -> dict[str, str]:
    with open(out_root / "train" / "fail_train.csv", encoding="utf-8-sig", newline="") as f:
        return {row["image_id"]: row["reason"] for row in csv.DictReader(f)}


def test_preprocess_serial_matches_parallel(tmp_path, monkeypatch, make_chessboard_image):
    """端到端:max_workers=1 串行与 =2 并行跑完整 preprocess,四份 npz 的
    uv/detected/source 逐帧一致;fail 清单同时含「SB 与 classic 均未检出」
    (噪声帧)与「读图失败」(坏 JPEG 帧)两类 reason——信封路径与双路线
    措辞都在真实编排里锁定。"""
    from stereo_calib.cli.main import preprocess

    _build_data_tree(tmp_path, make_chessboard_image)
    monkeypatch.chdir(tmp_path)

    for max_workers, out_root in ((1, "output_serial"), (2, "output_par")):
        (tmp_path / "config.yaml").write_text(
            _CONFIG_TEMPLATE.replace("@MAX@", str(max_workers)).replace("@OUT@", out_root),
            encoding="utf-8")
        preprocess()

    for cam in (1, 2):
        a = np.load(tmp_path / "output_serial" / "train" / f"cam{cam}_train.npz",
                    allow_pickle=False)
        b = np.load(tmp_path / "output_par" / "train" / f"cam{cam}_train.npz",
                    allow_pickle=False)
        assert (a["detected"] == b["detected"]).all()
        assert (a["source"] == b["source"]).all()
        assert np.allclose(a["uv"], b["uv"], atol=1e-9, equal_nan=True), (
            f"cam{cam} 串行与并行的角点逐帧不一致(串行回退必须与并行同果)"
        )

    fails = _read_fail_csv(tmp_path / "output_par")
    assert _NOISE_ID in fails and fails[_NOISE_ID] == "SB 与 classic 均未检出"
    assert _CORRUPT_ID in fails and fails[_CORRUPT_ID].startswith("读图失败")
    # 检出的帧 source 只可能是两颗检测器之一,绝不为空(带掩码稠密契约)
    data = np.load(tmp_path / "output_par" / "train" / "cam1_train.npz", allow_pickle=False)
    ok = data["detected"]
    assert set(data["source"][ok]) <= {"sb", "classic"} and ok.sum() == 2
