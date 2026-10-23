"""io/dataset.py 扫描韧性测试:两层嵌套、cam 号解析、单边缺文件、坏文件名、坏图。

测试维度照参考工程 tests/test_io_dataset.py(只读引用),数据布局按本项目
(<目录>/<组目录>/cam{1,2}.jpg 两层嵌套)用 make_tiny_jpg 造树。
"""

from pathlib import Path

import pytest

from stereo_calib.io.config import DataDirs
from stereo_calib.io.dataset import imread_checked, scan_dataset


@pytest.fixture
def in_data_dir(make_tiny_jpg, monkeypatch):
    """chdir 到数据根的父目录,让相对路径 data/train / data.test 生效。"""
    monkeypatch.chdir(Path(make_tiny_jpg.data_root).parent)
    return make_tiny_jpg


def test_scan_two_level_tree_records(in_data_dir):
    """正常两层树:扫描出记录,image_id/group_id/cam/split 全部正确。"""
    in_data_dir("data/train/20261023_154025/20261023_154025_519359/cam1.jpg", 8, 8)
    in_data_dir("data/train/20261023_154025/20261023_154025_519359/cam2.jpg", 8, 8)
    in_data_dir("data/train/20261023_154043/20261023_154043_321789/cam1.jpg", 8, 8)
    in_data_dir("data/train/20261023_154043/20261023_154043_321789/cam2.jpg", 8, 8)

    result = scan_dataset(DataDirs(train="data/train", test="data/test"))

    assert len(result.records) == 4
    by_id = {r.image_id: r for r in result.records}
    rec = by_id["20261023_154025_519359_cam1"]
    assert rec.cam == 1
    assert rec.split == "train"
    assert rec.group_id == "20261023_154025_519359"
    assert rec.path.name == "cam1.jpg"
    assert "20261023_154025_519359_cam2" in by_id
    assert "20261023_154043_321789_cam2" in by_id


def test_missing_test_dir_warns_not_raises(in_data_dir):
    """缺 test 目录:中文警告 + 按 0 张处理,不是异常(只有训练集是合法输入)。"""
    in_data_dir("data/train/20261023_154025/20261023_154025_519359/cam1.jpg", 8, 8)
    in_data_dir("data/train/20261023_154025/20261023_154025_519359/cam2.jpg", 8, 8)

    result = scan_dataset(DataDirs(train="data/train", test="data/test"))

    assert all(r.split == "train" for r in result.records)
    assert any("不存在" in w and "test" in w for w in result.warnings)


def test_both_splits_empty_raises(in_data_dir):
    """两个划分都扫不出记录:中文 ValueError(数据没就位不是可继续状态)。"""
    with pytest.raises(ValueError, match="没有扫到任何有效的双目图像"):
        scan_dataset(DataDirs(train="data/train", test="data/test"))


def test_single_cam_group_warns_and_keeps_present_side(in_data_dir):
    """叶子目录只有 cam1 无 cam2:警告,该 cam1 记录照常存在、cam2 记录不存在。"""
    in_data_dir("data/train/20261023_154025/20261023_154025_519359/cam1.jpg", 8, 8)

    result = scan_dataset(DataDirs(train="data/train", test="data/test"))

    ids = [r.image_id for r in result.records]
    assert ids == ["20261023_154025_519359_cam1"]
    assert any("缺" in w and "cam2" in w for w in result.warnings)


def test_unsupported_cam_filename_warns_and_skips(in_data_dir):
    """坏文件名(cam3.jpg):中文告警跳过,不产生记录。"""
    in_data_dir("data/train/20261023_154025/20261023_154025_519359/cam1.jpg", 8, 8)
    in_data_dir("data/train/20261023_154025/20261023_154025_519359/cam2.jpg", 8, 8)
    in_data_dir("data/train/20261023_154043/20261023_154043_321789/cam3.jpg", 8, 8)

    result = scan_dataset(DataDirs(train="data/train", test="data/test"))

    ids = [r.image_id for r in result.records]
    assert "20261023_154043_321789_cam3" not in ids
    assert len(ids) == 2
    assert any("cam3" in w and "跳过" in w for w in result.warnings)


def test_empty_leaf_dir_warns(in_data_dir):
    """空组目录:中文警告点名,不让"少了一半数据"静默发生。"""
    (Path(in_data_dir.data_root) / "train" / "20261023_154043" / "20261023_154043_000001") \
        .mkdir(parents=True, exist_ok=True)
    in_data_dir("data/train/20261023_154025/20261023_154025_519359/cam1.jpg", 8, 8)
    in_data_dir("data/train/20261023_154025/20261023_154025_519359/cam2.jpg", 8, 8)

    result = scan_dataset(DataDirs(train="data/train", test="data/test"))

    assert any("没有有效的 cam1.jpg/cam2.jpg" in w for w in result.warnings)
    assert len(result.records) == 2


def test_same_split_duplicate_leaf_dirs_warn_and_keep_first(in_data_dir):
    """同划分不同父目录下同名叶子目录(CR-01):不再静默合并组——
    「叶子目录重名」中文警告触发(既有去重措辞,现在对同划分碰撞可达),
    只保留先出现者的记录;image_id/group_id 契约不变(配对键仍是叶子名,
    不存在跨拍摄的双目错配)。"""
    in_data_dir("data/train/20261023_154025/dup_leaf/cam1.jpg", 8, 8)
    in_data_dir("data/train/20261023_154025/dup_leaf/cam2.jpg", 8, 8)
    in_data_dir("data/train/20261023_154043/dup_leaf/cam1.jpg", 8, 8)
    in_data_dir("data/train/20261023_154043/dup_leaf/cam2.jpg", 8, 8)

    result = scan_dataset(DataDirs(train="data/train", test="data/test"))

    assert any("重复" in w and "叶子目录重名" in w for w in result.warnings), \
        f"同划分叶子重名必须触发去重警告,实际警告:{result.warnings}"
    assert len(result.records) == 2, "同名叶子两组各两条,去重后恰留先出现者的 2 条"
    by_id = {r.image_id: r for r in result.records}
    assert set(by_id) == {"dup_leaf_cam1", "dup_leaf_cam2"}
    for rec in by_id.values():
        assert rec.group_id == "dup_leaf", "配对键契约:group_id 仍是叶子名"
        assert "20261023_154025" in str(rec.path), \
            f"保留的必须是先出现的 20261023_154025 下的文件,实际:{rec.path}"


def test_imread_checked_raises_on_corrupt_file(in_data_dir):
    """坏文件读图:cv2.imread 返回 None,imread_checked 抛含文件名的中文 ValueError。"""
    bad = in_data_dir("data/train/20261023_154025/20261023_154025_519359/cam1.jpg", 8, 8)
    bad.write_bytes(b"not a jpeg at all")
    with pytest.raises(ValueError, match="cam1.jpg"):
        imread_checked(bad)
