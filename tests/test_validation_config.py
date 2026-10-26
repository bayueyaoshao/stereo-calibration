"""validation 配置段测试:默认值、合法解析、逐字段中文守卫、数据安全互斥。

体例照 test_io_config.py:最小合法配置模板 + 字段变异;非法用例必须中文
报错(「不合法」);数据安全用例的三种重叠形态(相等 / 对方在验证目录内 /
验证目录在对方内)都要在加载层被拒。
"""

from pathlib import Path

import pytest

from stereo_calib.io.config import ValidationConfig, load_config

MINIMAL_YAML = """\
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

VALIDATION_SECTION = """\
validation:
  data_dir: "data/val"
  distance_bias_max_mm: 1.0
  distance_rms_max_mm: 2.0
  region_grid: [3, 3]
  min_pairs: 1
"""


def _write_config(tmp_path: Path, validation_section: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(MINIMAL_YAML + validation_section, encoding="utf-8")
    return path


def test_missing_validation_section_falls_back_to_defaults(tmp_path):
    """段缺省 = 全默认(向后兼容:老 config.yaml 不必为新模块补键)。"""
    cfg = load_config(_write_config(tmp_path, ""))
    assert cfg.validation == ValidationConfig()
    assert cfg.validation.data_dir == "data/val"
    assert cfg.validation.distance_bias_max_mm == 1.0
    assert cfg.validation.distance_rms_max_mm == 2.0
    assert cfg.validation.region_grid == (3, 3)
    assert cfg.validation.min_pairs == 1


def test_valid_validation_section_parses_all_fields(tmp_path):
    cfg = load_config(_write_config(tmp_path, VALIDATION_SECTION))
    assert cfg.validation.data_dir == "data/val"
    assert cfg.validation.distance_bias_max_mm == pytest.approx(1.0)
    assert cfg.validation.distance_rms_max_mm == pytest.approx(2.0)
    assert cfg.validation.region_grid == (3, 3)
    assert cfg.validation.min_pairs == 1


def test_region_grid_zero_zero_means_disabled(tmp_path):
    """[0, 0] = 关闭区域分析,是合法值(与 [3, 0] 这类半开组合相对)。"""
    section = VALIDATION_SECTION.replace("region_grid: [3, 3]",
                                         "region_grid: [0, 0]")
    cfg = load_config(_write_config(tmp_path, section))
    assert cfg.validation.region_grid == (0, 0)


@pytest.mark.parametrize(
    "bad_section",
    [
        'validation: {data_dir: ""}\n',                 # 空路径
        'validation: {data_dir: 3}\n',                  # 非文字
        "validation: {distance_bias_max_mm: 0}\n",      # 红线必须 > 0
        "validation: {distance_bias_max_mm: -1.0}\n",
        "validation: {distance_bias_max_mm: true}\n",   # 布尔冒充数
        "validation: {distance_rms_max_mm: 0}\n",
        "validation: {region_grid: [3]}\n",             # 长度必须恰 2
        "validation: {region_grid: [3, 3, 3]}\n",
        "validation: {region_grid: [true, 3]}\n",       # 布尔冒充 int
        "validation: {region_grid: [3.0, 3]}\n",        # 浮点冒充 int
        "validation: {region_grid: [-1, 3]}\n",         # 负桶数
        "validation: {region_grid: [3, 0]}\n",          # 半开组合退化成条带
        "validation: {min_pairs: 0}\n",                 # 0 对等于没验证
        "validation: {min_pairs: 1.5}\n",
        "validation: {min_pairs: true}\n",
    ],
)
def test_invalid_validation_values_rejected(tmp_path, bad_section):
    with pytest.raises(ValueError, match="不合法"):
        load_config(_write_config(tmp_path, bad_section))


@pytest.mark.parametrize(
    ("val_dir", "desc"),
    [
        ("data/train", "与 data.train 相等"),
        ("data/test", "与 data.test 相等"),
        ("output", "与 output_root 相等"),
        ("data", "验证目录是 data.train 的父目录"),
        ("data/train/inner", "验证目录位于 data.train 之内"),
        ("output/validation", "验证目录位于 output_root 之内(会被运行前清理删掉)"),
    ],
)
def test_validation_data_dir_overlaps_protected_dirs_rejected(
        tmp_path, monkeypatch, val_dir, desc):
    """验证目录与 data.train/test/output_root 相等或互相包含,加载层当场拒绝。"""
    monkeypatch.chdir(tmp_path)  # 相对路径按进程 cwd 解析,与运行目录无关
    section = VALIDATION_SECTION.replace('data_dir: "data/val"',
                                         f'data_dir: "{val_dir}"')
    with pytest.raises(ValueError, match="validation.data_dir"):
        load_config(_write_config(tmp_path, section))


def test_validation_data_dir_disjoint_path_accepted(tmp_path, monkeypatch):
    """不相交的路径正常通过(数据安全防线只拦重叠,不误伤常规配置)。"""
    monkeypatch.chdir(tmp_path)
    cfg = load_config(_write_config(tmp_path, VALIDATION_SECTION))
    assert cfg.validation.data_dir == "data/val"
