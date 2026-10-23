"""io/config.py 配置层测试:合法加载、非法值中文报错、output_root/data 互斥校验。

非法用例的报错必须带「不合法」或配置文件路径(中文一句,不吐堆栈);
数据安全用例的三种重叠形态(相等 / data 在 output 内 / output 在 data 内)
消息必须含实际解析路径——操作员要靠它定位是哪一段配错了。
"""

from pathlib import Path

import pytest

from stereo_calib.io.config import load_config

# 最小合法配置:Phase 1 全部键显式写出,加载结果不依赖隐式默认值
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


def _write_config(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def _mutate(field: str, value: str) -> str:
    """在最小合法配置里把某个键的值换成给定值(其余部分保持不变)。"""
    out = []
    replaced = False
    for line in MINIMAL_YAML.splitlines():
        key = line.strip().split(":")[0]
        if key == field and not replaced:
            indent = line[: len(line) - len(line.lstrip())]
            out.append(f"{indent}{field}: {value}")
            replaced = True
        else:
            out.append(line)
    assert replaced, f"字段 {field} 不在最小配置模板里"
    return "\n".join(out) + "\n"


def test_load_minimal_config_all_fields(tmp_path):
    """合法配置加载:每个键都解析到对应字段(含嵌套 corners/subpix/visualization)。"""
    cfg = load_config(_write_config(tmp_path, MINIMAL_YAML))
    assert cfg.board.cols == 11
    assert cfg.board.rows == 8
    assert cfg.board.square_size_mm == 25.0
    assert cfg.board.unit == "mm"
    assert cfg.data.train == "data/train"
    assert cfg.data.test == "data/test"
    assert cfg.output_root == "output"
    assert cfg.cv2_threads == 0
    assert cfg.preprocess.max_workers == 12
    assert cfg.preprocess.corners.classic_flags == ("adaptive_thresh", "normalize_image")
    assert cfg.preprocess.corners.sb_flags == 0
    assert cfg.preprocess.corners.sb_fallback_classic is True
    assert cfg.preprocess.corners.subpix.window_granularity == "frame"
    assert cfg.preprocess.corners.subpix.min_win == 3
    assert cfg.preprocess.corners.subpix.max_iter == 30
    assert cfg.preprocess.corners.subpix.epsilon == pytest.approx(0.01)
    assert cfg.preprocess.copy_fail_originals is True
    assert cfg.preprocess.jpeg_quality == 90


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("max_workers", "0"),                                    # 并行数下限
        ("square_size_mm", "0"),                                 # 方格边长必须为正
        ("classic_flags", "[adaptive_thresh, bogus_flag]"),      # 白名单外的选项名
        ("max_workers", "true"),                                 # 布尔冒充 int 必须被拦
    ],
)
def test_invalid_values_raise_with_chinese_message(tmp_path, field, bad_value):
    with pytest.raises(ValueError, match="不合法"):
        load_config(_write_config(tmp_path, _mutate(field, bad_value)))


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(ValueError, match="配置文件不存在"):
        load_config(tmp_path / "nope.yaml")


def test_yaml_syntax_error_raises_chinese(tmp_path):
    bad = tmp_path / "config.yaml"
    bad.write_text("board: [unclosed", encoding="utf-8")
    with pytest.raises(ValueError, match="yaml 语法有误"):
        load_config(bad)


@pytest.mark.parametrize(
    ("output_root", "desc"),
    [
        ("data/train", "与 data.train 相等"),
        (".", "data 目录位于 output_root 之内(清理输出会删数据)"),
        ("data/train/output", "output_root 位于 data.train 之内(产物写进数据)"),
    ],
)
def test_output_root_overlapping_data_rejected(tmp_path, monkeypatch, output_root, desc):
    """三种重叠形态都在加载层被拒,消息含实际解析路径(数据安全防线)。"""
    # 相对路径按进程 cwd 解析——chdir 到 tmp_path,让断言与运行目录无关
    monkeypatch.chdir(tmp_path)
    content = _mutate("output_root", f'"{output_root}"')
    with pytest.raises(ValueError) as excinfo:
        load_config(_write_config(tmp_path, content))
    msg = str(excinfo.value)
    assert "不得相等或互相包含" in msg, desc
    assert "output_root=" in msg, desc
    # 消息必须含实际解析出的绝对路径(操作员靠它定位配错的段)
    assert str(tmp_path.resolve()) in msg, desc


def test_default_output_root_form_is_legal(tmp_path):
    """默认形态(output_root: "output" 与 data/ 平级)合法加载。"""
    cfg = load_config(_write_config(tmp_path, MINIMAL_YAML))
    assert cfg.output_root == "output"
