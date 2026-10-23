"""intrinsic 配置段测试:合法加载全字段断言 + 非法值中文报错(参数化)。

体例照 tests/test_io_config.py:MINIMAL_YAML 全键显式(含 intrinsic 段,
不依赖隐式默认)+ _write_config/_mutate 辅助函数 + parametrize 非法值逐例
拦截。注意 _mutate 按键名替换第一处命中——YAML 里 refine 排在 outlier
之前,所以键 "enabled" 命中的是 intrinsic.refine.enabled(布尔冒充开关)。
"""

from pathlib import Path

import pytest

from stereo_calib.io.config import load_config

# 最小合法配置:Phase 1 全部键 + Phase 2 intrinsic 段全部键显式写出,
# 加载结果不依赖隐式默认值
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
intrinsic:
  refine:
    enabled: true
    method: "trf"
    max_nfev: 300
    ftol: 1.0e-10
    xtol: 1.0e-12
    gtol: 1.0e-10
  outlier:
    enabled: true
    rms_multiplier: 5.0
    max_drop_fraction: 0.2
    max_iterations: 3
    min_views: 10
  rms_test_max: 1.0
  exclude_images: []
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


def test_load_intrinsic_all_fields(tmp_path):
    """合法配置加载:intrinsic 段每个键都解析到对应字段(含嵌套 refine/outlier)。"""
    cfg = load_config(_write_config(tmp_path, MINIMAL_YAML))
    assert cfg.intrinsic.refine.enabled is True
    assert cfg.intrinsic.refine.method == "trf"
    assert cfg.intrinsic.refine.max_nfev == 300
    assert cfg.intrinsic.refine.ftol == pytest.approx(1.0e-10)
    assert cfg.intrinsic.refine.xtol == pytest.approx(1.0e-12)
    assert cfg.intrinsic.refine.gtol == pytest.approx(1.0e-10)
    assert cfg.intrinsic.outlier.enabled is True
    assert cfg.intrinsic.outlier.rms_multiplier == pytest.approx(5.0)
    assert cfg.intrinsic.outlier.max_drop_fraction == pytest.approx(0.2)
    assert cfg.intrinsic.outlier.max_iterations == 3
    assert cfg.intrinsic.outlier.min_views == 10
    assert cfg.intrinsic.rms_test_max == pytest.approx(1.0)
    assert cfg.intrinsic.exclude_images == ()


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("rms_multiplier", "1.0"),        # 乘数必须 > 1.0(1 倍无判别力)
        ("max_drop_fraction", "0"),       # 剔除比例保险丝必须落在开区间 (0, 1)
        ("enabled", "1"),                 # 布尔冒充开关必须被拦(命中 refine.enabled)
        ("method", '"lm"'),               # 白名单外的方法('lm' 只支持稠密雅可比)
        ("ftol", "0"),                    # 收敛容差必须 > 0
        ("max_iterations", "true"),       # 布尔冒充 int 必须被拦
        ("exclude_images", "cam1_001"),   # 裸字符串不是 list,必须被拦
    ],
)
def test_invalid_values_raise_with_chinese_message(tmp_path, field, bad_value):
    with pytest.raises(ValueError, match="不合法"):
        load_config(_write_config(tmp_path, _mutate(field, bad_value)))


def test_missing_intrinsic_section_falls_back_to_defaults(tmp_path):
    """缺 intrinsic 段的旧配置(Phase 1 形态)不炸:整段走默认值(向后兼容)。"""
    no_intrinsic = MINIMAL_YAML.split("intrinsic:")[0]
    cfg = load_config(_write_config(tmp_path, no_intrinsic))
    assert cfg.intrinsic.refine.enabled is True
    assert cfg.intrinsic.outlier.rms_multiplier == pytest.approx(5.0)
    assert cfg.intrinsic.exclude_images == ()
