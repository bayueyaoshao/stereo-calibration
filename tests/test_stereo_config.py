"""stereo 配置段测试:合法加载全字段断言 + 非法值中文报错(参数化)。

体例照 tests/test_intrinsic_config.py:MINIMAL_YAML 全键显式(含 stereo 段,
不依赖隐式默认)+ _write_config 辅助函数 + parametrize 非法值逐例拦截。
注意 _mutate_stereo 只扫描 stereo 段(stereo 在模板末尾)——键名 enabled /
method / max_nfev 等在 intrinsic 段也有同名键,按键名全局替换第一处命中会
改错段,所以必须以 "stereo:" 行为界;段内 refine 排在 outlier 之前,键
"enabled" 命中的是 stereo.refine.enabled(布尔冒充开关)。
"""

from pathlib import Path

import pytest

from stereo_calib.io.config import load_config

# 最小合法配置:Phase 1/2 全部键 + Phase 3 stereo 段全部键显式写出,
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
stereo:
  refine:
    enabled: true
    method: "trf"
    max_nfev: 200
    ftol: 1.0e-10
    xtol: 1.0e-12
    gtol: 1.0e-10
  outlier:
    enabled: true
    rms_multiplier: 5.0
    max_drop_fraction: 0.2
    max_iterations: 3
    min_pairs: 10
  flip_ratio: 0.5
  median_spread_max_deg: 1.0
  distance_rms_max_mm: 5.0
  baseline_band_mm: [20, 2000]
  exclude_pairs: []
"""


def _write_config(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def _mutate_stereo(field: str, value: str) -> str:
    """把 stereo 段里某个键的值换成给定值(只扫描 "stereo:" 行之后的段落)。"""
    lines = MINIMAL_YAML.splitlines()
    start = lines.index("stereo:")
    out: list[str] = []
    replaced = False
    for i, line in enumerate(lines):
        key = line.strip().split(":")[0]
        if i > start and key == field and not replaced:
            indent = line[: len(line) - len(line.lstrip())]
            out.append(f"{indent}{field}: {value}")
            replaced = True
        else:
            out.append(line)
    assert replaced, f"字段 {field} 不在 stereo 段模板里"
    return "\n".join(out) + "\n"


def test_load_stereo_all_fields(tmp_path):
    """合法配置加载:stereo 段每个键都解析到对应字段(含嵌套 refine/outlier)。"""
    cfg = load_config(_write_config(tmp_path, MINIMAL_YAML))
    assert cfg.stereo.refine.enabled is True
    assert cfg.stereo.refine.method == "trf"
    assert cfg.stereo.refine.max_nfev == 200
    assert cfg.stereo.refine.ftol == pytest.approx(1.0e-10)
    assert cfg.stereo.refine.xtol == pytest.approx(1.0e-12)
    assert cfg.stereo.refine.gtol == pytest.approx(1.0e-10)
    assert cfg.stereo.outlier.enabled is True
    assert cfg.stereo.outlier.rms_multiplier == pytest.approx(5.0)
    assert cfg.stereo.outlier.max_drop_fraction == pytest.approx(0.2)
    assert cfg.stereo.outlier.max_iterations == 3
    assert cfg.stereo.outlier.min_pairs == 10
    assert cfg.stereo.flip_ratio == pytest.approx(0.5)
    assert cfg.stereo.median_spread_max_deg == pytest.approx(1.0)
    assert cfg.stereo.distance_rms_max_mm == pytest.approx(5.0)
    assert cfg.stereo.baseline_band_mm == (20.0, 2000.0)
    assert cfg.stereo.exclude_pairs == ()


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("rms_multiplier", "1.0"),        # 乘数必须 > 1.0(1 倍无判别力)
        ("max_drop_fraction", "0"),       # 剔除比例保险丝必须落在开区间 (0, 1)
        ("enabled", "1"),                 # 布尔冒充开关必须被拦(命中 stereo.refine.enabled)
        ("method", '"lm"'),               # 白名单外的方法('lm' 只支持稠密雅可比)
        ("max_nfev", "0"),                # 迭代上限必须 >= 1
        ("min_pairs", "1"),               # 幸存对数下限必须 >= 2(一对定不了相对位姿)
        ("flip_ratio", "0"),              # 消歧阈值必须 > 0
        ("flip_ratio", "1.5"),            # 消歧阈值超过 1.0 即"证据不足也翻",拒收
        ("median_spread_max_deg", "0"),   # 离散度防线上限必须 > 0
        ("baseline_band_mm", "[2000, 20]"),  # sanity 带反序 → 中文报错
        ("baseline_band_mm", "[20]"),     # sanity 带必须恰好两个数
        ("exclude_pairs", "pair_001"),    # 裸字符串不是 list,必须被拦
    ],
)
def test_invalid_values_raise_with_chinese_message(tmp_path, field, bad_value):
    with pytest.raises(ValueError, match="不合法"):
        load_config(_write_config(tmp_path, _mutate_stereo(field, bad_value)))


def test_missing_stereo_section_falls_back_to_defaults(tmp_path):
    """缺 stereo 段的旧配置(Phase 2 形态)不炸:整段走默认值(向后兼容)。"""
    no_stereo = MINIMAL_YAML.split("stereo:")[0]
    cfg = load_config(_write_config(tmp_path, no_stereo))
    assert cfg.stereo.refine.enabled is True
    assert cfg.stereo.outlier.min_pairs == 10
    assert cfg.stereo.flip_ratio == pytest.approx(0.5)
    assert cfg.stereo.median_spread_max_deg == pytest.approx(1.0)
    assert cfg.stereo.baseline_band_mm == (20.0, 2000.0)
    assert cfg.stereo.exclude_pairs == ()


# ===========================================================================
# 03-05:stereo.joint 段(联合 BA 配置)
# ===========================================================================

# joint 段追加在 MINIMAL_YAML 末尾(stereo 段之后;键全显式,不依赖隐式默认)
JOINT_YAML = MINIMAL_YAML + """\
  joint:
    enabled: true
    variant: "full_prior"
    prior_weight: 0.01
    spread_ratio_max: 5.0
"""


def _mutate_joint(field: str, value: str) -> str:
    """把 stereo.joint 段里某个键的值换成给定值(只扫描 "joint:" 行之后的段落)。"""
    lines = JOINT_YAML.splitlines()
    start = lines.index("  joint:")
    out: list[str] = []
    replaced = False
    for i, line in enumerate(lines):
        key = line.strip().split(":")[0]
        if i > start and key == field and not replaced:
            indent = line[: len(line) - len(line.lstrip())]
            out.append(f"{indent}{field}: {value}")
            replaced = True
        else:
            out.append(line)
    assert replaced, f"字段 {field} 不在 stereo.joint 段模板里"
    return "\n".join(out) + "\n"


def test_load_stereo_joint_all_fields(tmp_path):
    """合法 joint 段加载:每个键解析到对应字段(含白名单内变体)。"""
    cfg = load_config(_write_config(tmp_path, JOINT_YAML))
    assert cfg.stereo.joint.enabled is True
    assert cfg.stereo.joint.variant == "full_prior"
    assert cfg.stereo.joint.prior_weight == pytest.approx(0.01)
    assert cfg.stereo.joint.spread_ratio_max == pytest.approx(5.0)


def test_stereo_joint_defaults_when_absent(tmp_path):
    """缺 joint 段(03-05 之前的配置形态)不炸:整段走默认值(默认关,
    rigid 行为零改动的开关保证)。"""
    cfg = load_config(_write_config(tmp_path, MINIMAL_YAML))
    assert cfg.stereo.joint.enabled is False
    assert cfg.stereo.joint.variant == "fc_prior"
    assert cfg.stereo.joint.prior_weight == pytest.approx(1.0)
    assert cfg.stereo.joint.spread_ratio_max == pytest.approx(5.0)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("enabled", "1"),                 # 布尔冒充开关必须被拦
        ("variant", '"bogus"'),           # 白名单 {fc, fc_prior, full_prior} 外拒收
        ("variant", "[fc]"),              # 容器型值不可哈希,先判 str 再查白名单
        ("prior_weight", "0"),            # 先验权重必须 > 0
        ("prior_weight", "-1.0"),         # 负先验权重拒收
        ("spread_ratio_max", "1.0"),      # 比值上限必须 > 1.0(≤1 无判别力)
        ("spread_ratio_max", "0.5"),      # 同上
    ],
)
def test_joint_invalid_values_raise_with_chinese_message(tmp_path, field, bad_value):
    with pytest.raises(ValueError, match="不合法"):
        load_config(_write_config(tmp_path, _mutate_joint(field, bad_value)))
