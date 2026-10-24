"""内参模型充分性体检测试:配置解析 / 逐检查单测 / verdict 规则 / 合成端到端双向
/ 管线 fail-closed 接入。

体例照 tests/test_intrinsic_pipeline.py:合成路线用真值参数 + 多样位姿
cv2.projectPoints(z 600~1200mm、xy 铺到画面边缘,保证 r_norm 覆盖到 0.9)+
0.05px 高斯噪声。**双向端到端是本任务的核心验收**——rational 畸变合成数据
经产线 M2 协议(OpenCV 初值 + scipy 精化)拟合后,体检必须判「不可靠」
(设计目的就是检测已知欠拟合;真值高阶项取 k4=0.35 量级:k6 单项会被 M2
的 k3 在本画面半径内完全吸收,红灯不亮,故按 PLAN 预案加大);M2 一致畸变
合成数据必须判「可靠」。
"""

import json
import pathlib

import cv2
import numpy as np
import pytest
import typer

from stereo_calib.cli import main as cli_main
from stereo_calib.intrinsic.diagnostics import (
    CheckResult,
    _finalize,
    diagnostics_to_dict,
    run_battery,
)
from stereo_calib.io.config import (
    BoardSpec,
    IntrinsicDiagnosticsConfig,
    RefineConfig,
    load_config,
)
from stereo_calib.preprocess.artifacts import build_object_points

# ===========================================================================
# 配置段:diagnostics 节解析 + 类型守卫
# ===========================================================================

# 最小合法配置(Phase 1/2 全键 + diagnostics 节全键显式,不依赖隐式默认)
_CFG_YAML = """\
board: {board_type: "chessboard", cols: 11, rows: 8, square_size_mm: 25.0, unit: "mm"}
data: {train: "data/train", test: "data/test"}
output_root: "output"
cv2_threads: 0
preprocess:
  corners:
    classic_flags: [adaptive_thresh]
    sb_fallback_classic: true
    subpix: {window_granularity: frame, min_win: 3, max_iter: 30, epsilon: 0.01}
  copy_fail_originals: true
  visualization: {jpeg_quality: 90}
intrinsic:
  refine: {enabled: true, method: "trf", max_nfev: 300, ftol: 1.0e-10, xtol: 1.0e-12, gtol: 1.0e-10}
  outlier: {enabled: true, rms_multiplier: 5.0, max_drop_fraction: 0.2, max_iterations: 3, min_views: 10}
  rms_test_max: 1.0
  exclude_images: []
  diagnostics:
    enabled: true
    a0_max_ratio: 2.0
    a1_max_bin_mean_px: 0.02
    a3_max_fx_range_pct: 0.5
    a3_min_frames: 20
    a4_max_fx_jump_pct: 2.0
    min_red_count: 2
"""


def _write_config(tmp_path: pathlib.Path, content: str) -> pathlib.Path:
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_load_diagnostics_all_fields(tmp_path):
    """合法配置:diagnostics 段每个键都解析到对应字段(数值按 float 收口)。"""
    cfg = load_config(_write_config(tmp_path, _CFG_YAML)).intrinsic.diagnostics
    assert cfg.enabled is True
    assert cfg.a0_max_ratio == pytest.approx(2.0)
    assert cfg.a1_max_bin_mean_px == pytest.approx(0.02)
    assert cfg.a3_max_fx_range_pct == pytest.approx(0.5)
    assert cfg.a3_min_frames == 20
    assert cfg.a4_max_fx_jump_pct == pytest.approx(2.0)
    assert cfg.min_red_count == 2


def test_diagnostics_defaults_when_section_absent(tmp_path):
    """缺 diagnostics 节的旧配置不炸:整段走默认值(向后兼容,体检默认开)。"""
    no_diag = _CFG_YAML.split("  diagnostics:")[0]
    cfg = load_config(_write_config(tmp_path, no_diag)).intrinsic.diagnostics
    assert cfg.enabled is True
    assert cfg.a0_max_ratio == pytest.approx(2.0)
    assert cfg.min_red_count == 2


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("enabled", "1"),                  # 布尔冒充开关必须被拦
        ("a0_max_ratio", "1.0"),           # 比值上限必须 > 1.0(≤1 无判别力)
        ("a1_max_bin_mean_px", "0"),       # 桶均值上限必须 > 0
        ("a3_max_fx_range_pct", "-0.5"),   # 极差上限必须 > 0
        ("a3_min_frames", "true"),         # 布尔冒充 int 必须被拦
        ("a4_max_fx_jump_pct", "0"),       # 跳变上限必须 > 0
        ("min_red_count", "5"),            # 红检只有 4 个,>4 永远不可达
        ("min_red_count", "0"),            # 0 红即判不可靠 = 恒不可靠,无意义
    ],
)
def test_invalid_diagnostics_values_raise_chinese(tmp_path, field, bad_value):
    """非法值逐例拦截:中文 ValueError(与 WR-01 同风格),绝不静默吞掉。"""
    lines = _CFG_YAML.splitlines()
    out = []
    replaced = False
    for line in lines:
        key = line.strip().split(":")[0]
        if key == field and not replaced:
            indent = line[: len(line) - len(line.lstrip())]
            out.append(f"{indent}{field}: {bad_value}")
            replaced = True
        else:
            out.append(line)
    assert replaced, f"字段 {field} 不在配置模板里"
    with pytest.raises(ValueError, match="不合法"):
        load_config(_write_config(tmp_path, "\n".join(out) + "\n"))


# ===========================================================================
# 合成夹具:真值投影 + 多样位姿(z 600~1200mm、xy 铺到画面边缘)+ 0.05px 噪声
# ===========================================================================

_K_TRUE = np.array([[2037.0, 0.0, 1280.0],
                    [0.0, 2037.0, 720.0],
                    [0.0, 0.0, 1.0]])
_D5_TRUE = np.array([-0.40, 0.23, 1.1e-4, -3.8e-5, -0.08])  # M2 一致真值
# rational 真值(8 系布局 [k1,k2,p1,p2,k3,k4,k5,k6]):k4=0.35 量级保证高阶
# 结构在本画面半径内不可被 M2 的 3 径向项吸收(k6 单项会被 k3 完全吸收,见文件头)
_D8_RATIONAL = np.array([-0.3, 0.2, 1.0e-4, -1.0e-4, -0.05, 0.35, 0.0, 0.0])
_SIZE = (2560, 1440)
_OBJP = build_object_points(BoardSpec())
_CENTER = np.array([_SIZE[0] / 2.0, _SIZE[1] / 2.0])
_HALF_DIAG = float(np.hypot(*_CENTER))


def _diverse_poses(n_views: int, seed: int, d_truth: np.ndarray,
                   min_r_norm_frame: float = 0.0) -> list:
    """多样位姿采样:z 600~1200mm,xy 铺开,每帧全部角点在画面内(8px 边距)
    且最远角点 r_norm >= 0.5(保证体检的边缘桶有数据)。

    min_r_norm_frame > 0 时再要求每帧最近角点 r_norm >= 该值(A0 中心桶
    为空的反例构造用)。候选打满 20000 次仍凑不齐 → 断言失败(采样器坏了)。
    """
    rng = np.random.default_rng(seed)
    poses = []
    tries = 0
    while len(poses) < n_views and tries < 20000:
        tries += 1
        z = rng.uniform(600.0, 1200.0)
        tx_max = _SIZE[0] / 2 - _K_TRUE[0, 0] * 125.0 / z - 12.0
        ty_max = _SIZE[1] / 2 - _K_TRUE[1, 1] * 87.5 / z - 12.0
        tvec = np.array([rng.uniform(-tx_max, tx_max),
                         rng.uniform(-ty_max, ty_max), z])
        rvec = rng.normal(0.0, 0.3, 3)
        proj, _ = cv2.projectPoints(_OBJP.reshape(-1, 1, 3), rvec, tvec,
                                    _K_TRUE, d_truth.reshape(-1, 1))
        uv = proj.reshape(-1, 2)
        if uv[:, 0].min() < 8 or uv[:, 0].max() > _SIZE[0] - 8:
            continue
        if uv[:, 1].min() < 8 or uv[:, 1].max() > _SIZE[1] - 8:
            continue
        r_norm = np.linalg.norm(uv - _CENTER, axis=1) / _HALF_DIAG
        if r_norm.max() < 0.5 or r_norm.min() < min_r_norm_frame:
            continue
        poses.append(uv)
    assert len(poses) == n_views, (
        f"位姿采样器只凑到 {len(poses)}/{n_views} 帧(tries={tries}),夹具坏了")
    return poses


def _synth_uv(d_truth: np.ndarray, n_views: int, seed: int,
              min_r_norm_frame: float = 0.0) -> np.ndarray:
    """真值投影 + 0.05px 高斯噪声 → (V,88,2) float64 角点。"""
    uv = np.stack(_diverse_poses(n_views, seed, d_truth, min_r_norm_frame))
    noise = np.random.default_rng(seed + 999).normal(0.0, 0.05, uv.shape)
    return uv + noise


# ===========================================================================
# 逐检查单测:数值口径 / 阈值接线 / 数据不足与探针失败路径 / 输入防线
# ===========================================================================

# 体检消费的「产线解」直接用真值 K/D(M2 一致数据下真值即产线解;
# 「拟合后再体检」的产线全链路在端到端节覆盖)
_DEFAULT_CFG = IntrinsicDiagnosticsConfig()


def test_battery_green_on_m2_consistent_truth_and_input_readonly():
    """M2 一致真值 K/D:全部检查绿、verdict 可靠;输入数组绝不被改写。"""
    uv = _synth_uv(_D5_TRUE, 24, seed=11)
    uv_before = uv.copy()
    report = run_battery(uv, _OBJP, _K_TRUE, _D5_TRUE, _SIZE, _DEFAULT_CFG)
    assert report.verdict == "可靠" and report.reliable is True
    assert report.red_count == 0 and report.red_names == ()
    by_name = {c.name: c for c in report.checks}
    assert set(by_name) == {"A0", "A1", "A2", "A3", "A4"}
    # A2 仅报告:无阈值、恒不红(实测对 M2 欠拟合无分辨力,设计如此)
    assert by_name["A2"].threshold is None and by_name["A2"].red is False
    # 数值口径抽查:噪声底下 A0 比值应近 1(中心桶与全体的噪声同源)
    assert 0.5 < by_name["A0"].value < 2.0
    assert by_name["A1"].value < 0.01, "0.05px 噪声下桶均值应远低于 0.02 红线"
    assert np.array_equal(uv, uv_before), "体检只读数据:输入 uv 绝不许被改写"


def test_battery_m8_production_layout_battery_runs_green():
    """M8 产线布局(D 8 长):体检按长度分派正常运转,一致真值判可靠。

    真值即产线解(cv2 rational8 一致数据):A3 切片重标自动带 rational
    flag(与产线同阶,见 _refit_fx);A4 探针走「同阶独立重标一致性」
    语义,detail 显式标 same_order_rational8(与 M2 的升阶探针区分)。
    """
    uv = _synth_uv(_D8_RATIONAL, 24, seed=11)
    cfg = IntrinsicDiagnosticsConfig(a3_min_frames=5)  # 24 帧切 6 片各 ~4 帧须放低下限
    report = run_battery(uv, _OBJP, _K_TRUE, _D8_RATIONAL, _SIZE, cfg)
    assert report.verdict == "可靠", f"M8 一致真值不许误判不可靠:{report.red_names}"
    by_name = {c.name: c for c in report.checks}
    a4 = by_name["A4"]
    assert a4.detail["probe"] == "same_order_rational8", (
        "M8 产线解下 A4 必须标注同阶探针语义(与升阶探针区分)")
    assert a4.value is not None and np.isfinite(a4.value)
    assert np.isfinite(by_name["A3"].value), "a3_min_frames=5 时判定量应走数值路径"


def test_battery_threshold_wiring_all_four_red():
    """阈值接线:收紧四条红线(数据本身干净)→ 四检查全红、判不可靠。

    单检查的红灯逻辑只有「判定量 > 阈值」一条路,用收紧阈值逐项点亮,
    比构造四种病态数据更直接地锁定「阈值来自配置」这一契约。
    """
    uv = _synth_uv(_D5_TRUE, 24, seed=11)
    tight = IntrinsicDiagnosticsConfig(
        a0_max_ratio=0.5,            # 干净数据比值 ~1 > 0.5 → 红
        a1_max_bin_mean_px=1.0e-9,   # 桶均值 ~1e-3 > 1e-9 → 红
        a3_min_frames=5,             # 24 帧切成 6 片,每片 ~8 帧有效
        a3_max_fx_range_pct=1.0e-9,  # 切片极差 ~0.08% > 1e-9 → 红
        a4_max_fx_jump_pct=1.0e-9,   # 升阶跳变 ~0.005% > 1e-9 → 红
    )
    report = run_battery(uv, _OBJP, _K_TRUE, _D5_TRUE, _SIZE, tight)
    assert report.verdict == "不可靠" and report.reliable is False
    assert report.red_count == 4
    assert set(report.red_names) == {"A0", "A1", "A3", "A4"}


def test_battery_a3_insufficient_frames_notes_not_red():
    """A3 数据不足路径:单切片帧数下限抬到全集之上 → 全部切片跳过,
    有效切片 <4 → 判定量 None +「数据不足」note、不计红(fail-closed
    的可信度前提:红灯必须来自真实判定量,不许由故障间接撑出来)。"""
    uv = _synth_uv(_D5_TRUE, 24, seed=11)
    cfg = IntrinsicDiagnosticsConfig(a3_min_frames=100)
    report = run_battery(uv, _OBJP, _K_TRUE, _D5_TRUE, _SIZE, cfg)
    a3 = next(c for c in report.checks if c.name == "A3")
    assert a3.value is None and a3.red is False
    assert "数据不足" in a3.note
    assert report.verdict == "可靠", "A3 数据不足不许单独把 verdict 推红"


def test_battery_a3_numeric_path_counts_subsets():
    """A3 数值路径:a3_min_frames 放低 → 6 切片全有效,判定量有限且干净
    数据下远低于默认红线(切片重标的 fx 极差应接近噪声限)。"""
    uv = _synth_uv(_D5_TRUE, 24, seed=11)
    cfg = IntrinsicDiagnosticsConfig(a3_min_frames=5)
    report = run_battery(uv, _OBJP, _K_TRUE, _D5_TRUE, _SIZE, cfg)
    a3 = next(c for c in report.checks if c.name == "A3")
    assert a3.value is not None and np.isfinite(a3.value)
    assert a3.value < _DEFAULT_CFG.a3_max_fx_range_pct, "干净数据切片 fx 极差不许过默认红线"
    assert len(a3.detail) == 6, "左右/上下/内外圈 6 切片必须全部计数"
    assert all(v["n_frames"] >= 5 for v in a3.detail.values())


def test_battery_a4_probe_failure_notes_not_red(monkeypatch):
    """A4 探针失败路径:rational 重标抛 cv2.error → 判定量 None + note、
    不计红,其余检查不受影响(verdict 仍由真实判定量决定)。"""
    import stereo_calib.intrinsic.diagnostics as diag_mod

    real_fit = cv2.calibrateCameraExtended

    def flaky_fit(*args, **kwargs):
        if kwargs.get("flags", 0) & cv2.CALIB_RATIONAL_MODEL:
            raise cv2.error("测试注入:rational 探针拟合失败")
        return real_fit(*args, **kwargs)

    monkeypatch.setattr(diag_mod.cv2, "calibrateCameraExtended", flaky_fit)
    uv = _synth_uv(_D5_TRUE, 24, seed=11)
    report = run_battery(uv, _OBJP, _K_TRUE, _D5_TRUE, _SIZE, _DEFAULT_CFG)
    a4 = next(c for c in report.checks if c.name == "A4")
    assert a4.value is None and a4.red is False
    assert "探针拟合失败" in a4.note
    assert report.verdict == "可靠", "探针故障不许单独把 verdict 推红"
    a1 = next(c for c in report.checks if c.name == "A1")
    assert a1.value is not None, "其余检查必须照常算出判定量"


def test_battery_a0_empty_center_bucket_notes_not_red():
    """A0 中心桶为空的反例:全部帧的角点都在 r_norm>=0.22 之外 → 噪声底
    无从取,判定量 None + note、不计红(不硬算、不编数)。"""
    uv = _synth_uv(_D5_TRUE, 12, seed=23, min_r_norm_frame=0.22)
    r_norm = np.linalg.norm(uv - _CENTER, axis=-1) / _HALF_DIAG
    assert r_norm.min() >= 0.22, "夹具自检:反例数据必须真的没有中心桶角点"
    report = run_battery(uv, _OBJP, _K_TRUE, _D5_TRUE, _SIZE, _DEFAULT_CFG)
    a0 = next(c for c in report.checks if c.name == "A0")
    assert a0.value is None and a0.red is False
    assert "中心桶" in a0.note


@pytest.mark.parametrize(
    ("bad_input", "match"),
    [
        (lambda uv, objp: run_battery(uv[:, :, 0], objp, _K_TRUE, _D5_TRUE, _SIZE,
                                      _DEFAULT_CFG), "uv 形状"),
        (lambda uv, objp: run_battery(uv, objp[:, :2], _K_TRUE, _D5_TRUE, _SIZE,
                                      _DEFAULT_CFG), "objp 形状"),
        (lambda uv, objp: run_battery(uv, objp, _K_TRUE[:2, :], _D5_TRUE, _SIZE,
                                      _DEFAULT_CFG), "K 形状"),
        (lambda uv, objp: run_battery(uv, objp, _K_TRUE, _D5_TRUE[:4], _SIZE,
                                      _DEFAULT_CFG), "D 形状"),
        (lambda uv, objp: run_battery(uv, objp, _K_TRUE, np.zeros(6), _SIZE,
                                      _DEFAULT_CFG), "D 形状"),
        (lambda uv, objp: run_battery(uv, objp, _K_TRUE, _D5_TRUE, (2560,),
                                      _DEFAULT_CFG), "image_size"),
    ],
)
def test_battery_input_validation_chinese_valueerror(bad_input, match):
    """输入防线:形状/取值不合一律中文 ValueError,绝不带病计算。"""
    uv = _synth_uv(_D5_TRUE, 4, seed=5)
    with pytest.raises(ValueError, match=match):
        bad_input(uv, _OBJP)


def test_battery_disabled_returns_not_examined_without_compute():
    """开关关闭:返回「未体检」报告、不做任何计算(checks 为空),
    JSON dict 路径同步可用(enabled=False 原样留痕)。"""
    report = run_battery(np.zeros((2, 88, 2)), _OBJP, _K_TRUE, _D5_TRUE, _SIZE,
                         IntrinsicDiagnosticsConfig(enabled=False))
    assert report.verdict == "未体检" and report.reliable is None
    assert report.checks == () and report.red_count == 0
    d = diagnostics_to_dict(report)
    assert d["enabled"] is False and d["verdict"] == "未体检" and d["checks"] == []


# ===========================================================================
# verdict 规则与 JSON 序列化
# ===========================================================================

def _check(name: str, value: float, threshold: float, red: bool) -> CheckResult:
    return CheckResult(name=name, label=name, value=value, threshold=threshold,
                       red=red, detail={})


def test_verdict_rule_min_red_count_boundary():
    """判定规则边界:恰好 1 红(阈值 2)→ 可靠;恰好 2 红 → 不可靠;
    阈值收到 1 时 1 红即不可靠(规则完全由配置驱动)。"""
    ok_a0 = _check("A0", 1.0, 2.0, False)
    red_a1 = _check("A1", 0.5, 0.02, True)
    ok_a3 = _check("A3", 0.1, 0.5, False)
    ok_a4 = _check("A4", 0.1, 2.0, False)

    one_red = _finalize((ok_a0, red_a1, ok_a3, ok_a4), (), 2, True)
    assert one_red.reliable is True and one_red.verdict == "可靠"
    assert one_red.red_count == 1 and one_red.red_names == ("A1",)

    two_red = _finalize((ok_a0, red_a1, _check("A3", 5.0, 0.5, True), ok_a4), (), 2, True)
    assert two_red.reliable is False and two_red.verdict == "不可靠"
    assert two_red.red_names == ("A1", "A3")

    strict = _finalize((ok_a0, red_a1, ok_a3, ok_a4), (), 1, True)
    assert strict.reliable is False and strict.verdict == "不可靠"


def test_diagnostics_to_dict_strict_json_safe():
    """to_dict 产出必须过严格 JSON(allow_nan=False):NaN(空桶)转 null,
    数值全部原生类型——intrinsics.json 不许写出非严格 JSON。"""
    checks = (
        CheckResult(name="A0", label="残差能量比", value=1.0, threshold=2.0, red=False,
                    detail={"pooled_rms_px": np.float64(0.5), "center_floor_px": 0.5}),
        CheckResult(name="A1", label="残差径向结构", value=0.5, threshold=0.02, red=True,
                    detail={"bin_means_px": [0.1, float("nan")], "bin_counts": [5, 0]}),
        CheckResult(name="A3", label="切片 fx 稳定性", value=None, threshold=0.5, red=False),
    )
    report = _finalize(checks, ("全局 note",), 2, True)
    d = diagnostics_to_dict(report)
    text = json.dumps(d, ensure_ascii=False, allow_nan=False)  # NaN 存在即抛
    assert '"value": null' in text or '"value":null' in text  # A3 数据不足
    assert json.loads(text)["checks"][1]["detail"]["bin_means_px"] == [0.1, None]
    assert json.loads(text)["checks"][1]["detail"]["bin_counts"] == [5, 0]
    assert json.loads(text)["red_names"] == ["A1"]


# ===========================================================================
# 合成端到端双向(核心验收):产线 M2 协议拟合 → 体检
# ===========================================================================

def test_e2e_rational_truth_m2_fit_judged_unreliable():
    """① rational 畸变合成(48 帧多样位姿)→ 产线 M2 协议拟合 → 体检
    必须判「不可靠」且红灯 >= 2(本测试就是设计来检测已知欠拟合的;
    真值 fx=2037,M2 拟合会把它推高——与真实 cam1 同一种病)。"""
    from stereo_calib.intrinsic.models import intrinsics_matrix
    from stereo_calib.intrinsic.solve import solve_calibration

    uv = _synth_uv(_D8_RATIONAL, 48, seed=11)
    sol = solve_calibration(uv, _OBJP, _SIZE, RefineConfig())
    report = run_battery(uv, _OBJP, intrinsics_matrix(sol.params),
                         np.asarray(sol.params[4:9]), _SIZE, _DEFAULT_CFG)
    assert report.verdict == "不可靠" and report.reliable is False, (
        f"rational 真值必须被判不可靠(红灯 {report.red_names});体检坏了")
    assert report.red_count >= 2
    assert "A1" in report.red_names, "结构性欠拟合必在残差径向结构上留痕"
    assert sol.params[0] > 2037.0, (
        "M2 拟合的 fx 应高于真值(欠拟合的畸变被焦距吸收的病灶;本真值下 "
        "实测 +0.7% 左右,病灶强弱随畸变形状变化,不设更紧的下限)")


def test_e2e_m2_truth_m2_fit_judged_reliable():
    """② M2 一致畸变合成 → 产线 M2 协议拟合 → 体检必须判「可靠」
    (绿灯不许是 A3 数据不足撑出来的:48 帧下 6 切片全部有效)。"""
    from stereo_calib.intrinsic.models import intrinsics_matrix
    from stereo_calib.intrinsic.solve import solve_calibration

    uv = _synth_uv(_D5_TRUE, 48, seed=11)
    sol = solve_calibration(uv, _OBJP, _SIZE, RefineConfig())
    report = run_battery(uv, _OBJP, intrinsics_matrix(sol.params),
                         np.asarray(sol.params[4:9]), _SIZE, _DEFAULT_CFG)
    assert report.verdict == "可靠" and report.red_count == 0, (
        f"M2 一致数据不许误报(红灯 {report.red_names})")
    a3 = next(c for c in report.checks if c.name == "A3")
    assert a3.value is not None, "48 帧下 A3 必须真的算了(绿灯不许靠跳过撑出来)"
    assert abs(sol.params[0] - 2037.0) / 2037.0 < 1e-3, "M2 一致数据应恢复真值 fx"


# ===========================================================================
# 管线 fail-closed 接入:体检不可靠 → 产物留痕 + 中文 RuntimeError
# ===========================================================================

def _pipeline_yaml(diagnostics_enabled: bool) -> str:
    """管线级测试配置:refine/outlier 关(体检与二者无交互,关掉省时,
    OpenCV 初值解同样带 M2 病灶);test 集 4 帧 → min_views 放到 2。"""
    return f"""\
board: {{board_type: "chessboard", cols: 11, rows: 8, square_size_mm: 25.0, unit: "mm"}}
data: {{train: "data/train", test: "data/test"}}
output_root: "output"
cv2_threads: 0
preprocess:
  corners:
    classic_flags: [adaptive_thresh]
    sb_fallback_classic: true
    subpix: {{window_granularity: frame, min_win: 3, max_iter: 30, epsilon: 0.01}}
  copy_fail_originals: false
  visualization: {{jpeg_quality: 90}}
intrinsic:
  refine: {{enabled: false, method: "trf", max_nfev: 300, ftol: 1.0e-10, xtol: 1.0e-12, gtol: 1.0e-10}}
  outlier: {{enabled: false, rms_multiplier: 5.0, max_drop_fraction: 0.2, max_iterations: 3, min_views: 2}}
  rms_test_max: 1.0
  exclude_images: []
  diagnostics:
    enabled: {str(diagnostics_enabled).lower()}
    a0_max_ratio: 2.0
    a1_max_bin_mean_px: 0.02
    a3_max_fx_range_pct: 0.5
    a3_min_frames: 20
    a4_max_fx_jump_pct: 2.0
    min_red_count: 2
"""


def _write_tiny_jpg(path: pathlib.Path, width: int = 640, height: int = 480) -> pathlib.Path:
    """最小噪声 JPEG(仅需"可读出分辨率"派生 image_size)。"""
    rng = np.random.default_rng(7)
    img = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), img), f"测试夹具写图失败:{path}"
    return path


def _build_pipeline_tree(tmp_path: pathlib.Path, d_truth: np.ndarray) -> None:
    """搭最小运行树:train 48 帧 rational/M2 合成角点 + test 4 帧,两路相机。"""
    img = _write_tiny_jpg(tmp_path / "img" / "cam.jpg", _SIZE[0], _SIZE[1])
    for cam in (1, 2):
        for split, n, seed in (("train", 48, 10 + cam), ("test", 4, 60 + cam)):
            uv = _synth_uv(d_truth, n, seed)
            ids = [f"{split}{k:03d}_cam{cam}" for k in range(n)]
            npz = tmp_path / "output" / split / f"cam{cam}_{split}.npz"
            npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                npz,
                image_ids=np.asarray(ids, dtype=np.str_),
                paths=np.asarray([str(img)] * n, dtype=np.str_),
                uv=uv,
                objp=_OBJP,
                source=np.asarray(["sb"] * n, dtype="<U16"),
                detected=np.ones(n, dtype=bool),
                sq_px=np.full(n, 18.0, dtype=np.float64),
                win_size=np.full(n, 5.0, dtype=np.float64),
            )
    (tmp_path / "config.yaml").write_text(_pipeline_yaml(True), encoding="utf-8")


def _run_pipeline(tmp_path: pathlib.Path) -> None:
    from stereo_calib.intrinsic.pipeline import run_intrinsic

    run_intrinsic(load_config(tmp_path / "config.yaml"))


def test_pipeline_fail_closed_on_unreliable(tmp_path, monkeypatch):
    """fail-closed 全链:rational 合成数据进产线管线 → 体检双路判不可靠 →
    run_intrinsic 抛中文 RuntimeError(cam1/cam2 红灯在案),且 intrinsics.json
    与 report.md 已落盘留痕(JSON diagnostics 键 + 报告 FAIL ⚠ 体检节)。"""
    _build_pipeline_tree(tmp_path, _D8_RATIONAL)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="内参模型充分性体检未通过") as ei:
        _run_pipeline(tmp_path)
    assert "cam1" in str(ei.value) and "cam2" in str(ei.value)
    assert "禁止消费该内参进入双目标定" in str(ei.value)

    root = tmp_path / "output" / "intrinsic"
    for cam in (1, 2):
        payload = json.loads((root / f"cam{cam}" / "intrinsics.json").read_text(encoding="utf-8"))
        d = payload["diagnostics"]
        assert d["enabled"] is True and d["verdict"] == "不可靠"
        assert d["reliable"] is False and d["red_count"] >= 2
        assert len(d["checks"]) == 5, "五检查数值必须全部落进 JSON 留痕"
    report = (root / "report.md").read_text(encoding="utf-8")
    assert "## 模型充分性体检" in report
    assert "FAIL ⚠ 不可靠" in report
    assert "判定规则" in report


def test_pipeline_diagnostics_disabled_skips_and_reports(tmp_path, monkeypatch):
    """开关关闭路径:体检跳过、管线正常完成不抛;JSON diagnostics 键如实
    记 enabled=false,报告注明「体检未启用」。"""
    _build_pipeline_tree(tmp_path, _D8_RATIONAL)
    (tmp_path / "config.yaml").write_text(_pipeline_yaml(False), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    _run_pipeline(tmp_path)  # 不抛即第一契约:体检关闭时旧流程行为零改动

    for cam in (1, 2):
        payload = json.loads(
            (tmp_path / "output" / "intrinsic" / f"cam{cam}" / "intrinsics.json")
            .read_text(encoding="utf-8"))
        assert payload["diagnostics"]["enabled"] is False
        assert payload["diagnostics"]["verdict"] == "未体检"
    report = (tmp_path / "output" / "intrinsic" / "report.md").read_text(encoding="utf-8")
    assert "体检未启用" in report
    assert "FAIL ⚠" not in report


def test_pipeline_reliable_data_completes_without_raise(tmp_path, monkeypatch):
    """绿灯全链:M2 一致合成数据进产线管线 → 体检判可靠 → 不抛、JSON
    diagnostics.verdict=可靠、报告 PASS(体检不许误伤好数据)。"""
    _build_pipeline_tree(tmp_path, _D5_TRUE)
    monkeypatch.chdir(tmp_path)
    _run_pipeline(tmp_path)
    for cam in (1, 2):
        payload = json.loads(
            (tmp_path / "output" / "intrinsic" / f"cam{cam}" / "intrinsics.json")
            .read_text(encoding="utf-8"))
        assert payload["diagnostics"]["verdict"] == "可靠"
        assert payload["diagnostics"]["red_count"] == 0
    report = (tmp_path / "output" / "intrinsic" / "report.md").read_text(encoding="utf-8")
    assert "PASS 可靠" in report


def test_cli_intrinsic_diagnostics_fail_chinese_exit_one(tmp_path, monkeypatch):
    """CLI 收口:体检不可靠的 RuntimeError 经命令异常网 → 中文一句 +
    typer.Exit code 1(与双目防线同风格:产物留痕 + 非零退出)。"""
    _build_pipeline_tree(tmp_path, _D8_RATIONAL)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli_main, "_config_path", pathlib.Path("config.yaml"))
    with pytest.raises(typer.Exit) as ei:
        cli_main.intrinsic()
    assert ei.value.exit_code == 1
    # 产物已落盘留痕(退出码非零不销毁证据)
    assert (tmp_path / "output" / "intrinsic" / "report.md").is_file()
    payload = json.loads(
        (tmp_path / "output" / "intrinsic" / "cam1" / "intrinsics.json")
        .read_text(encoding="utf-8"))
    assert payload["diagnostics"]["verdict"] == "不可靠"
