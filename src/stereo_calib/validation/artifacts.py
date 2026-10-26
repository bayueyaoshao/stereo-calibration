"""模块4 产物落盘:validation.json(契约新文件,字段自本文件定死)+
report.md(中文产线报告,判读矩阵写给操作工,零模板引擎)。

产物布局(output_root/validation/ 之下;D-11 固定覆盖,运行前清目录):

- validation.json  逐对行(group_id/双模式偏差/板中心/重投影参考/翻转)+
  区域分桶统计 + 判定闸门与 verdict + 参数指纹(内参 fx、baseline ‖t‖,
  便于跨时间对比参数本身)+ 配置快照;字段集由 _VALIDATION_JSON_KEYS
  断言锁定(契约漂移当场炸,与 extrinsics_stereo.json 同一防线体例)。
- report.md        单份中文报告:判读矩阵说明(冻结大+逐对小=外参漂移→
  重标 stereo;两者都大=内参退化或数据质量→跑体检;两者都小=健康)+
  逐对表 + 区域表 + 参数指纹 + verdict + 能力边界说明(绝对尺度盲)。

本模块只做落盘与排版,零数学(一切数值由 measure/pipeline 算好传入)。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

# validation.json 顶层字段全集(模块4 契约;增删必须同步本断言与规划文档)
_VALIDATION_JSON_KEYS = (
    "generated_at", "data_dir", "n_groups_found", "n_pairs_valid",
    "n_pairs_failed", "pair_fail_notes", "true_distance_mm",
    "fingerprint", "gates", "verdict", "verdict_meaning",
    "regions_enabled", "region_grid", "regions", "pairs", "config_snapshot",
)


def write_validation_json(path: Path | str, rep: dict) -> Path:
    """写模块4 契约文件(顶层字段集断言锁定,防契约漂移不报错)。"""
    if set(rep) != set(_VALIDATION_JSON_KEYS):
        raise AssertionError(
            f"validation.json 字段集漂移:实际 {sorted(rep)},"
            f"契约 {sorted(_VALIDATION_JSON_KEYS)}。"
            f"这是测量精度验证的唯一落盘接口,增删字段必须同步本文件契约断言与规划文档。")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2, default=list)
    return path


def _fmt(v) -> str:
    """报告数值显示:4 位小数;非有限(失败对)显式标 NaN。"""
    fv = float(v)
    return f"{fv:.4f}" if np.isfinite(fv) else "NaN"


# 判读矩阵(产线操作工的操作指引;冻结=健康判据,逐对=归因诊断)
_INTERPRET_MATRIX_ROWS = (
    ("冻结偏差大、逐对偏差小", "外参漂移(振动/碰撞把双目相对位姿挪了)",
     "重跑 stereo-calib stereo 重新标外参,内参不用动"),
    ("冻结偏差大、逐对偏差大", "内参退化,或该批图像数据质量有问题",
     "先确认板规格与拍摄正常,再跑 stereo-calib intrinsic 重标内参(含体检)"),
    ("冻结偏差小、逐对偏差小", "系统健康,无漂移", "无需处理,按周期复测即可"),
    ("冻结偏差小、逐对偏差大", "理论上不应出现(逐对自洽性反而更差)",
     "疑似个别对角点质量差:查该对重投影 RMS 与角点标注图"),
)


def _scale_flags(rep: dict) -> tuple[bool, bool]:
    """给判读矩阵自动选行的粗标尺:(冻结是否大, 逐对是否大)。

    冻结「大」= 任一红线未过;逐对「大」= 逐对偏差 RMS 超过 bias 红线
    (逐对模式吸收外参漂移,它的大小就是数据+内参的自洽噪声底);
    逐对全失败时退化为 (冻结判定, False)。
    """
    frozen_bad = not (rep["gates"]["bias_pass"] and rep["gates"]["rms_pass"])
    errs = [p["err_pair_mm"] for p in rep["pairs"] if np.isfinite(p["err_pair_mm"])]
    if not errs:
        return frozen_bad, False
    pair_rms = math.sqrt(sum(e * e for e in errs) / len(errs))
    return frozen_bad, pair_rms > rep["gates"]["distance_bias_max_mm"]


def write_report_md(path: Path | str, rep: dict) -> Path:
    """写模块4 中文报告(判读矩阵 + 逐对表 + 区域表 + 指纹 + verdict)。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    g = rep["gates"]
    is_fail = rep["verdict"] == "FAIL"
    lines: list[str] = []

    lines.append("# 测量精度验证报告(系统漂移体检)")
    lines.append("")
    lines.append(f"- 生成时间:{rep['generated_at']}")
    lines.append(f"- 验证数据:{rep['data_dir']}(有效对 {rep['n_pairs_valid']}"
                 f" / 扫到组 {rep['n_groups_found']},失败单帧 {rep['n_pairs_failed']})")
    lines.append(f"- 判定红线:|平均偏差| ≤ {g['distance_bias_max_mm']} mm、"
                 f"RMS ≤ {g['distance_rms_max_mm']} mm(冻结模式)")
    lines.append("")
    if is_fail:
        lines.append("**verdict:FAIL(红灯——冻结模式测量偏差超阈值;本结论同时以"
                     "退出码 1 上报,fail-closed。按下方判读矩阵定位与处理)**")
    else:
        lines.append("**verdict:PASS(绿灯——冻结外参下的测量偏差在阈值内,系统无漂移)**")
    lines.append("")

    # ---- 判定数值 ----
    lines.append("## 判定数值(冻结模式 = 健康判据)")
    lines.append("")
    lines.append("| 指标 | 实测 | 红线 | 判定 |")
    lines.append("|---|---|---|---|")
    lines.append(f"| \\|平均偏差\\| (mm) | {abs(g['mean_err_frozen_mm']):.4f} | "
                 f"{g['distance_bias_max_mm']} | "
                 f"{'PASS' if g['bias_pass'] else 'FAIL'} |")
    lines.append(f"| 偏差 RMS (mm) | {g['rms_err_frozen_mm']:.4f} | "
                 f"{g['distance_rms_max_mm']} | "
                 f"{'PASS' if g['rms_pass'] else 'FAIL'} |")
    lines.append(f"| 参与判定的对数 | {g['n_pairs_used']} | — | — |")
    lines.append("")
    lines.append(f"- 对角距离理论值:{rep['true_distance_mm']:.4f} mm"
                 f"(√((行−1)²+(列−1)²) × 方格边长,从配置棋盘导出,角点 0↔87 口径)")
    if rep["pair_fail_notes"]:
        lines.append(f"- 失败单帧备注:{rep['pair_fail_notes']}")
    lines.append("")

    # ---- 判读矩阵(产线操作指引)----
    lines.append("## 判读矩阵(怎么读这份报告)")
    lines.append("")
    lines.append("| 现象 | 最可能的原因 | 操作 |")
    lines.append("|---|---|---|")
    for row in _INTERPRET_MATRIX_ROWS:
        lines.append(f"| {row[0]} | {row[1]} | {row[2]} |")
    lines.append("")
    lines.append("- **冻结模式**(存档 R/t 直接三角化):参数一动它就变大——漂移体检的判据;")
    lines.append("- **逐对模式**(该对两路自己 PnP 再测):自洽量,外参漂移会被它吸收——用于归因。")
    frozen_bad, pair_bad = _scale_flags(rep)
    pair_vals = [abs(p["err_pair_mm"]) for p in rep["pairs"]
                 if np.isfinite(p["err_pair_mm"])]
    pair_mean = float(np.mean(pair_vals)) if pair_vals else float("nan")
    lines.append(f"- 本次实测读数:冻结偏差{'大(超红线)' if frozen_bad else '小(红线内)'}、"
                 f"逐对偏差{'大' if pair_bad else '小'}"
                 f"(逐对 |平均| = {_fmt(pair_mean)} mm)——对照上表取对应行的操作。")
    lines.append("")

    # ---- 逐对表 ----
    lines.append("## 逐对结果")
    lines.append("")
    lines.append("| group_id | 冻结偏差 (mm) | 逐对偏差 (mm) | 板中心 u,v (px) | "
                 "重投影 RMS (px) | 翻转 | 备注 |")
    lines.append("|---|---|---|---|---|---|---|")
    for p in rep["pairs"]:
        lines.append(
            f"| {p['group_id']} | {_fmt(p['err_frozen_mm'])} | {_fmt(p['err_pair_mm'])} | "
            f"{_fmt(p['center_u_px'])}, {_fmt(p['center_v_px'])} | "
            f"{_fmt(p['reproj_rms_px'])} | {'是' if any(p['flipped']) else '否'} | "
            f"{p['note']} |")
    lines.append("")

    # ---- 区域表 ----
    if rep["regions_enabled"]:
        grid_rows, grid_cols = rep["region_grid"]
        lines.append(f"## 区域精度分布(cam1 视场 {grid_rows}×{grid_cols} 分桶,"
                     "冻结模式 |偏差|)")
        lines.append("")
        if rep["n_pairs_valid"] < grid_rows * grid_cols:
            lines.append(f"**样本不足(有效对 {rep['n_pairs_valid']} 少于桶数 "
                         f"{grid_rows * grid_cols})——下表只如实罗列单对落桶,"
                         "区域级结论待补样本。**")
            lines.append("")
        lines.append("| 区域(行,列) | 样本数 | \\|偏差\\| 中位 (mm) | "
                     "\\|偏差\\| 最大 (mm) | 备注 |")
        lines.append("|---|---|---|---|---|")
        for r in rep["regions"]:
            med = "—" if r["median_abs_err_mm"] is None else f"{r['median_abs_err_mm']:.4f}"
            mx = "—" if r["max_abs_err_mm"] is None else f"{r['max_abs_err_mm']:.4f}"
            lines.append(f"| {r['region']} | {r['n']} | {med} | {mx} | {r['note']} |")
        lines.append("")

    # ---- 参数指纹(跨时间对比参数本身)----
    fp = rep["fingerprint"]
    lines.append("## 参数指纹(冻结产物的参数本身,便于跨时间对比)")
    lines.append("")
    lines.append("| 项 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| distortion_model | cam1 {fp['distortion_model_cam1']} / "
                 f"cam2 {fp['distortion_model_cam2']} |")
    lines.append(f"| fx (px) | cam1 {fp['fx_cam1']:.4f} / cam2 {fp['fx_cam2']:.4f} |")
    lines.append(f"| baseline ‖t‖ (mm) | {fp['baseline_mm']:.4f} |")
    lines.append(f"| rvec(存档)| [{', '.join(f'{v: .6f}' for v in fp['rvec'])}] |")
    lines.append(f"| 棋盘 | {fp['board_cols']}×{fp['board_rows']},"
                 f"方格 {fp['square_size_mm']} mm |")
    lines.append("")

    # ---- 能力边界 ----
    lines.append("## 能力边界(必读)")
    lines.append("")
    lines.append("- 本工具测的是「**与标定时刻的一致性**」,对**绝对尺度盲**:"
                 "理论距离继承棋盘方格边长假设(当前 25mm,建议卡尺复核);")
    lines.append("- 漂移监测看的是相对变化,不受尺度假设影响;换板/换方格规格后,"
                 "历史读数不可直接对比;")
    lines.append("- 单次验证只对当批图像负责;要判断趋势,请按周期复测并对比"
                 "各次 validation.json 的参数指纹与偏差。")
    lines.append("")

    # ---- 附录:配置快照 ----
    lines.append("## 附录:配置快照(config_snapshot)")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(rep["config_snapshot"], ensure_ascii=False, indent=2,
                            default=list))
    lines.append("```")
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path
