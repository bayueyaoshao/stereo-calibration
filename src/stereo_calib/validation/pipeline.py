"""模块4 主流程:测量精度验证编排(装载冻结产物 → 检测复用 → 双模式测量 →
区域分析 → 闸门判定 → 落盘)。

流程(run_validation 是本模块唯一编排入口,零数学——只调用 validation.measure
/ observe 与既有 evaluate/models 入口,投影/三角化/去畸变/真值通通不住本文件):

  1. 运行前清理 output_root/validation(D-11 固定覆盖 + 防残留,照 stereo 体例);
  2. 装载冻结产物:两路 intrinsics.json(distortion_model 感知)+
     extrinsics_stereo.json 的 rvec/t → R=cv2.Rodrigues;
  3. 扫描 validation.data_dir(<组目录>/cam{1,2}.jpg)→ 既有 detect_one
     检测链(SB 主路线 + classic 兜底 + canonicalize,与训练同一口径);
     有效对数 < min_pairs → 中文 ValueError fail-closed;
  4. 逐对双模式测量(冻结 = 健康判据;逐对 PnP = 归因诊断);单对失败
     NaN + 中文备注不中断整批;全部失败 → 中文 ValueError(不静默出空报告);
  5. 判定闸门:|mean err_frozen| ≤ distance_bias_max_mm 且 RMS ≤
     distance_rms_max_mm → PASS,否则 FAIL;逐对模式不设闸门;
  6. 落盘:validation.json(先落盘留痕)→ report.md;FAIL 结论由 CLI
     转成退出码 1(fail-closed,与体检防线同风格)。

数据安全(CLAUDE.md data 只读):全程只读 data_dir 下的图像(imread),全部
写路径仅从 cfg.output_root 派生;validation.data_dir 与 data/output_root 的
相等/包含配置已在加载层被拒(Phase 4 锁定),清理只可能命中产物目录。
"""

from __future__ import annotations

import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import typer

from stereo_calib.intrinsic.models import intrinsic_names
from stereo_calib.io.config import StereoCalibConfig
from stereo_calib.preprocess.artifacts import build_object_points
from stereo_calib.stereo.evaluate import true_board_distance
from stereo_calib.stereo.observation import IntrinsicsView, load_intrinsics
from stereo_calib.validation.artifacts import write_report_md, write_validation_json
from stereo_calib.validation.measure import (
    PairMeasurement,
    analyze_regions,
    evaluate_gates,
    load_frozen_extrinsics,
    measure_pair,
)
from stereo_calib.validation.observe import detect_val_pairs, scan_val_groups


def _fx_of(intr: IntrinsicsView) -> float:
    """按名取 fx(禁魔数下标;两模型布局里 fx 都有名可查)。"""
    named = dict(zip(intrinsic_names(intr.model), (float(v) for v in intr.params)))
    return named["fx"]


def run_validation(cfg: StereoCalibConfig) -> dict:
    """模块4 主流程;返回 validation.json 同构的 summary(dict,CLI 据此定退出码)。"""
    val_cfg = cfg.validation
    val_root = Path(cfg.output_root) / "validation"

    # ---- 运行前清理(D-11 固定覆盖,照 stereo 体例)----
    # 只删 output_root/validation 一个产物子目录;load_config 已拒绝
    # validation.data_dir 与 data/output_root 相等/互相包含的配置,这里的
    # rmtree 只可能命中产物目录,不可能触及 data/。
    shutil.rmtree(val_root, ignore_errors=True)

    # ---- 装载冻结产物(绝不重拟合的唯一含义:以下参数只读)----
    intrinsic_root = Path(cfg.output_root) / "intrinsic"
    intr1 = load_intrinsics(intrinsic_root / "cam1" / "intrinsics.json")
    intr2 = load_intrinsics(intrinsic_root / "cam2" / "intrinsics.json")
    ext = load_frozen_extrinsics(
        Path(cfg.output_root) / "stereo" / "extrinsics_stereo.json")
    objp = build_object_points(cfg.board)
    theory = true_board_distance(objp)
    typer.echo(f"测量精度验证:冻结内参 cam1 fx={_fx_of(intr1):.4f} / "
               f"cam2 fx={_fx_of(intr2):.4f},baseline ‖t‖={ext.baseline_mm:.2f} mm;"
               f"对角距离理论值 {theory:.4f} mm")

    # ---- 扫描 + 既有检测链(与训练同一代码路径)----
    groups, scan_warnings = scan_val_groups(val_cfg.data_dir)
    for w in scan_warnings:
        typer.echo(f"警告:{w}")
    typer.echo(f"  扫到 {len(groups)} 组验证对({val_cfg.data_dir}),"
               "走 preprocess 检测链(SB 主路线 + classic 兜底)……")
    pairs, fail_notes = detect_val_pairs(
        groups, cfg.board, cfg.preprocess.corners, cfg.cv2_threads,
        cfg.preprocess.max_workers)
    n_valid = len(pairs)
    if n_valid < val_cfg.min_pairs:
        raise ValueError(
            f"有效验证对数不足:扫到 {len(groups)} 组、两路均检出 {n_valid} 对,"
            f"低于 validation.min_pairs={val_cfg.min_pairs}——没数据不硬算"
            f"(fail-closed)。失败详情:{fail_notes or '无'};"
            f"请检查 validation.data_dir 下的图像对与 config.yaml。")
    typer.echo(f"  检出有效对 {n_valid} 组(失败单帧 {len(fail_notes)})")

    # ---- 逐对双模式测量(单对失败 NaN + 备注,不中断整批)----
    flip_ratio = cfg.stereo.flip_ratio
    measurements: list[PairMeasurement] = []
    pair_rows: list[dict] = []
    for vp in pairs:
        pm = measure_pair(vp.group_id, vp.uv1, vp.uv2, objp, intr1, intr2,
                          ext, flip_ratio)
        measurements.append(pm)
        pair_rows.append({
            "group_id": pm.group_id,
            "err_frozen_mm": pm.err_frozen_mm,
            "err_pair_mm": pm.err_pair_mm,
            "measured_frozen_mm": pm.measured_frozen_mm,
            "measured_pair_mm": pm.measured_pair_mm,
            "center_u_px": pm.center_u_px,
            "center_v_px": pm.center_v_px,
            "reproj_rms_px": pm.reproj_rms_px,
            "flipped": [bool(pm.flipped[0]), bool(pm.flipped[1])],
            "route1": vp.route1,
            "route2": vp.route2,
            "note": pm.note,
        })

    # ---- 判定闸门(冻结模式;逐对不设闸门)----
    finite_errs = [r["err_frozen_mm"] for r in pair_rows
                   if np.isfinite(r["err_frozen_mm"])]
    gates = evaluate_gates(finite_errs, val_cfg.distance_bias_max_mm,
                           val_cfg.distance_rms_max_mm)
    verdict = "PASS" if (gates["bias_pass"] and gates["rms_pass"]) else "FAIL"
    if verdict == "PASS":
        verdict_meaning = ("冻结模式测量偏差在红线内:当前内外参与标定时刻一致,"
                           "未见系统漂移。")
    else:
        verdict_meaning = ("冻结模式测量偏差超红线:参数已相对标定时刻漂移或该批"
                           "数据异常——按 report.md 判读矩阵归因(冻结大+逐对小="
                           "外参漂移,重标 stereo;两者都大=内参/数据问题,跑体检)。")

    regions = analyze_regions(measurements, val_cfg.region_grid,
                              intr1.image_size)

    # ---- 组装契约 payload(字段集与 artifacts 断言一一对应)----
    rep = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_dir": str(val_cfg.data_dir),
        "n_groups_found": len(groups),
        "n_pairs_valid": n_valid,
        "n_pairs_failed": len(fail_notes),
        "pair_fail_notes": dict(sorted(fail_notes.items())),
        "true_distance_mm": theory,
        "fingerprint": {
            "distortion_model_cam1": intr1.model,
            "distortion_model_cam2": intr2.model,
            "fx_cam1": _fx_of(intr1),
            "fx_cam2": _fx_of(intr2),
            "baseline_mm": ext.baseline_mm,
            "rvec": [float(v) for v in ext.rvec],
            "board_cols": cfg.board.cols,
            "board_rows": cfg.board.rows,
            "square_size_mm": cfg.board.square_size_mm,
        },
        "gates": gates,
        "verdict": verdict,
        "verdict_meaning": verdict_meaning,
        "regions_enabled": bool(val_cfg.region_grid[0] > 0 and val_cfg.region_grid[1] > 0),
        "region_grid": [int(val_cfg.region_grid[0]), int(val_cfg.region_grid[1])],
        "regions": regions,
        "pairs": pair_rows,
        "config_snapshot": {
            "validation": asdict(val_cfg),
            "board": asdict(cfg.board),
            "stereo_flip_ratio": cfg.stereo.flip_ratio,
            "distortion_model": cfg.intrinsic.distortion_model,
        },
    }

    # ---- 落盘:JSON 先留痕,报告随后(FAIL 也不丢证据)----
    val_root.mkdir(parents=True, exist_ok=True)
    json_path = write_validation_json(val_root / "validation.json", rep)
    report_path = write_report_md(val_root / "report.md", rep)

    # ---- 中文汇总 ----
    typer.echo("测量精度验证完成,产物已写盘:")
    typer.echo(f"  冻结偏差:mean |{gates['mean_err_frozen_mm']:.4f}| mm"
               f"(红线 {gates['distance_bias_max_mm']} )、"
               f"RMS {gates['rms_err_frozen_mm']:.4f} mm"
               f"(红线 {gates['distance_rms_max_mm']} mm)")
    finite_pair = [abs(r["err_pair_mm"]) for r in pair_rows
                   if np.isfinite(r["err_pair_mm"])]
    if finite_pair:
        typer.echo(f"  逐对偏差(归因诊断,不设闸门):|平均| "
                   f"{float(np.mean(finite_pair)):.4f} mm")
    typer.echo(f"  判定:{verdict}"
               + ("" if verdict == "PASS" else "(FAIL → 退出码 1,fail-closed)"))
    typer.echo("  全部产物(绝对路径):")
    for p in (json_path, report_path):
        typer.echo(f"    {Path(p).resolve()}")
    return rep
