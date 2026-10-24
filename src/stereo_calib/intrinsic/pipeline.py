"""模块2 编排:两路相机 内参标定 → train 离群自动剔除(全程留痕)→ test
只估位姿评分 → 五类产物落盘 → 中文汇总(OVERNIGHT AUTONOMY 机制落地)。

流程(run_intrinsic 是本模块唯一编排入口,零数学——只调用 02-01/02-02
的 observation / solve / scoring / metrics 入口、diagnostics 体检入口与
artifacts 落盘,投影与畸变公式只存在于 models.py,本文件一个公式都不出现):

运行前清理 output_root/intrinsic → 每路相机串行:
  1. 装载 train/test 幸存视图(load_views:检出过滤 + 手动剔除清单);
  2. 离群剔除循环(仅 train;剔除全程用 OpenCV 初值解——精化贵,只做
     幸存集定稿后的一次):每轮求解 → float64 回代逐帧 RMS(唯一判据尺,
     OpenCV 自己的逐帧表是 float32 口径,不读)→ 阈值 = rms_multiplier ×
     幸存集中位数 → 超阈剔除;min_views / max_drop_fraction /
     max_iterations 三保险丝,触发如实记入报告,绝不静默;
  3. 幸存集定稿:solve_calibration(精化开关由配置生效)→ 回代指标;
  4. 模型充分性体检(diagnostics.run_battery,2026-10-24 接入):标量
     RMS 看不出的畸变模型欠拟合由 A0/A1/A3/A4 揪出,数值/判定进
     intrinsics.json 的 diagnostics 键与 report.md 体检节;
  5. test 逐帧 score_pose_only(内参当尺子只估位姿;单帧失败 NaN + 中文
     备注不中断;全部失败属环境异常,中文报错不静默产出 NaN 报告);
  6. 落盘:intrinsics.json / 残差 npz×2 / 外参 npz / 逐帧 CSV。
两路完成后写单份 report.md(两路对比 + 体检 + 离群留痕)→ 中文汇总 →
**fail-closed 闸门**:任一路体检判「不可靠」抛中文 RuntimeError(产物
已落盘留痕),由 CLI 异常网收口为「内参标定失败」退出码 1。

数据安全(T-02-07):全部写路径仅从 cfg.output_root 派生;rmtree 目标仅
output_root/intrinsic 一个产物子目录。安全前提(不重复实现):load_config
已在加载层拒绝 output_root 与 data 目录相等/互相包含的配置(Phase 1
锁定),这里的 rmtree 只可能命中产物目录,不可能触及 data/。

剔除可审计(T-02-08):每个剔除帧的 image_id / rms / 轮次 / 阈值进
intrinsics.json 的 dropped_frames 与 report.md 离群表;test 帧永不剔除
(考卷不可改),高 RMS 只在 CSV note 与报告里标记。
"""

from __future__ import annotations

import shutil
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import typer

from stereo_calib.intrinsic.artifacts import (
    write_extrinsics_npz,
    write_intrinsics_json,
    write_per_frame_csv,
    write_report_md,
    write_residuals_npz,
)
from stereo_calib.intrinsic.diagnostics import diagnostics_to_dict, run_battery
from stereo_calib.intrinsic.metrics import compute_metrics
from stereo_calib.intrinsic.models import intrinsic_names, intrinsics_matrix
from stereo_calib.intrinsic.observation import load_views
from stereo_calib.intrinsic.scoring import score_pose_only
from stereo_calib.intrinsic.solve import solve_calibration
from stereo_calib.io.config import RefineConfig, StereoCalibConfig

_CAMS = (1, 2)


def _drop_outliers(cfg: StereoCalibConfig, train) -> tuple[list[dict], list[int], str]:
    """train 集离群剔除循环(Pattern 6):返回 (剔除留痕清单, 剔除行号, 循环说明)。

    每轮对幸存视图重新求解(OpenCV 初值解,精化不进循环)→ float64 回代
    逐帧 RMS → 阈值 = rms_multiplier × 幸存集中位数(中位数基准稳健,
    不受离群自身拖动)→ 超阈剔除。保险丝(绝不静默,触发原因进返回的
    循环说明并写入报告):
    - 剔除后幸存数将低于 max(min_views,(1-max_drop_fraction)×初始帧数)
      → 本轮停止,已剔的保留;
    - 打满 max_iterations 轮仍有超阈 → 停止(中位数随剔除漂移未收敛的
      信号,如实报告而非静默)。
    开关关闭(outlier.enabled=false)时一次都不剔,说明照常留痕。
    """
    outlier = cfg.intrinsic.outlier
    dropped: list[dict] = []
    dropped_idx: list[int] = []
    if not outlier.enabled:
        return dropped, dropped_idx, "自动剔除开关关闭(intrinsic.outlier.enabled: false),未做剔除"

    n_initial = len(train.image_ids)
    survived = list(range(n_initial))
    no_refine = RefineConfig(enabled=False)
    model = cfg.intrinsic.distortion_model
    for round_i in range(1, outlier.max_iterations + 1):
        res = solve_calibration(train.uv[survived], train.objp, train.image_size,
                                no_refine, model)
        m = compute_metrics(train.objp, train.uv[survived], res.params, res.rvecs, res.tvecs)
        med = float(np.median(m.per_view_rms))
        threshold = outlier.rms_multiplier * med
        bad = np.where(m.per_view_rms > threshold)[0]
        typer.echo(f"    第 {round_i} 轮剔除检查:幸存 {len(survived)} 帧,中位 RMS {med:.4f}px,"
                   f"阈值 {threshold:.4f}px,超阈 {int(bad.size)} 帧")
        if bad.size == 0:
            return dropped, dropped_idx, (
                f"第 {round_i} 轮无超阈帧,剔除循环收敛(共剔除 {len(dropped)} 帧)")
        floor = max(outlier.min_views, (1.0 - outlier.max_drop_fraction) * n_initial)
        if len(survived) - int(bad.size) < floor:
            return dropped, dropped_idx, (
                f"第 {round_i} 轮触发保险丝:本轮超阈 {int(bad.size)} 帧,剔除后幸存 "
                f"{len(survived) - int(bad.size)} 将低于下限 {floor:.1f}"
                f"(min_views={outlier.min_views},max_drop_fraction={outlier.max_drop_fraction}),"
                f"停止剔除并如实留痕(共剔除 {len(dropped)} 帧)")
        for j in bad:
            j = int(j)
            orig = survived[j]
            dropped.append({"image_id": train.image_ids[orig],
                            "rms": float(m.per_view_rms[j]),
                            "round": round_i, "threshold": float(threshold)})
            dropped_idx.append(orig)
        survived = [i for i in survived if i not in set(dropped_idx)]
    return dropped, dropped_idx, (
        f"剔除循环打满 max_iterations={outlier.max_iterations} 轮仍有超阈帧"
        f"(中位数随剔除漂移未收敛的信号,请检查数据),共剔除 {len(dropped)} 帧")


def _score_test_set(test, params) -> tuple[float, np.ndarray, np.ndarray, list[str]]:
    """test 集逐帧只估位姿评分:返回 (pooled RMS, 残差, 逐帧 RMS, 逐帧备注)。

    pooled 口径与 train 同一把尺:成功帧全部平方残差均值开根(不是逐帧
    RMS 的简写平均)。单帧失败 → 残差行 NaN + 中文备注,不中断整批;
    全部失败 → 中文 ValueError(考卷全糊属环境异常,不静默产出 NaN 报告,
    T-02-09)。
    """
    n_test = len(test.image_ids)
    n_points = test.objp.shape[0]
    residuals = np.full((n_test, n_points, 2), np.nan)
    per_view_rms = np.full(n_test, np.nan)
    notes: list[str] = []
    sq_chunks: list[np.ndarray] = []
    for i in range(n_test):
        score = score_pose_only(test.uv[i], test.objp, params)
        notes.append(score.note)
        if score.residuals is not None:
            residuals[i] = score.residuals
            per_view_rms[i] = score.rms
            sq_chunks.append(np.sum(score.residuals * score.residuals, axis=1))
    if not sq_chunks:
        raise ValueError(
            f"test 集全部 {n_test} 帧评分失败(考卷全糊,属环境异常而非数据波动):"
            f"内参无法解释任何一张没参与标定的照片,多半是角点 npz 与标定输入"
            f"不是同一批数据,或图像分辨率/棋盘规格错位。逐帧备注:{notes}。"
            f"请检查 Phase 1 产物与 config.yaml 后重跑。")
    pooled = float(np.sqrt(np.concatenate(sq_chunks).mean()))
    return pooled, residuals, per_view_rms, notes


def _calibrate_one_cam(cfg: StereoCalibConfig, cam: int,
                       intrinsic_root: Path) -> tuple[dict, list[Path]]:
    """单路相机全流程:装载 → 剔除 → 定稿 → test 评分 → 落盘。

    返回 (报告汇总 dict, 产物路径清单)。行序契约全程锁在装载层定死的
    image_ids 顺序上:剔除留痕、逐帧 RMS、CSV、npz 全按它对齐,绝不重排。
    """
    train_npz = Path(cfg.output_root) / "train" / f"cam{cam}_train.npz"
    test_npz = Path(cfg.output_root) / "test" / f"cam{cam}_test.npz"
    typer.echo(f"cam{cam}:装载 train={train_npz}")
    train = load_views(cfg.board, train_npz, cfg.intrinsic.exclude_images,
                       cfg.intrinsic.outlier.min_views)
    typer.echo(f"cam{cam}:装载 test={test_npz}")
    test = load_views(cfg.board, test_npz, cfg.intrinsic.exclude_images,
                      cfg.intrinsic.outlier.min_views)
    typer.echo(f"cam{cam}:幸存视图 train {len(train.image_ids)} 帧 / "
               f"test {len(test.image_ids)} 帧(已过检出过滤与手动剔除清单)")
    n_initial = len(train.image_ids)

    # ---- 离群剔除(仅 train;test 帧永不剔除,考卷不可改)----
    typer.echo(f"cam{cam}:train 离群剔除循环(multiplier={cfg.intrinsic.outlier.rms_multiplier})……")
    dropped, dropped_idx, loop_note = _drop_outliers(cfg, train)
    drop_by_idx = dict(zip(dropped_idx, dropped))

    # ---- 幸存集定稿(剔除循环全程 OpenCV 初值解,精化只做这一次)----
    # 畸变模型按配置(M2/M8)贯穿剔除循环与定稿求解(2026-10-24 增补)
    model = cfg.intrinsic.distortion_model
    keep = [i for i in range(n_initial) if i not in drop_by_idx]
    uvs_train = train.uv[keep]
    refine_state = "开" if cfg.intrinsic.refine.enabled else "关"
    typer.echo(f"cam{cam}:幸存 {len(keep)} 帧定稿求解(模型 {model},精化:{refine_state})……")
    final = solve_calibration(uvs_train, train.objp, train.image_size,
                              cfg.intrinsic.refine, model)
    m_final = compute_metrics(train.objp, uvs_train, final.params, final.rvecs, final.tvecs)
    typer.echo(f"cam{cam}:train 幸存集回代 RMS {m_final.rms_overall:.4f}px(solver={final.solver})")

    # ---- 模型充分性体检(fail-closed 闸门,2026-10-24 接入)----
    # 标量 RMS 看不出的畸变模型欠拟合(如 fx 被推高 11%)由 A0/A1/A3/A4
    # 揪出;数值与判定进 intrinsics.json 的 diagnostics 键(契约只增)与
    # report.md 体检节。verdict 在 run_intrinsic 收口(两路齐后再判)。
    # D 传产线解全尾部(5 长 M2 / 8 长 M8,体检内部按长度分派)。
    diag_report = run_battery(uvs_train, train.objp,
                              intrinsics_matrix(final.params),
                              np.asarray(final.params[4:], dtype=np.float64),
                              train.image_size, cfg.intrinsic.diagnostics)
    diagnostics = diagnostics_to_dict(diag_report)
    for chk in diag_report.checks:
        value = "数据不足" if chk.value is None else f"{chk.value:.4f}"
        state = "红灯" if chk.red else "通过"
        typer.echo(f"    体检 {chk.name} {chk.label}:{value}"
                   f"{'(阈值 ' + format(chk.threshold, 'g') + ')' if chk.threshold is not None else ''}"
                   f" → {state}")
    typer.echo(f"cam{cam}:体检 verdict={diag_report.verdict}"
               f"(红灯 {diag_report.red_count} 项:{','.join(diag_report.red_names) or '无'})")

    # ---- test 评分(内参当尺子,只估位姿)----
    typer.echo(f"cam{cam}:test {len(test.image_ids)} 帧只估位姿评分……")
    rms_test_overall, test_residuals, test_per_view_rms, test_notes = \
        _score_test_set(test, final.params)
    typer.echo(f"cam{cam}:test pooled RMS {rms_test_overall:.4f}px")

    # ---- 逐帧行组装(原 npz 行序;剔除帧 rms 用其被剔那轮的记录值)----
    rows: list[dict] = []
    survivor_rms = {orig: float(m_final.per_view_rms[k]) for k, orig in enumerate(keep)}
    for i, image_id in enumerate(train.image_ids):
        if i in drop_by_idx:
            d = drop_by_idx[i]
            rows.append({"image_id": image_id, "split": "train",
                         "source": train.source[i], "rms": f"{d['rms']:.4f}",
                         "dropped": "True",
                         "note": f"第{d['round']}轮剔除:RMS {d['rms']:.3f} 超阈值 {d['threshold']:.3f}"})
        else:
            rows.append({"image_id": image_id, "split": "train",
                         "source": train.source[i], "rms": f"{survivor_rms[i]:.4f}",
                         "dropped": "False", "note": ""})
    for i, image_id in enumerate(test.image_ids):
        rms = float(test_per_view_rms[i])
        rows.append({"image_id": image_id, "split": "test", "source": test.source[i],
                     "rms": f"{rms:.4f}" if np.isfinite(rms) else "",
                     "dropped": "False",  # test 帧恒不剔除,只标记
                     "note": test_notes[i]})

    # ---- 五类产物落盘(全部路径从 output_root 派生,T-02-07)----
    cam_dir = intrinsic_root / f"cam{cam}"
    kept_ids = [train.image_ids[i] for i in keep]
    json_path = write_intrinsics_json(
        cam_dir / "intrinsics.json", cam=cam, image_size=train.image_size,
        params=final.params, K=intrinsics_matrix(final.params), D=final.params[4:],
        solver=final.solver, refine_enabled=cfg.intrinsic.refine.enabled,
        rms_train_overall=m_final.rms_overall, rms_test_overall=rms_test_overall,
        n_views_train=len(keep), n_views_test=len(test.image_ids),
        dropped_frames=dropped, solver_meta=final.solver_meta,
        board=cfg.board, config_snapshot={"intrinsic": asdict(cfg.intrinsic)},
        diagnostics=diagnostics, distortion_model=model)
    train_res_path = write_residuals_npz(
        cam_dir / "train_residuals.npz", kept_ids, m_final.residuals, m_final.per_view_rms)
    test_res_path = write_residuals_npz(
        cam_dir / "test_residuals.npz", test.image_ids, test_residuals, test_per_view_rms)
    extr_path = write_extrinsics_npz(
        cam_dir / "extrinsics.npz", kept_ids, final.rvecs, final.tvecs)
    csv_path = write_per_frame_csv(intrinsic_root / f"per_frame_cam{cam}.csv", rows)

    summary = {
        "cam": cam,
        "distortion_model": model,
        "params": {name: float(v) for name, v in zip(intrinsic_names(model), final.params)},
        "solver": final.solver,
        "refine_enabled": bool(cfg.intrinsic.refine.enabled),
        "rms_train_overall": m_final.rms_overall,
        "rms_test_overall": rms_test_overall,
        "n_views_train": len(keep),
        "n_views_train_initial": n_initial,
        "n_views_test": len(test.image_ids),
        "dropped_frames": dropped,
        "outlier_loop": loop_note,
        "train_per_view_rms": [float(v) for v in m_final.per_view_rms],
        "test_per_view_rms": [float(v) for v in test_per_view_rms],
        "diagnostics": diagnostics,
    }
    artifacts = [json_path, train_res_path, test_res_path, extr_path, csv_path]
    return summary, artifacts


def run_intrinsic(cfg: StereoCalibConfig) -> None:
    """模块2 主流程:两路相机串行标定 + train 离群自动剔除 + test 评分 + 落盘。

    单路失败收口:某一路的求解/评分抛出数值路径异常(ValueError/
    ArithmeticError/RuntimeError/AssertionError/cv2.error,含精化期
    models.project 的 Z>0 数据防卫)时,该路降级为失败记录,另一路继续
    跑完并正常落盘,report.md 如实留痕——一路的退化帧绝不炸掉整批双路
    运行(评分器"单帧失败不中断整批"契约在编排层的对应兜底)。
    """
    intrinsic_root = Path(cfg.output_root) / "intrinsic"

    # ---- 运行前清理(D-11 固定覆盖 + 防残留)----
    # 只删 output_root/intrinsic 一个产物子目录,删后随落盘按需重建;上次
    # 运行的 JSON/npz/CSV/报告残留不会误导人工复核。output_root 本身与其他
    # 未知内容不动。数据安全前提(不重复实现):load_config 已在加载层拒绝
    # output_root 与 data 目录相等/互相包含的配置(Phase 1 锁定),这里的
    # rmtree 只可能命中 output_root 之下的产物目录,不可能触及 data/。
    shutil.rmtree(intrinsic_root, ignore_errors=True)

    typer.echo(f"开始内参标定:cam1、cam2 串行(剔除 multiplier={cfg.intrinsic.outlier.rms_multiplier},"
               f"保险丝 min_views={cfg.intrinsic.outlier.min_views} / "
               f"max_drop_fraction={cfg.intrinsic.outlier.max_drop_fraction} / "
               f"max_iterations={cfg.intrinsic.outlier.max_iterations})……")
    summaries: list[dict] = []
    artifacts: list[Path] = []
    for cam in _CAMS:
        try:
            summary, cam_artifacts = _calibrate_one_cam(cfg, cam, intrinsic_root)
        except (ValueError, ArithmeticError, RuntimeError, AssertionError, cv2.error) as e:
            # 单路失败降级:记录带 failed 标记的最小汇总(其余字段给中性
            # 空值,报告照常渲染),另一路继续——绝不向上抛中断整批。
            typer.echo(f"cam{cam}:该路标定失败,降级为失败记录(另一路继续):"
                       f"{type(e).__name__}——{e}")
            summaries.append({
                "cam": cam, "failed": True,
                "error": f"{type(e).__name__}——{e}",
                "dropped_frames": [],
                "outlier_loop": "该路求解失败,剔除循环未完成(失败原因见失败留痕)",
                "train_per_view_rms": [], "test_per_view_rms": [],
            })
            continue
        summaries.append(summary)
        artifacts.extend(cam_artifacts)

    report_path = write_report_md(intrinsic_root / "report.md", summaries,
                                  cfg.intrinsic.rms_test_max,
                                  {"intrinsic": asdict(cfg.intrinsic)})
    artifacts.append(report_path)

    # ---- 中文汇总:逐路幸存/剔除、train/test RMS、PASS/FAIL、产物路径 ----
    typer.echo("内参标定完成,产物已写盘。逐路指标:")
    for s in summaries:
        if s.get("failed"):
            typer.echo(f"  cam{s['cam']}:失败(该路无产物,失败原因见 report.md 失败留痕)"
                       f"——{s['error']}")
            continue
        ok = np.isfinite(s["rms_test_overall"]) and s["rms_test_overall"] <= cfg.intrinsic.rms_test_max
        verdict = "PASS" if ok else "FAIL"
        typer.echo(f"  cam{s['cam']}:train RMS {s['rms_train_overall']:.4f}px,"
                   f"test RMS {s['rms_test_overall']:.4f}px(阈值 {cfg.intrinsic.rms_test_max},{verdict}),"
                   f"train 幸存 {s['n_views_train']}/{s['n_views_train_initial']} 帧"
                   f"(剔除 {len(s['dropped_frames'])} 帧),test {s['n_views_test']} 帧")
        d = s.get("diagnostics") or {}
        if d.get("enabled"):
            typer.echo(f"    体检:{d['verdict']}(红灯 {d['red_count']} 项:"
                       f"{','.join(d['red_names']) or '无'})")
        else:
            typer.echo("    体检:未启用(内参可靠性未判定)")
        if s["dropped_frames"]:
            for d in s["dropped_frames"]:
                typer.echo(f"    剔除留痕:cam{s['cam']} {d['image_id']} 第 {d['round']} 轮,"
                           f"RMS {d['rms']:.4f}px 超阈值 {d['threshold']:.4f}px")
    typer.echo(f"  两路 report:{report_path.name}(离群留痕与判定明细见内)")
    typer.echo("  全部产物(绝对路径):")
    for p in artifacts:
        typer.echo(f"    {Path(p).resolve()}")

    # ---- fail-closed 闸门(2026-10-24):任一路体检判「不可靠」→ 中文
    #      RuntimeError,由 CLI 异常网收口为「内参标定失败」退出码 1。此刻
    #      intrinsics.json/report.md 已全部落盘留痕——与双目防线同风格:
    #      产物诚实留痕 + 非零退出,绝不让不可靠内参静默流入双目标定。
    bad = [(s, s["diagnostics"]) for s in summaries
           if (s.get("diagnostics") or {}).get("enabled")
           and s["diagnostics"].get("verdict") == "不可靠"]
    if bad:
        detail = ";".join(
            f"cam{s['cam']}:红灯 {','.join(d['red_names']) or '未命名'}" for s, d in bad)
        raise RuntimeError(
            f"内参模型充分性体检未通过({detail}——畸变模型不充分,内参不可靠;"
            f"禁止消费该内参进入双目标定。体检明细见 report.md 模型充分性体检节"
            f"与各路 intrinsics.json 的 diagnostics 键)")
