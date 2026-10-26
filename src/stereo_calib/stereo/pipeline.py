"""模块3 主流程:双目外参标定编排(train 剔除-重解循环 + test 评估 + 落盘)。

流程(run_stereo 是本模块唯一编排入口,零数学——只调用 03-01/03-02/03-03
的 observation/solve/evaluate/artifacts 入口,投影、三角化、精化通通不住
本文件,源码防线由测试锁):

  1. 运行前清理 output_root/stereo;
  2. 装载两路固定内参(Phase 2 契约产物)+ train/test 两 split 幸存对
     (exclude_pairs 手动剔除通道生效);
  3. train 剔除循环(决策 2:判据 = 逐对重投影 RMS;时序照 Phase 2:
     剔除循环全程用快解——精化开关临时关闭,精化只跑幸存集定稿后一次)。
     阈值 = rms_multiplier × 幸存集中位数(中位数基准稳健,不受离群自身
     拖动);保险丝照 Phase 2 三道:剔除后幸存数将低于
     max(min_pairs,(1−max_drop_fraction)×初始对数) → 停,已剔的保留;
     打满 max_iterations 轮仍有超阈 → 停;开关关闭一次都不剔。触发原因
     全部如实留痕进报告,绝不静默(Pitfall 8);
  4. baseline 量级硬门(决策 5):baseline 不在 sanity 带内 → 中文
     ValueError,绝不带病出报告(灾难型错解在落盘前拦死);
  5. 评估:test 全对(考卷不可改,永不剔除只标记)+ train 幸存集。
     距离残差只进报告与标记,绝不回流剔除判据(决策 2/决策 9);
  6. 落盘五件:JSON 契约 / 逐帧 CSV×2 / 残差 npz / report.md;
  7. 中文汇总:幸存/剔除、两 split 指标、PASS/FAIL(报告结论非退出
     条件)、全部产物绝对路径(OVERNIGHT AUTONOMY)。

数据安全(T-03-SC,照 Phase 2 体例):全部写路径仅从 cfg.output_root
派生。单段降级:train 求解或 test 评估某段失败 → 中文留痕降级,绝不
炸掉已成功段的产物(一路失败不中断整批的编排层兜底体例)。
"""

from __future__ import annotations

import shutil
from dataclasses import asdict, replace
from pathlib import Path

import cv2
import numpy as np
import typer

from stereo_calib.io.config import StereoCalibConfig
from stereo_calib.preprocess.artifacts import build_object_points
from stereo_calib.stereo.artifacts import (
    write_extrinsics_stereo_json,
    write_per_frame_csv,
    write_report_md,
    write_residuals_npz,
)
from stereo_calib.stereo.evaluate import evaluate_split, true_board_distance
from stereo_calib.stereo.observation import (
    IntrinsicsView,
    Pairs,
    load_intrinsics,
    load_pairs,
    select_pairs,
)
from stereo_calib.stereo.solve import solve_stereo

# 装载/评估两 split(行序与 Phase 1 目录约定一致)
_SPLITS = ("train", "test")

# test split 装载下限:成对几何的最小需求。train 的 min_pairs 保险丝
# 管的是"求解素材够不够",test 是考卷不是求解素材,不适用同一把尺
# (15 对 test 配 min_pairs=10 是生产形态;合成小样本测试不受钳制)。
_MIN_TEST_PAIRS = 2

# 单段数值异常收口家族(照 intrinsic/pipeline 同一体例)
_SEGMENT_EXC = (ValueError, OSError, ArithmeticError, RuntimeError,
                AssertionError, cv2.error)

# test 帧重投影离群的标记语(只标记不剔除——考卷不可改)
_TEST_MARK_NOTE = "test 帧重投影超中位乘数阈,仅标记不剔除(考卷不可改)"


def _load_split_pairs(cfg: StereoCalibConfig, split: str,
                      min_pairs: int) -> Pairs:
    """装载一个 split 的幸存双目对(路径全从 output_root 派生,D-05 布局)。"""
    split_dir = Path(cfg.output_root) / split
    csv_path = split_dir / f"common_fov_pairs_{split}.csv"
    npz1 = split_dir / f"cam1_{split}.npz"
    npz2 = split_dir / f"cam2_{split}.npz"
    return load_pairs(cfg.board, csv_path, npz1, npz2,
                      exclude_pairs=cfg.stereo.exclude_pairs,
                      min_pairs=min_pairs)


def _drop_outliers(cfg: StereoCalibConfig, train: Pairs,
                   intr1: IntrinsicsView, intr2: IntrinsicsView
                   ) -> tuple[list[dict], list[int], str, np.ndarray]:
    """train 集离群剔除循环(Pattern 6):留痕清单/剔除行号/循环说明/末轮 RMS。

    判据 = 逐对重投影 RMS(决策 2),阈值 = rms_multiplier × 幸存集中位;
    距离残差不进本函数(它还没算,也绝不回流——决策 9)。精化不进循环:
    全程用快解(refine 开关临时关闭),精化只在幸存集定稿后跑一次
    (时序照 Phase 2)。保险丝语义照 Phase 2:先核对剔除后幸存数,
    不足则整轮不剔并如实留痕。
    """
    outlier = cfg.stereo.outlier
    dropped: list[dict] = []
    dropped_idx: list[int] = []
    fast_cfg = replace(cfg, stereo=replace(cfg.stereo, refine=replace(
        cfg.stereo.refine, enabled=False)))
    if not outlier.enabled:
        res = solve_stereo(train, intr1, intr2, fast_cfg)
        return dropped, dropped_idx, (
            "自动剔除开关关闭(stereo.outlier.enabled: false),未做剔除"), res.per_pair_rms

    n_initial = len(train.image_ids)
    survived = list(range(n_initial))
    per_pair_rms = np.empty(0, dtype=np.float64)
    loop_note = ""
    for round_i in range(1, outlier.max_iterations + 1):
        res = solve_stereo(select_pairs(train, survived), intr1, intr2, fast_cfg)
        per_pair_rms = res.per_pair_rms
        med = float(np.median(per_pair_rms))
        threshold = outlier.rms_multiplier * med
        bad = np.where(per_pair_rms > threshold)[0]
        typer.echo(f"    第 {round_i} 轮剔除检查:幸存 {len(survived)} 对,"
                   f"中位 RMS {med:.4f}px,阈值 {threshold:.4f}px,"
                   f"超阈 {int(bad.size)} 对")
        if bad.size == 0:
            loop_note = (f"第 {round_i} 轮无超阈对,剔除循环收敛"
                         f"(共剔除 {len(dropped)} 对)")
            break
        floor = max(outlier.min_pairs, (1.0 - outlier.max_drop_fraction) * n_initial)
        if len(survived) - int(bad.size) < floor:
            loop_note = (
                f"第 {round_i} 轮触发保险丝:本轮超阈 {int(bad.size)} 对,剔除后幸存 "
                f"{len(survived) - int(bad.size)} 将低于下限 {floor:.1f}"
                f"(min_pairs={outlier.min_pairs},"
                f"max_drop_fraction={outlier.max_drop_fraction}),"
                f"停止剔除并如实留痕(共剔除 {len(dropped)} 对)")
            break
        for j in bad:
            j = int(j)
            orig = survived[j]
            dropped.append({"group_id": train.image_ids[orig],
                            "rms": float(per_pair_rms[j]),
                            "round": round_i, "threshold": float(threshold)})
            dropped_idx.append(orig)
            typer.echo(f"      剔除留痕:{train.image_ids[orig]} 第 {round_i} 轮,"
                       f"RMS {float(per_pair_rms[j]):.4f}px 超阈值 {threshold:.4f}px")
        survived = [i for i in survived if i not in set(dropped_idx)]
    else:
        loop_note = (f"剔除循环打满 max_iterations={outlier.max_iterations} 轮"
                     f"仍有超阈对(中位数随剔除漂移未收敛的信号,请检查数据),"
                     f"共剔除 {len(dropped)} 对")
    return dropped, dropped_idx, loop_note, per_pair_rms


def _pooled_frame_rms(evals) -> float:
    """逐帧 RMS 的 pooled 口径(等角点数下 = sqrt(平方均值));全 NaN 给 NaN。"""
    vals = np.asarray([e.reproj_rms for e in evals], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(vals ** 2)))


def run_stereo(cfg: StereoCalibConfig) -> None:
    """模块3 主流程:train 求解+剔除 → baseline 硬门 → 双 split 评估 → 落盘。"""
    stereo = cfg.stereo
    stereo_root = Path(cfg.output_root) / "stereo"

    # ---- fail-closed 硬门(CR-01):joint.enabled 当前没有生产消费者 ----
    # 联合 BA 引擎(joint.py)已交付,但生产接线(03-05 Task 3,评估链接线
    # + 24 键契约)因三变体裁决失败按失败协议跳过——run_stereo 全文不读
    # joint 子段,开关置 true 只会静默跑 rigid 路径(fail-open),违反本
    # 工程「不静默」红线(Pitfall 8)。此处硬拦:中文 ValueError 收口进
    # CLI 异常网(退出码 1),且发生在目录清理之前——既有产物原样保留,
    # 绝不制造「joint 已生效」的运行假象。接线任务落地后才允许打开
    # (deferred-items.md 已登记追踪)。
    if stereo.joint.enabled:
        raise ValueError(
            "stereo.joint.enabled=true 不合法:联合 BA 引擎已交付但生产接线"
            "(03-05 Task 3,评估链接线 + 24 键契约)因三变体裁决失败按失败"
            "协议跳过,当前开关没有任何消费者——置 true 只会静默跑 rigid "
            "路径。请保持 enabled: false,等待接线任务落地。")

    # ---- 运行前清理(D-11 固定覆盖 + 防残留,照 Phase 2 体例)----
    # 只删 output_root/stereo 一个产物子目录,删后随落盘按需重建;上次运行
    # 的 JSON/npz/CSV/报告残留不会误导人工复核。output_root 本身与其他未知
    # 内容不动。数据安全前提(不重复实现):load_config 已在加载层拒绝
    # output_root 与 data 目录相等/互相包含的配置(Phase 1 锁定),这里的
    # rmtree 只可能命中 output_root 之下的产物目录,不可能触及 data/。
    shutil.rmtree(stereo_root, ignore_errors=True)

    # ---- 装载:两路固定内参 + 两 split 幸存对 ----
    intrinsic_root = Path(cfg.output_root) / "intrinsic"
    intr1 = load_intrinsics(intrinsic_root / "cam1" / "intrinsics.json")
    intr2 = load_intrinsics(intrinsic_root / "cam2" / "intrinsics.json")
    typer.echo(f"开始双目标定:剔除 multiplier={stereo.outlier.rms_multiplier},"
               f"保险丝 min_pairs={stereo.outlier.min_pairs} / "
               f"max_drop_fraction={stereo.outlier.max_drop_fraction} / "
               f"max_iterations={stereo.outlier.max_iterations}……")
    train = _load_split_pairs(cfg, "train", stereo.outlier.min_pairs)
    test = _load_split_pairs(cfg, "test", _MIN_TEST_PAIRS)
    typer.echo(f"  装载完成:train {len(train.image_ids)} 对、"
               f"test {len(test.image_ids)} 对(固定内参 cam1/cam2 已按名读取)")

    # ---- train 求解 + 剔除循环(单段降级:失败留痕不炸整批)----
    try:
        dropped, dropped_idx, loop_note, _fast_rms = _drop_outliers(
            cfg, train, intr1, intr2)
        survivors = [i for i in range(len(train.image_ids)) if i not in set(dropped_idx)]
        train_survivors = select_pairs(train, survivors)
        final = solve_stereo(train_survivors, intr1, intr2, cfg)  # 精化只此一次
        # 自适应软剔除(基于相对位姿离散度)的对需从评估/报告中同步排除,
        # 否则会用剔除后的位姿去评估被剔除对 → 距离残差虚高(假象)
        _sd = list(final.flip_log.get("train", {}).get("auto_dropped", []))
        if _sd:
            _sd_set = set(_sd)
            _used_idx = [i for i, _g in enumerate(train_survivors.image_ids) if _g not in _sd_set]
            train_used = select_pairs(train_survivors, _used_idx)
        else:
            train_used = train_survivors
        typer.echo(f"  train 定稿:幸存 {len(survivors)}/{len(train.image_ids)} 对,"
                   f"重投影 pooled RMS {final.rms_reproj:.4f}px,"
                   f"baseline {final.baseline_mm:.2f} mm,solver={final.solver}")
    except _SEGMENT_EXC as e:
        typer.echo(f"双目标定失败,降级为失败记录:{type(e).__name__}——{e}")
        report_path = write_report_md(
            stereo_root / "report.md",
            {"ref_cam": 1, "failed": True, "error": f"{type(e).__name__}——{e}",
             "dropped_pairs": [], "outlier_loop": "train 求解失败,剔除循环未完成",
             "flip_log": {}, "train_evals": [], "test_evals": []},
            stereo.distance_rms_max_mm, stereo.baseline_band_mm,
            {"stereo": asdict(stereo)})
        typer.echo(f"  失败留痕已写:{report_path.resolve()}")
        return

    # ---- baseline 量级硬门(决策 5):灾难型错解绝不带病出报告 ----
    lo, hi = stereo.baseline_band_mm
    if not lo <= final.baseline_mm <= hi:
        raise ValueError(
            f"双目 baseline 量级越界:{final.baseline_mm:.2f} mm 不在 sanity 带 "
            f"[{lo}, {hi}] mm 内(baseline_band_mm)——多半是内参与角点不同源、"
            f"棋盘规格错位或离群对污染求解(可调 rms_multiplier 收紧剔除)。"
            f"硬门拦截,不落盘不出报告,请修正后重跑。")

    # ---- 评估:test 全对(永不剔除只标记)+ train 幸存集 ----
    # 距离残差只进报告与标记,绝不回流剔除判据(决策 2/决策 9)。train 的
    # 剔除判据只有逐对重投影 RMS(上面循环已完),这里评估纯粹是留证。
    # 两路评估同在单段降级收口内(WR-01):任一 split 评估全帧失败 → 中文
    # 留痕降级,已成功且昂贵的 train 求解+精化结果绝不丢弃——产物五件
    # 尽量齐(降级侧指标 NaN、npz 残差整块 NaN、CSV 幸存行缺席、报告追加
    # 降级留痕)。
    test_error = ""
    try:
        test_eval = evaluate_split(test, intr1, intr2, final.R, final.t,
                                   cfg.board, stereo.flip_ratio)
    except _SEGMENT_EXC as e:
        test_eval = None
        test_error = f"{type(e).__name__}——{e}"
        typer.echo(f"  test 评估失败,降级留痕(train 产物不受影响):{test_error}")
    train_error = ""
    try:
        train_eval = evaluate_split(train_used, intr1, intr2, final.R, final.t,
                                    cfg.board, stereo.flip_ratio)
    except _SEGMENT_EXC as e:
        train_eval = None
        train_error = f"{type(e).__name__}——{e}"
        typer.echo(f"  train 评估失败,降级留痕(train 求解结果保留):{train_error}")

    # test 逐帧标记:重投影超中位乘数阈 → note 追加标记(考卷不可改,
    # 恒不剔除);判据与 train 剔除同源(逐对重投影 × 中位乘数),但只标不剔
    test_rows = list(test_eval.evals) if test_eval is not None else []
    if test_rows:
        finite = np.asarray([e.reproj_rms for e in test_rows
                             if np.isfinite(e.reproj_rms)], dtype=np.float64)
        if finite.size:
            mark_threshold = stereo.outlier.rms_multiplier * float(np.median(finite))
            test_rows = [
                e if not (np.isfinite(e.reproj_rms) and e.reproj_rms > mark_threshold)
                else replace(e, note=(e.note + ";" if e.note else "") + _TEST_MARK_NOTE)
                for e in test_rows
            ]

    # ---- flip_log 归一到 Pattern 7 契约键(pass2_flipped / pairs)----
    train_flip = final.flip_log.get("train", {})
    flip_log = {
        "train": {"pass2_flipped": int(train_flip.get("pass2_flipped", 0)),
                  "pairs": [str(x) for x in train_flip.get("pair_ids", [])],
                  "auto_dropped": [str(x) for x in train_flip.get("auto_dropped", [])]},
        "test": {"pass2_flipped": 0, "pairs": []},
    }
    if test_rows:
        flipped_pairs = [e.group_id for e in test_rows if any(e.flipped)]
        flip_log["test"] = {"pass2_flipped": len(flipped_pairs),
                            "pairs": flipped_pairs}

    rms_reproj_test = _pooled_frame_rms(test_rows) if test_rows else float("nan")
    distance_rms_test = test_eval.distance_rms_mm if test_eval is not None else float("nan")
    distance_rms_train = (train_eval.distance_rms_mm if train_eval is not None
                          else float("nan"))
    # true_distance_mm 是配置棋盘的确定量(对角线长度),不依赖任何评估
    # 存活——train 评估降级时直接从配置导出,契约键照常给真值(WR-01)
    true_distance = (train_eval.true_distance_mm if train_eval is not None
                     else true_board_distance(build_object_points(cfg.board)))

    stereo_root.mkdir(parents=True, exist_ok=True)

    # ---- 落盘五件(路径全从 output_root 派生)----
    json_path = write_extrinsics_stereo_json(
        stereo_root / "extrinsics_stereo.json", ref_cam=1, R=final.R, t=final.t,
        rvec=final.rvec, rotation_angle_deg=final.rotation_angle_deg,
        baseline_mm=final.baseline_mm, K1=intr1.K, D1=intr1.D,
        K2=intr2.K, D2=intr2.D, n_pairs_train=len(train_used.image_ids),
        n_pairs_test=len(test.image_ids), rms_reproj_train=final.rms_reproj,
        rms_reproj_test=rms_reproj_test,
        distance_rms_train_mm=distance_rms_train,
        distance_rms_test_mm=distance_rms_test,
        true_distance_mm=true_distance,
        dropped_pairs=dropped, flip_log=flip_log, solver_meta=dict(
            final.solver_meta, solver=final.solver),
        board=cfg.board, config_snapshot={"stereo": asdict(stereo)})

    # 逐帧 CSV:train 全初始对(剔除行 dropped=True + 中文说明,幸存行接评估),
    # test 全对(恒 dropped=False)
    drop_by_gid = {d["group_id"]: d for d in dropped}
    train_csv_rows: list[dict] = []
    for pos, gid in enumerate(train.image_ids):
        r1, r2 = train.routes[pos]
        if gid in drop_by_gid:
            d = drop_by_gid[gid]
            train_csv_rows.append(
                {"group_id": gid, "split": "train", "route_cam1": r1,
                 "route_cam2": r2, "reproj_rms": f"{d['rms']:.4f}",
                 "dist_err_mm": "", "dropped": "True",
                 "note": f"第 {d['round']} 轮重投影超阈剔除(阈值 {d['threshold']:.4f}px)"})
    for pos, ev in enumerate(train_eval.evals if train_eval is not None else []):
        r1, r2 = train_survivors.routes[pos]
        train_csv_rows.append(
            {"group_id": ev.group_id, "split": "train", "route_cam1": r1,
             "route_cam2": r2, "reproj_rms": f"{ev.reproj_rms:.4f}",
             "dist_err_mm": f"{ev.dist_err_mm:.4f}", "dropped": "False",
             "note": ev.note})
    csv_train_path = write_per_frame_csv(
        stereo_root / "per_frame_train.csv", train_csv_rows)

    test_csv_rows = [
        {"group_id": ev.group_id, "split": "test",
         "route_cam1": test.routes[pos][0], "route_cam2": test.routes[pos][1],
         "reproj_rms": f"{ev.reproj_rms:.4f}",
         "dist_err_mm": f"{ev.dist_err_mm:.4f}", "dropped": "False",
         "note": ev.note}
        for pos, ev in enumerate(test_rows)]
    csv_test_path = write_per_frame_csv(
        stereo_root / "per_frame_test.csv", test_csv_rows)

    # 诊断 npz:评估降级时残差整块 NaN 留痕(形状契约不破,失败可审计)
    if train_eval is not None:
        npz_res1, npz_res2 = train_eval.residuals_cam1, train_eval.residuals_cam2
        npz_dist = np.asarray([ev.dist_err_mm for ev in train_eval.evals],
                              dtype=np.float64)
    else:
        n_surv, n_points = len(train_survivors.image_ids), cfg.board.rows * cfg.board.cols
        npz_res1 = np.full((n_surv, n_points, 2), np.nan)
        npz_res2 = np.full((n_surv, n_points, 2), np.nan)
        npz_dist = np.full(n_surv, np.nan)
    npz_path = write_residuals_npz(
        stereo_root / "train_residuals.npz", train_survivors.image_ids,
        npz_res1, npz_res2, final.per_pair_rms, npz_dist)

    summary = {
        "ref_cam": 1, "R": final.R, "t": final.t, "rvec": final.rvec,
        "rotation_angle_deg": final.rotation_angle_deg,
        "baseline_mm": final.baseline_mm,
        "rms_reproj_train": final.rms_reproj,
        "rms_reproj_test": rms_reproj_test,
        "distance_rms_train_mm": distance_rms_train,
        "distance_rms_test_mm": distance_rms_test,
        "true_distance_mm": true_distance,
        "n_pairs_train": len(train_used.image_ids),
        "n_pairs_train_initial": len(train.image_ids),
        "n_pairs_test": len(test.image_ids),
        "train_evals": [
            {"group_id": ev.group_id, "reproj_rms": ev.reproj_rms,
             "dist_err_mm": ev.dist_err_mm,
             "flipped": f"({ev.flipped[0]}, {ev.flipped[1]})",
             "dropped": False, "note": ev.note}
            for ev in (train_eval.evals if train_eval is not None else [])],
        "test_evals": [
            {"group_id": ev.group_id, "reproj_rms": ev.reproj_rms,
             "dist_err_mm": ev.dist_err_mm,
             "flipped": f"({ev.flipped[0]}, {ev.flipped[1]})",
             "dropped": False, "note": ev.note}
            for ev in test_rows],
        "dropped_pairs": dropped,
        "flip_log": flip_log,
        "outlier_loop": loop_note,
    }
    if test_error:
        summary["test_error"] = f"test 评估失败(降级留痕,train 产物不受影响):{test_error}"
    if train_error:
        summary["train_error"] = f"train 评估失败(降级留痕,train 求解结果保留):{train_error}"
    report_path = write_report_md(
        stereo_root / "report.md", summary, stereo.distance_rms_max_mm,
        stereo.baseline_band_mm, {"stereo": asdict(stereo)})

    # ---- 中文汇总:指标 + PASS/FAIL(报告结论非退出条件)+ 产物绝对路径 ----
    typer.echo("双目标定完成,产物已写盘:")
    typer.echo(f"  对数:train 幸存 {len(train_used.image_ids)}/{len(train.image_ids)}"
               f"(RMS 剔除 {len(train.image_ids) - len(survivors)} + 软剔除 "
               f"{len(survivors) - len(train_used.image_ids)})、test {len(test.image_ids)}")
    typer.echo(f"  重投影 RMS:train {final.rms_reproj:.4f}px、"
               f"test {rms_reproj_test:.4f}px")
    typer.echo(f"  距离 RMS:train {distance_rms_train:.4f} mm、"
               f"test {distance_rms_test:.4f} mm"
               f"(阈值 {stereo.distance_rms_max_mm} mm)")
    if test_error:
        typer.echo(f"  test 评估:{test_error}")
    if train_error:
        typer.echo(f"  train 评估:{train_error}")
    ok = np.isfinite(distance_rms_test) and distance_rms_test <= stereo.distance_rms_max_mm
    typer.echo(f"  判定:{'PASS' if ok else 'FAIL'}(按 distance_rms_max_mm,"
               f"报告结论非退出条件)")
    typer.echo(f"  baseline 量级:‖t‖ = {final.baseline_mm:.2f} mm,"
               f"|t| 分量 = [{', '.join(f'{v:.2f}' for v in final.t)}] mm,"
               f"sanity 带 [{lo}, {hi}] mm")
    typer.echo(f"  消歧翻转:train {flip_log['train']['pass2_flipped']} 对、"
               f"test {flip_log['test']['pass2_flipped']} 对")
    typer.echo("  全部产物(绝对路径):")
    for p in (json_path, csv_train_path, csv_test_path, npz_path, report_path):
        typer.echo(f"    {Path(p).resolve()}")
