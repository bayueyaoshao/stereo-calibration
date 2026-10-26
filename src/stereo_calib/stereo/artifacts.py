"""模块3 产物落盘:extrinsics_stereo.json(Pattern 7 契约)+ 残差 npz +
逐帧 CSV + 总报告(决策 8:契约 = Pattern 7 全集 + distance_rms_train_mm;
不导出逐对三角化板点 npz)。

产物布局(output_root/stereo/ 之下;字段层级自本文件定死,不再变更):

- extrinsics_stereo.json   双目外参契约文件,字段全集见
  write_extrinsics_stereo_json;语义 X_cam2 = R @ X_cam1 + t(cam1 为
  参考系,t 单位 mm),baseline_mm = ‖t‖;K1/D1/K2/D2 为所用固定内参
  快照(留痕,对账 Phase 2 产物);dropped_pairs 剔除留痕可审计
- per_frame_train.csv、per_frame_test.csv
    utf-8-sig 逐对表(Excel 双击直开),列序见 _PER_FRAME_FIELDS;
    dropped 列标记剔除对(test 恒 False——考卷不可改),note 中文说明
- train_residuals.npz      train 幸存对逐角点重投影残差(诊断用;
    group_ids 定长 unicode、残差/逐对 RMS/距离残差显式 float64,
    allow_pickle=False 可读)
- report.md                单份总报告:指标汇总(含 baseline 量级核对行,
    决策 5 的人工核对出口)+ 两 split 逐帧距离残差表 + 剔除留痕 +
    翻转留痕 + 配置快照附录(STER-04)

字符串一律定长 unicode 数组、数值一律显式 float64:npz 存取都不需要
pickle,读取方一律 allow_pickle=False(既有安全防线,不因新产物破例)。
本模块只做落盘、零数学(投影/畸变公式只存在于 intrinsic/models.py)。
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from stereo_calib.io.config import BoardSpec

# 逐对 CSV 列序契约(STER-04):utf-8-sig,一行一对,train 在前 test 在后
_PER_FRAME_FIELDS = ["group_id", "split", "route_cam1", "route_cam2",
                     "reproj_rms", "dist_err_mm", "dropped", "note"]

# extrinsics_stereo.json 顶层字段全集(Pattern 7 契约 + 决策 8 的
# distance_rms_train_mm;下游与报告的唯一接口,增删必须同步本断言)
_EXTRINSICS_STEREO_JSON_KEYS = (
    "ref_cam", "R", "t", "rvec", "rotation_angle_deg", "baseline_mm",
    "K1", "D1", "K2", "D2", "n_pairs_train", "n_pairs_test",
    "rms_reproj_train", "rms_reproj_test",
    "distance_rms_train_mm", "distance_rms_test_mm", "true_distance_mm",
    "dropped_pairs", "flip_log", "solver_meta", "board", "config_snapshot",
)


def _unicode_array(ids) -> np.ndarray:
    """字符串清单 → 定长 unicode 一维数组(空清单也得是 '<U1',防 numpy 把
    空列表推断成 float64 的经典陷阱);npz 全程不需要 pickle。"""
    arr = np.asarray([str(x) for x in ids], dtype=np.str_)
    if arr.size == 0:
        return np.empty(0, dtype="<U1")
    return arr


def write_extrinsics_stereo_json(path: Path | str, ref_cam: int, R, t, rvec,
                                 rotation_angle_deg: float, baseline_mm: float,
                                 K1, D1, K2, D2, n_pairs_train: int,
                                 n_pairs_test: int, rms_reproj_train: float,
                                 rms_reproj_test: float,
                                 distance_rms_train_mm: float,
                                 distance_rms_test_mm: float,
                                 true_distance_mm: float,
                                 dropped_pairs: list[dict], flip_log: dict,
                                 solver_meta: dict, board: BoardSpec,
                                 config_snapshot: dict) -> Path:
    """写双目外参契约文件(Pattern 7 全集 + 决策 8,STER-02 的落盘出口)。

    字段层级(与 RESEARCH Pattern 7 逐字段一致):
        ref_cam:            参考相机号(恒 1:R/t 语义为 cam1 → cam2);
        R / t:              3×3 旋转与 (3,) 平移(mm),语义
                            X_cam2 = R @ X_cam1 + t;
        rvec:               R 的 Rodrigues 向量(报告直读);
        rotation_angle_deg: ‖rvec‖ 转角(度);
        baseline_mm:        ‖t‖(与 t 的一致性在写前断言,防两处来源漂移);
        K1/D1/K2/D2:        所用固定内参快照(对账 Phase 2 产物);
        n_pairs_train:      train 幸存对数(剔除后);n_pairs_test:test 对数;
        rms_reproj_train / rms_reproj_test:精化后逐对重投影 pooled RMS(px);
        distance_rms_train_mm / distance_rms_test_mm:
                            板系对角距离残差 pooled RMS(mm,成功标准 3;
                            train 份为决策 8 补的幸存集口径);
        true_distance_mm:   真值(norm(objp[-1] − objp[0]),配置导出);
        dropped_pairs:      [{"group_id", "rms", "round", "threshold"}, …]
                            自动剔除留痕(T-03-12 可审计);
        flip_log:           {"train": {"pass2_flipped": N, "pairs": [...]},
                            "test": {...}} 消歧翻转留痕;
        solver_meta:        status/nfev/离散度诊断等对账信息,不静默;
        board:              棋盘规格快照;config_snapshot:stereo 段快照。
    """
    t_arr = np.asarray(t, dtype=np.float64).ravel()
    assert t_arr.shape == (3,), (
        f"平移向量必须恰 3 长(cam1 系下 mm),实际是 {t_arr.shape}。"
        f"出现此错说明调用方传入的位姿布局不是 [R | t],请检查求解层。")
    baseline = float(baseline_mm)
    assert abs(baseline - float(np.linalg.norm(t_arr))) < 1e-6, (
        f"baseline_mm({baseline!r})与 ‖t‖({float(np.linalg.norm(t_arr))!r})不一致:"
        f"契约规定 baseline_mm = ‖t‖,两处来源漂移会让报告量级核对失真。")
    r_arr = np.asarray(R, dtype=np.float64)
    assert r_arr.shape == (3, 3), (
        f"旋转矩阵必须 3×3,实际是 {r_arr.shape}。请检查求解层输出。")

    def _norm_flip(entry: dict) -> dict:
        # auto_dropped:自适应软剔除(基于相对位姿离散度)自动剥离的翻转/孪生解离群对,
        # 与 pairs(180° 消歧翻转对)同为 flip_log 的留痕字段,不可在归一化时丢弃——
        # 否则磁盘 JSON 查不到被自动剔除的是哪对(留痕意图落空)。
        return {"pass2_flipped": int(entry.get("pass2_flipped", 0)),
                "pairs": [str(x) for x in entry.get("pairs", [])],
                "auto_dropped": [str(x) for x in entry.get("auto_dropped", [])]}

    payload = {
        "ref_cam": int(ref_cam),
        "R": [[float(v) for v in row] for row in r_arr],
        "t": [float(v) for v in t_arr],
        "rvec": [float(v) for v in np.asarray(rvec, dtype=np.float64).ravel()],
        "rotation_angle_deg": float(rotation_angle_deg),
        "baseline_mm": baseline,
        "K1": [[float(v) for v in row] for row in np.asarray(K1, dtype=np.float64)],
        "D1": [float(v) for v in np.asarray(D1, dtype=np.float64).ravel()],
        "K2": [[float(v) for v in row] for row in np.asarray(K2, dtype=np.float64)],
        "D2": [float(v) for v in np.asarray(D2, dtype=np.float64).ravel()],
        "n_pairs_train": int(n_pairs_train),
        "n_pairs_test": int(n_pairs_test),
        "rms_reproj_train": float(rms_reproj_train),
        "rms_reproj_test": float(rms_reproj_test),
        "distance_rms_train_mm": float(distance_rms_train_mm),
        "distance_rms_test_mm": float(distance_rms_test_mm),
        "true_distance_mm": float(true_distance_mm),
        "dropped_pairs": [
            {"group_id": str(d["group_id"]), "rms": float(d["rms"]),
             "round": int(d["round"]), "threshold": float(d["threshold"])}
            for d in dropped_pairs
        ],
        "flip_log": {"train": _norm_flip(flip_log.get("train", {})),
                     "test": _norm_flip(flip_log.get("test", {}))},
        "solver_meta": solver_meta,
        "board": asdict(board),
        "config_snapshot": config_snapshot,
    }
    assert set(payload) == set(_EXTRINSICS_STEREO_JSON_KEYS), (
        f"extrinsics_stereo.json 字段集漂移:实际 {sorted(payload)},"
        f"契约 {sorted(_EXTRINSICS_STEREO_JSON_KEYS)}。"
        f"这是双目外参的唯一落盘接口,增删字段必须同步本文件契约断言与规划文档。")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def write_per_frame_csv(path: Path | str, rows: list[dict]) -> Path:
    """写逐对误差 CSV(utf-8-sig,Excel 双击直开;列序 = _PER_FRAME_FIELDS)。

    行由编排层组装:train 行在前(含被剔除对,dropped=True + 中文剔除
    说明),test 行在后(恒 dropped=False——考卷不可改,note 记评估
    备注)。reproj_rms / dist_err_mm 列为格式化字符串,失败对留空。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_PER_FRAME_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_residuals_npz(path: Path | str, group_ids, residuals_cam1,
                        residuals_cam2, per_pair_rms, dist_err_mm) -> Path:
    """写 train 幸存对的逐角点残差 npz(诊断用;allow_pickle=False 可读)。

    数组契约:residuals_cam1 / residuals_cam2 各 (V,88,2) float64 回代
    残差(BA 位姿回代,只经 models)、per_pair_rms (V,) float64、
    dist_err_mm (V,) float64(失败对 NaN 留痕)、group_ids (V,) 定长
    unicode 对账列(行序与幸存对一一对应)。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        group_ids=_unicode_array(group_ids),
        residuals_cam1=np.asarray(residuals_cam1, dtype=np.float64),
        residuals_cam2=np.asarray(residuals_cam2, dtype=np.float64),
        per_pair_rms=np.asarray(per_pair_rms, dtype=np.float64),
        dist_err_mm=np.asarray(dist_err_mm, dtype=np.float64),
    )
    return path


def _fmt(v: float) -> str:
    """报告里的数值显示格式:4 位小数;非有限(无有效帧)显式标 NaN。"""
    return f"{v:.4f}" if np.isfinite(v) else "NaN"


def write_report_md(path: Path | str, summary: dict, distance_rms_max_mm: float,
                    baseline_band_mm, config_snapshot: dict) -> Path:
    """写双目标定总报告(STER-04;f-string markdown,零模板引擎)。

    summary 的键(编排层组装):
        R / t / rvec / rotation_angle_deg / baseline_mm /
        rms_reproj_train / rms_reproj_test /
        distance_rms_train_mm / distance_rms_test_mm / true_distance_mm /
        n_pairs_train(幸存)/ n_pairs_train_initial / n_pairs_test /
        train_evals / test_evals(逐对清单:group_id/reproj_rms/dist_err_mm/
        flipped/note/dropped)/ dropped_pairs(剔除留痕清单)/
        flip_log {"train"/"test": {"pass2_flipped", "pairs"}} /
        outlier_loop(剔除循环结束原因,含保险丝触发说明)/ failed / error /
        test_error / train_error(单段评估降级留痕,可选)。

    判定列按 distance_rms_max_mm 给 PASS/FAIL——只作报告结论,不作退出
    条件(OVERNIGHT AUTONOMY:跑完留痕,人不被夜里叫起来看退出码)。
    baseline 量级核对行(决策 5):|t| 各分量 + baseline 对带上下限的
    落位说明——灾难型错解(平移跑飞/退化为零)在此人工一眼可辨。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lo, hi = float(baseline_band_mm[0]), float(baseline_band_mm[1])

    lines.append("# 双目标定报告")
    lines.append("")
    lines.append(f"- 生成时间:{datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- test 集判定阈值 distance_rms_max_mm:{distance_rms_max_mm} mm"
                 f"(报告判定用,不作退出条件)")
    lines.append(f"- 外参语义:X_cam2 = R @ X_cam1 + t(cam{summary['ref_cam']} 为参考系,"
                 f"t 单位 mm)")
    lines.append("")

    if summary.get("failed"):
        lines.append("## 失败留痕")
        lines.append("")
        lines.append(f"本段双目标定失败:{summary['error']}")
        lines.append("")
        lines.append("其余段产物不受影响(若有);排除数据问题后重跑本模块即可。")
        lines.append("")
    else:
        # ---- 指标汇总:R/t/baseline + 量级核对行(决策 5)----
        r_arr = np.asarray(summary["R"], dtype=np.float64)
        t_arr = np.asarray(summary["t"], dtype=np.float64).ravel()
        lines.append("## 指标汇总")
        lines.append("")
        lines.append("| 指标 | 值 |")
        lines.append("|---|---|")
        for i in range(3):
            rows_str = ", ".join(f"{v: .6f}" for v in r_arr[i])
            lines.append(f"| R 行 {i + 1} | [{rows_str}] |")
        for i in range(3):
            lines.append(f"| t[{i}] (mm) | {t_arr[i]:.4f} |")
        lines.append(f"| rvec | [{', '.join(f'{v: .6f}' for v in summary['rvec'])}] |")
        lines.append(f"| 旋转角 (deg) | {summary['rotation_angle_deg']:.4f} |")
        lines.append(f"| 基线 baseline_mm | {summary['baseline_mm']:.4f} |")
        in_band = lo <= summary["baseline_mm"] <= hi
        lines.append(f"| 基线量级核对 | ‖t‖ = {summary['baseline_mm']:.2f} mm,"
                     f"|t| 分量 = [{', '.join(f'{v:.2f}' for v in t_arr)}] mm,"
                     f"落在 sanity 带 [{lo}, {hi}] mm "
                     f"{'内(量级正常)' if in_band else '外(疑似错解,已在运行时硬门拦截—若见此行请核对内参与棋盘规格)'} |")
        ok_test = (np.isfinite(summary["distance_rms_test_mm"])
                   and summary["distance_rms_test_mm"] <= distance_rms_max_mm)
        verdict = "PASS" if ok_test else "FAIL"
        lines.append(f"| 重投影 RMS train/test (px) | {_fmt(summary['rms_reproj_train'])} / "
                     f"{_fmt(summary['rms_reproj_test'])} |")
        lines.append(f"| 距离 RMS train/test (mm) | {_fmt(summary['distance_rms_train_mm'])} / "
                     f"{_fmt(summary['distance_rms_test_mm'])} | {verdict} |")
        lines.append(f"| 对角距离真值 (mm) | {summary['true_distance_mm']:.4f} |")
        lines.append(f"| 对数 train 幸存/初始、test | {summary['n_pairs_train']} / "
                     f"{summary['n_pairs_train_initial']}、{summary['n_pairs_test']} |")
        lines.append("")
        lines.append(f"判定列按 distance_rms_max_mm = {distance_rms_max_mm} mm 给 "
                     f"PASS/FAIL——只作报告结论,不作退出条件。")
        lines.append("")
        if summary.get("test_error"):
            lines.append(f"**test 评估降级留痕**:{summary['test_error']}")
            lines.append("")
        if summary.get("train_error"):
            lines.append(f"**train 评估降级留痕**:{summary['train_error']}")
            lines.append("")

        # ---- 逐帧距离残差表:train / test 各一节 ----
        for split in ("train", "test"):
            evals = summary[f"{split}_evals"]
            lines.append(f"## 逐帧距离残差({split})")
            lines.append("")
            lines.append("| group_id | reproj_rms (px) | dist_err_mm | flipped | dropped | note |")
            lines.append("|---|---|---|---|---|---|")
            for ev in evals:
                lines.append(
                    f"| {ev['group_id']} | {_fmt(ev['reproj_rms'])} | "
                    f"{_fmt(ev['dist_err_mm'])} | {ev['flipped']} | "
                    f"{'是' if ev['dropped'] else '否'} | {ev['note']} |")
            lines.append("")

        # ---- 剔除留痕(可审计,T-03-12)----
        lines.append("## 剔除留痕(自动剔除对)")
        lines.append("")
        dropped = summary["dropped_pairs"]
        if dropped:
            lines.append("| group_id | 轮次 | RMS (px) | 阈值 (px) |")
            lines.append("|---|---|---|---|")
            for d in dropped:
                lines.append(f"| {d['group_id']} | {d['round']} | "
                             f"{d['rms']:.4f} | {d['threshold']:.4f} |")
        else:
            lines.append("无(本轮运行没有对超过离群阈值)。")
        lines.append("")
        lines.append(f"- 剔除循环:{summary['outlier_loop']}")
        lines.append("")

        # ---- 翻转留痕(消歧)----
        lines.append("## 翻转留痕(180° 消歧)")
        lines.append("")
        for split in ("train", "test"):
            entry = summary["flip_log"].get(split, {})
            lines.append(f"- {split}:pass2 翻转 {entry.get('pass2_flipped', 0)} 对"
                         f"({', '.join(entry.get('pairs', [])) or '无'})")
        lines.append("")

    # ---- 附录:配置快照留痕(照 Phase 1/2 模式)----
    lines.append("## 附录:配置快照(config_snapshot)")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(config_snapshot, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path
