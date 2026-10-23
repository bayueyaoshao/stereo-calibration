"""模块2 产物落盘:intrinsics.json(Pattern 7 契约)+ 残差/外参 npz 三件套 + 逐帧 CSV + 报告。

产物布局(output_root/intrinsic/ 之下;cam{n}/intrinsics.json 是 Phase 3
按名读取的唯一固定内参接口,字段层级自本文件定死,不再变更):

- cam{n}/intrinsics.json    每路一份,字段全集见 write_intrinsics_json;
  params 按名(fx/fy/cx/cy/k1/k2/p1/p2/k3),K/D 冗余存储避免模块 3 重建
  出错;dropped_frames 留痕自动剔除清单(image_id/rms/轮次/阈值,可审计)
- cam{n}/train_residuals.npz、cam{n}/test_residuals.npz(三件套之二)
    image_ids:   (V,)   定长 unicode,与行序一一对应(对账列,Pitfall 10)
    residuals:   (V,88,2) float64,float64 回代残差;test 份的单帧评分失败
                 行整行 NaN(失败留痕),train 份恒全有限
    per_view_rms:(V,)  float64,逐帧 RMS;NaN 语义同行序失败帧
- cam{n}/extrinsics.npz     train 幸存视图的 rvecs/tvecs 各 (V,3) float64,
  纯诊断用——模块 3 双目外参自己做 PnP,不消费本文件(调用方写 train 位姿)
- per_frame_cam{n}.csv      utf-8-sig 逐帧表(Excel 双击直开),列序见
  _PER_FRAME_FIELDS;dropped 列标记自动剔除帧,note 列中文说明
- report.md                 单份总报告:两路指标对比 + 按名参数表 + 离群
  留痕 + 逐帧误差摘要(INTR-04)

字符串一律定长 unicode 数组、数值一律显式 float64:npz 存取都不需要
pickle,读取方一律 allow_pickle=False(既有安全防线,不因新产物破例)。
本模块只做落盘、零数学(投影/畸变公式只存在于 models.py)。
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from stereo_calib.intrinsic.models import INTRINSIC_NAMES
from stereo_calib.io.config import BoardSpec

# 逐帧 CSV 列序契约(INTR-04):utf-8-sig,一行一帧,train 在前 test 在后
_PER_FRAME_FIELDS = ["image_id", "split", "source", "rms", "dropped", "note"]

# intrinsics.json 顶层字段全集(Pattern 7 契约;Phase 3 按名读取的唯一接口)
_INTRINSICS_JSON_KEYS = (
    "cam", "image_size", "params", "K", "D", "solver", "refine_enabled",
    "rms_train_overall", "rms_test_overall", "n_views_train", "n_views_test",
    "dropped_frames", "solver_meta", "board", "config_snapshot",
)


def _unicode_array(ids) -> np.ndarray:
    """字符串清单 → 定长 unicode 一维数组(空清单也得是 '<U1',防 numpy 把
    空列表推断成 float64 的经典陷阱);npz 全程不需要 pickle。"""
    arr = np.asarray([str(x) for x in ids], dtype=np.str_)
    if arr.size == 0:
        return np.empty(0, dtype="<U1")
    return arr


def write_intrinsics_json(path: Path | str, cam: int, image_size,
                          params, K, D, solver: str, refine_enabled: bool,
                          rms_train_overall: float, rms_test_overall: float,
                          n_views_train: int, n_views_test: int,
                          dropped_frames: list[dict], solver_meta: dict,
                          board: BoardSpec, config_snapshot: dict) -> Path:
    """写一路相机的内参契约文件(Pattern 7,模块 3 的唯一固定内参输入)。

    字段层级(与 RESEARCH Pattern 7 逐字段一致):
        cam:               相机号(int);
        image_size:        [宽, 高],与标定同源口径;
        params:            九参数按名(fx/fy/cx/cy/k1/k2/p1/p2/k3);
        K:                 3×3 冗余存储(避免模块 3 重建出错);
        D:                 [k1,k2,p1,p2,k3] OpenCV 布局,5 长冗余存储;
        solver:            "opencv" | "scipy"(精化开关关闭时为前者);
        refine_enabled:    精化开关留痕;
        rms_train_overall / rms_test_overall:float64 回代 pooled RMS;
        n_views_train / n_views_test:帧数(train 为剔除后幸存数);
        dropped_frames:    [{"image_id", "rms", "round", "threshold"}, …]
                           自动剔除留痕(T-02-08 可审计);
        solver_meta:       status/nfev/opencv_rms 等对账信息,不静默;
        board:             棋盘规格快照;config_snapshot:intrinsic 段快照。
    """
    d_arr = np.asarray(D, dtype=np.float64).ravel()
    assert d_arr.shape == (5,), (
        f"畸变向量必须恰 5 长([k1,k2,p1,p2,k3] OpenCV 布局),实际是 {d_arr.shape}。"
        f"出现此错说明调用方传入的参数布局不是 M2,请检查 params。")
    payload = {
        "cam": int(cam),
        "image_size": [int(image_size[0]), int(image_size[1])],
        "params": {name: float(v) for name, v in zip(INTRINSIC_NAMES, params)},
        "K": [[float(v) for v in row] for row in np.asarray(K, dtype=np.float64)],
        "D": [float(v) for v in d_arr],
        "solver": str(solver),
        "refine_enabled": bool(refine_enabled),
        "rms_train_overall": float(rms_train_overall),
        "rms_test_overall": float(rms_test_overall),
        "n_views_train": int(n_views_train),
        "n_views_test": int(n_views_test),
        "dropped_frames": [
            {"image_id": str(d["image_id"]), "rms": float(d["rms"]),
             "round": int(d["round"]), "threshold": float(d["threshold"])}
            for d in dropped_frames
        ],
        "solver_meta": solver_meta,
        "board": asdict(board),
        "config_snapshot": config_snapshot,
    }
    assert set(payload) == set(_INTRINSICS_JSON_KEYS), (
        f"intrinsics.json 字段集漂移:实际 {sorted(payload)},契约 {sorted(_INTRINSICS_JSON_KEYS)}。"
        f"这是 Phase 3 的固定输入接口,增删字段必须同步本文件契约断言与规划文档。")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def write_residuals_npz(path: Path | str, image_ids, residuals,
                        per_view_rms) -> Path:
    """写一份残差 npz(train/test 各一份,三件套之二;allow_pickle=False 可读)。

    数组契约:residuals (V,88,2) float64 回代残差、per_view_rms (V,)
    float64、image_ids (V,) 定长 unicode 对账列(行序与调用方幸存视图
    一一对应)。test 份的单帧评分失败行允许整行 NaN(留痕),train 份恒
    全有限—— NaN 只可能来自 test 评分失败,别无他源。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        image_ids=_unicode_array(image_ids),
        residuals=np.asarray(residuals, dtype=np.float64),
        per_view_rms=np.asarray(per_view_rms, dtype=np.float64),
    )
    return path


def write_extrinsics_npz(path: Path | str, image_ids, rvecs, tvecs) -> Path:
    """写 train 幸存视图的外参 npz(三件套之三,纯诊断用)。

    rvecs/tvecs 各 (V,3) float64,行序与 image_ids 一致。注意:模块 3 的
    双目外参标定自己做 PnP,不消费本文件——它只为人工排查位姿分布而写
    (RESEARCH Pattern 7 三件套模式)。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        image_ids=_unicode_array(image_ids),
        rvecs=np.asarray(rvecs, dtype=np.float64),
        tvecs=np.asarray(tvecs, dtype=np.float64),
    )
    return path


def write_per_frame_csv(path: Path | str, rows: list[dict]) -> Path:
    """写逐帧误差 CSV(utf-8-sig,Excel 双击直开;列序 = _PER_FRAME_FIELDS)。

    行由编排层组装:train 行在前(含被剔除帧,dropped=True + 中文剔除
    说明),test 行在后(恒 dropped=False,note 记评分备注)。rms 列为
    格式化字符串,失败帧留空。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_PER_FRAME_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _fmt_rms(v: float) -> str:
    """报告里的 RMS 显示格式:4 位小数;非有限(无有效帧)显式标 NaN。"""
    return f"{v:.4f}" if np.isfinite(v) else "NaN"


def write_report_md(path: Path | str, per_cam_summaries: list[dict],
                    rms_test_max: float, config_snapshot: dict) -> Path:
    """写单份总报告(两路并排对比,INTR-04;f-string markdown,零模板引擎)。

    per_cam_summaries 每项的键(编排层 _calibrate_one_cam 组装):
        cam / image_size / params(按名 dict)/ solver / refine_enabled /
        rms_train_overall / rms_test_overall /
        n_views_train(幸存)/ n_views_train_initial / n_views_test /
        dropped_frames / outlier_loop(剔除循环结束原因,含保险丝触发说明)/
        train_per_view_rms / test_per_view_rms(逐帧清单,NaN 表示失败帧)。

    判定列按 rms_test_max 给 PASS/FAIL——只作报告结论,不作退出条件
    (OVERNIGHT AUTONOMY:跑完留痕,人不被夜里叫起来看退出码)。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    lines.append("# 内参标定报告")
    lines.append("")
    lines.append(f"- 生成时间:{datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- test 集判定阈值 rms_test_max:{rms_test_max} px(报告判定用,不作退出条件)")
    lines.append(f"- 相机:cam1、cam2 各自独立标定(M2 模型,九参数按名),共 {len(per_cam_summaries)} 路")
    lines.append("")

    # ---- 指标汇总:两路对比表 ----
    lines.append("## 指标汇总")
    lines.append("")
    lines.append("| 相机 | train RMS (px) | test RMS (px) | test 阈值 (px) | 判定 | train 幸存帧数 | train 初始帧数 | test 帧数 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for s in per_cam_summaries:
        ok = np.isfinite(s["rms_test_overall"]) and s["rms_test_overall"] <= rms_test_max
        verdict = "PASS" if ok else "FAIL"
        lines.append(
            f"| cam{s['cam']} | {_fmt_rms(s['rms_train_overall'])} | "
            f"{_fmt_rms(s['rms_test_overall'])} | {rms_test_max} | {verdict} | "
            f"{s['n_views_train']} | {s['n_views_train_initial']} | {s['n_views_test']} |")
    lines.append("")

    # ---- 内参参数表:两路 9 参数按名 ----
    lines.append("## 内参参数(按名)")
    lines.append("")
    lines.append("| 参数 | " + " | ".join(f"cam{s['cam']}" for s in per_cam_summaries) + " |")
    lines.append("|---" * (len(per_cam_summaries) + 1) + "|")
    for name in INTRINSIC_NAMES:
        cells = " | ".join(f"{s['params'][name]:.6g}" for s in per_cam_summaries)
        lines.append(f"| {name} | {cells} |")
    lines.append("")
    lines.append("求解器与开关:" + ";".join(
        f"cam{s['cam']} solver={s['solver']}(refine={'开' if s['refine_enabled'] else '关'})"
        for s in per_cam_summaries))
    lines.append("")

    # ---- 离群帧表:自动剔除留痕(可审计,T-02-08)----
    lines.append("## 离群帧(自动剔除留痕)")
    lines.append("")
    all_dropped = [(s["cam"], d) for s in per_cam_summaries for d in s["dropped_frames"]]
    if all_dropped:
        lines.append("| 相机 | image_id | 轮次 | RMS (px) | 阈值 (px) |")
        lines.append("|---|---|---|---|---|")
        for cam, d in all_dropped:
            lines.append(f"| cam{cam} | {d['image_id']} | {d['round']} | "
                         f"{d['rms']:.4f} | {d['threshold']:.4f} |")
    else:
        lines.append("无(本轮运行没有帧超过离群阈值)。")
    lines.append("")
    for s in per_cam_summaries:
        lines.append(f"- cam{s['cam']} 剔除循环:{s['outlier_loop']}")
    lines.append("")

    # ---- 逐帧误差摘要:每路 train/test 的最高/最低/中位,明细指向 CSV ----
    lines.append("## 逐帧误差摘要")
    lines.append("")
    for s in per_cam_summaries:
        for split, key in (("train", "train_per_view_rms"), ("test", "test_per_view_rms")):
            vals = np.asarray([v for v in s[key] if np.isfinite(v)], dtype=np.float64)
            if vals.size == 0:
                lines.append(f"- cam{s['cam']} {split}:无有效帧(明细见 per_frame_cam{s['cam']}.csv)")
                continue
            lines.append(
                f"- cam{s['cam']} {split}:最高 {_fmt_rms(float(vals.max()))} / "
                f"最低 {_fmt_rms(float(vals.min()))} / 中位 {_fmt_rms(float(np.median(vals)))} px"
                f"(共 {vals.size} 个有效帧;逐帧明细见 per_frame_cam{s['cam']}.csv)")
    lines.append("")

    # ---- 附录:配置快照留痕(照 Phase 1 meta 模式)----
    lines.append("## 附录:配置快照(config_snapshot)")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(config_snapshot, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path
