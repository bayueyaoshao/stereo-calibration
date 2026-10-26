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

from stereo_calib.intrinsic.models import intrinsic_names
from stereo_calib.io.config import BoardSpec

# 逐帧 CSV 列序契约(INTR-04):utf-8-sig,一行一帧,train 在前 test 在后
_PER_FRAME_FIELDS = ["image_id", "split", "source", "rms", "dropped", "note"]

# intrinsics.json 顶层字段全集(Pattern 7 契约;Phase 3 按名读取的唯一接口)
# 2026-10-24 增补 "diagnostics"(模型充分性体检,只增不改既有字段)
# 与 "distortion_model"(M2/M8 布局自述,只增不改既有字段;读取方对缺该键
# 的旧产物默认 M2,向后兼容)
_INTRINSICS_JSON_KEYS = (
    "cam", "distortion_model", "image_size", "params", "K", "D", "solver",
    "refine_enabled", "rms_train_overall", "rms_test_overall",
    "n_views_train", "n_views_test", "dropped_frames", "solver_meta",
    "board", "config_snapshot", "diagnostics",
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
                          board: BoardSpec, config_snapshot: dict,
                          diagnostics: dict,
                          distortion_model: str = "M2") -> Path:
    """写一路相机的内参契约文件(Pattern 7,模块 3 的唯一固定内参输入)。

    字段层级(与 RESEARCH Pattern 7 逐字段一致;2026-10-24 增补 diagnostics
    与 distortion_model,只增不改既有字段):
        cam:               相机号(int);
        distortion_model:  "M2" | "M8" | "M4"(params/D 布局自述;读取方对缺该键的
                           旧产物默认 M2,向后兼容);
        image_size:        [宽, 高],与标定同源口径;
        params:            按模型布局按名(M2 九参 fx/fy/cx/cy/k1/k2/p1/p2/k3;
                           M8 十二参再加 k4/k5/k6;M4 八参 fx/fy/cx/cy/k1..k4);
        K:                 3×3 冗余存储(避免模块 3 重建出错);
        D:                 OpenCV dist 布局冗余存储(M2 5 长;M8 8 长
                           [k1,k2,p1,p2,k3,k4,k5,k6];M4 4 长 [k1,k2,k3,k4]
                           ——鱼眼与针孔的 D 不同义,必须凭本键解读);
        solver:            "opencv" | "scipy"(精化开关关闭时为前者);
        refine_enabled:    精化开关留痕;
        rms_train_overall / rms_test_overall:float64 回代 pooled RMS;
        n_views_train / n_views_test:帧数(train 为剔除后幸存数);
        dropped_frames:    [{"image_id", "rms", "round", "threshold"}, …]
                           自动剔除留痕(T-02-08 可审计);
        solver_meta:       status/nfev/opencv_rms 等对账信息,不静默;
        board:             棋盘规格快照;config_snapshot:intrinsic 段快照;
        diagnostics:       模型充分性体检报告(各检查数值/阈值/red/notes/
                           verdict/red_count/enabled,JSON 可序列化 dict)。
    """
    d_arr = np.asarray(D, dtype=np.float64).ravel()
    names = intrinsic_names(distortion_model)  # 模型名白名单防线(非法名中文报错)
    # D 长度 ↔ 布局对照表(M4 鱼眼 4 长 / M2 五长 / M8 rational 八长)。
    # 三个长度互不相同,checkpoint 表在这里明示而非 if-else 逐写
    # ——将来加布局只需加一行。
    expected_d = {"M2": 5, "M8": 8, "M4": 4}[distortion_model]
    assert d_arr.shape == (expected_d,), (
        f"畸变向量必须恰 {expected_d} 长(模型 {distortion_model} 的 OpenCV "
        f"dist 布局 {names}),实际是 {d_arr.shape}。出现此错说明调用方传入的 "
        f"params 布局与 distortion_model 不一致,请检查 params。")
    payload = {
        "cam": int(cam),
        "distortion_model": str(distortion_model),
        "image_size": [int(image_size[0]), int(image_size[1])],
        "params": {name: float(v) for name, v in zip(names, params)},
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
        "diagnostics": diagnostics,
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


def _fmt_diag_value(v: float | None) -> str:
    """体检判定量显示:None(数据不足/探针失败)显式标注,不冒充数值。"""
    return "数据不足" if v is None else f"{v:.4f}"


def _render_diagnostics_section(lines: list[str], per_cam_summaries: list[dict],
                                config_snapshot: dict) -> None:
    """模型充分性体检小节:判定规则一行 + 逐检查数值表格 + verdict(留痕)。

    判「不可靠」的相机以 **FAIL ⚠** 醒目标红;体检未启用/该路失败的相机
    如实注明,不编造体检结果。diagnostics 结构来自编排层
    diagnostics_to_dict(enabled/verdict/reliable/red_count/red_names/
    notes/checks[name,label,value,threshold,red,note])。
    """
    lines.append("## 模型充分性体检")
    lines.append("")
    snap = config_snapshot.get("intrinsic", {}) \
        if isinstance(config_snapshot, dict) else {}
    diag_model = snap.get("distortion_model", "M2") if isinstance(snap, dict) else "M2"
    lines.append(f"畸变模型:{diag_model}(A4 探针在 M8 产线解下为同阶 rational8 "
                 "独立重标一致性检验,M2 下为升阶探针)。")
    snap_diag = snap.get("diagnostics", {}) if isinstance(snap, dict) else {}
    min_red = snap_diag.get("min_red_count") if isinstance(snap_diag, dict) else None
    rule = f"A0/A1/A3/A4 任意 {min_red} 项" if min_red is not None \
        else "A0/A1/A3/A4 红灯数达 min_red_count"
    lines.append(f"判定规则:{rule}红灯 → 判「内参不可靠」(fail-closed:禁止消费该内参"
                 "进入双目标定);A2 去畸变直线度仅报告不判红。")
    lines.append("")
    for s in per_cam_summaries:
        d = s.get("diagnostics")
        if d is None:
            reason = "该路标定失败" if s.get("failed") else "体检未运行"
            lines.append(f"- cam{s['cam']}:无体检结果({reason})。")
            continue
        if not d.get("enabled", False):
            lines.append(f"- cam{s['cam']}:体检未启用(diagnostics.enabled: false),"
                         "内参可靠性未判定。")
            continue
        lines.append(f"**cam{s['cam']}**")
        lines.append("")
        lines.append("| 检查 | 判定量 | 阈值 | 结果 | 说明 |")
        lines.append("|---|---|---|---|---|")
        for c in d["checks"]:
            thr = "-" if c["threshold"] is None else f"{c['threshold']:.4g}"
            result = "红灯" if c["red"] else "通过"
            note = c["note"] if c["name"] != "A2" or c["note"] else ""
            lines.append(f"| {c['name']} {c['label']} | {_fmt_diag_value(c['value'])} | "
                         f"{thr} | {result} | {note} |")
        lines.append("")
        if d.get("reliable"):
            lines.append(f"- cam{s['cam']} 体检结论:PASS 可靠(红灯 {d['red_count']} 项,"
                         "低于判定线)")
        else:
            names = ",".join(d["red_names"]) or "未命名"
            lines.append(f"- cam{s['cam']} 体检结论:**FAIL ⚠ 不可靠(红灯 {names} "
                         f"共 {d['red_count']} 项)——畸变模型不充分,内参不可靠**")
        for n in d["notes"]:
            lines.append(f"  - {n}")
        lines.append("")


def write_report_md(path: Path | str, per_cam_summaries: list[dict],
                    rms_test_max: float, config_snapshot: dict) -> Path:
    """写单份总报告(两路并排对比,INTR-04;f-string markdown,零模板引擎)。

    per_cam_summaries 每项的键(编排层 _calibrate_one_cam 组装):
        cam / image_size / params(按名 dict)/ solver / refine_enabled /
        rms_train_overall / rms_test_overall /
        n_views_train(幸存)/ n_views_train_initial / n_views_test /
        dropped_frames / outlier_loop(剔除循环结束原因,含保险丝触发说明)/
        train_per_view_rms / test_per_view_rms(逐帧清单,NaN 表示失败帧)/
        diagnostics(模型充分性体检 dict,diagnostics_to_dict 产出;
        失败路无此键)。

    整路失败记录(编排层单路失败收口组装)只带 cam / failed=True / error
    与中性空字段:指标表按 FAIL 渲染,失败留痕节写明病因,参数表只列
    成功路——报告对失败路如实留痕,不编造指标。

    判定列按 rms_test_max 给 PASS/FAIL——只作报告结论,不作退出条件
    (OVERNIGHT AUTONOMY:跑完留痕,人不被夜里叫起来看退出码)。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    # 畸变模型从配置快照取(一次运行一份配置 → 两路同一模型)
    snap_intrinsic = config_snapshot.get("intrinsic", {}) \
        if isinstance(config_snapshot, dict) else {}
    model = snap_intrinsic.get("distortion_model", "M2") \
        if isinstance(snap_intrinsic, dict) else "M2"
    names = intrinsic_names(model)

    lines.append("# 内参标定报告")
    lines.append("")
    lines.append(f"- 生成时间:{datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- test 集判定阈值 rms_test_max:{rms_test_max} px(报告判定用,不作退出条件)")
    lines.append(f"- 相机:cam1、cam2 各自独立标定({model} 模型,{len(names)} 参数按名),"
                 f"共 {len(per_cam_summaries)} 路")
    lines.append("")

    # ---- 指标汇总:两路对比表 ----
    lines.append("## 指标汇总")
    lines.append("")
    lines.append("| 相机 | train RMS (px) | test RMS (px) | test 阈值 (px) | 判定 | train 幸存帧数 | train 初始帧数 | test 帧数 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for s in per_cam_summaries:
        if s.get("failed"):
            lines.append(
                f"| cam{s['cam']} | 失败 | 失败 | {rms_test_max} | FAIL | - | - | - |")
            continue
        ok = np.isfinite(s["rms_test_overall"]) and s["rms_test_overall"] <= rms_test_max
        verdict = "PASS" if ok else "FAIL"
        lines.append(
            f"| cam{s['cam']} | {_fmt_rms(s['rms_train_overall'])} | "
            f"{_fmt_rms(s['rms_test_overall'])} | {rms_test_max} | {verdict} | "
            f"{s['n_views_train']} | {s['n_views_train_initial']} | {s['n_views_test']} |")
    lines.append("")

    # ---- 失败留痕:整路失败如实记录(其余路不受影响)----
    failed_cams = [s for s in per_cam_summaries if s.get("failed")]
    if failed_cams:
        lines.append("## 失败留痕")
        lines.append("")
        lines.append("以下相机整路失败(其余路不受影响,产物已正常落盘):")
        lines.append("")
        for s in failed_cams:
            lines.append(f"- cam{s['cam']}:{s['error']}")
        lines.append("")

    # ---- 内参参数表:成功路按模型布局按名(M2 九参 / M8 十二参)----
    ok_cams = [s for s in per_cam_summaries if not s.get("failed")]
    lines.append("## 内参参数(按名)")
    lines.append("")
    if not ok_cams:
        lines.append("本批无成功标定的相机,无内参参数可列。")
        lines.append("")
    else:
        lines.append("| 参数 | " + " | ".join(f"cam{s['cam']}" for s in ok_cams) + " |")
        lines.append("|---" * (len(ok_cams) + 1) + "|")
        for name in names:
            cells = " | ".join(f"{s['params'][name]:.6g}" for s in ok_cams)
            lines.append(f"| {name} | {cells} |")
        lines.append("")
        lines.append("求解器与开关:" + ";".join(
            f"cam{s['cam']} solver={s['solver']}(refine={'开' if s['refine_enabled'] else '关'})"
            for s in ok_cams))
        lines.append("")

    # ---- 模型充分性体检:畸变模型是否充分(fail-closed 判定留痕)----
    _render_diagnostics_section(lines, per_cam_summaries, config_snapshot)

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
