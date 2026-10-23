"""模块1 数据预处理编排:扫描 → 并行角点检测(SB 主路线 + classic 兜底)→ 汇总落盘。

流程(编排体例移植自参考工程 calib.cli.main 的 _run_corners,只读引用):
读配置 → 扫描数据树 → 运行前清理产物子目录(D-11 防残留)→ 预分配每
(split, cam) 的带掩码稠密数组 → 构造任务元组 → run_parallel 并行跑
detect_one worker(每图一任务;worker 内 SB → classic 两级检测,任何
异常包 TaskFailed 信封)→ 主进程逐个消费填槽 → 按 group_id 配对两路
均检出的公共视场对 → 每相机每数据集一份 npz + meta.json + usable CSV,
每数据集一份 fail CSV + common_fov_pairs CSV → 全部检出帧画角点标注图
(D-07 全量无抽样)+ 失败帧原图 fail_ 前缀副本(开关控制,D-02)→
打印中文汇总。max_workers = 1 时同函数串行回退(调试用,结果与并行一致)。

按 D-01:检出即可用,不加任何额外质量闸门;检测路线记入 npz source 列
与 usable CSV 的 route 列("sb" / "classic")。

数据安全(INFRA-03):全程写路径仅从 cfg.output_root 派生;worker 只读
(imread),全部写操作集中在主进程汇总层,子进程之间不存在 data/ 写竞争
(T-01-08);output_root 与 data 目录相等/互相包含的配置已在 load_config
层被拒。
"""

from __future__ import annotations

import shutil
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import typer

from stereo_calib.corners.detect import detect_classic, detect_sb
from stereo_calib.io.config import StereoCalibConfig
from stereo_calib.io.dataset import imread_checked, scan_dataset
from stereo_calib.preprocess.artifacts import (
    build_object_points,
    save_corners_npz,
    write_common_pairs_csv,
    write_fail_csv,
    write_usable_csv,
)
from stereo_calib.preprocess.parallel import TaskFailed, run_parallel
from stereo_calib.visualization.overlay import copy_fail_original, render_marked

_SPLITS = ("train", "test")
_CAMS = (1, 2)

# 进度播报间隔(纯 CLI 提示,非算法参数):全量几百张,每 100 张报一次
_PROGRESS_EVERY = 100


def detect_one(task: tuple) -> dict:
    """单图检测 worker(顶层函数,在子进程内运行;像素数据不跨进程)。

    task 是可整读的小元组,元素序即下方解包序:
        (path_str, image_id, cam, group_id, split, cols, rows,
         corners_cfg, cv2_threads)
    路径传绝对路径字符串(子进程 cwd 不受主进程影响);corners_cfg 是
    frozen dataclass,可整读序列化,检测参数全部由它带进子进程(零散参数
    不在代码里写死)。worker 内任何异常(含 imread_checked 的读图
    ValueError)由 _guarded_worker 包成 TaskFailed 信封交回,绝不杀任务流。

    返回 {"path", "image_id", "cam", "group_id", "split", "result"};
    result 是 detect_sb / detect_classic 的 dict(双路线都未检出时 None),
    source 就取 result["detector"]("sb" / "classic")。
    """
    (path_str, image_id, cam, group_id, split, cols, rows,
     corners_cfg, cv2_threads) = task
    # 子进程没有经过 CLI callback 的统一准备,OpenCV 线程限制在这里补上
    # (值来自 config 的 cv2_threads,与主进程同源;防多进程叠满核线程)
    cv2.setNumThreads(cv2_threads)
    gray = imread_checked(Path(path_str))
    result = detect_sb(gray, cols, rows, corners_cfg)
    if result is None and corners_cfg.sb_fallback_classic:
        result = detect_classic(gray, cols, rows, corners_cfg)
    return {"path": path_str, "image_id": image_id, "cam": cam,
            "group_id": group_id, "split": split, "result": result}


def _fail_reason(error: BaseException) -> str:
    """TaskFailed 信封 → fail 清单的 reason 标签(按异常类型区分,WR-02)。

    imread_checked 的读图 ValueError(消息含「照片读取失败」)按「读图失败」
    标注;其余 worker 异常(derive_window_frame 的窗口断言 AssertionError、
    检测器 cv2.error、传输/序列化错误等)按「检测失败(异常类型名)」诚实
    标注——原始报错文本两种情况都完整保留,D-02 人工复核不再被一律指向
    「读图」而去找错地方。
    """
    if isinstance(error, ValueError) and "照片读取失败" in str(error):
        return f"读图失败:{error}"
    return f"检测失败({type(error).__name__}):{error}"


def _new_slot(records: list, n_corners: int) -> dict:
    """预分配一个 (split, cam) 组的带掩码稠密数组与清单行容器。

    未检出的帧照常占一行(uv 整行 NaN / detected False / source ""),
    绝不从数据里消失(D-01)。行号由 enumerate 直接给出,不经过
    "从列表构造行索引数组"这一步——空列表会被 numpy 推断成 float64,
    拿它做花式索引当场 IndexError(参考工程踩过的坑,此处结构性规避)。
    """
    n = len(records)
    return {
        "records": records,
        "image_ids": [r.image_id for r in records],
        "paths": [str(r.path) for r in records],
        "uv": np.full((n, n_corners, 2), np.nan),
        "sq_px": np.full(n, np.nan),
        "win_size": np.full(n, np.nan),
        "detected": np.zeros(n, dtype=bool),
        "source": np.full(n, "", dtype="<U16"),
        "usable_rows": [],
        "fail_rows": [],
        "fail_reasons": {},
    }


def run_preprocess(cfg: StereoCalibConfig) -> None:
    """模块1 主流程:扫描 → 并行角点检测(SB + classic 兜底)→ 落盘 → 中文汇总。"""
    board = cfg.board
    cols, rows = board.cols, board.rows
    n_corners = cols * rows

    scan = scan_dataset(cfg.data)
    for w in scan.warnings:
        typer.echo(f"警告:{w}")

    out_root = Path(cfg.output_root)

    # ---- 运行前清理(D-11 固定覆盖 + Pitfall 7 防残留)----
    # 只删 output_root/{train} 与 output_root/{test} 两个产物子目录,删后随
    # 落盘按需重建;上次运行的 visualization/、fail_originals/、npz、CSV、meta
    # 残留(如本次检出变少)不会误导人工复核。output_root 本身与其他未知
    # 内容不动。数据安全前提(不重复实现):load_config 已在加载层拒绝
    # output_root 与 data 目录相等/互相包含的配置(01-01 锁定),这里的
    # rmtree 只可能命中 output_root 之下的产物目录,不可能触及 data/。
    for split in _SPLITS:
        shutil.rmtree(out_root / split, ignore_errors=True)

    groups = {
        (split, cam): _new_slot(
            [r for r in scan.records if r.split == split and r.cam == cam], n_corners)
        for split in _SPLITS for cam in _CAMS
    }
    fail_by_split: dict[str, list[dict]] = {split: [] for split in _SPLITS}

    # 任务元组只带绝对路径字符串 + 标识 + 棋盘规格 + 检测配置(像素数据
    # 不跨进程);路径 → (槽, 行号, 记录) 的索引让消费端不依赖结果顺序
    tasks: list[tuple] = []
    slot_of: dict[str, tuple[dict, int, object]] = {}
    for (split, cam), slot in groups.items():
        for j, rec in enumerate(slot["records"]):
            path_str = str(rec.path.resolve())
            tasks.append((path_str, rec.image_id, rec.cam, rec.group_id,
                          rec.split, cols, rows, cfg.preprocess.corners,
                          cfg.cv2_threads))
            slot_of[path_str] = (slot, j, rec)

    total = len(tasks)
    mode = "串行" if cfg.preprocess.max_workers <= 1 else f"{cfg.preprocess.max_workers} 进程并行"
    typer.echo(f"开始预处理:共 {total} 张图像(SB 主路线 + classic 兜底,{mode})……")

    def _progress(info: dict) -> None:
        if info["done"] % _PROGRESS_EVERY == 0 or info["done"] == info["total"]:
            typer.echo(f"  已处理 {info['done']}/{info['total']} 张……")

    def _record_fail(slot: dict, rec, reason: str) -> None:
        """单帧失败不中断全批:记 fail 行(清单 + meta 留痕),继续下一帧。"""
        row = {"image_id": rec.image_id, "path": str(rec.path),
               "cam": rec.cam, "reason": reason}
        slot["fail_rows"].append(row)
        slot["fail_reasons"][rec.image_id] = reason
        fail_by_split[rec.split].append(row)

    # run_parallel 按任务清单顺序产出;zip 把结果对回任务(信封本身不带
    # 任务标识,靠同一位置的 task 元组识别是哪一帧)。
    consumed = 0
    for task, outcome in zip(tasks, run_parallel(
            cfg.preprocess.max_workers, tasks, worker=detect_one,
            progress=_progress)):
        consumed += 1
        slot, j, rec = slot_of[task[0]]
        if isinstance(outcome, TaskFailed):
            # 拆信封:该帧记 fail,绝不重抛(单帧坏图只废它自己);
            # 标签按异常类型区分(_fail_reason),原始报错文本完整保留
            _record_fail(slot, rec, _fail_reason(outcome.error))
            continue
        result = outcome["result"]
        if result is None:
            _record_fail(slot, rec, "SB 与 classic 均未检出")
            continue
        detector = result["detector"]  # "sb" / "classic"
        slot["uv"][j] = result["uv"]
        slot["sq_px"][j] = result["sq_px"]
        slot["win_size"][j] = result["win_size"]
        slot["detected"][j] = True
        slot["source"][j] = detector
        slot["usable_rows"].append({
            "image_id": rec.image_id, "path": str(rec.path), "cam": rec.cam,
            "route": detector, "sq_px": f"{result['sq_px']:.2f}",
        })
    if consumed != total:
        # 信封契约被击穿才会走到这里(joblib 任务流被截断):宁可全批报错,
        # 绝不静默少写行——npz/CSV 的行数契约必须与扫描记录数一致
        raise ValueError(
            f"并行任务流不完整:应到 {total} 个结果,实际收到 {consumed} 个。"
            f"这是异常信封防线被击穿的信号,请把本错误连同运行日志反馈维护者。"
        )

    # ---- 公共视场配对(PREP-04;编排层按组 ID 配对,只在主进程做,不进 worker)----
    # 同 group_id(叶子目录名)下 cam1 与 cam2 都检出才产一行;单边存在或
    # 单边检出 → 不入公共清单,也不额外记 fail:fail 清单只记检测失败,
    # 该帧的检出侧已在 usable 清单、失败对侧已在 fail 清单(D-01),把
    # "单边检出"误标为失败只会带来无谓的原图拷贝与复核噪音。
    common_rows_by_split: dict[str, list[dict]] = {split: [] for split in _SPLITS}
    for split in _SPLITS:
        detected_by_group: dict[str, dict[int, tuple]] = {}
        for cam in _CAMS:
            slot = groups[(split, cam)]
            for j, rec in enumerate(slot["records"]):
                if slot["detected"][j]:
                    detected_by_group.setdefault(rec.group_id, {})[cam] = (
                        rec, str(slot["source"][j]))
        for group_id in sorted(detected_by_group):
            cams_in_group = detected_by_group[group_id]
            if 1 in cams_in_group and 2 in cams_in_group:
                common_rows_by_split[split].append({
                    "group_id": group_id,
                    "cam1_path": str(cams_in_group[1][0].path),
                    "cam2_path": str(cams_in_group[2][0].path),
                    "route_cam1": cams_in_group[1][1],
                    "route_cam2": cams_in_group[2][1],
                })

    # ---- 落盘(D-05:output_root/{train,test}/ 分目录,D-11 直接覆盖)----
    objp = build_object_points(board)
    artifacts: list[Path] = []
    for split in _SPLITS:
        split_dir = out_root / split
        for cam in _CAMS:
            slot = groups[(split, cam)]
            if not slot["records"]:
                continue  # 空划分(如缺 test 目录)不产出文件,警告已在扫描层记过
            npz_path = save_corners_npz(
                split_dir, f"cam{cam}_{split}",
                image_ids=slot["image_ids"], paths=slot["paths"], uv=slot["uv"],
                objp=objp, source=slot["source"], detected=slot["detected"],
                sq_px=slot["sq_px"], win_size=slot["win_size"],
                meta={
                    "split": split,
                    "cam": cam,
                    "common_pairs": len(common_rows_by_split[split]),
                    "board": {"cols": cols, "rows": rows,
                              "square_size_mm": board.square_size_mm,
                              "unit": board.unit},
                    "config_snapshot": {"preprocess": asdict(cfg.preprocess)},
                    "scan_warnings": list(scan.warnings),
                    "fail_reasons": slot["fail_reasons"],
                },
            )
            usable_path = write_usable_csv(
                split_dir / f"usable_cam{cam}_{split}.csv", slot["usable_rows"])
            artifacts.extend([npz_path, split_dir / f"cam{cam}_{split}.meta.json",
                              usable_path])
        fail_path = write_fail_csv(split_dir / f"fail_{split}.csv", fail_by_split[split])
        common_path = write_common_pairs_csv(
            split_dir / f"common_fov_pairs_{split}.csv", common_rows_by_split[split])
        artifacts.extend([fail_path, common_path])

    # ---- 全量角点标注图 + 失败帧原图副本(PREP-06/D-07/D-08/D-02)----
    # 全部检出帧一张不落(D-07 绝不抽样——复核要挑"检出但角点偏移"的帧,
    # 抽样会漏);标注画在内存副本上,原图只 imread(INFRA-03)。
    # 路线文字用 npz 同一份 source 列的值("sb"/"classic"),翻图即知路线。
    n_marked = 0
    for split in _SPLITS:
        vis_dir = out_root / split / "visualization"
        for cam in _CAMS:
            slot = groups[(split, cam)]
            for j, rec in enumerate(slot["records"]):
                if not slot["detected"][j]:
                    continue
                render_marked(
                    rec.path, slot["uv"][j], cols, rows,
                    route=str(slot["source"][j]),
                    out_path=vis_dir / f"{rec.image_id}.jpg",
                    jpeg_quality=cfg.preprocess.jpeg_quality,
                )
                n_marked += 1

    # 失败帧原图副本(D-02 假失败复核入口):逐字节拷贝 fail_ 前缀副本,
    # 开关 copy_fail_originals 由配置控制(关掉就不产目录)
    n_copied = 0
    if cfg.preprocess.copy_fail_originals:
        for split in _SPLITS:
            fail_dir = out_root / split / "fail_originals"
            for row in fail_by_split[split]:
                copy_fail_original(Path(row["path"]), row["image_id"], fail_dir)
                n_copied += 1

    # ---- 中文汇总:逐 split×cam 检出率、方格边长统计、fail 计数、产物路径 ----
    typer.echo("预处理完成,产物已写盘。检出统计(检出 / 总数):")
    for split in _SPLITS:
        for cam in _CAMS:
            slot = groups[(split, cam)]
            if not slot["records"]:
                continue
            n_det = int(slot["detected"].sum())
            n_all = len(slot["records"])
            typer.echo(f"  cam{cam}/{split}:{n_det}/{n_all}({n_det / n_all * 100:.1f}%)")
    sq_all = np.concatenate([
        g["sq_px"][g["detected"]] for g in groups.values() if len(g["records"])])
    if sq_all.size:
        typer.echo(f"  方格像素边长(检出帧):中位数 {float(np.median(sq_all)):.1f}px"
                   f"(范围 {float(sq_all.min()):.1f} ~ {float(sq_all.max()):.1f})")
    total_fail = sum(len(v) for v in fail_by_split.values())
    n_sb = sum(int((g["source"] == "sb").sum()) for g in groups.values())
    n_classic = sum(int((g["source"] == "classic").sum()) for g in groups.values())
    typer.echo(f"  失败帧合计:{total_fail} 张(逐张原因见 fail_*.csv,人工复核入口 D-02)")
    typer.echo(f"  检测路线分布:sb {n_sb} 帧、classic {n_classic} 帧"
               f"(classic = SB 未检出、由经典检测器兜住的帧)")
    typer.echo("  公共视场对(同组两路均检出,common_fov_pairs_*.csv,Phase 3 双目标定输入清单):")
    for split in _SPLITS:
        if not any(groups[(split, cam)]["records"] for cam in _CAMS):
            continue
        typer.echo(f"    {split}:{len(common_rows_by_split[split])} 对")
    typer.echo(f"  角点标注图:{n_marked} 张(output_root/{{train,test}}/visualization/,"
               f"全部检出帧无抽样,图上烧录路径与路线 D-07/D-08)")
    if cfg.preprocess.copy_fail_originals:
        typer.echo(f"  失败帧原图副本:{n_copied} 张(fail_ 前缀,供人工复核假失败 D-02)")
    else:
        typer.echo("  失败帧原图拷贝已关闭(preprocess.copy_fail_originals: false)")
    typer.echo("  全部产物(绝对路径):")
    for p in artifacts:
        typer.echo(f"    {Path(p).resolve()}")
