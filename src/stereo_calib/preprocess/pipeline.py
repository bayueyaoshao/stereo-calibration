"""模块1 数据预处理编排:扫描 → 并行角点检测(SB 主路线 + classic 兜底)→ 汇总落盘。

流程(编排体例移植自参考工程 calib.cli.main 的 _run_corners,只读引用):
读配置 → 扫描数据树 → 预分配每 (split, cam) 的带掩码稠密数组 → 构造任务
元组 → run_parallel 并行跑 detect_one worker(每图一任务;worker 内
SB → classic 两级检测,任何异常包 TaskFailed 信封)→ 主进程逐个消费
填槽 → 每相机每数据集一份 npz + meta.json + usable CSV,每数据集一份
fail CSV → 打印中文汇总。max_workers = 1 时同函数串行回退(调试用,
结果与并行一致)。

按 D-01:检出即可用,不加任何额外质量闸门;检测路线记入 npz source 列
与 usable CSV 的 route 列("sb" / "classic")。

数据安全(INFRA-03):全程写路径仅从 cfg.output_root 派生;worker 只读
(imread),全部写操作集中在主进程汇总层,子进程之间不存在 data/ 写竞争
(T-01-08);output_root 与 data 目录相等/互相包含的配置已在 load_config
层被拒。
"""

from __future__ import annotations

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
    write_fail_csv,
    write_usable_csv,
)
from stereo_calib.preprocess.parallel import TaskFailed, run_parallel

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
            # 拆信封:该帧记 fail,绝不重抛(单帧坏图只废它自己)
            _record_fail(slot, rec, f"读图失败:{outcome.error}")
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
        artifacts.append(fail_path)

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
    typer.echo("  全部产物(绝对路径):")
    for p in artifacts:
        typer.echo(f"    {Path(p).resolve()}")
