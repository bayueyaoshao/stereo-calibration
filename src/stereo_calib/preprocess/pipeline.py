"""模块1 数据预处理编排:扫描 → 逐帧角点检测 → 汇总落盘 → 中文汇总。

流程(编排体例移植自参考工程 calib.cli.main 的 _run_corners,只读引用):
读配置 → 扫描数据树 → 预分配每 (split, cam) 的带掩码稠密数组 → 串行逐记录
检测填槽 → 每相机每数据集一份 npz + meta.json + usable CSV,每数据集一份
fail CSV → 打印中文汇总。

本计划(01-01)只跑 SB 主路线;01-02 把串行循环替换为 run_parallel +
detect_one worker(classic 兜底同批接入),结果组装逻辑不变。
按 D-01:检出即可用,不加任何额外质量闸门。

数据安全(INFRA-03):全程写路径仅从 cfg.output_root 派生,data/ 只 imread;
output_root 与 data 目录相等/互相包含的配置已在 load_config 层被拒。
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import typer

from stereo_calib.corners.detect import detect_sb
from stereo_calib.io.config import StereoCalibConfig
from stereo_calib.io.dataset import imread_checked, scan_dataset
from stereo_calib.preprocess.artifacts import (
    build_object_points,
    save_corners_npz,
    write_fail_csv,
    write_usable_csv,
)

_SPLITS = ("train", "test")
_CAMS = (1, 2)


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
    """模块1 主流程:扫描 → SB 角点检测 → npz/CSV/meta 落盘 → 中文汇总。"""
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

    total = len(scan.records)
    typer.echo(f"开始预处理:共 {total} 张图像(角点检测 SB 主路线,"
               f"classic 兜底与并行由后续计划接入)……")
    done = 0

    def _record_fail(slot: dict, rec, reason: str) -> None:
        """单帧失败不中断全批:记 fail 行(清单 + meta 留痕),继续下一帧。"""
        row = {"image_id": rec.image_id, "path": str(rec.path),
               "cam": rec.cam, "reason": reason}
        slot["fail_rows"].append(row)
        slot["fail_reasons"][rec.image_id] = reason
        fail_by_split[rec.split].append(row)

    for (split, cam), slot in groups.items():
        for j, rec in enumerate(slot["records"]):
            done += 1
            try:
                gray = imread_checked(rec.path)
            except ValueError as e:
                _record_fail(slot, rec, f"读图失败:{e}")
                continue
            result = detect_sb(gray, cols, rows, cfg.preprocess.corners)
            if result is None:
                # 01-02 classic 兜底落地后,该 reason 升级为双路线措辞
                _record_fail(slot, rec, "SB 未检出")
                continue
            slot["uv"][j] = result["uv"]
            slot["sq_px"][j] = result["sq_px"]
            slot["win_size"][j] = result["win_size"]
            slot["detected"][j] = True
            slot["source"][j] = "sb"
            slot["usable_rows"].append({
                "image_id": rec.image_id, "path": str(rec.path), "cam": rec.cam,
                "route": "sb", "sq_px": f"{result['sq_px']:.2f}",
            })
            if done % 100 == 0:
                typer.echo(f"  已处理 {done}/{total} 张……")

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
    typer.echo(f"  失败帧合计:{total_fail} 张(逐张原因见 fail_*.csv,人工复核入口 D-02)")
    typer.echo("  全部产物(绝对路径):")
    for p in artifacts:
        typer.echo(f"    {Path(p).resolve()}")
