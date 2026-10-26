"""模块4 对装载层:data/val 扫描 + 既有检测链复用(禁第二份检测)。

数据布局(PLAN 规格):data_dir 下 <组目录>/cam1.jpg + cam2.jpg(rglob 扫描,
容忍更深嵌套);group_id = 组目录名(公共配对键,与 train/test 同名约定)。
单边缺文件的组记中文警告跳过(与 Phase 1 扫描韧性同款);同名叶子目录重复
也警告跳过后出现者(防跨拍摄错配,CR-01 同源纪律)。

角点检测**只准走既有 preprocess.detect_one**(SB 主路线 + classic 兜底 +
canonicalize 规范化,角点行列朝向消歧与训练口径严格一致)——本文件绝不
重写任何检测/规范化逻辑,只负责组装 detect_one 的任务元组并用
preprocess.parallel.run_parallel 调度(单帧失败 TaskFailed 信封,任务流
不断)。两路都检出的组才算有效验证对;仅单路检出如实记失败备注。

装载防线(照 stereo.observation 体例):幸存对两路 uv np.isfinite 全真
断言——检出但含 NaN 直接中文报错,绝不带毒进测量链;角点数与棋盘规格
逐对核对(88 点),不合中文报错。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stereo_calib.io.config import BoardSpec
from stereo_calib.preprocess.parallel import TaskFailed, run_parallel
from stereo_calib.preprocess.pipeline import detect_one


@dataclass(frozen=True)
class ValGroup:
    """一个验证组:两路同步图像的路径档案(group_id 为配对键)。"""

    group_id: str
    cam1_path: Path
    cam2_path: Path


def scan_val_groups(data_dir: Path | str) -> tuple[list[ValGroup], list[str]]:
    """扫描验证根目录:<组目录>/cam1.jpg + cam2.jpg 齐全的组入册。

    返回(组清单按 group_id 字典序, 中文警告清单)。根目录不存在、组单边
    缺文件、同名组目录重复均中文警告/报错,绝不静默。
    """
    root = Path(data_dir)
    if not root.is_dir():
        raise ValueError(
            f"验证数据目录 {root} 不存在:请检查 config.yaml 的 "
            f"validation.data_dir(相对当前工作目录,期望结构 "
            f"<组目录>/cam1.jpg + cam2.jpg)。")
    warnings: list[str] = []
    by_group: dict[str, dict[int, Path]] = {}
    for cam1 in sorted(root.rglob("cam1.jpg")):
        group_key = cam1.parent.relative_to(root).as_posix()
        if group_key in by_group:
            warnings.append(
                f"验证组目录重名:{group_key} 出现多次,跳过后出现者({cam1.parent})——"
                f"防跨拍摄错配,请重命名组目录。")
            continue
        cams: dict[int, Path] = {1: cam1}
        cam2 = cam1.parent / "cam2.jpg"
        if cam2.is_file():
            cams[2] = cam2
        else:
            warnings.append(
                f"验证组 {group_key} 只有 cam1.jpg、缺 cam2.jpg——该组只入单路,"
                f"双目验证不可用,已跳过。")
        by_group[group_key] = cams
    for cam2 in sorted(root.rglob("cam2.jpg")):
        group_key = cam2.parent.relative_to(root).as_posix()
        if group_key not in by_group:
            warnings.append(
                f"验证组 {group_key} 只有 cam2.jpg、缺 cam1.jpg——该组只入单路,"
                f"双目验证不可用,已跳过。")
            by_group.setdefault(group_key, {2: cam2})

    groups = [ValGroup(group_id=key.split("/")[-1],
                       cam1_path=by_group[key][1], cam2_path=by_group[key][2])
              for key in sorted(by_group) if 1 in by_group[key] and 2 in by_group[key]]
    return groups, warnings


@dataclass(frozen=True)
class ValPair:
    """一个有效验证对:两路均检出的角点(规范化序)+ 检测路线留痕。"""

    group_id: str
    uv1: np.ndarray            # (P, 2) float64,canonicalize 后
    uv2: np.ndarray            # (P, 2) float64,canonicalize 后
    route1: str                # "sb" / "classic"
    route2: str


def detect_val_pairs(groups: list[ValGroup], board: BoardSpec,
                     corners_cfg, cv2_threads: int, max_workers: int,
                     progress=None) -> tuple[list[ValPair], dict[str, str]]:
    """对全部验证组跑既有 detect_one 检测链,装配两路均检出的有效对。

    任务元组与 preprocess 管线逐字段同构(split 位记 "val"),worker 就是
    preprocess.pipeline.detect_one 本体(SB → classic 兜底 + canonicalize
    全在里面)——检测口径与训练零漂移。单帧失败(TaskFailed 信封或两检测器
    都未检出)记入失败备注不中断整批;返回(有效对清单, 失败备注
    {f"{gid}_cam{n}": 中文原因})。
    """
    n_corners = board.rows * board.cols
    tasks: list[tuple] = []
    for group in groups:
        for cam, path in ((1, group.cam1_path), (2, group.cam2_path)):
            tasks.append((str(Path(path).resolve()), f"{group.group_id}_cam{cam}",
                          cam, group.group_id, "val", board.cols, board.rows,
                          corners_cfg, cv2_threads))

    detected: dict[tuple[str, int], tuple[np.ndarray, str]] = {}
    fail_notes: dict[str, str] = {}
    for task, outcome in zip(tasks, run_parallel(
            max_workers, tasks, worker=detect_one, progress=progress)):
        _path_str, image_id, cam, gid, _split, *_rest = task
        if isinstance(outcome, TaskFailed):
            fail_notes[image_id] = f"检测失败({type(outcome.error).__name__}):{outcome.error}"
            continue
        result = outcome["result"]
        if result is None:
            fail_notes[image_id] = "SB 与 classic 均未检出"
            continue
        detected[(gid, cam)] = (np.asarray(result["uv"], dtype=np.float64),
                                str(result["detector"]))

    pairs: list[ValPair] = []
    for group in groups:
        hit1 = detected.get((group.group_id, 1))
        hit2 = detected.get((group.group_id, 2))
        if hit1 is None or hit2 is None:
            # 单边检出/失败的组不成对:两侧病因都已在 fail_notes(或补记)
            if hit1 is not None:
                fail_notes.setdefault(f"{group.group_id}_cam2", "仅单路检出,不成对")
            if hit2 is not None:
                fail_notes.setdefault(f"{group.group_id}_cam1", "仅单路检出,不成对")
            continue
        uv1, route1 = hit1
        uv2, route2 = hit2
        for cam, uv in ((1, uv1), (2, uv2)):
            if uv.shape != (n_corners, 2):
                raise ValueError(
                    f"验证组 {group.group_id} cam{cam} 的角点数不合法:期望 "
                    f"{n_corners} 个(当前棋盘 {board.rows}×{board.cols}),"
                    f"实际 {uv.shape}。检测产物可疑,请重跑。")
            if not np.isfinite(uv).all():
                raise ValueError(
                    f"验证组 {group.group_id} cam{cam} 的角点混入非数值(NaN/无穷),"
                    f"检测产物可疑,请重跑。")
        if uv1.shape != uv2.shape:  # 两路角点数必须一致(同一棋盘)
            raise ValueError(
                f"验证组 {group.group_id} 两路角点数不一致:cam1 {uv1.shape} vs "
                f"cam2 {uv2.shape}——两路看到的不是同一块板,该组可疑。")
        pairs.append(ValPair(group_id=group.group_id, uv1=uv1, uv2=uv2,
                             route1=route1, route2=route2))
    return pairs, fail_notes
