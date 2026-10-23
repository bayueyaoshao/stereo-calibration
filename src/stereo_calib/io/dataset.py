"""数据扫描:双层嵌套双目数据树的只读发现(全部数据驱动,绝不写死张数)。

数据布局契约(实测结构,容忍 <ts>/<ts>_<us> 两层嵌套,不硬编码层数语义):
- data/{train,test}/<目录>/<组目录>/cam1.jpg + cam2.jpg
- 组目录(group_id)= 叶子目录名,是后续"公共视场对"的配对键;
- image_id = 叶子目录名 + "_cam{n}"——文件名 stem 全库只有 cam1/cam2 两个,
  直接用 stem 必碰撞,绝不能用(Pitfall 3);
- 同名叶子目录出现在同划分不同父目录下:按「父/叶子」相对路径分组,成为
  两个独立组,绝不静默合并;产生的同 image_id 由 scan_dataset 去重
  (中文警告 + 跳过后出现者),不存在跨拍摄的双目错配(CR-01)。

韧性规则(换数据集不崩不静默):
- 划分目录不存在:中文警告 + 按 0 张处理(只有训练集是合法输入);
- 组目录里没有有效双目图像(空目录/文件名不合模式):中文警告跳过;
- 单边缺文件(某组只有一路 cam):中文警告,存在的那路照常入册
  (内参标定可用,只是双目外参不可用);
- 文件名不是 cam1.jpg/cam2.jpg(如 cam3.jpg):中文告警跳过;
- train 与 test 都扫不出任何记录才 raise ValueError。

image_id 会进入输出文件名,过 _SAFE_IMAGE_ID 白名单正则,防止特殊字符
把产物写到 output/ 之外(T-01-03 路径安全)。

韧性体例移植自参考工程 calib.io.dataset(只读引用,绝不修改);
数据布局逻辑按本项目重写。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import cv2

from stereo_calib.io.config import DataDirs

# 安全的 image_id 模式:只许字母/数字/下划线/点/横线。
# image_id 会进入输出文件名,这里的限制防止特殊字符(路径分隔符等)
# 把产物写到 output/ 之外(T-01-03 路径安全)。
_SAFE_IMAGE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")

# 相机文件名模式:cam1.jpg / cam2.jpg(双目两路;cam3 及其余模式告警跳过)
_CAM_STEM = re.compile(r"^cam(\d+)$")
_SUPPORTED_CAMS = (1, 2)


@dataclass(frozen=True)
class ImageRecord:
    """一张双目图像的档案:它是谁、在哪、属于哪个划分的哪路相机哪个组。"""

    image_id: str        # 叶子目录名 + "_cam{n}"(stem 全库重复,绝不能用)
    path: Path           # 只读路径(相对当前工作目录)
    split: str           # "train" 或 "test"
    cam: int             # 1 或 2
    group_id: str        # 叶子目录名(公共视场配对键)


@dataclass(frozen=True)
class ScanResult:
    """一次目录扫描的完整结果:记录 + 全部警告(警告要写进 meta 留痕)。"""

    records: tuple[ImageRecord, ...]
    warnings: tuple[str, ...]


def _scan_split(split_dir: Path, split: str, warnings: list[str]) -> list[ImageRecord]:
    """扫描一个数据划分(train 或 test)下的全部双目图像。

    划分目录整个不存在:记中文警告并按 0 张处理(比如只有训练集、
    暂无独立测试集的数据是合法输入),两个划分都为空才在上一层报错。
    """
    split_dir = Path(split_dir)
    if not split_dir.is_dir():
        warnings.append(
            f"数据目录 {split_dir} 不存在,data.{split} 按 0 张照片处理——"
            f"如非有意为之,请检查 config 里 data.{split} 的路径(相对当前工作目录)。"
        )
        return []

    # 两层通配扫描(<目录>/<组目录>/cam*.jpg),容忍 <ts>/<ts>_<us> 嵌套,
    # 不硬编码层数语义;sorted 保证字典序稳定(时间戳目录名字典序 = 拍摄时间序)
    # 分组键 = 组目录相对划分的「父/叶子」路径(CR-01):同划分不同父目录下的
    # 同名叶子目录是两个独立组,绝不静默合并——若按叶子名合并,cam1/cam2 会
    # 跨拍摄错配成毒对,直接污染 common_fov_pairs(Phase 3 双目标定输入)。
    # 行为保持:当前数据叶子名全局唯一,路径键与原叶子名键产生完全相同的分组;
    # 排序方面叶子名 <ts>_<us> 以父目录 <ts> 开头,键排序与原叶子名排序给出
    # 相同记录顺序。
    by_leaf: dict[str, dict[int, Path]] = {}
    for jpg in sorted(split_dir.glob("*/*/cam*.jpg")):
        m = _CAM_STEM.match(jpg.stem)
        if m is None or int(m.group(1)) not in _SUPPORTED_CAMS:
            warnings.append(
                f"文件名 {jpg.name!r} 不是受支持的相机命名"
                f"(双目两路,期望 cam1.jpg / cam2.jpg),已跳过:{jpg}"
            )
            continue
        key = jpg.parent.relative_to(split_dir).as_posix()
        by_leaf.setdefault(key, {})[int(m.group(1))] = jpg

    # 空目录(或只有不合模式文件)也要点名,不让"少了一半数据"静默发生
    for d in sorted(split_dir.glob("*/*")):
        if d.is_dir() and d.relative_to(split_dir).as_posix() not in by_leaf:
            warnings.append(
                f"组目录 {d} 里没有有效的 cam1.jpg/cam2.jpg(空目录或文件名不合模式),已跳过。"
            )

    # 单边缺文件:存在的那路照常入册(内参可用),只警告不丢
    for leaf, cams in by_leaf.items():
        missing = [c for c in _SUPPORTED_CAMS if c not in cams]
        if missing:
            had = "/".join(f"cam{c}" for c in sorted(cams))
            want = "/".join(f"cam{c}" for c in missing)
            warnings.append(
                f"组目录 {leaf} 只有 {had}、缺 {want}——该组只入单路,"
                f"内参标定仍可用,双目外参不可用。"
            )

    records: list[ImageRecord] = []
    for key in sorted(by_leaf):
        leaf = Path(key).name  # 叶子名 = group_id / image_id 前缀(配对键契约,绝不变)
        for cam in sorted(by_leaf[key]):
            image_id = f"{leaf}_cam{cam}"
            if not _SAFE_IMAGE_ID.match(image_id) or image_id in {".", ".."}:
                warnings.append(
                    f"图像编号 {image_id!r} 含不安全的字符"
                    f"(只允许字母、数字、下划线、点、横线),已跳过——防止它破坏产物路径。"
                )
                continue
            records.append(ImageRecord(
                image_id=image_id, path=by_leaf[key][cam],
                split=split, cam=cam, group_id=leaf,
            ))
    return records


def scan_dataset(data_cfg: DataDirs) -> ScanResult:
    """按配置扫描 data.train / data.test 两个划分,返回记录 + 警告清单。

    image_id 全库唯一(叶子目录重名会撞编号):重复记中文警告并跳过后出现者。
    同划分不同父目录下的同名叶子目录按「父/叶子」路径分为独立组,产生的
    同 image_id 同样在此去重(警告 + 跳过后出现者),绝不静默合并组。
    两个划分都扫不出任何记录:抛中文 ValueError(数据没就位不是可继续状态)。
    """
    warnings: list[str] = []
    seen: set[str] = set()
    records: list[ImageRecord] = []
    for split, split_dir in (("train", data_cfg.train), ("test", data_cfg.test)):
        for rec in _scan_split(Path(split_dir), split, warnings):
            if rec.image_id in seen:
                warnings.append(
                    f"图像编号 {rec.image_id!r} 在数据集中重复(叶子目录重名),"
                    f"已跳过后出现者:{rec.path}"
                )
                continue
            seen.add(rec.image_id)
            records.append(rec)
    if not records:
        raise ValueError(
            f"目录 {data_cfg.train} 与 {data_cfg.test} 下没有扫到任何有效的双目图像,"
            f"请检查 config 里 data.train / data.test 的路径"
            f"(期望结构:<目录>/<组目录>/cam1.jpg + cam2.jpg)。"
        )
    return ScanResult(records=tuple(records), warnings=tuple(warnings))


def imread_checked(path: Path):
    """带检查的读图:cv2.imread 读坏文件不报错而是返回 None,
    这里立刻抛带文件名的中文错误,把报错位置钉在病因上。"""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(
            f"照片读取失败:{path}。文件可能损坏、路径不对或没有读权限,"
            f"请人工检查这张图。"
        )
    return img
