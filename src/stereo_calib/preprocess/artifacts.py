"""预处理产物落盘与读取:角点 npz(带掩码稠密)+ meta.json 旁车 + 清单 CSV。

npz 布局(带掩码的稠密格式,Phase 2/3 的模块间契约 D-04,每相机每数据集一份):
- image_ids: (N,)   图像编号(叶子目录名 + "_cam{n}"),与扫描记录顺序一致;
                   未检出的图像照样占一行,绝不从数据里消失(D-01)
- paths:    (N,)    对应图像路径(与 image_ids 同序,下游定位原图用)
- uv:       (N, 88, 2) float64,角点像素坐标;未检出整行为 NaN
- objp:     (88, 3) float64,板系物点网格,与 canonicalize 后的 uv 同序
- source:   (N,) <U16 每帧角点来源:"sb" = SB 本体检出;"classic" = SB 未检出、
                   由经典检测器兜底填入(01-02 起);"" = 未检出
- detected: (N,)    bool,是否检出
- sq_px:    (N,)    每帧实测的方格像素边长,未检出为 NaN
- win_size: (N,)    每帧记录的亚像素半窗口口径(供审查窗口断言 2*win+1 < sq_px)

字符串一律定长 unicode 数组,npz 存取都不需要 pickle。
安全防线:读取一律 np.load(..., allow_pickle=False)——npz 的 pickle 模式
可以执行任意代码,关死它,角点文件只可能是纯数字/定长字符串数组(T-01-01)。

CSV 清单用标准库 csv、encoding="utf-8-sig"(带 BOM,Excel 双击直开中文
不乱码,D-06)。npz 契约移植自参考工程 calib.corners.artifacts(只读引用)。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from stereo_calib.io.config import BoardSpec

# 规范化规则的版本字符串:规则一变,下游读到的产物含义就变,必须留痕
CANONICAL_RULE_VERSION = "canonical-v1:行方向朝画面右、列方向朝画面下"


def build_object_points(board: BoardSpec) -> np.ndarray:
    """构造物点网格:棋盘内角点在"棋盘平面"上的坐标,(cols*rows, 3) float64。

    Z 一律为 0(棋盘是平的),X/Y 按方格边长排开,单位毫米。
    契约:objp[row*cols+col] = (col*square_size_mm, row*square_size_mm, 0),
    与 canonicalize 后的 uv 同序(行方向朝画面右、列方向朝画面下)。
    square_size 只进这里——改错它只会把"相机离棋盘多远"算错,
    内参(焦距/主点)与全部像素级逻辑都不受影响。
    """
    xs = np.arange(board.cols, dtype=np.float64) * board.square_size_mm
    ys = np.arange(board.rows, dtype=np.float64) * board.square_size_mm
    objp = np.zeros((board.rows * board.cols, 3), dtype=np.float64)
    objp[:, 0] = np.tile(xs, board.rows)
    objp[:, 1] = np.repeat(ys, board.cols)
    return objp


def save_corners_npz(out_dir: Path | str, name: str,
                     image_ids, paths, uv, objp, source, detected,
                     sq_px, win_size, meta: dict) -> Path:
    """写一份角点 npz(八数组)+ 同名 meta.json 旁车,返回 npz 路径。

    参数:
        out_dir:  输出目录(如 output/train)
        name:     文件基名(如 cam1_train)
        meta:     board 规格、配置快照(preprocess 段)、失败原因表等留痕内容,
                  由调用方组装;detected/source 计数与规范化规则版本在此一并写入
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{name}.npz"
    arrays = {
        "image_ids": np.asarray(image_ids),
        "paths": np.asarray(paths),
        "uv": np.asarray(uv, dtype=np.float64),
        "objp": np.asarray(objp, dtype=np.float64),
        "source": np.asarray(source, dtype="<U16"),
        "detected": np.asarray(detected, dtype=bool),
        "sq_px": np.asarray(sq_px, dtype=np.float64),
        "win_size": np.asarray(win_size, dtype=np.float64),
    }
    np.savez(npz_path, **arrays)

    source_arr = np.asarray(source)
    full_meta = {
        "file": npz_path.name,
        "canonical_rule": CANONICAL_RULE_VERSION,
        "n_images": len(image_ids),
        "n_detected": int(np.asarray(detected).sum()),
        "n_source_sb": int((source_arr == "sb").sum()),
        "n_source_classic": int((source_arr == "classic").sum()),
        **meta,
    }
    with open(out_dir / f"{name}.meta.json", "w", encoding="utf-8") as f:
        json.dump(full_meta, f, ensure_ascii=False, indent=2)
    return npz_path


def load_corners(path: Path | str) -> dict[str, np.ndarray]:
    """读取角点 npz,返回 {键: 数组} 字典。

    强制 allow_pickle=False:遇到含 pickle 对象的文件直接抛错,
    绝不执行其中的任意代码(反序列化安全防线,测试有锁定)。
    """
    with np.load(Path(path), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


_USABLE_FIELDS = ["image_id", "path", "cam", "route", "sq_px"]
_FAIL_FIELDS = ["image_id", "path", "cam", "reason"]
_COMMON_PAIR_FIELDS = ["group_id", "cam1_path", "cam2_path", "route_cam1", "route_cam2"]


def write_usable_csv(path: Path | str, rows: list[dict]) -> Path:
    """写可用图像清单 CSV:列 = image_id, path, cam, route, sq_px(D-06)。

    行数与对应 npz 的 detected 计数一致(D-01:检出即可用,无额外质量闸门)。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_USABLE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_fail_csv(path: Path | str, rows: list[dict]) -> Path:
    """写失败帧清单 CSV:列 = image_id, path, cam, reason(D-02 人工复核入口)。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FAIL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_common_pairs_csv(path: Path | str, rows: list[dict]) -> Path:
    """写公共视场对清单 CSV:列 = group_id, cam1_path, cam2_path, route_cam1,
    route_cam2(PREP-04,Phase 3 双目外参标定的输入清单)。

    一行 = 一个两路均检出的叶子目录组(group_id);行数恒 <= min(cam1 检出,
    cam2 检出)(Pitfall 8 警告信号)。成对规则(两路均检出才入对)由编排层
    pipeline 决定,这里只负责落盘——清单写空(仅表头)也合法(该 split 无
    公共视场对)。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_COMMON_PAIR_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path
