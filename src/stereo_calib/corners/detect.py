"""角点检测(移植自参考工程 calib.corners.detect,只读引用,绝不修改)。

本计划(01-01)只落地 SB 主路线 detect_sb;classic 兜底路线(粗定位 →
二次亚像素精化 → 缺陷闸门 → 窗口阶梯恢复)由 01-02 追加移植。

三条硬不变量(随代码一起移植,测试锁定):
1. SB 输出已是亚像素精度,**严禁二次亚像素精化**——SB 内部的角点模型与
   经典路线的精化模型不是一回事,二次处理会互相污染、精度反降;
   因此本文件不得出现二次精化的调用形式(01-02 落地 classic 时才会有
   全文件唯一调用点,源码防线测试锁定出现次数;注释里也不写出该
   调用串原文,避免防线自伤)。
2. 亚像素窗口逐帧从实测方格像素边长推导并断言 2w+1 < sq_px(window.py)。
3. 检测输出必过规范化(canonicalize)——SB/classic 角点序不同已在本数据
   实测(原始差 355.9px 中位,规范化后 0.63px)。

检测失败返回 None(本帧未检出),绝不返回半成品。
"""

from __future__ import annotations

import cv2
import numpy as np

from stereo_calib.corners.canonical import canonicalize
from stereo_calib.corners.window import derive_window_frame, estimate_sq_px
from stereo_calib.io.config import CornersConfig


def detect_sb(gray: np.ndarray, cols: int, rows: int, cfg: CornersConfig) -> dict | None:
    """新款(SB)检测器:一次检测直接输出亚像素角点 → 规范化。

    SB 输出已是亚像素精度,**绝不再做二次亚像素精化**——它内部的角点模型
    与经典路线的精化模型不是一回事,二次处理会互相污染、精度反降。

    返回 dict(uv (rows*cols, 2) float64, sq_px, win_size, detector="sb",
    recovery=None);未检出返回 None。
    win_size 字段只为产物结构对齐而按同一公式推导入档(SB 路线并不使用
    窗口),下游审查两颗检测器的方格尺度时口径一致。
    """
    found, raw = cv2.findChessboardCornersSB(gray, (cols, rows), cfg.sb_flags)
    if not found:
        return None
    corners = raw.reshape(-1, 2).astype(np.float64)

    sq_px = estimate_sq_px(corners, cols, rows)
    win_size = derive_window_frame(sq_px, cfg.subpix.min_win)  # 仅记录,不参与任何精化

    return {
        "uv": canonicalize(corners, cols, rows),  # 不变量 3:输出必过规范化
        "sq_px": sq_px,
        "win_size": win_size,
        "detector": "sb",
        "recovery": None,  # SB 一次到位、不经二次精化,结构上与 classic 对齐
    }
