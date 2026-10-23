"""角点检测双检测器(移植自参考工程 calib.corners.detect,只读引用,绝不修改)。

经典路线(detect_classic,SB 未检出时的兜底):cv2.findChessboardCorners
(自适应阈值 + 归一化)先粗定位(整像素),再做二次亚像素精化,精化窗口从
本帧实测的方格像素边长推导(window.py,绝不写死像素)。精化之后过一道
**缺陷闸门**:数网格里的重复点(两角点叠在同处)与锯齿(间距突变);
带病帧从粗定位网格出发,用"安全窗口上限"往下逐级试更小的窗口重提,
第一个零缺陷的胜出,恢复过程(缺陷数/试过的窗口/最终窗口)全部记进
返回值留痕。

新款路线(detect_sb,主路线):cv2.findChessboardCornersSB 一次到位,
**输出已是亚像素精度,严禁再做二次亚像素精化**——两条路线的精化用的是
不同的角点模型,叠在一起互相污染、精度反降。因此本文件里二次精化的
调用形式**恰好只允许出现 1 次**(只在经典分支的 _subpix_refine;注释里
也不写出该调用串原文,避免源码防线测试自伤)。

三条硬不变量(随代码一起移植,测试锁定):
1. SB 输出严禁二次亚像素精化(源码防线测试锁定调用形式出现次数 == 1);
2. 亚像素窗口逐帧从实测方格像素边长推导并断言 2w+1 < sq_px(window.py);
3. 两条路线的输出都必过规范化(canonicalize)——SB/classic 角点序不同已在
   本数据实测(原始差 355.9px 中位,规范化后 0.63px),不过这层,兜底帧
   的角点数据对下游就是废数据。

两颗检测器失败都返回 None(本帧该检测器未检出),绝不返回半成品。
"""

from __future__ import annotations

import cv2
import numpy as np

from stereo_calib.corners.canonical import canonicalize
from stereo_calib.corners.window import (
    derive_window_frame,
    derive_window_per_corner,
    estimate_sq_px,
)
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


# ---- classic 兜底路线(01-02 追加移植:粗定位 → 精化 → 缺陷闸门 → 窗口阶梯恢复)----

# config.yaml 里经典检测器的选项名 → cv2 常量(名字在配置层已白名单校验)
_CLASSIC_FLAG_VALUES = {
    "adaptive_thresh": cv2.CALIB_CB_ADAPTIVE_THRESH,
    "normalize_image": cv2.CALIB_CB_NORMALIZE_IMAGE,
    "fast_check": cv2.CALIB_CB_FAST_CHECK,
}


def _classic_flag_value(flag_names: tuple[str, ...] | list[str]) -> int:
    """把配置里的选项名组合成 cv2 的选项位掩码。"""
    value = 0
    for name in flag_names:
        value |= _CLASSIC_FLAG_VALUES[name]
    return value


def _subpix_refine(gray: np.ndarray, pts: np.ndarray, win: int,
                   max_iter: int, epsilon: float) -> np.ndarray:
    """二次亚像素精化(检测器无关的薄封装):返回 (N, 2) float64 角点。

    全文件**只有这里**真正调用二次精化——经典路线的初次精化与缺陷恢复
    重提都经它走;SB 分支绝不调用本函数(它的输出已是亚像素精度,叠二次
    精化会互相污染)。源码防线测试锁定该调用形式在全文件恰好出现 1 次。
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon)
    refined = cv2.cornerSubPix(gray, pts.reshape(-1, 1, 2).astype(np.float32),
                               (int(win), int(win)), (-1, -1), criteria)
    return refined.reshape(-1, 2).astype(np.float64)


def _spacing_pool(uv: np.ndarray, cols: int, rows: int) -> np.ndarray:
    """把角点排成网格,返回全部相邻间距(行内 + 列内)的一维样本池。"""
    g = np.asarray(uv, dtype=np.float64).reshape(rows, cols, 2)
    d_h = np.linalg.norm(g[:, 1:, :] - g[:, :-1, :], axis=2)
    d_v = np.linalg.norm(g[1:, :, :] - g[:-1, :, :], axis=2)
    return np.concatenate([d_h.ravel(), d_v.ravel()])


def grid_defects(uv: np.ndarray, cols: int, rows: int) -> dict:
    """数一个角点网格里的两类硬伤(缺陷闸门的判据,来自参考工程事故复盘):

    - 重复点:相邻角点间距 < 0.25 × 稳健中位方格边长。刚性棋盘上物理不可能,
      只可能是精化窗口吞掉邻角、两点被拉到同一处;
    - 锯齿:行/列间距的二阶差分(间距的变化速度)绝对值超过中位边长的 30%。
      平面棋盘即使强透视,间距也只该平滑渐变,突变说明有角点错位。

    两个阈值都相对中位数取比例,个别坏点动不了基准。返回
    {"n_dup", "n_zigzag", "total", "sq_ref_px"};干净网格 total == 0。
    """
    pool = _spacing_pool(uv, cols, rows)
    sq_ref = float(np.median(pool))
    n_dup = int((pool < 0.25 * sq_ref).sum())

    g = np.asarray(uv, dtype=np.float64).reshape(rows, cols, 2)
    d_h = np.linalg.norm(g[:, 1:, :] - g[:, :-1, :], axis=2)
    d_v = np.linalg.norm(g[1:, :, :] - g[:-1, :, :], axis=2)
    n_zigzag = 0
    if cols >= 3:
        n_zigzag += int((np.abs(np.diff(d_h, n=2, axis=1)) > 0.3 * sq_ref).sum())
    if rows >= 3:
        n_zigzag += int((np.abs(np.diff(d_v, n=2, axis=0)) > 0.3 * sq_ref).sum())
    return {"n_dup": n_dup, "n_zigzag": n_zigzag,
            "total": n_dup + n_zigzag, "sq_ref_px": sq_ref}


# 恢复重提用比常规精化(配置默认 30 次 / 0.01px)更严的收敛判据:
# 候选窗口之间要按"缺陷数"比出高下,迭代不收敛会让比较结果抖动。
# 数值来自参考工程事故帧的逐窗实验(win11 重提后整体 RMS 3.99→0.318px)。
_RECOVERY_MAX_ITER = 100
_RECOVERY_EPSILON = 1e-6


def _window_ladder(cap: int) -> list[int]:
    """恢复重提的窗口阶梯:从 cap 往下每级减 2,并保证下限 3 恒在阶梯里。

    偶数 cap 时 range(cap, 2, -2) 不含 3(例:cap=4 → [4]、cap=6 → [6, 4]),
    必须补上——「下限 3 恒在阶梯里」是 _recover_grid_defects 尾部 assert 的
    前提,漏 3 就是文档化不变量「往下每级减 2 试到 3」被击穿(WR-01)。
    cap 由 max(3, ...) 保证 >= 3,阶梯绝不为空;奇数 cap 输出与原实现
    逐字节一致(例:cap=11 → [11, 9, 7, 5, 3])。
    """
    ladder = list(range(cap, 2, -2))
    if ladder[-1] != 3:
        ladder.append(3)
    return ladder


def _recover_grid_defects(gray: np.ndarray, raw_corners: np.ndarray,
                          cols: int, rows: int) -> tuple[np.ndarray, int, dict, list[int]]:
    """窗口阶梯恢复:初次精化有缺陷时,从粗定位网格换更小的窗口逐级重提。

    为什么从粗定位网格出发:初次精化的产物已经带病(重复点/锯齿),
    在它上面再精化只会把病留下来。粗定位虽只有整像素精度,但点位次序
    可靠,是干净的重提起点(参考工程实验验证:双起点收敛到同一不动点)。

    窗口上限(阶梯顶格)取粗定位间距池的 2% 分位、按同一公式推导:
    win = max(3, floor(分位值/2) − 1)。用 2% 分位而不是中位数,是因为
    病根恰恰是"中位边长正常、局部小方格远小于中位"——分位数贴近最小的
    那一批方格,个别坏点带不动它。往下每级减 2 试到 3,
    第一个零缺陷的胜出;全都有缺陷时取缺陷最少的(同级取更大窗口)。

    返回 (角点 float64 (N,2), 采用窗口, 缺陷摘要, 试过的窗口阶梯)。
    """
    raw = np.asarray(raw_corners, dtype=np.float64)
    pool = _spacing_pool(raw, cols, rows)
    cap = max(3, int(np.percentile(pool, 2) // 2) - 1)
    ladder = _window_ladder(cap)  # 例:cap=11 → [11, 9, 7, 5, 3](下限 3 恒在阶梯里)
    best: tuple[np.ndarray, int, dict] | None = None
    for w in ladder:
        cand = _subpix_refine(gray, raw.copy(), w,
                              _RECOVERY_MAX_ITER, _RECOVERY_EPSILON)
        defects = grid_defects(cand, cols, rows)
        if defects["total"] == 0:
            return cand, w, defects, ladder
        if best is None or defects["total"] < best[2]["total"]:
            best = (cand, w, defects)
    assert best is not None, "窗口阶梯不可能为空(下限 3 恒在阶梯里)"
    return best[0], best[1], best[2], ladder


def detect_classic(gray: np.ndarray, cols: int, rows: int,
                   cfg: CornersConfig) -> dict | None:
    """经典检测器:粗定位 → 二次亚像素精化(窗口按实测方格边长)→ 规范化,
    精化后过缺陷闸门,必要时窗口阶梯恢复。

    返回 dict(uv (rows*cols, 2) float64, sq_px, win_size, detector="classic",
    recovery),recovery 为 None 或恢复留痕 dict;未检出返回 None。
    """
    found, raw = cv2.findChessboardCorners(
        gray, (cols, rows), _classic_flag_value(cfg.classic_flags))
    if not found:
        return None
    corners = raw.reshape(-1, 2).astype(np.float64)
    coarse = corners.copy()  # 粗定位网格留底:精化若带病,恢复必须从这里重新出发

    sq_px = estimate_sq_px(corners, cols, rows)

    # 两种窗口粒度统一成"逐组精化":帧级模式下所有角点共用一个窗口值,
    # 逐角点模式下各点用自己的窗口值;同窗口的角点分一组、一组调用一次
    # (精化函数一次只收一个窗口)。精化调用全文件只有 _subpix_refine 一处,
    # 且只挂在经典分支——SB 路线严禁经过这里(见 detect_sb 的说明)。
    if cfg.subpix.window_granularity == "corner":
        wins = derive_window_per_corner(corners, cols, rows, cfg.subpix.min_win)
    else:
        wins = np.full(corners.shape[0],
                       derive_window_frame(sq_px, cfg.subpix.min_win), dtype=int)
    win_size = int(wins.max())
    refined = corners.astype(np.float32)  # 精化函数只收 32 位浮点点坐标
    for w in np.unique(wins):
        refined[wins == w] = _subpix_refine(
            gray, refined[wins == w], int(w),
            cfg.subpix.max_iter, cfg.subpix.epsilon)
    corners = refined.reshape(-1, 2).astype(np.float64)

    # 缺陷闸门:正常帧零缺陷、原样通过(逐字节不动的帧不受任何影响);
    # 有重复点/锯齿的帧才进窗口阶梯恢复,恢复过程整段留痕。
    recovery = None
    defects = grid_defects(corners, cols, rows)
    if defects["total"] > 0:
        fixed, used_win, defects_after, ladder = _recover_grid_defects(
            gray, coarse, cols, rows)
        recovery = {
            "defects_before": defects["total"],
            "defects_after": defects_after["total"],
            "initial_win": win_size,
            "used_win": int(used_win),
            "ladder": ladder,
        }
        corners = fixed
        win_size = int(used_win)

    return {
        "uv": canonicalize(corners, cols, rows),  # 不变量 3:输出必过规范化
        "sq_px": sq_px,
        "win_size": win_size,
        "detector": "classic",
        "recovery": recovery,
    }
