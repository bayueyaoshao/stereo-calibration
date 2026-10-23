"""方格像素边长估计与亚像素窗口推导(窗口按实测比例推导,绝不写死像素)。

头号陷阱(本项目第一优先防线的由来):亚像素精化(cornerSubPix,把角点位置
从整数像素细化到小数像素的迭代搜索)的窗口是"半窗口"——窗口值 w 对应的实际
搜索区是 (2w+1)×(2w+1) 个像素。一旦搜索区跨过相邻角点(即 2w+1 ≥ 方格像素
边长),搜索会把对边的黑白边界梯度混进来,角点被系统性地拉向一侧;更危险的
是,这个偏差在每个角点上方向相同,标定求解会把它整个吸收进内参和位姿,
重投影误差完全看不出来——错误悄无声息。

所以本模块是全流水线的断言防线:
1. 从每帧实测的角点间距估计方格像素边长(中位数稳健,保守取小);
2. 窗口 = max(下限, floor(方格边长/2) − 1),数学上保证 2w+1 < 方格边长;
3. 推导函数内部逐帧(或逐角点)断言该不等式,违反就当场报错,
   绝不让带病窗口流进检测与标定。

cols/rows(内部角点列数/行数)一律由调用方从 config.yaml 的 board 段传入,
本模块不设任何模块级魔法数字。

原样移植自参考工程 calib.corners.window(只读引用,绝不修改)。
"""

from __future__ import annotations

import numpy as np


def estimate_sq_px(corners: np.ndarray, cols: int, rows: int) -> float:
    """从检测到的角点估计方格的像素边长(每帧一个值)。

    corners: (rows*cols, 2) 行主序角点(先遍历一行内的列,再换下一行),
             顺序与检测器的原始输出经行主序 reshape 后一致。
    做法:把角点排成 (rows, cols, 2) 网格后,
    - 行内相邻角点的间距 → 行方向的方格边长样本;
    - 列内相邻角点的间距 → 列方向的方格边长样本;
    两个方向各取中位数(中位数对个别检测失准的角点不敏感),
    再**保守取小**——透视下远离相机的那排方格更小,取小值保证后续窗口不超限。
    """
    g = np.asarray(corners, dtype=np.float64).reshape(rows, cols, 2)
    d_h = np.linalg.norm(g[:, 1:, :] - g[:, :-1, :], axis=2)
    d_v = np.linalg.norm(g[1:, :, :] - g[:-1, :, :], axis=2)
    return float(min(np.median(d_h), np.median(d_v)))


def derive_window_frame(sq_px: float, min_win: int = 3) -> int:
    """按帧推导亚像素半窗口:win = max(min_win, floor(sq_px/2) − 1)。

    推导出来后立即断言 2 * w + 1 < sq_px(搜索区严格小于方格边长)。
    数学上 floor(sq/2)−1 这一支恒满足:偶数 sq → 2w+1 = sq−1;奇数 sq → sq−2。
    唯一可能违反的是 min_win 下限托底生效而方格又极小的情形(如方格 5px、
    下限 3 → 搜索区 7 ≥ 5)——这正是要当场报错的情况:数据里方格小到
    无法安全精化,继续跑只会产出看不见的系统性偏差。
    断言失败的信息用中文说明头号陷阱,便于排查时理解"为什么这么严"。
    """
    w = max(int(min_win), int(sq_px // 2) - 1)
    assert 2 * w + 1 < sq_px, (
        f"亚像素窗口断言失败:搜索区 {2 * w + 1} 像素未严格小于方格像素边长 {sq_px:.2f}。"
        f"窗口一旦跨过相邻角点,定位会被系统性拉偏,且重投影误差完全看不出来"
        f"(本项目头号陷阱)。请检查角点检测质量,或确认方格像素边长是否过小、"
        f"preprocess.corners.subpix.min_win 下限是否需要人工复核。"
    )
    return w


def _corner_local_spacing(corners: np.ndarray, cols: int, rows: int) -> np.ndarray:
    """逐角点算局部方格边长:取该角点上下左右现有邻居的间距中位数。

    内部角点有 4 个邻居,边上 3 个,四角 2 个——有几个用几个。
    返回 (rows*cols,) 数组。强畸变画面里边缘处方格明显更小,
    这个局部值比帧级中位数更贴近每个角点自己的真实尺度。
    """
    g = np.asarray(corners, dtype=np.float64).reshape(rows, cols, 2)

    # 收集每个角点的邻居间距:有几个邻居用几个
    samples: list[list[float]] = [[[] for _ in range(cols)] for _ in range(rows)]
    dist_h = np.linalg.norm(g[:, 1:, :] - g[:, :-1, :], axis=2)  # 行内相邻 (rows, cols-1)
    for r in range(rows):
        for c in range(cols - 1):
            samples[r][c].append(dist_h[r, c])
            samples[r][c + 1].append(dist_h[r, c])
    dist_v = np.linalg.norm(g[1:, :, :] - g[:-1, :, :], axis=2)  # 列内相邻 (rows-1, cols)
    for r in range(rows - 1):
        for c in range(cols):
            samples[r][c].append(dist_v[r, c])
            samples[r + 1][c].append(dist_v[r, c])

    local = np.zeros((rows, cols), dtype=np.float64)
    for r in range(rows):
        for c in range(cols):
            local[r, c] = float(np.median(samples[r][c]))
    return local.reshape(-1)


def derive_window_per_corner(corners: np.ndarray, cols: int, rows: int,
                             min_win: int = 3) -> np.ndarray:
    """逐角点推导半窗口(强畸变画面下的备选粒度,由配置选择启用)。

    每个角点用它自己的局部方格边长(4 邻域间距中位数,边界角点用现有邻域)
    走与帧级完全相同的公式:max(min_win, floor(局部边长/2) − 1),
    并逐点断言 2 * w + 1 < 局部方格边长(同一不等式,逐点成立)。

    返回 (rows*cols,) 的整数数组。帧级窗口大概率够用(棋盘集中画面中部,
    强畸变边缘通常没有角点);什么时候切换到本模式由真实数据说了算——
    若帧级模式下"窗口违反局部方格边长"的角点占比明显,配置
    preprocess.corners.subpix.window_granularity 改成 corner 即可,无需改代码。

    01-02 的 classic 逐角点粒度会用到,一并移植保持文件完整。
    """
    local = _corner_local_spacing(corners, cols, rows)
    wins = np.maximum(int(min_win), (local // 2).astype(int) - 1)
    for idx, (w, sq_px_local) in enumerate(zip(wins, local)):
        assert 2 * w + 1 < sq_px_local, (
            f"逐角点窗口断言失败:第 {idx} 号角点搜索区 {2 * w + 1} 像素未严格小于"
            f"局部方格边长 {sq_px_local:.2f}。窗口跨过相邻角点会产生重投影误差里"
            f"看不见的系统性定位偏差(头号陷阱);该角点附近的方格可能小到无法安全精化,"
            f"请人工检查这张图的检测质量。"
        )
    return wins.astype(int)
