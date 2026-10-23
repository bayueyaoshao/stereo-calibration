"""测试夹具:最小合法 JPEG 与合成棋盘投影图(带真值角点,检测对错有标准答案)。

移植自参考工程 tests/conftest.py(只读引用,绝不修改):
- make_tiny_jpg:造一张最小合法 JPEG(随机灰度噪声),用于扫描/读图韧性测试;
- make_chessboard_image:合成 11×8 棋盘灰度图,返回 (图像 uint8, 真值角点
  (88, 2) float64, 方格像素边长)。supersample 抗锯齿与主点对齐的注释一并
  照抄——它们保证"真值角点误差 < 0.5px"断言成立(硬边光栅化会带来约半个
  像素的系统性画格偏差,主点不对齐会带来 ≈0.375 像素的整幅偏移)。
"""

from pathlib import Path

import cv2
import numpy
import pytest


@pytest.fixture
def make_tiny_jpg(tmp_path: Path):
    """造图夹具:make_tiny_jpg("data/train/<ts>/<ts>_<us>/cam1.jpg", 宽, 高) → 路径。"""

    def _make(rel_path: str, width: int, height: int) -> Path:
        target = tmp_path / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        rng = numpy.random.default_rng(42)
        gray = rng.integers(0, 256, size=(height, width), dtype=numpy.uint8)
        if not cv2.imwrite(str(target), gray):
            raise RuntimeError(f"测试夹具写图失败:{target}")
        return target

    # 附带数据根路径,测试直接取用,不用再猜目录
    _make.data_root = tmp_path / "data"  # type: ignore[attr-defined]
    return _make


def _fill_quad(img: numpy.ndarray, pts: numpy.ndarray, value: int) -> None:
    """把一个四边形区域填成指定灰度值(纯 numpy 判断"点在四边形内")。

    做法:取四边形的外接矩形,对矩形内每个像素判断它是否落在四边形内——
    判断依据是"点在每条边的同一侧"(用叉积的符号,叉积就是两个向量张成
    平行四边形的有向面积,同侧则符号一致)。透视/畸变下四边形会轻微变形,
    这种逐点判断天然支持,不需要强行按矩形填。
    """
    height, width = img.shape[:2]
    x0 = max(int(numpy.floor(pts[:, 0].min())), 0)
    y0 = max(int(numpy.floor(pts[:, 1].min())), 0)
    x1 = min(int(numpy.ceil(pts[:, 0].max())), width - 1)
    y1 = min(int(numpy.ceil(pts[:, 1].max())), height - 1)
    if x1 < x0 or y1 < y0:
        return  # 整块落在画面外,跳过

    xs = numpy.arange(x0, x1 + 1)
    ys = numpy.arange(y0, y1 + 1)
    xx, yy = numpy.meshgrid(xs, ys)
    pixels = numpy.stack([xx.ravel(), yy.ravel()], axis=1).astype(numpy.float64)

    signs = []
    for k in range(4):
        a = pts[k]
        b = pts[(k + 1) % 4]
        cross = (b[0] - a[0]) * (pixels[:, 1] - a[1]) - (b[1] - a[1]) * (pixels[:, 0] - a[0])
        signs.append(cross)
    side = numpy.stack(signs, axis=1)
    inside = (side >= 0).all(axis=1) | (side <= 0).all(axis=1)
    block = img[y0:y1 + 1, x0:x1 + 1]
    block[inside.reshape(block.shape)] = value


@pytest.fixture
def make_chessboard_image():
    """合成棋盘投影图工厂:make_chessboard_image(...) → (图像, 真值角点, 方格像素边长)。

    参数(全部有默认值,测试只写关心的那一两项;默认即 11×8 规格,
    不带参数调用得到固定 11×8 / 方格约 20px 的棋盘):
        cols / rows:  内部角点的列数/行数(默认 11×8 = 88 个角点)
        square_px:    期望的方格像素边长(通过反推物理方格尺寸来近似实现)
        K:            3×3 相机内参矩阵;默认按 image_size 取中心为光心、焦距 800
        dist_coeffs:  畸变系数(默认无畸变;合成图重点是"已知真值",有小畸变也行)
        rvec / tvec:  棋盘相对相机的旋转/平移;默认棋盘正对相机、只带极轻微侧转
        image_size:   图像尺寸 (高, 宽),与 numpy 数组形状同序
        degrade:      退化变体:None / "blur"(高斯模糊)/ "noise"(加噪声)

    返回 (图像 uint8 (高,宽), 真值角点 (cols*rows, 2) float64, 方格像素边长约等于值)。
    """

    def _make(cols: int = 11, rows: int = 8, square_px: float = 20.0,
              K: numpy.ndarray | None = None,
              dist_coeffs: numpy.ndarray | None = None,
              rvec: numpy.ndarray | None = None,
              tvec: numpy.ndarray | None = None,
              image_size: tuple[int, int] = (480, 640),
              degrade: str | None = None):
        height, width = image_size
        if K is None:
            K = numpy.array([[800.0, 0.0, width / 2.0],
                             [0.0, 800.0, height / 2.0],
                             [0.0, 0.0, 1.0]])
        if dist_coeffs is None:
            dist_coeffs = numpy.zeros(5)
        if rvec is None:
            rvec = numpy.array([0.0, 0.05, 0.0])  # 极轻微侧转:真实感,又不破坏方格均匀
        if tvec is None:
            tvec = numpy.array([0.0, 0.0, 800.0])  # 棋盘放在相机正前方 800(单位与物理方格一致)

        # 反推物理方格边长,使投影后方格像素边长约等于 square_px:
        # 像素尺寸 ≈ 焦距 × 物理尺寸 / 距离,倒过来即得物理尺寸。
        focal_x = float(K[0, 0])
        depth = float(tvec[2])
        square_world = square_px * depth / focal_x

        # 内部角点物点(行主序:行在外、列在内),棋盘中心对齐光轴
        xs = (numpy.arange(cols) - (cols - 1) / 2.0) * square_world
        ys = (numpy.arange(rows) - (rows - 1) / 2.0) * square_world
        objp = numpy.zeros((rows * cols, 3), dtype=numpy.float64)
        objp[:, 0] = numpy.tile(xs, rows)
        objp[:, 1] = numpy.repeat(ys, cols)
        corners, _ = cv2.projectPoints(objp, rvec, tvec, K, dist_coeffs)
        corners = corners.reshape(-1, 2).astype(numpy.float64)

        # 画图:把网格向外扩一圈(即多出一圈边框方格),从 -1 到 cols / rows 闭区间,
        # 相邻网格点围成的每一格就是一个方格;黑白交替,黑格填 0、白格与背景填 255。
        # 先在 4 倍边长的大画布上画硬边棋盘,再用"区域平均"缩小回目标尺寸——
        # 等效于真实镜头的光学模糊(抗锯齿),否则硬边的光栅化台阶会让角点
        # 带上约半个像素的系统性画格偏差,污染"对真值误差 < 0.5px"这类断言。
        supersample = 4
        # 大画布的内参:焦距同比放大;主点按 OpenCV 缩小的像素中心映射平移——
        # cv2.resize 缩小 SS 倍时,小图像素 i 取大图第 4i..4i+3 列的均值,
        # 其中心在大图坐标 SS*i + (SS-1)/2 处,所以必须用
        # c_big = SS*(c_small + 0.5) − 0.5 对齐,否则整幅图带 (SS−1)/2/SS ≈
        # 0.375 像素的系统性偏移,污染一切"对真值误差"断言。
        k_big = K.astype(numpy.float64).copy()
        k_big[0, 0] *= supersample
        k_big[1, 1] *= supersample
        k_big[0, 2] = supersample * (k_big[0, 2] + 0.5) - 0.5
        k_big[1, 2] = supersample * (k_big[1, 2] + 0.5) - 0.5
        gx = (numpy.arange(-1, cols + 1) - (cols - 1) / 2.0) * square_world
        gy = (numpy.arange(-1, rows + 1) - (rows - 1) / 2.0) * square_world
        grid_obj = numpy.zeros(((rows + 2) * (cols + 2), 3), dtype=numpy.float64)
        grid_obj[:, 0] = numpy.tile(gx, rows + 2)
        grid_obj[:, 1] = numpy.repeat(gy, cols + 2)
        grid, _ = cv2.projectPoints(grid_obj, rvec, tvec, k_big, dist_coeffs)
        grid = grid.reshape(rows + 2, cols + 2, 2).astype(numpy.float64)

        big = numpy.full((height * supersample, width * supersample), 255, dtype=numpy.uint8)
        for j in range(rows + 1):
            for i in range(cols + 1):
                quad = numpy.stack([grid[j, i], grid[j, i + 1],
                                    grid[j + 1, i + 1], grid[j + 1, i]])
                if (i + j) % 2 == 0:
                    _fill_quad(big, quad, 0)
        img = cv2.resize(big, (width, height), interpolation=cv2.INTER_AREA)

        if degrade == "blur":
            kernel = max(3, (round(square_px / 4) // 2) * 2 + 1)  # 奇数边长
            img = cv2.GaussianBlur(img, (kernel, kernel), 0)
        elif degrade == "noise":
            rng = numpy.random.default_rng(42)
            noisy = img.astype(numpy.float64) + rng.normal(0.0, 12.0, size=img.shape)
            img = numpy.clip(noisy, 0, 255).astype(numpy.uint8)
        elif degrade is not None:
            raise ValueError(f"未知的退化变体:{degrade!r}(支持 blur / noise)")

        return img, corners, float(square_px)

    return _make
