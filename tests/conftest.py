"""测试夹具:最小合法 JPEG 与合成棋盘投影图(带真值角点,检测对错有标准答案)。

移植自参考工程 tests/conftest.py(只读引用,绝不修改):
- make_tiny_jpg:造一张最小合法 JPEG(随机灰度噪声),用于扫描/读图韧性测试;
- make_chessboard_image:合成 11×8 棋盘灰度图,返回 (图像 uint8, 真值角点
  (88, 2) float64, 方格像素边长)。supersample 抗锯齿与主点对齐的注释一并
  照抄——它们保证"真值角点误差 < 0.5px"断言成立(硬边光栅化会带来约半个
  像素的系统性画格偏差,主点不对齐会带来 ≈0.375 像素的整幅偏移)。

本工程新增(03-01 起的双目合成夹具,供 03-02 求解恢复、03-03 e2e/评估链
复用;纯 numpy/cv2 投影合成,零渲染、零读图,秒级):
- make_stereo_synth:已知 R_rel/t_rel 真值几何的两视角角点合成器。两路
  K/D 取 tests/test_intrinsic_scoring.py 的 _PARAMS_TRUE 同源量级(focal
  ~1850、k1≈-0.4 的强畸变域)——让夹具几何与生产数据同性质,一切在弱
  畸变下才成立的捷径在测试里就先暴露;
- write_stereo_synth_files:按 Phase 1 八数组契约 + common_fov_pairs CSV
  契约落盘合成产物,供 stereo.observation.load_pairs 的装载契约测试。
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy
import pytest

from stereo_calib.io.config import BoardSpec
from stereo_calib.preprocess.artifacts import build_object_points, write_common_pairs_csv


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


# ===========================================================================
# 03-01 任务 2:双目合成夹具(真值几何两视角合成器 + npz/pairs CSV 工厂)
# ===========================================================================


@dataclass(frozen=True)
class SynthStereo:
    """一次双目合成观测的全量真值(不可变):03-02/03-03 测试的唯一数据源。

    uv1/uv2 形状 (V, 88, 2) float64,行序与 group_ids 一一对应;rvecs1/
    tvecs1 / rvecs2/tvecs2 是每对的 board→各相机真值位姿(OpenCV world→cam
    方向);R_rel_true/t_rel_true 语义为 X_cam2 = R_rel @ X_cam1 + t_rel
    (cam1→cam2 外参,baseline = ‖t_rel_true‖,单位毫米)。
    """

    objp: numpy.ndarray            # (88, 3) 板系物点
    uv1: numpy.ndarray             # (V, 88, 2) cam1 观测角点
    uv2: numpy.ndarray             # (V, 88, 2) cam2 观测角点
    rvecs1: numpy.ndarray          # (V, 3) board→cam1 旋转向量
    tvecs1: numpy.ndarray          # (V, 3) board→cam1 平移(mm)
    rvecs2: numpy.ndarray          # (V, 3) board→cam2 旋转向量
    tvecs2: numpy.ndarray          # (V, 3) board→cam2 平移(mm)
    R_rel_true: numpy.ndarray      # (3, 3) cam1→cam2 旋转真值
    t_rel_true: numpy.ndarray      # (3,)   cam1→cam2 平移真值(mm)
    K1: numpy.ndarray              # (3, 3) cam1 内参
    D1: numpy.ndarray              # (5,)   cam1 畸变(OpenCV dist 布局)
    K2: numpy.ndarray              # (3, 3) cam2 内参
    D2: numpy.ndarray              # (5,)   cam2 畸变
    params1: numpy.ndarray         # (9,)   models.INTRINSIC_NAMES 布局
    params2: numpy.ndarray         # (9,)   models.INTRINSIC_NAMES 布局
    baseline_mm_true: float        # ‖t_rel_true‖
    group_ids: tuple[str, ...]     # (V,) 对清单 group_id(<ts>_<us> 形态)


def make_stereo_synth(n_pairs: int = 8, seed: int = 7,
                      t_rel_true: tuple[float, float, float] = (120.0, 15.0, 8.0),
                      rot_rel_deg: float = 2.0, noise_px: float = 0.05,
                      flip_pair_indices: tuple[int, ...] = (),
                      board: tuple[int, int, float] = (11, 8, 25.0)) -> SynthStereo:
    """双目合成夹具工厂:固定种子下由已知 R_rel/t_rel 真值合成两路观测角点。

    参数(全部有默认值,测试只写关心的那一两项):
        n_pairs:           合成对数(每对一组随机 board→cam1 位姿);
        seed:              位姿与噪声的 numpy default_rng 种子(固定即复现);
        t_rel_true:        cam1→cam2 平移真值(mm,~120mm 落在 baseline 带内);
        rot_rel_deg:       相对旋转转角(度,刚架小夹角,绕固定任意轴);
        noise_px:          角点高斯检测噪声 std(像素);
        flip_pair_indices: 这些对的 uv2 取 arr[::-1](模拟 180° 角点序翻转,
                           供 03-02 消歧测试注入);
        board:             (cols, rows, square_size_mm) 棋盘规格,默认 11×8/25mm。

    零渲染、零读图:objp 直接由 build_object_points 构造,观测由
    cv2.projectPoints 从真值位姿合成(点位 float32、K/dist float64——
    参考工程实测的合法 dtype 组合)。
    """
    cols, rows, square_mm = board
    spec = BoardSpec(cols=cols, rows=rows, square_size_mm=square_mm)
    objp = build_object_points(spec)

    # 两路 K/D:_PARAMS_TRUE 同源量级(强畸变域,与生产数据同性质);cam2
    # 微调 fy/cy 避免与 cam1 完全相同。布局零畸变轴对齐 models.INTRINSIC_NAMES
    # = (fx, fy, cx, cy, k1, k2, p1, p2, k3);D 取后五项(OpenCV dist 布局)。
    params1 = numpy.array([1850.0, 1840.0, 1280.0, 720.0,
                           -0.40, 0.23, 1.1e-4, -3.8e-5, -0.08])
    params2 = numpy.array([1850.0, 1836.0, 1280.0, 724.0,
                           -0.40, 0.23, 1.1e-4, -3.8e-5, -0.08])

    def _kdt(params: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
        K = numpy.array([[params[0], 0.0, params[2]],
                         [0.0, params[1], params[3]],
                         [0.0, 0.0, 1.0]])
        return K, params[4:9].copy()

    K1, D1 = _kdt(params1)
    K2, D2 = _kdt(params2)

    # 相对位姿真值:绕任意固定轴 rot_rel_deg 的小转角(刚架两相机近同向,
    # 远离 π 的 Rodrigues 中位退化域,Pitfall 1)
    axis = numpy.array([0.3, -0.5, 0.8])
    axis = axis / numpy.linalg.norm(axis)
    R_rel_true, _ = cv2.Rodrigues(axis * numpy.deg2rad(rot_rel_deg))
    t_rel_true = numpy.asarray(t_rel_true, dtype=numpy.float64)
    baseline_mm_true = float(numpy.linalg.norm(t_rel_true))

    rng = numpy.random.default_rng(seed)
    objp_f32 = objp.astype(numpy.float32).reshape(-1, 1, 3)
    uv1_rows: list[numpy.ndarray] = []
    uv2_rows: list[numpy.ndarray] = []
    rvecs1: list[numpy.ndarray] = []
    tvecs1: list[numpy.ndarray] = []
    rvecs2: list[numpy.ndarray] = []
    tvecs2: list[numpy.ndarray] = []
    for _ in range(n_pairs):
        # board→cam1 随机位姿:适度倾角(0.3 rad std)+ 距离 500–900mm,
        # 与 test_intrinsic_pipeline 合成视图同一量级体例
        rvec1 = rng.normal(0.0, 0.3, 3)
        tvec1 = numpy.array([rng.normal(0.0, 40.0), rng.normal(0.0, 40.0),
                             rng.uniform(500.0, 900.0)])
        # cam2 真值位姿由相对位姿组合导出(方向推导,Pitfall 6 防搞反):
        #   X_cam1 = R1 @ X_board + t1;X_cam2 = R_rel @ X_cam1 + t_rel
        #   代入 → X_cam2 = (R_rel @ R1) @ X_board + (R_rel @ t1 + t_rel)
        #   即 R2 = R_rel @ R1,t2 = R_rel @ t1 + t_rel
        R1, _ = cv2.Rodrigues(rvec1)
        R2 = R_rel_true @ R1
        tvec2 = R_rel_true @ tvec1 + t_rel_true
        rvec2, _ = cv2.Rodrigues(R2)

        proj1, _ = cv2.projectPoints(objp_f32, rvec1, tvec1, K1, D1.reshape(-1, 1))
        proj2, _ = cv2.projectPoints(objp_f32, rvec2.astype(numpy.float64), tvec2,
                                     K2, D2.reshape(-1, 1))
        uv1 = proj1.reshape(-1, 2).astype(numpy.float64) + rng.normal(0.0, noise_px, (len(objp), 2))
        uv2 = proj2.reshape(-1, 2).astype(numpy.float64) + rng.normal(0.0, noise_px, (len(objp), 2))
        # 真值自检:板必须整体在两路相机前方(Z 全 > 0),否则位姿采样坏了
        pts2 = (R2 @ objp.T).T + tvec2
        assert (pts2[:, 2] > 0).all(), "合成夹具自检失败:cam2 系下板点 Z<=0"
        uv1_rows.append(uv1)
        uv2_rows.append(uv2)
        rvecs1.append(rvec1)
        tvecs1.append(tvec1)
        rvecs2.append(rvec2.ravel().astype(numpy.float64))
        tvecs2.append(tvec2)

    uv1_arr = numpy.asarray(uv1_rows)
    uv2_arr = numpy.asarray(uv2_rows)
    for idx in flip_pair_indices:
        uv2_arr[idx] = uv2_arr[idx][::-1]  # 180° 角点序翻转注入(消歧测试用)

    group_ids = tuple(f"20261024_0000{idx:02d}_{idx * 1000:06d}" for idx in range(n_pairs))
    return SynthStereo(
        objp=objp, uv1=uv1_arr, uv2=uv2_arr,
        rvecs1=numpy.asarray(rvecs1), tvecs1=numpy.asarray(tvecs1),
        rvecs2=numpy.asarray(rvecs2), tvecs2=numpy.asarray(tvecs2),
        R_rel_true=R_rel_true, t_rel_true=t_rel_true,
        K1=K1, D1=D1, K2=K2, D2=D2, params1=params1, params2=params2,
        baseline_mm_true=baseline_mm_true, group_ids=group_ids)


def write_stereo_synth_files(root: Path, synth: SynthStereo, split: str,
                             extra_undetected: bool = True) -> Path:
    """把合成观测按 Phase 1 契约落盘,返回 pairs CSV 路径(装载测试的入口)。

    写 root/output/{split}/ 下:
    - cam{1,2}_{split}.npz:八数组契约(image_ids = group_id + "_cam{n}";
      paths 造占位字符串即可——双目链路零读图,仅作路径交叉核对;source 全
      "sb"、detected 全 True、sq_px 全 18.0、win_size 全 5.0;字符串定长
      unicode,allow_pickle=False 可读);
    - common_fov_pairs_{split}.csv:utf-8-sig + 五列契约(列序照
      write_common_pairs_csv),cam{n}_path 与 npz paths 一致。
    extra_undetected=True 时在两路 npz 各追加一个 detected=False、uv 全 NaN
    的额外组(测装载过滤;它不进 pairs CSV)。额外组 group_id 用 990000 微
    秒段,与合成组(idx < n_pairs 的 0–99000 微秒段)不冲突。
    """
    root = Path(root)
    out_dir = root / "output" / split
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(synth.group_ids)
    extra_gid = "20261024_000099_990000"
    gids = list(synth.group_ids) + ([extra_gid] if extra_undetected else [])
    detected = [True] * n + ([False] if extra_undetected else [])

    def _paths_for(cam: int) -> list[str]:
        return [f"data/{split}/{gid}/{gid}/cam{cam}.jpg" for gid in gids]

    for cam, uv in ((1, synth.uv1), (2, synth.uv2)):
        rows = list(uv)
        if extra_undetected:
            rows.append(numpy.full((len(synth.objp), 2), numpy.nan))
        uv_arr = numpy.asarray(rows, dtype=numpy.float64)
        numpy.savez(
            out_dir / f"cam{cam}_{split}.npz",
            image_ids=numpy.asarray([f"{gid}_cam{cam}" for gid in gids]),
            paths=numpy.asarray(_paths_for(cam)),  # 定长 unicode 自动按最长路径取宽(禁手写宽度,会截断)
            uv=uv_arr,
            objp=numpy.asarray(synth.objp, dtype=numpy.float64),
            source=numpy.asarray(["sb" if d else "" for d in detected], dtype="<U16"),
            detected=numpy.asarray(detected, dtype=bool),
            sq_px=numpy.where(numpy.asarray(detected), 18.0, numpy.nan),
            win_size=numpy.where(numpy.asarray(detected), 5.0, numpy.nan),
        )

    csv_path = out_dir / f"common_fov_pairs_{split}.csv"
    # 列序与 utf-8-sig 由 Phase 1 写侧原函数保证(契约零漂移);额外未检出
    # 组不进 pairs CSV(它只测装载的 detected 过滤)
    write_common_pairs_csv(csv_path, [
        {"group_id": gid, "cam1_path": _paths_for(1)[i], "cam2_path": _paths_for(2)[i],
         "route_cam1": "sb", "route_cam2": "sb"}
        for i, gid in enumerate(synth.group_ids)
    ])
    return csv_path
