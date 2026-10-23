"""视图装载层:Phase 1 角点 npz → 进标定/评分的唯一输入形态(Views)。

输入契约(Phase 1 八数组带掩码 npz:image_ids / paths / uv / objp / source /
detected / sq_px / win_size):读取一律经 preprocess.artifacts.load_corners——
其 allow_pickle=False 防线已在 Phase 1 测试锁定(npz 的 pickle 模式可执行
任意代码,关死),本模块绝不直接用 numpy 的加载接口(T-02-01)。

三重防线(参考工程 solve/observation.py 体例,RESEARCH Pitfall 9):
  1. exclude_images 命中 image_id 先跳过(手动逻辑剔除通道,原数据不动);
  2. detected=False 行跳过(整行 NaN 不进求解);
  3. 幸存行 np.isfinite 全真断言——检出但含 NaN/Inf 直接中文报错,
     绝不带毒进求解器。
另有 objp 与配置重建网格的 allclose 交叉核对(Pitfall 11:board 规格在
Phase 1 运行后被改过 → 角点与物点错位,RMS 虚高难排查),不一致中文报错
提示重跑 preprocess。

行序锁(Pitfall 10):image_ids 元组在装载时定死,幸存视图顺序与之一一
对应;后续 rvecs/tvecs/逐帧 RMS/报告全部按该顺序,绝不重排——行序一旦
漂移,逐帧表错行且不报错、只是悄悄算错。

image_size 派生(Pitfall 2):npz 与 meta.json 都不含分辨率,幸存首帧经
io.dataset.imread_checked 读图后 height, width = img.shape[:2] 返回
(宽, 高)——严禁把 (高, 宽) 顺序传下去;首/中/末 3 帧抽查分辨率一致
(A2 防混分辨率,不足 3 帧时对现有帧抽查)。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stereo_calib.io.config import BoardSpec
from stereo_calib.io.dataset import imread_checked
from stereo_calib.preprocess.artifacts import build_object_points, load_corners

# Phase 1 八数组契约里装载层消费的键(sq_px/win_size 是预处理内部口径,不进 Views)
_REQUIRED_KEYS = ("image_ids", "paths", "uv", "objp", "source", "detected")


@dataclass(frozen=True)
class Views:
    """一次装载的幸存视图集合(不可变):一切下游结果按 image_ids 顺序对齐。

    uv 形状 (V, P, 2) float64(V = 幸存视图数,P = rows*cols 个内部角点);
    objp 形状 (P, 3) float64(与 uv 角点同序的板系物点);
    image_size 为 (宽, 高)——OpenCV 口径,与 numpy shape 的 (高, 宽) 相反。
    """

    image_ids: tuple[str, ...]
    paths: tuple[str, ...]
    source: tuple[str, ...]
    uv: np.ndarray
    objp: np.ndarray
    image_size: tuple[int, int]


def _image_size_from_paths(paths: tuple[str, ...]) -> tuple[int, int]:
    """读幸存首帧派生 (宽, 高);首/中/末 3 帧抽查分辨率一致(A2 防混分辨率)。

    统一 height, width = img.shape[:2] 后返回 (width, height)——imread 的
    numpy 形状是 (高, 宽),标定与 OpenCV 调用要的是 (宽, 高),顺序在此
    一次性翻转并锁定,下游绝不再碰 img.shape。
    """
    size: tuple[int, int] | None = None
    spot = sorted({0, len(paths) // 2, len(paths) - 1})
    for i in spot:
        img = imread_checked(Path(paths[i]))
        height, width = img.shape[:2]
        if size is None:
            size = (int(width), int(height))
        elif (int(width), int(height)) != size:
            raise ValueError(
                f"图像分辨率不一致:第 {i} 帧 {paths[i]} 是 (宽{width}, 高{height}),"
                f"而幸存首帧是 {size}。混分辨率会破坏单 image_size 标定语义,"
                f"请检查数据采集,或把不一致的帧加进 intrinsic.exclude_images 后重跑。"
            )
    assert size is not None  # paths 非空由调用方幸存视图数下限保证
    return size


def load_views(board: BoardSpec, npz_path: Path,
               exclude_images: tuple[str, ...] = (),
               min_views: int = 10) -> Views:
    """把一份 Phase 1 角点 npz 装载、过滤成幸存视图集(行序契约在此定死)。

    参数:
        board:          棋盘规格(物点网格由此重建,与 npz 内 objp 交叉核对);
        npz_path:       Phase 1 产物路径(如 output/train/cam1_train.npz);
        exclude_images: 手动逻辑剔除的图像编号清单(原数据不动,与自动机制叠加);
        min_views:      幸存视图数下限,低于此数中文报错不硬算(惯例 ≥10)。
    """
    data = load_corners(npz_path)  # allow_pickle=False 防线复用,绝不绕过
    missing = [key for key in _REQUIRED_KEYS if key not in data]
    if missing:
        raise ValueError(
            f"角点 npz {Path(npz_path).name} 缺少数组键 {missing}——它不是 Phase 1"
            f"八数组契约的预处理产物,请先运行 stereo-calib preprocess 生成角点。"
        )

    ids = [str(x) for x in np.asarray(data["image_ids"]).ravel()]
    paths = [str(x) for x in np.asarray(data["paths"]).ravel()]
    uv = np.asarray(data["uv"], dtype=np.float64)
    objp = np.asarray(data["objp"], dtype=np.float64)
    source = [str(x) for x in np.asarray(data["source"]).ravel()]
    detected = np.asarray(data["detected"], dtype=bool)

    n = len(ids)
    if not (len(paths) == len(source) == len(detected) == n) or uv.shape[0] != n:
        raise ValueError(
            f"角点 npz {Path(npz_path).name} 各数组行数不一致:image_ids={n},"
            f"paths={len(paths)},source={len(source)},detected={len(detected)},"
            f"uv={uv.shape[0]}。文件可疑,请重跑 preprocess 重新生成。"
        )

    n_points = board.rows * board.cols
    if uv.ndim != 3 or uv.shape[1:] != (n_points, 2):
        raise ValueError(
            f"角点数组形状不合:期望 (行数,{n_points},2)(棋盘 {board.cols}×{board.rows}"
            f" 个内部角点),实际是 {uv.shape}。多半是角点文件与 config.yaml 的棋盘规格"
            f"不是同一套:请核对配置后重跑 preprocess。"
        )

    expected_objp = build_object_points(board)
    if objp.shape != expected_objp.shape or not np.allclose(objp, expected_objp):
        raise ValueError(
            f"角点 npz {Path(npz_path).name} 内的 objp 与当前配置 board 规格重建的"
            f"物点网格不一致(配置的棋盘规格在 Phase 1 运行后被改过?)。角点与物点"
            f"错位会让标定结果整批作废:请用当前 config.yaml 重跑 preprocess 后再标定。"
        )

    exclude = set(exclude_images)
    kept_ids: list[str] = []
    kept_paths: list[str] = []
    kept_source: list[str] = []
    views: list[np.ndarray] = []
    for i, image_id in enumerate(ids):
        if image_id in exclude:
            continue  # 手动剔除通道:逻辑剔除,原数据不动
        if not detected[i]:
            continue  # 未检出的帧整行 NaN,进不了标定
        if not np.isfinite(uv[i]).all():
            raise ValueError(
                f"照片 {image_id}(第 {i} 行)的 uv 字段标记为已检出,但角点里混入了"
                f"非数值(NaN/无穷),角点文件可疑:{Path(npz_path)}。"
                f"请重跑 preprocess 重新生成角点,或人工检查这张图。"
            )
        views.append(uv[i])
        kept_ids.append(image_id)
        kept_paths.append(paths[i])
        kept_source.append(source[i])

    if len(views) < min_views:
        raise ValueError(
            f"{Path(npz_path).name} 过滤后只剩 {len(views)} 个可用视图"
            f"(检出且未被手动剔除),低于下限 {min_views}——两三个视角定不了内参,"
            f"硬算必错。请检查 exclude_images 清单、角点检出率与 min_views 配置。"
        )

    uvs = np.asarray(views, dtype=np.float64)
    if uvs.shape != (len(views), n_points, 2):  # 幸存形状断言(防线自证)
        raise ValueError(
            f"幸存角点数组形状不合:期望 ({len(views)},{n_points},2),"
            f"实际是 {uvs.shape}。这是装载层内部防线被击穿的信号,请把本错误"
            f"连同运行日志反馈维护者。"
        )

    image_size = _image_size_from_paths(tuple(kept_paths))
    return Views(image_ids=tuple(kept_ids), paths=tuple(kept_paths),
                 source=tuple(kept_source), uv=uvs, objp=objp, image_size=image_size)
