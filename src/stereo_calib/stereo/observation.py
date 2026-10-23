"""对装载层:pairs CSV + 两路角点 npz → 双目求解/评估的唯一输入形态(Pairs)。

输入契约(Phase 1 产物,只读):
- common_fov_pairs_{split}.csv:utf-8-sig(带 BOM,DictReader 否则首列名带
  BOM 前缀),列 = group_id / cam1_path / cam2_path / route_cam1 / route_cam2;
- cam{n}_{split}.npz:Phase 1 八数组(image_ids / paths / uv / objp /
  source / detected / sq_px / win_size),读取一律经
  preprocess.artifacts.load_corners——其 allow_pickle=False 防线已在
  Phase 1 测试锁定,本模块绝不直接调 numpy 的加载接口(T-03-01);
- intrinsics.json(Phase 2):15 字段按名读,见 load_intrinsics。

连接键(Pattern 1 实测规则):npz 的 image_id = group_id + "_cam{n}"
(实测形态 20261023_154025_519359_cam1)。逐 CSV 行按连接键在两路 npz 的
image_ids 里定位行,不命中即中文报错(Pitfall 10:对不齐 = 角点错位配对,
外参静默错)。

三重防线(intrinsic/observation 体例,T-03-03):
  1. exclude_pairs 命中 group_id 先跳过(手动逻辑剔除通道,与自动机制
     叠加,原数据不动);
  2. 该对两路 detected 必须全 True(清单说成对,npz 必须检出);
  3. 幸存行 np.isfinite 全真断言——检出但含 NaN/Inf 直接中文报错,
     绝不带毒进求解器。
另有 objp 三方 allclose 交叉核对(两路各自与配置重建网格、两路互相;
Pitfall 11:board 规格漂移 → 角点与物点错位)与路径交叉核对(csv 的
cam{n}_path 与 npz paths 对应行一致,连接键错位防线),不一致中文报错
提示重跑 preprocess。

行序锁(Pitfall 10):image_ids 元组在装载时定死,幸存对顺序与 pairs CSV
行序一一对应;后续 rvecs/tvecs/逐帧 RMS/报告/剔除清单全部按该顺序,绝不
重排——行序一旦漂移,逐帧表错行且不报错、只是悄悄算错。

双目链路零读图:全程消费角点数组,绝不打开任何图像文件(比 Phase 2
更强的 data/ 只读保证;paths 仅作路径交叉核对,不读图)。
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stereo_calib.intrinsic.models import INTRINSIC_NAMES
from stereo_calib.io.config import BoardSpec
from stereo_calib.preprocess.artifacts import build_object_points, load_corners

# Phase 1 八数组契约里装载层消费的键(sq_px/win_size 是预处理内部口径,不进 Pairs)
_REQUIRED_KEYS = ("image_ids", "paths", "uv", "objp", "source", "detected")

# Phase 2 intrinsics.json 15 字段里装载层消费的子集(其余是 Phase 2 留痕)
_REQUIRED_JSON_FIELDS = ("cam", "image_size", "params", "K", "D")


@dataclass(frozen=True)
class Pairs:
    """一次装载的幸存双目对集合(不可变):一切下游结果按 image_ids 顺序对齐。

    uv1/uv2 形状 (V, P, 2) float64(V = 幸存对数,P = rows*cols 个内部角点,
    两路角点按连接键逐对对齐);objp 形状 (P, 3) float64(与两路 uv 同序的
    板系物点);image_ids 是 group_id 串(非 image_id);routes 是每对的
    (route_cam1, route_cam2)。
    """

    image_ids: tuple[str, ...]
    source1: tuple[str, ...]
    source2: tuple[str, ...]
    routes: tuple[tuple[str, str], ...]
    uv1: np.ndarray
    uv2: np.ndarray
    objp: np.ndarray


def _load_corners_checked(npz_path: Path, cam: int) -> dict[str, np.ndarray]:
    """读一路角点 npz 并做八数组键齐全性检查(缺键 = 不是 Phase 1 产物)。"""
    data = load_corners(npz_path)  # allow_pickle=False 防线复用,绝不绕过
    missing = [key for key in _REQUIRED_KEYS if key not in data]
    if missing:
        raise ValueError(
            f"角点 npz {Path(npz_path).name}(cam{cam})不合法:缺少数组键 {missing}——"
            f"它不是 Phase 1 八数组契约的预处理产物,请先运行 stereo-calib preprocess。"
        )
    return data


def _check_objp(objp1: np.ndarray, objp2: np.ndarray, expected: np.ndarray,
                npz1_path: Path, npz2_path: Path) -> np.ndarray:
    """objp 三方 allclose 交叉核对:两路各自 vs 配置重建网格、两路互相。"""
    for path, objp in ((npz1_path, objp1), (npz2_path, objp2)):
        if objp.shape != expected.shape or not np.allclose(objp, expected):
            raise ValueError(
                f"角点 npz {Path(path).name} 的 objp 不合法:与当前配置 board 规格重建的"
                f"物点网格不一致(配置的棋盘规格在 Phase 1 运行后被改过?)。角点与物点"
                f"错位会让外参整批作废:请用当前 config.yaml 重跑 preprocess 后再标定。"
            )
    if not np.allclose(objp1, objp2):
        raise ValueError(
            f"两路角点产物的 objp 不合法:cam1({Path(npz1_path).name})与 "
            f"cam2({Path(npz2_path).name})的物点网格互相不一致——两路不是同一次"
            f" preprocess 的产物。请重跑 preprocess 重新生成后再装载。"
        )
    return np.asarray(objp1, dtype=np.float64)


def load_pairs(board: BoardSpec, pairs_csv: Path | str,
               npz1_path: Path | str, npz2_path: Path | str,
               exclude_pairs: tuple[str, ...] = (), min_pairs: int = 10) -> Pairs:
    """把对清单与两路角点 npz 对齐装载成幸存双目对集(行序契约在此定死)。

    参数:
        board:         棋盘规格(objp 由此重建,与两路 npz 三方交叉核对);
        pairs_csv:     Phase 1 对清单(utf-8-sig,列 group_id/cam{n}_path/route_cam{n});
        npz1_path:     cam1 角点 npz(如 output/train/cam1_train.npz);
        npz2_path:     cam2 角点 npz;
        exclude_pairs: 手动逻辑剔除的 group_id 清单(原数据不动,与自动机制叠加);
        min_pairs:     幸存对数下限,低于此数中文报错不硬算(一对定不了相对位姿)。
    """
    data1 = _load_corners_checked(Path(npz1_path), 1)
    data2 = _load_corners_checked(Path(npz2_path), 2)

    uv1_all = np.asarray(data1["uv"], dtype=np.float64)
    uv2_all = np.asarray(data2["uv"], dtype=np.float64)
    detected1 = np.asarray(data1["detected"], dtype=bool)
    detected2 = np.asarray(data2["detected"], dtype=bool)
    paths1 = [str(x) for x in np.asarray(data1["paths"]).ravel()]
    paths2 = [str(x) for x in np.asarray(data2["paths"]).ravel()]
    source1_all = [str(x) for x in np.asarray(data1["source"]).ravel()]
    source2_all = [str(x) for x in np.asarray(data2["source"]).ravel()]

    objp = _check_objp(np.asarray(data1["objp"], dtype=np.float64),
                       np.asarray(data2["objp"], dtype=np.float64),
                       build_object_points(board), Path(npz1_path), Path(npz2_path))

    # 连接键(Pattern 1 实测规则):image_id = group_id + "_cam{n}"
    id_to_row1 = {str(x): i for i, x in enumerate(np.asarray(data1["image_ids"]).ravel())}
    id_to_row2 = {str(x): i for i, x in enumerate(np.asarray(data2["image_ids"]).ravel())}

    exclude = set(exclude_pairs)
    kept_ids: list[str] = []
    kept_source1: list[str] = []
    kept_source2: list[str] = []
    kept_routes: list[tuple[str, str]] = []
    uv1_rows: list[np.ndarray] = []
    uv2_rows: list[np.ndarray] = []

    # utf-8-sig:Phase 1 写盘带 BOM,裸 utf-8 会让首列 fieldnames 带 BOM 前缀
    with open(pairs_csv, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for lineno, row in enumerate(reader, start=2):  # start=2:第 1 行是表头
            gid = str(row["group_id"]).strip()
            if gid in exclude:
                continue  # 手动剔除通道:逻辑剔除,原数据不动
            keys = (f"{gid}_cam1", f"{gid}_cam2")
            rows: list[int] = []
            for cam, key, id_to_row in ((1, keys[0], id_to_row1), (2, keys[1], id_to_row2)):
                if key not in id_to_row:
                    raise ValueError(
                        f"对清单 {Path(pairs_csv).name} 第 {lineno} 行的 group_id "
                        f"{gid!r} 不合法:cam{cam} 角点产物里找不到连接键 {key!r}"
                        f"(缺 cam{cam} 路)。Phase 1 产物不一致,请重跑 preprocess。"
                    )
                rows.append(id_to_row[key])
            row1, row2 = rows

            if not (detected1[row1] and detected2[row2]):
                raise ValueError(
                    f"对清单 {Path(pairs_csv).name} 第 {lineno} 行的 group_id "
                    f"{gid!r} 不合法:清单声明两路成对,但 npz 检出标记为 "
                    f"cam1={bool(detected1[row1])}/cam2={bool(detected2[row2])}。"
                    f"Phase 1 产物不一致,请重跑 preprocess。"
                )

            csv1 = str(row["cam1_path"]).strip()
            csv2 = str(row["cam2_path"]).strip()
            if csv1 != paths1[row1] or csv2 != paths2[row2]:
                raise ValueError(
                    f"对清单 {Path(pairs_csv).name} 第 {lineno} 行的 group_id "
                    f"{gid!r} 不合法:路径交叉核对失败(csv cam1_path={csv1!r} vs "
                    f"npz {paths1[row1]!r};csv cam2_path={csv2!r} vs npz "
                    f"{paths2[row2]!r})——连接键命中但图像对不上,产物可疑,"
                    f"请重跑 preprocess。"
                )

            for cam, uv_row in ((1, uv1_all[row1]), (2, uv2_all[row2])):
                if not np.isfinite(uv_row).all():
                    raise ValueError(
                        f"对清单 {Path(pairs_csv).name} 第 {lineno} 行的 group_id "
                        f"{gid!r} 不合法:cam{cam} 路的 uv 字段标记为已检出,但角点里"
                        f"混入了非数值(NaN/无穷),角点文件可疑。请重跑 preprocess。"
                    )

            kept_ids.append(gid)
            kept_source1.append(source1_all[row1])
            kept_source2.append(source2_all[row2])
            kept_routes.append((str(row["route_cam1"]).strip(), str(row["route_cam2"]).strip()))
            uv1_rows.append(uv1_all[row1])
            uv2_rows.append(uv2_all[row2])

    if len(kept_ids) < min_pairs:
        raise ValueError(
            f"{Path(pairs_csv).name} 过滤后只剩 {len(kept_ids)} 个可用双目对"
            f"(未被手动剔除且两路检出),低于下限 {min_pairs}——一对定不了相对位姿,"
            f"硬算必错。请检查 exclude_pairs 清单、对清单行数与 min_pairs 配置。"
        )

    uvs1 = np.asarray(uv1_rows, dtype=np.float64)
    uvs2 = np.asarray(uv2_rows, dtype=np.float64)
    n_points = board.rows * board.cols
    if uvs1.shape != (len(kept_ids), n_points, 2) or uvs2.shape != (len(kept_ids), n_points, 2):
        raise ValueError(  # 幸存形状断言(防线自证)
            f"幸存角点数组形状不合:期望 ({len(kept_ids)},{n_points},2),"
            f"实际是 {uvs1.shape} / {uvs2.shape}。这是装载层内部防线被击穿的信号,"
            f"请把本错误连同运行日志反馈维护者。"
        )
    return Pairs(image_ids=tuple(kept_ids), source1=tuple(kept_source1),
                 source2=tuple(kept_source2), routes=tuple(kept_routes),
                 uv1=uvs1, uv2=uvs2, objp=objp)


def select_pairs(pairs: Pairs, indices) -> Pairs:
    """按索引子集重建幸存对集(剔除循环的幸存集重建入口,绝不原地改)。

    indices 保持调用方给定的相对顺序(如 [2, 0] 就按第 2、第 0 对的顺序
    重建);行序锁的"下游绝不重排"约束由调用方保证——本函数只做与索引
    一一对应的子集抽取。
    """
    idx = np.asarray(list(indices), dtype=np.int64)
    return Pairs(image_ids=tuple(pairs.image_ids[i] for i in idx),
                 source1=tuple(pairs.source1[i] for i in idx),
                 source2=tuple(pairs.source2[i] for i in idx),
                 routes=tuple(pairs.routes[i] for i in idx),
                 uv1=np.asarray(pairs.uv1[idx], dtype=np.float64),
                 uv2=np.asarray(pairs.uv2[idx], dtype=np.float64),
                 objp=pairs.objp)


@dataclass(frozen=True)
class IntrinsicsView:
    """一路相机的固定内参快照(Phase 3 全程只读):K/D 与九名 params 双轨。

    params 按 models.INTRINSIC_NAMES = (fx, fy, cx, cy, k1, k2, p1, p2, k3)
    布局;D 是 OpenCV dist 布局 (k1, k2, p1, p2, k3)——两者切向/径向次序
    不同,按位置硬拼必错(Pitfall 13),故读取时两轨并存、形状当场断言。
    """

    cam: int
    image_size: tuple[int, int]
    params: np.ndarray
    K: np.ndarray
    D: np.ndarray


def load_intrinsics(json_path: Path | str) -> IntrinsicsView:
    """按名读取 Phase 2 的 intrinsics.json(固定内参的唯一读取口)。

    params 九名按 models.INTRINSIC_NAMES 导航(禁魔数下标);优先直读
    K/D 冗余字段并断言 D.shape == (5,)(Pitfall 13 的布局错位防护)。
    缺字段/形状不合均中文报错。
    """
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        # 不用 TypeError:CLI 异常网只收 ValueError 家族,配置/产物错误必须留在网内
        raise ValueError(  # noqa: TRY004
            f"内参文件 {Path(json_path).name} 不合法:顶层应当是 JSON 对象。")
    missing = [key for key in _REQUIRED_JSON_FIELDS if key not in data]
    if missing:
        raise ValueError(
            f"内参文件 {Path(json_path).name} 不合法:缺字段 {missing}——"
            f"它不是 Phase 2 intrinsics 契约的产物,请先运行 stereo-calib intrinsic。"
        )

    named = data["params"]
    if not isinstance(named, dict):
        raise ValueError(  # noqa: TRY004  同上:类型不对也必须走 ValueError 收口
            f"内参文件 {Path(json_path).name} 不合法:params 应当是按名九字段的"
            f"JSON 对象,实际是 {type(named).__name__}。请重跑 stereo-calib intrinsic。"
        )
    missing_names = [name for name in INTRINSIC_NAMES if name not in named]
    if missing_names:
        raise ValueError(
            f"内参文件 {Path(json_path).name} 不合法:params 缺参数名 {missing_names}"
            f"(布局 {INTRINSIC_NAMES})。请重跑 stereo-calib intrinsic。"
        )
    params = np.array([float(named[name]) for name in INTRINSIC_NAMES], dtype=np.float64)

    K = np.asarray(data["K"], dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(
            f"内参文件 {Path(json_path).name} 不合法:K 的形状应当是 (3, 3),"
            f"实际是 {K.shape}。请重跑 stereo-calib intrinsic。"
        )
    D = np.asarray(data["D"], dtype=np.float64).ravel()
    if D.shape != (5,):
        # 布局错位防护:OpenCV dist 布局恰五项,四项/六项都说明写侧坏掉,
        # 静默塞错切向/径向比报错可怕得多(Pitfall 13)
        raise ValueError(
            f"内参文件 {Path(json_path).name} 不合法:D 的形状应当是 (5,)"
            f"(OpenCV dist 布局 k1,k2,p1,p2,k3),实际是 {D.shape}。"
            f"请重跑 stereo-calib intrinsic。"
        )

    image_size_raw = data["image_size"]
    if not isinstance(image_size_raw, (list, tuple)) or len(image_size_raw) != 2:
        raise ValueError(
            f"内参文件 {Path(json_path).name} 不合法:image_size 应当是 [宽, 高]"
            f"两项,实际是 {image_size_raw!r}。请重跑 stereo-calib intrinsic。"
        )
    image_size = (int(image_size_raw[0]), int(image_size_raw[1]))
    return IntrinsicsView(cam=int(data["cam"]), image_size=image_size,
                          params=params, K=K, D=D)
