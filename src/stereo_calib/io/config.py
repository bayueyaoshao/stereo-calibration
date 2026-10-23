"""配置层:单一 config.yaml 的解析与中文校验(D-10 / INFRA-02)。

体例移植自参考工程 calib.io.config(只读引用,绝不修改):frozen dataclass
载体、统一中文 _fail 报错、yaml.safe_load 防线(绝不 yaml.load,防配置执行
任意代码)、逐字段 isinstance 校验、布尔字段先判 bool 再判 int(True 是 int
的陷阱,yaml 手写 1/0 必须拦下)。

本工程新增(参考没有,checker 修订定案):分段解析完成后做跨字段数据安全
校验——data.train / data.test 与 output_root 相等或互相包含的配置当场拒绝,
防后续运行前清理输出目录时误删数据、防产物写进 data/。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

# 经典检测器允许的选项名(写在 config.yaml 里的名字,代码内翻译成 cv2 常量)
_ALLOWED_CLASSIC_FLAGS = {"adaptive_thresh", "normalize_image", "fast_check"}

# 亚像素窗口粒度的合法取值:frame = 每帧一个窗口;corner = 逐角点局部自适应
_ALLOWED_GRANULARITY = {"frame", "corner"}


@dataclass(frozen=True)
class BoardSpec:
    """棋盘规格(来自 config.yaml 的 board 段)。

    square_size_mm 决定外参平移的物理尺度,不影响内参——它绝不进入任何
    像素级的逻辑(角点窗口等用实测方格像素边长,不用它)。
    """

    cols: int = 11          # 内部角点列数
    rows: int = 8           # 内部角点行数
    square_size_mm: float = 25.0
    unit: str = "mm"


@dataclass(frozen=True)
class DataDirs:
    """数据目录(相对当前工作目录)。"""

    train: str = "data/train"
    test: str = "data/test"


@dataclass(frozen=True)
class SubpixConfig:
    """亚像素精化参数(把角点位置从整数像素细化到小数像素)。

    窗口从每帧实测的方格像素边长按比例推导,这里只控制粒度与下限;
    半窗口绝不允许大到跨过方格对边,否则会产生重投影误差里完全看不见的
    系统性定位偏差(本项目头号陷阱)。
    """

    window_granularity: str = "frame"
    min_win: int = 3
    max_iter: int = 30
    epsilon: float = 0.01


@dataclass(frozen=True)
class CornersConfig:
    """角点检测参数:SB 主路线 + classic 兜底(双检测器)。"""

    classic_flags: tuple[str, ...] = ("adaptive_thresh", "normalize_image")
    sb_flags: int = 0
    # SB 未检出的帧是否用经典检测器的结果兜底(01-02 接入消费)。
    sb_fallback_classic: bool = True
    subpix: SubpixConfig = field(default_factory=SubpixConfig)


@dataclass(frozen=True)
class PreprocessConfig:
    """模块1 数据预处理段:并行/开关/可视化。"""

    max_workers: int = 12
    corners: CornersConfig = field(default_factory=CornersConfig)
    copy_fail_originals: bool = True
    jpeg_quality: int = 90   # 来自 preprocess.visualization.jpeg_quality(01-03 消费)


@dataclass(frozen=True)
class StereoCalibConfig:
    """config.yaml 解析结果:全流程参数的唯一载体(D-10 单一文件分段)。"""

    board: BoardSpec = field(default_factory=BoardSpec)
    data: DataDirs = field(default_factory=DataDirs)
    output_root: str = "output"
    cv2_threads: int = 0
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)


def _fail(field_name: str, actual, requirement: str) -> None:
    """抛出带字段名与实际值的中文配置错误(统一格式,便于定位)。"""
    raise ValueError(
        f"配置字段 {field_name} 不合法:{requirement};实际读到的值是 {actual!r}。"
        f"请检查对应的 yaml 文件。"
    )


def _check_is_dict(raw, field_name: str) -> dict:
    """确认 yaml 里某一段是"键: 值"结构,否则给中文报错。"""
    if not isinstance(raw, dict):
        _fail(field_name, raw, "应当是一组缩进的配置项(字典)")
    return raw


def _read_yaml(path: Path | str) -> object:
    """读 yaml 文件,safe_load(禁 yaml.load,防配置执行任意代码)。

    yaml **语法**错误(如缩进用了 tab)也归入中文报错防线:转成带文件
    路径的 ValueError,消息附 PyYAML 原始信息(行号/列号对定位有用),
    并保留因果链(raise ... from)——PyYAML 自己抛的 ScannerError 不是
    ValueError 子类,不转的话 CLI 入口的 except (ValueError, OSError)
    防线会被击穿。
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"配置文件不存在:{path}。请确认路径(相对当前工作目录)是否正确。")
    with open(path, encoding="utf-8") as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(
                f"配置文件 {path} 的 yaml 语法有误,请检查缩进与冒号:{e}"
            ) from e


def _parse_board(raw: dict) -> BoardSpec:
    """解析 board 段:board_type 只支持 chessboard;cols/rows/square 校验。"""
    if "board" not in raw:
        _fail("board", raw.get("board"), "文件顶层必须有 board: 这一段")
    section = _check_is_dict(raw["board"], "board")

    board_type = section.get("board_type")
    if board_type != "chessboard":
        _fail("board.board_type", board_type, "当前只支持 chessboard(棋盘格)")

    cols = section.get("cols")
    if not isinstance(cols, int) or isinstance(cols, bool) or cols < 3:
        _fail("board.cols", cols, "必须是 >= 3 的整数(内部角点列数)")

    rows = section.get("rows")
    if not isinstance(rows, int) or isinstance(rows, bool) or rows < 3:
        _fail("board.rows", rows, "必须是 >= 3 的整数(内部角点行数)")

    square = section.get("square_size_mm")
    if not isinstance(square, int | float) or isinstance(square, bool) or square <= 0:
        _fail("board.square_size_mm", square, "必须是 > 0 的数(方格物理边长,毫米)")

    unit = section.get("unit", "mm")
    if not isinstance(unit, str) or not unit:
        _fail("board.unit", unit, "必须是长度单位的文字(如 mm)")

    return BoardSpec(cols=cols, rows=rows, square_size_mm=float(square), unit=unit)


def _parse_data(raw: dict) -> DataDirs:
    """解析 data 段:train/test 必须是非空路径文字。"""
    if "data" not in raw:
        _fail("data", raw.get("data"), "文件顶层必须有 data: 这一段")
    section = _check_is_dict(raw["data"], "data")

    train = section.get("train")
    if not isinstance(train, str) or not train:
        _fail("data.train", train, "必须是训练数据目录的非空路径文字(相对当前工作目录)")

    test = section.get("test")
    if not isinstance(test, str) or not test:
        _fail("data.test", test, "必须是测试数据目录的非空路径文字(相对当前工作目录)")

    return DataDirs(train=train, test=test)


def _parse_subpix(raw: dict) -> SubpixConfig:
    """解析 preprocess.corners.subpix 段:粒度白名单 + 数值下限。"""
    cfg = SubpixConfig()
    section = _check_is_dict(raw.get("subpix", {}), "preprocess.corners.subpix")
    granularity = section.get("window_granularity", cfg.window_granularity)
    if granularity not in _ALLOWED_GRANULARITY:
        _fail("preprocess.corners.subpix.window_granularity", granularity,
              f"只能是 {' 或 '.join(sorted(_ALLOWED_GRANULARITY))}")
    min_win = section.get("min_win", cfg.min_win)
    if not isinstance(min_win, int) or isinstance(min_win, bool) or min_win < 1:
        _fail("preprocess.corners.subpix.min_win", min_win, "必须是 >= 1 的整数(半窗口下限,像素)")
    max_iter = section.get("max_iter", cfg.max_iter)
    if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter < 1:
        _fail("preprocess.corners.subpix.max_iter", max_iter, "必须是 >= 1 的整数(迭代次数上限)")
    epsilon = section.get("epsilon", cfg.epsilon)
    if not isinstance(epsilon, int | float) or isinstance(epsilon, bool) or epsilon <= 0:
        _fail("preprocess.corners.subpix.epsilon", epsilon, "必须是 > 0 的数(收敛容差,像素)")
    return SubpixConfig(window_granularity=granularity, min_win=min_win,
                        max_iter=max_iter, epsilon=float(epsilon))


def _parse_corners(raw: dict) -> CornersConfig:
    """解析 preprocess.corners 段:双检测器开关与选项。"""
    section = _check_is_dict(raw.get("corners", {}), "preprocess.corners")
    flags = section.get("classic_flags", CornersConfig.classic_flags)
    if not isinstance(flags, (list, tuple)) or not all(isinstance(f, str) for f in flags):
        _fail("preprocess.corners.classic_flags", flags, "必须是一列选项名(如 [adaptive_thresh])")
    bad = [f for f in flags if f not in _ALLOWED_CLASSIC_FLAGS]
    if bad:
        _fail("preprocess.corners.classic_flags", bad,
              f"含有不认识的选项;可选:{sorted(_ALLOWED_CLASSIC_FLAGS)}")
    sb_flags = section.get("sb_flags", CornersConfig.sb_flags)
    if not isinstance(sb_flags, int) or isinstance(sb_flags, bool) or sb_flags < 0:
        _fail("preprocess.corners.sb_flags", sb_flags, "必须是 >= 0 的整数(选项位掩码)")
    fallback = section.get("sb_fallback_classic", CornersConfig.sb_fallback_classic)
    # 注意顺序:Python 里 True 同时也是 int(issubclass(bool, int)),但 1 不是 bool,
    # 所以先判 bool 再拒其余一切——整数 1/0 是 yaml 手写时最常见的笔误,必须拦下
    if not isinstance(fallback, bool):
        _fail("preprocess.corners.sb_fallback_classic", fallback,
              "必须是 true 或 false(SB 未检出的帧是否用经典检测器结果兜底)")
    return CornersConfig(classic_flags=tuple(flags), sb_flags=sb_flags,
                         sb_fallback_classic=fallback,
                         subpix=_parse_subpix(section))


def _parse_preprocess(raw: dict) -> PreprocessConfig:
    """解析 preprocess 段:max_workers / corners / 开关 / 可视化。"""
    section = _check_is_dict(raw.get("preprocess", {}), "preprocess")

    max_workers = section.get("max_workers", PreprocessConfig.max_workers)
    if not isinstance(max_workers, int) or isinstance(max_workers, bool) or max_workers < 1:
        _fail("preprocess.max_workers", max_workers, "必须是 >= 1 的整数(并行进程数;1 = 串行)")

    corners = _parse_corners(section)

    copy_fail = section.get("copy_fail_originals", PreprocessConfig.copy_fail_originals)
    if not isinstance(copy_fail, bool):
        _fail("preprocess.copy_fail_originals", copy_fail,
              "必须是 true 或 false(失败帧原图是否拷贝到输出目录)")

    vis = _check_is_dict(section.get("visualization", {}), "preprocess.visualization")
    quality = vis.get("jpeg_quality", PreprocessConfig.jpeg_quality)
    if not isinstance(quality, int) or isinstance(quality, bool) or not 1 <= quality <= 100:
        _fail("preprocess.visualization.jpeg_quality", quality,
              "必须是 1 到 100 之间的整数(JPEG 质量)")

    return PreprocessConfig(max_workers=max_workers, corners=corners,
                            copy_fail_originals=copy_fail, jpeg_quality=quality)


def _check_output_data_disjoint(cfg: StereoCalibConfig) -> None:
    """跨字段数据安全校验(CLAUDE.md data 只读 + INFRA-03 前置防线)。

    output_root 与 data.train / data.test 相等或互相包含的配置当场拒绝:
    运行前清理输出目录会把数据一起删掉(数据只读被击穿);反向包含则会把
    产物写进 data/。resolve() 用默认非 strict 模式——目录不存在也能解析
    出规范路径,加载层就要拦得住(不是等目录真建出来才报错)。
    """
    out = Path(cfg.output_root).resolve()
    for name in ("train", "test"):
        data_dir = Path(getattr(cfg.data, name)).resolve()
        if data_dir == out:
            direction = "两目录指向同一位置"
        elif out in data_dir.parents:
            direction = "数据目录位于输出目录之内(运行前清理输出目录会把数据一起删除)"
        elif data_dir in out.parents:
            direction = "输出目录位于数据目录之内(产物会写进数据目录)"
        else:
            continue
        raise ValueError(
            f"配置字段 data.{name}/output_root 不合法:数据目录与输出目录不得相等或互相包含"
            f"(清理输出目录会删除数据/产物会写进数据目录);本次重叠形态:{direction};"
            f"实际:data.{name}={data_dir},output_root={out}。请检查对应的 yaml 文件。"
        )


def load_config(path: Path | str) -> StereoCalibConfig:
    """解析并校验单一 config.yaml,返回不可变的 StereoCalibConfig(D-10)。

    一次 _read_yaml 后分段解析;分段全部通过后做跨字段数据安全校验。
    任何不合:抛 ValueError,消息含字段名与实际值(中文)。
    """
    raw = _read_yaml(path)
    raw = _check_is_dict(raw, "(顶层)")

    board = _parse_board(raw)
    data = _parse_data(raw)

    output_root = raw.get("output_root", StereoCalibConfig.output_root)
    if not isinstance(output_root, str) or not output_root:
        _fail("output_root", output_root, "必须是输出根目录的非空路径文字(如 \"output\")")

    cv2_threads = raw.get("cv2_threads", StereoCalibConfig.cv2_threads)
    if not isinstance(cv2_threads, int) or isinstance(cv2_threads, bool):
        _fail("cv2_threads", cv2_threads, "必须是整数(OpenCV 内部线程数,0 = 关闭)")

    preprocess = _parse_preprocess(raw)

    cfg = StereoCalibConfig(board=board, data=data, output_root=output_root,
                            cv2_threads=cv2_threads, preprocess=preprocess)
    _check_output_data_disjoint(cfg)
    return cfg
