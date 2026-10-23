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

# scipy 精化的求解方法白名单:'lm' 只支持稠密雅可比,本工程走稀疏路径,
# 参考工程配置层直接拒收的防呆照搬(白名单只收 "trf")
_ALLOWED_REFINE_METHODS = {"trf"}


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
class RefineConfig:
    """模块2 scipy 精化:开关 + 求解方法 + 迭代上限 + 收敛容差。

    OpenCV 初值(calibrateCameraExtended)之后的一次联合精化开关由配置
    控制(成功标准 1);容差默认值取参考工程实测收敛组合(ftol=1e-10 /
    xtol=1e-12 / gtol=1e-10),xtol 压到机器精度量级保住收敛精度。
    """

    enabled: bool = True
    method: str = "trf"
    max_nfev: int = 300
    ftol: float = 1.0e-10
    xtol: float = 1.0e-12
    gtol: float = 1.0e-10


@dataclass(frozen=True)
class OutlierConfig:
    """模块2 自动离群帧剔除:判离群阈值 + 双保险丝 + 幸存视图数下限。

    逐帧 RMS > rms_multiplier × 幸存集中位数判离群(中位数基准稳健);
    max_drop_fraction / max_iterations 是剔除循环的双保险丝(防阈值漂移
    不收敛 / 防吃掉过多视图);min_views 是剔除后幸存视图数下限,低于
    此数直接报错不硬算(两三个视角定不了内参)。只作用于 train,test
    永不剔除只标记(考卷不可改)。
    """

    enabled: bool = True
    rms_multiplier: float = 5.0
    max_drop_fraction: float = 0.2
    max_iterations: int = 3
    min_views: int = 10


@dataclass(frozen=True)
class IntrinsicConfig:
    """模块2 内参标定段:精化 / 离群剔除 / test 阈值 / 手动剔除清单。"""

    refine: RefineConfig = field(default_factory=RefineConfig)
    outlier: OutlierConfig = field(default_factory=OutlierConfig)
    rms_test_max: float = 1.0
    exclude_images: tuple[str, ...] = ()


@dataclass(frozen=True)
class StereoCalibConfig:
    """config.yaml 解析结果:全流程参数的唯一载体(D-10 单一文件分段)。"""

    board: BoardSpec = field(default_factory=BoardSpec)
    data: DataDirs = field(default_factory=DataDirs)
    output_root: str = "output"
    cv2_threads: int = 0
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    intrinsic: IntrinsicConfig = field(default_factory=IntrinsicConfig)


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


def _parse_refine(raw: dict) -> RefineConfig:
    """解析 intrinsic.refine 段:开关(bool)+ 方法白名单 + 迭代/容差校验。"""
    cfg = RefineConfig()
    section = _check_is_dict(raw.get("refine", {}), "intrinsic.refine")

    enabled = section.get("enabled", cfg.enabled)
    if not isinstance(enabled, bool):
        _fail("intrinsic.refine.enabled", enabled,
              "必须是 true 或 false(OpenCV 初值之后是否做 scipy 联合精化)")

    method = section.get("method", cfg.method)
    if method not in _ALLOWED_REFINE_METHODS:
        _fail("intrinsic.refine.method", method,
              f"只能是 {' 或 '.join(sorted(_ALLOWED_REFINE_METHODS))}"
              f"('lm' 只支持稠密雅可比,本工程走稀疏路径,直接拒收防呆)")

    max_nfev = section.get("max_nfev", cfg.max_nfev)
    if not isinstance(max_nfev, int) or isinstance(max_nfev, bool) or max_nfev < 1:
        _fail("intrinsic.refine.max_nfev", max_nfev, "必须是 >= 1 的整数(精化函数求值次数上限)")

    ftol = section.get("ftol", cfg.ftol)
    if not isinstance(ftol, int | float) or isinstance(ftol, bool) or ftol <= 0:
        _fail("intrinsic.refine.ftol", ftol, "必须是 > 0 的数(代价函数相对收敛容差)")

    xtol = section.get("xtol", cfg.xtol)
    if not isinstance(xtol, int | float) or isinstance(xtol, bool) or xtol <= 0:
        _fail("intrinsic.refine.xtol", xtol, "必须是 > 0 的数(参数相对收敛容差)")

    gtol = section.get("gtol", cfg.gtol)
    if not isinstance(gtol, int | float) or isinstance(gtol, bool) or gtol <= 0:
        _fail("intrinsic.refine.gtol", gtol, "必须是 > 0 的数(梯度无穷范数收敛容差)")

    return RefineConfig(enabled=enabled, method=method, max_nfev=max_nfev,
                        ftol=float(ftol), xtol=float(xtol), gtol=float(gtol))


def _parse_outlier(raw: dict) -> OutlierConfig:
    """解析 intrinsic.outlier 段:开关(bool)+ 阈值/保险丝/视图下限校验。"""
    cfg = OutlierConfig()
    section = _check_is_dict(raw.get("outlier", {}), "intrinsic.outlier")

    enabled = section.get("enabled", cfg.enabled)
    if not isinstance(enabled, bool):
        _fail("intrinsic.outlier.enabled", enabled,
              "必须是 true 或 false(是否自动剔除离群帧后重标定)")

    rms_multiplier = section.get("rms_multiplier", cfg.rms_multiplier)
    if (not isinstance(rms_multiplier, int | float) or isinstance(rms_multiplier, bool)
            or rms_multiplier <= 1.0):
        _fail("intrinsic.outlier.rms_multiplier", rms_multiplier,
              "必须是 > 1.0 的数(逐帧 RMS > k×幸存集中位数判离群;k<=1 无判别力)")

    max_drop_fraction = section.get("max_drop_fraction", cfg.max_drop_fraction)
    if (not isinstance(max_drop_fraction, int | float) or isinstance(max_drop_fraction, bool)
            or not 0 < max_drop_fraction < 1):
        _fail("intrinsic.outlier.max_drop_fraction", max_drop_fraction,
              "必须是 0 到 1 之间(不含端点)的小数(累计剔除比例保险丝)")

    max_iterations = section.get("max_iterations", cfg.max_iterations)
    if (not isinstance(max_iterations, int) or isinstance(max_iterations, bool)
            or max_iterations < 1):
        _fail("intrinsic.outlier.max_iterations", max_iterations,
              "必须是 >= 1 的整数(剔除-重标定循环轮数上限)")

    min_views = section.get("min_views", cfg.min_views)
    if not isinstance(min_views, int) or isinstance(min_views, bool) or min_views < 2:
        _fail("intrinsic.outlier.min_views", min_views,
              "必须是 >= 2 的整数(剔除后幸存视图数下限;两三个视角定不了内参)")

    return OutlierConfig(enabled=enabled, rms_multiplier=float(rms_multiplier),
                         max_drop_fraction=float(max_drop_fraction),
                         max_iterations=max_iterations, min_views=min_views)


def _parse_intrinsic(raw: dict) -> IntrinsicConfig:
    """解析 intrinsic 段(模块2 全部算法参数/开关,CLI 零算法参数)。"""
    section = _check_is_dict(raw.get("intrinsic", {}), "intrinsic")

    rms_test_max = section.get("rms_test_max", IntrinsicConfig.rms_test_max)
    if (not isinstance(rms_test_max, int | float) or isinstance(rms_test_max, bool)
            or rms_test_max <= 0):
        _fail("intrinsic.rms_test_max", rms_test_max,
              "必须是 > 0 的数(test 集整体 RMS 判 PASS/FAIL 的阈值,像素)")

    exclude = section.get("exclude_images", IntrinsicConfig.exclude_images)
    if not isinstance(exclude, (list, tuple)) or not all(isinstance(x, str) for x in exclude):
        _fail("intrinsic.exclude_images", exclude,
              "必须是一列图像编号文字(如 [20261023_154025_519359_cam1]),逐项手动逻辑剔除")

    return IntrinsicConfig(refine=_parse_refine(section), outlier=_parse_outlier(section),
                           rms_test_max=float(rms_test_max), exclude_images=tuple(exclude))


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
    intrinsic = _parse_intrinsic(raw)

    cfg = StereoCalibConfig(board=board, data=data, output_root=output_root,
                            cv2_threads=cv2_threads, preprocess=preprocess,
                            intrinsic=intrinsic)
    _check_output_data_disjoint(cfg)
    return cfg
