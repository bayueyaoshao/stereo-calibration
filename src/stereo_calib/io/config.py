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

# 联合 BA 的结构化变体白名单(03-05):fc = 只放两路 f/c(D 冻结);
# fc_prior = fc + Tikhonov 先验(f/c 相对偏差);full_prior = 全内参放开 + 先验
_ALLOWED_JOINT_VARIANTS = {"fc", "fc_prior", "full_prior"}

# 畸变模型白名单(2026-10-24 增补 M8,2026-10-26 增补 M4):
# M2 = 五参 Brown-Conrady(legacy);M8 = rational 8(本相机实测充分);
# M4 = 鱼眼等距 KB4(cv2.fisheye 同族,8 参),用于后续鱼眼双目;布局统一
# 见 intrinsic.models(INTRINSIC_NAMES_M2/M8/M4,按长度一一对应,禁止按位置猜)
_ALLOWED_DISTORTION_MODELS = {"M2", "M8", "M4"}


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
class IntrinsicDiagnosticsConfig:
    """模块2 内参模型充分性体检段:四个检查的红线阈值 + verdict 判定规则。

    背景(2026-10-24 根因):M2 五参畸变模型对本相机欠拟合时,标量重投影
    RMS 看不出来(每帧自由位姿 + fx 会吸收模型误差)。体检用四个仅凭自家
    数据 + 自家结果的检查揪出这类问题,任意两项红灯即判「内参不可靠」
    (fail-closed,禁止消费该内参进入双目标定):
      A0 残差能量:    PnP 逐点池化 RMS / 中心桶噪声底,比值上限(倍);
      A1 残差径向结构:按归一化半径 5 桶的径向分量桶均值绝对值上限(px);
      A3 切片 fx 稳定性:6 切片重标 fx 极差上限(%),单切片最少帧数下限;
      A4 模型阶次探针:升阶 rational8 重标的 fx 跳变上限(%)。
    阈值依据:rational8 对照实测通过值(A1 0.0035 / A3 0.32 / A0 ~1)留
    3~5× 裕量。A2 去畸变直线度仅报告不判红(实测无分辨力)。
    """

    enabled: bool = True
    a0_max_ratio: float = 2.0
    a1_max_bin_mean_px: float = 0.02
    a3_max_fx_range_pct: float = 0.5
    a3_min_frames: int = 20
    a4_max_fx_jump_pct: float = 2.0
    min_red_count: int = 2


@dataclass(frozen=True)
class IntrinsicConfig:
    """模块2 内参标定段:畸变模型 / 精化 / 离群剔除 / test 阈值 / 手动剔除清单。

    distortion_model 选畸变布局("M2" 五参 Brown-Conrady,legacy;
    "M8" rational 8——本相机 8mm C-mount 4MP 实测充分,2026-10-24 增补;
    "M4" 鱼眼等距 KB4——8 参 cv2.fisheye 同族,2026-10-26 增补,走另一条
    求解路线(见 intrinsic.solve._solve_opencv_m4)),
    默认 "M2" 保持旧配置行为零改动。生产当前用 "M8"。
    """

    distortion_model: str = "M2"
    refine: RefineConfig = field(default_factory=RefineConfig)
    outlier: OutlierConfig = field(default_factory=OutlierConfig)
    diagnostics: IntrinsicDiagnosticsConfig = field(
        default_factory=IntrinsicDiagnosticsConfig)
    rms_test_max: float = 1.0
    exclude_images: tuple[str, ...] = ()


@dataclass(frozen=True)
class StereoRefineConfig:
    """模块3 scipy 外参精化:开关 + 求解方法 + 迭代上限 + 收敛容差。

    与模块2 RefineConfig 同构(θ = [6 外参 | 6×N 位姿] 的联合 BA);
    容差默认值沿用 Phase 2 实测收敛组合,max_nfev 取 200(112 对 678 参数
    规模留裕量;参考工程同规模实测数十次内收敛)。
    """

    enabled: bool = True
    method: str = "trf"
    max_nfev: int = 200
    ftol: float = 1.0e-10
    xtol: float = 1.0e-12
    gtol: float = 1.0e-10


@dataclass(frozen=True)
class StereoOutlierConfig:
    """模块3 双目自动离群对剔除:判离群阈值 + 双保险丝 + 幸存对数下限。

    判据对象是逐对重投影 RMS(两路 pooled,float64 回代;Phase 2 是逐帧);
    min_pairs 是剔除后幸存对数下限,双目侧对应 Phase 2 的 min_views。只
    作用于 train pair,test 永不剔除只标记(考卷不可改)。
    """

    enabled: bool = True
    rms_multiplier: float = 5.0
    max_drop_fraction: float = 0.2
    max_iterations: int = 3
    min_pairs: int = 10


@dataclass(frozen=True)
class StereoJointConfig:
    """模块3 联合 BA 段(03-05):内参 + 公共外参 + 逐对位姿联合精化。

    结构化变体裁决体例(03-02 实测全内参放开始联合 descent 不收敛,必须
    结构化约束防内参无界漂移):fc 只放两路焦距/主点(D 冻结在 Phase 2
    初值);fc_prior 在 fc 之上加 Tikhonov 先验(f/c 相对偏差拉回);
    full_prior 全内参放开 + 先验(D 绝对偏差拉回)。prior_weight 是先验行
    的统一乘子;spread_ratio_max 是联合解散布防线阈值(精化后逐对残差
    max/median,防翻转帧/孪生解污染的联合解带病出链)。enabled 默认 false:
    rigid 路径行为零改动的开关保证。
    """

    enabled: bool = False
    variant: str = "fc_prior"
    prior_weight: float = 1.0
    spread_ratio_max: float = 5.0


@dataclass(frozen=True)
class StereoConfig:
    """模块3 双目外参标定段:精化 / 离群剔除 / 消歧与防线阈值 / 手动剔除清单。

    flip_ratio 是 180° 角点序消歧的判定阈值(证据充分才翻,默认 0.5 取自
    参考工程实测;上限 1.0——超过即"证据不足也翻",拒收);median_spread
    是 Rodrigues 中位离散度防线;baseline_band_mm 是基线量级 sanity 硬带
    (反序或塌缩成一点都会让 sanity 检查失效);joint 是联合 BA 子段
    (默认关)。
    """

    refine: StereoRefineConfig = field(default_factory=StereoRefineConfig)
    outlier: StereoOutlierConfig = field(default_factory=StereoOutlierConfig)
    flip_ratio: float = 0.5
    median_spread_max_deg: float = 1.0
    distance_rms_max_mm: float = 5.0
    baseline_band_mm: tuple[float, float] = (20.0, 2000.0)
    exclude_pairs: tuple[str, ...] = ()
    joint: StereoJointConfig = field(default_factory=StereoJointConfig)


@dataclass(frozen=True)
class ValidationConfig:
    """模块4 测量精度验证段:消费冻结标定产物对 data/val 做漂移体检。

    data_dir 是验证图像对根目录(<组目录>/cam1.jpg + cam2.jpg);两个阈值是
    冻结模式对角距离偏差的判定红线(distance_bias_max_mm 管 |平均偏差|,
    distance_rms_max_mm 管 RMS;现役系统 test 距离 RMS 0.20mm,1mm 红线约
    5× 噪声底,超线即「参数已漂」的强信号);region_grid 是按板中心在
    cam1 视场的区域分桶(行,列),[0, 0] 关闭区域分析;min_pairs 是有效
    验证对数下限,低于此数中文报错不硬算(fail-closed)。
    """

    data_dir: str = "data/val"
    distance_bias_max_mm: float = 1.0
    distance_rms_max_mm: float = 2.0
    region_grid: tuple[int, int] = (3, 3)
    min_pairs: int = 1


@dataclass(frozen=True)
class StereoCalibConfig:
    """config.yaml 解析结果:全流程参数的唯一载体(D-10 单一文件分段)。"""

    board: BoardSpec = field(default_factory=BoardSpec)
    data: DataDirs = field(default_factory=DataDirs)
    output_root: str = "output"
    cv2_threads: int = 0
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    intrinsic: IntrinsicConfig = field(default_factory=IntrinsicConfig)
    stereo: StereoConfig = field(default_factory=StereoConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)


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
    # 先判 str 再做集合成员检查:容器型 yaml 值(如 [frame])不可哈希,
    # 直接 in set 会抛 TypeError(unhashable)击穿 CLI 异常网(WR-01)
    if not isinstance(granularity, str) or granularity not in _ALLOWED_GRANULARITY:
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
    # 先判 str 再做集合成员检查:容器型 yaml 值(如 [trf])不可哈希,
    # 直接 in set 会抛 TypeError(unhashable)击穿 CLI 异常网(WR-01)
    if not isinstance(method, str) or method not in _ALLOWED_REFINE_METHODS:
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


def _parse_diagnostics(raw: dict) -> IntrinsicDiagnosticsConfig:
    """解析 intrinsic.diagnostics 段:体检总开关 + 四检查红线阈值 + verdict 规则。"""
    cfg = IntrinsicDiagnosticsConfig()
    section = _check_is_dict(raw.get("diagnostics", {}), "intrinsic.diagnostics")

    enabled = section.get("enabled", cfg.enabled)
    if not isinstance(enabled, bool):
        _fail("intrinsic.diagnostics.enabled", enabled,
              "必须是 true 或 false(是否在标定完成后做模型充分性体检)")

    a0_max_ratio = section.get("a0_max_ratio", cfg.a0_max_ratio)
    if (not isinstance(a0_max_ratio, int | float) or isinstance(a0_max_ratio, bool)
            or a0_max_ratio <= 1.0):
        _fail("intrinsic.diagnostics.a0_max_ratio", a0_max_ratio,
              "必须是 > 1.0 的数(A0 红:池化 RMS / 中心桶噪声底 的倍数上限;≤1 无判别力)")

    a1_max_bin_mean = section.get("a1_max_bin_mean_px", cfg.a1_max_bin_mean_px)
    if (not isinstance(a1_max_bin_mean, int | float) or isinstance(a1_max_bin_mean, bool)
            or a1_max_bin_mean <= 0):
        _fail("intrinsic.diagnostics.a1_max_bin_mean_px", a1_max_bin_mean,
              "必须是 > 0 的数(A1 红:径向分量桶均值绝对值上限,像素)")

    a3_max_fx_range = section.get("a3_max_fx_range_pct", cfg.a3_max_fx_range_pct)
    if (not isinstance(a3_max_fx_range, int | float) or isinstance(a3_max_fx_range, bool)
            or a3_max_fx_range <= 0):
        _fail("intrinsic.diagnostics.a3_max_fx_range_pct", a3_max_fx_range,
              "必须是 > 0 的数(A3 红:切片重标 fx 极差上限,%)")

    a3_min_frames = section.get("a3_min_frames", cfg.a3_min_frames)
    if (not isinstance(a3_min_frames, int) or isinstance(a3_min_frames, bool)
            or a3_min_frames < 1):
        _fail("intrinsic.diagnostics.a3_min_frames", a3_min_frames,
              "必须是 >= 1 的整数(A3 单切片最少帧数,低于则跳过该切片)")

    a4_max_fx_jump = section.get("a4_max_fx_jump_pct", cfg.a4_max_fx_jump_pct)
    if (not isinstance(a4_max_fx_jump, int | float) or isinstance(a4_max_fx_jump, bool)
            or a4_max_fx_jump <= 0):
        _fail("intrinsic.diagnostics.a4_max_fx_jump_pct", a4_max_fx_jump,
              "必须是 > 0 的数(A4 红:升阶 rational8 重标的 fx 跳变上限,%)")

    min_red_count = section.get("min_red_count", cfg.min_red_count)
    if (not isinstance(min_red_count, int) or isinstance(min_red_count, bool)
            or not 1 <= min_red_count <= 4):
        _fail("intrinsic.diagnostics.min_red_count", min_red_count,
              "必须是 1 到 4 之间的整数(verdict 判不可靠所需最少红灯数;"
              "红检只有 A0/A1/A3/A4 四个,>4 永远判不可达)")

    return IntrinsicDiagnosticsConfig(
        enabled=enabled, a0_max_ratio=float(a0_max_ratio),
        a1_max_bin_mean_px=float(a1_max_bin_mean),
        a3_max_fx_range_pct=float(a3_max_fx_range), a3_min_frames=a3_min_frames,
        a4_max_fx_jump_pct=float(a4_max_fx_jump), min_red_count=min_red_count)


def _parse_intrinsic(raw: dict) -> IntrinsicConfig:
    """解析 intrinsic 段(模块2 全部算法参数/开关,CLI 零算法参数)。"""
    section = _check_is_dict(raw.get("intrinsic", {}), "intrinsic")

    model = section.get("distortion_model", IntrinsicConfig.distortion_model)
    # 先判 str 再做集合成员检查:容器型 yaml 值(如 [M8])不可哈希,
    # 直接 in set 会抛 TypeError(unhashable)击穿 CLI 异常网(WR-01)
    if not isinstance(model, str) or model not in _ALLOWED_DISTORTION_MODELS:
        _fail("intrinsic.distortion_model", model,
              f"只能是 {' 或 '.join(sorted(_ALLOWED_DISTORTION_MODELS))}"
              f"(M2=五参 Brown-Conrady legacy;M8=rational 8;M4=鱼眼 KB4)")

    rms_test_max = section.get("rms_test_max", IntrinsicConfig.rms_test_max)
    if (not isinstance(rms_test_max, int | float) or isinstance(rms_test_max, bool)
            or rms_test_max <= 0):
        _fail("intrinsic.rms_test_max", rms_test_max,
              "必须是 > 0 的数(test 集整体 RMS 判 PASS/FAIL 的阈值,像素)")

    exclude = section.get("exclude_images", IntrinsicConfig.exclude_images)
    if not isinstance(exclude, (list, tuple)) or not all(isinstance(x, str) for x in exclude):
        _fail("intrinsic.exclude_images", exclude,
              "必须是一列图像编号文字(如 [20261023_154025_519359_cam1]),逐项手动逻辑剔除")

    return IntrinsicConfig(distortion_model=model,
                           refine=_parse_refine(section), outlier=_parse_outlier(section),
                           diagnostics=_parse_diagnostics(section),
                           rms_test_max=float(rms_test_max), exclude_images=tuple(exclude))


def _parse_stereo_refine(raw: dict) -> StereoRefineConfig:
    """解析 stereo.refine 段:开关(bool)+ 方法白名单 + 迭代/容差校验。"""
    cfg = StereoRefineConfig()
    section = _check_is_dict(raw.get("refine", {}), "stereo.refine")

    enabled = section.get("enabled", cfg.enabled)
    if not isinstance(enabled, bool):
        _fail("stereo.refine.enabled", enabled,
              "必须是 true 或 false(PnP 初值之后是否做 scipy 外参联合精化)")

    method = section.get("method", cfg.method)
    # 先判 str 再做集合成员检查:容器型 yaml 值(如 [trf])不可哈希,
    # 直接 in set 会抛 TypeError(unhashable)击穿 CLI 异常网(WR-01)
    if not isinstance(method, str) or method not in _ALLOWED_REFINE_METHODS:
        _fail("stereo.refine.method", method,
              f"只能是 {' 或 '.join(sorted(_ALLOWED_REFINE_METHODS))}"
              f"('lm' 只支持稠密雅可比,本工程走稀疏路径,直接拒收防呆)")

    max_nfev = section.get("max_nfev", cfg.max_nfev)
    if not isinstance(max_nfev, int) or isinstance(max_nfev, bool) or max_nfev < 1:
        _fail("stereo.refine.max_nfev", max_nfev, "必须是 >= 1 的整数(精化函数求值次数上限)")

    ftol = section.get("ftol", cfg.ftol)
    if not isinstance(ftol, int | float) or isinstance(ftol, bool) or ftol <= 0:
        _fail("stereo.refine.ftol", ftol, "必须是 > 0 的数(代价函数相对收敛容差)")

    xtol = section.get("xtol", cfg.xtol)
    if not isinstance(xtol, int | float) or isinstance(xtol, bool) or xtol <= 0:
        _fail("stereo.refine.xtol", xtol, "必须是 > 0 的数(参数相对收敛容差)")

    gtol = section.get("gtol", cfg.gtol)
    if not isinstance(gtol, int | float) or isinstance(gtol, bool) or gtol <= 0:
        _fail("stereo.refine.gtol", gtol, "必须是 > 0 的数(梯度无穷范数收敛容差)")

    return StereoRefineConfig(enabled=enabled, method=method, max_nfev=max_nfev,
                              ftol=float(ftol), xtol=float(xtol), gtol=float(gtol))


def _parse_stereo_outlier(raw: dict) -> StereoOutlierConfig:
    """解析 stereo.outlier 段:开关(bool)+ 阈值/保险丝/对数下限校验。"""
    cfg = StereoOutlierConfig()
    section = _check_is_dict(raw.get("outlier", {}), "stereo.outlier")

    enabled = section.get("enabled", cfg.enabled)
    if not isinstance(enabled, bool):
        _fail("stereo.outlier.enabled", enabled,
              "必须是 true 或 false(是否自动剔除离群对后重解)")

    rms_multiplier = section.get("rms_multiplier", cfg.rms_multiplier)
    if (not isinstance(rms_multiplier, int | float) or isinstance(rms_multiplier, bool)
            or rms_multiplier <= 1.0):
        _fail("stereo.outlier.rms_multiplier", rms_multiplier,
              "必须是 > 1.0 的数(逐对重投影 RMS > k×幸存中位数判离群;k<=1 无判别力)")

    max_drop_fraction = section.get("max_drop_fraction", cfg.max_drop_fraction)
    if (not isinstance(max_drop_fraction, int | float) or isinstance(max_drop_fraction, bool)
            or not 0 < max_drop_fraction < 1):
        _fail("stereo.outlier.max_drop_fraction", max_drop_fraction,
              "必须是 0 到 1 之间(不含端点)的小数(累计剔除比例保险丝)")

    max_iterations = section.get("max_iterations", cfg.max_iterations)
    if (not isinstance(max_iterations, int) or isinstance(max_iterations, bool)
            or max_iterations < 1):
        _fail("stereo.outlier.max_iterations", max_iterations,
              "必须是 >= 1 的整数(剔除-重解循环轮数上限)")

    min_pairs = section.get("min_pairs", cfg.min_pairs)
    if not isinstance(min_pairs, int) or isinstance(min_pairs, bool) or min_pairs < 2:
        _fail("stereo.outlier.min_pairs", min_pairs,
              "必须是 >= 2 的整数(剔除后幸存对数下限;一对定不了相对位姿)")

    return StereoOutlierConfig(enabled=enabled, rms_multiplier=float(rms_multiplier),
                               max_drop_fraction=float(max_drop_fraction),
                               max_iterations=max_iterations, min_pairs=min_pairs)


def _parse_stereo_joint(raw: dict) -> StereoJointConfig:
    """解析 stereo.joint 段(03-05 联合 BA):开关(bool)+ 变体白名单 +
    先验权重正性 + 散布比判别力校验(照 _parse_stereo_refine 体例)。"""
    cfg = StereoJointConfig()
    section = _check_is_dict(raw.get("joint", {}), "stereo.joint")

    enabled = section.get("enabled", cfg.enabled)
    if not isinstance(enabled, bool):
        _fail("stereo.joint.enabled", enabled,
              "必须是 true 或 false(是否启用内参+外参+逐对位姿联合 BA;默认关,"
              "rigid 路径行为零改动)")

    variant = section.get("variant", cfg.variant)
    # 先判 str 再做集合成员检查:容器型 yaml 值(如 [fc])不可哈希,
    # 直接 in set 会抛 TypeError(unhashable)击穿 CLI 异常网(WR-01)
    if not isinstance(variant, str) or variant not in _ALLOWED_JOINT_VARIANTS:
        _fail("stereo.joint.variant", variant,
              f"只能是 {' 或 '.join(sorted(_ALLOWED_JOINT_VARIANTS))}"
              f"(结构化变体:约束越强越防内参无界漂移,03-02 实测全放开不收敛)")

    prior_weight = section.get("prior_weight", cfg.prior_weight)
    if (not isinstance(prior_weight, int | float) or isinstance(prior_weight, bool)
            or prior_weight <= 0):
        _fail("stereo.joint.prior_weight", prior_weight,
              "必须是 > 0 的数(Tikhonov 先验行的统一乘子;先验防内参漂移,"
              "权重越大拉回越强)")

    spread_ratio_max = section.get("spread_ratio_max", cfg.spread_ratio_max)
    if (not isinstance(spread_ratio_max, int | float) or isinstance(spread_ratio_max, bool)
            or spread_ratio_max <= 1.0):
        _fail("stereo.joint.spread_ratio_max", spread_ratio_max,
              "必须是 > 1.0 的数(精化后逐对残差 max/median 散布比上限;"
              "max/median ≤ 1 数学上不存在,接近 1 无判别力)")

    return StereoJointConfig(enabled=enabled, variant=variant,
                             prior_weight=float(prior_weight),
                             spread_ratio_max=float(spread_ratio_max))


def _parse_stereo(raw: dict) -> StereoConfig:
    """解析 stereo 段(模块3 全部算法参数/开关,CLI 零算法参数)。"""
    section = _check_is_dict(raw.get("stereo", {}), "stereo")

    flip_ratio = section.get("flip_ratio", StereoConfig.flip_ratio)
    # 上限 1.0:err_rev < ratio×err_fwd 的判定里 ratio 超过 1 即
    # "证据不足也翻"(Pitfall 11 的边界抖动),与"证据充分才翻"的设计相反
    if (not isinstance(flip_ratio, int | float) or isinstance(flip_ratio, bool)
            or not 0 < flip_ratio <= 1.0):
        _fail("stereo.flip_ratio", flip_ratio,
              "必须是 0 到 1.0 之间(含上端点)的数(消歧翻转判定阈值;超过 1.0 即证据不足也翻)")

    median_spread = section.get("median_spread_max_deg", StereoConfig.median_spread_max_deg)
    if (not isinstance(median_spread, int | float) or isinstance(median_spread, bool)
            or median_spread <= 0):
        _fail("stereo.median_spread_max_deg", median_spread,
              "必须是 > 0 的数(Rodrigues 中位离散度防线上限,度)")

    distance_rms_max = section.get("distance_rms_max_mm", StereoConfig.distance_rms_max_mm)
    if (not isinstance(distance_rms_max, int | float) or isinstance(distance_rms_max, bool)
            or distance_rms_max <= 0):
        _fail("stereo.distance_rms_max_mm", distance_rms_max,
              "必须是 > 0 的数(test 集对角距离残差 RMS 判 PASS/FAIL 的阈值,毫米)")

    band = section.get("baseline_band_mm", StereoConfig.baseline_band_mm)
    if (not isinstance(band, (list, tuple)) or len(band) != 2
            or any(not isinstance(x, int | float) or isinstance(x, bool) for x in band)):
        _fail("stereo.baseline_band_mm", band,
              "必须是一列恰好两个数的区间(如 [20, 2000],毫米)")
    if not 0 < float(band[0]) < float(band[1]):
        _fail("stereo.baseline_band_mm", band,
              "必须满足 0 < 下限 < 上限(反序或塌缩成一点都让 sanity 带失效)")

    exclude = section.get("exclude_pairs", StereoConfig.exclude_pairs)
    # 默认值是元组,yaml 解析出来是 list:两种序列形态都收,裸字符串才拒
    # (照 intrinsic.exclude_images 的体例;只收 list 会把缺段默认值 () 拒掉)
    if not isinstance(exclude, (list, tuple)) or not all(isinstance(x, str) for x in exclude):
        _fail("stereo.exclude_pairs", exclude,
              "必须是一列 group_id 文字(如 [20261023_154025_519359]),逐对手动逻辑剔除")

    return StereoConfig(refine=_parse_stereo_refine(section),
                        outlier=_parse_stereo_outlier(section),
                        flip_ratio=float(flip_ratio),
                        median_spread_max_deg=float(median_spread),
                        distance_rms_max_mm=float(distance_rms_max),
                        baseline_band_mm=(float(band[0]), float(band[1])),
                        exclude_pairs=tuple(exclude),
                        joint=_parse_stereo_joint(section))


def _parse_validation(raw: dict) -> ValidationConfig:
    """解析 validation 段(模块4 测量精度验证:路径/红线阈值/区域网格/对数下限)。"""
    cfg = ValidationConfig()
    section = _check_is_dict(raw.get("validation", {}), "validation")

    data_dir = section.get("data_dir", cfg.data_dir)
    if not isinstance(data_dir, str) or not data_dir:
        _fail("validation.data_dir", data_dir,
              "必须是验证图像对根目录的非空路径文字(相对当前工作目录,"
              "期望结构:<组目录>/cam1.jpg + cam2.jpg)")

    bias_max = section.get("distance_bias_max_mm", cfg.distance_bias_max_mm)
    if (not isinstance(bias_max, int | float) or isinstance(bias_max, bool)
            or bias_max <= 0):
        _fail("validation.distance_bias_max_mm", bias_max,
              "必须是 > 0 的数(|平均偏差| 红线,毫米;漂移判据,≤0 无判别力)")

    rms_max = section.get("distance_rms_max_mm", cfg.distance_rms_max_mm)
    if (not isinstance(rms_max, int | float) or isinstance(rms_max, bool)
            or rms_max <= 0):
        _fail("validation.distance_rms_max_mm", rms_max,
              "必须是 > 0 的数(冻结模式偏差 RMS 红线,毫米)")

    grid = section.get("region_grid", cfg.region_grid)
    if (not isinstance(grid, (list, tuple)) or len(grid) != 2
            or any(not isinstance(x, int) or isinstance(x, bool) for x in grid)):
        _fail("validation.region_grid", grid,
              "必须是一列恰好两个整数的 (行, 列)(如 [3, 3];[0, 0] = 关闭区域分析)")
    if grid[0] < 0 or grid[1] < 0:
        _fail("validation.region_grid", grid,
              "行/列桶数必须 >= 0(负数没有几何意义)")
    if (grid[0] == 0) != (grid[1] == 0):
        _fail("validation.region_grid", grid,
              "非法组合:要么行/列都 > 0(开区域分析),要么都 = 0(关闭);"
              "只关一个方向会让分桶网格退化成条带")

    min_pairs = section.get("min_pairs", cfg.min_pairs)
    if not isinstance(min_pairs, int) or isinstance(min_pairs, bool) or min_pairs < 1:
        _fail("validation.min_pairs", min_pairs,
              "必须是 >= 1 的整数(有效验证对数下限;0 对等于没验证)")

    return ValidationConfig(data_dir=data_dir, distance_bias_max_mm=float(bias_max),
                            distance_rms_max_mm=float(rms_max),
                            region_grid=(int(grid[0]), int(grid[1])),
                            min_pairs=min_pairs)


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


def _check_validation_data_safety(cfg: StereoCalibConfig) -> None:
    """跨字段数据安全校验(validation 侧,照 _check_output_data_disjoint 体例)。

    validation.data_dir 与 data.train / data.test / output_root 相等或互相
    包含的配置当场拒绝:验证目录混进 train/test 会让「漂移体检」测到训练
    考卷自己(结论失真);落进 output_root 之内则运行前清理产物目录会把
    验证数据一起删掉。resolve() 用默认非 strict 模式——目录不存在也能解析
    出规范路径,加载层就要拦得住。
    """
    val = Path(cfg.validation.data_dir).resolve()
    for name, other in (("data.train", cfg.data.train), ("data.test", cfg.data.test),
                        ("output_root", cfg.output_root)):
        other_dir = Path(other).resolve()
        if val == other_dir:
            relation = "两目录指向同一位置"
        elif other_dir in val.parents:
            relation = f"{name} 是验证目录的父目录(验证数据位于 {name} 之内)"
        elif val in other_dir.parents:
            relation = f"验证目录是 {name} 的父目录({name} 位于验证目录之内)"
        else:
            continue
        raise ValueError(
            f"配置字段 validation.data_dir/{name} 不合法:验证数据目录与 "
            f"{name} 不得相等或互相包含(验证数据混入数据/产物目录会让体检结论失真,"
            f"或被运行前清理一并删除);本次重叠形态:{relation};"
            f"实际:validation.data_dir={val},{name}={other_dir}。请检查对应的 yaml 文件。"
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
    stereo = _parse_stereo(raw)
    validation = _parse_validation(raw)

    cfg = StereoCalibConfig(board=board, data=data, output_root=output_root,
                            cv2_threads=cv2_threads, preprocess=preprocess,
                            intrinsic=intrinsic, stereo=stereo, validation=validation)
    _check_output_data_disjoint(cfg)
    _check_validation_data_safety(cfg)
    return cfg
