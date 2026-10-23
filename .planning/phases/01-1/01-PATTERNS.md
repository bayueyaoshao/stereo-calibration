# Phase 1: 工程骨架 + 模块1 数据预处理 - Pattern Map

**Mapped:** 2026-10-23
**Files analyzed:** 17(全部为新建;greenfield 仓库无源码)
**Analogs found:** 17 / 17(全部来自只读参考工程 `/home/lx/Works/src/lx/github/intrinsic-calibration/`,下文简称"参考")

**总原则(来自 RESEARCH §Don't Hand-Roll):** 除胶水编排(扫描配对、并行调度、落盘汇总)外,算法路径是"移植 + 不变量测试锁定",不是重写。三条硬不变量必须随代码一起移植并在测试中锁定:
1. SB 输出严禁二次亚像素精化(`cornerSubPix` 全文件只许出现在 classic 分支,源码防线测试锁定)
2. 亚像素窗口逐帧从实测方格边长推导,断言 `2w+1 < sq_px`
3. 两条检测路线输出必过 `canonicalize`(SB/classic 角点序不同已在本数据实测:原始差 355.9px 中位)

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog(参考工程) | Match Quality |
|-------------------|------|-----------|--------------------------|---------------|
| `pyproject.toml` | config(构建清单) | — | `pyproject.toml`(61 行) | exact |
| `config.yaml` | config | — | `config/board.yaml` + `config/calibration.yaml` L1-62 | exact(两文件合一,D-10) |
| `src/stereo_calib/__init__.py` 及各子包 `__init__.py` | package | — | `src/calib/corners/__init__.py` | exact |
| `src/stereo_calib/cli/main.py` | CLI 入口 | request-response | `src/calib/cli/main.py` L132-162, L253-259, L753-770, L2592-2598 | role-match(仅骨架,不要九个子命令) |
| `src/stereo_calib/io/config.py` | 配置加载/校验 | transform | `src/calib/io/config.py` L1-143, L478-641 | exact(移植体例,合并两文件加载) |
| `src/stereo_calib/io/dataset.py` | 数据扫描 | file-I/O | `src/calib/io/dataset.py` L25-154 | role-match(数据布局不同,见 Deviations) |
| `src/stereo_calib/corners/detect.py` | 检测算法 | transform(纯函数) | `src/calib/corners/detect.py`(239 行) | exact(近乎原样移植) |
| `src/stereo_calib/corners/window.py` | 窗口推导 | transform(纯函数) | `src/calib/corners/window.py`(113 行) | exact(近乎原样移植) |
| `src/stereo_calib/corners/canonical.py` | 排序规范化 | transform(纯函数) | `src/calib/corners/canonical.py` L28-51 | exact(只移植 canonicalize) |
| `src/stereo_calib/preprocess/parallel.py` | 并行调度 | batch(任务扇出) | `src/calib/evaluation/parallel.py`(109 行) | exact(近乎原样移植) |
| `src/stereo_calib/preprocess/pipeline.py` | 模块编排 | batch | `src/calib/cli/main.py` `_run_corners` L440-750 | role-match(参考串行,本工程改并行) |
| `src/stereo_calib/preprocess/artifacts.py` | 产物落盘 | file-I/O | `src/calib/corners/artifacts.py`(98 行) | role-match(npz 移植 + CSV 新增) |
| `src/stereo_calib/visualization/overlay.py` | 可视化渲染 | file-I/O | `src/calib/visualization/overlay.py` `render_marked` L29-90 + L184-198 | exact(核心函数) |
| `tests/conftest.py` | test 夹具 | — | `tests/conftest.py` L22-37, L73-172 | exact |
| `tests/test_corners.py` | test | — | `tests/test_corners.py`(246 行) | exact |
| `tests/test_io_dataset.py` | test | — | `tests/test_io_dataset.py`(120 行) | role-match |
| `tests/test_preprocess_pipeline.py`(或 test_cli_main.py) | test | — | `tests/test_cli_main.py`(86 行) | role-match |

---

## Pattern Assignments

### `pyproject.toml`(构建清单)

**Analog:** 参考工程 `pyproject.toml`(全文 61 行,直接照抄改包名)

```toml
# 参考工程 L1-26 关键节选 —— 本工程改动点:name/description、去掉 matplotlib+jinja2
[project]
name = "stereo-calibration"          # 包目录名;import 名为 stereo_calib
requires-python = ">=3.12"
dependencies = [
    "opencv-python>=4.14,<5.0",      # 注释一并抄:5.0 calib3d 行为易变
    "numpy==2.5.*",                  # 与 scipy 成对锁
    "scipy==1.18.*",
    "typer>=0.15",
    "pyyaml>=6.0",
    "joblib>=1.4",                   # 注释抄:return_as="generator" 自 1.3 引入
]

[dependency-groups]                   # 参考 L28-36
dev = ["pytest>=8", "ruff>=0.8"]      # 不需要 numdifftools

[project.scripts]                     # 参考 L38-40
stereo-calib = "stereo_calib.cli.main:main"

[build-system]                        # 参考 L42-47
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/stereo_calib"]       # 连字符包名不能 import,src 下目录用下划线

[tool.ruff]
line-length = 100                     # 参考 L60-61
```

---

### `config.yaml`(唯一参数来源,D-10)

**Analog:** 参考工程 `config/board.yaml`(全文 7 行)+ `config/calibration.yaml` L1-62

**合并方式:** board 段作为全局段并入单一 config.yaml(参考是两个文件,本工程 D-10 锁定单一文件)。命名风格照抄:snake_case 键、`# 中文注释解释为什么`、段名即模块名。

```yaml
# 参考 board.yaml 全文(并作本工程 board: 段)
board:
  board_type: "chessboard"
  cols: 11                             # 内部角点列数
  rows: 8                              # 11×8 = 88 个内部角点
  square_size_mm: 25.0                 # 方格物理边长
  unit: "mm"

# 参考 calibration.yaml L9-17 风格(数据位置 + 全局)
data:
  train: "data/train"
  test: "data/test"
output_root: "output"                  # D-11:固定根目录,直接覆盖
cv2_threads: 0                         # 0 = 关 OpenCV 内部多线程,并行交上层控制

# 参考 calibration.yaml L35-62 的 corners 段 → 本工程挪到 preprocess.corners 下
preprocess:
  max_workers: 12                      # 1 = 串行调试(体例照参考 injection_max_workers L90-93)
  corners:
    classic_flags: [adaptive_thresh, normalize_image]   # 名字在配置层校验,代码内翻译成 cv2 常量
    sb_flags: 0
    sb_fallback_classic: true          # SB 未检出时 classic 兜底
    subpix: {window_granularity: frame, min_win: 3, max_iter: 30, epsilon: 0.01}
  copy_fail_originals: true            # D-02 失败帧原图拷贝开关
  visualization: {jpeg_quality: 90}
```

**不移植的参考键:** `burst_check`(本数据无同姿态连拍)、`force_win`(参考工程实验专用)、`exclude_images`(D-03 明确不做)。

---

### `src/stereo_calib/cli/main.py`(CLI 入口,INFRA-01)

**Analog:** 参考工程 `src/calib/cli/main.py`——只抄骨架四件,不抄九个子命令的实现体。

**App 定义 + 运行时 callback**(参考 L132-157):
```python
app = typer.Typer(
    name="stereo-calib",
    help="双目相机标定生产工具:preprocess 数据预处理 / intrinsic 内参标定 / stereo 双目外参标定。",
    no_args_is_help=True,
)

CONFIG_PATH = Path("config.yaml")       # D-10 单一配置文件
OUTPUT_ROOT 从配置读,不写死

@app.callback()
def _setup_runtime() -> None:
    """每个命令执行前的统一准备:按配置限制 OpenCV 线程数。"""
    try:
        cfg = load_config(CONFIG_PATH)
        cv2.setNumThreads(cfg.cv2_threads)
    except (ValueError, OSError):
        pass  # --help / 配置未就位时不拦着用户

def _not_implemented(name: str) -> None:   # 参考 L160-162:预留子命令占位
    typer.echo(f"命令「{name}」由后续阶段实现。")
```

**本工程差异(见 Deviations #10):** 参考用固定常量 `CONFIG_PATH`;本工程把配置路径做成 callback 上的可选 `--config` typer 选项(默认 `Path("config.yaml")`,赋值模块级 `_config_path`,callback 与 preprocess 命令统一用它)——满足 D-10 字面「命令行只传配置文件路径」,仍零算法参数。

**命令入口报错收口**(参考 L753-770,每个子命令同构):
```python
@app.command(help="模块1 数据预处理:…")
def preprocess() -> None:
    """坏输入收口为中文一句 + 退出码 1,不吐堆栈。"""
    try:
        run_preprocess()
    except (ValueError, OSError) as e:
        typer.echo(f"数据预处理失败:{e}")
        typer.echo("  请检查 config.yaml 配置与 data/ 数据,修正后重跑 stereo-calib preprocess。")
        raise typer.Exit(code=1) from e

@app.command(help="模块2 单相机内参标定(Phase 2 实现)")
def intrinsic() -> None:
    _not_implemented("intrinsic")
```

**入口函数**(参考 L2592-2598):
```python
def main() -> None:
    """[project.scripts] 的入口函数。"""
    app()

if __name__ == "__main__":
    main()
```

---

### `src/stereo_calib/io/config.py`(配置层,dataclass + 中文校验,PREP-01/INFRA-02)

**Analog:** 参考工程 `src/calib/io/config.py`

**白名单常量 + frozen dataclass**(参考 L22-26, L57-77, L99-143):
```python
_ALLOWED_CLASSIC_FLAGS = {"adaptive_thresh", "normalize_image", "fast_check"}
_ALLOWED_GRANULARITY = {"frame", "corner"}

@dataclass(frozen=True)
class BoardSpec:
    cols: int = 11
    rows: int = 8
    square_size_mm: float = 25.0
    unit: str = "mm"

@dataclass(frozen=True)
class SubpixConfig:
    window_granularity: str = "frame"
    min_win: int = 3
    max_iter: int = 30
    epsilon: float = 0.01

@dataclass(frozen=True)
class CornersConfig:
    classic_flags: tuple[str, ...] = ("adaptive_thresh", "normalize_image")
    sb_flags: int = 0
    sb_fallback_classic: bool = True
    subpix: SubpixConfig = field(default_factory=SubpixConfig)

@dataclass(frozen=True)
class PreprocessConfig:        # 模块段:并行/开关/可视化
    max_workers: int = 12
    corners: CornersConfig = field(default_factory=CornersConfig)
    copy_fail_originals: bool = True
    jpeg_quality: int = 90

@dataclass(frozen=True)
class StereoCalibConfig:       # 顶层:全局段 + 每模块段(D-10 分段)
    board: BoardSpec = field(default_factory=BoardSpec)
    data: DataDirs = field(default_factory=DataDirs)
    output_root: str = "output"
    cv2_threads: int = 0
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
```

**统一中文报错 + safe_load**(参考 L496-529,逐字抄体例):
```python
def _fail(field_name: str, actual, requirement: str) -> None:
    raise ValueError(
        f"配置字段 {field_name} 不合法:{requirement};实际读到的值是 {actual!r}。"
        f"请检查对应的 yaml 文件。")

def _read_yaml(path: Path | str) -> object:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"配置文件不存在:{path}。请确认路径(相对当前工作目录)是否正确。")
    with open(path, encoding="utf-8") as f:
        try:
            return yaml.safe_load(f)      # 绝不 yaml.load(执行任意代码)
        except yaml.YAMLError as e:
            raise ValueError(f"配置文件 {path} 的 yaml 语法有误,请检查缩进与冒号:{e}") from e
```

**逐字段校验体例**(参考 `_parse_subpix` L586-603 / `_parse_corners` L620-641):每字段 isinstance 检查 + 白名单 + `_fail`,注意 bool 先于 int 判断(参考 L633-637 注释:True 是 int,1 不是 bool,yaml 手写 1/0 必须拦)。加载入口 `load_config(path) -> StereoCalibConfig` 把参考的 `load_board` + `load_calibration` 合并为一次 `_read_yaml` 后分段解析(单一文件,D-10)。

**本工程新增(参考没有,checker 修订定案):** 分段解析完成后做跨字段数据安全校验——data.train、data.test、output_root 各取 `Path(...).resolve()` 后双向互斥(任一 data 目录与 output_root 相等、位于其内、或 output_root 位于 data 目录之内 → 中文 ValueError,防运行前清理误删 data/ 与产物写进 data/)。参数化测试锁定。

---

### `src/stereo_calib/io/dataset.py`(数据扫描,PREP 配套)

**Analog:** 参考工程 `src/calib/io/dataset.py`——**韧性规则与代码体例照抄,数据布局逻辑重写**(见 Deviations #2)。

**要抄的部分**——frozen dataclass 记录 + 警告清单模式(参考 L33-51):
```python
@dataclass(frozen=True)
class ImageRecord:
    image_id: str        # 本工程:叶子目录名 + "_cam{n}"(stem 全库重复,绝不能用!)
    path: Path
    split: str           # "train" / "test"
    cam: int             # 1 或 2(本工程新增字段)
    group_id: str        # 叶子目录名 <ts>_<us>(公共视场配对键,本工程替代参考的 pose_id)

@dataclass(frozen=True)
class ScanResult:
    records: tuple[ImageRecord, ...]
    warnings: tuple[str, ...]   # 警告全部中文、写进 meta 留痕(参考体例)
```

**安全 image_id 正则**(参考 L27-30,抄):
```python
_SAFE_IMAGE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
# image_id 会进入输出文件名,限制字符防止特殊字符把产物写出 output/ 之外
```

**imread_checked**(参考 L145-154,逐字抄——Pitfall: cv2.imread 读坏文件返回 None 不报错):
```python
def imread_checked(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(
            f"照片读取失败:{path}。文件可能损坏、路径不对或没有读权限,请人工检查这张图。")
    return img
```

**容错扫描体例**(参考 L67-117):目录不存在 → 中文警告按 0 张处理;空目录/不合模式 → 警告 + 继续;两划分全空才 raise ValueError(参考 L126-130)。本工程扫描实现:`Path(split_dir).glob("*/*/cam*.jpg")` 两层通配,`cam` 号从文件名解析,group_id = 叶子目录名;单边缺文件记警告。

---

### `src/stereo_calib/corners/detect.py`(双检测器,PREP-02)

**Analog:** 参考工程 `src/calib/corners/detect.py`(239 行)——**近乎原样移植**,只做三处适配:import 路径 `calib.` → `stereo_calib.`;`CornersConfig` 引用本工程配置类;可去掉 `force_win` 实验分支(L164-175)。

**核心结构**(`_CLASSIC_FLAG_VALUES` L38-50、`_subpix_refine` L53-64、`grid_defects` L75-100、`_recover_grid_defects` L110-140、`detect_classic` L143-212、`detect_sb` L215-239 全部保留)。

**SB 主路线——严禁二次精化**(L215-239):
```python
def detect_sb(gray: np.ndarray, cols: int, rows: int, cfg: CornersConfig) -> dict | None:
    found, raw = cv2.findChessboardCornersSB(gray, (cols, rows), cfg.sb_flags)
    if not found:
        return None                      # 未检出返回 None,绝不返回半成品
    corners = raw.reshape(-1, 2).astype(np.float64)
    sq_px = estimate_sq_px(corners, cols, rows)
    win_size = derive_window_frame(sq_px, cfg.subpix.min_win)  # 仅记录口径,不参与精化
    return {
        "uv": canonicalize(corners, cols, rows),   # 不变量 3:输出必过规范化
        "sq_px": sq_px, "win_size": win_size,
        "detector": "sb", "recovery": None,
    }
```

**classic 兜底——窗口推导 + 缺陷闸门 + 阶梯重提**(L150-204 精髓):
```python
found, raw = cv2.findChessboardCorners(gray, (cols, rows), _classic_flag_value(cfg.classic_flags))
if not found:
    return None
corners = raw.reshape(-1, 2).astype(np.float64)
coarse = corners.copy()   # 粗定位网格留底:精化若带病,恢复必须从这里重新出发
sq_px = estimate_sq_px(corners, cols, rows)
wins = np.full(corners.shape[0], derive_window_frame(sq_px, cfg.subpix.min_win), dtype=int)
win_size = int(wins.max())
# ... 分组精化(L183-186,cornerSubPix 只经 _subpix_refine 这一处) ...
defects = grid_defects(corners, cols, rows)
if defects["total"] > 0:                # 重复点(<0.25×中位方格)或锯齿(二阶差分>0.3×中位)
    fixed, used_win, defects_after, ladder = _recover_grid_defects(gray, coarse, cols, rows)
    recovery = {"defects_before": defects["total"], "defects_after": defects_after["total"],
                "initial_win": win_size, "used_win": int(used_win), "ladder": ladder}
```

恢复重提用更严收敛判据(参考 L106-107 模块常量):`_RECOVERY_MAX_ITER = 100`、`_RECOVERY_EPSILON = 1e-6`。

---

### `src/stereo_calib/corners/window.py`(窗口推导断言防线)

**Analog:** 参考工程 `src/calib/corners/window.py`(113 行)——**原样移植** `estimate_sq_px`(L25-39)、`derive_window_frame`(L42-59)、`derive_window_per_corner`(L91-113,含 `_corner_local_spacing`)。

**头号陷阱断言**(L52-59,断言文案逐字抄):
```python
w = max(int(min_win), int(sq_px // 2) - 1)
assert 2 * w + 1 < sq_px, (
    f"亚像素窗口断言失败:搜索区 {2 * w + 1} 像素未严格小于方格像素边长 {sq_px:.2f}。"
    f"窗口一旦跨过相邻角点,定位会被系统性拉偏,且重投影误差完全看不出来"
    f"(本项目头号陷阱)。请检查角点检测质量,或确认方格像素边长是否过小、"
    f"corners.subpix.min_win 下限是否需要人工复核。")
```

`estimate_sq_px` 关键:行/列相邻间距各取中位数后**保守取小**(`min(np.median(d_h), np.median(d_v))`)——透视下远处方格更小。

---

### `src/stereo_calib/corners/canonical.py`(排序规范化)

**Analog:** 参考工程 `src/calib/corners/canonical.py`——只移植 `canonicalize`(L28-51),**不移植** `burst_consistency`(本数据无同姿态连拍)。

```python
def canonicalize(corners: np.ndarray, cols: int, rows: int) -> np.ndarray:
    g = np.asarray(corners, dtype=np.float64).reshape(rows, cols, 2)
    e_row_x = float((g[:, 1:, :] - g[:, :-1, :])[..., 0].mean())
    if e_row_x < 0:          # 行方向朝左 → 每行内反转
        g = g[:, ::-1, :]
    e_col_y = float((g[1:, :, :] - g[:-1, :, :])[..., 1].mean())
    if e_col_y < 0:          # 列方向朝上 → 反转行顺序
        g = g[::-1, :, :]
    return g.reshape(-1, 2).copy()
```

实证依据(RESEARCH Pattern 4):本数据 SB vs classic 原始同编号角点距离中位 355.9px,规范化后 0.63px——不过这层,兜底帧 npz 就是废数据。

---

### `src/stereo_calib/preprocess/parallel.py`(joblib loky 并行,PREP-02)

**Analog:** 参考工程 `src/calib/evaluation/parallel.py`(109 行)——**原样移植全文件**:`TaskFailed` 信封类(L31-45)、`_guarded_worker`(L48-58)、`run_parallel`(L61-109)。

**核心模式**(L85-109):
```python
if max_workers <= 1:
    # 串行回退:不开子进程,与并行同序同果(调试用)
    for done, task in enumerate(task_list, start=1):
        yield _guarded_worker(worker, task)
        if progress:
            progress({"done": done, "total": total})
    return

threads_each = max(1, (os.cpu_count() or 1) // max(1, max_workers))
with parallel_config(backend="loky", inner_max_num_threads=threads_each):
    results = Parallel(n_jobs=max_workers, return_as="generator")(
        delayed(_guarded_worker)(worker, task) for task in task_list)
    for done, result in enumerate(results, start=1):
        yield result
        if progress:
            progress({"done": done, "total": total})
```

**为什么必须异常信封**(参考模块 docstring L15-19 + RESEARCH Pitfall 6):joblib 在任一任务抛异常时关闭整条结果流,2026-10-19 真实事故 700+ 任务全没。worker 内任何异常(含 imread_checked 的 ValueError)包 `TaskFailed` 交回,汇总层 `isinstance` 拆信封 → 该帧记 fail,不是全批崩溃。

**本工程适配:** worker = 检测单图(任务小元组 = 路径/cam/group_id/split/board 规格/corners 配置),worker 自己 imread(像素数据不跨进程),返回小 dict(uv 88×2 可序列化)。示例见 RESEARCH §Code Examples `detect_one`。

---

### `src/stereo_calib/preprocess/pipeline.py`(模块编排)

**Analog:** 参考工程 `src/calib/cli/main.py` `_run_corners`(L440-750)——编排**流程**照抄,**实现**改为并行 + 本工程产物。参考的流程骨架:读配置 → 读上游清单 → 预分配 NaN 数组 → 逐帧检测填槽 → 按 split 切行落盘 npz → 质量摘要 json → 全量标记副本 → 失败帧独立证据 → 中文汇总 typer.echo。

**预分配带掩码稠密数组**(参考 L474-489,体例照抄):
```python
n = len(images)
arrays = {
    (split, cam): {
        "uv": np.full((n_split, board.cols * board.rows, 2), np.nan),  # 未检出整行 NaN
        "sq_px": np.full(n_split, np.nan),
        "win_size": np.full(n_split, np.nan),
        "detected": np.zeros(n_split, dtype=bool),
    }
    ...
}
sources = np.full(n, "", dtype="<U16")   # 定长 unicode,不需要 pickle
```

**兜底填槽逻辑**(参考 L515-531,改为 SB→classic 两级):
```python
sb_result = detect_sb(gray, cols, rows, cfg.corners)
if sb_result is not None:
    ...; sources[i] = "sb"
elif classic_result is not None and cfg.corners.sb_fallback_classic:
    # classic 的角点连同 sq_px/win_size 一起填进槽位,真实来源单独记 source
    ...; sources[i] = "classic"
```

**空划分保护**(参考 L594-598 注释,坑已踩过):
```python
# dtype=int 必须显式写:空列表会被 numpy 推断成 float64,拿它做花式索引当场 IndexError
rows = np.array([i for i, rec in enumerate(images) if rec["split"] == split], dtype=int)
```

**中文汇总输出**(参考 L690-750 体例):成功率表、方格像素边长统计、fail 数、产物路径逐行 typer.echo。

**与参考的差异:** ① 检测循环外包 `run_parallel`(参考是串行 for);② 输出无 `run_<时间戳>` 目录层(D-11 固定覆盖,见 Deviations #5);③ 运行开始先清理产物子目录(visualization/、fail_originals/、CSV、npz——防残留,Pitfall 7;清理安全性由 io/config.py 的 output_root/data 互斥校验保证)。

---

### `src/stereo_calib/preprocess/artifacts.py`(npz/CSV 落盘,PREP-03/04/05)

**Analog:** 参考工程 `src/calib/corners/artifacts.py`(98 行)——npz 契约与 allow_pickle 防线照抄;CSV 写入为本工程新增(见 No Analog)。

**npz 保存模式**(参考 L59-88):
```python
out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
arrays = {
    "image_ids": np.asarray(image_ids),        # 本工程扩展:paths、objp 数组(D-04)
    "uv": np.asarray(uv, dtype=np.float64),    # (N, 88, 2),未检出整行 NaN
    "sq_px": np.asarray(sq_px, dtype=np.float64),
    "win_size": np.asarray(win_size, dtype=np.float64),
    "detected": np.asarray(detected, dtype=bool),
    "source": np.asarray(source),              # 定长 unicode,存取不需要 pickle
}
np.savez(npz_path, **arrays)
meta = {...}                                   # 配置快照/规范化规则版本/detected 计数
with open(out_dir / f"{name}.meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
```

**规范版本字符串旁车**(参考 L28-29,抄概念):
```python
CANONICAL_RULE_VERSION = "canonical-v1:行方向朝画面右、列方向朝画面下"
```

**安全读取**(参考 L91-98,逐字抄):
```python
def load_corners(path: Path | str) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as data:   # pickle 可执行任意代码,关死
        return {key: data[key] for key in data.files}
```

**本工程 schema(D-04 + RESEARCH Pattern 7 推荐,planner 定案记为契约):** 文件名 `cam{n}_{split}.npz`;数组 `image_ids (N,) <U` = 叶子目录名+`_cam{n}`、`paths (N,) <U`、`uv (N,88,2) float64`、`objp (88,3) float64`、`source (N,) <U16` ∈ {"sb","classic",""}、`detected (N,) bool`、`sq_px (N,)`、`win_size (N,)`。

**objp 静态生成**(参考 `src/calib/solve/baseline.py` L35-47——`build_object_points`,近逐字):
```python
def build_object_points(board: BoardSpec) -> np.ndarray:
    """构造物点网格:棋盘内角点在棋盘平面上的坐标,(cols*rows, 3) float64。
    Z 一律为 0,X/Y 按方格边长排开,单位毫米。"""
    xs = np.arange(board.cols, dtype=np.float64) * board.square_size_mm
    ys = np.arange(board.rows, dtype=np.float64) * board.square_size_mm
    objp = np.zeros((board.rows * board.cols, 3), dtype=np.float64)
    objp[:, 0] = np.tile(xs, board.rows)
    objp[:, 1] = np.repeat(ys, board.cols)
    return objp
# 契约:objp[row*cols+col] = (col*25.0, row*25.0, 0),与 canonicalize 后 uv 同序
```

---

### `src/stereo_calib/visualization/overlay.py`(角点标注图,PREP-06/D-07/D-08)

**Analog:** 参考工程 `src/calib/visualization/overlay.py` `render_marked`(L29-90)——核心渲染函数照抄,本工程**新增**烧录路径/路线文字(D-08)。

**标准角点叠加**(参考 L54-58,Don't Hand-Roll 定案):
```python
img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)   # 原图只读进内存,标记画在副本
if img is None:
    raise ValueError(f"照片读取失败:{image_path},无法生成标记副本,请人工检查这张图。")
corners = np.asarray(uv_or_none, dtype=np.float32).reshape(-1, 1, 2)
cv2.drawChessboardCorners(img, (int(cols), int(rows)), corners, True)  # 圆点+顺序连线,一行出图
```

**按画面尺寸比例推导字号/线宽,不写死像素**(参考 L61-70 的 MISS 分支体例,用于 D-08 烧录文字):
```python
height, width = img.shape[:2]
thickness = max(4, min(height, width) // 80)
font_scale = max(1.0, min(height, width) / 480)
cv2.putText(img, f"{path} [{route}]", org=(thickness * 2, thickness * 2 + int(font_scale * 22)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale,
            color=(0, 0, 255), thickness=max(2, thickness // 2), lineType=cv2.LINE_AA)
# ASCII only:cv2.putText 不支持中文字形(Pitfall 9);路径/cam/sb/classic 恰为 ASCII
```

**写盘 + 质量参数**(参考 L85-89):
```python
out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
if not cv2.imwrite(str(out_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]):
    raise ValueError(f"标记副本写盘失败:{out_path}")
```

**失败帧原图拷贝(D-02)**——模式来自参考 `render_sb_failed_review`(L184-188):
```python
originals_dir.mkdir(parents=True, exist_ok=True)
shutil.copy2(src_path, originals_dir / src_path.name)   # 逐字节复制,源文件绝不改写
# 本工程:文件名加 fail_ 前缀 → fail_{image_id}.jpg;受 preprocess.copy_fail_originals 开关控制
```

---

### `tests/conftest.py`(合成棋盘夹具)

**Analog:** 参考工程 `tests/conftest.py`——`make_tiny_jpg`(L22-37)与 `make_chessboard_image`(L73-172)**原样移植**(连 supersample 抗锯齿与主点对齐注释 L126-139 一起抄——它们防止半像素画格偏差污染"误差<0.5px"断言)。`make_true_views`(L175-204)是 Phase 2 夹具,可不移植。

夹具返回 `(图像 uint8, 真值角点 (88,2) float64, 方格像素边长)`——检测对错有"标准答案"可断言,不靠肉眼。

---

### `tests/test_corners.py`(不变量锁定测试)

**Analog:** 参考工程 `tests/test_corners.py`(246 行)——结构照抄,按本工程范围裁剪。

**必抄的源码防线测试**(参考 L156-163,锁不变量 1):
```python
def test_detect_sb_source_guard_single_refinement_call():
    """源码防线:二次亚像素精化在 detect.py 里恰好只出现 1 次(只在经典分支)。"""
    source = pathlib.Path("src/stereo_calib/corners/detect.py").read_text(encoding="utf-8")
    assert source.count("cv2.cornerSubPix(") == 1
```

**必抄的不等式锁定**(参考 L49-69,锁不变量 2):
```python
@pytest.mark.parametrize("sq_px", [15.0, 20.0, 25.0, 40.0])
def test_derive_window_frame_inequality(sq_px):
    w = derive_window_frame(sq_px)
    assert w == int(sq_px // 2) - 1
    assert 2 * w + 1 < sq_px

def test_derive_window_frame_min_win_floor():
    assert derive_window_frame(7.5, min_win=3) == 3
    with pytest.raises(AssertionError):
        derive_window_frame(5.0, min_win=3)   # 方格过小必须当场报错
```

**必抄的四变体规范化锁定**(参考 L95-114,锁不变量 3):
```python
def _four_variants(corners, cols, rows):    # 恒等 / 水平翻转 / 垂直翻转 / 旋转180°
    g = np.asarray(corners, dtype=np.float64).reshape(rows, cols, 2)
    return {"恒等": g.copy(), "水平翻转": g[:, ::-1, :], "垂直翻转": g[::-1, :, :], "旋转180": g[::-1, ::-1, :]}

def test_canonicalize_maps_all_variants_to_same_sequence(make_chessboard_image):
    _img, corners, _ = make_chessboard_image()
    base = canonicalize(corners, cols=11, rows=8)
    for name, variant in _four_variants(corners, 11, 8).items():
        out = canonicalize(variant.reshape(-1, 2), cols=11, rows=8)
        assert np.allclose(out, base), f"变体「{name}」没有被规范化到唯一形态"
```

其余照抄:双检测器精度 <0.5px(L123-136)、纯噪声返回 None(L148-153)、干净网格零缺陷(L168-175)、缺陷注入被抓(L178-188)。`integration`/真实事故帧回归(参考 L191-213)本工程无对应数据,不移植。

---

### `tests/test_io_dataset.py`(扫描韧性测试)

**Analog:** 参考工程 `tests/test_io_dataset.py`(120 行)——测试**维度**照抄(空目录警告/坏文件名/缺目录中文报错/分辨率一致性),用 `make_tiny_jpg` 造临时数据树。本工程新增维度:两层嵌套扫描、cam 号解析、单边缺文件(仅 cam1 无 cam2)告警不入对、image_id 唯一性(叶子目录名去重)。

坏文件读图测试照抄(参考 L109-114):
```python
def test_imread_checked_raises_on_corrupt_file(make_tiny_jpg):
    bad = make_tiny_jpg("data/train/<ts>/<ts>_<us>/cam1.jpg", 8, 8)
    bad.write_bytes(b"not a jpeg at all")
    with pytest.raises(ValueError, match="cam1.jpg"):
        imread_checked(bad)
```

---

### `tests/test_preprocess_pipeline.py`(端到端行为测试)

**Analog:** 参考工程 `tests/test_cli_main.py`(86 行)——**进程内直接调命令函数 + monkeypatch.chdir 临时目录**的模式照抄(L19-66):
```python
def test_preprocess_...(tmp_path, monkeypatch, make_chessboard_image):
    # 1. 造合成双目数据树:tmp_path/data/train/<ts>/<ts>_<us>/cam{1,2}.jpg
    # 2. 写最小 config.yaml
    # 3. monkeypatch.chdir(tmp_path);preprocess()   # 产物落 tmp,真实 output/ 不受影响
    # 4. 断言:npz 齐全、CSV 行数、detected 计数、可视化图数量 == 检出数
```

**本工程必加的防线测试(INFRA-03,data 只读):** 跑完 preprocess 后断言 `data/` 树 mtime/hash 不变(参考没有此测试,RESEARCH Pitfall 4 要求新增);可视化文件数 == npz detected 计数(防 Pitfall 7 残留);公共对数 ≤ min(cam1 检出, cam2 检出)(防 Pitfall 8)。

---

## Shared Patterns(横切,全部相关文件适用)

### 1. 中文报错收口(所有 io/算法/CLI 文件)
**Source:** 参考 `io/config.py` L496-501 `_fail` + `cli/main.py` L764-770 命令边界
库层抛带字段名/文件名/实际值的中文 `ValueError`;CLI 入口 `except (ValueError, OSError)` 收口为中文一句 + `typer.Exit(code=1)`,不吐堆栈。

### 2. data/ 只读 + 输出路径派生(INFRA-03)
**Source:** 参考 `overlay.py` L49-52(读图只读进内存)、L186-188(shutil.copy2 副本)、`dataset.py` imread_checked
全部写路径从配置 `output_root` 派生;fail 标记只体现在输出文件名 `fail_` 前缀与清单,绝不触碰 data/。测试锁定 mtime 不变。本工程加固:`io/config.py` 加载层拒绝 output_root 与 data 目录相等/互相包含的配置(参考没有,checker 修订定案)。

### 3. 安全防线(artifacts + dataset)
**Source:** 参考 `artifacts.py` L97(`np.load(..., allow_pickle=False)`)、`dataset.py` L27-30(`_SAFE_IMAGE_ID` 正则)
字符串一律定长 unicode 数组存 npz;image_id 过安全字符正则才可进输出文件名。

### 4. 不写死像素/张数(算法与可视化)
**Source:** 参考 `overlay.py` L61-70(字号线宽按画面尺寸比例)、`window.py` 全文(窗口按实测推导)、`dataset.py` L3("绝不写死张数")
阈值以倍数/比例/分位数表达;cols/rows/square_size 全部从配置 BoardSpec 传入。

### 5. 依赖安装(环境)
**Source:** RESEARCH §Pitfall 5——系统 pip 指向清华镜像(numpy 滞后),全部依赖操作走 uv(默认 pypi.org),绝不向系统 python 装任何东西。Python 用 uv 管理的 3.12.13(系统 3.10 不可用)。

---

## Deviations from Analog(参考 → 本工程,planner 必须写进任务)

| # | 参考工程做法 | 本工程做法 | 依据 |
|---|-------------|-----------|------|
| 1 | 两个配置文件 board.yaml + calibration.yaml | 单一 config.yaml,board 为全局段 | D-10 |
| 2 | train=每姿态一目录多张连拍、test 平铺 | `data/{split}/<ts>/<ts>_<us>/cam{1,2}.jpg` 两层嵌套;group_id=叶子目录名 | 数据契约(实测) |
| 3 | image_id = 文件名 stem(单相机唯一) | image_id = 叶子目录名 + "_cam{n}"(stem 全库只有 cam1/cam2 两个,必碰撞) | RESEARCH Pitfall 3 |
| 4 | npz 按划分×检测器 4 份(corners_train_sb.npz 等) | npz 按相机×数据集 4 份(cam1_train.npz 等),增加 paths/objp 数组;source ∈ {"sb","classic",""} | D-04 |
| 5 | 输出 `output/<模块>/run_<时间戳>/` 版本化 | 固定 `output_root` 下 train// test/ 分目录直接覆盖;运行开始清理产物子目录防残留 | D-11 + Pitfall 7 |
| 6 | manifest.json 为清单载体 | 清单用 CSV(usable/common_fov_pairs/fail_list),Excel 可开 | D-06 |
| 7 | 标记副本仅画角点(MISS 红框给未检出) | 成功图全部绘制 + 烧录图像路径与检测路线 ASCII 文字 | D-07/D-08 |
| 8 | 检测串行 for 循环 | joblib loky 并行(run_parallel + TaskFailed 信封),max_workers 入配置 | PREP-02 |
| 9 | burst_consistency 连拍断言 | 不移植(本数据无同姿态连拍) | CONTEXT 裁量 |
| 10 | CLI 配置路径为模块级固定常量 CONFIG_PATH | callback 上可选 `--config` typer 选项(默认 config.yaml,赋值模块级 _config_path) | D-10 字面「命令行只传配置文件路径」(checker 修订定案) |

---

## No Analog Found(参考工程中无对应物,按 RESEARCH 模式实现)

| File/Function | Role | Data Flow | 实现依据 |
|---------------|------|-----------|----------|
| `artifacts.py` 的 CSV 三清单写入(usable / common_fov_pairs / fail_list) | utility(writer) | file-I/O | D-06 + RESEARCH §Code Examples 列建议:可用清单 `image_id, path, cam, route, sq_px`;公共对 `group_id, cam1_path, cam2_path, route_cam1, route_cam2`;fail `image_id, path, cam, reason`。用标准库 `csv` 模块,utf-8(带 BOM 供 Excel 直开可选,planner 定) |
| 公共视场配对逻辑(编排层) | service | transform | 同 group_id 下 cam1 与 cam2 均 detected 才入对;单边存在/单边检出 → 不入公共清单,**不额外记 fail**(planner 定案,checker 修订记录:检出侧已按 D-01 在 usable 清单、未检出侧已按检测失败入 fail 清单,fail 清单只记检测失败——把"单边检出"误标为失败会触发无谓的原图拷贝与复核噪音;01-03 任务 2 落地)。警告信号:公共对数 > min(cam1 检出, cam2 检出) 即逻辑错 |
| `io/config.py` 的 output_root 与 data 目录互斥校验 | guard(加载层) | — | CLAUDE.md data 只读 + INFRA-03:load_config 对 data.train/data.test/output_root 各 Path.resolve() 后双向互斥(相等/互相包含即中文报错),防 01-03 运行前 rmtree 误删数据、防产物写进 data/(checker 修订新增,参考工程无对应物) |
| `visualization/` 烧录文字实现细节 | renderer | transform | D-08:putText 位置/字号体例抄 overlay.py MISS 分支,内容 = 路径 + [sb/classic] |

---

## Metadata

**Analog search scope:** `/home/lx/Works/src/lx/github/intrinsic-calibration/`(src/calib 全部 60+ 文件结构已列;精读 14 个 analog 文件 + 2 个大文件目标区段);本仓库确认 greenfield(仅 CLAUDE.md/readme.md/.gitignore/data/.planning)
**Files scanned:** 参考工程 18 个文件(约 4,700 行,其中 cli/main.py 与 io/config.py 按区段精读)
**Pattern extraction date:** 2026-10-23
**Last updated:** 2026-10-23(checker 修订:Deviations #10 --config 选项;No-Analog 配对行记录最终定案;新增 output_root/data 互斥校验行)
