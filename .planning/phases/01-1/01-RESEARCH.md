# Phase 1: 工程骨架 + 模块1 数据预处理 - Research

**Researched:** 2026-10-23
**Domain:** Python/OpenCV 棋盘格角点检测批处理流水线 + 配置驱动工程骨架(uv + typer + src layout)
**Confidence:** HIGH(核心算法逐行读过参考工程实现;关键假设在本机 data/ 上实测验证)

<user_constraints>

## User Constraints (from CONTEXT.md)

### Locked Decisions

#### 可用图像判定标准
- **D-01:** "棋盘格完整可用" = 仅检测成功——88 个角点全部检出即入可用清单,**不加**边缘距离、清晰度等额外质量闸门
- **D-02:** 失败图像输出 fail 清单(含图像路径与失败原因,如 cam1 SB+classic 均失败),并将失败帧**原图拷贝**到输出目录(文件名 `fail_` 前缀),拷贝行为由配置开关控制——用户要人工复核其中是否存在"棋盘格在但角度/参数不当导致没提出来"的假失败
- **D-03:** Phase 1 **不做**自动剔除机制——用户复核发现角点明显偏移的图像后,人工另行处理(不引入排除清单配置)

#### 角点数据文件格式(定义模块 2/3 输入接口)
- **D-04:** 每相机每数据集一个 npz(如 `cam1_train.npz`、`cam2_test.npz`),内含:2D 角点坐标数组、3D 板系坐标、对应图像路径数组、检测路线标志(SB/classic)
- **D-05:** 输出根目录下按 `train/` 与 `test/` 分目录组织,各自包含清单、角点 npz、可视化图
- **D-06:** 清单文件(可用清单、公共视场对清单、fail 清单)统一用 CSV——Excel 可直接打开筛选排序,服务人工复核场景

#### 可视化抽查图
- **D-07:** **全部**成功图像绘制角点图(不是抽样),存 visualization/ 子目录——用户复核目标是把"成功检出但角点明显偏移"的帧挑出来,抽样会漏
- **D-08:** 绘制内容:角点圆点 + 顺序连线(显示角点序,核验角点序一致性)+ 图上烧录图像路径与检测路线文字,翻图时无需对照清单

#### 工程骨架形态
- **D-09:** 单 CLI 多子命令(typer,如 `stereo-calib preprocess/intrinsic/stereo`)——本阶段实现 preprocess,骨架预留三模块结构;三模块各自可单独运行调试
- **D-10:** 单一 `config.yaml` 分段:全局段(棋盘格规格 11×8/25mm、数据路径、输出路径)+ 每模块段(算法参数、开关);命令行只传配置文件路径,零算法参数
- **D-11:** 固定输出根目录(配置指定),每次运行直接写入覆盖,不做时间戳版本化——路径稳定,下游引用与复核沟通成本低
- **D-12:** 工程风格跟随参考工程1:src layout + uv 构建 + Python ≥3.12 + opencv 4.x + numpy + scipy

### Claude's Discretion
- npz 内部 schema 细节(数组命名、dtype 用 float64、axis 布局)
- CSV 清单的列定义(可用清单:路径/相机/检测路线;公共视场对清单:组目录、两路路径、两路路线等)
- classic 兜底的窗口推导与阶梯重提实现细节——直接跟随参考工程1 `src/calib/corners/detect.py`,不重新发明
- 并行实现方式(多进程 Pool,进程数等参数入配置段)
- 图像发现方式(glob 递归找 `cam1.jpg`/`cam2.jpg`,容忍 `<ts>/<ts>_<us>/` 两层嵌套结构)

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope

</user_constraints>

<phase_requirements>

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PREP-01 | 读 YAML 配置文件(棋盘格规格、数据路径、输出路径、角点检测参数),命令行零算法参数 | 参考 config/board.yaml + calibration.yaml 风格;dataclass 配置层 + 中文校验报错模式(参考 io/config.py) |
| PREP-02 | SB 主路线(严禁二次精化)+ classic 兜底(窗口从实测方格边长推导),多进程并行 | 参考 detect.py/window.py/canonical.py 逐行研究(下文 Pattern 1-3);本机实测 12 样本双检测器全通过;joblib loky 并行模式(参考 parallel.py) |
| PREP-03 | 输出每路相机"棋盘格完整可用"图像清单 | D-01 判定 = 检出即可用;CSV 格式(D-06) |
| PREP-04 | 输出两路相机公共视场图像对清单 | 组目录配对逻辑:叶子目录 `<ts>_<us>` 两路均检出(数据结构已实地核实,203 组每组恰好 cam1+cam2) |
| PREP-05 | 输出角点数据文件(2D + 3D 板系坐标 + 图像索引) | D-04 npz schema(下文 Code Examples);3D 板系坐标由 board 规格静态生成,序与 canonicalize 后 2D 序严格对应 |
| PREP-06 | 输出角点可视化抽查图 | `cv2.drawChessboardCorners` + `cv2.putText` 烧录 ASCII 文字(参考 overlay.py,D-07 全量绘制) |
| INFRA-01 | 三模块独立 CLI 入口,可单独运行调试 | typer 多子命令骨架,本阶段实现 preprocess、预留 intrinsic/stereo(参考 cli/main.py) |
| INFRA-02 | 全部传参由配置文件控制,禁止硬编码算法参数 | D-10 单一 config.yaml 分段;配置校验在加载层完成 |
| INFRA-03 | data/ 只读,所有产物写独立输出目录 | 参考工程模式:imread 只读 + 全部输出走 output root;测试可锁定 data/ 目录 mtime/hash 不变 |

</phase_requirements>

## Summary

本阶段的核心资产不在外部生态,而在参考工程1 `intrinsic-calibration` 的已验证实现:角点检测三件套(`detect.py` 双检测器策略、`window.py` 窗口推导与断言防线、`canonical.py` 排序规范化)可以近乎原样移植。研究逐行读完了这三个模块加配置层、数据扫描层、npz 落盘层、可视化层和 joblib 并行层,提取出可直接复用的模式与必须保持的不变量。

最有价值的验证是在本机 data/ 上实测完成的:SB 与 classic 两颗检测器在 12 张抽样图(train 前后段 + test,两路相机)上全部检出;**两颗检测器原始角点序确实不同**(同编号角点原始距离中位 355.9px、最大 590.1px),经参考工程 `canonicalize` 规范化后同编号距离中位 0.63px——这实证了规范化层必须移植,不能省;实测方格像素边长约 43.9px,窗口推导公式(得 20,满足 2×20+1=41 < 43.9)在本数据上成立。单图单检测器约 0.29s(cv2.setNumThreads(0) 单线程),406 张全量串行约 2-4 分钟、12 进程并行约 20-30s,性能无风险。

外部依赖极简且全部验证通过:opencv-python 4.14.0.94 / numpy 2.5.4 / scipy 1.18.1 / typer 0.27.3 / pyyaml 6.0.3 / joblib 1.6.0(与参考工程 uv.lock 逐项一致,slopcheck 全部 OK)。本机 uv 0.11.32 已装好 cpython-3.12.13,无需下载。一个环境陷阱:系统 pip 走清华镜像(numpy 最高只见 2.2.6,严重滞后),本工程必须用 uv 默认源(pypi.org)解析。

**Primary recommendation:** 骨架照抄参考工程1(pyproject.toml + src layout + typer + dataclass 配置层),角点三件套(detect/window/canonical)整块移植并保持"SB 严禁二次精化、窗口 2w+1<sq_px 断言、输出必过 canonicalize"三条不变量;并行用 joblib loky + 异常信封模式;输出按 train/test 分目录,npz 用带掩码稠密格式(全部图像占行 + detected 掩码 + source 路线标志)。

## Architectural Responsibility Map

单机批处理 CLI,无网络分层;按模块职责划分:

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| CLI 入口与参数解析 | `cli/main.py`(typer) | — | 命令行只收 config 路径,入口层只做编排与报错收口 |
| 配置加载与校验 | `io/config.py`(dataclass) | — | 所有算法参数唯一来源;非法值在加载层中文报错 |
| 图像发现与配对 | `io/dataset.py`(扫描) | preprocess 编排层 | 扫描层产记录(路径/相机/组 ID),配对(公共视场)在编排层按组 ID 做 |
| 角点检测(SB/classic) | `corners/detect.py` | `corners/window.py`、`corners/canonical.py` | 检测算法纯函数化:灰度图进、结构化 dict 出,便于子进程调用 |
| 并行调度 | `preprocess/parallel`(joblib loky) | — | 图像级任务切分;像素数据不跨进程,只传路径等小元组 |
| 落盘(npz/CSV/可视化) | `preprocess/artifacts.py` | `visualization` | 上游输出即下游输入的契约层 |
| data/ 只读保证 | 全部读路径(imread) | 测试锁定 | 只读是横切约束,不归任何单模块 |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| opencv-python | 4.14.0.94(`<5.0`) | 角点检测、绘图、图像读写 | 参考工程锁定同版本区间;5.0(2026-08)刚发布 calib3d 行为易变 [VERIFIED: 参考 pyproject.toml + PyPI] |
| numpy | 2.5.4 | 角点数组、npz 落盘 | 与 scipy 1.18 成对验证过的组合,必须一起锁 [VERIFIED: PyPI registry, 2026-10-10 发布;参考 uv.lock] |
| scipy | 1.18.1 | Phase 1 仅作依赖就位(Phase 2 用) | 同上;CLAUDE.md 技术栈约束 [VERIFIED: PyPI registry, 2026-08-21 发布] |
| typer | 0.27.3 | 多子命令 CLI(`preprocess/intrinsic/stereo`) | 参考工程同款 [VERIFIED: PyPI registry] |
| pyyaml | 6.0.3 | config.yaml 加载 | 参考1配置驱动模式的载体 [VERIFIED: PyPI registry] |
| joblib | 1.6.0(`>=1.4`) | loky 常驻进程池并行角点检测 | `return_as="generator"`(1.3+)+ `parallel_config(inner_max_num_threads)` 是参考工程压测过的组合 [VERIFIED: PyPI registry;参考 parallel.py] |

### Supporting (dev)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest | >=8 | 单元测试(detect/window/canonical 的不变量锁定) | dev 依赖组 |
| ruff | >=0.8 | lint | dev 依赖组 |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| joblib loky | stdlib `multiprocessing.Pool` | Pool 更零依赖,但丢掉参考工程已踩平的三件事:异常信封(任一任务抛异常 joblib 会关整条结果流)、生成器式进度、`inner_max_num_threads` 防多进程叠满线程互抢内存带宽——建议直接移植 parallel.py 模式 |
| matplotlib 可视化 | (不用) | Phase 1 角点标注用 `cv2.drawChessboardCorners` 一次成图,比 matplotlib 快且标准;不引入 |
| findChessboardSBWithNovelAreas | findChessboardCornersSB | NovelAreas 变体服务"棋盘出画检测",本项目不需要 |

**Installation:**
```bash
uv init --package .  # 或手写 pyproject(hatchling + src layout,照参考模板)
uv add "opencv-python>=4.14,<5.0" "numpy==2.5.*" "scipy==1.18.*" "typer>=0.15" "pyyaml>=6.0" "joblib>=1.4"
uv add --dev pytest ruff
```

**Version verification(全部已完成):**
- numpy 2.5.4、scipy 1.18.1、typer 0.27.3、joblib 1.6.0:pypi.org JSON API 逐个确认存在 [VERIFIED: PyPI registry]
- opencv-python 4.14.0.94、pyyaml 6.0.3:参考工程 uv.lock 解析结果(SHA256 锁定)+ 镜像列表确认 [VERIFIED: 参考 uv.lock]
- **陷阱:系统 pip 配置了清华镜像**(`global.index-url='https://pypi.tuna.tsinghua.edu.cn/simple'`),numpy 只见到 2.2.6——本工程必须用 uv 默认源,绝不用系统 pip 装依赖 [VERIFIED: 本机 pip config list]

## Package Legitimacy Audit

slopcheck 0.6.1 已安装并运行:`slopcheck install opencv-python numpy scipy typer pyyaml joblib` → **6 packages, 6 OK, 0 SUS, 0 SLOP**。

| Package | Registry | Age | Downloads | Source Repo | slopcheck | Disposition |
|---------|----------|-----|-----------|-------------|-----------|-------------|
| opencv-python | PyPI | 14+ yrs | 极高 | github.com/opencv/opencv-python | [OK] | Approved |
| numpy | PyPI | 19+ yrs | 极高 | github.com/numpy/numpy | [OK] | Approved |
| scipy | PyPI | 24+ yrs | 极高 | github.com/scipy/scipy | [OK] | Approved |
| typer | PyPI | 6+ yrs | 高 | github.com/fastapi/typer | [OK] | Approved |
| pyyaml | PyPI | 18+ yrs | 极高 | github.com/yaml/pyyaml | [OK] | Approved |
| joblib | PyPI | 15+ yrs | 极高 | github.com/joblib/joblib | [OK] | Approved |

额外佐证:六个包全部出现在参考工程 uv.lock 中(带哈希),即"官方文档级"使用先例。postinstall 检查:均无可疑 postinstall 网络行为(opencv-python 仅编译验证型脚本)。

**Packages removed due to slopcheck [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

## Architecture Patterns

### System Architecture Diagram

```
                     config.yaml(唯一参数来源)
                            │
                            ▼
        ┌───────── stereo-calib preprocess <config路径> ─────────┐
        │                    typer CLI(零算法参数)               │
        ▼                                                     │
  io/config.py 加载+校验(dataclass,中文报错)                     │
        ▼                                                     │
  io/dataset.py 扫描 data/{train,test}/<ts>/<ts>_<us>/cam{1,2}.jpg │
  → ImageRecord(路径/相机/组ID/数据集)+ 警告清单                  │
        ▼                                                     │
  并行调度(joblib loky, max_workers 来自配置)                   │
  ┌───────┬───────┬───────┬─── … 每图一任务(路径小元组) … ─────┐ │
  ▼       ▼       ▼       ▼                                  │ │
 worker: imread → detect_sb ──成功──► canonicalize ──► dict    │ │
                      │失败(配置开关)                          │ │
                      ▼                                       │ │
              detect_classic(粗定位→窗口推导精化→缺陷闸门       │ │
                      │        →必要时窗口阶梯重提)             │ │
                      ▼失败                                    │ │
                 None(该帧记 fail)                             │ │
  └───────┴───────┴───────┴──────────────────────────────────┘ │
        ▼  (异常装 TaskFailed 信封,不杀任务流)                  │
  结果汇总(主进程)                                              │
  ├─► 按 {train,test} × {cam1,cam2} 组装                       │
  ├─► npz: cam{n}_{split}.npz(uv/objp/paths/source/detected)   │
  ├─► CSV: usable清单 / common_fov_pairs / fail_list           │
  ├─► visualization/: 全量成功图角点标注(drawChessboardCorners  │
  │    + putText 烧录路径/路线)                                 │
  └─► fail_originals/: 失败帧原图副本(fail_ 前缀,开关控制)      │
        │                                                     │
        ▼                                                     │
  output/(固定根目录,直接覆盖,无时间戳)  ◄── data/ 全程只读 ────┘
```

### Recommended Project Structure
```
stereo-calibration/
├── pyproject.toml            # hatchling + src layout + [project.scripts]
├── config.yaml               # 单一配置:全局段(board/data/output)+ 模块段
├── src/stereo_calib/         # 包名不可带连字符
│   ├── cli/
│   │   └── main.py           # typer app:preprocess(实现)/intrinsic/stereo(预留)
│   ├── io/
│   │   ├── config.py         # dataclass 配置层 + load_config(校验+中文报错)
│   │   └── dataset.py        # 扫描 <ts>/<ts>_<us>/cam{1,2}.jpg → 记录+警告
│   ├── corners/
│   │   ├── detect.py         # 移植参考:detect_sb / detect_classic(+兜底编排)
│   │   ├── window.py         # 移植参考:estimate_sq_px / derive_window_*(断言防线)
│   │   └── canonical.py      # 移植参考:canonicalize(burst_check 不需要,见下)
│   ├── preprocess/
│   │   ├── pipeline.py       # 编排:扫描→并行检测→汇总→落盘
│   │   ├── parallel.py       # joblib loky + TaskFailed 信封(移植参考 parallel.py)
│   │   └── artifacts.py      # npz/CSV 读写 + meta.json
│   └── visualization/
│       └── overlay.py        # drawChessboardCorners + putText 标注
├── tests/                    # 不变量测试(见下)
├── data/                     # 只读输入(train 188 组 / test 15 组)
└── output/                   # 全部产物(gitignore)
```

### Pattern 1: SB 主路线,严禁二次亚像素精化
**What:** `cv2.findChessboardCornersSB` 一次输出亚像素角点;SB 内部角点模型与 `cornerSubPix` 模型不同,叠加二次精化会互相污染、精度反降(参考工程 Pitfall 4,三处注释反复强调)。
**When to use:** 每图先跑 SB;`cv2.cornerSubPix` 的调用全文件只允许出现在 classic 分支。参考工程用源码防线测试锁定"cornerSubPix 只在 classic 分支被调用"。
**Example:**
```python
# Source: 参考工程 src/calib/corners/detect.py(L215-239)
found, raw = cv2.findChessboardCornersSB(gray, (cols, rows), cfg.sb_flags)
if not found:
    return None                      # 未检出返回 None,绝不返回半成品
corners = raw.reshape(-1, 2).astype(np.float64)
sq_px = estimate_sq_px(corners, cols, rows)   # 仅记录口径,不参与精化
return {"uv": canonicalize(corners, cols, rows), "detector": "sb", ...}
```

### Pattern 2: classic 兜底——窗口从本帧实测方格边长推导
**What:** `findChessboardCorners` 粗定位(整像素)→ `cornerSubPix` 精化,半窗口 `w = max(min_win, floor(sq_px/2) - 1)`,断言 `2w+1 < sq_px`(头号陷阱:搜索区跨过相邻角点会产生重投影误差里完全看不见的系统性偏差)。`sq_px` = 行/列相邻角点间距中位数取小(保守,防透视下远处方格更小)。
**When to use:** SB 未检出且配置开关 `sb_fallback_classic: true` 时;SB 结果绝不进入此路径。
**Example:**
```python
# Source: 参考工程 src/calib/corners/window.py(L42-59)
def derive_window_frame(sq_px: float, min_win: int = 3) -> int:
    w = max(int(min_win), int(sq_px // 2) - 1)
    assert 2 * w + 1 < sq_px, ( ...中文报错,解释头号陷阱... )
    return w

# Source: 参考工程 detect.py(L150-151)经典粗定位
found, raw = cv2.findChessboardCorners(gray, (cols, rows), _classic_flag_value(cfg.classic_flags))
# classic_flags: ["adaptive_thresh", "normalize_image"] → cv2 常量位掩码(配置层校验合法名)
```

### Pattern 3: 缺陷闸门 + 窗口阶梯重提
**What:** classic 精化后数两类硬伤——重复点(相邻间距 < 0.25×中位方格)与锯齿(间距二阶差分 > 0.3×中位方格);有缺陷则**从粗定位网格**(整数精度但次序可靠)出发,用阶梯窗口 `cap → cap-2 → … → 3`(cap 按间距 2% 分位同一公式推导)逐级重提,第一个零缺陷的胜出;恢复过程(缺陷数/试过的窗口/最终窗口)全部记入返回值留痕。重提用更严收敛判据(max_iter=100, eps=1e-6)。
**When to use:** 每帧 classic 精化后无条件过闸门;正常帧零缺陷原样通过(逐字节不动)。
**Empirical note:** 本数据方格约 44px(实测),远大于参考事故帧的临界情形,预计极少触发;但闸门必须移植——它是唯一能拦截"窗口吞邻角"类不可见偏差的运行时防线。 [VERIFIED: 本机实测 sq_px=43.9px]

### Pattern 4: canonicalize 排序规范化(必须移植,已实证)
**What:** 两颗检测器对同一块板的角点序起点/方向可以不同。规范化规则纯几何:行主方向 `.x < 0` 则每行反转,列主方向 `.y < 0` 则行序反转,最终唯一形态 = 行向画面右、列向画面下。
**Empirical proof(本机 data/,2026-10-23):** 同一帧 SB vs classic 原始同编号角点距离中位 355.9px / 最大 590.1px(序完全不同);规范化后中位 0.63px / 最大 1.53px(classic 未精化的整像素差)。**不过规范化层,兜底机制产出的 npz 就是废数据。** [VERIFIED: 本机实测]
**Example:**
```python
# Source: 参考工程 src/calib/corners/canonical.py(L28-51)
g = np.asarray(corners, dtype=np.float64).reshape(rows, cols, 2)
if float((g[:, 1:, :] - g[:, :-1, :])[..., 0].mean()) < 0:  # 行方向朝左→行内反转
    g = g[:, ::-1, :]
if float((g[1:, :, :] - g[:-1, :, :])[..., 1].mean()) < 0:  # 列方向朝上→行序反转
    g = g[::-1, :, :]
return g.reshape(-1, 2).copy()
```

### Pattern 5: joblib loky 并行 + 异常信封
**What:** 图像级任务切分(每任务一个小元组:路径/相机/组ID/配置参数),loky 常驻工作进程避免逐任务重导入 cv2;`return_as="generator"` 逐个消费驱动进度;`parallel_config(inner_max_num_threads=核数//进程数)` 防多进程叠满线程互抢内存带宽(参考实测 8 路并发从 8 倍掉到 2 倍);worker 内任何异常包 `TaskFailed` 信封交回——joblib 后端在任一任务抛异常时会关掉整条结果流(参考 2026-10-19 真实事故:700+ 任务全没)。
**When to use:** `max_workers <= 1` 时同函数串行回退(调试用,结果与并行一致)。
**Key detail:** 像素数据不跨进程——worker 自己 imread,返回小 dict(uv 数组 88×2 很小,可序列化)。 [VERIFIED: 参考 parallel.py L93-109 压测注释]

### Pattern 6: 配置层 dataclass + 中文校验报错
**What:** 全部配置键映射 frozen dataclass;加载时校验取值合法(如 classic_flags 只许已知名字、max_workers ≥ 1),非法值抛带中文说明的 ValueError;CLI 入口收口为中文一句 + exit code 1,不吐堆栈。typer `@app.callback()` 里按配置 `cv2.setNumThreads(cfg.cv2_threads)`(0 = 关 OpenCV 内部多线程,并行交给上层显式控制)。
**Example(config.yaml 骨架,风格照参考 board.yaml + calibration.yaml):**
```yaml
# ---- 全局 ----
board: {cols: 11, rows: 8, square_size_mm: 25.0, unit: "mm"}
data: {train: "data/train", test: "data/test"}
output_root: "output"
cv2_threads: 0
# ---- 模块1:数据预处理 ----
preprocess:
  max_workers: 12            # 并行进程上限;1 = 串行调试
  corners:
    classic_flags: [adaptive_thresh, normalize_image]
    sb_flags: 0
    sb_fallback_classic: true # SB 未检出时 classic 兜底(路线记入 npz source)
    subpix: {window_granularity: frame, min_win: 3, max_iter: 30, epsilon: 0.01}
  copy_fail_originals: true   # D-02 失败帧原图拷贝开关
  visualization: {jpeg_quality: 90}
```

### Pattern 7: npz 带掩码稠密格式 + meta.json 旁车
**What:** 参考工程的角点 npz 契约:每图占一行(未检出 NaN + detected=False,绝不从数据里消失),定长 unicode 数组存 source 路线(不需要 pickle),npz 旁配 meta.json 记录配置快照/规范化规则版本/统计;读取一律 `np.load(..., allow_pickle=False)`(npz pickle 模式可执行任意代码,安全防线)。本工程扩展出 D-04 要求的 3D 板系坐标与路径数组。
**Schema 建议(细节属 Claude 裁量,给出推荐):**
```python
# cam{n}_{split}.npz, N = 该数据集该相机全部图像数
image_ids: (N,) <U32    # 唯一 ID = 叶子目录名 + "_cam{n}",如 "20261023_154025_519359_cam1"
paths:      (N,) <U str  # 相对仓库根的图像路径(Excel/下游可读)
uv:         (N, 88, 2) float64  # 规范化序角点;未检出整行 NaN
objp:       (88, 3) float64     # 3D 板系:x=col*25.0, y=row*25.0, z=0(静态,序与 uv 严格对应)
source:     (N,) <U16    # "sb" / "classic" / ""
detected:   (N,) bool
sq_px:      (N,) float64 # 每帧实测方格边长(审查窗口断言用)
win_size:   (N,) float64 # classic 实际窗口;SB 帧按同一公式记录(仅口径)
# meta.json: board 规格、配置快照、规范化规则版本、detected/source 计数、失败原因表
```
**Critical contract:** `objp[k]` 必须对应 `uv[:, k]`——canonicalize 后角点序恒为"行方向朝右、列方向朝下"的行主序,故 `objp[row*11+col] = (col*25.0, row*25.0, 0)`,静态生成一次即可。

### Anti-Patterns to Avoid
- **对 SB 结果再跑 cornerSubPix:** 精度反降(两套角点模型互相污染);参考工程用源码防线测试锁定调用位置
- **窗口写死像素:** 必须逐帧从实测方格边长推导 + 断言 `2w+1 < sq_px`;写死在换分辨率/换距离数据上直接产生不可见系统偏差
- **跳过 canonicalize 直接存角点:** SB 与 classic 序不同(本数据实测 356px 级差异),兜底帧混入后下游全错
- **cv2.imread 不检查返回值:** 读坏文件返回 None 而非报错,后续报错位置远离病因;必须 `imread_checked` 模式
- **np.load 不带 allow_pickle=False:** pickle 反序列化可执行任意代码
- **图上烧录中文:** `cv2.putText` 不支持 CJK 字形,图内文字只能 ASCII(路径/路线名恰为 ASCII);中文说明写 CSV/summary
- **并行 worker 抛裸异常:** joblib 会关闭整条结果流,一个坏帧赔掉整批;必须异常信封
- **数据扫描假设固定两层:** 用 glob 容忍 `<ts>/<ts>_<us>/` 结构(实测 203 个 ts 目录每个恰好 1 个子目录,但扫描要容错并告警,不硬编码)
- **时间戳版本化输出:** D-11 明确固定根目录直接覆盖;若上次运行残留孤儿文件(本次已检出图像变少),重跑前应清空 preprocess 产物子目录或全量覆盖同名文件

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| 角点可视化绘制 | 自绘圆圈+连线 | `cv2.drawChessboardCorners(img, (11,8), corners, True)` | 一行出标准叠加(交替色圆点+顺序连线,自带角点序信息);参考工程研究结论:自画既费工又不标准 |
| 亚像素窗口推导 | 固定窗口/经验值 | 移植 window.py(实测边长+断言) | 头号陷阱防线,数学上保证搜索区不吞邻角 |
| 角点排序统一 | 自定起始角约定 | 移植 canonical.py(纯几何规范化) | 已实证两检测器序不同;纯几何规则四种翻转变体恰有一个映射到唯一形态 |
| 进程池与异常隔离 | 手写 multiprocessing + try/except | 移植 parallel.py(joblib loky) | 异常信封/生成器进度/线程配额三件事都是踩过坑的定制品 |
| YAML 校验 | 手写 dict 检查 | dataclass + 加载层校验(参考 io/config.py) | 非法配置在入口中文报错,不进算法层 |
| 唯一图像 ID | 直接用文件名 stem | 叶子目录名 + "_cam{n}" | 本数据文件名全部是 cam1.jpg/cam2.jpg,**stem 全库重复**,直接用会在 npz/CSV/可视化文件名上碰撞(参考工程单相机 stem 唯一的前提在此不成立)|

**Key insight:** 本阶段的"不要自己造"边界比一般项目更宽——参考工程1 是同作者、同风格、同类相机的已验证实现,连中文报错文案风格都应保持一致;除胶水编排(扫描配对、并行调度、落盘汇总)外,算法路径应当是"移植 + 不变量测试锁定",不是重写。

## Common Pitfalls

### Pitfall 1: SB 与 classic 角点序不一致(已在数据上实证)
**What goes wrong:** 兜底帧的 classic 角点序与 SB 帧不同,同编号角点指向板上不同物理点,Phase 2/3 全部对不上。
**Why it happens:** OpenCV 已知行为——两检测器起点/方向可不同。
**How to avoid:** 两条路线输出都过 canonicalize(移植);npz meta 记录规范化规则版本字符串。
**Warning signs:** 抽查可视化图上连线顺序方向不一(SB 帧与兜底帧连线走向不同即规范化失效)。

### Pitfall 2: 亚像素窗口吞邻角(头号陷阱)
**What goes wrong:** `cornerSubPix` 半窗口 w 的实际搜索区 (2w+1)² ≥ 方格像素边长时,角点被系统性拉偏,且偏差被标定求解整体吸收,重投影误差完全看不出来。
**Why it happens:** 窗口写死或推导越界;透视下画面边缘方格比中位数小。
**How to avoid:** 逐帧 derive_window_frame + 断言;npz 存 sq_px 与 win_size 供事后复核 `2*win+1 < sq_px`。
**Warning signs:** 断言报错即数据方格过小;npz 复核脚本发现违反即流程有漏。

### Pitfall 3: 图像文件名不唯一
**What goes wrong:** 用文件名 stem 当 image_id,406 张图只有 2 个不同 stem(cam1/cam2),npz 对不上图、CSV 筛选失效、可视化文件互相覆盖。
**How to avoid:** image_id = 叶子目录名(时间戳+微秒,天然唯一)+ "_cam{n}";fail 副本命名 `fail_{image_id}.jpg`。
**Warning signs:** 可视化目录文件数远小于成功数。

### Pitfall 4: 向 data/ 写入任何文件
**What goes wrong:** 违反 INFRA-03 与 CLAUDE.md 硬约束;cv2.imwrite 路径拼错时可能命中源目录。
**How to avoid:** 全部输出路径从 output_root 派生;测试锁定:跑完 preprocess 后对 data/ 做 mtime/tree hash 断言不变。
**Warning signs:** git status 出现 data/ 变更。

### Pitfall 5: pip 镜像滞后导致装错版本
**What goes wrong:** 系统 pip 指向清华镜像(numpy 最高 2.2.6),按参考工程的 2.5.4/1.18.1 组合装不上或降级,numpy/scipy 版本错配引发 ABI/行为差异。
**How to avoid:** 全部依赖操作走 uv(uv 默认 pypi.org);不向系统 python 装任何东西。
**Warning signs:** uv.lock 解析结果与参考组合不一致。

### Pitfall 6: 并行任务异常杀整批
**What goes wrong:** 某帧 imread 损坏文件抛 ValueError,joblib 关闭结果流,已完成的几百帧结果全丢。
**How to avoid:** worker 内全捕获包 TaskFailed 信封;汇总层拆信封记入 fail 清单(单帧读图失败 = 该帧 fail,不是全批崩溃)。
**Warning signs:** 进程池提前结束、结果数 < 任务数。

### Pitfall 7: 固定输出目录的残留文件
**What goes wrong:** 上次运行检出 400 帧,本次因配置收紧只检出 380 帧,visualization/ 里 20 张旧图残留,人工复核看到已失效的标注。
**How to avoid:** 运行开始时清理 preprocess 产物子目录(visualization/、fail_originals/、CSV、npz 全部重写);或整目录删除重建。
**Warning signs:** 可视化图数量 > npz detected 计数。

### Pitfall 8: 公共视场配对逻辑错位
**What goes wrong:** 用 `<ts>` 层目录当组 ID 而某组有多个 `<ts>_<us>` 子目录时配对错;或两路只有一个文件时仍算一对。
**How to avoid:** 组 ID = 叶子目录 `<ts>_<us>`;配对条件 = 同叶子目录下 cam1 与 cam2 都检出;单边存在/单边检出 → 不入公共清单(可记入 fail 清单注明原因)。当前数据 203 个叶子目录全部恰好两路齐全(实测),但代码必须容错。
**Warning signs:** 公共对数 > min(cam1 检出数, cam2 检出数)。

### Pitfall 9: cv2.putText 中文乱码
**What goes wrong:** 烧录中文路径/说明画成问号。
**How to avoid:** 图内文字只用 ASCII(路径、cam、sb/classic、px 值);中文写 CSV 与 summary。

## Code Examples

### 检测 worker(单图,子进程内运行)
```python
# 组合参考 detect.py 三件套后的本工程形态(编排示意)
def detect_one(task: tuple) -> dict:
    path, cam, group_id, split, board_cfg, corners_cfg = task
    gray = imread_checked(path)                       # 读坏文件当场中文报错(TaskFailed 包住)
    result = detect_sb(gray, board_cfg.cols, board_cfg.rows, corners_cfg)
    if result is None and corners_cfg.sb_fallback_classic:
        result = detect_classic(gray, board_cfg.cols, board_cfg.rows, corners_cfg)
        # source 记 "classic"(detect_classic 内部返回 detector 字段)
    return {"path": str(path), "cam": cam, "group_id": group_id, "split": split,
            "result": result}                          # result None = 双路线失败
```

### 3D 板系坐标(与规范化角点序严格对应)
```python
# objp[k] 其中 k = row*cols + col;canonicalize 保证 uv 同序
def build_object_points(cols: int, rows: int, square_mm: float) -> np.ndarray:
    pts = np.zeros((rows * cols, 3), dtype=np.float64)
    pts[:, 0] = np.tile(np.arange(cols, dtype=np.float64), rows) * square_mm
    pts[:, 1] = np.repeat(np.arange(rows, dtype=np.float64), cols) * square_mm
    return pts   # 11×8/25mm → (88,3),x∈[0,250],y∈[0,175]
```

### 可视化标注(全量成功图)
```python
# Source: 参考工程 visualization/overlay.py(L54-58)+ 本工程烧录要求
img = cv2.imread(str(path), cv2.IMREAD_COLOR)          # 原图只读进内存,标记画在副本
corners = uv.astype(np.float32).reshape(-1, 1, 2)
cv2.drawChessboardCorners(img, (cols, rows), corners, True)   # 圆点+顺序连线(交替色=角点序)
cv2.putText(img, f"{path} [{route}]", org=(10, 30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
            color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)  # ASCII only
cv2.imwrite(str(out_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
```

### 扫描与公共视场配对
```python
# 数据契约(本机实测核实):data/{split}/<ts>/<ts>_<us>/cam{1,2}.jpg
# 203 个叶子目录(train 188 + test 15),每个恰好 cam1.jpg + cam2.jpg,共 406 张
# 扫描:Path(split_dir).glob("*/*/cam*.jpg"),容忍告警;组 ID = 叶子目录名
# 配对:同 group_id 下两路 detected 均为 True → common_fov_pairs.csv 一行
# CSV 列建议(公共视场对):group_id, cam1_path, cam2_path, route_cam1, route_cam2
# CSV 列建议(可用清单):image_id, path, cam, route, sq_px
# CSV 列建议(fail 清单):image_id, path, cam, reason(如 "sb+classic both failed" / "read error: ...")
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| findChessboardCorners + cornerSubPix(经典) | findChessboardCornersSB(一次亚像素) | OpenCV 4.3+(2020) | SB 对光照/倾斜更稳,但大倾斜下会系统性拒检 → 双检测器策略是当前最优实践(参考工程三轮实验定案:多尺度重试无增量价值,classic 兜底更简单且多覆盖帧) |
| OpenCV 5.0 已发布(2026-08) | 本工程锁 `>=4.14,<5.0` | 2026-08 | calib3d 行为易变,与"可审计可复现"冲突;参考工程明确等 5.x 攒够补丁再评估 |

**Deprecated/outdated:**
- OpenCV 5.x:刚发布,calib3d 数值行为未稳,本阶段禁用(约束已锁)

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | SB+classic 在全量 406 张上预计高成功率(抽样 12 张全过;多数帧两路都能看到棋盘) | Summary / Pitfalls | 若全量跑出大量双失败,属数据本身问题(假失败复核流程即为此设计);不阻塞架构 |
| A2 | 每叶子目录恰好 cam1+cam2 两个文件(当前实测成立),扫描层仍按容错实现 | Pattern 7 / Pitfall 8 | 若后续数据变化,告警+fail 清单兜住,不需改架构 |
| A3 | 固定输出覆盖制下,清理子目录重写即可满足"无残留"(无需增量对账) | Pitfall 7 | 若用户要求保留上次结果,D-11 已锁死不版本化——属用户决策域,非技术风险 |

**其余论断均直接读自参考工程源码或本机实测,无需用户确认。**

## Open Questions

1. **CLI 命令名与包名定稿**
   - What we know: CONTEXT D-09 示例 `stereo-calib`;Python 包名不能带连字符,需 `stereo_calib`。
   - What's unclear: 命令名最终取 `stereo-calib` 还是其他。
   - Recommendation: 按 CONTEXT 示例用 `stereo-calib`,包名 `stereo_calib`,planner 直接定案即可(低风险)。

2. **npz 是否包含未检出图像行**
   - What we know: 参考工程带掩码稠密格式(全图占行 + detected 掩码);D-04 只说"内含四类数组"。
   - What's unclear: Phase 2 消费方(尚未实现)期望检出子集还是掩码。
   - Recommendation: 用带掩码稠密格式(与参考一致、行对齐 CSV、下游 `detected` 过滤一行代码);planner 可在 PLAN 中把该选择记为 schema 契约。

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| uv | 构建/依赖/运行 | ✓ | 0.11.32 | — |
| Python ≥3.12 | 全部(D-12) | ✓ uv 管理版已装 | cpython-3.12.13(另有 3.14.6) | 系统是 3.10.12,**不可用**于本工程 |
| pypi.org 网络 | uv 解析依赖 | ✓(偶发超时需重试) | — | 参考 uv.lock 版本组合已验证存在 |
| data/ 数据集 | preprocess | ✓ | train 188 组 + test 15 组 = 406 张,2560×1440 JPEG(~368KB/张) | — |
| CPU/内存 | 并行 | ✓ | 20 核 / 31GB(当前可用 ~10GB) | max_workers 配置默认 12,留余量 |
| 磁盘(输出) | 可视化+副本 | ✓ | 全量成功图 JPEG q90 约 300-500KB/张 → 406 张约 150-200MB | jpeg_quality 可调 |

**Missing dependencies with no fallback:** none
**Missing dependencies with fallback:** none

## Security Domain

ASVS Level 1;离线批处理 CLI,无网络服务、无认证、无会话、无密钥。适用项极少:

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | 离线单用户工具 |
| V3 Session Management | no | 无会话 |
| V4 Access Control | no | 无多用户 |
| V5 Input Validation | yes | 配置层 dataclass 校验(非法值中文报错);图像路径白名单化扫描(只认 `cam*.jpg`);image_id 安全字符模式(参考 `_SAFE_IMAGE_ID` 正则,防特殊字符把产物写出 output/ 之外)——建议移植 |
| V6 Cryptography | no | 无加密需求 |
| V12 File Security | partial | `np.load(allow_pickle=False)`(npz pickle 可执行任意代码);输出路径全部从配置 output_root 派生,不拼接用户自由文本 |

### Known Threat Patterns for 离线图像批处理 Python CLI

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| 恶意 npz(pickle 载荷) | Tampering/EoP | 读取一律 allow_pickle=False;定长 unicode 数组存字符串 |
| 配置注入异常路径 | Tampering | 配置校验 + 输出路径派生约束 + image_id 安全字符模式 |
| 参考工程误写 | Tampering | 参考路径只出现在读操作;测试断言写路径均在 output/ 下 |

## Sources

### Primary (HIGH confidence)
- 参考工程1 `/home/lx/Works/src/lx/github/intrinsic-calibration/src/calib/corners/detect.py` — 双检测器策略、缺陷闸门、窗口阶梯重提(逐行精读)
- 参考工程1 `src/calib/corners/window.py`、`src/calib/corners/canonical.py` — 窗口推导断言、排序规范化(逐行精读)
- 参考工程1 `src/calib/io/dataset.py`、`src/calib/io/config.py` — 扫描容错模式、dataclass 配置层
- 参考工程1 `src/calib/corners/artifacts.py` — npz schema 契约与 allow_pickle 防线
- 参考工程1 `src/calib/evaluation/parallel.py` — joblib loky + 异常信封并行模式
- 参考工程1 `src/calib/visualization/overlay.py` — drawChessboardCorners 标注模式
- 参考工程1 `pyproject.toml` + `uv.lock` — 骨架模板与已验证依赖组合(numpy 2.5.4 / scipy 1.18.1 / opencv 4.14.0.94 / typer 0.27.3 / pyyaml 6.0.3 / joblib 1.6.0)
- 本机实测(2026-10-23):数据结构 406 张全量核实、12 图双检测器全检出、SB/classic 序差异 355.9px→规范化 0.63px、sq_px=43.9px、单检测器 ~0.29s/图

### Secondary (MEDIUM confidence)
- pypi.org JSON API — numpy 2.5.4(2026-10-10)、scipy 1.18.1(2026-08-21)、typer 0.27.3(2026-10-06)、joblib 1.6.0(2026-08-31)存在性
- slopcheck 0.6.1 — 六包全部 [OK]

### Tertiary (LOW confidence)
- 无(本研究未依赖未验证网络来源;OpenCV 行为论断全部来自参考工程已验证代码注释与本机实测)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — 与参考工程 uv.lock 逐项一致 + PyPI 验证 + slopcheck 全过
- Architecture: HIGH — 全部模式取自参考工程同构实现;数据结构实测核实
- Pitfalls: HIGH — 关键陷阱(SB 序差异、窗口推导)在本数据上实测复现/验证;其余来自参考工程真实事故记录(158294 事故、joblib 结果流事故)

**Research date:** 2026-10-23
**Valid until:** 2026-11-22(依赖版本 30 天内稳定;数据与参考工程为本地资产,不过期)
