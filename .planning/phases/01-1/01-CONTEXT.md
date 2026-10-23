# Phase 1: 工程骨架 + 模块1 数据预处理 - Context

**Gathered:** 2026-10-23
**Status:** Ready for planning

<domain>
## Phase Boundary

建立配置文件驱动的工程骨架(src layout + uv + typer + 单一 YAML 配置),并交付可独立运行的数据预处理模块:对 data/train(188 组)与 data/test(15 组)全部 cam1/cam2 图像完成棋盘格角点检测(SB 主路线 + classic 兜底,多进程并行),产出每路相机可用图像清单、公共视场图像对清单、角点数据 npz 文件、可视化角点图与失败帧复核材料。覆盖 PREP-01..06 与 INFRA-01..03。

不含:内参标定(Phase 2)、双目外参标定(Phase 3)、任何图像剔除机制的自动化(复核结果人工另行处理)。

</domain>

<decisions>
## Implementation Decisions

### 可用图像判定标准
- **D-01:** "棋盘格完整可用" = 仅检测成功——88 个角点全部检出即入可用清单,**不加**边缘距离、清晰度等额外质量闸门
- **D-02:** 失败图像输出 fail 清单(含图像路径与失败原因,如 cam1 SB+classic 均失败),并将失败帧**原图拷贝**到输出目录(文件名 `fail_` 前缀),拷贝行为由配置开关控制——用户要人工复核其中是否存在"棋盘格在但角度/参数不当导致没提出来"的假失败
- **D-03:** Phase 1 **不做**自动剔除机制——用户复核发现角点明显偏移的图像后,人工另行处理(不引入排除清单配置)

### 角点数据文件格式(定义模块 2/3 输入接口)
- **D-04:** 每相机每数据集一个 npz(如 `cam1_train.npz`、`cam2_test.npz`),内含:2D 角点坐标数组、3D 板系坐标、对应图像路径数组、检测路线标志(SB/classic)
- **D-05:** 输出根目录下按 `train/` 与 `test/` 分目录组织,各自包含清单、角点 npz、可视化图
- **D-06:** 清单文件(可用清单、公共视场对清单、fail 清单)统一用 CSV——Excel 可直接打开筛选排序,服务人工复核场景

### 可视化抽查图
- **D-07:** **全部**成功图像绘制角点图(不是抽样),存 visualization/ 子目录——用户复核目标是把"成功检出但角点明显偏移"的帧挑出来,抽样会漏
- **D-08:** 绘制内容:角点圆点 + 顺序连线(显示角点序,核验角点序一致性)+ 图上烧录图像路径与检测路线文字,翻图时无需对照清单

### 工程骨架形态
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

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### 参考工程(只读,绝不修改)
- `/home/lx/Works/src/lx/github/intrinsic-calibration/src/calib/corners/detect.py` — SB 主路线 + classic 兜底(窗口从本帧实测方格像素边长推导 + 缺陷闸门 + 窗口阶梯重提)的已验证实现,Phase 1 角点检测直接参照此文件
- `/home/lx/Works/src/lx/github/intrinsic-calibration/config/board.yaml`、`config/calibration.yaml` — 配置文件风格样板(结构、命名、分段方式)
- `/home/lx/Works/src/lx/github/intrinsic-calibration/pyproject.toml` — 工程骨架样板(src layout、uv、typer 入口、Python ≥3.12)
- `/home/lx/Works/src/bolight_robot/vvd_daq/multi_cameras` — 多目标定参考(主要服务 Phase 3,Phase 1 无需深读)

### 项目规划文档
- `.planning/PROJECT.md` §Context — 数据结构、参考工程方案要点、约束汇总
- `.planning/REQUIREMENTS.md` — PREP-01..06、INFRA-01..03 需求定义
- `.planning/ROADMAP.md` Phase 1 节 — 阶段目标与成功判据(5 条)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- 无本仓库代码——greenfield 项目。全部复用资产在参考工程(见 canonical refs),方案与代码只读引用

### Established Patterns
- 参考工程1 已验证模式:双检测器策略(SB 严禁二次亚像素精化;classic 兜底窗口自适应推导)、配置文件驱动、typer CLI、输出 npz+json
- 数据目录模式:`data/{train,test}/<ts>/<ts>_<us>/cam{1,2}.jpg` 两层嵌套(已实地核实,train=188 组、test=15 组,结构一致)

### Integration Points
- Phase 1 输出(npz 角点数据 + 清单)即 Phase 2/3 输入;npz schema 与 train/test 分目录结构是模块间契约

</code_context>

<specifics>
## Specific Ideas

- 用户的复核工作流是本阶段输出设计的核心驱动:翻 fail 原图找假失败(角度/参数问题),翻成功绘制图找角点偏移帧——所以 fail 原图要拷贝出来、成功图要全量绘制且烧录路径
- data/ 只读,fail 标记只能体现在输出目录文件名/清单中,不能改动 data/ 内任何文件

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 1-工程骨架 + 模块1 数据预处理*
*Context gathered: 2026-10-23*
