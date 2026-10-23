# Phase 1: 工程骨架 + 模块1 数据预处理 - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-10-23
**Phase:** 1-工程骨架 + 模块1 数据预处理
**Areas discussed:** 可用图像判定标准, 角点数据文件格式, 可视化抽查图策略, 工程骨架形态

---

## 可用图像判定标准

### Q1: "棋盘格完整可用"的判定标准是什么?

| Option | Description | Selected |
|--------|-------------|----------|
| 仅检测成功 | 88 角点全部找到即入清单,简单直接 | ✓ |
| 成功+边缘闸门 | 角点距图像边缘 ≥ N px(配置控制),剔除贴边棋盘格 | |
| 成功+边缘+清晰度 | 更严格,但清晰度阈值难调,参考工程未验证收益 | |

**User's choice:** 仅检测成功(自由文本澄清)
**Notes:** 用户明确"全部检出就算成功"。同时提出:失败的文件夹名要标记 fail 供人工复核——有些可能是角度/参数不当没提取出来(假失败),成功检出的里面也可能有角点明显偏的需要挑出来。因 data/ 只读,fail 标记落在输出目录(清单+文件名前缀)而非改 data/。

### Q2: 人工复核后,角点明显偏移的图像如何从可用清单中剔除?

| Option | Description | Selected |
|--------|-------------|----------|
| 配置排除清单 | exclude.txt 式文件,加路径重跑生效,可追溯 | |
| 手工改输出文件 | 简单但重跑被覆盖 | |
| 不需要机制 | Phase 1 不做,复核发现问题人工另行处理 | ✓ |

**User's choice:** 不需要机制

### Q3: 失败图像的复核形态用哪种?

| Option | Description | Selected |
|--------|-------------|----------|
| 仅 fail 清单 | 清单含路径+失败原因,自己去 data/ 翻原图 | |
| 清单+原图拷贝 | 失败帧原图拷贝到输出目录(fail_ 前缀),拷贝开关由配置控制 | ✓ |

**User's choice:** 清单+原图拷贝

---

## 角点数据文件格式

### Q1: 角点数据主文件用什么格式组织?

| Option | Description | Selected |
|--------|-------------|----------|
| 每相机一 npz | 每相机每数据集一个 npz,含角点数组+路径+路线标志,参考工程1 同风格 | ✓ |
| 每图像一文件 | 单图单查方便,但文件数 ~400 | |
| JSON | 人类可读但体积大、加载慢 | |

**User's choice:** 每相机一 npz

### Q2: 输出目录按 train/test 怎么组织?

| Option | Description | Selected |
|--------|-------------|----------|
| train/test 分目录 | 各含清单、npz、可视化,模块2 取数路径即语义 | ✓ |
| 混合+清单列区分 | 下游需过滤,易混淆 | |

**User's choice:** train/test 分目录

### Q3: 清单文件用什么格式?

| Option | Description | Selected |
|--------|-------------|----------|
| CSV | Excel 直接打开筛选排序,复核顺手 | ✓ |
| JSON | 参考工程1 风格,程序处理方便 | |
| CSV+JSON 双格式 | 内容一致双输出 | |

**User's choice:** CSV

---

## 可视化抽查图策略

### Q1: 成功图像的角点绘制范围?

| Option | Description | Selected |
|--------|-------------|----------|
| 全部成功图像 | 角点+顺序连线+烧录路径/路线,全量翻页排查角点偏移帧 | ✓ |
| 抽样 N 张 | 输出小,但会漏坏帧,与复核目标冲突 | |
| 全量+抽样开关 | 默认全量,配置降级 | |

**User's choice:** 全部成功图像
**Notes:** 用户在领域1 已说明复核需求(把成功但角点偏移的挑出来),全量绘制由此驱动。

---

## 工程骨架形态

### Q1: 三个模块的 CLI 入口怎么组织?

| Option | Description | Selected |
|--------|-------------|----------|
| 单 CLI 多子命令 | typer,如 stereo-calib preprocess/intrinsic/stereo,参考工程1 风格 | ✓ |
| 三独立入口 | 隔离彻底但重复配置、安装项多 | |

**User's choice:** 单 CLI 多子命令

### Q2: 配置文件用什么粒度?

| Option | Description | Selected |
|--------|-------------|----------|
| 单一 YAML 分段 | 全局段+每模块段,命令行只传这一个文件 | ✓ |
| 全局+每模块 YAML | 隔离彻底但要传多路径、共享项改两处 | |

**User's choice:** 单一 YAML 分段

### Q3: 输出目录采用固定直写还是时间戳版本化?

| Option | Description | Selected |
|--------|-------------|----------|
| 固定根目录直写 | 配置指定根目录,直接写入覆盖,路径稳定 | ✓ |
| 时间戳版本目录 | 保留历史但路径每次变 | |

**User's choice:** 固定根目录直写

---

## Claude's Discretion

- npz 内部 schema 细节(数组命名、dtype float64、布局)
- CSV 清单列定义
- classic 兜底窗口推导与阶梯重提细节(跟随参考工程1 detect.py)
- 并行实现方式与进程数配置
- 图像发现 glob 方式(容忍两层嵌套目录)

## Deferred Ideas

None — discussion stayed within phase scope
