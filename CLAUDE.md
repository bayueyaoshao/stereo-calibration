<!-- GSD:project-start source:PROJECT.md -->

## Project

**双目相机标定生产工具 (stereo-calibration)**

基于已采集的同步双目图像对(data/train 188 组、data/test 15 组,cam1.jpg/cam2.jpg),快速计算可靠的双目相机标定结果的生产工具。分三个相互独立、可单独调试的模块:数据预处理(角点检测与筛选)、单相机内参标定与验证、双目外参标定与验证。使用者是标定产线操作人员/工程师,输入图像数据,输出标定参数与量化指标。

**Core Value:** 快速、可靠地产出经过测试集指标验证的双目标定结果(内参 + 双目外参 R/T)。

### Constraints

- **Tech stack**: Python + OpenCV 4.x + numpy + scipy,参考工程1风格(配置文件驱动,命令行零算法参数)— 与已验证工程保持一致
- **配置文件**: 所有传参必须用配置文件控制(棋盘格规格、路径、算法参数、开关),禁止硬编码
- **模块分离**: 三个模块相互独立,每个可单独运行调试,输入输出文件明确;上游输出即下游输入
- **依赖**: 只读引用两个参考工程的方案与代码,绝不修改其任何内容
- **性能**: 生产用途,快速计算——角点检测可并行(两路相机、多图像)
- **数据只读**: data/ 目录图像不可修改,所有中间产物和结果写到独立输出目录

<!-- GSD:project-end -->

<!-- GSD:stack-start source:STACK.md -->

## Technology Stack

Technology stack not yet documented. Will populate after codebase mapping or first phase.
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->

## Conventions

Conventions not yet established. Will populate as patterns emerge during development.
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->

## Architecture

Architecture not yet mapped. Follow existing patterns found in the codebase.
<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->

## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, `.github/skills/`, or `.codex/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->

## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:

- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->

## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
