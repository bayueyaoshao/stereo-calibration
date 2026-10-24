---
type: quick
status: complete
priority: high
created: 2026-10-24
completed: 2026-10-24
source: 用户直接下达(内参模型充分性体检 + fail-closed 接入内参管线)
plan: 261024-intrinsic-adequacy-battery
---

# Quick Task Summary: 内参模型充分性体检(A0/A1/A3/A4)fail-closed 接入

**一句话:** 内参管线新增独立体检模块——五项检查量化「M2 畸变模型是否配得住这台相机」,生产实证双机均判「不可靠」(cam1 红 A1/A3/A4、cam2 红 A1/A3),产物留痕后 fail-closed 拦截退出码 1,不可靠内参从此进不了双目标定。

## 背景

Phase 2 内参标量指标很好看(cam1/cam2 test RMS 0.1407/0.1361px 远小于 1.0),但双目 Phase 3 暴露 M2 对本相机欠拟合(cam1 fx 虚高约 11%)。标量 RMS 无法区分「噪声底」与「模型失配」,需要一套模型充分性体检把这件事变成可判定、可拦截的闸门。

## 交付内容(对应 PLAN Tasks,提交按序)

| 提交 | 内容 |
|---|---|
| `0ec36e4` | 体检模块 `src/stereo_calib/intrinsic/diagnostics.py`(A0 残差能量比 / A1 残差径向结构 / A2 去畸变直线度仅报告 / A3 切片 fx 稳定性 / A4 升阶 rational8 探针,红检数 ≥ min_red_count 判不可靠)+ `IntrinsicDiagnosticsConfig` 解析与类型守卫 + config.yaml diagnostics 节(阈值全进配置,禁硬编码) |
| `2406c97` | 管线接入:定稿求解后体检,`intrinsics.json` 增 diagnostics 键、`report.md` 增体检表格(契约只增不改);verdict 不可靠 → 先落盘全部产物再抛中文 RuntimeError,CLI 收口退出码 1(fail-closed) |
| `f2bb429` | 30 个测试(阈值接线/verdict 边界/严格 JSON/输入防线)+ 合成端到端双向 + 管线级 fail-closed/CLI 退出码;REQUIREMENTS 增补 INTR-05;deferred-items 登记后续待办 |

算法口径借鉴参考实现 `/tmp/model_adequacy_battery.py` 但不 import,按产线风格重写:float64 回代、frozen dataclass 配置、中文注释与报错、类型守卫(先判 bool 再判 int)。

## 验收结果

### 1. 合成端到端双向(测试内,48 视图,σ≈0.05px,z 600~1200mm、xy 铺到画面边缘)

- rational8 真值 `[k1,k2,p1,p2,k3,k4,k5,k6]=[-0.3,0.2,1e-4,-1e-4,-0.05,0.35,0,0]` → M2 拟合 → **必判不可靠**(红灯 3 项:A1≈2.93px / A3≈29.0% / A4≈4.57%,含 A1)
- M2 一致真值 → M2 拟合 → **必判可靠**(0 红灯,A3≈0.079%,距阈值 0.5% 有 6 倍裕量)

### 2. 生产实证(`uv run stereo-calib intrinsic`,主仓真实数据)

| 项 | cam1 | cam2 |
|---|---|---|
| A0 残差能量比(阈 2) | 1.4730 通过 | 1.0578 通过 |
| A1 残差径向结构(阈 0.02px) | **0.0220 红灯** | **0.0772 红灯** |
| A2 去畸变直线度(仅报告) | 0.1852 | 0.1835 |
| A3 切片 fx 稳定性(阈 0.5%) | **3.6567 红灯** | **1.2795 红灯** |
| A4 模型阶次探针(阈 2%) | **11.2404 红灯** | 0.7395 通过 |
| verdict | **不可靠**(红 A1,A3,A4) | **不可靠**(红 A1,A3) |
| train/test RMS(px) | 0.5645 / 0.1407 | 0.2389 / 0.1361 |
| train 幸存帧 | 116/141 | 155/155 |

- 退出码 **1**;中文报错点名 cam1/cam2 红灯项并声明「禁止消费该内参进入双目标定」
- 留痕:`output/intrinsic/report.md` 模型充分性体检节(双机逐项表格)、`output/intrinsic/cam{1,2}/intrinsics.json` diagnostics 键
- 与 PLAN 背景表逐项吻合:A4 cam1≈11%(fx 虚高实证)、cam2≈0.7%;A1/A3 双机同红

### 3. 质量闸门

- 全量 `uv run pytest -q`:**244 passed, 0 failed**(开工基线 214,零回退)
- `uv run ruff check .`:本任务全部新建/改动文件干净;仅剩 6 个存量告警(EXE001×1 指向未跟踪目录 .codebuddy、RUF007×1 + C408×4 均为 Phase 1 旧测试文件),已登记 deferred-items,按 Scope Boundary 不修

## 偏差与决策留痕(最保守选项)

1. **A0 背景数值口径**:PLAN 背景表「6.3×/2.3×」实为参考脚本里 M2/rational8 **pooled RMS 对最优模型的倍数**;按 PLAN 算法规格定义(pooled RMS / 中心桶 r_norm<0.2 噪声底)实测 cam1=1.47、cam2=1.06。按规格实现,verdict 不变(双机仍 ≥2 红灯)。
2. **rational 合成真值系数**:k6-only 真值(如 k6=0.05)会被 M2 的 k3 完全吸收(分母 1+k6r⁶ 展开只贡献 r⁶ 项),红灯不亮;按 PLAN 预案加大高阶项,实验确定 **k4=0.35** 量级不可吸收,端到端红灯稳定。
3. **视图数**:24 视图时 A3 全部切片不足 a3_min_frames=20 帧被跳过记「数据不足」;改用 **48 视图** 激活 A3,合成双向与阈值接线测试均基于 48 帧。
4. **A2 实测无分辨力**:双机 A2≈0.18px,对 M2 欠拟合不敏感,维持 PLAN 规格「仅报告不判红」,报告注明实测结论。
5. **data/ 只读指纹方法更正**:`find data -name '*.jpg' | xargs md5sum | md5sum`(未排序)在文件系统返回顺序变化下**不可复现**(连续三次三个不同值),此前记录的基线即此伪差异;改用排序版 `find … | sort | xargs md5sum | md5sum` = `8a430c9395e91f256923b2a6eabb4354`(两次复测一致)。内容级证据:无任何 jpg 的 mtime 晚于本任务 PLAN.md(今日创建),最新 mtime 为采集日 10-23 17:08——**data/ 未被触碰**。

## 后续待办(已登记 `.planning/phases/03-3/deferred-items.md`,待用户决策)

1. 产线支持 rational8 畸变模型(扩 models.py 或新增 M8 契约);
2. rational8 重标内参(Phase 2 重标,体检复核应转绿);
3. 内参就绪后 Phase 3 双目标定重跑。

## 测试

- `tests/test_intrinsic_diagnostics.py`(新增,30 个):配置解析 3+8 参数化;green 只读断言;阈值接线全 4 检查改阈值必红;A3 数据不足/数值路径;A4 探针失败注入;A0 中心桶空;输入防线参数化;disabled;verdict 边界直测;严格 JSON;e2e 双向;管线级 fail-closed/disabled/绿灯全链/CLI exit 1
- 受契约影响:`tests/test_intrinsic_pipeline.py`(JSON 键集 + roundtrip 传 diagnostics)、`tests/test_stereo_pipeline.py`(种子夹具补 diagnostics 占位)

---
*completed 2026-10-24 · 提交 0ec36e4 / 2406c97 / f2bb429*
