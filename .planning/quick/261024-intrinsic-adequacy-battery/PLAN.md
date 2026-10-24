---
type: quick
status: pending
priority: high
created: 2026-10-24
source: 用户指令「增加进去,保证下次你能判定内参数是否可靠」(2026-10-24 晨;根因分析见 .planning/MORNING-REPORT-2026-10-24.md 与会话记录)
---

# Quick Task: 内参模型充分性体检(model-adequacy battery)fail-closed 接入内参管线

## Description

新增 `src/stereo_calib/intrinsic/diagnostics.py`:内参求解完成后自动体检畸变模型是否充分,**仅凭自家数据 + 自家结果**判定内参是否可靠,不依赖任何参考工程。

**为什么需要(根因背景)**:2026-10-24 晨确认产线 M2 五参畸变模型对本相机(8mm C-mount 4MP)欠拟合——cam1 fx 被推高 11%(2265 vs 数据自证真值 ~2037),双目阶段才以「逐对相对位姿假散布 12.16°」暴露。而单相机标量重投影 RMS 0.14px 完全看不出:每帧自由位姿 + fx 会吸收模型误差。已用本仓实测数据验证四个有效检查(/tmp/model_adequacy_battery.py,M2 双机全亮红 / rational8 全绿):

| 检查 | 量 | cam1 M2 | cam2 M2 | rational8 对照 |
|---|---|---|---|---|
| A0 残差能量 | PnP 逐点池化 RMS / 中心桶噪声底 | 6.3× | 2.3× | ~1× |
| A1 残差径向结构 | 按归一化半径 5 桶径向分量均值 max\|mean\| | 0.12px | 0.077px | ≤0.0035px |
| A3 切片 fx 稳定性 | 6 切片重标 fx 极差 % | 2.36% | 1.28% | 0.18%/0.32% |
| A4 模型阶次探针 | 内部升阶 rational8 重标,fx 跳变 % | 11.2% | 0.74% | — |

**判定规则:A0/A1/A3/A4 任意两项红 → verdict「畸变模型不充分,内参不可靠」(fail-closed)。**

## 算法规格(全部 float64 口径)

输入:某相机 train 角点 `uv (F,88,2)`、`objp (88,3)`(只取 detected=True)、产线解出的 `K (3,3)`、`D (5,)`、图像尺寸 (2560,1440)。中心 C=(W/2,H/2),半对角 half_diag=hypot(W/2,H/2)。

**逐点残差(公共前置)**:每帧 `cv2.solvePnP(objp.float32, uv.float32, K, D, SOLVEPNP_ITERATIVE)` → `cv2.projectPoints` 回代 → `res = uv - proj`,shape (F,88,2)。

- **A0 残差能量**:`pooled = sqrt(mean(||res||²))`(全点);`floor = sqrt(mean(||res||²))` 限于 r_norm<0.2 的点(r_norm=|uv−C|/half_diag);`ratio = pooled/floor`。红:> `a0_max_ratio`。
- **A1 残差径向结构**:r_norm ∈ [0,0.2,...,1.0) 5 桶;桶内 `radial_mean = mean(res·û)`,û=(uv−C)/|uv−C|;取 5 桶 max|radial_mean|。红:> `a1_max_bin_mean_px`。报告记录全部桶均值(带符号)。
- **A2 去畸变直线度(仅报告,不判红)**:每帧 `cv2.undistortPoints(uv, K, D, None, K)` → 按 objp 坐标分组行(同 Y,11 点)/列(同 X,8 点)→ 每线 SVD 拟合取正交偏差 max → 每帧取最弯线 → 报中位数。实测该统计量无分辨力(M2 0.20 vs R8 0.15),仅作信息项。
- **A3 切片 fx 稳定性**:按每帧板中心(角点均值)切 6 子集:u<C.x / u≥C.x / v<C.y / v≥C.y / |板中心−C|<median / ≥median。每子集 `cv2.calibrateCameraExtended(objp.float32×n, 子集角点.float32, (W,H), K0=产线K, D0=产线D, flags=CALIB_USE_INTRINSIC_GUESS)` → 记 fx。极差% = (max−min)/mean×100(只统计 n≥`a3_min_frames` 的子集;有效子集 <4 → 记「数据不足」note 不计红)。红:> `a3_max_fx_range_pct`。
- **A4 模型阶次探针**:全量数据同协议升阶重标:同上但 flags=`CALIB_RATIONAL_MODEL|CALIB_USE_INTRINSIC_GUESS` → fx8。跳变% = |fx产线 − fx8|/fx8×100。红:> `a4_max_fx_jump_pct`。探针拟合失败(cv2 异常/不收敛)→ 记 note 不计红。

**verdict**:`red_count = sum(各检查 red)`,≥ `min_red_count`(默认 2)→ unreliable。

**性能预算**:141 帧全电池 ≤2 分钟(本晨实测);A3 每子集拟合与 A4 探针均一次性 cv2 调用,无需 scipy。

## 实现要点

1. **配置**(`io/config.py` + `config.yaml`):`intrinsic:` 下新增 `diagnostics:` 节
   ```yaml
   diagnostics:
     enabled: true            # 体检总开关(默认 true;false 时跳过并在报告注明)
     a0_max_ratio: 2.0        # A0 红:池化 RMS / 中心桶噪声底 上限(倍)
     a1_max_bin_mean_px: 0.02 # A1 红:径向分量桶均值绝对值上限(px)
     a3_max_fx_range_pct: 0.5 # A3 红:切片 fx 极差上限(%)
     a3_min_frames: 20        # A3 单切片最少帧数(低于则跳过该切片)
     a4_max_fx_jump_pct: 2.0  # A4 红:升阶 fx 跳变上限(%)
     min_red_count: 2         # verdict 判不可靠所需最少红灯数(判定规则「任意两项」)
   ```
   dataclass `IntrinsicDiagnosticsConfig`,解析带类型守卫(数值型配置收到非数值给中文 ValueError,与 WR-01 同风格)。阈值依据:rational8 实测通过值(A1 0.0035 / A3 0.32 / A0 ~1)留 3~5× 裕量。
2. **模块** `diagnostics.py`:纯函数 + 一个入口 `run_battery(uv, objp, K, D, image_size, cfg) -> DiagnosticsReport`(dataclass 或 dict:各检查数值、阈值、red 布尔、note 列表、verdict、red_count)。不 import models.py 的 M2 契约(A3/A4 探针拟合直接用 cv2);不改产线内参模型。
3. **管线接入** `intrinsic/pipeline.py`:每路相机求解成功后(run_intrinsic 现有双路编排内)调用体检;结果:
   - `intrinsics.json` 新增顶层键 `diagnostics`(契约**只增不改**):各检查数值/阈值/red/notes/verdict/red_count/enabled;
   - report.md 新增「模型充分性体检」中文小节:判定规则一行 + 逐检查数值表格 + verdict;
   - verdict=不可靠 → 报告醒目标红(`**FAIL**`/`⚠` 标记),`run_intrinsic` 结束后抛中文 RuntimeError(如「内参模型充分性体检未通过(cam1/cam2:红灯 A1,A3——畸变模型不充分,内参不可靠;禁止消费该内参进入双目标定)」),由 CLI 现有异常网收口为「内参标定失败:…」退出码 1(intrinsics.json/report.md 已落盘留痕,与双目防线同风格:产物诚实留痕 + 非零退出)。两路都求解但只有一路不可靠:同样 fail(报告注明哪路)。
4. **测试**(tests/test_intrinsic_diagnostics.py 新文件):
   - 每检查独立单测:合成小数据(如 12~20 帧、已知 K/D)验证数值口径与红绿阈值边界;
   - **合成端到端双向**:① 用 rational 畸变(K=(2037,2037,1280,720),D8 含显著高阶项如 k1=-0.2,k2=0.15,k6=0.05 + p1,p2,25mm 11×8 网格,~24 个多样位姿,z 600~1200mm、xy 铺开到画面边缘)投影出无噪+微噪(σ≈0.05px)角点 → 用 M2 协议拟合 → 体检**必须判不可靠**(若红灯不亮,加大高阶系数/位姿幅度调到亮,这是设计检测已知欠拟合的测试);② 用 M2 一致畸变(D5)合成 → 拟合 → 体检**必须判可靠**;
   - verdict 规则单测:构造恰好 1 红/2 红的报告验证 min_red_count 行为;enabled=false 跳过路径。
   - 现有 214 测试保持全绿。
5. **实证验收(生产重跑)**:`uv run stereo-calib intrinsic` → cam1/cam2 当前 M2 结果**必须被判不可靠**(exit 1,report.md 体检节红灯,intrinsics.json 含 diagnostics),产物落主仓 output/intrinsic/。这就是「下次能自己揪出来」的验收。
6. **登记**:`.planning/REQUIREMENTS.md` Phase 2 增补验收条目(模型充分性体检闸门,引用判定规则);`.planning/phases/03-3/deferred-items.md` 登记「rational8 产线支持 + Phase 2 重标 + Phase 3 重跑」待办(体检判红后的出路,待用户决策)。

## Constraints

- 不改 data/(md5 a183699da79875fc63f7cbfe75874398 不变)
- intrinsics.json / npz 契约**只增字段**,不改不删既有字段
- 不改 models.py 产线 M2 契约(INTRINSIC_NAMES、D shape 断言、9 参布局)——rational8 产线支持是另一个任务
- 全部阈值/开关进 config.yaml,禁止硬编码
- ruff 干净;原子提交 `feat(02)` 前缀(模块/配置/管线/测试可合并为 2~3 个原子提交)
- 全部测试绿:`uv run pytest -q`(当前 214)

## Tasks

1. `IntrinsicDiagnosticsConfig` + config.yaml 节 + 解析与类型守卫(含测试)
2. `diagnostics.py` 五检查 + verdict(含逐检查单测)
3. 管线接入:JSON/report 落盘 + fail-closed 退出(含管线级测试)
4. 合成端到端双向测试(rational 合成→不可靠;M2 合成→可靠)
5. 生产重跑实证验收(双机判不可靠)+ REQUIREMENTS/deferred-items 登记 + SUMMARY.md

## Success Criteria

- [ ] `uv run pytest -q` 全绿(214 + 新增,预期 ≥222)
- [ ] 合成双向测试:rational 畸变数据判不可靠、M2 畸变数据判可靠
- [ ] `uv run stereo-calib intrinsic`:cam1/cam2 均判「不可靠」,退出码 1,report.md 含体检红灯表格,intrinsics.json 含 diagnostics 键,产物落 output/intrinsic/
- [ ] 判定规则可配置(min_red_count),阈值全部出自 config
- [ ] data/ md5 不变;JSON 契约只增不改;ruff 干净;原子提交
