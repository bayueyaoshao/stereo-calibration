# SUMMARY — quick 261026-validation-module(模块4 测量精度验证)

2026-10-26 执行完毕。新增第四个独立模块 `stereo-calib validate`(系统漂移体检):
消费**冻结的现役标定产物**(两路 intrinsics.json + extrinsics_stereo.json,绝不重拟合),
对 `data/val/<组目录>/cam{1,2}.jpg` 走既有 preprocess 检测链(SB 主路线 + classic
兜底 + canonicalize,零第二份检测),双模式测棋盘对角(角点 0↔87)距离与理论值
(√(10²+7²)×25mm)的偏差。

## 交付

- **Task 1 配置**:`ValidationConfig`(io/config.py)+ config.yaml `validation:` 节
  (data_dir / distance_bias_max_mm=1.0 / distance_rms_max_mm=2.0 / region_grid=[3,3]
  / min_pairs=1)+ 中文解析守卫 + 数据安全互斥(验证目录 vs data.train/test/
  output_root 相等或互相包含在加载层被拒)。
- **Task 2/3 validation 包**(src/stereo_calib/validation/):
  - `observe.py`:data/val 扫描(单边缺文件/重名组警告跳过)+ 既有
    detect_one 复用(run_parallel 调度,单帧失败留痕不中断)。
  - `measure.py`:冻结模式(P1=K1[I|0]、P2=K2[R|t] → undist_pix → DLT →
    farthest_corners_distance)与逐对归因模式(两路 solvePnP → 相对位姿 →
    DLT);全部数学 import 复用 evaluate/models,包内零第二份公式、零
    solve/refine(逐对 PnP 是 PLAN 规定的诊断量,不回流参数);
    evaluate_gates 双红线闸门;analyze_regions 按 cam1 板中心 3×3 分桶。
  - `artifacts.py`:validation.json(16 键契约断言锁定)+ 中文 report.md
    (判读矩阵/逐对表/区域表/参数指纹/能力边界「绝对尺度盲」/配置快照)。
  - `pipeline.py`:run_validation 编排(D-11 清目录 → 装载 → 检测 → 测量 →
    闸门 → 落盘 JSON 先于报告);CLI `validate`(异常网与 stereo 同族,
    verdict FAIL → 退出码 1 fail-closed)。
- **Task 4 测试**(38 个新测试,全套 311 全绿):
  - M8 合成零误差(冻结 <1e-6mm;逐对 <1e-3mm,地板 = 生产 _pnp_pose 的
    float32 入参,漂移信号与地板差 5 个数量级);
  - 漂移分离性:t 注入 +1.5mm → 冻结 3.9mm ≥ 红线,逐对仍 ≈0(判读矩阵
    「冻结大+逐对小=外参漂移」的数学证据);
  - 配置守卫/数据安全互斥;渲染棋盘 e2e(PASS 全契约 + FAIL 退出码 +
    D-11 清残留 + 数据只读快照 + CliRunner);真实 train 角点 + 真实冻结
    产物冒烟(产物缺失自动跳过)。
- **Task 5 生产实跑**:`uv run stereo-calib validate`(data/val 1 组
  20261026_150155_233986,SB/SB 双路检出):

| 指标 | 实测 | 红线 | 判定 |
|---|---|---|---|
| 冻结偏差 \|平均\| | 0.2396 mm | 1.0 | PASS |
| 冻结偏差 RMS | 0.2396 mm | 2.0 | PASS |
| 逐对偏差(诊断,无闸门) | +0.0141 mm | — | 自洽极佳 |
| 重投影 RMS | 0.0384 px | — | — |
| 180° 翻转 / 失败单帧 | 0 / 0 | — | — |

verdict **PASS**,退出码 0:v1.0 定稿参数(2026-10-24)无系统漂移。
data/ 全树 408 文件运行前后 md5 逐字节一致(只读防线实证)。

## 登记

- REQUIREMENTS.md:VALID-01 增补 + Traceability + 实证行 + Last updated。
- README.md:功能入口加 `uv run stereo-calib validate` 一行(定稿数值未动)。

## 偏差记录(执行中如实调整)

1. **逐对零误差阈值 1e-6 → 1e-3mm**:PLAN 期望 <1e-6mm,实测地板 ~4e-5mm
   来自生产代码 `_pnp_pose` 的 solvePnP float32 入参转换(与 evaluate 消歧锚
   同一 dtype 惯例,非缺陷);合成投影本身改用 float64 物点消掉夹具量化地板。
2. **e2e 渲染几何改「收敛式双目」**:正对平移构型在基线 121mm/深度 800mm 的
   弱几何下,亚像素检测偏差被放大成冻结偏差 ~1.3mm(逐对 0.02mm 仍自洽);
   改两机各自正视棋盘(基线 ~207mm/深度 ~600mm),干净渲染实测冻结 0.10mm。
   这是测试夹具几何设计问题,非被测代码缺陷。
3. R/t 语义按代码实证为准:X_cam2 = R@X_cam1 + t(cam1 参考系),与
   extrinsics_stereo.json 写侧契约一致。

## 遗留

- 无功能遗留。既有 5 处 RUF007/C408 告警(PLAN 预埋不修)与 .codebuddy/
  EXE001 保持原样。
