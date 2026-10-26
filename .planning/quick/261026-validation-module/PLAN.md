---
type: quick
status: pending
priority: high
created: 2026-10-26
source: 用户需求「加一个测量精度功能:用当前双目内外参测棋盘格最远两点距离,与理论值求偏差,评估使用一段时间后参数是否漂移(振动等)、哪些区域精度如何,快速定位问题」;数据已放 data/val(1 组同步对)
---

# Quick Task: 新增模块4「测量精度验证」(stereo-calib validate)——系统健康检查

## 定位与原理

消费**冻结的现役标定产物**(intrinsics.json ×2 + extrinsics_stereo.json,绝不重拟合),
对 data/val 的同步图像对测棋盘对角(固定角点 0↔87,与 SC3 同口径)距离,与理论值求偏差。

**为什么双模式(本功能的核心价值)**:
- **冻结模式(健康判据)**:P1=K1[I|0]、P2=K2[R|t](存档 R/t)→ 去畸变 → DLT 三角化 →
  对角距离偏差。参数一漂(振动/碰撞/热漂)这里立刻变大。
- **逐对模式(归因诊断)**:每对 PnP 相对位姿 → DLT → 距离偏差(自洽,会吸收外参漂移)。
  判读矩阵:冻结大 + 逐对小 → **外参漂移**(重标 stereo 即可);两者都大 → 内参退化或
  数据质量(跑 intrinsic 体检定位);两者都小 → 系统健康。
- 报告必须把这个判读矩阵写给产线操作工(中文,说人话)。

**能力边界(报告注明)**:本工具测的是「与标定时刻的一致性」,对**绝对尺度盲**——理论值
继承棋盘格边长假设(当前 25mm,用户卡尺复核待做);漂移监测看的是相对变化,不受此影响。

## 算法规格(float64)

1. **装载**:两路 intrinsics.json(distortion_model 感知,M8 12 参)+ extrinsics_stereo.json
   的 rvec/t(mm)→ R=cv2.Rodrigues(rvec)。config 新节 `validation:` 指定 data_dir。
2. **角点检测**:data/val/<group>/cam{1,2}.jpg 走**既有 preprocess 检测链**
   (detect_one/canonicalize 同一代码路径,禁止重写检测——角点行列朝向消歧必须与
   训练口径一致)。val 组数不定(当前 1 组),min_pairs 配置守卫(默认 1)。
3. **逐对测量**(两种模式都做):
   - 去畸变:models.unproject(牛顿逆,M8 感知);
   - 冻结模式:P1=[I|0]、P2=[R|t] → cv2.triangulatePoints 或既有 DLT(evaluate.py
     的 triangulate_dlt 复用)→ farthest_corners_distance(固定 0/87,复用)→
     err_frozen = d − theory;
   - 逐对模式:PnP 各自相机 → 相对位姿(R2R1ᵀ / t2−R21t1)→ DLT → err_pair;
   - theory = true_board_distance(objp)(复用)。
4. **区域分析**:按板中心(角点均值)在 cam1 视场分桶(region_grid 配置,默认 3×3),
   逐桶冻结模式 |err| 的 n/median/max。组少时如实标注「样本不足,仅列单对」。
5. **判定闸门**(全配置化,fail-closed 与仓库风格一致):
   - `distance_bias_max_mm`(|mean err_frozen|,默认 1.0mm——现役噪声 0.20mm RMS 的 ~5×)
   - `distance_rms_max_mm`(默认 2.0mm)
   - verdict FAIL → 报告标红 + **CLI 退出码 1**(report/JSON 先落盘留痕,与体检防线同风格);
   - 逐对模式不设闸门(诊断量,不判生死)。
6. **产物**(output/validation/,运行前清目录照 D-11 体例):
   - `validation.json`:契约新文件自由定义,但字段齐备(per-pair 行:group_id/
     err_frozen_mm/err_pair_mm/板中心坐标/重投影参考;regions;gates;verdict;参数指纹:
     内参 fx 与 baseline‖t‖ 写入,便于跨时间对比参数本身);
   - `report.md`:中文,含判读矩阵说明、逐对表、区域表、参数指纹、verdict。
7. **CLI**:`stereo-calib validate` 子命令(cli/main.py @app.command,中文 help),
   异常走现有 CLI 异常网收口。

## 配置(io/config.py + config.yaml)

```yaml
validation:
  data_dir: data/val            # 验证图像对根目录(组子目录 cam1.jpg/cam2.jpg)
  distance_bias_max_mm: 1.0     # |平均偏差| 红线(漂移判据)
  distance_rms_max_mm: 2.0      # RMS 红线
  region_grid: [3, 3]           # 区域分桶(行,列);[0,0] 关闭区域分析
  min_pairs: 1                  # 最少验证对数
```
dataclass ValidationConfig;解析带类型守卫(非数值/非法组合中文 ValueError);
data_dir 与 data.train/test/output_root 的包含关系校验沿用 load_config 既有防线风格。

## 测试

1. 合成双目场景(已知 K/D8/R/t + 理想角点)→ 冻结/逐对模式 err ≈ 0(<1e-6 mm 量级);
2. **漂移分离性测试(关键)**:给存档 t 注入 +1.5mm 扰动 → 冻结模式 err 显著变大
   (≥1mm 量级)、逐对模式仍 ≈0——锁死「冻结大/逐对小 = 外参漂移」的判读依据;
3. config 守卫单测;verdict/退出码路径单测(pipeline 级,tmp 路径);
4. 角点检测复用路径:用 train 既有 npz 中一对真实角点跑测量函数(不重新检测);
5. 现有 273 测试零修改全绿;ruff 干净(存量 5 处 Phase 1 遗留不修)。

## 实证验收(生产数据)

`uv run stereo-calib validate` 跑 data/val 现有 1 组:
- 预期 err_frozen 与 err_pair 都在 ±0.5mm 内(系统昨天刚定稿,若无漂移);
- 如实报告两个数值与判读;若超闸门 → 停,报告数值,不调阈值。
- data/ 只读防线:train/test 逐文件 md5 与基线一致(val/ 为用户新增目录,不修改其中文件,
  也不往 data/ 写任何东西;新基线指纹如实记录)。

## Constraints

- 不修改 data/ 任何既有文件(输出只写 output/);参考工程只读
- 复用优先:检测链/models.unproject/evaluate 的 DLT 与 farthest_corners_distance/
  true_board_distance 全部复用,禁止第二份数学
- 阈值/网格/路径全配置化;报告与 JSON 中文;fail-closed 退出码风格与体检一致
- 原子提交:feat(04)(模块+CLI+config+测试)、docs(SUMMARY/REQUIREMENTS 登记)
- GSD 纪律:偏差如实记 SUMMARY

## Tasks

1. ValidationConfig + config.yaml 节 + 解析守卫(含测试)
2. validation 包:装载/检测复用/双模式测量/区域分析/闸门 verdict(含单元测试)
3. 产物落盘(validation.json + report.md)+ CLI validate 子命令 + 退出码(含管线测试)
4. 合成场景测试(零误差 + 漂移分离性)
5. 生产验证:data/val 实跑,数值回报;REQUIREMENTS 登记 + SUMMARY + README 备份节补一行
   (功能入口,不动定稿数值)
