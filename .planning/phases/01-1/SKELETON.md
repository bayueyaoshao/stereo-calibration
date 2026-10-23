# Walking Skeleton — 双目相机标定生产工具 (stereo-calibration)

**Phase:** 1
**Generated:** 2026-10-23

## Capability Proven End-to-End

一条只传配置文件路径的命令 `uv run stereo-calib preprocess`(命令行零算法参数)对图像数据完成扫描 → SB 角点检测(输出必过 canonicalize)→ 每相机每数据集带掩码稠密 npz + usable/fail CSV + meta.json 落盘;全程 data/ 只读,产物全部写入配置指定的输出根目录。

(01-01 计划交付此骨架;01-02 在同一骨架上补 classic 兜底与并行,01-03 补可视化/公共视场对/全量生产运行——三者不改骨架契约,只增加内容。)

## Architectural Decisions

| Decision | Choice | Rationale |
|---|---|---|
| CLI 框架 | typer 多子命令 `stereo-calib preprocess/intrinsic/stereo` | D-09;参考工程1 同款;命令行只收 config 路径 |
| 工程形态 | uv + hatchling + src layout,包名 `stereo_calib`,Python >=3.12 | D-12;与参考工程1 完全同构,移植成本最低 |
| 依赖 | opencv-python >=4.14,<5.0 / numpy 2.5.* / scipy 1.18.* / typer / pyyaml / joblib >=1.4;uv.lock 入库 | 参考工程 uv.lock 逐项一致 + PyPI 验证 + slopcheck 全 OK;绝不走系统 pip(清华镜像滞后) |
| 配置 | 单一 config.yaml 分段(全局 board/data/output_root/cv2_threads + preprocess 模块段),dataclass 加载层中文校验 | D-10 / PREP-01 / INFRA-02;非法值在入口收口 |
| 数据契约(npz) | 每相机×数据集一个 `cam{n}_{split}.npz`,带掩码稠密格式:image_ids/paths/uv/objp/source/detected/sq_px/win_size 八数组,未检出整行 NaN,读取一律 allow_pickle=False;objp[row*cols+col]=(col*25,row*25,0) 与 canonicalize 后 uv 同序 | D-04 + RESEARCH Pattern 7;行对齐 CSV,Phase 2/3 一行代码过滤 detected;定长 unicode 不需要 pickle |
| 清单契约(CSV) | usable_cam{n}_{split}.csv / fail_{split}.csv / common_fov_pairs_{split}.csv,utf-8-sig(Excel 直开) | D-06;服务人工复核筛选排序 |
| 输出组织 | output_root/{train,test}/ 分目录,npz+CSV+meta+visualization/+fail_originals/;固定根目录每次覆盖、运行前清理重建,无时间戳 | D-05 / D-11 / Pitfall 7 |
| 角点检测 | 参考工程1 detect/window/canonical 三件套移植:SB 主路线(严禁二次亚像素)+ classic 兜底(窗口逐帧实测推导、缺陷闸门、阶梯重提);三条硬不变量由测试锁定 | PREP-02;本数据实证 SB/classic 原始序差 355.9px,canonicalize 必须保留 |
| 图像身份 | image_id = 叶子目录名 + "_cam{n}"(过 ^[A-Za-z0-9_.-]+$ 白名单);group_id = 叶子目录名 | stem 全库只有 cam1/cam2 必碰撞(Pitfall 3);group_id 是公共视场配对键 |
| 并行 | joblib loky + TaskFailed 异常信封 + inner_max_num_threads 配额;max_workers=1 串行回退 | PREP-02;参考工程真实事故定制品,单帧失败不杀整批 |
| 部署/运行方式 | 本地 CLI(uv run stereo-calib …),无网络、无服务、无 DB | 离线生产标定工具;每次刚性支架数据只算一次 |

## Stack Touched in Phase 1

- [x] Project scaffold(uv + hatchling + src layout + pytest + ruff + uv.lock)
- [x] CLI — 一个真实子命令端到端(preprocess;intrinsic/stereo 预留占位)
- [x] 数据层 — 真实图像读取(cv2.imread 只读)+ 真实结构化产物写入(npz + CSV + meta.json)
- [x] 算法层 — 真实角点检测(SB + classic 兜底,不变量测试锁定)
- [x] 本地全流程运行命令:`uv run stereo-calib preprocess`(全量 406 张并行约 30-60s)
- (无 UI/DB/部署要求——离线 CLI 工具,按 orchestrator 指示不适用)

## Out of Scope (Deferred / Never)

- intrinsic 与 stereo 子命令实现 → Phase 2 / Phase 3(骨架已留位)
- 自动剔除/排除清单机制 → 不做(D-03:用户人工复核后另行处理)
- 输出时间戳版本化 → 不做(D-11:固定根目录直接覆盖)
- cv2.stereoCalibrate / 鱼眼模型 / 图像采集 / 在线标定 → 见 REQUIREMENTS.md Out of Scope
- 环境注意:系统 pip 指向清华镜像(numpy 滞后),一切依赖操作只走 uv;系统 Python 3.10 不可用,uv 管 3.12

## Subsequent Slice Plan

每个后续阶段在本骨架上加一个垂直切片,不改变上述架构决策:

- Phase 2: `stereo-calib intrinsic` — 消费 cam{n}_{train}.npz(detected 行)→ M2 模型标定 + scipy 精化 → 内参文件 + test 集 float64 RMS
- Phase 3: `stereo-calib stereo` — 消费 common_fov_pairs + 公共视场角点 + 固定内参 → PnP-median + 精化求 R/T → test 集最远角点距离残差 RMS
