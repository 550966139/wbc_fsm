# 当前状态快照（STATUS）

> **本文件是项目现状的唯一落点**：在役部署、分支、进行中战役、悬置决策、已知问题。
> 历史细节与复现命令查 [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)（追加式账本）；"怎么用"查手册层文档。
> 快照日期：2026-10-16（与账本末条 / `feature/g1-23dof-dance` HEAD `6d1eadd` 对齐）。

## 目录

1. [在役部署矩阵](#1-在役部署矩阵)
2. [代码分支状态](#2-代码分支状态)
3. [进行中的战役](#3-进行中的战役)
4. [悬置的用户决策](#4-悬置的用户决策)
5. [已知问题与设施坑](#5-已知问题与设施坑)
6. [候选与回滚件清单](#6-候选与回滚件清单)
7. [维护约定](#7-维护约定)

## 1. 在役部署矩阵

| 位置 | 走跑（速度策略） | 舞蹈 slot 0（R2+B） | 舞蹈 slot 1（R2+Y） |
|---|---|---|---|
| repo（`feature/g1-23dof-dance`） | v1（81 obs） | take6-F10 | **take2-F17c**（2026-10-12 起） |
| 224（`~/sim2sim/wbc_fsm`） | 同 repo | take6-F10 | **take2-F17c**（2026-10-12 起） |
| PC2（机器人 aarch64） | 同 repo | take6-F10 | **take2-F2**（2026-10-09 起，F17c 换装待现场会） |

速度策略要点：契约 v1，单输入 `[1,81]` / 命令 4 维（pitch 维恒 0），限幅 `[0.3, 0.3, 0.4, 0.25]`；守卫 `max_target_step` 1.0 / `max_tracking_error` 1.5（`config/g1_23dof.json`）。硬件验证 2026-09-18 站立+行走首胜。

舞蹈槽位关键数值（**以 `config/g1_23dof_tracking.json` 为准**，本表是快照）：

| 动作 | 模型/动作文件 | 守卫 step/error | motion_end_target | handoff 闸 | 已知特性 |
|---|---|---|---|---|---|
| take6-F10 | `policy.onnx`（take6F10fullending@100000）+ `motion.csv`（1832 帧 36.6 s） | 1.5 / 5.5（顶层继承） | `"stand"`（2026-09-22 决策） | — | 结尾走位 1-2 m，**3 m 半径清场**；硬件 3/3（2026-09-28） |
| take2-F17c | `policy_take2f17c_200000.onnx` + `motion_take2f2_115000.csv`（共享 take2 链，2907 帧 58.14 s） | 1.9 / 4.3 | `"policy23"` | **0.8**（结尾稳态偏差 0.672 rad r_shoulder_yaw） | MPJPE 0.21-0.24 / turn1 推力 73% 为已接受代价（决策 (a)） |
| take2-F2（PC2 在役 / repo 回滚） | `policy_take2f2_115000.onnx` + 同上 motion | 1.7 / 4.0 | `"policy23"` | 0.5 | 硬件首舞 4/4 全绿（2026-10-09） |

- 电机槽位 `[0..12, 15..19, 22..26]`；29DoF 旧模型绝不截断复用。
- 操作链（现役）：START → R2+A（策略站立，稳 ≥5 s）→ R2+B/R2+Y 起舞 → take2 舞完沉降→自动交接→策略站立 / take6 舞完回 FixedStand 再手动 R2+A。详见 [tracking23.md](tracking23.md) §4。

## 2. 代码分支状态

```
main (b352409, 2026-04-29)                      ← 上游 WBC_Deploy 原版 29DoF
  └─ feature/g1-23dof-port (fccaf75, 2026-09-20) ← 23DoF 移植 + 速度策略（走跑）
       └─ feature/g1-23dof-dance (6d1eadd, 2026-10-16) ← 当前：双舞 + obs-v2 + MC 工具，含全部走跑代码
```

线性链、无平行分支；走跑与舞蹈**同分支**，运行时靠 FSM 状态 + 两套配置区分。本地与 origin 同步，工作区干净。

## 3. 进行中的战役

### 3.1 数据质量线（当前主线，权威路线图 [data-quality.md](data-quality.md)）

- **训练线冻结**（2026-10-15 用户决策）：交付物 = 机器判定合格的参考数据（物理不变量、最坏帧门禁）；解冻条件 = 双舞数据全过全部不变量。
- 当前卡点：**project_g1 v3 不变量闭环——设计已完成，待实现**（四角贴地 ±2 mm 硬目标 + 每步复测回炉）。
- 后续优先级：分级平滑治抖动 → 双舞全过 → take1 新视频零人工全链（泛化终测）。

### 3.2 take2-F20 验收链（已训完，验收未做）

- F20/F20b 双种子**已训完**（model_200000 + onnx 就位；修正跑 `2026-10-15_08-07-36_take2F20v4`，暖启 F17b@175000，r5 参考）。
- r5 参考重建背景：用户"抽帧"投诉定案为参考数据手臂瞬移（GMR 盆地翻转 + GVHMR 肘跳变）；r5 修复后跳变行 190→33、max 0.824→0.406 rad。
- 待做链条：导出核对 → 守卫/handoff 闸重标（从日志结尾偏差）→ wsk 仿真验收（**MPJPE ≤0.18 线重开**——F17 阶梯"联合不可达"结论是对瞬移参考测的）→ 用户目检 → 换槽决策。

## 4. 悬置的用户决策

| 决策 | 内容 | 影响 |
|---|---|---|
| **P5（take6-V2）** | take6-V2 验收 3/3、深蹲墙已破（F18 死点消失），但 MPJPE 0.222-0.228 超预注册线 0.20（+0.02-0.03），与 F17c 被接受的交换同类 | 接受 → 需 take2-v2 同变体重训（~13h GPU）才能组双 v2 配置（一 config 一契约版本）；拒绝 → 保留 F10。在役零改动 |
| **F20 换槽** | F20/F20b 验收通过后是否换下 F17c（repo+224+PC2） | 验收链走完后决策 |
| **PC2 F17c 换装** | repo+224 已 F17c，PC2 仍 F2 | 有人值守现场会：RUNBOOK 键链更新 + 吊带先跑全链 + live A/B vs F2（换 motions[1] 配置重启） |

## 5. 已知问题与设施坑

- **224 无头显示栈断供**：Xvfb 假活（pgrep 在但 MuJoCo `monitor != NULL` 断言崩），重建 X 服务配方未定 → **仿真验收/观看一律 wsk 本机**（`~/sim2sim_g1/`）；F16 的部署-sim FK 指标（drift/float）被阻塞。
- **采集负载扰动**：940 Hz 双采集使 F10 在 f=600 深蹲窗 2/2 锁存 → 验收采集降载（CAP_STRIDE=2 / 减字段），彻底方案待定。
- **wsk /tmp 有清理器**：暂存/构建一律放 `~/sim2sim_g1/` 持久路径（/tmp 一日两清有前科）。
- **224 训练四坑**（每次启动必查清单）：任务必须 `Unitree-G1-23Dof-Tracking-No-State-Estimation`（SE 变体是 130 obs，导出即废）；`--agent.max-iterations` 是**增量**语义；暖启必带 `--agent.resume True`（否则静默从头训）；`pkill -f` 经 ssh 自匹配（按 PID 杀）。
- sdk2py 单进程双订阅采集段错误 → 必须双进程。
- 固件里程计 `rt/sportmodestate` 在 ai 模式静默（与 lowcmd 路径互斥）→ 部署侧死路，只用于事后分析。
- relay 脚本 224 重启即死（查 `f20b_relaunch.log`）；wsk known_hosts 丢条目曾致轮询器误报"训练已死"（先 ssh 复核再动作）。
- 台账遗留待办：机器人固件版本号待首次在场会话记录；PC2 无持久 journal（取证靠控制器自带日志）。

## 6. 候选与回滚件清单

**候选（未上役）**：

| 候选 | 状态 | 关键指标 | 工件位置 |
|---|---|---|---|
| take6-V2（127 obs） | 验收 3/3，**待 P5 决策** | 膝 std 0.915-0.918（最佳）/ 深蹲墙破 / MPJPE 0.222-0.228 / 守卫 1.0/3.5 | wsk `~/sim2sim_g1/v2_staging/` + 224 `take6_pipeline/` |
| take6-F16 | 已判决，提名下次真机会话候选 | 踢腿 92.1%（历史最高）/ 恶劣角折叠 −95% / 守卫 1.1/3.7 | wsk `~/sim2sim_g1/f16_staging/` |
| take2-F20/F20b | 训完待验收（见 §3.2） | — | 224 runs + wsk（r5 参考 `tools/fidelity23/data/refs/`） |
| take6-F13 / F15 | 方向性成功未换版（F13 结尾修复 / F15 R6+配方） | F13 踢腿 73%；F15 顶点 77% | 224 `/tmp/f13_deploy/`、`/tmp/f15_deploy/`（/tmp 易失，可能已清） |
| take6-F18/F18a | 否决（深蹲能力墙，由 obs-v2 取代） | 幅度买到（深蹲 0.005 rad）但 f=550 确定性锁存 | wsk `~/sim2sim_g1/f18_staging/` |

**回滚件（全部在 repo `model/g1_23dof_tracking/` 及 `.bak` 集）**：take2-F2 全套（换回 = motions[1] 改回 F2 条目 1.7/4.0 + 闸 0.5）；take2-F1r4；take6-F9 `.bak` 集；速度策略 v0（`policy_v0_80obs.onnx` + `model_4900.pt`）。PC2 侧另有换装前 tarball（`~/wbc_fsm_backup_predual_20261009.tar.gz` 等）。

## 7. 维护约定

- 战役结束 / 换装 / 用户决策落地 → **更新本文件对应小节** + 账本追加条目（真实命令+结果）。
- 手册层文档（tracking23.md / g1-23dof-port.md / safety-supervisor.md）只写"怎么用"，不写"现在是什么版本"——版本状态只在本文件。
- 数值以 config JSON 为准；本文件表格是快照不是配置源。
- 本索引层约定见 [README.md](README.md)。
