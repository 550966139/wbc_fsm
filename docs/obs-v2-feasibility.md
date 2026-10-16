# obs-v2 可行性备忘录（2026-10-12，决策支持文档）

> **状态（2026-10-16 补注）：已落地。** 实际实现为速度反馈子集的 127 维契约 v2（机体系基座线速度，未加参考根速度/位置）：估计器 `src/common/base_state_estimator.cpp` + 双契约运行时已合入主分支（2026-10-13/14，全链 7/7 测试绿）。契约细节见 `tracking23.md` §1"契约 v2"；过程记录见账本 obs-v2 节；验收状态（take6-V2 候选，待 P5 决策）见 `STATUS.md`。§六"若立项"的执行骨架与实际落地基本一致，唯训练变体为 Vel127（仅 `base_lin_vel` 一项入 actor）且 CSV 格式未改。

> 背景：take6-F18/F18a 梯子已证明 F17 腿项在 take6 上**买得到幅度**（深蹲 1.90 rad 跟到 0.005 rad 误差、踢腿顶点 90% vs F10 的 82%），但在深蹲单支撑窗（t≈11 s, f=550）**确定性失去平衡**（0/3 × 两轮，+25k 只改签名不改结局）——能力墙，见 `IMPLEMENTATION_STATUS.md` F18a verdict 节。失败形态是"姿态到位但看不见自己正在倾倒"，正是根状态反馈（obs-v2）针对的类别。本备忘录回答：obs-v2 在本部署上到底可不可行、怎么做、风险在哪。

## 一、结论（TL;DR）

- **训练侧已建好**：任务变体 `Unitree-G1-23Dof-Tracking`（`has_state_estimation=True`）恰好把 `motion_anchor_pos_b`（3 维，参考根位置误差·机体系）+ `base_lin_vel`（3 维）放进 actor 观测（124→130）。训练侧成本 = 翻变体 + DR/入场对齐复核。
- **真机估计的原始输入全部已在现有 DDS 消息里**：`rt/lowstate` 的 IMU 加速度计、`rpy`、电机 `tau_est`、`ddq`——但其中三类当前被 `IOSDK::LowStateHandler` 丢弃（只拷了 q/dq/quat/gyro/accel）。
- **固件里程计路线已死**（硬件验证）：`rt/sportmodestate` 在 ai 模式下静默，而我们的部署要求 motion service 释放——互斥。仿真 rig 上它常流，故 sim 高估了其可用性。
- **难度按维度陡峭分层**：z / vz 便宜；vxy 中等；xy 位置误差高危（业界多数派明确拒绝）。
- **建议**：以速度反馈子集（vxy+vz+参考速度，共 6 维）作为对 F18 失败的最小响应；估计器按 z/vz → vxy →（仅在需要时）xy 自底向上建设；部署侧第一个改动是 `IOSDK.cpp` 捞起 `tau_est`/`ddq`；训练变体从第一天起注入估计器误差 DR；验收专门盯深蹲单支撑窗。

## 二、本地事实（全部核对过文件）

### 2.1 现行契约与版本机制

`include/common/tracking23.h`：124 维 = `[0:46]` 参考关节 pos+vel，`[46:52]` 锚姿态误差 6 维，`[52:55]` 陀螺，`[55:78]` q−默认，`[78:101]` dq，`[101:124]` 上一步动作。**无根位置、无基座线速度、无绝对重力**——仅相对锚姿态与机体系角速度。机器人锚四元数 = `imu_quat ⊗ Rz(q_waist_yaw)`。

版本门三重校验：契约串 `unitree_rl_mjlab_g1_23dof_tracking_v1` + `observation_size==124`（`include/control/config_tracking23.h:63-73`）+ ONNX 形状探测 `[1,124]/[1,23]` + 50 列 CSV 严格加载。任何增量都需要：换 SE 变体重训、契约升 `..._v2`、`kTrackingObsSize`/偏移、config 校验值、下游工具链（`npz_to_motion_csv.py`、`motion_transitions23.py`、`replay_tracking23.py`、`export_from_ckpt.py`——导出已按 obs 维参数化）、测试。

### 2.2 LowState IDL 里比控制器用到的多

`src/interface/IOSDK.cpp` 订阅 `rt/lowstate`（CRC 校验、`mode_machine==4`），但只拷出电机 q/dq、IMU quat/gyro/accel、手柄字节。IDL 实际还有：

- `IMUState_`：`quaternion[4]`、`gyroscope[3]`、`accelerometer[3]`、**`rpy[3]`**——无线速度、无位置。
- `MotorState_`：`q`、`dq`、**`ddq`**、**`tau_est`**——关节力矩/加速度在流、被丢弃（C++ 镜像 `include/message/LowlevelState.h` 有字段、从未填充）。`scripts/cap_hw.py` 已在真机 900 Hz 记录 `tau[29]`，证明字段活着。
- `LowState_`：无足底力、无任何基座位姿。

### 2.3 SDK 客户端与话题——G1 无里程计可读

`unitree_sdk2py` v1.0.1 与 C++ SDK：G1 loco 客户端 getter 只有 FsmId/FsmMode/BalanceMode/SwingHeight/StandHeight/Phase（全命令类之外）；`go2/robot_state` 是服务管理器不是估计器；G1 的 dds_wrapper 只暴露 `rt/lowcmd`、`rt/lowstate`、`rt/arm_sdk`、`rt/inspire/state`、`rt/dex3/*/state`——**本 SDK 构建内 G1 无里程计话题**。对照：go2 的 `SportModeState_` 有 `position[3]/velocity[3]/foot_force[4]`，hg 版只有 `fsm_id/fsm_mode/task_id/task_time`。

### 2.4 决定性硬件事实：部署模式下固件里程计不可用

G1 高层服务运行时会发 go 类型的 `rt/sportmodestate`（含骨盆 `position[0..2]`）——sim/224 的 `f2_{low,sport}.npz` 双主题采集就是这么来的。但台账验尸（line 670）明确：**motion service 释放（ai 模式）后 `rt/sportmodestate` 静默**，而 `IOSDK::start()` 反向要求 motion service 不活跃才发布 lowcmd——两者在真 G1 上互斥。该话题只能用于 ai 模式下的事后分析，永远进不了闭环。

### 2.5 训练侧不对称与数据路径

`model/g1_23dof_tracking/env.yaml`：actor 项 = command(46)+锚姿态(6)+角速度(3)+q(23)+dq(23)+动作(23)=124，部署逐项复现。**Critic 特权项已含 `motion_anchor_pos_b`、`body_pos`、`body_ori`、`base_lin_vel`**——训练环境本来就在算 obs-v2 要的那些量，奖励也在罚 actor 看不见的东西（`motion_global_root_pos` std 0.3 / `motion_body_lin_vel` w 1.0）。动作 NPZ 带世界系机体阵列（`body_quat_w` 已验证、`body_lin_vel_w` 已确认；`body_pos_w` 键名需在改列清单前从上游 `csv_to_npz.py` 核对）。部署 CSV 现为 50 列（`src/common/motion_csv.h:16`）——加根轨迹列是转换器扩展，不是重新重定向。

## 三、业界实践

| 系统 | actor 用根线速度？ | 用相对根位置？ | 硬件来源 |
|---|---|---|---|
| PHC (ICCV'23) | 师特权/生估计 | 师特权 | 纯仿真；teacher→student Transformer 估计器（RMA 式） |
| H2O (2024) | 是 | — | **动捕** |
| OmniH2O (2024) | **否——刻意移除**；25 步历史 | 否 | 无；VIO 线速度变体**摔**，学习式 MLP/GRU 估计器**劣于**不用 |
| ExBody (RSS'24) | **否——"真机特权信息"** | 否（仅偏航差） | 按设计无估计器 |
| BeyondMimic (SR'25) | 是（IMU twist ℝ⁶） | **是（9 维锚误差 = 位置+Rot6D）** | 依赖机器人状态估计器（自认局限） |
| HOVER (ICRA'25) | 学生不用 | 否 | `[q,q̇,ω,g]`×25 步历史蒸馏；在 G1 上板载运行 |
| unitree_rl_mjlab（本任务血统） | 否 | 否 | 任务名就叫 `...-No-State-Estimation` |

四种原型：(a) 接触辅助 Kalman/EKF——经典但无 surveyed WBC 跟踪器在环内使用；(b) 学习式估计器——显式（PHC）或隐式历史（HOVER/OmniH2O），OmniH2O 消融表明显式估计器可能有害；(c) 固件里程计——dev 模式静默，死路；(d) 不要根位置、接受漂移——多数派，BeyondMimic 是最强反例。

来源：OmniH2O arXiv:2406.08858；ExBody arXiv:2402.16796；HOVER arXiv:2410.21229；BeyondMimic arXiv:2508.08241；PHC arXiv:2305.06456；H2O arXiv:2406.17732；RMA arXiv:2107.04034；接触辅助 InEKF arXiv:1809.06529。

## 四、六个维度的分层可行性

| 增量 | 传感/估计器 | dev 模式可用？ | 部署数据路径 | 风险 |
|---|---|---|---|---|
| **(v) 参考根速度/位置（CSV 侧）** | 无（纯数据） | N/A | CSV 50→56 列（NPZ 世界系根 pos+vel）；扩 `npz_to_motion_csv.py` + `motion_transitions23.py`（首尾过渡段需像冻结根四元数一样插值根位置） | **低** |
| **z 误差 + vz** | 接触足 FK + IMU roll/pitch（quat/rpy 已收） | **是，零新 DDS** | FSM 内估计器组件；`tracking23.h` 加偏移 | **低-中**（最少漂移的维度；训练里 z 本就有专属终止项） |
| **vxy（机体系）** | 接触辅助腿式里程计（支撑检测用 `tau_est`/`ddq`——**在 IDL 里、被 IOSDK 丢弃，需捞起** + 足 FK）；或从 critic 标签蒸馏学习式估计器 | **是**——原始输入全在流；`cap_hw.py` 证明 `tau_est` 900 Hz 活 | IOSDK 扩展 + 估计器；obs +3 | **中**——OmniH2O 证据：坏估计比没有更糟；足滑（实测 slide p90 0.19 m/s）是精度地板 |
| **xy 位置误差** | 由 vxy 积分 + 入场锚定（现有入场偏航对齐 `ref_aligned(k)=Rz(Δyaw)⊗ref(k)` 是天然模板，扩为平移参考到入场位置） | 无固件源（§2.4） | 同上；obs +2 | **高**——单调漂移进策略；业界多数派拒绝；BeyondMimic 先例需接受估计器质量依赖 |
| 固件里程计 | — | **否（硬件验证静默）** | — | 部署侧**死路** |

## 五、与 F18 结果的综合

F18 的失败（幅度买到、深蹲单支撑确定性失稳）是**平衡反馈形状的失败，不是幅度形状的失败**——深蹲单腿时机器人欠驱动度最高，恢复由基座速度与 CoM/根偏移主导，恰是 124 维契约删掉的信号。推论：

1. **速度反馈子集（vxy+vz+参考速度）是对 F18 的最小响应**：喂进平衡回路而不引入单调漂移的位置项；也是唯一零新增 DDS 依赖（只需捞起 `tau_est`/`ddq`）的路径。
2. **xy 位置误差与 F18 修复解耦**，仅在速度反馈不足时再加；入场锚定模板保证架构上随时可加。
3. **估计器质量是新失败面**：必须在 sim rig（`sportmodestate` 在 sim 里常流 = 免费真值）上先验证，再上真机；训练从第一天注入估计器误差 DR（噪声/延迟），哲学同现有 `motion_anchor_ori_b` ±0.05 / `base_ang_vel` ±0.2。

## 六、若立项：建议的执行骨架

1. 训练：翻 `Unitree-G1-23Dof-Tracking` 变体（130 维），对 6 个新维度加噪声/延迟 DR；warm start 不可跨维度（124→130 需从头/加宽层）——成本高于 F18 的热启动。
2. 部署：`IOSDK.cpp` 捞 `tau_est`/`ddq`/`rpy` → FSM 内估计器（先 z/vz 后 vxy）→ 契约 v2（`tracking23.h` 偏移、config、ONNX 探测）→ `npz_to_motion_csv.py` 56 列 → 全链测试。
3. 验收：sim 双主题真值下先过估计器精度关；舞蹈验收专门盯深蹲单支撑窗（F18 的死点）；真机上用 1 Hz `[track]` 日志 + 接触锚定 FK 重建做替代真值（sport 采集为空）。
