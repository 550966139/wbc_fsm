# G1 23DoF 动作跟踪(dance)策略部署说明

分支 `feature/g1-23dof-dance`,基于 `feature/g1-23dof-port`。本文档是 `State_Tracking23`(23DoF 全身动作跟踪,即跳舞)的部署侧契约与操作说明。该状态与现有速度策略 `State_Policy23` 并存,所有改动均为新增;未提供跟踪配置时,速度/站立/阻尼路径的行为与之前完全一致。**版本状态(在役模型、守卫数值、悬置决策)见 [STATUS.md](STATUS.md),本手册只写"怎么用"。**

训练侧任务:`unitree_rl_mjlab_g1_23dof_tracking_v1`(mjlab tracking 任务)。速度任务与跟踪任务共用同一套机器人配置(相同的 kp/kd、默认关节角、关节限位、`G1_23DOF_ACTION_SCALE`),因此部署侧复用同一 `setTargets23` / `validateCommand` 命令形式,不新增校验形式。

## 目录

1. [策略契约(ONNX)](#1-策略契约onnx)
2. [动作文件格式与训练/数据门禁](#2-动作文件格式与训练数据门禁)
3. [配置](#3-配置)
4. [操作](#4-操作)
5. [安全说明](#5-安全说明)

## 1. 策略契约(ONNX)

- 单输入 float `[1,124]`,单输出 float `[1,23]`;观测归一化嵌入图内(与 v1 速度策略一致,直接喂原始值)。
- 输入维度错误时启动即拒绝(`requireShape`,报错信息指明需要 `[1,124]` / `[1,23]`)。

### 观测布局(124 维,顺序固定)

| 区间 | 维度 | 内容 |
| --- | --- | --- |
| `[0:23]` | 23 | 参考关节位置(当前帧,策略关节序) |
| `[23:46]` | 23 | 参考关节速度(当前帧) |
| `[46:52]` | 6 | 锚点姿态误差 `motion_anchor_ori_b`(见下) |
| `[52:55]` | 3 | 基座角速度(IMU gyro) |
| `[55:78]` | 23 | 关节位置 − 默认值 |
| `[78:101]` | 23 | 关节速度 |
| `[101:124]` | 23 | 上一帧动作 |

实现:`include/common/tracking23.h` 的 `g1::observationTracking23`,段偏移常量为 `kTrackingRefPos/RefVel/Anchor/Gyro/JointPos/JointVel/LastAction`。

### 锚点姿态误差(6 维)

- 机器人锚点(torso_link)世界四元数 = `imu_quat ⊗ Rz(q_waist_yaw)`,waist_yaw 为策略关节索引 12(电机槽位 12),轴向 (0,0,1)。
- 相对姿态 `quat_rel = robot_anchor⁻¹ ⊗ ref_anchor`(与训练侧 `subtract_frame_transforms` 的姿态部分一致),取旋转矩阵前两列按**行主序**展开:`{R[0][0], R[0][1], R[1][0], R[1][1], R[2][0], R[2][1]}`。
- 实现:`g1::anchorError6`;四元数均为 wxyz 约定(`g1::quatMul23` 与 `mathTools.h` 的 `quat_mul` 相同)。

### 入场 yaw 对齐

机器人世界系重力对齐但 yaw 任意,参考动作是 MuJoCo z-up 世界系。`enter()` 时计算机机器人锚点与参考第 0 帧锚点的 yaw 差,整段参考按 `ref_aligned(k) = Rz(yaw_robot − yaw_ref0) ⊗ ref(k)` 逐帧旋转。仅作用于锚点姿态项,关节参考不变。

### 动作

`target = default_joint_pos + a * action_scale`(逐关节 23 维)→ `g1::clampTargets23` 限位截断(训练侧 MuJoCo 在 ctrlrange 截断)→ 以速度策略相同的 kp/kd 经 `setTargets23` 下发。

### 契约 v2(obs-v2 速度反馈,127 维,2026-10-13)

第二个契约 `unitree_rl_mjlab_g1_23dof_tracking_v2`:输入 `[1,127]`,输出不变 `[1,23]`。**同一布局在陀螺前插入 3 维基座线速度(机体系,IMU-site 速度)**,镜像训练侧 actor 顺序(command/anchor_ori/**base_lin_vel**/base_ang_vel/…):

| 区块 | 维度 | 内容 |
| --- | --- | --- |
| `[0:46]` | 46 | 同 v1(参考关节 pos+vel) |
| `[46:52]` | 6 | 锚点姿态误差(同 v1) |
| `[52:55]` | 3 | **基座线速度(估计)** `kTrackingBaseLinVel` |
| `[55:58]` | 3 | 基座角速度(IMU gyro) |
| `[58:81]` `[81:104]` `[104:127]` | 23×3 | 关节位置−默认 / 关节速度 / 上一帧动作(同 v1 后移 3) |

- **双契约规则:一个 config 文件一个契约版本**——`policy_contract` 写 v1 则全 config 必须 124,写 v2 则 127,`observation_size` 必须与契约串一致(v1/v2 模型不可在同一 config 混装)。v1 配置在 v2 代码上线后行为不变(`--check` 复验)。
- **线速度来源 = `g1::BaseStateEstimator23`**(`src/common/base_state_estimator.cpp`,离线冻结算法的 C++ 移植):接触辅助腿式里程计(腿 FK+雅可比、足底 4 候选点择优)+ IMU 互补融合;输入全部来自 `rt/lowstate`(q/dq/**tau_est**/quat/gyro/accel——`IOSDK.cpp` 自 2026-10-13 起捞起 tau_est/ddq 与 IMU rpy),不依赖 ddq(仿真桥不填)。离线精度(以 sim `rt/sportmodestate` 为真值,双采集验证):深蹲窗 vxy RMSE 0.19 m/s、vz 0.09-0.10;训练侧该观测项自带 ±0.5 噪声 DR,估计误差远在其内。
- 动作 CSV 格式不变(50 列)——速度增量是机器人侧估计量,无需新参考数据;`scripts/replay_tracking23.py` 自动按 ONNX 输入宽度适配(v2 理想化重放在速度位插 0)。
- 候选/验收状态(take6-V2 的 P5 决策等)见 [STATUS.md](STATUS.md);可行性论证见 [obs-v2-feasibility.md](obs-v2-feasibility.md);观看脚本 `bash ~/sim2sim_g1/run_view_v2.sh [--band]`。

## 2. 动作文件格式与训练/数据门禁

### 部署 CSV 格式

纯 CSV,无表头,每行一帧(50 Hz,与控制周期对齐,**不做插值**),每行 50 列:

```
ref_joint_pos[23], ref_joint_vel[23], ref_anchor_quat_wxyz[4]
```

加载校验(`src/common/motion_csv.cpp`,`g1::MotionCsv23::load`):文件可读、≥2 帧、每行恰好 50 列、全部有限值、锚点四元数单位模长(±1e-3)。任何违反抛 `std::runtime_error`,报错含文件路径与行号。该文件由训练 NPZ 经 `scripts/npz_to_motion_csv.py` 生成,保证数值与训练逐位一致。光标每个控制周期前进 1 帧(与训练 time_steps 一致)。

### 过渡段规范(2026-09-21 sim2sim 教训,硬性要求)

跟踪训练用参考状态初始化(RSI,episode 直接重置到参考位形),策略从未见过"站姿 → 远距第 0 帧"的过渡——强行进入首拍动作会猛跳(实测 1.72 rad)并被入门禁拒绝。因此**部署动作文件必须以默认站姿开始、以默认站姿结束**:用 `scripts/motion_transitions23.py` 给源 CSV 前补 ~1s 引入段(站姿插值到第 0 帧,根四元数保持第 0 帧值不转 yaw)、后补 ~1s 引出段,再经 `csv_to_npz.py` 转训练 NPZ(**训练与部署用同一份带过渡段的动作**,门禁自然通过、结束平滑交接)。

### 训练/数据三条硬性门禁(全部有失败前科,新动作/新训练必查)

1. **num_envs**:追踪任务基座 `tracking_env_cfg.py` 写死 `num_envs=1`,必须 CLI 覆盖 `--env.scene.num-envs 512`(批量 24→12288/迭代);启动后检查 run 目录 `params/env.yaml` 落盘值。1-env 下 PPO 梯度纯噪声,永远学不会跟踪(前 8 个 run 全废)。
2. **动作物理可行性**:GMR 输出可能脚底悬空(human_scale_table 绕原点缩放、无触地约束)——每个新 NPZ 必查 FK 鞋底净空(`take6_pipeline/ground_motion.py` 修平)。
3. **首尾过渡段**:见上节过渡段规范,训练与部署用同一份带过渡动作。

已知参考数据质量备注:SMPL-X 脚部朝向弱约束导致 ankle_pitch 系统性背屈偏置 ~5°+摆动期幻影背屈(量化病根与修复路线见 [data-quality.md](data-quality.md) §2)。

## 3. 配置

`config/g1_23dof_tracking.json`(严格加载,`include/control/config_tracking23.h`)。**各键的现役取值(哪个模型在哪个槽、守卫/闸数值)以 config 文件为准,快照见 [STATUS.md](STATUS.md)。**

- `policy_contract` 必须为 `unitree_rl_mjlab_g1_23dof_tracking_v1`(124 维)或 `unitree_rl_mjlab_g1_23dof_tracking_v2`(127 维,见 §1 契约 v2 小节);`observation_size` 必须与契约串一致(124/127),`policy_dof` 23、`motor_count` 29、`joint_ids_map` 与 `kPolicyToMotor` 完全一致,否则拒绝。
- `default_joint_pos` / `joint_lower` / `joint_upper` / `stiffness` / `damping` / `action_scale` 与 `config/g1_23dof.json` 相同(共用机器人配置)。
- `model_path`、`motion_path`:相对路径按仓库根解析(与 config23 相同);指向 `model/g1_23dof_tracking/` 下的策略与动作文件。
- `safety.max_target_step` / `safety.max_tracking_error`:TRACKING 状态入场时换入、出场恢复(见第 5 节)。**守卫是逐模型标定值,绝不跨动作共用;换模型必须用 `scripts/replay_tracking23.py --write` 重新标定**,协议与历史数值留痕 [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)。
- `entry_gate.entry_max_joint_error`(0.15 rad)/ `entry_gate.entry_max_joint_vel`(1.0 rad/s):入场门槛(站立/走跑 → 跟踪,与走跑 R2+A 入场同阈值)。
- `motion_end_target`(默认 `"policy23"`):动作播完或 R2+B 中止时的交接目标——`"policy23"` = 过走跑入场门禁后零命令交接给速度策略(门禁不过则回退 FIXEDSTAND 并打印原因);`"stand"` = 回退 FIXEDSTAND(Phase 1 旧行为,调试/bring-up 用)。
- `handoff_settle_timeout`(默认 3.0 s,范围 (0, 30]):motion-end 沉降保持窗——播完后光标冻结在引出段末帧、跟踪策略继续运行,每周期重评走跑入场门禁,通过即交接;超时仍不过则回退 FIXEDSTAND 并打印原因。R2+B 中止路径无沉降窗(舞步中段位姿等也过不了门禁),保持单发判定。
- `handoff_entry_max_joint_error`(默认 0.15 rad = FixedStand R2+A 闸,范围 (0, 1];2026-10-10 新增,per-motion 可覆盖):motion-end/中止交接门禁的**姿态**阈值;`|dq|` 阈值固定 1.0 不变。存在原因:跟踪策略在结尾保持帧上有**确定性的稳态跟踪偏差**(策略侧无法消除——已实测 F2 左肘 0.387 rad、F17c 右肩 yaw 0.672 rad,|dq| 均已收敛 <0.08),按 FixedStand 保持精度设定的 0.15 永远过不了。**换跟踪策略必须用日志实测结尾偏差重标此值。**
- `loop_motion`(默认 false):动作播完是否循环。
- **多动作库 `motions[]`(2026-09-30 起,双舞支持)**:可选数组(1-4 项)替代单一 `model_path`/`motion_path`(两种形式互斥,同时出现即拒绝)。每项必含 `name`(非空、唯一)、`model_path`、`motion_path`,可选携带自己的 `safety` / `entry_gate` / `motion_end_target` / `handoff_settle_timeout` / `handoff_entry_max_joint_error` / `loop_motion` 覆盖块;未覆盖的项继承顶层默认值(顶层的 `safety`/`entry_gate` 因此始终必填)。手柄映射:**R2+B = 第 1 支(保持肌肉记忆),R2+Y = 第 2 支**;入场门禁按所选动作自己的帧 0 与阈值判定,守卫在 `enter()` 换入该动作的值。R2+Y 在跟踪进行中只记日志不切参考(先 R2+B 中止);未配置第 2 支时 R2+Y 仅记日志。`[track:名称]` 1 Hz 快照与 FSM 日志均携带动作名。自动探测要求**所有**已配置动作的模型+动作文件齐备才启用跟踪。

CLI:`--tracking-config <path>` 显式指定(严格:任何缺失/非法即拒绝启动)。未指定时自动探测 `config/g1_23dof_tracking.json` —— **仅当其模型与动作文件同时存在才启用**;否则打印提示并禁用跟踪,行为与之前完全一致。`--check` 模式在启用跟踪时会完整校验:配置解析、动作 CSV 加载、模型存在性、ONNX 形状探针及一次真实推理(不启动 DDS)。

## 4. 操作

推荐操作链(两端都用主动平衡的策略站立,FixedStand 只做引导/兜底):**START → 站立 → R2+A(策略站立,稳定 ≥5 s)→ R2+B/R2+Y 起舞 → 播完后按该动作的 `motion_end_target` 交接**——`"policy23"` 走沉降窗自动交接回策略站立,`"stand"` 回 FixedStand 再手动 R2+A(哪个动作配哪个目标见 [STATUS.md](STATUS.md))。

**⚠️ 结尾走位风险(现场必读)**:部分动作结尾段策略会自平衡迈步、可能向任意方向移动 1-2 m 再站定(take6-F10 仿真实测漂移 0.8-2 m;根因=张臂定格的幅度衰减)。**机器人周围 3 m 半径必须清空**(该风险是否适用于当前动作,以 STATUS.md 的动作特性表为准)。

1. START:阻尼 → 定点站立(插值完成)。
2. R2+A:站立 → 走跑策略(速度策略零命令原地站立是真机+仿真验证过的主动平衡态)。
3. **R2+B:站立 或 走跑 → 跟踪(第 1 支舞)**。门槛:逐关节 `|q − ref_pos[第0帧]| ≤ 0.15` 且 `|dq| ≤ 1.0`;不满足时打印最差关节与原因并保持原状态(与 R2+A 同一模式)。跟踪配置不可用(未提供模型/动作文件)时 R2+B 仅提示,不触发任何故障。走跑中按 R2+B 走同一门禁。
3b. **R2+Y:站立 或 走跑 → 跟踪第 2 支舞**(`motions[]` 配置存在时)。与 R2+B 完全同一门禁逻辑,但对照**第 2 支动作**的帧 0 与其自身阈值;守卫换入第 2 支的标定值。跟踪进行中按 R2+Y 只记日志不切参考(先 R2+B 中止再选);仅配置一支舞时 R2+Y 记日志忽略。仿真键盘:**1=START、2=R2+A、3=L2+B、4=SELECT、5=R2+B、6=R2+Y、8=吊带、9=脱吊**。
4. 播放结束(`motion_end_target="policy23"` 时,`handoff_settle_timeout` 默认 3.0 s):**沉降保持窗**——光标冻结在引出段末帧(站姿),跟踪策略继续运行保持主动平衡,每周期重评走跑入场门禁(`|q − 默认站姿| ≤ handoff_entry_max_joint_error(默认 0.15,per-motion)` 且 `|dq| ≤ 1.0`);通过则**交接给速度策略**(零命令,主动平衡);舞后沉降抖动(短时 |dq|>1)不会让交接"一锤定音"地失败。超时仍不过则回退 FIXEDSTAND 并打印原因行(含已等待时长与最差关节)。`loop_motion=true` 时回卷到第 0 帧继续,不交接。
5. **R2+B(再次):中止**——**无沉降窗**(中止时位姿在舞步中段,等也过不了门禁):单发评估走跑入场门禁,过则 LOCO,否则 FIXEDSTAND。
6. L2+B(锁定阻尼急停)、SELECT/SIGINT(退出)、故障恢复流程与现有 FSM 完全一致,不经过跟踪状态特殊处理。

设计动机:FixedStand 纯位置 PD 无重力前馈,无外部支撑(吊带)不能自保——仿真里脱吊跳完舞回 FixedStand 即倒;生产环境没有吊带。部署动作文件的引出段末帧≈站姿(过渡段规范保证),落在走跑入场门禁内,因此交接给速度策略是最优落地态。

`[track:名称]` 1 Hz 关节快照日志(`f=帧/总数 worst j<i> e<误差> q=<23值>`)使每次真机会话天然可离线分析。

## 5. 安全说明

- **分状态安全限值**:`SafetySupervisor::overrideCommandGuards` 在 TRACKING `enter()` 换入跟踪配置(或该动作 motions[] 覆盖块)的 `max_target_step` / `max_tracking_error`,`exit()` 经 `restoreCommandGuards` 恢复(幂等;状态切换、故障、enter 抛异常等所有出场路径都会经过 `exit()`)。其余限值(关节速度、倾斜阈值、看门狗等)仍由速度配置统一提供。
- 两个 guard 是逐模型回放标定值(理想化开链回放包络 ×1.5)。注意:误差包络来自开链回放的 last_action 发散伪影,闭环仿真实测零守卫压力——**tracking-error 守卫实质空转,operative 保护是 max_target_step**(历次失败候选全部死于步长守卫——F14 的 1.73/2.58 rad 关节 6 步长爆炸即被它拦下)。**换模型必须 `replay_tracking23.py --write` 重新标定**;上机前如需收紧请重新评估。
- 控制回路 40 ms 截止时间、状态/命令看门狗、目标步进/跟踪误差校验(`validateTargets23` 读 `safety.limits()`,即换入后的值)全部生效;推理或观测异常当周期锁定阻尼。
- **交接门禁**:motion-end 沉降窗内每周期重评走跑入场门禁(`g1::entryGate23`,与 FixedStand R2+A/R2+B 同一实现,阈值 0.15 rad / 1.0 rad/s,姿态阈值可被 `handoff_entry_max_joint_error` 覆盖);R2+B 中止为单发评估;超时或不过则回退 FIXEDSTAND 并打印原因行。
- **入场首周期步长基线(2026-09-22 起)**:FSM 每次状态跳转后,`validateTargets23` 的 `previous` 在**首个有效控制周期从当前实测 q 播种**,而不是继承上一状态的最后指令(`CtrlComponents::noteStateEntry`,仅影响首周期,随后恢复指令链)。入场门禁已保证 q 贴近入场姿态,首周期步长因此是真实的指令跳变;这消除了状态入场/载荷切换瞬态的守卫误判(走跑踝膝指令与舞步帧 0 目标相撞、脱吊进舞、舞中载荷突变),且不放宽任何阈值。阻尼后的首指令原本就从实测播种,行为不变。
- 全部动作切换均由遥控器门控,控制器不含任何自主动作。
- **入场淡入(2026-09-24 起)**:`State_Tracking23::enter()` 起 20 个控制周期(0.4 s)内,输出目标从入场实测姿态 smoothstep 混入策略目标(`kTrackingEntryRampCycles`),消除控制器交接的指令跳变(实跳首秒机身摆动 −22%);观测/lastAction 链不受影响,入场门禁语义不变。
- **仿真侧踝蜂鸣(双稳态极限环)运维协议(2026-09-24 实测)**:FixedStand 下踝关节可能陷入 ~81 Hz 自激振荡(踝 dq 峰值 10-16 rad/s),与吊带长度/有无无关,重启实例随机落入安静/振荡盆地,瞬态(锁存/恢复/舞后交接)可激发。它会被入场门禁(|dq|≤1.0)正确拦下——**遇到门禁报荒谬恒定 |dq| 即按此处理**:重启 sim+wbc 抽签,入场前读踝 |dq|<0.5 才算安静盆地;入场一律**带吊按 5 进跟踪、跟踪跑起来后再按 9 脱吊**(避免脱吊站立的倾斜窗口)。踝抖还可能升级为 excessive-velocity 安全锁存(之后按键全哑仅 START 有效)——抖动重试必须整机重启而非重按。该现象为 MuJoCo 接触/PD 共振(高于 50 Hz 控制奈奎斯特),真机踝部动力学不同,真机站立已验证无此现象。

---

*历史决策与复测记录(motion_end_target 的 2026-09-22 决策链、沉降窗与首周期播种的发现过程、F10 首舞等)见 [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) 对应条目;策略链设计动机的完整推演见 "Dance ↔ walk policy chain" 节。*
