# G1 23DoF 动作跟踪(dance)策略部署说明

分支 `feature/g1-23dof-dance`,基于 `feature/g1-23dof-port`。本文档是 `State_Tracking23`(23DoF 全身动作跟踪,即跳舞)的部署侧契约与操作说明。该状态与现有速度策略 `State_Policy23` 并存,所有改动均为新增;未提供跟踪配置时,速度/站立/阻尼路径的行为与之前完全一致。

训练侧任务:`unitree_rl_mjlab_g1_23dof_tracking_v1`(mjlab tracking 任务)。速度任务与跟踪任务共用同一套机器人配置(相同的 kp/kd、默认关节角、关节限位、`G1_23DOF_ACTION_SCALE`),因此部署侧复用同一 `setTargets23` / `validateCommand` 命令形式,不新增校验形式。

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

## 2. 动作文件格式(部署侧)

纯 CSV,无表头,每行一帧(50 Hz,与控制周期对齐,**不做插值**),每行 50 列:

```
ref_joint_pos[23], ref_joint_vel[23], ref_anchor_quat_wxyz[4]
```

加载校验(`src/common/motion_csv.cpp`,`g1::MotionCsv23::load`):文件可读、≥2 帧、每行恰好 50 列、全部有限值、锚点四元数单位模长(±1e-3)。任何违反抛 `std::runtime_error`,报错含文件路径与行号。该文件由训练 NPZ 经 `scripts/npz_to_motion_csv.py` 生成,保证数值与训练逐位一致。光标每个控制周期前进 1 帧(与训练 time_steps 一致)。

**过渡段规范(2026-09-21 sim2sim 教训,硬性要求)**:跟踪训练用参考状态初始化(RSI,episode 直接重置到参考位形),策略从未见过"站姿 → 远距第 0 帧"的过渡——dance1 第 0 帧距站姿 0.97 rad,入门禁必然拒绝,强行进入首拍动作猛跳 1.72 rad。因此**部署动作文件必须以默认站姿开始、以默认站姿结束**:用 `scripts/motion_transitions23.py` 给源 CSV 前补 ~1s 引入段(站姿插值到第 0 帧,根四元数保持第 0 帧值不转 yaw)、后补 ~1s 引出段,再经 `csv_to_npz.py` 转训练 NPZ(训练与部署用同一份带过渡段的动作,门禁自然通过、结束平滑交接 FixedStand)。

## 3. 配置

`config/g1_23dof_tracking.json`(严格加载,`include/control/config_tracking23.h`):

- `policy_contract` 必须为 `unitree_rl_mjlab_g1_23dof_tracking_v1`;`observation_size` 124、`policy_dof` 23、`motor_count` 29、`joint_ids_map` 与 `kPolicyToMotor` 完全一致,否则拒绝。
- `default_joint_pos` / `joint_lower` / `joint_upper` / `stiffness` / `damping` / `action_scale` 与 `config/g1_23dof.json` 相同(共用机器人配置)。
- `model_path`、`motion_path`:相对路径按仓库根解析(与 config23 相同)。指向 `model/g1_23dof_tracking/policy.onnx` 与 `model/g1_23dof_tracking/motion.csv`(**2026-09-29 起 = take6F10fullending@100000 部署版**:`unitree_rl_mjlab_g1_23dof_tracking_v1` 契约、R4 手术版参考 + **真人真实结尾**(GMR 暖启动锁死修复后的 1832 帧 36.6 s 版本,首尾站姿过渡 1s/3s;数据链见 `docs/IMPLEMENTATION_STATUS.md` "P0 real-ending recovery" 节)。F9 回滚件 `*_take6f9r4_90000*.bak`、F13(结尾保真候选,踢腿 73%)暂存 224 `/tmp/f13_deploy/`)。
- `safety.max_target_step` / `safety.max_tracking_error`:TRACKING 状态入场时换入、出场恢复(见第 5 节)。当前为 take6F10fullending@100000 的标定值(**1.5 / 5.5**;换模型必须重新标定,协议与数值留痕 `docs/IMPLEMENTATION_STATUS.md`)。
- `entry_gate.entry_max_joint_error`(0.15 rad)/ `entry_gate.entry_max_joint_vel`(1.0 rad/s):入场门槛(站立/走跑 → 跟踪,与走跑 R2+A 入场同阈值)。
- `motion_end_target`(默认 `"policy23"`):动作播完或 R2+B 中止时的交接目标——`"policy23"` = 过走跑入场门禁后零命令交接给速度策略(门禁不过则回退 FIXEDSTAND 并打印原因);`"stand"` = 回退 FIXEDSTAND(Phase 1 旧行为,调试/ bring-up 用)。**现役配置(2026-10-10 起):take6-F10 slot 0 保持 `"stand"`(2026-09-22/23 决策,其结尾稳态偏差未按新闸复测),take2-F2 slot 1 改为 `"policy23"`**(sim 6/6 验证:舞完→沉降→交接→速度策略脱带站立;take2 末帧距默认站姿仅 0.014 rad)。
- `handoff_settle_timeout`(默认 3.0 s,范围 (0, 30]):motion-end 沉降保持窗——播完后光标冻结在引出段末帧、跟踪策略继续运行,每周期重评走跑入场门禁,通过即交接;超时仍不过则回退 FIXEDSTAND 并打印原因。R2+B 中止路径无沉降窗(舞步中段位姿等也过不了门禁),保持单发判定。
- `handoff_entry_max_joint_error`(默认 0.15 rad = FixedStand R2+A 闸,范围 (0, 1];2026-10-10 新增,per-motion 可覆盖):motion-end/中止交接门禁的**姿态**阈值;`|dq|` 阈值固定 1.0 不变。存在原因:跟踪策略在结尾保持帧上有**确定性的稳态跟踪偏差**(实测 F2 左肘 0.387 rad、F17c 右肩 yaw 0.672 rad,|dq| 均已收敛到 <0.08)——按 FixedStand 保持精度设定的 0.15 永远过不了。**换跟踪策略必须用日志实测结尾偏差重标此值**(现役 take2-F2 = 0.5;候选 F17c 需 0.8)。
- `loop_motion`(默认 false):动作播完是否循环。
- **多动作库 `motions[]`(2026-09-30 起,双舞支持)**:可选数组(1-4 项)替代单一 `model_path`/`motion_path`(两种形式互斥,同时出现即拒绝)。每项必含 `name`(非空、唯一)、`model_path`、`motion_path`,可选携带自己的 `safety` / `entry_gate` / `motion_end_target` / `handoff_settle_timeout` / `handoff_entry_max_joint_error` / `loop_motion` 覆盖块——**守卫是逐模型标定值,绝不跨动作共用**;未覆盖的项继承顶层默认值(顶层的 `safety`/`entry_gate` 因此始终必填)。手柄映射:**R2+B = 第 1 支(保持肌肉记忆),R2+Y = 第 2 支**;入场门禁按所选动作自己的帧 0 与阈值判定,守卫在 `enter()` 换入该动作的值。R2+Y 在跟踪进行中只记日志不切参考(先 R2+B 中止);未配置第 2 支时 R2+Y 仅记日志。`[track:名称]` 1 Hz 快照与 FSM 日志均携带动作名。自动探测要求**所有**已配置动作的模型+动作文件齐备才启用跟踪。

CLI:`--tracking-config <path>` 显式指定(严格:任何缺失/非法即拒绝启动)。未指定时自动探测 `config/g1_23dof_tracking.json` —— **仅当其模型与动作文件同时存在才启用**;否则打印提示并禁用跟踪,行为与之前完全一致。`--check` 模式在启用跟踪时会完整校验:配置解析、动作 CSV 加载、模型存在性、ONNX 形状探针及一次真实推理(不启动 DDS)。

## 4. 操作

链路(**现役 2026-10-10 起:take2-F2 舞完自动交接速度策略;take6-F10 舞完回 FixedStand 再手动 R2+A**;推荐操作链两端都用主动平衡的策略站立,FixedStand 只做引导/兜底):**START → 站立 → R2+A(策略站立,稳定 ≥5 s)→ R2+B/R2+Y 起舞 → take2:播完沉降→自动交接→策略站立 / take6:播完回站立**。F10 的 1 Hz 关节快照日志 `[track] f=帧/总数 worst j<i> e<误差> q=<23值>`(2026-09-28 起,双舞格式 `[track:名称] f=…/N`)使每次真机会话天然可离线分析。

**⚠️ F10 结尾走位(已知行为,现场必读)**:结尾段(入场后 t≈31-35 s)策略会自平衡迈步、可能向任意方向移动 1-2 m 再站定(仿真实测漂移 0.8-2 m;根因=张臂定格的幅度衰减,改善中,F13 候选已压到 0.83 m)。**机器人周围 3 m 半径必须清空**。

1. START:阻尼 → 定点站立(插值完成)。
2. R2+A:站立 → 走跑策略(速度策略零命令原地站立是真机+仿真验证过的主动平衡态)。
3. **R2+B:站立 或 走跑 → 跟踪(第 1 支舞)**。门槛:逐关节 `|q − ref_pos[第0帧]| ≤ 0.15` 且 `|dq| ≤ 1.0`;不满足时打印最差关节与原因并保持原状态(与 R2+A 同一模式)。跟踪配置不可用(未提供模型/动作文件)时 R2+B 仅提示,不触发任何故障。走跑中按 R2+B 走同一门禁。
3b. **R2+Y:站立 或 走跑 → 跟踪第 2 支舞**(2026-09-30 起,`motions[]` 配置存在时)。与 R2+B 完全同一门禁逻辑,但对照**第 2 支动作**的帧 0 与其自身阈值;守卫换入第 2 支的标定值。跟踪进行中按 R2+Y 只记日志不切参考(先 R2+B 中止再选);仅配置一支舞时 R2+Y 记日志忽略。仿真键盘:**1=START、2=R2+A、3=L2+B、4=SELECT、5=R2+B、6=R2+Y、8=吊带、9=脱吊**。
4. 播放结束(`motion_end_target="policy23"` 时,`handoff_settle_timeout` 默认 3.0 s):**沉降保持窗**——光标冻结在引出段末帧(站姿),跟踪策略继续运行保持主动平衡,每周期重评走跑入场门禁(`|q − 默认站姿| ≤ handoff_entry_max_joint_error(默认 0.15,per-motion)` 且 `|dq| ≤ 1.0`);通过则**交接给速度策略**(零命令,主动平衡);舞后沉降抖动(短时 |dq|>1)不会让交接"一锤定音"地失败。超时仍不过则回退 FIXEDSTAND 并打印原因行(含已等待时长与最差关节)。`loop_motion=true` 时回卷到第 0 帧继续,不交接。
5. **R2+B(再次):中止**——**无沉降窗**(中止时位姿在舞步中段,等也过不了门禁):单发评估走跑入场门禁,过则 LOCO,否则 FIXEDSTAND。
6. L2+B(锁定阻尼急停)、SELECT/SIGINT(退出)、故障恢复流程与现有 FSM 完全一致,不经过跟踪状态特殊处理。

设计动机:FixedStand 纯位置 PD 无重力前馈,无外部支撑(吊带)不能自保——仿真里脱吊跳完舞回 FixedStand 即倒(见台账 E1@3000 条目);生产环境没有吊带。部署动作文件的引出段末帧≈站姿(实测末帧与默认站姿最大偏差 0.107 rad),恰落在走跑入场门禁内,因此交接给速度策略是最优落地态。

## 5. 安全说明

- **分状态安全限值**:`SafetySupervisor::overrideCommandGuards` 在 TRACKING `enter()` 换入跟踪配置的 `max_target_step` / `max_tracking_error`,`exit()` 经 `restoreCommandGuards` 恢复(幂等;状态切换、故障、enter 抛异常等所有出场路径都会经过 `exit()`)。其余限值(关节速度、倾斜阈值、看门狗等)仍由速度配置统一提供。
- 两个 guard 已按 take6F10fullending@100000 部署模型标定(2026-09-28):`max_target_step` **1.5**、`max_tracking_error` **5.5**(回放包络 0.934/3.640 ×1.5)。注意:误差包络来自理想化开链回放的 last_action 发散伪影,闭环仿真实测零守卫压力;**tracking-error 守卫仍实质空转,operative 保护是 max_target_step**(历次失败候选全部死于步长守卫——F14 的 1.73/2.58 rad 关节 6 步长爆炸即被它拦下)。标定协议与数值留痕 `docs/IMPLEMENTATION_STATUS.md`,上机前如需收紧请重新评估。
- 控制回路 40 ms 截止时间、状态/命令看门狗、目标步进/跟踪误差校验(`validateTargets23` 读 `safety.limits()`,即换入后的值)全部生效;推理或观测异常当周期锁定阻尼。
- **交接门禁**:motion-end 沉降窗内每周期重评走跑入场门禁(`g1::entryGate23`,与 FixedStand R2+A/R2+B 同一实现,阈值 0.15 rad / 1.0 rad/s);R2+B 中止为单发评估;超时或不过则回退 FIXEDSTAND 并打印原因行。
- **入场首周期步长基线(2026-09-22 起)**:FSM 每次状态跳转后,`validateTargets23` 的 `previous` 在**首个有效控制周期从当前实测 q 播种**,而不是继承上一状态的最后指令(`CtrlComponents::noteStateEntry`,仅影响首周期,随后恢复指令链)。入场门禁已保证 q 贴近入场姿态,首周期步长因此是真实的指令跳变;这消除了状态入场/载荷切换瞬态的守卫误判(走跑踝膝指令与舞步帧 0 目标相撞、脱吊进舞、舞中载荷突变),且不放宽任何阈值。阻尼后的首指令原本就从实测播种,行为不变。
- 全部动作切换均由遥控器门控,控制器不含任何自主动作。
- **入场淡入(2026-09-24 起)**:`State_Tracking23::enter()` 起 20 个控制周期(0.4 s)内,输出目标从入场实测姿态 smoothstep 混入策略目标(`kTrackingEntryRampCycles`),消除控制器交接的指令跳变(实跳首秒机身摆动 −22%);观测/lastAction 链不受影响,入场门禁语义不变。
- **仿真侧踝蜂鸣(双稳态极限环)运维协议(2026-09-24 实测)**:FixedStand 下踝关节可能陷入 ~81 Hz 自激振荡(踝 dq 峰值 10-16 rad/s),与吊带长度/有无无关,重启实例随机落入安静/振荡盆地,瞬态(锁存/恢复/舞后交接)可激发。它会被入场门禁(|dq|≤1.0)正确拦下——**遇到门禁报荒谬恒定 |dq| 即按此处理**:重启 sim+wbc 抽签,入场前读踝 |dq|<0.5 才算安静盆地;入场一律**带吊按 5 进跟踪、跟踪跑起来后再按 9 脱吊**(避免脱吊站立的倾斜窗口)。该现象为 MuJoCo 接触/PD 共振(高于 50 Hz 控制奈奎斯特),真机踝部动力学不同,09-18 真机站立已验证无此现象。

## 6. 状态与经验教训(2026-09-22 更新)

Phase 2/3 已完成:take6g512@8000(num_envs=512 修复后)sim2sim 全项通过——全程 1565 帧零守卫触发、motion-end 自动回站立、二次进入/中止/急停/恢复全过。训练与数据的三条门禁(全部有失败前科):

1. **num_envs**:追踪任务基座 `tracking_env_cfg.py` 写死 `num_envs=1`,必须 CLI 覆盖 `--env.scene.num-envs 512`(批量 24→12288/迭代);启动后检查 run 目录 `params/env.yaml` 落盘值。1-env 下 PPO 梯度纯噪声,永远学不会跟踪。
2. **动作物理可行性**:GMR 输出可能脚底悬空(human_scale_table 绕原点缩放、无触地约束)——每个新 NPZ 必查 FK 鞋底净空(`take6_pipeline/ground_motion.py` 修平,take6 修正 −0.058 m)。
3. **首尾过渡段**:训练用 RSI,策略没见过站姿→远距帧 0 的过渡;动作文件必须首尾各接 ~1 s 站姿过渡(`scripts/motion_transitions23.py`,引入段保持帧 0 根四元数不转 yaw),训练与部署用同一份带过渡动作。

已知数据质量瑕疵(可接受):SMPL-X 脚部朝向弱约束导致 ankle_pitch 系统性背屈偏置 ~5°+摆动期幻覚背屈,已 1:1 传入参考但在限位内;尾部 2.7s GMR 手臂朝向 target 与 G1 运动学冲突导致的前倾已随 trim880 裁除(机制与复现见台账)。

剩余待办:Phase 4 有人值守硬件验证(PC2,流程见 `SIM2SIM_HANDOFF.md` 与 `RUNBOOK_ONSITE.md`),舞蹈章节:release_ai→L1+X→START→R2+B。

**舞蹈↔走跑策略链(2026-09-22,用户拍板)**:motion-end 与 R2+B 中止默认(`motion_end_target="policy23"`)过走跑入场门禁后零命令交接给速度策略,不再回 FixedStand(脱吊跳完即倒的缺陷链由此闭合);走跑中 R2+B 亦可直接进舞蹈(同一帧 0 门禁)。交接/入场门禁统一为 `g1::entryGate23` 单一实现。部署动作文件实测:帧 0 距站姿 0.018 rad、末帧 0.107 rad,均在 0.15 门禁内。

**策略链 224 复测修复(2026-09-22,agent-5 实测驱动)**:复测暴露两问题,均不靠放宽阈值解决,已修复。(A) motion-end 门禁"一锤定音":舞后沉降抖动(|dq| 短时 >1)+ 残余误差让单帧门禁判负(0/3 交接成功,每次安全回退 FixedStand;沉降后手动 R2+A 可进,证明只需等沉降)——改为**沉降保持窗**:播完后光标冻结在引出段末帧、跟踪策略继续运行,每周期重评门禁,通过即交接;超过 `handoff_settle_timeout`(默认 3.0 s)仍不过才回退 FixedStand。R2+B 中止无沉降窗(单发判定不变)。(B) 步长守卫对入场瞬态盲视:E2@7000(步长护栏 0.6)脱吊四连败全部是入场/载荷切换瞬态(走跑进舞 joint10 步 0.68、脱吊进舞 0.95、舞中载荷突变 0.73),而 E1@3000 的 1.0 护栏恰好容纳——标定协议(理想 replay)对瞬态盲视,不是策略回退。修法:FSM 状态跳转后**首个控制周期 `previous` 从当前实测 q 播种**(`CtrlComponents::noteStateEntry`),只影响首周期,随后恢复指令链;入场门禁已保证 q 贴近入场姿态,首周期步长即真实指令跳变。两项修复待 224 再复测确认。

**阶段状态(2026-09-29 收官)**:在役 = **F10(真人结尾版)**,仓库+224+PC2 真机三方一致;**真机首舞 2026-09-28 3/3 全程零锁存**(详见台账 "F10 first hardware session" 节)。候选格局:F10(踢腿 86%,结尾走位 2 m)vs F13(宽 std 臂项:结尾保真 0.83 m/张臂定格误差 0.07-0.11 rad,踢腿 73%,暂存 224 `/tmp/f13_deploy/`)。奖励空间已探索收敛:腿部位置项任何锐化都触发踢腿窗踝过冲折叠(F11/F12 0/2 同签名,踝过冲 0.09→0.37 rad),臂位置宽 std 项修复结尾但以踢腿幅度为价(F13),续训退化(F14 0/2,踢腿段步长爆炸)。下一步杠杆:R6 级参考手术(踢腿窗踝需求钳位)+ F13 配方;H0 真机采集(cap 工具已备)出权威差距表。
