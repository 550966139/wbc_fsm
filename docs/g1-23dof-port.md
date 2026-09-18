# G1 23DoF 适配与部署

本分支以参考项目 `b352409` 为基础，当前入口面向 G1 23DoF 的阻尼、站立和速度策略。原 29DoF 的 AMP、MJAmp、Loco、WBC 源码和资源仍保留供参考，但 23DoF 入口不构造这些状态，也不能进入舞蹈/WBC。23DoF 动作跟踪需要另行训练与适配观测，不能直接复用原舞蹈模型。

## 当前边界

已实现 23DoF 控制链路及离线回归，并于 2026-09-17 在 192.168.3.224 的 unitree_mujoco 集成仿真中完成 Sim2Sim 闭环验证（进入策略、零命令站立、0.2 m/s 行走、停止、L2+B 锁存与吊带辅助恢复全部通过，数据见 [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)）。**尚未完成 aarch64 构建和真机验证。**

2026-09-17 核对官方 `unitree_rl_mjlab` commit `1425b15f73bd4095f0df53709d7c389c3eb9e790`：23DoF 目录提供训练及部署配置，没有附带匹配的预训练 ONNX。同日在训练机 192.168.3.224 上完成 `Unitree-G1-23Dof-Flat` 训练（官方 4096 envs 配置，2 小时 08 分），训练中途第 ~5030 次迭代发生 PPO 坍缩，最终选用坍缩前稳定期的 `model_4900.pt`（其后 20 次迭代窗口：速度误差 0.515、平均奖励 45.6、摔倒率 0.004）经官方导出路径重导出。当前 `model/g1_23dof/policy.onnx` 已就位并通过 `--check` 真实推理与全部回归；溯源文件（checkpoint、env.yaml、agent.yaml、sha256）同目录保存。仓库原有四个 ONNX 全部输出 29 个关节动作，不可截断后使用。

训练环境注意（224 上 conda env `unitree_rl_mjlab`）：需钉住 `mujoco==3.5.0`（3.13.0 与 `mujoco-warp==3.5.0` 的 `mjENBL_MULTICCD` 不兼容）、`warp-lang==1.12.1`（1.13+ 移除 `wp.context`，破坏 mjlab 1.2.0）、补装 `scipy`，训练命令使用 `--agent.logger tensorboard`（rsl-rl 5.0.1 的 wandb writer 与 wandb 0.30.0 不兼容）。完整记录见 [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)。

## 模型契约

- CPU float ONNX，单输入 `[1,80]`、单输出 `[1,23]`；batch 维声明可为动态，实际推理必须返回 `[1,23]`。
- 观测依次为：角速度 3、投影重力 3、速度命令 3、步态 sin/cos 2、相对默认关节位置 23、关节速度 23、上一帧原始动作 23。
- 控制步长 0.02 秒，步态周期 0.6 秒。零命令也推进相位，命令范数小于 0.1 时屏蔽 sin/cos。
- 动作按 `default_joint_pos + action_scale * action` 转为目标，PD、默认姿态、顺序和缩放必须与实际训练导出一致。
- SDK 电机槽位依次为 `[0..12,15..19,22..26]`。其余六槽 `13,14,20,21,27,28` 的 `mode=0`，全部命令字段为零；它们没有位置控制。
- 当前配置来自官方 Flat velocity 部署契约，包含腰 yaw 的 `action_scale=0.55`。Rough 地形观测或带循环隐藏状态的模型不能直接接入。

从训练运行导出 ONNX 时，同时保存该运行的 `deploy.yaml`、环境/训练参数、模型哈希和任何 `policy.onnx.data` 外部权重。核对并更新 `config/g1_23dof.json` 后，将真实策略放入 `model/g1_23dof/policy.onnx`。模型相对路径按配置文件所在目录的上一级解析；使用自定义位置时建议写绝对模型路径。

`--check` 会检查配置、张量接口并执行一次推理；它不能证明模型经过正确训练或能够在真机上稳定行走。`tests/onnx_preflight_test.py` 生成的常数图仅用于软件测试，不得作为机器人策略。

## 构建与离线验证

以下命令从 `wbc_fsm/` 执行。依赖为 C++17 编译器、CMake >=3.14、Eigen3、nlohmann_json >=3.7.3、Unitree SDK2、ONNX Runtime（本次使用官方 CPU 1.22.0）。当前路径无需 CUDA 或 OpenSSL。

无 SDK/ONNX 的基础测试：

```sh
cmake -S . -B build-tests -DWBC_BUILD_CONTROLLER=OFF
cmake --build build-tests -j2
ctest --test-dir build-tests --output-on-failure
```

纯 C++ 策略及安全测试不需要外部依赖；配置与命令测试需 JSON/Eigen。CMake 会明确提示跳过的测试。也可执行 `bash scripts/run-port-tests.sh`，通过 `NLOHMANN_JSON_INCLUDE_DIR` 指定非系统 JSON 头文件路径。

完整控制器（将路径替换成当前机器的安装位置）：

```sh
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH=/opt/unitree_robotics \
  -DONNXRUNTIME_ROOT=/absolute/path/to/onnxruntime-linux-x64-1.22.0 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
ctest --test-dir build --output-on-failure
./build/wbc_fsm --check --config "$PWD/config/g1_23dof.json"
```

完整构建在 Python3 可用时还运行 ONNX/状态机离线集成测试：生成临时常数图，验证模型拒绝、状态切换、阻尼和故障恢复；不会初始化 DDS。无参数运行也等价于 `--check`。只有显式 `--run` 才连接 DDS。

## 仿真验证

使用与训练一致的 G1 **23DoF** MuJoCo 资产与关节顺序，确认模拟器的 HG LowState `mode_machine=4`。29DoF 场景或错误机型会被拒绝。训练服务器负责训练/导出，控制器推理与 DDS 闭环在部署计算机本地运行。

```sh
./build/wbc_fsm --run --config "$PWD/config/g1_23dof.json" --interface lo --domain 1
```

验证顺序：阻尼 → START 站立 → 站立插值完成且姿态稳定 → R2+A 速度策略。初始速度上限为前后 0.2 m/s、侧向 0.1 m/s、偏航 0.2 rad/s。逐一验证零命令、低速、L2+B、通信丢帧、异常输出、倾角保护及恢复。记录循环与推理耗时，PC2 上应稳定满足 20 ms 周期，40 ms 控制超时会锁存阻尼。

2026-09-17 Sim2Sim 实测结论（unitree_mujoco + 弹性吊带 + FIFO 按键注入，细节与数据见 [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)）：

- 训练策略可稳定进入与行走：零命令松吊带站立 30 s（z=0.784 m）；W（0.2 m/s）5 s 前进 0.81 m，稳态 vx 0.14–0.21 m/s；松开指令后平滑停止。
- 为让策略可进入，做了两处部署侧修正：安全护栏按策略实际幅度标定（`max_target_step` 0.35→1.0、`max_tracking_error` 0.8→1.2，原值会在切入首拍必然拒收）；策略/站立目标在发送前按 `joint_lower/upper` 钳位（`g1::clampTargets23`，训练中 MuJoCo 对超出 ctrlrange 的位置目标同样只是饱和而非报错）。
- 吊带挂在锚点 (0,0,3) 且放长约 0.9 m 时承担约 75% 体重，行走会失效——验证行走必须松开吊带（键 9）。模拟器运行期间禁止 Backspace 重置（tick 回跳会被拒收，等效长时间失联）。

## PC2 部署与吊装验证

1. 记录实际机型、固件、`uname -m`、系统版本、底层 DDS 网卡和 SSH 管理地址。不要把管理 IP 当作 DDS 接口，也不要假定网卡一定叫 `eth0`。
2. 通过 SSH 将源码和已验证模型放到 PC2 的新版本目录，保留原控制器作为回退。不要设置开机自启或覆盖旧控制器。
3. **若 PC2 为 aarch64**，安装/使用 aarch64 SDK2 与 `onnxruntime-linux-aarch64-1.22.0`，在 PC2 原生执行上面的构建命令并替换 `ONNXRUNTIME_ROOT`。x64 可执行文件和库不能直接复制到 ARM 运行。交叉编译需另配 PC2 对应 sysroot；本分支没有声称交叉编译已验证。
4. 先运行 CTest、`ldd build/wbc_fsm` 和 `--check`。核对模型/配置哈希以及外部权重齐全。
5. 由现场操作者吊装机器人、准备急停，按该机型官方步骤进入开发模式，停止原运动服务及其他 `rt/lowcmd` 发布者（包括旧策略启动器）。
6. 现场准备完成且仿真通过后，使用实际底层网卡执行：

```sh
./build/wbc_fsm --run --config "$PWD/config/g1_23dof.json" --interface eth0 --domain 0
```

程序对 domain 0 调用 `CheckMode`，运动服务未释放或查询失败时拒绝继续；不会自动调用 `ReleaseMode`。随后先订阅状态，最多等待 3 秒，要求有效 CRC、新鲜数据和 `mode_machine=4`；再监听已有 `rt/lowcmd` 命令 300 ms，发现活动发布者时拒绝启动。该检测不是 DDS 独占锁，操作者必须防止启动后又有其他控制器加入。

首次真机仅在吊装和现场急停条件下验证通信、关节对应、阻尼和缓慢站立，确认方向、PD、限位及恢复行为后再进行受控落地测试。仅有离线测试通过不足以放行真机行走。

### 2026-09-18 现场修订（本机固件实测）

- **官方"悬挂+阻尼下 L2+R2 进调试模式"在本机固件上无效**：正确手势（按住 L2+单击 R2，来源为遥控器手册 V1.1 与固件 V1.0.2 按键表）也做了两轮对照实验，`CheckMode` 的 `name` 始终为 `'ai'`；L2+A/L2+B 的诊断动作正常只说明固件层功能在，不代表模式释放发生。**可靠途径 = 官方 RPC `MotionSwitcherClient.ReleaseMode()`**（本机验证 name 变空、随后 lowcmd 电机真实出力）。机器人每次重启 ai 自动恢复，需重新释放。
- **本机固件不对外部 lowcmd 做模式校验**（历史 Python 策略从未释放也跑通）；`CheckMode` 门禁是我们自己的保守设计，防的是官方文档警告的"运控并发零速指令→抖动冲突"。
- **站立增益**：23DoF 移植曾让 FixedStand 复用策略契约增益，真机吊带下无法把机器人拉进站姿（踝误差 0.245 卡 0.15 门禁）。现恢复作者 29DoF 站立增益映射（`stand_stiffness`/`stand_damping`，沿插值爬坡），Sim2Sim 实测带支持下最大误差 0.039、R2+A 首试进策略。FixedStand 的定位仍是"外部支撑下的姿态整形器"——完全无支撑的独立起立需要重力前馈（可选项，未做）。

## 操作与退出

| 输入 | 作用 |
| --- | --- |
| START（无故障） | 从阻尼进入默认姿态站立 |
| R2+A | 站立完成、每关节误差 ≤0.15 rad 且速度 ≤1 rad/s 后进入速度策略 |
| 左摇杆 | 前后/侧向速度；右摇杆水平轴为偏航 |
| L2+B | 锁存阻尼并退出当前动作 |
| START（故障恢复后） | 仅确认并解除锁存；松开后再次 START 才站立 |
| SELECT / SIGINT / SIGTERM | 退出，尝试发送 20 ms 阻尼后停止 DDS |

DDS 默认 `lo` / domain 1；可用 `UNITREE_DDS_IFACE`、`UNITREE_DDS_DOMAIN` 配置，CLI 参数优先。domain 0 必须显式指定非 loopback 网卡。异常机型变化禁止继续发布；状态/命令 100 ms 看门狗由独立线程执行。进程崩溃、断电或网络彻底中断时的停机依赖机器人侧保护，20 ms 退出阻尼只是尽力交接。

## 核对来源

- [官方 23DoF 部署参数](https://github.com/unitreerobotics/unitree_rl_mjlab/blob/1425b15f73bd4095f0df53709d7c389c3eb9e790/deploy/robots/g1_23dof/config/policy/velocity/v0/params/deploy.yaml)
- [官方 23DoF 入口与机型检查](https://github.com/unitreerobotics/unitree_rl_mjlab/blob/1425b15f73bd4095f0df53709d7c389c3eb9e790/deploy/robots/g1_23dof/main.cpp)
- [官方观测实现](https://github.com/unitreerobotics/unitree_rl_mjlab/blob/1425b15f73bd4095f0df53709d7c389c3eb9e790/deploy/include/isaaclab/envs/mdp/observations/observations.h)
- [官方训练与真机部署步骤](https://github.com/unitreerobotics/unitree_rl_mjlab#4-real-deployment)
- [SDK2 G1 低层控制示例](https://github.com/unitreerobotics/unitree_sdk2/blob/main/example/g1/low_level/g1_ankle_swing_example.cpp)
