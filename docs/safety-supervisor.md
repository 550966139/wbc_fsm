# 安全监控集成

`common/safety_supervisor.h` 提供无 DDS 依赖的检查器。DDS 回调收到有效 CRC 的 `LowState` 后调用 `stateReceived()`；控制循环在发布动作前检查通信超时、投影重力、有限数值、关节速度和目标跳变。任一检查失败时应立即切换 `FSMStateName::PASSIVE`，并清零力矩、设置阻尼。

阈值必须先在仿真和吊装实机上标定，不能把默认值视为经过实机认证的安全边界。
