#!/bin/bash
# release_ai.sh — 释放厂商内置运控（ai），使 wbc_fsm 可以接管电机命令。
#
# 背景（2026-09-18 现场实测，本机固件）：
#   - 官方文档的遥控 L2+R2 进调试模式在本机无效（正确手势亦然，CheckMode 不变）。
#   - 机器人每次重启后 ai 自动恢复（CheckMode name='ai'），必须重新释放。
#   - 本脚本是官方 RPC 途径（motion_switcher ReleaseMode），已在本机验证有效。
#
# 用法（PC2 上）：bash ~/wbc_fsm/scripts/release_ai.sh
# 判据：输出 RELEASED 即成功（CheckMode name 变空）。
set -euo pipefail
export PYTHONPATH=/home/unitree/unitree_sdk2_python/unitree_sdk2_python-master
python3 - <<'EOF'
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

ChannelFactoryInitialize(0, "eth0")
m = MotionSwitcherClient()
m.SetTimeout(4.0)
m.Init()
code, mode = m.CheckMode()
print("before:", code, mode)
if mode.get("name"):
    print("ReleaseMode ->", m.ReleaseMode())
code, mode = m.CheckMode()
print("after:", code, mode)
if mode.get("name"):
    print("STILL_ACTIVE:", mode["name"])
    raise SystemExit(1)
print("RELEASED")
EOF
