#!/usr/bin/env python3
"""cap_hw_sport.py — hardware pelvis capture (passive listener on rt/sportmodestate; read-only).

Kept in its own process on purpose: SportModeState_ lives in unitree_go and
mixing unitree_go + unitree_hg IDL modules in one process segfaults
cyclonedds (2026-09-28 F10 acceptance ops finding).

Same hardening as cap_hw.py: 10 s progress lines, 60 s checkpoints,
graceful SIGTERM, non-zero exit on zero samples.

Usage: python3 cap_hw_sport.py <seconds> <output.npz>
Output: ts(N,) pos(N,3) — same layout as the sim sport captures.
PC2 deps: PYTHONPATH=/home/unitree/unitree_sdk2_python/unitree_sdk2_python-master
"""
import os
import signal
import sys
import time

import numpy as np
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

dur = float(sys.argv[1])
out = sys.argv[2]
ckpt = out + ".ckpt.npz"

ChannelFactoryInitialize(0, "eth0")
sport = []
ChannelSubscriber("rt/sportmodestate", SportModeState_).Init(
    lambda m: sport.append((time.time(), m.position[0], m.position[1], m.position[2])))

stop = {"flag": False}
signal.signal(signal.SIGTERM, lambda *a: stop.__setitem__("flag", True))
signal.signal(signal.SIGINT, lambda *a: stop.__setitem__("flag", True))


def dump(path):
    n = len(sport)
    a = np.array(sport, dtype=np.float64) if n else np.zeros((0, 4))
    np.savez_compressed(path, ts=a[:, 0], pos=a[:, 1:4])
    return n


t0 = time.time()
last_ckpt = 0.0
last_alive = 0.0
while not stop["flag"] and time.time() - t0 < dur:
    time.sleep(0.05)
    now = time.time()
    if now - last_ckpt > 60:
        n = dump(ckpt)
        print("[%s] cap_hw_sport checkpoint: %d samples (%.0f Hz avg)" % (
            time.strftime("%H:%M:%S"), n, n / max(now - t0, 1e-9)), flush=True)
        last_ckpt = now
    elif now - last_alive > 10:
        print("[%s] cap_hw_sport alive: %d samples" % (time.strftime("%H:%M:%S"), len(sport)), flush=True)
        last_alive = now

n = dump(out)
if os.path.exists(ckpt):
    os.remove(ckpt)
print("cap_hw_sport DONE: %d samples, %.0f Hz avg, -> %s" % (
    n, n / max(time.time() - t0, 1e-9), out), flush=True)
sys.exit(1 if n == 0 else 0)
