#!/usr/bin/env python3
"""cap_hw.py v3 — hardware full-motor capture, STREAMING append (crash-proof).

v2 died silently mid-session on 2026-09-30 (0 bytes salvaged) — in-memory
buffers plus periodic checkpoints cannot survive an unknown kill. v3 appends
packed float32 rows to a raw file and flushes every 64 rows (~65 ms at 1 kHz):
any death — kill, OOM, reboot — loses at most one flush window. The .raw file
survives everything and converts to .npz afterwards (--convert).

Row layout: float64 t + 97 float32 (q[29] dq[29] tau[29] quat[4 wxyz]
gyro[3] accel[3]) = 396 bytes.

Usage:
  cap_hw.py <seconds> <out.npz>          record (writes <out>.raw alongside)
  cap_hw.py --convert <a.raw> <b.npz>    salvage-convert a raw log
Exit non-zero if zero samples.
PC2 deps: PYTHONPATH=/home/unitree/unitree_sdk2_python/unitree_sdk2_python-master
"""
import signal
import struct
import sys
import time

import numpy as np

ROW = struct.Struct("<d" + "f" * 97)  # t + 97 floats = 8 + 388 bytes
MAGIC = b"CAPHW3\n"


def record(dur, out):
    from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

    raw_path = out.replace(".npz", "") + ".raw"
    f = open(raw_path, "wb", buffering=0)
    f.write(MAGIC)

    state = {"n": 0, "buf": [], "t0": time.time()}

    def h(m):
        state["buf"].append(ROW.pack(
            time.time(),
            *[m.motor_state[i].q for i in range(29)],
            *[m.motor_state[i].dq for i in range(29)],
            *[m.motor_state[i].tau_est for i in range(29)],
            *list(m.imu_state.quaternion),
            *list(m.imu_state.gyroscope),
            *list(m.imu_state.accelerometer)))
        state["n"] += 1
        if len(state["buf"]) >= 64:
            f.write(b"".join(state["buf"]))
            state["buf"] = []

    ChannelFactoryInitialize(0, "eth0")
    ChannelSubscriber("rt/lowstate", LowState_).Init(h)

    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *a: stop.__setitem__("flag", True))
    signal.signal(signal.SIGINT, lambda *a: stop.__setitem__("flag", True))

    last = 0.0
    while not stop["flag"] and time.time() - state["t0"] < dur:
        time.sleep(0.05)
        now = time.time()
        if now - last > 10:
            print("[%s] cap_hw alive: %d samples" % (time.strftime("%H:%M:%S"), state["n"]), flush=True)
            last = now
    if state["buf"]:
        f.write(b"".join(state["buf"]))
    f.close()
    elapsed = max(time.time() - state["t0"], 1e-9)
    print("cap_hw stream done: %d samples, %.0f Hz avg, raw=%s" % (
        state["n"], state["n"] / elapsed, raw_path), flush=True)
    convert(raw_path, out)
    return state["n"]


def convert(raw_path, out):
    data = open(raw_path, "rb").read()
    off = len(MAGIC) if data.startswith(MAGIC) else 0
    n = (len(data) - off) // ROW.size
    if n == 0:
        print("cap_hw convert: EMPTY raw log", flush=True)
        return 0
    rec = np.frombuffer(data, dtype=np.uint8, count=n * ROW.size, offset=off)
    rows = rec.view([("t", "<f8"), ("v", "<f4", (97,))])
    np.savez_compressed(out, t=rows["t"], q=rows["v"][:, 0:29], dq=rows["v"][:, 29:58],
                        tau=rows["v"][:, 58:87], quat=rows["v"][:, 87:91],
                        gyro=rows["v"][:, 91:94], accel=rows["v"][:, 94:97])
    print("cap_hw converted %d rows -> %s" % (n, out), flush=True)
    return n


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--convert":
        sys.exit(0 if convert(sys.argv[2], sys.argv[3]) > 0 else 1)
    n = record(float(sys.argv[1]), sys.argv[2])
    sys.exit(1 if n == 0 else 0)
