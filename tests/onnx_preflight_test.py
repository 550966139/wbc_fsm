#!/usr/bin/env python3
"""Offline CLI regression with tiny synthetic ONNX graphs; never robot policies.

Uses only Python's standard library and the ONNX protobuf schema:
https://github.com/onnx/onnx/blob/v1.18.0/onnx/onnx.proto
"""
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


def varint(value):
    result = bytearray()
    while value >= 128:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def number(field, value):
    return varint(field << 3) + varint(value)


def blob(field, value):
    if isinstance(value, str):
        value = value.encode()
    return varint((field << 3) | 2) + varint(len(value)) + value


def value_info(name, width, data_type=1):
    shape = blob(1, number(1, 1)) + blob(1, number(1, width))
    tensor_type = number(1, data_type) + blob(2, shape)
    return blob(1, name) + blob(2, blob(1, tensor_type))


def constant_model(input_width=81, output_width=23, value=0.0, data_type=1):
    """Untrained constant output, deliberately kept outside model/ and deployment."""
    raw = struct.pack("<" + ("f" if data_type == 1 else "q") * output_width,
                      *([value] * output_width))
    tensor = (number(1, 1) + number(1, output_width) + number(2, data_type)
              + blob(8, "actions") + blob(9, raw))
    graph = (blob(2, "TEST_ONLY_NEVER_DEPLOY") + blob(5, tensor)
             + blob(11, value_info("obs", input_width))
             + blob(12, value_info("actions", output_width, data_type)))
    return number(1, 8) + blob(2, "offline-regression-only") + blob(7, graph) + blob(8, number(2, 13))


def motion_csv_text(defaults, frames=8):
    """50-column tracking motion fixture; frame 0 pose equals the robot defaults."""
    rows = []
    for k in range(frames):
        pos = list(defaults)
        pos[0] += 0.005 * k
        rows.append(",".join(f"{v:.9g}" for v in pos + [0.0] * 23 + [1.0, 0.0, 0.0, 0.0]))
    return "\n".join(rows) + "\n"


def main():
    executable, source, *fsm_executable = map(Path, sys.argv[1:])
    config = json.loads((source / "config/g1_23dof.json").read_text())
    tracking_config = json.loads((source / "config/g1_23dof_tracking.json").read_text())
    # The checked-in config ships the motions[] library (2026-09-30); the
    # single-motion synthetic cases below run on its slot 0 rewritten into the
    # legacy form.
    if "motions" in tracking_config:
        slot0 = tracking_config["motions"][0]
        tracking_config["model_path"] = slot0["model_path"]
        tracking_config["motion_path"] = slot0["motion_path"]
        for key in ("safety", "entry_gate", "motion_end_target", "handoff_settle_timeout",
                    "loop_motion"):
            if key in slot0:
                tracking_config[key] = slot0[key]
        del tracking_config["motions"]
    env = os.environ.copy()
    # Test isolation: these tests never request --run or initialize DDS.
    env.pop("UNITREE_DDS_IFACE", None)
    env.pop("UNITREE_DDS_DOMAIN", None)
    count = 0
    with tempfile.TemporaryDirectory(prefix="g1-onnx-offline-") as temporary:
        directory = Path(temporary)

        def check(label, model, success, expected):
            nonlocal count
            path = directory / (label + ".onnx")
            if isinstance(model, bytes):
                path.write_bytes(model)
            elif model is not None:
                path = model
            config["model_path"] = str(path)
            config_path = directory / (label + ".json")
            config_path.write_text(json.dumps(config))
            result = subprocess.run([str(executable), "--check", "--config", str(config_path)],
                                    env=env, capture_output=True, text=True, timeout=15)
            output = result.stdout + result.stderr
            if (result.returncode == 0) != success or expected not in output or "DDS interface" in output:
                raise AssertionError(f"{label}: exit={result.returncode}\n{output}")
            count += 1
            return config_path

        good_config = check("valid-constant", constant_model(), True, "DDS was not started")
        check("missing", None, False, "model not found")
        check("corrupt", b"this is not an ONNX model", False, "stopped:")
        check("wrong-input", constant_model(input_width=96), False, "float input [1,81]")
        check("wrong-output", constant_model(output_width=29), False, "float output [1,23]")
        check("wrong-type", constant_model(value=0, data_type=7), False, "float output [1,23]")
        check("nan-action", constant_model(value=float("nan")), False, "non-finite action")
        for model in sorted((source / "model").rglob("*.onnx")):
            if "g1_23dof" not in model.parts:
                check(model.stem, model, False, "23DoF policy")
        if fsm_executable:
            subprocess.run([str(fsm_executable[0]), str(good_config)], check=True, timeout=30)

        # Tracking (dance) contract: 124-input synthetic graph plus a temp
        # motion CSV. Guard values differ from the velocity config so the FSM
        # test can observe the per-state swap.
        motion_path = directory / "motion.csv"
        motion_path.write_text(motion_csv_text(config["default_joint_pos"]))

        def check_tracking(label, model, success, expected, motion=motion_path, end_target=None):
            nonlocal count
            path = directory / (label + "-tracking.onnx")
            if isinstance(model, bytes):
                path.write_bytes(model)
            tracking_config["model_path"] = str(path)
            tracking_config["motion_path"] = str(motion)
            tracking_config["safety"] = {"max_tracking_error": 1.4, "max_target_step": 0.9}
            # Short window so the FSM test can exercise the settle timeout.
            tracking_config["handoff_settle_timeout"] = 0.2
            if end_target is not None:
                tracking_config["motion_end_target"] = end_target
            config_path = directory / (label + "-tracking.json")
            config_path.write_text(json.dumps(tracking_config))
            tracking_config["motion_end_target"] = "policy23"
            result = subprocess.run([str(executable), "--check", "--config", str(good_config),
                                     "--tracking-config", str(config_path)],
                                    env=env, capture_output=True, text=True, timeout=15)
            output = result.stdout + result.stderr
            if (result.returncode == 0) != success or expected not in output or "DDS interface" in output:
                raise AssertionError(f"{label}: exit={result.returncode}\n{output}")
            count += 1
            return config_path

        good_tracking = check_tracking("valid", constant_model(input_width=124), True,
                                       "ONNX contract checked", end_target="policy23")
        # Fallback-switch variant for the FSM test's phase-1 behaviour checks.
        stand_tracking = directory / "stand-tracking.json"
        stand_config = json.loads(good_tracking.read_text())
        stand_config["motion_end_target"] = "stand"
        stand_tracking.write_text(json.dumps(stand_config))
        check_tracking("missing-model", None, False, "tracking policy missing")
        check_tracking("wrong-input", constant_model(input_width=81), False,
                       "tracking policy requires float input [1,124]")
        check_tracking("wrong-output", constant_model(input_width=124, output_width=29), False,
                       "tracking policy requires float output [1,23]")
        check_tracking("missing-motion", constant_model(input_width=124), False,
                       "cannot open tracking motion CSV", motion=directory / "absent.csv")
        check_tracking("bad-end-target", constant_model(input_width=124), False,
                       "unsupported motion_end_target", end_target="loco")

        # Multi-motion library ("motions" array): two synthetic policies with
        # distinct per-motion guards; the FSM test drives R2+B / R2+Y through
        # both entries.
        def dual_tracking_config():
            dual = json.loads((source / "config/g1_23dof_tracking.json").read_text())
            dual.pop("model_path", None)
            dual.pop("motion_path", None)
            motion2 = directory / "motion2.csv"
            motion2.write_text(motion_csv_text(config["default_joint_pos"]))
            model2 = directory / "dual-motion2.onnx"
            model2.write_bytes(constant_model(input_width=124))
            dual["motions"] = [
                {"name": "dance-one",
                 "model_path": str(directory / "valid-tracking.onnx"),
                 "motion_path": str(motion_path),
                 "safety": {"max_tracking_error": 1.4, "max_target_step": 0.9},
                 "handoff_settle_timeout": 0.2,
                 "motion_end_target": "policy23"},
                {"name": "dance-two",
                 "model_path": str(model2),
                 "motion_path": str(motion2),
                 "safety": {"max_tracking_error": 1.3, "max_target_step": 0.7},
                 "motion_end_target": "stand"},
            ]
            return dual

        def check_tracking_json(label, value, success, expected):
            nonlocal count
            config_path = directory / (label + ".json")
            config_path.write_text(json.dumps(value))
            result = subprocess.run([str(executable), "--check", "--config", str(good_config),
                                     "--tracking-config", str(config_path)],
                                    env=env, capture_output=True, text=True, timeout=15)
            output = result.stdout + result.stderr
            if (result.returncode == 0) != success or expected not in output or "DDS interface" in output:
                raise AssertionError(f"{label}: exit={result.returncode}\n{output}")
            count += 1
            return config_path

        dual_tracking = check_tracking_json("dual-valid", dual_tracking_config(), True,
                                            "ONNX contract checked")
        # obs-v2 (2026-10-13): the v2 contract with a 127-input graph loads and
        # the width must agree with the contract string.
        v2_config = json.loads(json.dumps(tracking_config))
        v2_config["policy_contract"] = "unitree_rl_mjlab_g1_23dof_tracking_v2"
        v2_config["observation_size"] = 127
        v2_model = directory / "v2-tracking.onnx"
        v2_model.write_bytes(constant_model(input_width=127))
        v2_config["model_path"] = str(v2_model)
        v2_config["motion_path"] = str(motion_path)
        check_tracking_json("v2-valid", v2_config, True, "ONNX contract checked")
        v2_wrong = json.loads(json.dumps(v2_config))
        v2_wrong["observation_size"] = 124
        check_tracking_json("v2-wrong-width", v2_wrong, False, "does not match policy_contract")
        v2_bad_model = json.loads(json.dumps(v2_config))
        v2_bad_model["model_path"] = str(directory / "v2-bad.onnx")
        (directory / "v2-bad.onnx").write_bytes(constant_model(input_width=124))
        check_tracking_json("v2-wrong-model-width", v2_bad_model, False,
                            "requires float input [1,127]")
        mixed = dual_tracking_config()
        mixed["model_path"] = str(directory / "valid-tracking.onnx")
        check_tracking_json("dual-legacy-mixed", mixed, False, "cannot be combined")
        duplicate = dual_tracking_config()
        duplicate["motions"][1]["name"] = "dance-one"
        check_tracking_json("dual-duplicate-name", duplicate, False, "duplicate tracking motion name")
        missing = dual_tracking_config()
        missing["motions"][1]["model_path"] = str(directory / "absent2.onnx")
        check_tracking_json("dual-missing-model", missing, False, "tracking policy missing")
        if len(fsm_executable) > 1:
            subprocess.run([str(fsm_executable[1]), str(good_config), str(good_tracking),
                            str(stand_tracking), str(dual_tracking)], check=True, timeout=60)
    print(f"{count} offline ONNX preflight cases passed; DDS never started")


if __name__ == "__main__":
    main()
