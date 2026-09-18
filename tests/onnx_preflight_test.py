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


def constant_model(input_width=80, output_width=23, value=0.0, data_type=1):
    """Untrained constant output, deliberately kept outside model/ and deployment."""
    raw = struct.pack("<" + ("f" if data_type == 1 else "q") * output_width,
                      *([value] * output_width))
    tensor = (number(1, 1) + number(1, output_width) + number(2, data_type)
              + blob(8, "actions") + blob(9, raw))
    graph = (blob(2, "TEST_ONLY_NEVER_DEPLOY") + blob(5, tensor)
             + blob(11, value_info("obs", input_width))
             + blob(12, value_info("actions", output_width, data_type)))
    return number(1, 8) + blob(2, "offline-regression-only") + blob(7, graph) + blob(8, number(2, 13))


def main():
    executable, source, *fsm_executable = map(Path, sys.argv[1:])
    config = json.loads((source / "config/g1_23dof.json").read_text())
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
        check("wrong-input", constant_model(input_width=96), False, "float input [1,80]")
        check("wrong-output", constant_model(output_width=29), False, "float output [1,23]")
        check("wrong-type", constant_model(value=0, data_type=7), False, "float output [1,23]")
        check("nan-action", constant_model(value=float("nan")), False, "non-finite action")
        for model in sorted((source / "model").rglob("*.onnx")):
            if "g1_23dof" not in model.parts:
                check(model.stem, model, False, "23DoF policy")
        if fsm_executable:
            subprocess.run([str(fsm_executable[0]), str(good_config)], check=True, timeout=30)
    print(f"{count} offline ONNX preflight cases passed; DDS never started")


if __name__ == "__main__":
    main()
