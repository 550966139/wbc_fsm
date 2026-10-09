#!/usr/bin/env python
"""Deployment ONNX export from a tracking/velocity rsl-rl checkpoint.

Reconstruction of the volatile /tmp/export_from_ckpt.py (2026-09-30, take2
campaign): loads a checkpoint saved by MotionTrackingOnPolicyRunner
(actor_state_dict with obs_normalizer + mlp.{0,2,4,6}, ELU activations per
agent.yaml) and exports the deterministic deployment contract:

    input  [1, OBS] float32
    output [1, ACT] float32
    normalization embedded as leading Sub/Div nodes (obs - mean) / std

Usage: export_from_ckpt.py <ckpt.pt> <out.onnx> <obs_dim> <act_dim>
      [--verify <reference.onnx>]  (compare outputs on random inputs)

Verified against the in-service F10 policy.onnx (sha 8d364f462d8c6e5c) before
reuse: max |diff| < 1e-5 on random inputs.
"""
import argparse

import numpy as np
import torch
from torch import nn


class DeployPolicy(nn.Module):
    def __init__(self, actor_sd):
        super().__init__()
        linears = [(k, v) for k, v in actor_sd.items() if k.startswith("mlp.") and k.endswith(".weight")]
        linears.sort(key=lambda kv: int(kv[0].split(".")[1]))
        layers = []
        for i, (k, w) in enumerate(linears):
            b = actor_sd[k.replace(".weight", ".bias")]
            lin = nn.Linear(w.shape[1], w.shape[0])
            lin.weight.data = w
            lin.bias.data = b
            layers.append(lin)
            if i < len(linears) - 1:
                layers.append(nn.ELU())  # agent.yaml: activation elu
        self.mlp = nn.Sequential(*layers)
        self.register_buffer("mean", actor_sd["obs_normalizer._mean"])
        # Deployment denominator verified against the in-service F10 ONNX
        # (2026-09-30): sqrt(var) + 1e-2, NOT the checkpoint's stored _std.
        self.register_buffer("std", torch.sqrt(actor_sd["obs_normalizer._var"]) + 1e-2)

    def forward(self, x):
        return self.mlp((x - self.mean) / self.std)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("obs_dim", type=int)
    ap.add_argument("act_dim", type=int)
    ap.add_argument("--verify", default=None,
                    help="reference ONNX to compare outputs against (random inputs)")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("actor_state_dict", ck.get("model_state_dict", ck))
    model = DeployPolicy(sd).eval()
    obs = torch.zeros(1, args.obs_dim)
    torch.onnx.export(
        model, obs, args.out,
        export_params=True, opset_version=18, dynamo=False,
        input_names=["obs"], output_names=["actions"],
        dynamic_axes=None,
    )
    print(f"wrote {args.out} [{args.obs_dim} -> {args.act_dim}] from iter {ck.get('iter', '?')}")

    if args.verify:
        import onnxruntime as ort
        ref = ort.InferenceSession(args.verify, providers=["CPUExecutionProvider"])
        new = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
        rng = np.random.default_rng(0)
        worst = 0.0
        for _ in range(64):
            x = rng.normal(0, 1.2, size=(1, args.obs_dim)).astype(np.float32)
            a = ref.run(None, {"obs": x})[0]
            b = new.run(None, {"obs": x})[0]
            worst = max(worst, float(np.abs(a - b).max()))
        print(f"verify vs {args.verify}: max|diff| {worst:.2e} over 64 random inputs")
        assert worst < 1e-5, "export does not reproduce the reference policy"


if __name__ == "__main__":
    main()
