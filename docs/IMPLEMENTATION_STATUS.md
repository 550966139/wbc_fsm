# G1 23DoF implementation ledger

Updated: 2026-09-17 (trained policy obtained; MuJoCo Sim2Sim closed loop validated). Branch: `feature/g1-23dof-port`.
Reference baseline: `b352409`; previous adaptation HEAD: `2434c85`.

Goal: adapt the reference controller for a G1 23DoF robot, then validate and deploy on its PC2. The implemented policy path covers damping, fixed stand and flat-terrain velocity control. Original 29DoF sources/resources remain as references. 23DoF WBC/dance tracking is not implemented by truncating those models.

## Implemented locally

- [x] Validated 23-joint motor map, defaults, PD, action scales and 80-observation contract.
- [x] Dedicated 80-input / 23-output ONNX adapter, CPU inference and startup trial inference.
- [x] 23DoF Passive / interpolated Stand / velocity FSM. Legacy policy states are not instantiated.
- [x] Receive → compute → validate → send ordering, latched damping, explicit recovery and exit handling.
- [x] Joint/IMU/joystick finite checks, target limits/tracking/steps, exact configured gains, disabled absent slots.
- [x] Training-faithful target saturation (`g1::clampTargets23`) on policy and stand targets: MuJoCo clamps position targets at actuator ctrlrange during training, so deployment clamps to configured joint limits instead of tripping; `validateTargets23` remains the post-clamp last line of defense.
- [x] Monotonic 50 Hz control timing, 40 ms control deadline, independent approximately 500 Hz publishing and 100 ms state/command watchdog.
- [x] Offline-by-default CLI; explicit `--run` is required for DDS.
- [x] Startup checks for interface, real motion-service ownership, fresh CRC-valid state, machine type 4 and an existing active LowCmd stream.
- [x] Configurable target-specific ONNX Runtime root and SDK-independent test build.
- [x] Full native **x86_64** Release build against official SDK2 and ONNX Runtime 1.22.0.
- [x] Five CTest groups pass, including synthetic ONNX/real FSM offline integration. No DDS connection is initialized by these tests.

## Actual validation

```sh
/tmp/g1-build-tools/cmake/data/bin/cmake -S wbc_fsm -B /tmp/g1-full-build \
  -DCMAKE_PREFIX_PATH='/tmp/g1-sdk2-install;/tmp/g1-deps/usr' \
  -DONNXRUNTIME_ROOT=/tmp/onnxruntime-linux-x64-1.22.0 \
  -DCMAKE_BUILD_TYPE=Release
/tmp/g1-build-tools/cmake/data/bin/cmake --build /tmp/g1-full-build -j4
/tmp/g1-build-tools/cmake/data/bin/ctest --test-dir /tmp/g1-full-build --output-on-failure
```

Result: **5/5 passed**. Base tests run 458 checks even with `NDEBUG`; integration additionally covers valid/invalid model shapes, wrong tensor type, NaN, missing/corrupt models, rejection of the four legacy models, stand entry gating, operator stop, watchdog recovery, stale state and SELECT while faulted. ONNX fixtures are temporary untrained constant graphs and are never placed in the deployment model directory.

Tooling/dependencies above are temporary local validation paths, not paths to copy to PC2. Use the portable commands in [g1-23dof-port.md](g1-23dof-port.md) on the target machine. Existing legacy headers still generate non-fatal reorder/unused-parameter warnings in the command test.

## Trained 23DoF policy (2026-09-17)

Trained on `192.168.3.224` (RTX 4090, driver 580.173.02) in a new conda env `unitree_rl_mjlab` (Python 3.11). Official repo `unitree_rl_mjlab` at contract-pinned commit `1425b15f73bd4095f0df53709d7c389c3eb9e790`, installed `pip install -e .` plus required fixes: `mujoco==3.5.0` (pinned; 3.13.0 breaks `mujoco-warp==3.5.0` on `mjENBL_MULTICCD`), `warp-lang==1.12.1` (1.13+ removes `wp.context`, breaking mjlab 1.2.0), `scipy` (missing dependency), and `--agent.logger tensorboard` (rsl-rl 5.0.1 wandb writer incompatible with wandb 0.30.0).

Training command (run directory `logs/rsl_rl/g1_23dof_velocity/2026-09-17_10-16-57`, 0.78 s/iteration, 2 h 08 m):

```sh
python scripts/train.py Unitree-G1-23Dof-Flat --env.scene.num-envs 4096 --agent.logger tensorboard
```

PPO collapsed near iteration 5030 (fall rate spike to 0.54) and recovered only to a worse plateau (twist error ~0.80, reward ~35 vs ~45 pre-collapse). Checkpoint `model_4900.pt` from the pre-collapse stable phase was selected (post-checkpoint window 4901–4920: twist error 0.515, mean reward 45.6, fall rate 0.004) and re-exported through the same official exporter path (`MLPModel.as_onnx`, opset 18; observation normalization embedded in the graph as leading Sub/Div nodes).

Delivered to `model/g1_23dof/`: `policy.onnx` (836 710 bytes, single input `[1,80]`, single output `[1,23]`, sha256 `70082fb2f102d55c4f833a22998da98626e3e19b2e9f538cc8f77e10000d4129`), `model_4900.pt` (training checkpoint, sha256 `3900fae24f9c2a6d059fe371fe8e1bea4c4816097b65208829bcde1015fe5312`), and the run's `env.yaml`/`agent.yaml` parameters. No external `policy.onnx.data` is produced (weights are embedded).

Validation after placing the real policy:

```sh
/tmp/g1-full-build/wbc_fsm --check --config wbc_fsm/config/g1_23dof.json
# PASS: 23DoF config and ONNX contract checked; DDS was not started.
/tmp/g1-build-tools/cmake/data/bin/ctest --test-dir /tmp/g1-full-build --output-on-failure
# 5/5 passed (same suite as above)
NLOHMANN_JSON_INCLUDE_DIR=/tmp/g1-deps/usr/include bash scripts/run-port-tests.sh
# 205 + 25 + 51 checks, 0 failures
```

`--check` performs one real inference through the trained graph; it does not establish sim-to-real validity.

## MuJoCo Sim2Sim validation (2026-09-17, host 192.168.3.224)Closed-loop validation in the mjlab-integrated `unitree_mujoco` simulator (patched: keyboard-joystick + FIFO key injection, elastic band, 1 mrad inward joint ranges; scene `scene_g1_23dof.xml`, `mode_machine=4`). Results with the deployed `policy.onnx`:

- **Policy entry**: R2+A from completed fixed stand enters `23DoF locomotion` and stays latched-free; entry reproduced twice after fixes.
- **Zero-command standing, band released** (`9`): 30 s at z=0.784 m, |x| drift < 5 mm, no safety latch.
- **Commanded walking** (W = 0.2 m/s limit for 5 s): 0.81 m travelled, steady-state vx 0.14–0.21 m/s (mean ≈ 0.16), z 0.782–0.785 m throughout; clean deceleration to stand after release.
- **Operator stop**: L2+B latches damping (`operator requested damping`), robot settles to the floor as expected without the band.
- **Recovery flow**: band re-attach + retract → upright → START (acknowledge) → START (fixed stand) verified; the stand interpolation no longer trips on joints resting at limits (see clamp below).
- Regression after the changes: `ctest` **5/5**, `run-port-tests.sh` **210 + 25 + 51 checks, 0 failures** (224, fresh `build-tests` tree + full `build`).

Two deployment fixes were required to make the trained policy enterable (both discovered in sim, neither a training-contract mismatch):

1. **Safety guard calibration** (`config/g1_23dof.json`): `max_target_step` 0.35 → 1.0, `max_tracking_error` 0.8 → 1.2. Offline ONNX replay at the exact entry observation showed the policy's intrinsic first-cycle step is ≈ 0.36 rad from the training-default pose and up to ≈ 0.8 rad with measured offsets (action scale 0.44–0.55 with actions ≈ ±1.5–1.9 on the shoulders), so the original guards rejected the policy on every entry. These guards are port-authored values, not upstream hardware limits; recalibrate again if the policy is retrained.
2. **Target clamping** (`g1::clampTargets23` in `include/common/policy23.h`, applied in `State_Policy23::run` and the 23DoF branch of `State_FixedStand::run`): a few cycles after entry, closed-loop dynamics drove shoulder/ankle targets past configured joint limits and the old throw-to-damping behavior aborted the run. Training never sees this because MuJoCo saturates position targets at ctrlrange; the clamp reproduces that. This also removes the earlier "1 mrad XML shrink" real-robot concern for stand recovery (scene shrink remains harmless).

Known sim-only caveats: with the elastic band attached at 0.9 m the band unloads ≈ 75 % of the robot's weight and walking degrades — validate locomotion only with the band off; do not press Backspace in the viewer while the controller runs (tick rollback looks like a long state outage to the IOSDK layer).

### Repeat campaign (2026-09-18): three walk cycles, e-stop, fault injection

- **Walk cycle 1/3 (band off)**: 0.807 m in 5 s, vx 0.148–0.210 m/s, z 0.782–0.785, stop within ~1 s, no latch.
- **Yaw (Q/E, band off)**: quaternion-derived yaw +0.573 rad over 4 s under rx=−1, −0.394 rad back under rx=+1, settles stable; raw stick values verified in the `wireless_remote` frame (±1.00) end-to-end.
- **Lateral (A/D): NOT functional — policy limitation.** At the deployed 0.1 m/s limit the robot is perfectly static (verified band on and off, command verified in-frame). With the limit correctly raised to 0.5 (Python JSON edit; an earlier single-line `sed` silently missed the pretty-printed file) the robot does **not** move sideways — it runs **forward** at 0.38–0.46 m/s (2.8 m in 6 s, far above the 0.2 x-limit) and after release kept a slow zero-command limit cycle for ~150 s (reached (11, 9) m). Keep `deployment_command_limits = [0.2, 0.1, 0.2]`; do not raise the y limit with this checkpoint; lateral needs retraining.
- **Walk cycle 3/3 (band off)**: 0.78 m in 5 s, vx 0.147–0.170, post-stop 5 s drift < 2 mm.
- **E-stop mid-walk (L2+B, band off)**: latch line logged immediately, robot drops within one 0.8 s sample, settles on the floor; full band-assisted recovery afterwards.
- **State-loss injection**: `kill -STOP` on the simulator for 1.5 s → wbc latched `DDS state/command watchdog` within the watchdog window; after `CONT` the robot held ~standing under band + kd=3 damping; START×2 recovery and policy re-entry verified.
- Operational lessons recorded: enabling the elastic band while the robot is far from the anchor (≈14 m) produces a ≈2.7 kN slingshot — re-enable the band only near the anchor; edit the pretty-printed config with JSON tooling, not line `sed`.

## PC2 deployment prep (2026-09-18, host 10.10.16.184:15389)

Deployed to the robot's PC2 per the owner's instruction; **the controller has never been started with `--run` and no motor command has been sent** — startup is reserved for on-site personnel (hard rule re-confirmed by the owner: the program must contain no autonomous robot commands; all motion is remote-gated).

- **PC2**: aarch64 (Tegra, kernel 5.10.104), Ubuntu 20.04.6, 8 cores/15 GB, DDS interface `eth0` (192.168.123.x), `mode_machine=4` observed, `rt/lowcmd` verified silent before any change.
- **Historical policies removed from service**: `dance_controller`, `g1-ctrl-guard`, `g1-policy-launcher` system units disabled+stopped; leftover dance processes killed (robot verified passive first: joints still, no lowcmd traffic); all historical deployment dirs archived (not deleted) under `~/archive_historical_20260918/` for rollback. System services (master/ota/remote/web/video) untouched.
- **Build**: `~/wbc_fsm` (78 files: source with clamp + calibrated guards, `model/g1_23dof/policy.onnx`+yaml, no training checkpoint). Dependencies resolved on-robot: SDK2 from `/usr/local` (a proper `unitree_sdk2Targets.cmake` installed to `/usr/local/lib/cmake/unitree_sdk2/` — the stock install was missing it), ONNX Runtime aarch64 1.22.0 from the existing `~/unitree_rl_mjlab/deploy/thirdparty`, Eigen3 system package, nlohmann_json 3.11.3 built into `~/deps/json-build` (apt was broken by a pre-existing chrony/timesyncd conflict — system packages deliberately not modified).
- **Offline validation on PC2**: `ctest` **5/5** (including the 5 s real-inference ONNX/FSM integration test), `--check` **PASS** (one real inference through the aarch64 runtime; DDS not started), `ldd` clean (onnxruntime resolved via baked rpath).
- Operator runbook written to `~/wbc_fsm/RUNBOOK_ONSITE.md` (start command `--run --interface eth0 --domain 0`, remote flow, known limits, rollback).
- **Remote one-key launch (owner request, 2026-09-18 later)**: listen-only guard `g1-wbc-launcher.service` (enabled at boot, `~/wbc_fsm/scripts/launch_guard.py`) watches `rt/lowstate` on eth0/domain 0 and launches `wbc_fsm --run` on an **L1+X** edge, gated by: not already running, lowstate fresh ≤2 s (dev mode), `rt/lowcmd` quiet ≥3 s (no competing controller). Safety net: pkills wbc_fsm if lowstate dies >10 s while it runs. The guard never publishes; after launch the controller sits in passive damping awaiting the remote (all motion stays remote-gated). Detailed remote manual in `RUNBOOK_ONSITE.md`.
- Disclosure: in `--run` the publisher emits passive damping frames (kp=0, kd=3, zero position/torque) while awaiting operator input — the idle handoff state; every posture/motion transition requires the remote.

## Hardware validation of the stand controller (2026-09-18 16:16, attended)

First fully successful hardware session (log `wbc_20260918_161650`): L1+X guard launch → START → FixedStand pulled the robot to the default stance and **held it with no tilt/tracking latch and no guardrail trip** (previous sessions always latched) → **R2+A entered locomotion on the first press** (no rejection line; gate met position <0.15 rad and |dq|<1.0) → operator confirmed standing and walking physically → session closed with a deliberate L2+B (the only latch in the log, an operator-requested stop, not a fault). Comparison across the three hardware sessions: contract gains (10:25) = slumped + silent R2+A rejection; M1 author-mapped gains (15:20) = no visible stand + silent rejection; M2 stiff gains + gravity feedforward (16:16) = stand + first-try policy entry. Deployment path: files pushed to PC2, native aarch64 rebuild, ctest 5/5, `--check` PASS, `release_ai.sh` RELEASED before launch.

## Stand-gain restoration for FixedStand (2026-09-18 afternoon)

Root cause of "START does not stand" from the first attended hardware session, then reproduced in sim: the 23DoF port made `State_FixedStand` reuse the policy-contract gains (legs 40.2/99.1/28.5, tau=0). Upstream 29DoF stand used its own gains (`State_FixedStand.h`: hips 100, knee 150, ankles 40, waist 300) and assumed the robot entered already standing ("Please make the robot stand first"). With contract gains a sling-hung robot is not pulled into stance (ankle sag ≈0.245 rad blocked the 0.15 rad policy-entry gate; sling support was the missing physical premise).

- Config: optional `stand_stiffness`/`stand_damping` (23 values each, mapped from the authored 29DoF gains by joint name: hips 100/100/100, knee 150, ankles 40, waist 300, arms 100/100/50/50/20; kd legs 2/2/2/4/2/2, waist 3, arms 2/2/2/2/1). Absent keys fall back to the policy contract, preserving the previous behaviour exactly.
- Command layer: `CtrlComponents::setStandTargets23(targets, blend)` — stand form whose kp/kd lerp contract→stand along the same smoothstep as the FixedStand position interpolation. `setTargets23` (policy form) is unchanged; `validateCommand` accepts exactly the policy form (blend 0) or the stand form recomputed bit-identically from the recorded blend — any other gain combination still throws. Damping resets the form.
- Tests: config23 (stand-key validation + contract fallback), control23 (blend values, tamper rejection, blend bounds, form restore), fsm23 (interpolation ends at stand stiffness). Host-only ctest 4/4 and `run-port-tests.sh` 293 checks locally; full 224 rebuild ctest **5/5**.
- Sim2Sim (224, elastic band as the external support premise): band-supported FixedStand max joint error **0.039 rad** (right ankle) → R2+A first-try policy entry → band off → solo policy standing 30 s, zero drift; W-walk regression 0.80 m in 5 s with clean stop, no latch.
- Boundary, deliberately documented: FixedStand with the band fully removed still falls (forward, tilt latch) even with the authored gains — position PD without gravity feedforward is a posture shaper under external support, not a solo stander. A true one-button solo stand-up would need model-computed gravity feedforward (`stand_gravity_ff`, feet-grounded inverse dynamics); deferred as optional M2 because the attended flow (sling-supported stand → policy entry → slack sling) is validated end to end above.
- Deployment state: 224 rebuilt and validated; PC2 file sync + aarch64 rebuild pending (PC2 SSH unreachable at session end). The controller is never started remotely.

## Remaining work / missing inputs

- [x] Obtain/train a compatible `Unitree-G1-23Dof-Flat` policy and preserve its actual exported configuration and external weights. (2026-09-17: trained on 192.168.3.224; `model_4900.pt` checkpoint selected after a mid-run PPO collapse; provenance in `model/g1_23dof/`.)
- [x] Verify the policy in a matching 23DoF MuJoCo scene, including failure and recovery cases. (2026-09-17: sim2sim closed loop on 192.168.3.224 — entry, zero-command standing, 0.2 m/s walking, stop, L2+B latch and band-assisted recovery all pass; details above.)
- [ ] Confirm robot model/firmware, PC2 architecture/OS, DDS interface and SSH access. (2026-09-18: PC2 confirmed aarch64/Ubuntu 20.04/eth0, `mode_machine=4`; robot firmware version string still to be recorded at first attended session.)
- [x] Build and run offline checks on PC2 (aarch64 if that is the actual architecture). (2026-09-18: native aarch64 build, ctest 5/5, `--check` PASS; controller deliberately not started.)
- [ ] Validate actual DDS timing, mode reporting, ownership checks, remote controls and motor behavior under attended suspension, then controlled ground tests. (First attended session 2026-09-18: controller launched on hardware via the L1+X guard, startup gating passed after RPC `ReleaseMode()`, per-joint `tau_est ≈ kp×(cmd_q−fb_q)` confirmed real motor execution; standing blocked by the stand-gain root cause now fixed in software — see "Stand-gain restoration". Re-validate with the stand gains.)
- [x] On-hardware stand-gain validation (2026-09-18 16:16): START stands unaided, R2+A enters the policy on the first press, standing/walking confirmed by the operator; no fault latches during stand or locomotion. Details in "Hardware validation" above.
- [ ] Optional M2: model-computed gravity feedforward if a solo (no-sling) FixedStand is ever required; not needed for the attended flow.
- [ ] Optional: retrain without the mid-run PPO collapse (new seed or resume from `model_4900.pt`) and recalibrate the two safety guards for the new action profile.

Hardware contact so far: DDS round trip, FSM transitions, safety latches and motor torque execution have been verified on the robot under attended suspension; no untended motion has been commanded and the controller has never been started remotely.
