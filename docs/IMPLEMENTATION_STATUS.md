# G1 23DoF implementation ledger

Goal: adapt upstream wbc_fsm on feature/g1-23dof-port for 23DoF standing and velocity policy, with integrated safety. Preserve upstream main and legacy 29DoF sources. Do not deploy unvalidated policies or start robot motors automatically.

Baseline: upstream b352409. Early commits added mapping, partial config, environment selection, and a standalone safety helper. These were not yet a working controller.

Current implementation tasks:
- [ ] Full validated policy configuration, 80 observation / 23 action ONNX adapter
- [ ] 23DoF fixed stand and FSM routing; prevent entry to 29DoF modes
- [ ] Integrated latched safety, receive-before-compute-before-send
- [ ] DDS synchronized snapshots, fresh-state/command watchdog and strict configuration
- [ ] Dependency-independent regression tests, controller compile validation
- [ ] Simulation with a matching trained 23DoF policy
- [ ] Isolated aarch64 build and attended suspended hardware validation

Parallel agents have repeatedly failed at model capacity before producing any edits. Root continues implementation locally; this ledger is the handoff source of truth.
