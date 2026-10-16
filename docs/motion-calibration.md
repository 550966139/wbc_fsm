# 动作校准层(MC)战役备忘录 — motion calibration campaign

状态(2026-10-15 收官更新):**战役定位重定——交付物是合格的参考数据,训练线冻结**(用户决策,详见 `data-quality.md` §1)。S0-S8 工程验证完成,S9 取消。执行流水与真实数字见 `IMPLEMENTATION_STATUS.md` 的 MC campaign 条目;问题根因分析与后续路线以 **`data-quality.md`** 为权威。

## 1. 动机与架构

- 手工 R 系列手术逐舞一对一,不可持续;用户方向:动作数据应在 GVHMR 出口被严格校准,一个"动作特征模型/约束校准层"而非下游补丁;
- 目标链(任何 take 同一条链,零逐舞常量):
```
视频 → GVHMR(不动) → [可选] calibrate_smplx(SMPL-X 域 z 锚定,泛化件待做)
  → GMR 重定向(不动;ws 版已修盆地翻转) → project_g1 全自动 G1 投影
  → 不变量验收(data-quality.md §3,机器 PASS/FAIL)
  → [训练线冻结中;解冻条件=数据全过] transitions → NPZ → 训练 → 部署
```

## 2. 已验证结果(S0-S8;细节与命令在账本)

- **尺子双复现门槛通过**(防特调):take6 trans_r2 ZMP p50 −4.0cm/R 89% 精确复现账本;两舞原始浮脚 +0.053;take2 估计滑步 p90 0.363/0.378 vs 账本 0.375;
- **新量化的病**:GVHMR 世界系 Y-up;世界根垂直漂移 take6 7.5cm/take2 10.9cm;估计抖动 take2 p95 0.29 rad/帧;
- **project_g1 v1**(同一组常数双舞):单支撑 ZMP 出界 take6 100/87%→16/33%、take2 →13/13%(手工 R3 后 69/69%、在役 full_r6 71/70%);浮脚 5.3cm→0.5cm;28-39 秒/支。
  ⚠️ **v1 未达验收**:翘脚/悬脚/穿地(−9~16mm)/抖动四类缺陷由用户目检发现,根因与修复见 `data-quality.md` §2-§3;v1 工件不用于训练;
- **零训练复放(S7)**:F18(盲观测策略)× MC 参考 0/3,死点关节 6@f=550 与原参考同签名——参考治不了盲策略;
- **探针(已取消的对照线,仅存档)**:15k 冷启动曲线 MC 109/0.080 vs V2 122/0.066(奖励/error_body_pos,@13-15k 同迭代)——校准参考早期更难跟(钉足后不能靠滑步作弊);窗口存活评测按冻结决策未执行。

## 3. 工具、环境与工件位置

| 件 | 位置 | 说明 |
|---|---|---|
| `measure.py` | `wbc_fsm/scripts/motion_calib/` | G1 域尺子(ZMP/接触/滑步/浮脚;XML 经 `MC_G1_XML_DIR` 或 `default_g1_xml()` 查找) |
| `measure_smplx.py` | 同上 | 估计域尺子(Y-up;吃 mc_dump 的 npz) |
| `project_g1.py` | 同上 | 自动投影求解器 v1(接地→钉足→ZMP 迭代→去折角) |
| `play_compare3d.py` | 同上 | MuJoCo 原生双窗 3D 对比播放(scene 地板;显示平滑 k 可调,不动数据) |
| `make_compare_preview.py` | 同上 | HTML 骨架对比(次级,已由 3D 播放器取代) |
| `mc_dump_smplx.py` | 同上(**在 224 上跑**,gmr env) | GVHMR .pt → npz 一次性 dump |
| 环境 | wsk `~/miniconda3/envs/mc` | py3.11 + mujoco 3.5.0 + torch-cpu + onnxruntime |
| 数据/工件 | wsk `~/sim2sim_g1/mc_work/` | 两舞全链 CSV(58+ 文件 md5 对账)、MC 输出、报告 JSON |
| 224 侧 | `~/dsh-workspace/mc_work/`、`mc_dump_smplx.py`、run `2026-10-15_15-31-15_take6MCprobe` | 探针 run 与转换暂存 |
| 观看服务 | wsk `~/sim2sim_g1/mc_web/`(端口 8231) | 静态页(SVG 版已弃用为存档) |

## 4. 预注册门槛的证据修订(留档)

- ZMP 出界 ≤15% 物理不可达(根平移+0.12m 预算饱和于 21-31%)——F17 式"门槛 vs 物理"教训在数据层重演,门槛修订为"显著优于被替代方案+预算内忠实度";
- 后续所有门槛以 `data-quality.md` §3 不变量(最坏帧)为准,百分位仅作参考输出。

## 5. 风险与边界

- 校准=物理一致性≠真值:特征保持靠忠实度预算+窗口级最小修改;
- G1≠人的形态差永久存在:project_g1 是链上永久环节;
- 在役工件零改动(F10/F17c/PC2 未动);原始参考与 take6V2 主线零污染。
