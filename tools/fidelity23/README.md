# fidelity23 — G1 23DoF 动作保真对比工作台

行业标准的动作保真评估 + 本地常驻 Web 对比服务。指标服务于**优化路由**（哪段差多少 →
动哪个杠杆），不是观赏报表。

## 快速开始

```sh
cd wbc_fsm/tools/fidelity23
python3 -m venv .venv && .venv/bin/pip install numpy mujoco   # 首次
.venv/bin/python evaluator.py           # 评估 data/runs/ 下缺 metrics.json 的采集
python3 server.py                       # 常驻服务（标准库，零依赖）
# 浏览器打开 http://localhost:8230
```

重启后只需 `python3 server.py`（数据与指标都在 `data/`，评估结果不丢失）。

## 数据流

```
224 验收采集（脱吊，900Hz 双主题 npz）
  └─ rsync 到 data/runs/<版本>/<tag>/{<tag>.npz, <tag>_sport.npz, <tag>.t_entry}
       ├─ data/runs/<版本>/meta.json  指定该版本使用的参考链（refs/ 下的 NPZ+CSV）
       └─ evaluator.py 逐 run 计算 → <tag>.metrics.json（server 刷新即见）
```

新增一个候选版本的步骤：`mkdir data/runs/F16 && scp ... && 写 meta.json && .venv/bin/python evaluator.py`。

meta.json 可选键 `segments` / `kick_window` / `ending_drift_from` 描述**该版本动作自身的结构**
（分段边界、踢腿顶点窗、结尾漂移起点，单位秒，舞蹈时钟）——不写则用 take6 缔造值
（开场 0-8 / 踢腿 8-12 / 律动 12-28 / 结尾 28-36.6，踢腿窗 8.5-12.0，漂移自 28.5 起）。
换一支舞（如 take2）必须从其参考链重新标定这些窗口；舞蹈窗/序列窗随末段结束时刻自动推导。

## 指标定义（行业分类法）

| 指标 | 定义 | 阈值/口径 |
|---|---|---|
| fsr_ref 足滑移率 | 参考支撑相中足底水平速度 >0.10 m/s 的帧占比 | GMD 分类法；同时报告 actual-contact 口径（`fsr_*`），差异大说明是"浮着蹭" |
| float 悬浮 | 参考支撑相中鞋底 >5mm 的时长占比 | 入场吊带期（前 ~5s）不计 |
| penetration 穿地 | 鞋底低于 −2mm 的时长/深度 | |
| mpjpe | 世界系 24 刚体位置误差（入场帧刚体对齐，复现控制器入场 yaw 对齐语义）p50/p90 | 分段成功率阈值 0.15 m |
| 关节误差 | 23 关节 vs 参考链 joint_pos，每关节 p50/p90 | |
| apex 顶点比 | 踢腿窗(8.5–12s) 左鞋底最高 / **本版参考链**同窗最高 | 注意：不同链参考不同，跨链比较看绝对 cm |
| drift | 结尾段(td≥28.5s) 骨盆净位移/路径长 | |
| ankle 余量 | 支撑踝距 −0.8727 rad 机械限位 | 0.15 rad 达标线 |

分类法来源：Mimicking-Bench (arXiv 2412.17730)、BeyondMimic (arXiv 2508.07200)、
Benchmarking with Motion Difficulty (arXiv 2512.07248)。

## 已知口径注记（2026-09-29 首批数据）

- 历史台账的"踢腿 86%/77%/73%"用旧链常数 36.6cm 做分母；本工具按各版自己的参考链计算
  （full 链参考约 34cm），同日同管线结果：F15(84.7%) ≈ F10(82.2%) > F13(78.1%)——
  此前"F15 踢腿大幅退步"系分母口径错误。
- F10 的 2026-09-28 原始采集为空文件，本目录数据为 2026-09-29 同管线补采（3 遍）。
- F13 轮次无 t_entry 文件，入场时刻由轨迹对齐推断（页面标 Ⓘ）。
- "滑动"的视觉感受主要由 float（三版均 17-22% 悬浮）贡献，真实贴地滑移三版都 <1.3cm/s。

## 文件

- `evaluator.py` 指标核心（CLI：`--data-root data --xml <g1_23dof.xml> [--force]`；XML 默认取本机 sim rig 的资产）
- `server.py` 常驻服务（端口 8230，`FIDELITY23_PORT` 可改；只读 data/，绝不写）
- `web/` 中文前端（ECharts 已 vendor，离线可用）
- `data/` 采集与指标（gitignore，不进版本库）
- 可选开机自启：`systemd/fidelity23.service`（`systemctl --user enable` 用，默认不装）

## 红线

服务只读本机 data 目录；不动控制器/训练代码；真机零接触。
