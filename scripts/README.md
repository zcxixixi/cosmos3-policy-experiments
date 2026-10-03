# 实际运行脚本快照

archive保存A6000上使用的实验助手，不是一个已经打包好的独立Python库。脚本内ROOT、LIBERO导入路径、模型目录和输入状态依赖原环境，先看[推理设置](../provenance/runtime.md)。

采集与执行脚本使用有MuJoCo/robosuite/LIBERO的仿真环境；probe和policy server使用有torch/diffusers的GPU环境。完整复现需要输入快照与内部states.pt，权重和这些大文件没有纳入Git。

没有运行中的训练、自动下载或自动发布功能。此仓库当前职责是保留实验、结果和所用代码。不要把离线预测、逐层替换的局部效应当作机械臂任务成功。

2026-10-03新增实际运行快照：system-prompt probe/rollout、identity-position rollout/repeats、图表脚本及官方transformer首轮及四组30次采样对照。`summarize_cosmos_identity_position.py <本仓库路径>`只读原始轨迹并写实验10的汇总，已在本地重算。官方网络脚本保留原主机环境和cuDNN要求，复用已编码输入检验网络核心，显式 `--run --sample30 --scheduler-python <已有环境Python>` 才运行连续30次采样；没有宣称完整官方服务或闭环等价。

实验12新增四个实际运行快照：`check_cosmos_official_pixels_cpu.py`核对官方整数像素处理，`probe_cosmos_pixel_schedule.py`做16组真实预测，`rollout_cosmos_pixel_schedule.py`执行四条128步MuJoCo闭环，`present_cosmos_pixel_schedule.py`从真实动作和轨迹重新计算汇总与图。预测脚本在当前程序内采用官方像素函数和默认时间表，不替换隐藏状态；不代表完整官方服务等价。1帧／17帧首latent核验仍使用当前VAE。结果及范围见[实验12](../experiments/12-pixel-schedule/README.md)。

实验13新增路径助手、CPU多次反馈测试、40组实际预测、六条闭环与重算画图脚本。`cosmos_attention_routes.py`在原attention processor中限制哪些查询读另一句文本、哪些画面K/V固定为来源；不修改模型库。全路径精确重放与阻断动作精确恢复通过，但这六条执行没有改变抓取对象。全GEN边界只在运行时比较，未全部落盘；FP32未来视觉初始噪声也未单独保存。细节见[实验13](../experiments/13-language-routes/README.md)。

实验14使用官方已有 `pi05_libero`。`probe_pi05_target_control.py`保存12组首段输入与输出；`probe_pi05_replay_precision.py`核对四次直接调用。服务与旧离线输出的差异没有解决，保留失败记录；后来的`serve_pi05_target_control.py`在同一服务核对实际与保存原输入的两次计算，逐字节一致才由`rollout_pi05_target_control.py`执行第一份原生7维动作。四条每条128步，预测10步、执行5步再规划。历史服务源码可从提交`67cde60`和`a4b2000`查看，当前快照是四条实际执行用的版本。`plot_pi05_actual_eef_xy.py`仅从真实轨迹计算水平距离、绘图；不调用模型。见[实验14](../experiments/14-pi05-target-control/README.md)。
