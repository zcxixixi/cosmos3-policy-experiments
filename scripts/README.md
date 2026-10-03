# 实际运行脚本快照

archive保存A6000上使用的实验助手，不是一个已经打包好的独立Python库。脚本内ROOT、LIBERO导入路径、模型目录和输入状态依赖原环境，先看[推理设置](../provenance/runtime.md)。

采集与执行脚本使用有MuJoCo/robosuite/LIBERO的仿真环境；probe和policy server使用有torch/diffusers的GPU环境。完整复现需要输入快照与内部states.pt，权重和这些大文件没有纳入Git。

没有运行中的训练、自动下载或自动发布功能。此仓库当前职责是保留实验、结果和所用代码。不要把离线预测、逐层替换的局部效应当作机械臂任务成功。
