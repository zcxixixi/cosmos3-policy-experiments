# 推理设置与复现范围

- 权重：fwd4xl/cosmos3-nano-policy-liberoall-5k，revision `64798337c642c53f9e22332554ae92f76a86cf04`。
- Diffusers：`fef717ffb01f407d2637584ed936c16db908587a`。
- cosmos-framework：`cf5d68c00d97ccd2480a2320ed652b92dec63102`。
- RTX A6000 48GiB，BF16；Python3.10，PyTorch2.14/CUDA13；模型与仿真在两个独立环境中运行。
- 两个当前RGB视角：各256×256，旋转180度后拼接为512×256；不是历史4帧输入。
- policy模式，domain libero（5），tier256，fps20，16步动作块；原生Flow UniPC、30次去噪、shift1、guidance1。
- JSON提示补充 `idle_frame: 0 out of 16.`；Diffusers运行时LIBERO raw动作维度设为10。
- 模型10D：xyz3、rotation6d、gripper1；按官方normalizer_stats的q01/q99反归一化，转成原生7D并clip到[-1,1]。
- 归一化夹爪约+1张开、−1闭合；MuJoCo原生7D命令+1闭合、−1张开。
- 主要环境：LIBERO_OBJECT task7，初始状态0，10个idle步后的同一快照；闭环种子198+query，8轮×16动作=128步。

脚本快照保留实验时的主机路径和数据目录依赖。仅凭本仓库不能保证任意机器一键复现；运行时仍需上述代码、权重、LIBERO资源和原始内部states.pt（约数百MiB，未上传）。scripts/archive对应的快照经过语法解析，相关实际运行结果已保留；没有声称在本地无GPU电脑重跑了全部实验。

仓库不含权重、凭据或巨大的hidden-state文件。report.html可下载后用浏览器打开，图和视频已内嵌；JSON/MP4/PNG可单独核验。

尚未完成官方framework server与自定义Diffusers runner的完整逐数值交叉验证，不能排除所有输入格式/实现差异。源码目录在核验时没有变更模型库；实验hook仅作用于当次运行。

libero-milk-task.bddl来自安装的LIBERO资源，遵循上游[LIBERO仓库](https://github.com/Lifelong-Robot-Learning/LIBERO)的许可证。它记录任务默认摆放范围，不能据此证明这份社区权重实际训练数据的范围。
