# 推理设置与复现范围

- 权重：fwd4xl/cosmos3-nano-policy-liberoall-5k，revision `64798337c642c53f9e22332554ae92f76a86cf04`。
- Diffusers：`fef717ffb01f407d2637584ed936c16db908587a`。
- cosmos-framework：`cf5d68c00d97ccd2480a2320ed652b92dec63102`。
- RTX A6000 48GiB，BF16；Python3.10，PyTorch2.14/CUDA13；模型与仿真在两个独立环境中运行。
- 两个当前RGB视角：各256×256，旋转180度后拼接为512×256；不是历史4帧输入。
- policy模式，domain libero（5），tier256，fps20，16步动作块；原生Flow UniPC、30次去噪、shift1、guidance1。
- JSON提示补充 `idle_frame: 0 out of 16.`；Diffusers运行时LIBERO raw动作维度设为10。
- 模型10D：xyz3、rotation6d、gripper1；按官方normalizer_stats的q01/q99反归一化，转成原生7D并clip到[-1,1]。XYZ是无量纲的OSC控制命令，不是直接以米表示的位移。
- 归一化夹爪约+1张开、−1闭合；MuJoCo原生7D命令+1闭合、−1张开。
- 主要环境：LIBERO_OBJECT task7，初始状态0，10个idle步后的同一快照；闭环种子198+query，8轮×16动作=128步。

脚本快照保留实验时的主机路径和数据目录依赖。仅凭本仓库不能保证任意机器一键复现；运行时仍需上述代码、权重、LIBERO资源和原始内部states.pt（约数百MiB，未上传）。scripts/archive对应的快照经过语法解析，相关实际运行结果已保留；没有声称在本地无GPU电脑重跑了全部实验。

仓库不含权重、凭据或巨大的hidden-state文件。report.html可下载后用浏览器打开，图和视频已内嵌；JSON/MP4/PNG可单独核验。

尚未完成官方framework server与自定义Diffusers runner的完整逐数值交叉验证，不能排除所有输入格式/实现差异。源码目录在核验时没有变更模型库；实验hook仅作用于当次运行。

2026-10-03补充：实验09找到并关闭当前调用额外的19个system tokens。两布局、两目标词、三对应噪声种子及位置编号控制，共36组离线预测；新增5条128步执行，原设置复用3条。关闭后仍出现相同错误目标，不能把失败仅归因于多余system。实验10沿用关闭system的设置，四份固定初态分别以195、196、198为起始噪声种子，8轮查询使用base_seed+query；除该开关之外，仍保留历史像素处理和时间表。12条中两条复用09、两条为首次交换、八条为新增重复，不能按12独立场景解释。

实验11已对同一真实无system输入完成一个官方原生transformer forward。814参数键全部匹配，动作输入边界逐项相同；最终readout相对RMS差0.858%，同一个官方head重放的前10维去噪velocity差0.325%。原生计算使用独立Python3.12/PyTorch2.11、现有cuDNN9.24，主体BF16/时间MLP FP32；没有安装新依赖或下载权重。原生padding通过明确索引排除后采集。该首轮检查不是完整官方server或新闭环的等价核验。[原始数组与限制](../experiments/11-official-transformer/README.md)。

libero-milk-task.bddl来自安装的LIBERO资源，遵循上游[LIBERO仓库](https://github.com/Lifelong-Robot-Learning/LIBERO)的许可证。它记录任务默认摆放范围，不能据此证明这份社区权重实际训练数据的范围。

实验11随后补完两场景×两指令、种子198的四组官方核心30次采样。沿用已编码输入、FP32初始噪声和历史显式时间表，官方网络从自己的上一状态连续预测，未插入保存的隐藏状态。最终归一化XYZ RMSE为0.001058–0.001864；最后一次readout相对RMS差4.63%–5.02%。FlowUniPC在原有CPU兼容环境运行，旧记录在CUDA运行；固定同一组更新值的CUDA重放与CPU最终动作RMSE为3.68e-11–6.66e-9，但没有测试这些舍入差异反馈给网络后的影响。完整官方像素/VAE预处理、默认时间表和闭环尚未核验。[四组真实数组、单位和限制](../experiments/11-official-transformer/README.md)。

图像编码的尺寸还需区分：两视角拼图缩放为320×160内容，底部reflection pad到320×192后进入VAE，随后裁切latent中的padding区域。最终latent高度对应160像素内容，不能倒推VAE只看过160像素高的图。两套实现都有这个编码后裁切流程；此前查到的uint8／float像素处理差异仍需单独测试。

实验12按公开官方顺序缩放uint8像素、clamp/round后normalize，VAE仍为现有BF16实现；同时采用同源FlowUniPC默认`set_timesteps(30, shift=1.)`。16组分项预测包含旧／新像素×旧／新时间表×两场景×两指令，种子198；四条新增闭环每轮均采用新像素、新时间表、关闭system、seed198+query。两个既有模型库工作区无改动，没有安装依赖或下载权重。旧设置四组逐数组精确复现实验09；新闭环初态／首轮照片／预测动作精确对应，四条抓取对象与旧对照一致。单帧／17帧首latent在当前VAE里精确相同，但未核验完整官方VAE或server。[原始数组、逐步执行和限制](../experiments/12-pixel-schedule/README.md)。
