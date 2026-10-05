# 末层换入另一个目标的文字，仍未改变抓取对象

2026-10-05。使用当前 Cosmos3 Nano LIBERO 权重，没有新增训练。这轮确实做了局部网络干预，但没有找到原始牛奶任务抓错的根因。

我们怀疑：末层注意力读到另一个目标后，后面的 MLP 可能把动作变化抵消。于是选用上一轮自然能按名称抓对的 15 厘米／未来噪声198配对，双向交换“milk box”和“cream cheese”的文字状态。两句都11个token，目标都是两个token、位置相同，避免把句长变化混进来。预登记主预测是：只替换这个接口，就能让两个方向都转抓另一个名称指向的物体；固定 MLP 响应是辅助检验。

干预只发生在首轮预测的最后一层，即第36层（零编号 L35），覆盖30次去噪。保留当前动作的读取者、视觉和其他输入，只把目标词开始至文字结尾的 K/V 换成另一条自然指令的状态，再把新结果送给16行动作。后续七轮继续使用原接收者指令，按真实反馈原生推理，全部八条都执行。

| 首轮条件 | 原指令为牛奶盒 | 原指令为奶酪盒 |
|---|---|---|
| 自身替换：完整计算但不换内容 | [牛奶，64–68](future-instruction-interaction-execution-egl/closed-loop/x15_milk_box_from_cream_cheese_V198_A195_live_self/actual.mp4) | [奶酪，67–71](future-instruction-interaction-execution-egl/closed-loop/x15_cream_cheese_from_milk_box_V198_A195_live_self/actual.mp4) |
| J：换入另一个名称的文字，MLP正常响应 | [牛奶，64–68](future-instruction-interaction-execution-egl/closed-loop/x15_milk_box_from_cream_cheese_V198_A195_suffix_joint/actual.mp4) | [奶酪，67–71](future-instruction-interaction-execution-egl/closed-loop/x15_cream_cheese_from_milk_box_V198_A195_suffix_joint/actual.mp4) |
| C：同样换文字，MLP用本次调用原生读法下的输出 | [牛奶，64–68](future-instruction-interaction-execution-egl/closed-loop/x15_milk_box_from_cream_cheese_V198_A195_suffix_joint_native_mlp/actual.mp4) | [奶酪，67–71](future-instruction-interaction-execution-egl/closed-loop/x15_cream_cheese_from_milk_box_V198_A195_suffix_joint_native_mlp/actual.mp4) |
| 随机：同接口、量化后强度匹配的随机变化 | [牛奶，64–68](future-instruction-interaction-execution-egl/closed-loop/x15_milk_box_from_cream_cheese_V198_A195_matched_random/actual.mp4) | [奶酪，67–71](future-instruction-interaction-execution-egl/closed-loop/x15_cream_cheese_from_milk_box_V198_A195_matched_random/actual.mp4) |

数字是首个严格抓起窗口：双侧指垫同时接触、抬高超过2厘米，连续五条记录。八条均保留原目标，主预测为 [rejected](future-instruction-interaction-execution-egl/primary-prediction.json)。抓起不等于放进篮子。

内部也没有支持“MLP几乎把变化反向抹平”这个简单图景。首个去噪调用中，两个方向的 attention／MLP 输出差余弦为 −0.035／−0.043，合量范数是 attention 差的2.168／2.132倍。这些仅描述数组加法，不能称为牛奶概念被保留、增强或压制。J/C 在这个首调用的八个端点逐字节相同，可以直接比较；此后两条轨迹输入会分开，不能把跨轨迹的 MLP 差当成同输入因果效应。

随机只匹配投影前的实际 BF16 总范数，误差不超过2%；经过输出投影后的强度并未匹配。结果否定的是这组场景中“末层后缀替换足以双向换目标”的具体预测，不能推出语言无用或找到了运动／语言脑区。也不能排除更早层、其他通路或后续原指令反馈的作用。原始“milk”指令在15厘米位置的问题、失败的噪声195以及半程撤刺激检验，都没有被本轮解决。

首次物理启动因 EGL 设备解析失败，发生在环境创建完成前，零预测请求、零动作。保留 [失败记录](future-instruction-interaction-execution/wrapper_failed.json)，只用指定 EGL 设备的 [恢复入口](code/run_cosmos_milk_instruction_interaction_egl.py) 在新目录执行原八条；科学条件和首轮预测没有重算或改动。

[实际分析与图](instruction-interaction-analysis/analysis.json)、[可复核 NPZ](instruction-interaction-analysis/arrays.npz)、[240处原始捕获](instruction-interaction/)、[完整物理轨迹与视频](future-instruction-interaction-execution-egl/closed-loop/)及[逐文件 SHA](publication-manifest.json)一起保留。

<details>
<summary>技术核验与复现入口</summary>

首轮8×30共240个真实调用、17520次官方 attention dispatch；后续56次预测、1680个模型调用。CPU分析重读240处捕获、64份完整预测、1024次官方动作转换、49对后续完整噪声，以及八条严格抓起轨迹。两个自身替换完整复现旧自然轨迹。原始数组逐字节门与浮点范数统计分开；CPU分析没有新模型或物理调用。

每个调用保存实际 `x,W0,W1,m0,m1,Z0,Zdonor,Zselected`。`r0=BF16(x+W0)`、`r1=BF16(x+W1)`；比较 `BF16(r1+m1)` 和 `BF16(r1+m0)` 相对 `BF16(r0+m0)` 的变化。C中的前者是该调用的局部候选，不是J整条轨迹的值；没有构成完整 attention×MLP 四格实验，也没有证明唯一中介。

冻结代码：[producer](code/probe_cosmos_milk_instruction_interaction.py)（745c4d88）、[执行主体](code/run_cosmos_milk_instruction_interaction.py)（1e9e5ab0）、[EGL恢复](code/run_cosmos_milk_instruction_interaction_egl.py)（617a1169）、[CPU分析](code/analyze_cosmos_milk_instruction_interaction.py)（554e0669）。完整SHA与来源见各阶段的 complete/provenance。代码依赖原A6000冻结路径及合格供体，不是跨电脑一键包；完成和失败目录都不允许覆盖重跑。

方法动机来自 [The Hydra Effect](https://arxiv.org/abs/2307.15771) 中LLM下游补偿的研究；它不证明Cosmos的MLP具有同样机制或记忆功能。

</details>
