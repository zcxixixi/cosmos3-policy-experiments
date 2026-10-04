# VLA 内部因果定位：文献与 Cosmos3 可验证问题

核验日期：2026-10-03。仅依据原论文、作者仓库和本地脚本阅读；本轮未运行模型。下述文献均没有验证 Cosmos3 社区 LIBERO checkpoint。文献事实与拟议测试分开陈述。

## 三篇核心论文

**Not All Features Are Created Equal（v1，2026-03-19）** 使用同场景换指令、空提示、跨场景/随机种子 activation injection，并结合逐 token SAE、概念消融与动作线性探针。研究包括 π0.5、OpenVLA-OFT、X-VLA、SmolVLA、GR00T N1.5 和无语言 ACT。同场景的供体状态可改变目标选择，跨场景交换可拉向供体空间轨迹；这支持被替换状态携带行为相关信息，不能单凭全状态交换把它称为孤立的物体概念。语言重要性依 suite 歧义程度变化。

特别限制：附录 F.3/G.5 承认 full-layer hooks 破坏残差连接、夸大 SAE 效应；改为 MLP-targeted 后，2–5 概念特征消融无显著作用（p=0.975）。探针方向投影后探针失效，不等于真实策略必需该方向。大倍率 boosting 导致失败也不能排除分布外损坏。G.4 的 step0≈0%效应与 G.6 的 step0 后0%成功存在口径张力，不宜拿来证明首步普遍决定轨迹。[原文 §3–4、F.3、G.4–G.6](https://arxiv.org/html/2603.19233v1)

**LangForce（最新 v7，2026-05-29）** 将语言忽略解释为训练数据低条件熵：I(语言;动作|视觉)≤H(语言|视觉)。64 个可学习 action queries 作为 VLM→DiT 瓶颈，以不同序列位置建立视觉 prior / 多模态 posterior，并增加 LLR 训练目标。重要版本差别：v7 §2 的 vision-only 对照是重新训练的策略；不能当作同一冻结策略推理时去掉语言的实验。LIBERO Goal 的视觉对照9.8%、完整模型97.4%，支持同场景多目标评价的必要性。训练改进及消融支持方法效用，未定位冻结模型内部语言丢失层。早期版本名为 BayesianVLA；引用应标注版本。[原文 §2–3、4.5](https://arxiv.org/html/2601.15197v7)、[作者代码](https://github.com/ZGC-EmbodyAI/LangForce)

**IGAR（v1，2026-03-06）** 的 ICBench 固定视觉、只改变不存在属性或不可满足空间关系，比较 π0、π0.5、OpenVLA-OFT；LGS=正常提示成功率−矛盾提示下原任务成功率。IGAR 用 hidden spike 找 sinks，选 attention head/query 并把 text sinks 的注意力预算转向非 sink 文本；注意公式(9)预算来自文本 sinks，不能概括为直接把视觉注意力转移到文本。正常提示是已有保真对照。它提供内部干预影响行为的证据；矛盾下原任务少成功，仍不足以区分语义理解、悬停和一般损坏，必须补可满足替代目标。IVAR/注意力图本身不是因果定位。[原文 §3–5，特别式(7)–(10)](https://arxiv.org/html/2603.06001v1)

## Activation patching 应如何解释

clean→counterfactual 的恢复支持在指定背景中的充分性；反向替换造成损失支持对该扰动的必要性。两方向可能因冗余或串联依赖而不对称，零效应不证明信息不存在。先粗粒度扫描再测细粒度组件/路径；普通组件 patch 会影响所有下游，不能独自区分直接传递与中介。[Heimersheim & Nanda，§2–3](https://arxiv.org/html/2404.15255v1)

噪声、置零、token 选择、指标和多层滑窗都可能改变定位。优先自然且可满足的最小替换；同时报告原始效应，避免接近零的 clean/counterfactual 分母制造夸大恢复率；多层效果属于整个窗口，不属于中心层。LLM 的 logit difference 建议不能直接照搬到连续动作；这里需要目标方向、速度和执行行为的对比指标。[Zhang & Nanda，§3–6](https://arxiv.org/html/2309.16042v2)

## 面向冻结 Cosmos3 的具体测试（建议，尚未执行）

本地 `probe_cosmos_state_exchange.py` 可读取层12/30 GEN action token，在首个策略查询内每次去噪迭代替换供体状态；同提示、相同初始 action/future-video noise 与 self-patch 是其已有检查。`probe_cosmos_readout.py` 区分最终 norm 后 hidden、动作 velocity、scheduler 更新和可执行动作。这些脚本显示可观测接口，不自动证明实验结论。UND 文本、GEN 视频、子层/路径接口需按当前 diffusers 实现验证 mask、缓存与 hook 的确切位置。

1. **先构造有行为差异的最小对。** 固定同一完整模拟器状态、图像、随机种子、噪声、schedule、proprioception、prompt JSON 和 domain；只将 milk 指令换成场景里真实可抓的 butter/cheese 指令。附加同义改写、空提示、不存在物体提示。Goal suite 同场景可满足双目标为主，矛盾提示为辅。无独立供体差异时，不定义“恢复正确行为”的百分比。
2. **记录在哪里出现差异。** 逐层、逐 token、逐去噪迭代记录 UND text/vision 与 GEN action/video 状态差异；核对首个查询与首个去噪 forward 是不同时间轴。Layer norms、cosine、CKA、linear probes只用于提出假设，不能证明转化失败位置。
3. **分块、双向、分时间 patch。** 在合法 token 对齐下分别交换 UND 指令位置、UND 视觉位置、GEN action、GEN video；先层12/30并逐层细化。每次仅改一个块；设置同状态 self-patch、同义供体、无关供体、等范数随机方向和不同种子供体。区分单个去噪 step 与全部30 steps、只首个查询与全部闭环查询。若缓存 UND，要确认 patch 作用于 GEN 真正读到的 K/V；仅替换未使用副本不算消融。
4. **测 UND→GEN 的路径，而非只换整层。** 在实际实现允许时，只交换 GEN action queries 读取的 UND-text K/V，保留UND vision与GEN self-attention；再单独测GEN video→action。让下游其余组件维持recipient原生计算，比较局部attention输出/MLP输出与完整残差状态交换。阻断所有UND连同其metadata，仅支持整条通路贡献，不能定位语言。实装时以self-patch验证残差、mask、RoPE位置和scheduler全部保持。
5. **用真实输出头解释第一步。** 在norm后的hidden重放原LIBERO动作头，分别报告每动作维度的velocity差、scheduler后的动作差；预测clean action须按实际flow convention推导和校验。用实际动作头行空间比较hidden差异投影，避免把4096维状态差大而10维输出差小称作“坏头”。Layer12的探针好而真实输出差小，仅说明可解码，不说明下游本该使用。
6. **以闭环与语义特异性确认。** 在同一初始模拟器状态重复独立seed，记录先靠近哪个物体、抓取身份、目标距离、gripper、替代任务完成率及原任务完成率。单首步变化只能证明局部输出影响；供体一致、双向、跨布局的可满足目标切换，且正常提示/无关供体不损坏，才加强内部因果归因。首个查询注入之后恢复原生策略，测试效应是否持续；持续也可能是环境反馈累积，不等于“内部锁死”。

## 最低可发表表述

只有层状态/注意力变化：**表征对输入敏感**。真实readout变化：**输入差异传到动作读出**。受控局部patch引起对应目标行为变化：**该内部位置在此条件下因果影响目标选择**。通过双向、路径、语义/范数控制和独立布局复现：才进一步主张**某条信息通路介导语言到动作**。单次跨场景GEN交换、全UND移除、探针准确率、失败或预测视频看起来合理，均不能单独证明“milk概念被定位”“语言已理解但输出头坏了”或“Cosmos内部规划形成后才失效”。

## 功能分工：语言、目标与运动（2026-10-04）

下列证据分别涉及可干预的运动特征、已有通路之间的功能差异、以及模型计算与脑活动的对应；三者不能合并成“VLA 已有类似皮层的语言区和运动区”。这些研究均未验证当前 Cosmos3 checkpoint。

1. **Häon 等，Mechanistic Interpretability for Steering Vision-Language-Action Models（2509.00328v1；CoRL 2025）。** 作者用 FFN 输出权重的词表投影选择 fast/slow、high/low 等单元组，再在推理时覆盖这些组的激活。OpenVLA 的 LIBERO 实验及 π0-FAST 的 UR5 实验表明部分干预可改变运动幅度或运输高度，并比较未干预、改提示和随机单元对照。这是内部特征能够因果影响动作的证据，而非完整功能区域或目标身份回路的证明。特别注意：UR5 checkpoint 先经机器人数据适配；速度实验只执行未干预动作，其他方案在相同观测上比较预测位移，不能把全部速度结果称为独立真实闭环执行。论文也报告 high/fast 未明显超过原生，并承认词义聚类可能混淆“慢”和“卡住”。可迁移到 Cosmos3 的是候选单元、self/随机控制与行为检验；具体层号、词义标签及词表投影不能直接移植到其连续动作头。[原文 §3–4、§6](https://arxiv.org/html/2509.00328v1)、[CoRL 正式论文](https://proceedings.mlr.press/v305/haon25a.html)

2. **Grant 等，Not All Features Are Created Equal（2603.19233v1），§4.5。** 在 π0.5 中，错任务 expert 状态注入导致主动向错误位置运动，而 PaliGemma 注入更易导致停滞；在 SmolVLA 的 732 个 MetaWorld 跨任务配对中，expert 与 VLM 注入的 source-like 行为分别为 15.8% 和 9.0%。结合目标/状态探针，这支持既有 VLM 与 action-expert 通路在测试背景中的功能差异。限度是：通路划分本身来自架构，整状态注入还携带位置、轨迹等信息；停滞不能独自证明目标信息被删除，探针可解码也不能证明策略使用。旧节所述残差/hook 与 SAE 消融局限继续适用。对 Cosmos3 可迁移的是分通路、分输入内容的干预和目标/运动指标对照；不能直接把 UND/GEN 命名为语言区/运动区，尤其 GEN 同时处理视频与动作。[原文 §3.2、§4.5](https://arxiv.org/html/2603.19233v1#S4.SS5)、[附录 E.1、F.3、G.5](https://arxiv.org/html/2603.19233v1)

3. **Kumar 等，Shared functional specialization in transformer-based language models and the human brain（Nature Communications，2024）。** 作者分析 BERT attention heads 对词间上下文的实际变换，并以自然故事聆听时的 fMRI 做编码预测；不同 head 的变换对不同语言皮层位置有差异化预测，其对应关系呈层级与上下文长度梯度。这支持模型计算具有可测的功能分化，并与部分脑活动组织存在对应。这里的脑对应证据是预测关联，不能推出模型与大脑采用同一机制、head 等同一个脑区，或已经建立语言与运动的因果双重分离。可迁移的研究思路是按具体计算/任务区分组件，而不是只看激活强弱；该论文未研究 VLA 运动控制，不能替 Cosmos3 提供功能区标签。[原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC11217339/)、[作者实验室记录](https://hassonlab.princeton.edu/publications/shared-functional-specialization-transformer-based-language-models-and-human-brain)
