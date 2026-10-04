# Cosmos3 奶盒位置阈值：局部机制假设、文献核验和干预

核验日期：2026-10-04。范围收敛到同一场景奶盒沿 X 位移 0 / 6 / 15 cm 引起的目标切换。本文是研究笔记和实验设计；没有运行 Cosmos3、GPU 或 SSH，没有新增机制实验结果。已读同目录 `latent-causal-literature.md`，这里补充 head / MLP / 短脉冲层面的可区分预测。

## 要先回答的两个问题

**预测 A：存在由局部视觉通路触发、在去噪过程中延续的目标分支。** 在固定 recipient 画面、同一 sigma / 初始噪声的一次去噪 forward 中，只给一个中层 GEN attention head 的 action queries 一次自然供体脉冲。脉冲超过某个有限幅度后，撤掉 hook，后续层和后续去噪步骤仍产生供体方向的动作；效应在 6→15 cm 的行为转折附近比 0→6 cm 明显。如果每次都要重新 patch 才保留供体效应，或者只有动作向量范数增大而目标方向不变，该强预测不成立。这里首先称“分支延续”，不称已发现吸引子。

**预测 B：某个晚层 GEN MLP 对目标相关信号进行局部覆写。** 同样的 attention 脉冲在进入该 MLP 之前有供体方向的因果效应，经过该 MLP 后效应明显减弱或反向；仅把这一个 MLP 的输出保持为未受脉冲时的自然输出，便能恢复最终动作效应。反向自然输出替换应把 0 / 6 cm recipient 拉向 15 cm 供体的动作方向。若效应主要损失在后续 attention、最终 norm / 输出头、scheduler，或所有 MLP 都不满足“冻结后救回”，则该局部覆写预测不成立。负的 attention–MLP 差向量 cosine 只能作为候选排序，不能证明覆写。

两个预测可以局部共存，因此先查明**首次稳定阻断脉冲的具体组件**：是一个 MLP、后续视觉 attention，还是跨去噪步的状态更新。需要用局部干预量化它，不能用首末层差异百分比替代。

## 两篇机制论文，以及脑科学类比的边界

### 1. VLA FFN 神经元干预：Häon 等，CoRL 2025

[Mechanistic Interpretability for Steering Vision-Language-Action Models，原文 §4 / §6](https://arxiv.org/html/2509.00328v1)；[作者代码](https://github.com/Physical-AI-Safety-Institute/mechanistic-steering-vlas)。[arXiv 记录](https://arxiv.org/abs/2509.00328)标明 CoRL 2025。

作者将 FFN 看作输入相关系数乘固定 value vectors 的和；按词表投影选语义簇，覆写部分中间神经元系数。OpenVLA 的 hook 放在 `down_proj`，实质改变其输入系数；π0-FAST 用改写的 FFN 实现同一操作。LIBERO 与 UR5 的干预能改变位移、速度和高度。真实机器人 slow / low 效应较明显，fast / high 接近基线；随机神经元和改提示是对照。论文也明确语义聚类可能混合“slow / careful / stuck”，含义会随模型、任务和时间漂移。

**可借用：** MLP 内部确有可干预的行为控制接口；候选需要自然激活与语义特异性对照。**不可借用：** 文本词表投影里的“milk”就等于 Cosmos3 的纯物体身份变量；大倍率 steering 改变动作也不等于该神经元正常负责选择奶盒。Cosmos3 的 GEN MLP / 连续 Flow velocity 与该文 autoregressive action-token 设置不同。

### 2. 物体定位的 head 因果定位：Schaumlöffel 等，2026

[Mechanisms of Object Localization in Vision–Language Models，原文 §3.3 / §10](https://arxiv.org/html/2605.19792v1)。这里按所核验的 arXiv v1 引用，未凭搜索引擎条目认定会议审稿状态。

作者先按层组阻断对 object-aligned visual tokens 的读取，再对单 attention head 做 source→base activation patch。source 有物体，base 用 inpainting 移除物体；patch 对象是 prompt token 的 head activations，评分使用去掉固定模板 token 后的 teacher-forced perplexity。大效应集中在少数头；分类和定位的关键头重合有限。再按 patch 排名累计 ablate heads，以相同数量低效应头为对照。论文也观察到负 mediation fraction，指出供体与 recipient 激活可能不兼容。

**可借用：** “哪个区域”应落到 object ROI tokens → 指定 query → 指定 head → 输出的因果边，而不只展示 attention 热图；用 head patch 后再消融做双向证据。**不可借用：** 把 LLaVA / InternVL 的层号搬给 Cosmos，或把视觉移位供体的全 head 输出说成纯身份概念。视觉 latent 的一个 token 也未必严格对应一个像素方块。

### 脑科学类比只提供实验逻辑

[Li 等，Robust neuronal dynamics in premotor cortex during motor planning，Nature 2016；作者机构页面](https://www.janelia.org/publication/robust-neuronal-dynamics-premotor-cortex-during-motor-planning)。这篇旧文用于解释控制逻辑，非近两年 VLA 证据。小鼠 ALM 单侧短时抑制后动作选择相关活动恢复；双侧抑制后不恢复，提示另一侧通路可补回。它支持同时测短脉冲与剩余通路，而非仅凭“扰动后来变小”命名吸引子。Cosmos3 没有同样的双半球结构；类比不提供任何具体层号或神经元结论。

## Goodfire 报告：来源性质和确切干预

[Causal analysis improves VLA efficiency，2026-08-02](https://docs.goodfire.com/assets/reports/vla-interface.html)是 Goodfire 域下第一人称机构技术报告。核验页面未见作者署名、正式发表 venue / DOI 或同行评审标记；exact-title 搜索仅找到此页面。现有证据不足以把它当同行评审论文，也不足以排除它另有发表版本。

其确切组件是 **action-expert block k 读取 VLM layer k 的 cross-attention context / shared-prefix interface**。它按接口层和 token span patch；没有定位单 head、GEN MLP 或吸引子。MolmoBot 在同图、同机器人状态换 tomato / plate 指令；64 个匹配 flow-noise draws，L24 patch 恢复约 79% reach gap，L21–26 联合 patch 恢复完整 gap。object token 本身没效应，周围指令位置有；representation difference 继续随深度增大，却不对应因果效应。报告自述这些因果定位是首个 action chunk 的 open-loop 结果；约 1,000 次闭环 rollout 检验的是重新预训练或 50K-step fine-tune 后的裁剪架构，并非一次 patch 的闭环目标切换。

**对 Cosmos 的用途：** 提醒我们寻找狭窄读取通路，并采用连续动作的有符号方向评分；不是“Cosmos 第 24 层负责目标”的证据。正文部分 π0.5 instruction-bypass 结果主要依赖 paraphrase-style counterfactuals，也不能据此宣称它普遍忽略语言。

## 本地结构核验：干预必须放对位置

依据 `work/network-source/cosmos3/transformer_cosmos3.py` 的 processor / decoder layer / forward，以及 `config.json`：

- UND 是 causal text prefix，`hidden_states[:und_len]`；vision / action 在 GEN，`hidden_states[und_len:]`。旧笔记的“UND 视觉位置”不能直接用于此实现。
- GEN query 读取拼接的 UND 与 GEN K/V。可区分 UND text → GEN action 直接路径，和文本 → GEN vision → GEN action 中介路径。位置实验则重点测当前画面 GEN vision → GEN action。
- 32 个 query heads、8 个 KV heads、每头 128 维。改一个 KV head 会同时影响 4 个 query heads，不能据此称找到一个 head。单头实验应改 `to_add_out` **之前**某一个 query head 的 `z_h`，其他 head 的 `z` 保持 recipient 自然值。
- GEN 层计算为 `r = x + attention_out`，随后 `x_next = r + mlp_moe_gen(norm(r))`。patch `mlp_moe_gen` 输出或 `down_proj` 输入可保留 residual；替换整个 decoder layer 输出则不能分清 attention、MLP 与残差。
- 已有 `probe_cosmos_language_route.py` 记录 / 重算 attention 并验证与 native GEN attention 误差小于 3%；其 `all_gen` 模式同时改变 action 和 vision queries。新的直接 action 路径实验必须只替换 action query indices。

上述为文件阅读所得。实际运行时仍应验证加载 checkpoint 的 runtime config、token indices、位置编码和 native reconstruction；本地源码不自动代表远端正在加载的版本。

## 三个具体干预

### 干预 1：区域 × head 的单步自然脉冲

**问题：** 0→6 cm 的连续位置追踪与 6→15 cm 的目标切换，是否经过同一局部视觉读取头？

固定 recipient 的完整第一 query 输入。供体 / recipient 使用同一指令、初始 action / future-video noise、sigma、domain、schedule，仅奶盒位置不同；主比较 0↔6 与 6↔15。优先在已有位置交换指向的 L12 / L30 附近筛选，不能因为夹爪语言实验的 L13–18 有效便认定位置实验同样有效。

对选中层、一个 query head、16 个 action queries，保持 recipient Q 和所有非选中 token K/V，用供体自然 K/V 重算选中 ROI 的贡献；只把该 head 的 pre-output-projection `z_h` 换入。ROI 应包含奶盒新 / 旧位置；另分 third-object ROI 和相同面积背景 ROI。先做完整 vision span 找到有影响的头，再细分区域，避免初始 ROI 定义错误导致假阴性。K-only、V-only、KV 三种替换区分局部寻址效应与携带内容效应；K 替换会改变整个 softmax 分母，因此不能把它解释成只改了一个物体的注意力预算。

**预测：** 若同一头主要读坐标，0→6 自然供体 patch 应产生沿位移方向的连续变化，6→15 同头 patch 可能大幅改变目标分支；若两个对比的关键头 / ROI 不同，则支持位置更新和目标选择依赖不同通路。对整层有效但任何单头弱的情况，再测少数头组合，零单头效应不证明没有冗余通路。

**对照：** self-patch、反向 patch、相同面积背景、非目标物体、同头等范数随机方向；每个 seed 供体 / recipient 共用随机数。self-patch 与 native 输出误差需远小于待解释效应。

**评分和局限：** 固定第一 query，首先报告真实最终 velocity / 完整 scheduler 后的动作块，按供体–recipient XYZ 动作差方向做有符号投影，并报告原始量；0→6 与 6→15 gap 太小时不报告恢复百分比。动作单位不是米；实际末端坐标需要模拟器。头 patch 证明局部状态因果影响此背景下的输出，尚不证明它编码纯物体身份。ROI 依赖 VAE 的空间 / 时间感受野，需使用真实 packing metadata；不能把 latent heatmap直接解释成物体部位。

### 干预 2：attention–MLP 的 2×2 因果拆解

**问题：** 若自然局部脉冲的供体方向后来变弱，是否由特定晚层 MLP 的响应造成？

在同一个 recipient 输入、同一个 sigma 下，令自然 attention 输出为 A0，按干预 1 重算的局部输出为 A1，层输入 residual 为同一个 x。定义 `r0=x+A0`、`r1=x+A1`，以及实际 MLP `m0=f(norm(r0))`、`m1=f(norm(r1))`。让四组层输出分别为：

| 组别 | 层输出 | 含义 |
| --- | --- | --- |
| C00 | r0 + m0 | 完整自然基线 |
| C10 | r1 + m0 | attention 脉冲进入，MLP 保持自然基线输出 |
| C01 | r0 + m1 | 只加入 MLP 对脉冲的响应 |
| C11 | r1 + m1 | 正常运行脉冲后的 attention 和 MLP |

四组都继续跑真实剩余层，记录最终实际动作输出；不要把中间层直接套输出头的 readout lens 当真实预测。先在 late GEN MLP 扫描，再把满足条件的局部响应细化到 `down_proj` 输入中的自然神经元系数。候选神经元可由系数差和投影大小排序，因果结论仍需有限数目的实际替换验证。

**预测 B 的验收：** C10 的最终动作有供体方向效应，C11 明显回到 recipient；C01 单独沿 recipient 方向推动输出；该模式在反向自然脉冲和独立噪声下重现。仅晚层 full-state patch 有效、但保持 MLP 原输出不能救回，不满足 MLP 覆写。

**对照：** 同 prompt self-patch；同层 random neuron set、同数量及等输出范数；前层 MLP 与相邻 attention 输出；0→6 与 6→15 两种自然差值。gain 先限于自然插值 / 自然供体幅度；极端 gain 只能作为分布外探针。

**局限：** 四组是受控的局部 counterfactual，仍可能与下游激活不兼容。C11−C10 是在特定背景下加入 MLP 响应的总下游效应，不能假定全网络线性。0 / 6 / 15 cm 只改变奶盒位置，没有独立操纵奶盒身份；因此先说“目标 / 空间相关信号被覆写”，不能说定位了 milk 身份神经元。完整运动计划和物体身份仍可能混在同一系数里。

### 干预 3：撤去脉冲后的延续与回拉，并排除求解器记忆

**问题：** 一次局部脉冲影响后续去噪，是目标分支延续、后续通路每次补回，还是 scheduler 对数值扰动的继承？

将时间明确记为 `(物理 query q, 去噪 step t, layer l)`。先固定 q=0 与原 recipient 图像，选定一个 `(t,l)` 只 patch 一次；在当次余下层、下一未受干预 step 和最终动作中追踪有符号供体方向效应。与同层每一步 patch 的正对照比较。不同 t 的实验必须各自和**同 sigma**的自然基线配对；30 步结果不可直接按 step index 当同一个动力系统比较。

选中 step 的输入要完整保存：action sample、所有 future-video latents、time / position / domain 输入，以及 UniPC multistep scheduler 的 history。第一步之后的两个 run 即使 sigma 一致，也可能具有不同 action / video sample；这应视为脉冲介导的状态变化，不能假称输入仍完全相同。

至少增加两个解释控制：

1. **solver-only 配对：** 对自然 run 的 scheduler 输入直接施加同样的最终 action 和 video velocity 差值，不改内部网络；之后用同一原图、同 schedule 继续。若它复现全部后效应，则持续性可由外部 sample / solver history 的继承解释，尚未识别特殊内部自维持回路。不能只匹配 action velocity 而忽略未来视觉 latent 的更新。
2. **同 sigma 输入重放：** 在同一个未干预 `(t+1)`，分别重放自然 sample / 完整 solver state 与脉冲后的 sample / solver state；一次只交换 action、future-video 或 history，确认后效应由哪种状态携带。重置全部可见状态后若仍不同，先查未重置 cache / RNG / hook 和实现错误。Transformer 每次 forward 的中间 hidden 不自动跨步存活。

**预测 A 的加强证据：** 在完全固定原画面、正常 sigma schedule 下，有限自然脉冲形成持续的目标方向 / 目标身份切换；幅度扫出现转折，并在反向脉冲和多个匹配种子中重现。若位置 0 / 6 / 15 的动作 endpoint都连续变化，没有离散目标选择切换，就不支持目标分支阈值说法。先获得这种有限时域证据，再讨论局部 basin / 通路竞争。

**回拉的通路判别：** 脉冲后只在一个后续候选 MLP 把输出保持自然值，或只把 candidate visual→action head 的输出保持自然值；哪一个撤去后“回拉”消失，就定位到哪个恢复通路。不能把向基线方向恢复自动归给 MLP。

**严格边界：** 36 层是有不同参数的 feedforward 复合，30 个去噪步的 sigma / sample / solver history 不同；扰动范数收缩、残差相似或最后回到原目标都不是 stationary attractor 的充分条件。固定 sigma 重复人工 update map 可研究局部稳定性，但那是新定义的动力系统，不能直接称已证明原 Cosmos policy 的吸引子。实际闭环第一块执行后，新视觉与机械臂位置已改变；持续行为可能来自环境反馈，必须与上述固定画面的去噪实验分开报告。

## 本轮最小可解释结论

最有价值的交付是一个具体 `(t,l,head / MLP,query span,visual ROI)`，说明“自然位置供体脉冲如何改变真实动作、何处首次被保留或抹去、哪个局部干预能阻断这个过程”，并给出 self-patch / 反向 / 背景 / seed 对照。只测到差异增大，不支持机制定位；只测到输出失败，不支持身份覆写；只测到后效应，不支持吸引子。
