# Cosmos3 奶盒位置阈值：局部机制假设、文献核验和干预

核验日期：2026-10-04。范围收敛到同一场景奶盒沿 X 位移 0 / 6 / 15 cm 引起的目标切换。本文开头保留当时的只读研究笔记和实验设计；当时没有运行 Cosmos3、GPU 或 SSH。后续封存实验的状态追加在末尾，不能把原设计当成实验结论。已读同目录 `latent-causal-literature.md`，这里补充 head / MLP / 短脉冲层面的可区分预测。

## 要先回答的两个问题

**预测 A：存在由局部视觉通路触发、在去噪过程中延续的目标分支。** 在固定 recipient 画面、同一 sigma / 初始噪声的一次去噪 forward 中，只给一个中层 GEN attention head 的 action queries 一次自然供体脉冲。脉冲超过某个有限幅度后，撤掉 hook，后续层和后续去噪步骤仍产生供体方向的动作；效应在 6→15 cm 的行为转折附近比 0→6 cm 明显。如果每次都要重新 patch 才保留供体效应，或者只有动作向量范数增大而目标方向不变，该强预测不成立。这里首先称“分支延续”，不称已发现吸引子。

**预测 B：某个晚层 GEN MLP 对目标相关信号进行局部覆写。** 同样的 attention 脉冲在进入该 MLP 之前有供体方向的因果效应，经过该 MLP 后效应明显减弱或反向；仅把这一个 MLP 的输出保持为未受脉冲时的自然输出，便能恢复最终动作效应。反向自然输出替换应把 0 / 6 cm recipient 拉向 15 cm 供体的动作方向。若效应主要损失在后续 attention、最终 norm / 输出头、scheduler，或所有 MLP 都不满足“冻结后救回”，则该局部覆写预测不成立。负的 attention–MLP 差向量 cosine 只能作为候选排序，不能证明覆写。

两个预测可以局部共存，因此先查明**首次稳定阻断脉冲的具体组件**：是一个 MLP、后续视觉 attention，还是跨去噪步的状态更新。需要用局部干预量化它，不能用首末层差异百分比替代。

## 两篇机制论文，以及脑科学类比的边界

### 1. VLA FFN 神经元干预：Häon 等，CoRL 2025

[Mechanistic Interpretability for Steering Vision-Language-Action Models，原文 §4 / §6](https://arxiv.org/html/2509.00328v1)；[作者代码](https://github.com/Physical-AI-Safety-Institute/mechanistic-steering-vlas)。[arXiv 记录](https://arxiv.org/abs/2509.00328)标明 CoRL 2025。

作者将 FFN 看作输入相关系数乘固定 value vectors 的和；按词表投影选语义簇，覆写部分中间神经元系数。OpenVLA 的 hook 放在 `down_proj`，实质改变其输入系数；π0-FAST 用改写的 FFN 实现同一操作。LIBERO 有实际模拟器执行。UR5 的 π0-FAST 先经过机器人数据的 LoRA 适配；运输高度实验分别执行各干预版本，速度实验则只执行未改模型的动作，在相同观测上比较各版本预测的位移，不能把后者叫独立真实闭环速度实验。随机神经元和改提示是对照。论文也明确语义聚类可能混合“slow / careful / stuck”，含义会随模型、任务和时间漂移。

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

## 2026-10-04 实际 Cosmos3 结构：哪些分工由架构保证

本节只读核验实际源码、checkpoint 配置和已保存的真实输入；没有运行模型或新增干预。它回答“有没有管语言、管运动的区域”中的架构问题，不以活动热图命名功能区。

冻结的本地源码是 [transformer_cosmos3.py](/home/cenxi/Documents/Codex/2026-09-28/https-cvlab-kaist-github-io-geometric/work/network-source/cosmos3/transformer_cosmos3.py:50)，与远端实际 `/home/current/work/cosmos3/diffusers/src/diffusers/models/transformers/transformer_cosmos3.py` 字节一致，SHA256 为 `ac951f6b4966a485bf2c3420c754e9f8366bbceda3471a8a5de75606767f2b22`。实际 pipeline 为 `/home/current/work/cosmos3/diffusers/src/diffusers/pipelines/cosmos/pipeline_cosmos3_omni.py`，SHA256 为 `ec2f051ec5703926c82145cca24bfcb93222019234fade9aafa905f7ae589eca`。以下行号以这两份源码为准。

实际 checkpoint 配置 `/home/current/work/cosmos3/checkpoints/Cosmos3-Nano-Policy-LIBEROall-5k/transformer/config.json` 的 SHA256 是 `eeb0aa0c7e905aa941163d38a9a7af6ed65232f29af01a093633a40ed86f170c`；第 12–16、23–26 行确认 head_dim=128、SiLU、hidden_size=4096、intermediate_size=12288、32 个 query heads、8 个 KV heads、36 层。第 4 行 action_dim=64，第 24 行 num_embodiment_domains=32。本地 `work/network-source/cosmos3/config.json` 与此实际 checkpoint 配置的文件 SHA 不同，不能把本地配置文件默认当作运行配置。

| 结构与实际调用 | 可核验源码位置 | 通俗含义与边界 |
| --- | --- | --- |
| UND 和 GEN 各自独立的 Q/K/V、attention 输出投影和归一化参数 | transformer 第 257–288 行构造两套；第 68–79、119–122 行 forward 真正分别调用 | 文字处理流与生成流有设计好的参数分工；不是仅在一个共享模块中把输入换个标签。 |
| UND 和 GEN 各自独立的 MLP、输入 Norm、post-attention Norm | 第 335–353 行各自实例化；第 361–371 行分别调用，各自加 attention 和 MLP 残差 | 每层同时包含两条处理流。两者 MLP 都是 dense gated MLP；变量名 `mlp_moe_gen` 不代表这里存在稀疏 router 或按物体身份分工的专家。 |
| UND 只因果读取 UND；GEN 读取 UND 与全部 GEN 的 K/V | 第 93–117 行实际两次 dispatch | 语言信息可以逐层进入生成流；本实现的 GEN 不能反向进入 UND。GEN 中当前视觉、未来视觉和动作可互相读取。 |
| 视觉与动作使用各自输入投影，随后一起进入同一个 GEN trunk | 第 743–756 行视觉输入；第 773–794 行动作输入；第 807–818 行按 `und_len` 分流并运行 stack；第 656–664 行逐层执行 | 视觉与动作共享 GEN 的 attention/MLP 参数，没有另外一套独立的“动作 36 层”。不同模态依靠 token 位置、输入投影、动作模态 embedding 和 timestep 等区分。当前与未来视觉也没有各自独立的 GEN trunk。 |
| 视觉与动作在末端用不同输出投影 | 第 472–484 行构造；第 829–855 行实际按对应 token 索引读出 | 动作专用输出接口存在，但它读取前面共享计算的结果，不能把这个薄输出头当作完成全部运动规划的独立皮层。 |
| 构造了 `lm_head`，当前 policy forward 不使用它 | 第 471 行构造；第 857–860 行仅返回 vision/sound/action；本 pipeline 没有 `lm_head` 调用 | 不能因为参数里有语言输出头，就称本轮是在先生成一段文字计划再转换成动作。 |

MLP 的实际计算在 [第 179–194 行](/home/cenxi/Documents/Codex/2026-09-28/https-cvlab-kaist-github-io-geometric/work/network-source/cosmos3/transformer_cosmos3.py:179)：`down_proj(SiLU(gate_proj(x)) * up_proj(x))`，4096 维输入经真实 12288 个门控中间单元后回到 4096 维。32 个 query heads 各 128 维指的是 attention 输出投影之前的真实 head 轴；不能把任意 4096 维残差向量每 128 维切一块，称为一个运动或语言 head。

### 本次 policy 的真实调用链

实际 pipeline 第 1735–1741 行将当前去噪的 vision/action samples 转成模型输入 dtype；第 1756–1782 行把同一次输入的文字、视觉、动作和 embodiment domain 一起交给 transformer。模型预测经第 1783 行的 condition/padding mask 后，在第 1838–1843 行转成 FP32 velocity。第 1855–1857 行更新视觉 sample，第 1873–1882 行用动作 scheduler 更新动作 sample；第 1901–1905 行最终取出动作。动作 head 的单次返回是去噪 velocity，不等于已经执行的末端位移；之后还有求解器、动作反归一化及模拟器控制过程。

已保存的 `target-reader-inputs/seed195/dispatch-t29.pt` 实际 kwargs 中，UND 是 121 个 text-side tokens，包含指令包装、任务元数据和特殊 token，不是 121 个普通词；真实 10-token 指令在全局索引 45–54。GEN 是 266 个 tokens：50 个当前条件视觉、200 个未来视觉、16 个动作。声音相关输入全为 `None`；当前 UND 中没有图像 token。实际 action domain id 是 **5**，选择 LIBERO 机器人域的输入/输出权重，不能解释成牛奶或奶酪类别；`DomainAwareLinear` 第 197–225 行的定义明确是每个 embodiment domain 一组权重。

因此可以说：**有文字处理流、共享视觉动作生成流和动作专用读出这些架构分工。** 是否在它们内部学出了对某类运动、词义或物体选择更特异的局部通路，仍要看自然输入对照和真实因果干预；层或通道更亮只说明这一输入下数值幅度或净更新量更大，不足以命名“运动区”“语言区”或“海马体”。

## 2026-10-04 功能分工与“皮层区域”：三项原始研究

本节回应网络是否也有语言区、运动区。复核原始研究和作者代码，未新增模型或机器人运行。需要分清：**设计好的参数分支、训练后出现的功能倾向、具有空间邻近关系的功能地图**。三者不等价。

### 语言 head 的训练后分工：Kumar 等，Nature Communications 2024

[Shared functional specialization in transformer-based language models and the human brain](https://doi.org/10.1038/s41467-024-49173-5)；[开放原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC11217339/)；[作者 PDF](https://hassonlab.princeton.edu/sites/g/files/toruqf3591/files/documents/Kumar_NatCommun_2024.pdf)。核验位置：Results 的 “Interpreting transformations via headwise analysis”、Fig. 4–5，以及 Methods 的 “Encoding model estimation and evaluation” 和 “Summarizing headwise transformation weights”。

训练后的 BERT 部分 attention head 对不同语言依赖呈现不同偏好；研究还用这些上下文变换预测听故事时的人类皮层 fMRI。它没有预先指定某 head 专管某种语法，但结构和训练制度会影响分工。关键方法是时间连续分段的三折编码回归验证：所谓 headwise 分数在测试时只保留某 head 对应的**编码回归系数**，不是删除 BERT 中其他 head。打乱回归系数也是统计对照。

**支持：** 普通 Transformer 可学出局部功能倾向，并与脑响应形成预测关联。**不支持：** head 等于脑区、该 head 对理解具有已证必要性，或语言与运动的双重因果分离。研究没有机器人动作实验。

### 可干预的运动控制通路：Häon 等，CoRL 2025

[正式会议原文与 PDF](https://proceedings.mlr.press/v305/haon25a.html)；[开放全文 §3–4、§6](https://arxiv.org/html/2509.00328v1)；[作者代码](https://github.com/Physical-AI-Safety-Institute/mechanistic-steering-vlas)。

这项研究提供比活动相关更强的证据：保持模型权重不变，覆写 FFN 的选中中间单元，可改变 OpenVLA 模拟执行的运动，以及适配后 π0-FAST 的实际 UR5 运输高度。真实速度测试的执行范围已在上文澄清。词表投影用于筛选，干预用于检查输出作用；投影关联本身不是单元语义证明。其动作相关权重遍布各层，论文未找到一条“前面纯语义、后面纯运动”的硬分界。

**支持：** 少数内部单元可以成为可控的行为接口。**不支持：** 所选单位仅编码速度或某个物体、存在一块纯运动皮层，或相同单元编号和功能可直接迁移到 Cosmos3。

### 真正的空间功能地图：Margalit 等，Neuron 2024（TDANN）

[A unifying framework for functional organization in early and higher ventral visual cortex](https://doi.org/10.1016/j.neuron.2024.04.018)；[开放作者稿](https://pmc.ncbi.nlm.nih.gov/articles/PMC11257790/)；[官方代码](https://github.com/neuroailab/TDANN)。核验位置：Fig. 1–3、Fig. 5，Methods 的 “Loss functions” 与训练六步骤，Discussion。

作者先给各层单元二维模拟皮层位置，并加入“近邻响应更相似”的空间损失；位置还经过保持粗略视网膜映射的预优化，再冻结位置、重新随机初始化网络权重训练。自监督自然图像训练后，早层出现方向偏好图，较高层出现面孔、身体等类别选择性簇；同样布局下仅任务训练的模型没有相同聚集。类别偏好和聚集在这些约束下形成，并非事先把某一片贴上“面孔区”标签。

**支持：** 加入空间组织约束后，人工网络能学出类似视觉皮层的功能地图。**不支持：** 无此约束的普通 Transformer 或 Cosmos3 天然存在二维皮层地图；也不证明语言、运动或海马体功能。论文只基准比较 V1/VTC，部分类别簇的布局仍与真实皮层不同。

### 对当前 Cosmos 实验的判读

当前可核事实是 UND 与 GEN 的设计分工，以及在共享 GEN 内切断特定读取连接会改变部分场景的真实行为。后者定位的是**在固定输入、噪声和数值背景下有作用的通路**。第 1–9 层的切边效果不能单独命名成“牛奶区”或通用运动区；4096 个残差坐标也没有 TDANN 的皮层邻近含义。

若要进一步声称语言与运动存在功能分离，需要独立操纵词义/目标和运动需求，检验同一局部干预是否选择性损伤其中一个、保留另一个，并在未用于挑选通路的布局上确认。活动幅度、可读出信息、局部干预和真实任务行为应分别报告，不能互相代替。

### 补充：直接借用脑科学功能定位，再消融验证（NAACL 2025）

[The LLM Language Network: A Neuroscientific Approach for Identifying Causally Task-Relevant Units](https://aclanthology.org/2025.naacl-long.544/)；[修订全文](https://arxiv.org/html/2411.02280v2)；[作者代码](https://github.com/BKHMSI/llm-localization)。本轮只读核验 §3、§6–8、Limitations，以及 `localize.py`、`generate_lesion.py` 和 `models/modeling_llama.py`。

作者比较240条正常句子与240条非词串的内部响应，用正 Welch t 值定位语言选择性单元，再在独立语言基准上消融。18个模型参与定位，语言任务消融评估覆盖10个模型；消融全网最具选择性的0.125%–1%单元，比等数量的随机消融造成更大的语言表现下降。随机对照重复三个种子。代码将语言刺激截到最多12个token，并取绝对激活比较；“12个词”不应直接称为所有模型都天然token等长。

这里的单元是每个 Transformer block 输出 hidden 向量的坐标，**不是 FFN 中间神经元**。公开 LLaMA 实现用各层掩码将选中输出坐标在所有token上置零，干预包括完整残差状态。算术、代码是活动偏好的对照；MATH/ToMi 中消融的是另行定位的 MD/ToM 单元。本文未证明消融语言单元后数学能力完整保留，也没有机器人运动实验，不能据此宣称语言／运动双重因果分离。

**可借鉴：** 用独立刺激找候选，再检验局部干预的实际功能，并保留同数量随机对照。**不能照搬：** 将任意 hidden 坐标直接当成生物神经元，或把已有活动差异命名成纯语言区。

### 本轮机制分解的预注册边界

已接受的切边操作同时去掉 future value contribution，并重新归一化其余 keys 的读取权重。新实验在同一次实际 Q/K 下区分这两项：官方 all-allowed 输出 Z0、保留全部 keys 但 future V 置零的 Zv、官方切边输出 Zk，以及 pre-W 加法反事实 Z0+(Zk−Zv)。最后一项不是合法 softmax：它保留原 future contribution，同时增加其余 contribution，幅度变化仍是替代解释。差和用 FP32 运算、仅最后转一次 BF16；三个官方端点不由离线重建替代。

大胆主预测是：在 X12、固定 A195、V195/V198 两份源噪声下，仅 value-zero 也能使两例都严格只抓牛奶。其他分解臂全部执行并报告，不能用它们替换被否定的主预测。若 value-zero 和加法臂都救回，就不能排他归因；若两者都失败而切边仍成功，则要检验联合或交互作用。该实验尚不能把 future 向量命名为“熟悉抓法记忆”，也没有建立物体身份专属单元。

## 2026-10-04 语言目标 × 冻结九层切边：尚未运行的预注册假设

**本节是拟议方法和独立质疑，不是新实验结果，也不是文献已经证明 Cosmos 机制。** 固定 X12 初态、图像、A195 和 V195/V198；继续使用去噪步0..14、零起算层0..8的 future→current 切边。所有 masked 臂仍采用贯穿 q0 的 union66 all-allowed 数值背景。主比较只把指令中的 `milk` 与 `cheese` 作等长、单 token 替换：必须由实际 tokenizer 检查目标位置及其他 token 完全一致，不能默默补 token。Canonical `cream cheese` 的长度不同，只作辅助措辞测试，不能混入主词义比较；它还改变文本长度、索引和 attention 计算形状。

### 为什么要比较两次干预增量

真实结构中 UND 只作因果自注意力，GEN 读取 UND 与全部 GEN；当前视觉和动作共享 GEN 参数。切 future→current 时文本读取入口仍开，并同时重分配其余 keys 的权重。因此“换词产生内部差异”和“切边把动作推向同一对象”都不足以命名语言区或识别区。可核源为 `work/network-source/cosmos3/transformer_cosmos3.py:68` 的实际 QKV 与 dispatch、`:105` 的 UND+GEN 拼接、`:355` 的 attention/MLP 残差计算。

一级主位置是 **t0 的动作 boundary36**：最终 norm 之前、完整16×4096 action hidden 的实际数组，按相同 token/坐标顺序展平，用 Float64 作统计；不是 XYZ 命令、厘米或抓取概率。记同一 V 下的 all-allowed 数组为 A_m、A_c，cut 数组为 K_m、K_c：

- δ_m = K_m − A_m；δ_c = K_c − A_c。
- B = (δ_m + δ_c)/2 是两种指令共有的改变量。
- I = δ_m − δ_c 是“切边 × 目标指令”的交互量。
- D = A_m − A_c 是数值背景内的原始指令对比；D_cut = K_m − K_c，恒有 I = D_cut − D。

共同加上同一偏移时 I=0，而原指令差 D 可以仍然存在；若 D_cut 被压小或反向，I 就揭示切边改变了指令到动作的对比。固定两因子、配对初始样本和 sigma 后，这个差分中的差分量化的是**该边集在该数值背景内是否受这两条指令调制**。它不把“内部可区分指令”直接等同于模型正确理解对象身份。仅 t0 保证初始采样状态相同；后续去噪及闭环输入会动态分化，晚期差异不能全归为直接语言读取。

### 冻结的主检验与数值解释门

主假设 H_common_shift 是这两条指令下的**近似共同 hidden 偏移**。分别对 V195、V198 要求：

- cos(δ_m, δ_c) ≥ 0.95。
- r = ||I|| / (||δ_m|| + ||δ_c||) ≤ 0.10。

两 V 都通过才能支持这个受限的主预测；不得以其他层、时刻、canonical 措辞或物理最好的一例替代。零范数使 cosine 或 r 无定义，应保留无效标记，不能填成“1”或“0”以得到支持。

另设 E = max(||N_m−A_m||, ||N_c−A_c||)，其中 N 是相同目标指令下的 native t0 action boundary36。只有 **||D|| > 10E** 且 D 明显非零，才解释语言方向投影，例如 g = ⟨D_cut,D⟩ / ||D||²。须同时保存 D、E 原范数及投影外的残差，避免近零分母制造大百分数。门未过时该机制解释不可判，不能把它当作“语言无作用”。g≈1 表示该原始对比的平行分量保留，g≈0 或负值表示这一分量缩小或反向；即使 g≈1，正交分量仍可显著变化。

### 独立质疑：这两个阈值并不独立

在理想实数、两个非零向量下，r≤0.10 已经推出 cos≥0.98：

`r² = 1 − 2ab(1+cos)/(a+b)² ≥ (1−cos)/2`，其中 a=||δ_m||、b=||δ_c||。

它也推出较大/较小的增量范数比不超过 1.1/0.9≈1.222。故 cos≥0.95 是较松的冗余报告门，不是第二份独立证据；等范数且 cos=0.95 时 r≈0.158，仍会失败。**保留已指定阈值，不据此自行改主假设。** 0.95、0.10 和10倍数值差异是本轮事先选定的操作标准，不是论文给出的自然界线、显著性检验或总体置信度。原生/all-allowed 的差异也是确定性的计算背景差，不能冒充独立随机测量噪声。两 V 是两次受控案例，不支持总体频率估计。

完整16×4096范数可能由少数大幅度 token/通道主导，满足门也不排除较小但行为关键的语言交互。建议保留真实数组和逐 token 原范数供复核，作为描述性诊断，不另设获胜主位置。统计应记录运行库版本；不能通过换归约实现、改阈值或把 FP32 数值误差当语义证据来改变结论。

### 行为与脑区类比的边界

即使两条指令切边后都抓奶，也只能否定“都恢复了相应语言目标”；它可以来自共同接近偏移，也可以来自动作对比压缩、局部默认目标或闭环阈值。共同 hidden 偏移本身不推出固定世界坐标：对象位置未独立变化。它同样不推出语义无用：D 可以保留，语言还可以影响幅度较小的关键坐标或后续推理。已有 `trajectory.eef_xyz` 的 step0→16 位移可沿初始 milk−cheese 世界坐标方向投影，用米描述接近偏移；最终仍需逐对象验收接触、抬升与搬运，横推不算抓起。

方法依据是 [AlKhamissi 等，NAACL2025](https://aclanthology.org/2025.naacl-long.544/) 的独立刺激定位与独立任务消融、随机单元对照（原文 §§3、6–7），以及 [Häon 等，CoRL2025](https://proceedings.mlr.press/v305/haon25a.html) 的局部 FFN 干预、对立运动概念和行为验收（[作者全文 §§4、6](https://arxiv.org/html/2509.00328v1#S4)）。前者支持用对照和实际干预检查选择性，后者支持用真实运动检验 steering；二者没有证明我们的切边、差分中的差分阈值或 Cosmos 身份通路。Sham 只控制数值背景，不能替代随机结构对照。仅两条目标指令的交互也不构成语言／运动双重分离，更不能把九层边集命名为一块生物皮层。

## 2026-10-04 旧四头实验的实际输出链补查：探索性，只读CPU

本节是已有冻结 `future-head-groups` 的真实数组补查，**不是新语言×切边实验的结果**。Torch2.14.0+cu130 CPU、无模型加载或新forward。读取每个V下的all_allowed、head_G6、full_hardmask各自states.pt和boundaries-t00.pt；全部先Float64再相减。原生JSON/数组没有改动，正式新实验的主位置、阈值和条件没有因这份探索性统计改变。

实际结构：`_run_decoder_stack`返回`self.norm_moe_gen(gen_seq)`，随后`DomainAwareLinear`以domain5的权重把4096维readout映射成64维velocity。模型保留64维输出，LIBERO实际动作只用前10维；XYZ是前三维。已有record.readout来自action_proj_out的真实pre-hook，record.action_velocity来自其真实post-hook；L36来自最后decoder层输出，位于最终归一化之前。

相对差异定义为 `RMS(arm−same_V_all_allowed)/RMS(same_V_all_allowed)`，各位置有各自的分母，不是厘米、抓取概率或可相加的功能占比：

| 起点 | 改动 | t0 L36动作hidden | t0真实readout | t0真实XYZ velocity |
|---|---|---:|---:|---:|
|195|第1层24–27号头|1.9202%|1.4951%|0.6537%|
|195|前9层全部头切边|6.4131%|4.1707%|4.3871%|
|198|第1层24–27号头|1.6547%|1.4077%|0.6260%|
|198|前9层全部头切边|5.2210%|3.5557%|3.1838%|

195的all_allowed分母RMS分别为0.9305340768、0.1879823573、1.1058695312；198为0.9276149682、0.1876185396、1.0993695012。195/head_G6的实际差异RMS为0.0178681347、0.0028105525、0.0072289952；195/full_hardmask为0.0596765064、0.0078402196、0.0485156936。不能只凭原始RMS在归一化后降低就说信息被抹掉，因为整份向量的尺度也改变。

有限结论：四头干预的差异仍经过归一化进入实际输出，但其首步XYZ相对效应比L36小；全切边没有同样程度的相对缩小。这提示检查**改变的方向与实际动作输出投影的耦合**，而不是继续只报hidden范数。首步差异不能解释整条轨迹：195/full_hardmask最终q0 XYZ距离0.2669724791，198/full_hardmask为0.0131341398；二者不能称为同一单步传播链里的信息保留率。此处未建立输出头导致目标错误的因果证据，也未修复15cm失败。新实验将保存一次实际归一化和输出头的只读参数，同时测同一词替换经各输出位置的变化；这些只作为描述性机制检验，不替代冻结主预测。


## 18条件语言×九层切边：封存结果（2026-10-04）

H_common_shift 已按原定义执行，不再是未运行的计划。18条实际MuJoCo执行、540首轮forward、3780后续forward、540动作边界和2430干预site的CPU审计均完成。

- milk↔cheese真实packed输入同长、目标48单token改变。九层切边后两V、两指令都抓牛奶；此前milk下救回不能等于恢复语言目标选择。
- canonical cream cheese多一个token，两V、三臂都抓奶酪盒。措辞、语义和长度／packing混杂，不据此声称固定偏向牛奶或完全不读语言。
- 主位置q0t0actionL36：D/E为2.01055／2.43895，均未过预登记>10有效门，判定**insufficient**。cosδ=.89456／.83645、r=.24075／.29749只是描述，不能绕过门改称拒绝或支持。
- 实际domain5输出头首10维实数行空间中，语言D平方范数比例5.428%／9.010%；低维投影不是语义保留率，实数重算也不等价BF16内核。不写“95%语言丢失”。

[实际封存统计](future-language-selectivity-analysis/analysis.json)、[真实数组](future-language-selectivity-analysis/arrays.npz)、[18条实际结果](future-language-selectivity-execution/comparisons.json)。现有局部替换文献支持先验证自然供体与数值背景，不支持从本批热图给head或MLP命名脑区。下一轮先做合法等长名称对照，匹配后才用局部UND K/V自然双向替换检验直接文字→动作入口；此后续尚未执行。15厘米错误根因仍未定位。
