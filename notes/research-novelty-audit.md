# Cosmos3 Policy：研究价值与机制证据审计

核验日期：2026-10-03。范围为公开原论文、作者项目页/代码、直接 Cosmos3 讨论和本仓库现有结果。本轮没有运行模型、仿真或训练，没有对外发帖。判断针对 `fwd4xl/cosmos3-nano-policy-liberoall-5k` 与当前自定义 Diffusers runner；不推广为全部 Cosmos 模型的结论。

## 判断

当前结果值得继续，但“VLA 忽略语言”“跨场景状态注入带来供体位置上的运动”已有直接先例。现在最有价值的成果是：保留了可核验的错目标行为，并排除了语言完全未进入动作计算的简单解释。现有数据还没有证明训练轨迹记忆、纯位置编码、某层语言断路、输出头损坏，或未来视频分支导致了错误。

下一步应把问题收紧为：**在联合视频—动作的 flow policy 中，目标身份相关的信息通过哪些读取路径影响真实目标选择；这些路径何时足以改变行为，何时被布局相关动作压过？** 这是一条可证伪的研究方向，不是本轮已经得到的机制结论。

## 直接 Cosmos3 资料与工程边界

- 社区 checkpoint 的模型卡声明：四套 LIBERO 混合微调、JSON prompt、10D rot6d 动作、16 步块，iter 5250；LIBERO-10 的 200 episodes 成功率为 95.5%。它没有报告同场景换目标的语言忠实性，也没有列出完整训练 episode 清单。因此高默认任务成功率与当前反事实错目标可以同时出现。上述数值是发布者报告，未在本轮复现。[模型卡](https://huggingface.co/fwd4xl/cosmos3-nano-policy-liberoall-5k)
- 官方仓库 [Discussion #369](https://github.com/NVIDIA/cosmos/discussions/369) 是我们自己的早期夹爪报告；核验时为 unanswered、0 comments。它不是另一团队的独立复现，也不是 NVIDIA 已承认的根因。
- [Cosmos Framework Issue #50](https://github.com/NVIDIA/cosmos-framework/issues/50) 是直接 Cosmos3 LIBERO 复现记录，报告匹配 97.4%，并明确指出图像旋转、夹爪符号/区间和 normalization 可显著影响执行。它解释为什么必须先核对推理契约，但没有诊断语言忽略。
- 官方报告的 policy 模式联合去噪视频与动作；AR/UND 与 DM/GEN 有独立参数，在共享 attention 中相遇，GEN 能读取 AR 和 GEN 上下文。LIBERO-10 默认评测成功率不等于目标反事实忠实性。[Cosmos 3 v2，§2.2–2.3、§6.2.5](https://arxiv.org/html/2606.02800v2)
- 本仓库 [runtime](../provenance/runtime.md) 尚未完成官方 server 与自定义 runner 的完整逐数值交叉验证。先在同一权重、输入、噪声、schedule、normalizer、chunk 设置下核对首个动作块；若差异来自 runner，先处理工程问题，再评价模型机制。不能用现有自定义 runner 结果证明官方发布栈具有相同缺陷。

检索没有找到公开的一手 Cosmos3 wrong-target 根因说明；这是本轮检索结果，不等于全球不存在相关实验。没有把基于 Predict2 的 `NVlabs/cosmos-policy` 当作 Cosmos3 的直接证据。

## 最相关的五组研究来源

### 1. LIBERO-CF / Counterfactual Action Guidance：最接近当前外部现象

最新论文为 v2，2026-07-15。它在保留视觉场景的情况下分配可满足的替代目标，分别测目标 grounding 与完整任务成功，以及对训练任务目标的偏置；移除训练任务物体的 CF-Focused 是重要对照。CAG 有冻结 VLA 的空语言分支版本和另外训练 VA 的版本，二者应分开比较。实验支持视觉偏置与方法效用，未定位某个内部语言断路。[论文 §III–VI](https://arxiv.org/html/2602.17659v2)

作者实际发布了条件 JSON、BDDL、初态、`eval/main_cf.py`，以及 `openpi-cf`、`openvla-oft-cf` 的 CAG 推理入口。仓库的 touch rate 不能直接等同本仓库“接触并抬起”的抓取指标；若复用需统一定义。[作者仓库](https://github.com/yuffish/LIBERO-CF)

**新意影响：** 同场景 milk→butter 却抓旧物体，属于已有反事实失败范式。若将内部路径引导作为新方法，至少与 CAG 这种简单输出端对照比较收益、额外推理成本及正常任务保真度。

### 2. Not All Features Are Created Equal / Action Atlas：最接近当前内部状态实验

arXiv 当前仍为 v1，2026-03-19。论文已有 null、same-scene、cross-task、cross-seed 注入及分支比较，报告供体空间轨迹的影响。附录明确承认 full residual hooks 夸大消融效应；改为 MLP hook 后，少量概念特征消融无显著效果。因而“注入后动作改变”不能自动解释为概念被定位或必要性已证实。[论文 §3–4、F.3、G](https://arxiv.org/html/2603.19233v1)

当前作者代码比论文摘要更值得注意：`silent_prompt_steering.py` 已实现 instruction/tail span 编辑、真指令与同长度 filler 的状态差、位置化/平均方向、等范数 random 与 wrong-task 对照，并用真实环境成功检查。`cross_task_injection.py` 测双向注入；`hooks.py` 按每次 forward 重放整块输出，shape 不匹配时跳过。这些代码公开可读，本轮未重跑；不能把当前 GitHub 的新增接口当成 v1 论文已验证的结果。[作者仓库](https://github.com/CWRU-AISM/action-atlas)、[silent steering 实现](https://github.com/CWRU-AISM/action-atlas/blob/main/experiments/silent_prompt_steering.py)、[注入 hook](https://github.com/CWRU-AISM/action-atlas/blob/main/experiments/hooks.py)

**新意影响：** 供体状态追随位置、文本后缀替换、方向引导及随机控制本身已有先例。Cosmos3 的特定 MoT 路径、去噪时间与目标行为之间的受控联系，才可能超出架构复现。

### 3. LangForce：训练数据为何允许语言不起决定作用

最新论文为 v7，2026-05-29；早期题名是 BayesianVLA。其 vision-only pilot 是重新训练语言被屏蔽的策略，并非同一冻结策略临时清空 prompt。LIBERO Goal 的视觉策略与完整策略差异很大，强调同场景多目标监督的重要性；latent action queries 和双分支 LLR 是训练方法，不是对 Cosmos3 冻结模型的因果定位。[论文 §2–3](https://arxiv.org/html/2601.15197v7)

作者仓库发布 `LangForce.py`，含 prior/posterior 的 query 排序、prior condition detach 选项、LLR span 检查及 posterior-only 推理；推荐训练入口通过 starVLA。仅引用抽象“信息坍缩”不能证明当前社区 checkpoint 的实际训练分布满足其假设。[作者仓库](https://github.com/ZGC-EmbodyAI/LangForce)、[实现](https://github.com/ZGC-EmbodyAI/LangForce/blob/main/LangForce.py)

**新意影响：** 视觉可推断任务导致模仿学习不必使用语言，是已有解释。需检查当前权重的数据覆盖，或明确将它保留为假设。

### 4. LIT：最近、且已经覆盖 world-action 架构

论文 v1 发布于 2026-09-11。其两阶段方法先学无图像的空间目标动作 prior，再以 pose supervision 约束唯一视觉接口；覆盖 π0.5、MolmoAct2、FAST-WAM、ImageWAM。MolmoAct2 分析已比较任务不变的视觉扰动和同视觉改目标：baseline 沿原目标，LIT 改向新目标。六项消融包括恢复直接视觉路径、去掉 Stage 1、去掉 pose loss、简单 staged training 和单独 pose supervision。它支持该训练设计，不证明冻结 Cosmos3 中哪个通路失效。[论文 §IV-E](https://arxiv.org/html/2609.12641v1)

作者代码中心链接四个框架 fork，指定分支/commit；FAST-WAM 明确把 video DiT→ActionDiT 的直接 feature injection 改为接口。作者列出 interface 被使用、pose 可重建、参数可训练和保存三项实装检查；ImageWAM 的最新环境核验未重新跑 evaluation，需要 gated autoencoder。[作者代码中心](https://github.com/jianmanlincjx/LIT)、[FAST-WAM fork](https://github.com/jianmanlincjx/fastwam)

**新意影响：** “把语言/视觉通过 latent bridge 接动作”“在 WAM 上压制视觉捷径”也已有方法和结果。研究不能只靠换成 Cosmos3 或添加 pose probe 获得机制新意。

### 5. IGAR：内部干预有效不等于完成正确替代任务

论文 v1 发布于 2026-03-06。ICBench 固定视觉并给矛盾指令，IGAR 调整 sink 相关 attention；正常指令保真是重要对照。项目页展示干预后悬停/空抓，符合其矛盾场景评价，但不足以证明会执行另一条可满足目标。公式的文本预算来自 text sinks，不能泛称把视觉注意力转到语言。本轮核对的论文/项目页未提供可验证的作者实现链接。[论文 §3–5](https://arxiv.org/html/2603.06001v1)、[作者项目页](https://ray-nh.github.io/igar/)

**新意影响：** 若本项目取得“可满足指令之间的正确切换”，将比“矛盾提示让成功率下降”更直接；若干预只让机器人停住，机制主张应降级为行为抑制。

## 另一篇新近邻：CofactVLA

2026-08-05 的 v1 已提出语言屏蔽双分支、flow velocity 正交引导及 K/V covariance intervention。它与任何“放大语言相对空提示差异”的方案重叠。但“空提示分支等于纯视觉偏置”“正交分量等于纯语义”不是无需条件的事实；论文的 latent 分离依赖明确的正交子空间与 eigengap 假设。本轮未在原论文或可确认作者来源中找到发布代码，不应借其强因果措辞替代本项目的识别实验。[原文 §3、Appendix A](https://arxiv.org/html/2608.04396v1)

## 当前证据能支持到哪里

| 本仓库结果 | 可以说 | 仍不能说 |
|---|---|---|
| 换目标词，动作头前 hidden 和 flow/action 输出有小幅差异 | 指令变化传入了动作计算 | 已理解物体身份；语言完全失效 |
| 目标词前 K/V patch 为零，全部文本 patch 精确复现 | 操作触及实际使用的语言读取路径，且对齐检查有效 | 已找到错目标原因 |
| 后缀文本 patch 接近完整换指令数值效应 | 前词的信息可能经因果文本网络传播到后续位置 | 后缀独自存储一个纯 milk 概念 |
| 原布局供体 GEN action 状态让接收者抓供体原位置的黄油 | 注入状态因果影响行为，效应依赖布局 | 该状态只编码坐标；模型逐字记忆某条训练轨迹 |
| 移动供体后目标变化，牛奶 6 cm 可跟随、15 cm 失败 | 空间适应存在边界，简单完全不看视觉的解释不符 | 坐标记忆是唯一机制；训练只覆盖默认 BDDL 范围 |
| 前10次文本替换通常比后10次 XYZ RMSE 大 | 语言数值效应依去噪阶段变化，后期也有作用 | 早期已经锁定错误目标；干预提高抓取成功率 |

特别是 [07](../experiments/07-language-path/README.md) 的 K/V 替换作用于全部 GEN queries，包括视频与动作。因此它尚未分开 UND text→GEN action 的直接读取与经过 GEN video 的间接影响。[08](../experiments/08-language-timing/README.md) 只有离线动作块数值，不能当闭环语义修复。

## 下一条最小且可证伪的证据桥梁

以下是建议，尚未执行。优先顺序与退出条件比大规模扫描更重要。

1. **先建立可解释的行为差异。** 在同一模拟器状态中，两条目标指令都可执行；优先现有 Goal 或 LIBERO-CF 条件，分别记录正确物体接触/抬起、错误物体接触/抬起、完整任务成功。先找至少一对原生策略确实能随指令切换的成功对照。当前 milk/butter 两提示抓同物体，不能定义“恢复另一条正确行为”的百分比；只能测小幅指令数值差异。
2. **解开身份、位置和布局。** 对真实可抓物体在两个位置交叉安排身份与指令，加入第三种独立布局。分别统计接近当前目标与接近历史目标位置。交换不同形状物体会同时改变可见外形和接触几何，所以预接触运动与完整抓取要分开。只有效应跟身份跨位置迁移，才支持身份相关使用；只有跟位置也不能证明训练轨迹记忆。需要自然渲染、可达性/遮挡检查和成对比较。
3. **从现有全部 GEN patch 收缩到 query 路径。** 单独替换 action queries 读取的 UND-text K/V；另一个条件只替换 future-video queries 读取文本。保留 recipient 的 vision、action 输入、residual、mask、RoPE、schedule 与缓存契约，验证 self-patch 和全路径重放。查询组 mask 必须真正限制作用对象，不能只在日志中标注 action-only。
4. **验证未来视频是否介导。** 对已建立语义行为差异的最小对，做“直接文本→action 开/关 × video→action 使用原/另一指令轨迹”的成对条件；在每个指定层、去噪步保持其余输入一致。若只改文本→video 能改变目标，且固定 video→action 为 recipient 后此效应消失，支持所测计算图上的中介贡献。若 action-only 仍能切目标，说明存在直接路；若只有整块交换能切，不能定位纯语义通路。视频与动作联合迭代存在交互，四格效应不可简单相加成百分比，也不能把“生成视频看起来对”当实际执行成功。
5. **以目标特异性和闭环完成结论。** 双向 milk↔butter、同义供体、wrong-target 供体、等范数随机方向、同状态 self-patch 为主要对照；训练分布内正常任务保真必须同时记录。使用多个独立初态/布局并配对种子，报告原始效应和不确定性，避免把同一场景的多个去噪 step 当独立样本。只首查询与全闭环干预分别测，后续环境反馈放大早期差异不等于内部不可逆锁定。

## 继续投入的门槛

**可形成公开复现笔记：** 完成官方 runner 契约核对，扩大行为对照，诚实保留零效应与失败，交付这一 checkpoint 上的反事实现象和可靠 instrumentation。

**可形成机制研究：** 同场景存在可满足、可观测的目标切换；局部读取路径产生双向、目标特异的闭环变化；身份与位置交叉实验支持同一解释；语义/范数控制不造成相同泛损坏；跨独立场景复现。还应验证至少一个额外 checkpoint 或相近联合视频—动作模型，以明确结果是权重个例还是架构规律。

**可形成修复方法：** 在独立任务/布局上提高正确目标与完整成功，保持原任务性能，并比较简单 CAG/语言重加权与相同计算预算的对照。单个漂亮视频或一层 4096 维 hidden 变化大、10 维 readout 变化小，都不足以诊断输出头，也不足以支持普适方法。

**明确推翻条件：** 若官方路径不复现，研究转为 runner 兼容性问题；若干预只停住/乱动，不能宣称恢复语义；若效应只跟供体位置，不能宣称抽象身份迁移；若每种局部语义干预均不稳定而整块 patch 有效，保留“混合状态驱动行为”结论，停止寻找单一概念层的过强叙事。
