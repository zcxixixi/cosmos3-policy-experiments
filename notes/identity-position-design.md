# Cosmos3：把抓取身份、外观与位置分开的最小测试

日期：2026-10-03。本文基于实验 01–08 的本地 README/JSON、已归档脚本、远端安装的 LIBERO 源码及论文原文。仅做只读核验；未运行模型或新增物理 rollout。下述方案尚未执行。

## 当前结论和缺口

实验 02：牛奶 X+6cm 时仍抓牛奶。实验 03：牛奶/黄油 XY 互换后，两种目标指令都抓奶油奶酪盒。实验 06：原生策略、牛奶 X+15cm，也抓奶酪盒（63 步首次抓取、最高抬起 23.65cm），牛奶没有抓取接触。[实验02](../experiments/02-object-position/README.md)、[实验03原始量化](../experiments/03-target-swap/quantified.json)、[实验06原始量化](../experiments/06-milk-moved-15cm/quantified.json)

因此，抓错并非只能由 hidden-state 替换造成；原生策略存在位置相关的目标选择失败。但两种失败场景都没有移动奶酪盒，**奶酪盒身份/外观与它原来的位置完全重合**。当前不能判断它偏好奶酪盒、备用空间槽，还是凭整体视觉布局选择某种抓取行为。

第30层整块动作状态搬运的结果也不是身份检验：原始供体在接收场景中诱发抓旧牛奶位置上的黄油，+6cm供体诱发抓奶酪盒；第一块末端轨迹只相差约9.1mm，后来闭环反馈扩大差异。它证明行为受供体布局影响，不能单独定位“milk 概念”或证明身份信息不存在。[实验04](../experiments/04-state-exchange/physical.json)、[实验05](../experiments/05-donor-position/result.json)

这里的“身份”应区分三个意思：MuJoCo 的物体实例名、可见物体类别/外观、指令指定的语义目标。策略没有因为模拟器实例名叫 `cream_cheese_1` 就自动看到这个名字；全物体位置交换只能区分“跟随某个可见物体”与“跟随位置”，尚不能把外观、类别语义与可抓形状拆开。

## LIBERO Object 的真实槽位结构

核验路径：`/home/current/work/openpi-demo/work/cosmos-venv/lib/python3.10/site-packages/libero/libero/bddl_files/libero_object/*.bddl`。用 S-expression 解析全部10个文件的 `:regions` / `:init`，而非凭任务名推断。默认 `task_order_index=0` 的编号如下。

所有任务使用同样六个地面位置和同一篮子区域，但**目标并非全在同一个位置**。A–F 的中心与范围为：

| 槽 | XY中心（m） | XY范围（xmin, ymin, xmax, ymax） |
|---|---|---|
| A | (-0.12, -0.24) | (-0.145, -0.265, -0.095, -0.215) |
| B | (+0.05, -0.10) | (+0.025, -0.125, +0.075, -0.075) |
| C | (-0.15, +0.06) | (-0.175, +0.035, -0.125, +0.085) |
| D | (+0.10, -0.20) | (+0.075, -0.225, +0.125, -0.175) |
| E | (+0.15, +0.03) | (+0.125, +0.005, +0.175, +0.055) |
| F | (-0.20, -0.08) | (-0.225, -0.105, -0.175, -0.055) |

篮子中心 `(0, 0.26)`，范围 `(-0.01, 0.25, 0.01, 0.27)`。每个任务六个可抓物体如下；粗体是 BDDL 的目标，其他项是干扰物，因此表同时给出每任务完整物体集合。

| ID | 目标 | A | B | C | D | E | F |
|---|---|---|---|---|---|---|---|
| 0 | alphabet soup | **alphabet soup** | salad dressing | cream cheese | milk | tomato sauce | butter |
| 1 | cream cheese | alphabet soup | **cream cheese** | milk | tomato sauce | butter | orange juice |
| 2 | salad dressing | ketchup | **salad dressing** | alphabet soup | cream cheese | milk | tomato sauce |
| 3 | bbq sauce | chocolate pudding | **bbq sauce** | ketchup | salad dressing | alphabet soup | cream cheese |
| 4 | ketchup | **ketchup** | bbq sauce | salad dressing | alphabet soup | cream cheese | milk |
| 5 | tomato sauce | milk | **tomato sauce** | butter | orange juice | chocolate pudding | bbq sauce |
| 6 | butter | **butter** | tomato sauce | orange juice | chocolate pudding | bbq sauce | ketchup |
| 7 | milk | **milk** | cream cheese | tomato sauce | butter | orange juice | chocolate pudding |
| 8 | chocolate pudding | **chocolate pudding** | orange juice | bbq sauce | ketchup | salad dressing | alphabet soup |
| 9 | orange juice | butter | **orange juice** | chocolate pudding | bbq sauce | ketchup | salad dressing |

源码来源：[官方BDDL目录](https://github.com/Lifelong-Robot-Learning/LIBERO/tree/master/libero/libero/bddl_files/libero_object)、[milk原文](https://raw.githubusercontent.com/Lifelong-Robot-Learning/LIBERO/master/libero/libero/bddl_files/libero_object/pick_up_the_milk_and_place_it_in_the_basket.bddl)、[cream-cheese原文](https://raw.githubusercontent.com/Lifelong-Robot-Learning/LIBERO/master/libero/libero/bddl_files/libero_object/pick_up_the_cream_cheese_and_place_it_in_the_basket.bddl)、[任务编号源码](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/libero/benchmark/libero_suite_task_map.py)。实际执行依据上述安装版本，不假定远端 master 永远不变。milk/cream-cheese 文件 SHA256 分别为 `9910aabf6717e8ba3e24a3f8500d9bce9ee075346ea6961e8fd766f1257d2996` / `7019f37ee158d67a21338a6df0c441dd8f84979b946b5e20bb49463ef0508ea2`。

这提出一个具体可证伪的候选解释：牛奶在某段位置范围时选择A附近的牛奶，偏离后选择B附近物体。当前失败选中的奶酪盒恰在B；B也确实是另外5个原生任务的目标槽。它仍只是候选解释：X+6cm的牛奶已在A的BDDL范围之外而仍被抓，排除了把该方框边界直接当作策略阈值。BDDL是环境配置，**不证明 Cosmos3 checkpoint 看过什么训练样本、各布局频率或目标选择规则**。跨原生任务同时改变目标词、物体集合和槽位身份，也不能单独隔离位置或语义。

## 先做一项最小新增实验

从实验06的完整初态开始：牛奶保持 `(-0.1171608 + 0.15, -0.2442955)`，所有其余条件、首个种子198、后续每查询种子递增、128执行步均保持现有约定。只交换奶酪盒与黄油的XY：奶酪盒从B约 `(0.047, -0.104)` 到D约 `(0.096, -0.196)`，黄油相反。取值使用模拟器原始完整精度，不用这里的四舍五入值。保留各物体自己的Z、四元数、关节速度和机械臂状态。指令仍是 `pick up the milk and place it in the basket`，全程原生推理。

可区分的预测：

| 新场景观察 | 对候选解释的含义 |
|---|---|
| 先去D、抓D的奶酪盒 | 反对“此失败条件下一直选B”；支持跟随奶酪盒的可见外观/类别/形状，不能单独称语义身份 |
| 先去B、抓B的黄油 | 反对稳定跟随奶酪盒；支持B位置占据者在此条件下控制选择 |
| 先去原来牛奶位置A，那里空了 | 支持旧位置运动倾向；最终没抓到不等于信息丢失 |
| 改去远处牛奶 | 说明干扰物布局因果影响牛奶选择，简单“奶酪盒偏好”与“固定B”都不足以解释 |
| 早期无明确定向或全面失败 | 暂不能在身份/位置间裁决；先查碰撞、遮挡和操纵是否破坏抓取可行性 |

必须同时报告首次趋近和最后抓取；只报告最后成功/失败会把目标选择与抓取技术混在一起。奶酪盒移到D后与远处牛奶相距约7.9cm，先检查静态碰撞和两个政策相机的可见性。若初态不合法，调整对照布局后全部重建成匹配组，不只修改失败格。

## 最小完整的原生因子矩阵

上面的一个新增场景能快速排除一个简单解释；要形成更有说服力的结果，补成牛奶位置 × 奶酪盒位置的2×2，再在每格使用两条可满足的目标指令（milk / cream cheese），总共8格。相同物体集合、篮子、相机和机器人初态；没有 hidden-state 干预。

| 场景 | milk | cream cheese | butter | milk指令下简单备用B假说 | milk指令下偏好cream假说 |
|---|---|---|---|---|---|
| S00 | 原位置 | B | D | milk | milk（与已知成功一致） |
| S10 | X+15cm | B | D | cream@B | cream@B |
| S01 | 原位置 | D | B | milk | milk（若偏好仅在牛奶选择失败时触发） |
| S11 | X+15cm | D | B | butter@B | cream@D |

“依据指令选择身份”的理想预测则是：四场景在milk指令下都选当下的milk、在cream-cheese指令下都选当下的cream-cheese。若S11在两个指令下仍选同一物体，身份跟随仅能解释物体外观偏好，语言指定目标仍未获证明。若S01把原本正确的milk也变成错误目标，交换干扰物触发了更广的场景变化，不能只称失败时的备用规则。

执行顺序：先S11/milk（与现有S10直接成对）；随后S01/milk（检验操纵是否在正确场景也造成失败）；再补同场景cream指令。先用已有种子序列形成探索结果，对能产生区分的条件再用三个预先确定的独立种子序列重复，每个seed在所有相应条件保持匹配。3次不宜宣称总体成功率或统计显著。场景状态在每次运行前精确还原，不把上次rollout末态当下次初态。

原生替代指令的环境目标判定要单独计算：milk BDDL内置 `done/success` 仍只检查牛奶进篮子。换指令抓/放奶酪盒时，必须检查奶酪盒接触、抬起和进篮子，不能因为原环境返回false就说替代任务失败。[原生成功判定](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/libero/envs/problems/libero_floor_manipulation.py)

## 观测和验收

每格保存完整MuJoCo状态、全物体初始姿态、接触检查、两相机输入、指令及完整prompt JSON、policy/noise seed、每块动作、每步EEF和每个物体姿态。初态对照必须验证除预期milk X、cream/butter XY外没有其他状态变化；图像差异仅作为呈现验证，不能替代状态检查。

目标选择用三个互补量：首16步及前48步的EEF轨迹；每物体首次进入预注册的水平距离范围（例如4cm，另保存原始连续距离）；首个双指接触且随后持续抬起至少2cm的物体身份。保存首次双指接触步、接触持续帧、最大抬起、最终放置。4cm适合作为粗略趋近判据，不是统一抓取几何，所有原始距离需保留。近乎平局的趋近标作不明确，不能靠最终成败倒推早期目标。

同一物体在交换后的位置上若从未能被正常目标指令抓起，最后抓错仍可能含抓取可行性因素；用早期趋近判定意向，并如实标注可行性未验证。奶酪盒/黄油碰撞盒接近但不完全相同：远端资产中半尺寸分别为 `(0.008936,0.021336,0.040608)` / `(0.00871,0.01977,0.03809)`，不能称等几何控制。milk/orange-juice也仅近似同形，碰撞盒偏移/方向不完全相同。

## 内部干预：先有自然行为差异，再定位中介

最稳的下一步是**同一完整场景、只换语言的路径干预**。实验07–08已证明：GEN读全部UND文本K/V时可精确复现另一个指令的Flow/动作，词前替换零效应，时序替换有原样复现。这比跨布局整块动作状态更贴近可归因的自然干预，但当时尚未证明抓取目标发生切换。[实验07](../experiments/07-language-path/result.json)、[实验08](../experiments/08-language-timing/result.json)

1. **前提与停止条件。** 先在上述8格寻找同场景milk/cream指令原生选择不同物体的情况。存在这样的对照，才有目标层面的clean/counterfactual差异。若两个指令仍抓同一个物体，不把微小Flow差异归一化成“成功恢复”，不靠更大注入倍率追求抓取切换；行为问题还需要视觉对照。
2. **精确输入控制。** 同场景、相同初态/图像/proprioception，固定action/future-video noise、scheduler、domain、去噪步和query。供体text K/V由换目标指令的UND自然前向生成。只替换GEN读取UND文本的路径，保留recipient残差、GEN输入和vision K/V；明确当前实现作用于全部GEN还是仅action queries。
3. **最小patch。** 先全集text路径作为“等价原生换指令”阳性检查，再测目标词、词后上下文的6层窗口。每个去噪forward在recipient当前噪声状态下读取匹配供体文本，禁止把供体整段动作轨迹/动作hidden塞进去冒充语义。首query干预后其余query恢复原生，和全query干预分别标注；选有离线差异的最小窗口进行闭环验证。
4. **双向与控制。** 同态self-patch必须逐输出精确一致，全集text patch必须复现原生另一指令的动作。milk→cream与cream→milk双向测试；另有同义目标供体、与目标无关但结构相近的指令、仅词前供体。若局部patch造成的变化只在一边出现，报告背景依赖，不能强行概括为唯一语言通路。重复seed并换奶酪盒B/D位置：语义效果应跟随当下物体，不能只把EEF固定拉向B。

如果原生实验显示身份/外观或位置确实控制选择，才值得继续做视觉路径的局部干预。建议仅在已知观测帧、用两相机可见物体分割和真实token映射选择被操纵区域，patch相关attention输出或MLP增量并保留skip/residual；保留recipient的action/future-video状态、RoPE和时间编码。正反向交换和self-patch、同槽无关物体区域、相同大小的邻近背景区域为对照。视觉编码可能有跨patch感受野，必须记录mask映射，并承认区域token仍携带位置与上下文。

**局部视觉token来自不同布局，仍可能形成不自然组合。** 精确self-patch只能验证实现；局部patch能复现特定行为变化，才支持该视觉路径在该背景下的因果贡献，不能自动解释为纯身份。要进一步分离外观和物理身份，可另行设计渲染外观与物理位置交叉：只交换visual mesh/material，保留collision mesh、mass、qpos和goal实例；匹配对照必须检验图像可识别性、UV/尺寸与真实物体间隙。这种不一致物理/外观场景也可能OOD，因此以早期趋近为主，并把结论限定为“外观线索因果影响”。单纯改实例名没有改变策略输入，应得到精确相同动作，不能拿它检验语义身份。

## 文献怎样约束表述

《Not All Features Are Created Equal》v1 §3.2/§4.3报告same-scene与cross-task injection；后者也可能使运动偏向供体位置。G.5的单特征致0%成功使用full-layer hooks；F.3承认skip/residual损坏夸大效应，修正MLP-targeted后2–5特征消融无显著效应（p=0.975）。因此Cosmos3的下一步应以自然最小对照、保留残差的局部路径和对象特异的行为变化为证据，不能用全面失败来宣布概念必要性。[论文原文，§3.2、4.3、F.3、G.5](https://arxiv.org/html/2603.19233v1)

《LangForce》v7 §2的vision-only对照经过重新训练；LIBERO Goal上9.8%对完整模型97.4%体现同视觉多任务歧义。它提出低条件熵导致视觉捷径的解释；这种理论与其他模型结果提供实验动机，并未诊断冻结Cosmos3，也没有测量其真实训练分布。这里只能把同一场景两个可满足指令当作必要的行为检验，不把LIBERO的BDDL或语言小差异当作Cosmos3训练捷径的证明。[论文原文，§2、特别§2.2与§2.4](https://arxiv.org/html/2601.15197v7)

可交付的最小因果结论形式是：**在固定milk位置和匹配初态下，改变cream/butter空间安排导致策略选择身份或目标位置改变；同场景可满足指令及局部路径对照进一步检验语言是否控制这个选择。** 只有完成这些分离，才讨论具体身份/位置通路；不根据整块hidden跨场景搬运断言某一层“理解了但没有输出”。

## 实际运行更新

2026-10-03已完成[实验10](../experiments/10-identity-position/README.md)：四种固定布局，各三个噪声种子；牛奶移远后，交换奶酪/黄油让被抓对象从奶酪变成其旧位置上的黄油；牛奶原位置时仍抓牛奶。此设计笔记中的建议不是全部已运行，具体执行范围以实验记录为准。
