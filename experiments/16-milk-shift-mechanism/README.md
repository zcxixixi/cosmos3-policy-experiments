# 牛奶移远：从真实内部数字到实际抓取

[打开最新网页和录像](https://zcxixixi.github.io/cosmos3-policy-experiments/#latest)。

我们只查一个问题：指令一直是“抓牛奶，放进篮子”，牛奶移6厘米能抓，移15厘米却抓奶酪盒。这一批没有重新训练，也没有额外喂示范。

## 归档：原样搬到15、18厘米，前9层干预没能救回牛奶

这是当时的完整记录；最新的内容／分配检验与四头因果干预见本文末尾，首页只展示最新批次。

我们想推翻一个很诱人的解释：之前改第1～9层能让它抓牛奶，会不会就找到了管牛奶的部件？所以不再调层数或改强度，原样保留这套改法，把牛奶移到15厘米和未用于选择范围的18厘米处。

**十二条实际执行全都严格抓起了奶酪盒。**四条不改网络、四条使用相同接口但保留全部连接、四条使用前9层干预，结果如下。指令仍为 `pick up the milk and place it in the basket`。

| 牛奶位置／未来随机起点 | 原生 | 保留连接的接口对照 | 前9层切边 |
|---|---|---|---|
| X+15cm / V195 | 奶酪，第70步 | 奶酪，第68步 | 奶酪，第70步 |
| X+15cm / V198 | 奶酪，第71步 | 奶酪，第71步 | 奶酪，第71步 |
| X+18cm / V195 | 奶酪，第72步 | 奶酪，第72步 | 奶酪，第71步 |
| X+18cm / V198 | 奶酪，第71步 | 奶酪，第71步 | 奶酪，第71步 |

步数是严格五份记录窗口的第一步：左右指垫同时接触且相对初态抬高严格超过2厘米。检查全部物体，不只检查牛奶。这不是最终放入篮子的成功率。

**“这套改法能在新位置恢复跟着牛奶抓”的主要预测被否定。**四条接口对照都保留各自原生的抓取类别，不能用接口改变了分类解释失败。旧12厘米的有效结果仍成立；它是局部场景中的因果效果，不是通用的牛奶识别区。

内部确实变了，不能说干预完全没有影响。四条干预的首轮最终XYZ动作，与各自接口对照的RMS差异分别为 **0.011865、0.021922、0.005319、0.017624**。这是归一化动作数字的距离，不是厘米或概率。改变一部分内部计算，没有让后续真实抓取改回牛奶。

固定改动为：仅首轮去噪0～14、网络第1～9层，禁止50个当前视觉query读取200个未来视觉key。动作直接读未来仍开放；之后7轮使用实际新照片和原生推理。硬mask也会重新分配剩余权重，尚未区分未来内容与读取比例。

[真实新初态及恢复检查](future-current-position-inputs) · [12份首轮、协议与360边界SHA](future-current-positions) · [全部12条真实录像和96份模型记录](future-current-position-execution) · [CPU重算、实际NPZ和图](future-current-position-analysis)。首轮360次、后续2520次网络计算，每条128步。作图未新跑模型或物理；CPU核对1536次动作转换、96份模型记录、77组后续噪声配对、360份边界和540处切边。四条原生是新位置的首次观察，不是独立重复，也没有与旧12厘米整轨迹作相等性对照。

GitHub公开12份t0完整边界，余348份完整边界保留A6000，全部360份SHA和全部时刻统计公开；执行目录中的重复t0边界不重复提交。没有公开全部QKV，也没有把4096个坐标硬分成head。代码见 [采集](code/collect_cosmos_milk_future_current_positions.py)、[首轮推理](code/probe_cosmos_milk_future_current_positions.py)、[实际执行](code/run_cosmos_milk_future_current_position_execution.py)、[CPU分析](code/analyze_cosmos_milk_future_current_positions.py)。

[独立原数组重算](future-current-position-analysis/independent-audit.json)另检查10836个真实接触与抬高条件、36份PT导出字节、24个最终动作距离、32个t0内部距离和24个t0整层净更新距离，误差0。它不重跑主分析器、模型或物理，也不冒称独立重算了全部QKV。[可复现代码](code/audit_cosmos_milk_future_current_positions.py)。

**该阶段提出的下一步：切边有效时，究竟是移除了干扰内容，还是重新分配读取比例？**保持同一实际Q/K和所有key，只改未来V的内容，再与硬mask比较。当时尚未执行；现已完成，结果见下文“未来内容与读取分配拆开测试”。仍未找到具体功能head或MLP单元，研究目标未完成。

## 归档：12厘米缩到9层，救回A并保住B

怀疑先前一次切18层太粗，会把有用的信息一起切掉。这次固定首轮前15次去噪，将第1～36层分成四个9层组；只禁止当前视觉读取预测未来的向量，动作直接读取未来的连接仍保留。照片、指令、机器人初态和动作随机起点相同；A／B只区别未来生成的初始随机数组。

| 首轮干预 | A：未来起点195 | B：未来起点198 |
|---|---|---|
| 原生／全连接接口对照 | 奶酪盒 | 牛奶 |
| 第1～9层（事先选定的主要检验） | 牛奶61～65步 | 牛奶64～68步 |
| 第10～18层 | 牛奶62～66步 | 无严格抓起 |
| 第19～27层 | 黄油76～80步 | 无严格抓起 |
| 第28～36层 | 奶酪盒71～75步 | 牛奶64～68步 |

**主要预测成立：第1～9层的干预在这两条轨迹里，救回了A，又保住了B。**全12条都实际执行，没有从其他组挑替代赢家。这表明该范围里的连接改动会影响实际抓取；还不能命名为牛奶区，也未解决15厘米或新位置。旧18层的预测仍然被否定，9层与18层效果不可以相加。

[首轮设置、真实数组、全部360边界SHA](future-current-layer-groups) · [全12条MuJoCo执行及96份模型记录](future-current-layer-execution) · [实际统计、NPZ与图](future-current-layer-analysis)。首轮360次网络计算，后续2520次；每条首16步干预后恢复原生反馈，共128步。严格抓起须左右指垫同时接触、相对初态抬高超过2厘米，连续5份记录；不等同于最终放进篮子。

[CPU审计](future-current-layer-analysis/complete.json)核对360份边界、1536次动作转换、96整记录、77后续整噪声配对及4条完整对照；作图没有新跑模型、求解器或物理。[独立重算](future-current-layer-analysis/independent-audit.json)另检查10836个真实接触／抬高条件、112个t0逐层距离、144个t0层净更新差、36份导出原数组字节和24个最终动作RMS，误差0。

三个冻结脚本见 [probe](code/probe_cosmos_milk_future_current_layer_groups.py)、[物理执行](code/run_cosmos_milk_future_current_layer_execution.py)、[CPU分析](code/analyze_cosmos_milk_future_current_layer_groups.py)。GitHub公开12个t0完整边界，余348个保留A6000；全部时刻统计和原始SHA公开。热力图是整层attention＋MLP净更新相对自身同起点接口对照的差异，不是某个head活动或牛奶概率。

**下一步：固定第1～9层改法，回15厘米并改变牛奶位置，检验是否真的跟着牛奶抓。**当前研究目标尚未完成。

## 归档：18层窗口的分层分时刻试验


**先说结果：事先的主要猜想被否定，但范围缩小了。**我们原本猜测，只在首轮前15次去噪、网络第1～18层切断未来到当前视觉的连接，能让抓错的A改抓牛奶，同时保住B。实际A抓牛奶，B没有严格抓起。后半段的同等规模改动保留两组原抓取对象。不能把这18层命名成牛奶区或运动区。

### 实际改了哪里

[首轮probe](code/probe_cosmos_milk_future_current_windows.py) SHA `8fef06cae12291884ee8a8fa60c981605f7813ccaa1c372f76f426be9662eb69`；[物理adapter](code/run_cosmos_milk_future_current_execution.py) SHA `19b907e5760f5873f4b1d0b834018a594f8d7420827369e731bc701ae4bcb08a`。未训练。当前X+12cm照片／指令／机器人初态／A195动作起点相同，只区分V195/V198未来生成起点；不是不同指令或画面。

四个事先冻结的窗口：去噪零基0..14或15..29，层零基0..17或18..35，各270实际cut位置。所有10个masked条件全1080位置都用同输入allallowed替换同66个当前／动作行；窗口内改用future→current cut的对应结果，200future行与UND保留本次原生。只阻断current50 queries对future200 keys，action16仍可读future。每active站点unblocked GEN对同输入allallowed逐字节同；不做native−sham残差补偿。4control所有30完整输出／37current-action边界和8窗before-firstcut完整kwargs/FP32样本前缀通过。

### 全部12条真实执行

| 首轮连接怎么改 | A：V195_A195 | B：V198_A195 |
|---|---|---|
| 原生 | 奶酪68..72 | 牛奶64..68 |
| 全连接接口对照 | 奶酪70..74 | 牛奶64..68 |
| 前半段、第1～18层（主要检验） | 牛奶59..63 | 无严格抓起 |
| 前半段、第19～36层 | 奶酪69..73 | 无严格抓起 |
| 后半段、第1～18层 | 奶酪70..74 | 牛奶64..68 |
| 后半段、第19～36层 | 奶酪70..74 | 牛奶64..68 |

严格条件是左右指垫同时接触且相对初态抬高严格>.02m，连续5份记录；所有物体／129份记录均检查。无严格抓起不等于没动作、也不能当确定选了哪个目标；抓起不是最终放篮成功。两组无严格条件的milk最高抬高分别0.314917cm（early_low/B）、0.110148cm（early_high/B）。

主要criterion是early_low两组 `selected_objects == ['milk_1']`。A满足、B不满足，结论 **rejected**。其他窗口只作预登记的描述性结果，不替换主要假设。前段两窗都影响B正确抓取，只有低18层救回A；后段两窗对象不变。支持此背景中前段读取更影响抓取类别，不证明晚段完全无作用、唯一必要区域、身份表示或背训练轨迹。

12cached首轮＋84新后续＝96请求，360首轮＋2520后续＝2880真实forwards，84后续各用自己的新照片，seed196..202，77次actual whole-noise配对。首轮16步执行后恢复原生推理，共128步／129状态。4原生／sham对照分别逐字节复现旧完整129state、5JSON、8PNG、8整模型record与16步controller，不用native代替sham参考。全部12执行自己的q0，没有成绩筛选。[完整执行与96模型record](future-current-execution)，[complete](future-current-execution/complete.json)，[主要预测](future-current-execution/primary-prediction.json) SHA `4ae592b7a0972c1a6b43ac3a549b6a6271d14722d188aa302cb95c427c4b978b`，[summary](future-current-execution/summary.json) SHA `a59218548c6d0016636b4a551d77efabd8a34c44a65788339c11c927d4e59beb`。

### 真正内部数组与公开范围

[CPU分析](future-current-analysis)读取所有360份真实boundary：current `[37,50,4096]`、action `[37,16,4096]`。boundary0输入，1..36整block attention＋MLP残差后的输出，最终norm之前。净更新是实际相邻边界差；图中亮度与原同V全连接接口对照的净更新差有关，不是head激活、12288MLP单元、牛奶概率或行为解释比例。比较对同Vsham与同arm跨V分开；窗口后输入已受先前干预影响，不能要求归回独立sham，也不把跨去噪后效应当网络内部记忆。

[全部首轮metadata/noise/readout/velocity/FP32solver/actions与t0逐层PT](future-current-windows)；[公开范围说明](future-current-windows/public-export.json)。公开12个t0完整boundary；另348个rawboundary在A6000，全部360SHA仍封存。公开统计覆盖所有层／时刻，连续FP32solver `[12,31,16,64]`、BF16速度的无损FP32导出 `[12,30,16,64]` 和final10d动作也保存。没有称公开全部7GB rawhidden，更没有称每site完整QKV另存。

[中文四窗结果图](../../docs/media/milk-future-current-windows.svg)、[实际层净更新图](future-current-analysis/primary-block-net-updates.png)、[同Vsham与同arm跨V曲线](future-current-analysis/same-sham-and-cross-source.png)。首页单播放器可切全部12条。旧结果保留供核验。

冻结[CPU分析器](code/analyze_cosmos_milk_future_current_windows.py) SHA `04758f1ea7768f6bdd725b328787b391b4b12dbc08a32626768d9110fdfcc609`；[analysis](future-current-analysis/analysis.json) SHA `ae529cc86bc136a18fdb35ac61c436b8b824282af66f80d4db831189b8bfd5fa`；[实际NPZ](future-current-analysis/actual-arrays.npz) 2,220,392 bytes，SHA `5d581afa50eb1c6e8d6d72654fbc36d54b3dc29ec52f9ec7d8d90b0da8b59fb6`。唯一CPU运行57.8秒，全360文件／1536转换／96record／77pair及四control核验通过，CUDA未初始化，0model／attention重放／solver更新／physics。35个导出key逐dtype/shape/Cbytes回读，FP32solver、BF16速度的FP32导出及最终动作对原PT精确。相对距离分母无无效值。另一位独立复核者另从全12实际数组重算物理／转换与source SHA，通过。

最终首轮XYZ对各自同Vsham的真实RMS：early_low A/B为0.355598/0.184003，early_high为0.191940/0.133736，late_low为0.008958/0.008276，late_high为0.003789/0.006487。是归一化动作数字的距离，不是厘米、抓取概率或恢复比例。前后去噪的sigma与可累积的后续计算不同，所以这个早晚差异也可能来自采样敏感性；尚未排除，更不能命名“选牛奶区域”。

[独立只读审计原件](future-current-analysis/independent-audit.json)另以实际PT重新计算80点hidden距离、64点整block净更新差、8组最终XYZ距离，均与公开NPZ误差0；24份solver、速度和最终动作导出对原PT逐dtype/shape/字节相同。它不调用原分析器的RMS函数，不新跑模型或物理。SHA `343ba76a09a76d61ac196cf329441b14448cfff5e4b1fad218db629c2172c6a0`。

**下一步：**把前段低18层再缩小，检验能否保住B，并加相同规模无关连接控制。最后仍须15厘米和新位置验证；当前研究goal未完成。

## 归档：活动图、跨层绕行和10条真实执行

**能记录哪些层改了较多数字，也已找到会改变实际抓取的一组连接；还没缩到具体层／head，更未验证15厘米救回和新位置跟随。**

### 活动地图与跨层传播

[冻结CPU活动分析器](code/analyze_cosmos_milk_future_read_activity.py) SHA `2835bf585095f143adc13ff9d5b61a392b9c30069bb937908a7961c24d8b5843`，[真实NPZ](future-read-activity/activity-arrays.npz) SHA `768f6565d162c8d80674192da95d911e1102ba0e72f004f065d60861348d2908`，[analysis](future-read-activity/analysis.json) SHA `17c46077d1b6312deb1ee6e3f16b529b12635c629a300e3062133b9e77d1c563`。只读CPU，0model／attention／solver／physics，CUDA未初始化。十组全300个逐层PT容器SHA核验、逐张量首次rawhash封存；独立复核10个t0完整PT并重算全部5arm两population b0/1/2/36，误差<1e-12。

真正保存的是 current `[37,50,4096]` 与 action `[37,16,4096]`，每组30次。boundary0输入，1..36完整block输出，均在最终norm之前。`RMS(Hb−Hb-1)` 是attention＋MLP残差的合并净更新，不能当全部计算强度、放电率、功能或重要性。nativeV195的平均动作净更新最大三层是人类第36／4／35层（0.321156／0.243269／0.172843）；当前视觉为36／35／4层（0.440760／0.420168／0.297605），任意decoder单位，仅此运行。4096个坐标不是head，更不是12288个MLP单元；future200完整逐层hidden未保存。

下表是t0两份V在**同一个arm**内的真实相对L2距离（分母V195同arm）。不能将native与sham差异混作路径效果，不能当成功率。

| 条件 | current第1层 | current第2层 | action第1层 | action第2层 |
|---|---:|---:|---:|---:|
| allallowed同接口 | 3.478% | 4.294% | 3.725% | 3.607% |
| 切future→current | 0% | 0.375% | 3.725% | 3.619% |
| 切future→action | 3.478% | 4.272% | 0% | 0.312% |
| 两条都切 | 0% | 0% | 0% | 0% |

实际t0每arm的两population入口b0均byteexact；一条入口被切时其b1也byteexact，b2重新出现差异。结合已核的UND不读GEN、逐行MLP/norm和真实mask，可以定位到仍开放的另一population跨层桥。不是同层循环、持续放电、吸引子、海马体或牛奶身份编码。Both全30×37两population逐位相同，是隔离V影响的工程结果，不是活动为零或语义理解证据。

[简单活动图](future-read-activity/native-net-update-overview.png)、[跨V同arm传播图](future-read-activity/cross-source-same-arm-propagation.png)、[对同V接口对照的切断影响](future-read-activity/cut-vs-same-source-sham.png)、[真5×10当前token grid](future-read-activity/current-token-grid-first-two-layers.png)。黑格在传播图表示两组没有差异，不表示无活动。逐token map只平均4096坐标，没有解码成像素或物体ROI。

### 正式网络干预和MuJoCo结果

[冻结q0 probe](code/probe_cosmos_milk_future_read_cuts.py) SHA `c106bba687c6eaf0853ac9abe7aa2767f9464365162399c201e98b2369af7a4a`。两个V195_A195／V198_A195来源、5arm，原照片/指令/prepare/schedule/54维padding保持。每masked站点同66行替换足迹，官方fullGEN dispatch上构建4D布尔mask；每实际site未屏蔽query对同输入allallowed byteexact，本次原生future200与UND保留，完整266行原投影仅一次。不补偿mask数值漂移。

10个新q0=300 forwards，UND10800＋GEN10800＋allallowed8640＋cut6480＝36720实际官方dispatch。两native整个记录byteexact各自固定A195旧源。记录readout `[30,16,4096]`、velocity `[30,16,64]`、solver `[31,1,16,64]`、actions `[16,10]`，全部原始draw/RNG/初始kwargs和30份边界。Both这些动作字段及所有current/action边界精确同。成功site fullQKV与每站原始返回没额外导出，不能冒称CPU独立重算全部live gate；真实GPU断言、事件和SHA仍保留。

[物理adapter](code/run_cosmos_milk_future_read_execution.py) SHA `e0c5073ee320bbc173eebf827126c9431043a3efaa960eb0a8ccff289cae30ed`。10cachedq0＋70新q1..7＝80请求；后者2100forwards，与300合计2400，63次实际后续whole噪声pair精确同。每轮用各自真实新画面。全部预登记10条执行，未按成绩选择、未训练。首轮切全部36层、30次去噪，后续原生；干预不是仅切一次局部层。

| 首轮条件 | V195_A195 | V198_A195 |
|---|---|---|
| 原生 | 奶酪68..72 | 牛奶64..68 |
| 全连接接口对照 | 奶酪70..74 | 牛奶64..68 |
| 切未来→当前视觉 | 牛奶60..64 | 无严格抓起 |
| 切未来→动作 | 奶酪79..83 | 奶酪80..84 |
| 两个入口都切 | 奶酪79..83 | 奶酪79..83，整个轨迹精确同 |

严格标准：每份同时左右指垫接触、比初态抬高严格>.02m，连续5记录；列出整个窗口，不是终局放篮成功。两native全部129仿真states、5JSON、8PNG、8整个模型records精确复现旧源；Both129states／5JSON／8PNG／后7wholemodelrecords精确同。两接口对照对象保留，**但数字并非noop**：归一化最终actions对native L2195=.01851345、198=.01964879；切断只与同V接口对照比较。全10窗口／实际首轮执行转换另有独立CPU复核。

**结论：**future→current切断能让A抓取改善，却让B未严格抓起；future→action或Both切断会损害本样本B正确抓取。未来读取关系参与行为，但不是只有有害信息。尚不能将网络区域叫记忆区／牛奶区，不能声称根因或通用修复。屏蔽会重归一化attention，且范围较大；下一步预登记分层窗口，区分对A有益与B有害的部位，再细分head，最后15厘米与新位置验证。

[10条原始闭环录像、全部轨迹动作及80模型记录](future-read-execution)、[complete](future-read-execution/complete.json)、[controls](future-read-execution/controls.json)、[summary](future-read-execution/summary.json) SHA `3d460470f1e7e6e844729952cac93ce95a964194e6fb44eaa9a0fe82d7439175`。

**公开原始文件范围：**q0十组所有states/noise/indexes＋每组t0完整boundary（共10份），见[public-export](future-read-cuts/public-export.json)；其余290份逐层文件存A6000，原results列全300SHA。统计NPZ覆盖全部层和30时刻，并包括预定0／10／20／29时刻的token0真实坐标与逐channel聚合；不是全部6GB rawhidden的公开副本。全部physical文件已公开。不是权重下载包。

方法灵感：[On the Biology of a Large Language Model2025](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)的活动、连接与干预思路；[Heimersheim/Nanda2024](https://arxiv.org/html/2404.15255v1)的patching与多路径限制。这里没有训练SAE/CLT/替代网络，也不证明模型有海马体。此前[Rao/Ballard1999](https://www.nature.com/articles/nn0199_79)预测反馈只是提出假设的灵感。

**此前四起点结果：这四次实际抓取，跟着未来生成一路的随机起点改变。**当前12厘米初态、当前照片和指令都相同，首轮交叉V（未来画面）和A（动作）两份真实初始噪声。V195两次抓奶酪盒，V198两次抓牛奶；后续采样统一base195。具体内部通路尚未定位，不能写成记住答案、把想象当现实或最终放篮成功。

## 新完成：未来生成起点 × 动作起点，四条真实执行

| 首轮V来源 | 首轮A195 | 首轮A198 |
|---|---|---|
| V195 | 奶酪盒，68..72 | 奶酪盒，70..74 |
| V198 | 牛奶，64..68 | 牛奶，65..69 |

真实当前帧保持；改变的是未来预测所用的初始随机数组，不是换照片或输入目标标签。q1..7按各自真实新画面原生推理，统一seed196..202。严格抓起要求5份连续记录均左右指垫接触且抬高>2厘米；数字是窗口范围，不是单帧确认。全部四组、所有物体的129帧严格指标已独立只读重算，没有其他合格物体。

[首轮探针](code/probe_cosmos_milk_initial_noise_factors.py) SHA `9081abb51200d1335bb4f84fde87fc23eec45a97ca46789f2dcf700acaf26c1e`。先运行两个原生条件，整个q0记录逐张量byteexact旧源，才运行两混合条件。实际randn seam消费两次原生draw，保存消费前后RNG与返回源；未来FP32 draw `[1,48,5,10,20]`、动作FP32 draw `[16,64]` 分别逐字节等指定源。12项prepare返回除实际噪声两项外不变；全部30次真实布局、当前帧和padding控制通过。30次读出、速度、31份FP32求解器状态都保存了。4份新q0预测=120次model，0physics；该探针没有网络路径屏蔽。

[实际执行适配](code/run_cosmos_milk_initial_noise_execution.py) SHA `985f10b079684689dcd9735f8f55159c97976eaef1ac33ff163011980f6ca9d8`。两个原生完整129simstate、128动作、5JSON、8PNG控制byteexact旧source195与此前source198/later195交叉；后者窗口65..69，不能误用原生198/later198的64..68参考。四组各自首轮16动作精确执行。另独立用冻结仿真体重算全部512条norm→raw仿射还原、6D旋转转换、夹爪映射与裁剪，原dtype/shape/bytes全部吻合。

32请求=4已保存q0＋28新预测；后者840次model，与前段120合计960。实际21组后续whole纯噪声配对byteexact；q0两份噪声来源分别记录，不假装是单一共同seed。后续输入画面受执行影响而不同。两原生录像与已有MP4完全相同，网页复用；两混合录像为新真实物理产物。四组首次动作／第一步状态就出现差异，终局标签不能反推q0已经认出或决定目标。

[四份完整q0真实数组与核验](initial-noise-factors)、[四条执行、全部trajectory/controller/模型records与噪声](initial-noise-execution)、[物理汇总](initial-noise-execution/summary.json)、[控制](initial-noise-execution/controls.json)、[complete](initial-noise-execution/complete.json)。summary SHA `f1a55fe88087b0e2f2044804805df102ce6f4379a5d84855a149fc7237a1820c`。仅此初态和两对V/A样本，不是统计成功率。

## 已执行：未来→当前视觉、未来→动作的入口

这10组与活动图已完成，见本文件顶部。此前[15次独立官方attention算术核验](current-mask-dispatch-output)只是接口可行性控制，0model／physics；其allallowed数字漂移被保留，没有用native−sham残差修补。正式10组使用fullGEN266 Q读UND121＋GEN266 K/V，非因果；统一替换current50＋action16的66行，保留本次原生future200和UND输出，整个原生投影仍做一次。掩码同时改变剩余读取权重的归一化，不是从结果简单减去未来项。

**内部单元实验：**预选128个真实MLP单元没有承担一半响应，门控路也没有主导作用。三个无关方向的输入也受这组位置影响，不能把它们叫作牛奶身份单元。抓错根因尚未定位。此前持续改这块MLP的实际四条执行仍抓奶酪盒。

## 此前解释：把真正的Flow计算公式带回实数组

[CPU分析代码](code/analyze_cosmos_milk_flow_response.py)，SHA `e49296bd7565ccb4a8ad47e70a4949b8b37e4b4165fb6d3e4906efc13ad8d1ff`，复用此前24组全数组审查，另核真实scheduler/pipeline源码、完整时间表、有效输出mask与FP32状态→实际BF16模型kwargs逐字节关系。0model、0physics、0scheduler.step；只在CPU调用真实 `convert_model_output`，不是重新采样。真实σ21=0.299699991941452，公式 `x_clean=x−σv`，参考比例1/σ=3.3366700930554423。

| 既有数据的约定 | clean差沿输入方向的投影 / 输入差 | 全部XYZ clean差长度 / 输入差长度 |
|---|---:|---:|
| P原生：FP32动作＋P未来视觉 | −7.867% | 32.290% |
| PR反事实：保存的FP32动作＋固定R未来视觉 | **−2.113%** | **26.764%** |
| PR实际BF16模型输入约定 | −1.669% | 22.377% |
| 无关方向1：BF16模型输入约定 | −0.857% | 21.438% |
| 无关方向2：BF16模型输入约定 | −3.558% | 25.800% |
| 无关方向3：BF16模型输入约定 | −6.183% | 21.114% |

公式中速度差与输入差大体同向、幅度约为1/σ，因而在clean估计相减时抵消大部分原方向差异。这支持普通sample去噪是主要解释候选，**不是严格恒定clean、不是错误抓取吸引子、不是说明因果解释了某个百分比**。总长度还有21%～32%，主要在原轴以外；不能把接近零的投影当成全部不变。不同约定有不同分母，不混为成功率。

实际source对前10个有效动作输出保持原头值，padding后54维置0；mask前后原始raw头逐字节不变。所有输入padding54为0。`x_clean`是本轮对无噪动作的估计，和UniPC下一步state或最终16步动作不是一回事。generic没有保存的FP32求解器轨迹，三个BF16约定均明确写为模型输入反事实；尝试FP32lift仅第一组cast吻合，后两组不吻合，没有捏造solver样本。

[真实NPZ数组](flow-response-analysis/actual-flow-arrays.npz)、[完整数值与sourceSHA](flow-response-analysis/analysis.json)、[图](flow-response-analysis/milk-flow-response.png)、[complete](flow-response-analysis/complete.json)。analysis SHA `c366b113092da301217d2fdc97ca50cdea7cdd9ae4bf75b9e546454c2862273a`，actual-flow-arrays SHA `874583fa571dd8f4f5e7afb54497807f7c3237d983a8972b6ae92246a5787c2b`。这次分析改变了对先前反向响应的解释，尚未定位抓奶酪盒的原因。

## 此前：真的拆到第4层内部单元

固定 seed198、去噪索引21、PR输入（受扰动动作＋原未来视觉），使用同索引RR的自然MLP数字。实际结构为 `down(SiLU(gate(x)) * up(x))`，宽度4096→12288→4096，bias均关闭。我们直接在真实 `down_proj` 的输入替换16个动作行的选定系数；残差、非动作行、未选系数和其他模块保留原值，后续32层继续实际计算。

名单来自原生 RR/PR 的实际系数差和真实 down 权重的局部投影，按有符号proxy降序、同分以编号排序。32/128/512名单嵌套，见[冻结名单](mlp-units/selection.json)及[实际权重列与系数](mlp-units/selection-arrays.pt)，先冻结再看干预输出，没有按12厘米抓取标签重排。这种排序是找候选，不是最终作用的线性分解。

| 事先的预测 | 实际数值 | 结论 |
|---|---|---|
| 128个承担至少整块作用的一半 | 全部12288个恢复67.871%；半数阈值33.936%，实际128个17.282% | 未成立 |
| 自然门控路比另一分支强 | g-only5.777%、u-only18.047%、joint17.282% | 未成立 |

32个为8.071%，512个为36.936%。这是此预选集合的结果，没有排除其他可能128集合。两分支的自然改动幅度不同，不能据此判断哪路更“理解”物体；joint也不是两单独效应相加。

这里的恢复指标是 `dot(Vtrial−VPR, VRR−VPR)/||VRR−VPR||²`，取16×64原始Flow输出的XYZ三列。0为原PR在此轴上的位置，1为RR；不是抓牛奶概率、实际位移、最终动作或任务成功率。

### 同一群单元会不会也处理无关方向？

以同三份预注册CPU种子构造三个XYZ方向，和原PR−RR的XYZ输入差近乎垂直，实际长度匹配容差0.1%；其他61维动作及全部其他kwargs保持RR。128个仍用旧冻结名单，每个方向先实际跑native，再在它自己的真实MLP输入上换入RR系数。

| CPU input seed | 该输入自己轴上的恢复 | 原生XYZ输出差与原PR轴的余弦 |
|---|---:|---:|
| 5219801 | 5.751% | 0.00566 |
| 5219802 | 9.552% | 0.05695 |
| 5219803 | 9.891% | 0.02344 |

每组分母是 `VRR−VGi_native`，不能直接当相同语义指标。XYZ控制没有改原PR另外61维的差值；真实差值范数 XYZ=0.047259、其他61维=0.011639、全部64维=0.048671，保存于[input-contract](mlp-units/input-contract.json)。结果表明这组位置对多个动作方向有响应，削弱专属目标竞争解释，但三个方向不足以证明整个MLP没有目标信息。

两类随机对照均匹配真实BF16 down输出差的L2，容差0.1%：随机支持128个位置，沿各自自然系数差，恢复6.712%、10.142%、7.774%；同选定128个位置、独立随机系数方向，恢复−1.586%、0.563%、1.600%。前者是均匀从12288里抽位置，不是从top512抽。样本量小，没有报告显著性或成功率。

### 中断、续跑与检查范围

[原probe](code/probe_cosmos_milk_mlp_units.py) SHA `6818a6a3c3ae2f665040f2196c00055fc9487b7775642601e7ca7e9e69c95a6a` 跑了18次实际单步计算，因第一次generic输入在BF16舍入后长度／垂直检查失败而停止。[failed.json](mlp-units/failed.json)保留，原目录没有complete或退出后参数检查。

修正只涉及CPU输入构造：[helper](code/cosmos_quantized_orthogonal_input.py) SHA `15cd6e671014c69dc1db4b6a06b6b86a9c1ba615f2a96c08918a8e8f3690b842`。原随机draw、三个种子、阈值、128名单均不改；同时反馈舍入后的长度和方向误差，取第一个合格输入。三个输入在任何新模型输出前冻结，分别150／494／840轮CPU构造后过门，长度误差0.01209%／0.02637%／0.07055%，实际cos绝对值<0.001。这些CPU迭代不是模型推理。

[continuation](code/probe_cosmos_milk_generic_inputs.py) SHA `614434e2e52cf4340c0533adb16d46c62f01c1b4a46918679b13aae626564a66` 只补6次实际single forward，47.06秒完成；前18次不重跑。它独立读取旧真实数组、重新计算全部指标、检查完整native/self/all12288/joint等价、实际分支/系数/非动作足迹及已存在SHA链。旧18个文件的全case manifest是在续跑时第一次完整封存，不冒称原过程已有完整results。

合计18＋6=24次模型计算，另82次真实down线性投影用于匹配随机改变量；额外投影不计成模型forward。新6次运行前后所有模型参数字节一致，运行前与旧初始参数一致；旧退出后参数状态未测，verdict=null。新[complete](mlp-generic-inputs/complete.json)、[results](mlp-generic-inputs/results.json)、[旧18独立检查](mlp-generic-inputs/phase1-validation.json)、[三个输入冻结记录](mlp-generic-inputs/inputs-frozen.json)均保留。新results SHA `feaaf050a038e15ccd3354d5b17af36fb82404536e0a46fddef0305b7574a791`。

GitHub保存24组动作行真实 `unit_arrays.npz`（BF16转FP32逐项一致，值可精确表示）、原生RR/PR全GEN `unit_site.pt`、实际选择权重列、输入纠偏完整audit、JSON指标与源哈希；其余完整PT留A6000原目录。不是只保存画图百分比。完整非动作足迹检查可由A6000原PT复验。

本批没有新MuJoCo执行。结合此前四条持续干预都抓奶酪盒，只能说第4层参与输入响应；尚不能称抓错根因。下一步转查同12厘米画面下正确／错误采样的实际文本读取通路，把“内容是否存在”和“动作是否用它”分开。


### 原生注意力输入核验：完成90次网络计算＋3次官方注意力重放

固定12厘米初态，重放seed195／196／198的原生q0。使用[新只读观察脚本](code/probe_cosmos_milk_target_reader_inputs.py)，SHA `a71604115ee797cef97662f29c9f76e536d84f7e5515d1a725526c0a0d08d2eb`；只在第18层（索引17）、去噪索引29的真实官方dispatch保存post-normalization/RoPE Q、K和原生V，另保存fullGEN原输出、真实pre-to_add_out、完整kwargs及原生tuple返回。

三份完整q0记录（包括两份噪声、全部30次输出及未来latent）均逐字节等旧threshold记录；保留实际GPU args、shape、stride、kwargs，额外每份重算一次完整GEN官方attention，输出逐字节一致。不是用手写softmax或截短16个query来宣称忠实重算。真实长度UND121、GEN266、joint387；Q形状[1,266,32,128]、K/V[1,387,8,128]，16个动作行在GEN的250..265。backend配置为None；只记录启用flags，没有通过profiler证明具体使用了哪种内核。

真实UND的121个input_ids是Qwen聊天包装、JSON caption及边界；指令10个token占全局45..54，milk在48（id14074）。没有image_pad或vision_hidden_states，当前图像latent单独在GEN。实际tokenizer对完整原字符串重编码逐项等原IDs；milk→butter长度仍121、只有位置48改为14100，cream cheese则122，不能偷补padding说完全同位置。此处只核合法同长词的输入条件，尚未跑内容干预。

UND的实际causal K/V与GEN使用的UND-prefix K/V分别跨三个seed逐字节一致；当前condition frame0也一致，两份采样噪声确实不同。此结果是结构和算术核验，**不是牛奶语义解码或目标选择因果结论**。[完整实数组](target-reader-inputs)、[complete](target-reader-inputs/complete.json)、[results](target-reader-inputs/results.json)保存全部源链；results SHA `19dca4a76998e8b9212c3c286ef17f354525bf8a97c758656c18831b805128e8`。没有新动作干预、没有新物理执行。

检查已有真实trajectory后发现一个重要边界：第16步seed196与198沿奶酪→牛奶方向的累计位移几乎相同，相差约0.061厘米，且均未接触物体；第24步才明显分流，已经包含q1新画面与新噪声。完整执行每轮使用base+query的不同噪声。因此不能从终局对象标签反推q0 Q198就是正确目标读取。下一步先交叉q0真实动作来源195／198与后续q1..7采样base195／198，原生两组要求全部物理状态和动作精确复现，再判断该查哪轮网络。四组交叉已经完成，结果见下一节。

[24次单步效应图](mlp-generic-inputs/figures/milk-mlp-units.png)、[真实128×16乘积变化热图](mlp-generic-inputs/figures/milk-mlp-unit-values.png)、[128单元换数后的37边界传播图](mlp-generic-inputs/figures/milk-mlp-unit-propagation.png)、[逐数与源SHA核验](mlp-generic-inputs/figures/plot.json)。传播图只显示改动相对原PR的数组差，前边界0..3相同，第4层换数后开始变化；不是牛奶信息量或目标概率。


## 新完成：首轮动作来源 × 后续采样序列，四条真实执行

[薄执行适配代码](code/run_cosmos_milk_query_noise_cross.py)，SHA `7e684b1a330877b8dbaad2c27621c3bc8db4893eb25da0c9171ff6d8b56aa2fe`。复用未修改的冻结物理执行体，固定12厘米初态和指令。q0直接复用source195／198封存的真实16步动作；q1..7各拍自身当前画面正常推理，使用laterbase195／198+query。没有局部网络干预或新训练。

| q0真实动作来源 | laterbase195 | laterbase198 |
|---|---|---|
| source195 | 奶酪盒，合格窗口68..72 | 奶酪盒，合格窗口68..72 |
| source198 | 牛奶，合格窗口65..69 | 牛奶，合格窗口64..68 |

合格窗口每份记录都同时双指接触且相对初态抬高>2厘米。JSON `first_selection_step`/`first_2cm_for_5frames`是连续5帧窗口的**开始**，不是第五帧确认时刻；例如68表示68..72。以往页面写“第68步确认”不够准确，最新表和视频已明确窗口。这一抓起指标不是篮内放置完成。

两原生195/198对照先跑：全129 simstates dtype/bytes、128动作及trajectory五份JSON和8PNG逐字节等旧threshold。交叉两组q1入口的前17state、16各格式动作、下一帧PNG和controller JSON值精确等各自q0源；新真实controller原dtype数组也逐字节等已完成native counterpart。旧trajectory只封存controller值，未虚构旧rawbytes manifest。

总32请求，其中4q0读取旧动作、28新预测=840model forward；重复的16份纯噪声对照均逐字节相同。缓存q0 metadata保留历史model_calls30，同时显式current_execution_model_calls0；真实8个query seed完整保存，不将cross写成错误的单base规则。后续画面因动作状态不同而不同，不能把相同noise stream叫同一输入。

**结论：** 在这个初态和两条后续采样序列中，最终抓谁跟着首轮动作来源；第一轮造成的手臂状态/视角等差异是重要因果条件。但整轮动作同时改变多个分量，尚未区分哪个状态、相机线索或网络通路起作用，也未证明q0已选定目标。不能说之后任何随机采样都无效。随后完成了q0初始video/action两份噪声来源交叉，见本文开头的新结果；具体网络路径尚待检验。

[四条真实视频与完整记录](query-noise-cross/closed-loop)、[汇总](query-noise-cross/summary.json)、[控制检查](query-noise-cross/controls.json)、[完整source与8个实际seed](query-noise-cross/provenance.json)、[complete](query-noise-cross/complete.json)。summary SHA `bef36c2c7c433a21b7314077a03a06e06acff03f8e406bc58fe9c5c2b787fb0e`。文件留A6000且GitHub保存四组全部模型records、真实物理状态、动作、输入、视频和controller数组。

## 我们实际做了什么

### 先复现九条正常执行

只改牛奶自由关节的 X 坐标，分别移动0、6、15厘米。机器人、物体其他坐标、相机、控制器和初态保持一致。每种位置使用配对种子195、196、198；后续每轮用“初始种子 + 轮次”的同一规则。

| 牛奶的位置 | 三个种子的抓起对象 |
|---|---|
| 原位 | 牛奶、牛奶、牛奶 |
| 移动6厘米 | 牛奶、牛奶、牛奶 |
| 移动15厘米 | 奶酪盒、奶酪盒、奶酪盒 |

这是一个场景、三份配对噪声，不是泛化成功率。抓起要求两侧夹爪同时接触、抬起超过2厘米，并持续至少5份记录。它与最后是否放好分开判断。每条执行8轮推理、每轮16步动作，共128步，保存129份真实状态。

[九条摘要](summary.json)、[真实输入](inputs)、[九条录像和逐步轨迹](closed-loop)、[实际机械臂轨迹图](trajectory-analysis/actual_eef_xy.png)。

### 新补测：9厘米还能抓，12厘米开始出现两种选择

再跑0、9、12厘米，每个位置仍配对195、196、198，共9条128步闭环。没有改推理设置或其他物体；原位三组首轮完整推理记录逐字节复现，而且128步动作、129份仿真状态、完整轨迹和每轮真实输入画面都完全等于旧原位基线。

| 新位置 | seed195 | seed196 | seed198 |
|---|---|---|---|
| 9厘米 | 牛奶，第60步确认 | 牛奶，第59步确认 | 牛奶，第61步确认 |
| 12厘米 | 奶酪盒，第68步确认 | 奶酪盒，第67步确认 | 牛奶，第64步确认 |

**12厘米同一物理初态，改变随机噪声就能改变抓取对象。**因此这里不是“超过某个厘米数必然抓错”的硬阈值。9至12厘米之间开始出现转抓，15厘米三个种子都抓奶酪盒。这个结果给“内部目标选择发生竞争”的猜想提供了条件，但三次噪声不够说明概率分布，也没有证明训练轨迹记忆或吸引子。

[新九条实际摘要](position-threshold/closed-loop/summary.json)、[三个原位完整推理对照](position-threshold/zero-model-control.json)、[真实物理与动作对照](position-threshold/physical-checks.json)、[新输入和实际录像](position-threshold/closed-loop)、[执行代码](code/run_cosmos_milk_position_threshold.py)。完整9份component captures留在A6000；GitHub保留真实q0记录、全部动作、实际输入和状态。

### 然后在网络里做56个短暂换数实验

**猜想：** 牛奶移出熟悉范围后，某种熟悉抓法压过了牛奶信息。暂时换入“移6厘米时”的局部活动，也许能推开错误动作。

受体是 `x15_seed198`，供体是 `x06_seed198`。供体实际抓起牛奶，受体实际抓起奶酪盒；它们的位置也不同，因此供体数字不能直接叫作“牛奶表示”。

测了7层（人类编号6、12、18、24、30、32、36）、两个部件（attention / MLP）、四次去噪时刻（代码索引0、10、20、29），共56次。每次只在一个计算时刻，替换一个部件给16个动作行提供的真实 `[16,4096]` 数组；残差、文字、视觉行等继续正常计算。干预只发生在首轮输入，不会在之后7轮反复施加。

先做4次对照：R和D原生复算完整记录逐项一致；attention和MLP换回自己的数字，完整记录也逐项一致。所有配对运行核对两份FP32原始噪声、时间表、全部30次结构与前置计算。实际供体、受体和换后数组都保存了，不根据假想的token位置干预。

| 三个最大正向变化 | 最终动作沿参考方向的分量 | 实际执行结果 |
|---|---:|---|
| 第18层 attention，去噪索引29 | 7.324% | 仍抓奶酪盒 |
| 第32层 attention，去噪索引29 | 3.494% | 仍抓奶酪盒 |
| 第30层 attention，去噪索引29 | 3.038% | 仍抓奶酪盒 |

指标为 `dot(P-R,D-R) / dot(D-R,D-R)`，取16步归一化XYZ。P是改后，R是15厘米原结果，D是6厘米原结果。0是R沿此轴的位置，1是D；垂直于此轴的变化未表示。**这不是抓牛奶概率、成功率，也不是纯身份分数。**

[56次全部数值](component-pulses/results.json)、[四个完整对照](component-pulses/validation.json)、[换数位置与效果图](component-pulses/figures/signed_xyz_pulses.png)、[最强三条后续动作数组曲线](component-pulses/figures/post_pulse_action_state_axis.png)。

### 把三条改后的真实动作交给 MuJoCo

首轮使用实际网络干预产生的16步动作。之后每轮用真实新拍画面重新正常推理。另跑一条不干预对照。

四条均抓奶酪盒，没有恢复抓牛奶。未干预对照的完整物理轨迹、128步原生动作、129份仿真状态与九条基线中的 `x15_seed198` 逐项完全相同。第18层改法在前16步使EEF参考点最多偏移约2.97毫米；整个128步最多偏移43.46毫米，但仍抓同一个错误对象。

[四条执行结果](candidate-execution/summary.json)、[录像](candidate-execution/closed-loop)、[候选选择与原始文件核验](candidate-execution/selection.json)。

## 新线索：当次有效，后续消失

第18层attention在去噪索引20短暂换数，当次动作更新沿该时刻的D−R方向偏45.567%；撤掉后，索引21的更新偏−9.933%；最终XYZ投影为−0.0766%。这些更新来自模型的实际速度输出，不是物理速度或厘米数。不同去噪时刻用各自的配对基准，不能拿百分比当同一把固定尺子证明“吸引子”。

### 已完成：拆开动作输入与未来画面输入

保持同一个去噪时刻（索引21）、同一sigma、当前画面和其他输入不变。R是未经干预的真实输入，P是索引20换过一次第18层attention后，真实产生的下一次输入。四组是：R动作+R未来画面；P动作+R未来画面；R动作+P未来画面；P动作+P未来画面。原生两组完整transformer输出（含视觉输出）逐项复现，另两组只改明确指定的输入数组。三份配对噪声全部核验通过。

| 后续反向响应从哪来 | seed195 | seed196 | seed198 |
|---|---:|---:|---:|
| 只换动作数组 | −9.366% | −10.399% | −9.427% |
| 只换未来画面数组 | +0.125% | −0.450% | +0.342% |
| 二者共同作用的额外交互 | +0.095% | +0.461% | −0.849% |
| 两者都换的总变化 | −9.145% | −10.388% | −9.933% |

这里所有项都用**同一索引21的D−R XYZ速度方向与分母**，可以按四组计算分解。它们不是最终动作的7.3%那种指标，也不是物理速度或成功率。

**这批不支持“未来画面是主要回拉载体”的猜想。**只换动作数组就重现大部分反向响应，未来画面单独的作用小且三份噪声不一致。这支持“动作输入路径产生了反向响应”；仍没有说明它为什么偏好抓奶酪盒。每份都保存真实37个层边界、72个部件增量和输入/输出，接着进行了下面的逐部件干预。

[四组真实数据与摘要](feedback-inputs/summary.json)、[代码](code/probe_cosmos_milk_feedback_inputs.py)。四组是固定输入的单次网络计算，没有经过新的环境反馈或后续调度器积分，不能当作实际抓取结果。

### 最新：第4层 MLP 是这次反向响应的重要部件

**可推翻的猜想：** 改后的动作输入进入下一次计算，有一个局部部件产生反方向的更新。保持改后的输入不变，只把这个部件的输出换回未扰动时的真实值，反方向更新应该减小。

在 seed198、去噪索引21，固定 PR 输入（P动作+R画面），依次替换36层的 attention 和 MLP，共72次。每次只替换一个部件的16个动作行增量 `[16,4096]`，使用 RR（R动作+R画面）同部件的真实输出；残差、非动作行和其余模块不被直接替换，后续层自由计算。每次替换前必须逐字节等于原PR缓存，替换后必须等于请求的RR数组。实际输入、sigma与时间完全固定。

| 局部改法 | 沿 RR−PR 方向恢复的分量 |
|---|---:|
| 第4层 MLP，真实RR输出替换 | 67.871% |
| 同位置、实际BF16改变量范数匹配，随机方向1 | 0.145% |
| 同位置、实际BF16改变量范数匹配，随机方向2 | 5.041% |
| 同位置、实际BF16改变量范数匹配，随机方向3 | 2.432% |

这个新指标为 `dot(patchedPR-PR, RR-PR) / ||RR-PR||²`，只取实际16×64原始Flow输出的XYZ三列。**它表示一次动作输入响应减少了多少，不是抓牛奶概率，也不是6厘米供体的方向投影。**用此前同一索引21的D−R轴表达，反向响应从−9.427%减到−2.932%。两种指标分别保存，不能混用。

反向对照在RR输入中换入PR的第4层MLP输出，沿独立的PR−RR参考轴改变65.985%。原生RR/PR，以及PR的attention/MLP自身替换四组，完整网络返回、动作头、37层边界及72个部件记录全都逐字节相同。三个随机方向在BF16实际舍入后匹配改变量范数，误差均小于0.1%。总计80次单次网络计算，不是80条闭环执行。

**支持什么：** 第4层MLP对这一次反向响应有显著、方向相关的因果作用；我们已经从整体输入路径缩小到一个具体计算部件。

**仍未证明什么：** 普通去噪也可能反对一个扰动，这不能直接证明错误动作吸引子或“记住答案”。这块是用同一种子、同一步的干预效果选出来的，尚需跨种子验证。也不能称它为“奶酪神经元”或“压制牛奶区域”。下一步持续改动这块，检查最终动作与真实抓取是否变化。

[全部72个部件和对照](action-correction-sites/results.json)、[真实热力图](action-correction-sites/figures/action-correction-sites.png)、[位级校验与绘图来源](action-correction-sites/figures/plot.json)、[实际干预代码](code/probe_cosmos_milk_action_correction_sites.py)。GitHub保存80条metadata、被选中部件/四个对照/反向与随机组的真实37层数组、动作头与换前换后数组；全部原始80组component caches及数组保留A6000，未全部上传。

### 已完成：持续改动与四条预登记实际执行

**检验：** 先在索引20短暂换第18层attention为配对6厘米供体数字；随后在索引21～29，把第4层MLP动作行增量换回未扰动的15厘米原生输出。当前输入只在首轮做干预，后面7轮按新拍画面原生推理。比较只改一次、连续9次、0.5/1/2倍强度及等强度随机方向，三种子×8组共24次首轮推理。之后输入样本自由变化，不被强迫等于原P或R。

| 最终XYZ沿每个种子自身的6cm−15cm配对参考方向偏移 | seed195 | seed196 | seed198 |
|---|---:|---:|---:|
| 只推一下 P | +0.156% | +0.238% | −0.077% |
| 后续第4层MLP只换回一次 | +0.196% | +0.193% | +0.008% |
| 后续连续9次、半强度 | +0.363% | +0.490% | +0.221% |
| 后续连续9次、正常强度 | +0.819% | +0.964% | +0.733% |
| 后续连续9次、双强度 | +4.492% | +4.391% | +4.138% |
| 后续连续9次、逐次等强度随机方向 | +0.146% | +0.286% | −0.055% |

正常R与没有先推的R自身连续替换，各种子完整推理记录严格等于原R；原P复算完整等于旧P，9个整记录控制均通过。首次替换的实际输入必须等于真实P索引21，替换前MLP必须等于PP replay（P动作+P画面）；后续只固定结构与sigma，真实样本允许变化。随机方向逐步按正常强度分支实际BF16输出改变量L2匹配，每步误差≤0.1%，使用独立CPU RNG，不改模型采样噪声。

**它说明：** 连续替换使一部分变化保留下来，方向比随机更一致；强度加倍时投影也更大。68%即时响应的恢复与这里的最终动作投影是不同轴、不同分母，不能互换。它们都不是抓牛奶概率，更不能把移动6厘米的供体动作当成15厘米位置的正确动作答案。

在看分数前，协议已登记 seed198 的四条物理候选：原生R、P、连续正常强度替换、随机替换。执行结果：

| 首轮候选 | 实际抓起谁 | 首次确认步 | 前16步EEF相对正常对照最大距离 |
|---|---|---:|---:|
| 原生 R | 奶酪盒 | 68 | 0 mm |
| P，只推一下 | 奶酪盒 | 68 | 0.069 mm |
| 连续换回第4层MLP9次 | 奶酪盒 | 68 | 0.372 mm |
| 连续随机方向 | 奶酪盒 | 68 | 0.075 mm |

每条后面7轮都读取真实新画面正常推理，共4次保存首轮＋28次原生新预测、32次请求。正常控制129份仿真状态、所有动作与轨迹JSON、8张输入PNG逐字节等于原 `x15_seed198`。连续换回组在整个128步中EEF最大偏移14.287毫米，但没有改变抓取对象。没有改为执行分数最高的双强度组；本批没有验证双强度的物理效果。

**推翻：** 在这个首轮、部件与干预协议下，“减弱反向响应就足以恢复抓牛奶”。

**尚不能区分：** 它是通用去噪部件，还是目标相关计算的一部分但作用不足。下一步把真实12288维MLP门控系数拆开，测试少数单元、gate/up两条支路及无关动作扰动，而不是继续把MLP整体变化命名为牛奶表示。

[24次真实内部计算与协议](correction-persistence/results.json)、[预登记协议](correction-persistence/protocol.json)、[四条实际执行](correction-execution/closed-loop/summary.json)、[完整正常控制](correction-execution/native-control.json)、[真实EEF偏移数值](correction-execution/physical-effects.json)。GitHub提供全部24条连续动作数值和metadata，以及上述seed198四候选的真实states/actualsession；其他完整原始PT均保留A6000。

[24次实际动作对照图](correction-persistence/figures/correction-persistence.png)由真实PT重算最终投影，独立核验9个整记录控制、首次PP输入/部件、实际BF16随机范数与封存哈希链。[绘图核验与限制](correction-persistence/figures/plot.json)说明：原探针保存了非动作行的原始字节SHA和相等报告，没有保存非动作原数组；绘图检查封存报告，不虚称独立重构这些数组。

### 已完成：第18层拆成32个真实attention head

在最终去噪时刻（索引29）只换一个真实head的128维输出，作用于16个动作行。钩子放在 `self_attn.to_add_out` **输入**，不是把投影后的4096个坐标随意分成32块。32个head全换时，完整推理记录与此前整attention换数逐项完全相同。

最强为代码head23（人类编号第24个）：单独换数，使最终XYZ沿6厘米参考方向偏2.085%；三个实际BF16差值范数匹配的随机对照分别是0.057%、0.089%、0.026%。反过来在6厘米输入中换入15厘米同head数字，动作沿反向参考方向偏2.041%。这是40次预测（4严格对照+32单头+反向+3随机），并非40条物理执行。

它表明这个head对两组动作差异有方向性作用，不能称为牛奶身份head，也未证明单头改动能恢复抓取。自然供体的各head改变量不同；没有把32项相加或当成独立作用占比。

[32head全部数值和随机对照](attention-heads/results.json)、[真实图](attention-heads/figures/head_xyz_and_random_controls.png)、[真实head数组](attention-heads/actual_head_captures.pt)。40次完整states保留A6000；GitHub提供每个head的真实换前/换后数组、连续XYZ输出与原始文件SHA。

## 真正的数组、代码和复现范围

- `inputs/`：恢复用的仿真状态、相机、控制器、物体数组，以及模型收到的真实画面；没有仅凭截图认定初态相同。
- `closed-loop/*/chunk_00/states.pt`：九条首轮实际输入、两份噪声、30次动作readout与速度、31份动作样本、最终归一化动作和预测视频latent。
- `component-excerpt/actual-bf16-bits.npz`：162块真实 attention/MLP 数组的位级摘录；保持BF16原始位模式。附每块key、shape和SHA256。读取：`torch.from_numpy(np.load(path)[key].copy()).view(torch.bfloat16).float()`。这些是动作行的部件增量，不是完整层残差，也不带“牛奶”标签。
- `component-pulses/runs/`：已执行三候选的真实states、换前/换后数组和metadata。其余完整56次记录保留A6000。
- `candidate-execution/closed-loop/`：实际MuJoCo状态、原生动作、轨迹和录像。
- `action-correction-sites/`：逐个部件的因果干预结果；选中部件、四个对照和四组跟进的真实层数组与输出，剩余原始数组保留A6000。
- `correction-persistence/`：24个正常／局部持续干预分支，连续动作、逐步改变量、原始哈希、预登记协议；四条物理候选有完整真实张量。
- `correction-execution/`：四条预登记候选实际动作、物理状态、contact/lift记录、录像和正常控制。
- `code/`：实际执行脚本、诊断与打包程序。`local-mechanism-literature.md` 说明文献如何对应可验证的猜想。

A6000完整原始目录：

```text
/home/current/work/cosmos3/outputs/official-demos/milk-posttrain/gripper-context/deep-dive/milk-shift-mechanism
```

九份完整component capture每份约270MiB，包含36层×30次×attention/MLP的真实动作行数组。它们没有塞进GitHub；位级摘录记录完整文件hash和来源。所有72次基线推理、56次换数和控制记录保留在该目录。没有删除原始实验产物。

运行顺序见 `code/` 的命令行帮助：基线server + simulator（先collect-only）；CPU component分析；56次component pulse；候选server + simulator。绘图使用 `/home/current/work/openpi-demo/work/cosmos-venv/bin/python`；模型推理使用 `/home/current/work/cosmos3/.venv/bin/python`。模型环境没有matplotlib，首次组件分析已产出数组后仅绘图失败；后来用 `--plot-only` 和绘图环境完成，未重新跑网络。该失败记录没有被当成模型失败。

设置：社区 `fwd4xl/cosmos3-nano-policy-liberoall-5k`；官方像素处理和默认30步整数时间表；system prompt关闭；domain5、raw10维动作、chunk16、guidance1、fps20；两路当前相机各256×256。模型BF16，原始随机噪声FP32。归一化XYZ/OSC命令无米单位，实际米单位来自MuJoCo状态；转换保留当前原生动作链路，没有加高度锁或其他纠正动作。

## 这批实验支持什么、不支持什么

支持：错误在当前匹配设置下能复现；特定层、特定去噪时刻的局部活动确实因果影响动作数组；单次这些干预不足以恢复实际目标。

尚不支持：已经定位抓错根因；第18层是“牛奶区域”；MLP在压制牛奶；模型背了某条训练轨迹；存在固定吸引子；逐头效果就是抓取效果。attention/MLP变化的负相关只是找候选的线索。

下一步要形成“具体内部改动 → 后续路径变化 → 实际目标切换”的证据，再把牛奶移到新位置，检查是否跟着牛奶走。

## 文献如何帮助这次实验

- [Inagaki等，运动计划与短暂扰动](https://www.nature.com/articles/s41586-019-0919-7)：启发暂时改活动、撤掉后看后续计算；没有证明模型和大脑机制相同。
- [Mante等，情境相关选择](https://www.nature.com/articles/nature12742)：启发区分信息存在与信息实际影响动作。
- [Mechanistic Interpretability and Steering of VLAs](https://arxiv.org/html/2509.00328v1)：借鉴模型内部的局部干预方法；需在当前模型重新核验。
- [How to Use and Interpret Activation Patching](https://arxiv.org/html/2404.15255v1)：匹配输入、干预范围与对照，避免把向量差异直接叫作语义或根因。

## 归档：未来内容与读取分配拆开测试（2026-10-04）

问题：X12前9层切断未来→当前读取，能让两组都抓牛奶。是去掉未来内容起作用，还是其余来源重新分配权重起作用？事先唯一主预测是 **H_content：只去未来内容，应使195改抓牛奶、198保留牛奶**。

| 操作 | 未来195／动作195 | 未来198／动作195 |
|---|---|---|
| 原生 | 奶酪，第68步 | 牛奶，第64步 |
| 全连接接口对照 | 奶酪，第70步 | 牛奶，第64步 |
| 同数值FP32计算／V复制对照 | 奶酪，第70步 | 牛奶，第64步 |
| 同Q/K只把未来V置零 | 未满足严格抓起 | 奶酪，第70步 |
| 保留未来贡献，只加其余来源重分配增量 | 奶酪，第66步 | 奶酪，第67步 |
| 原始前9层硬切边 | 牛奶，第61步 | 牛奶，第64步 |

主预测 **rejected**。未满足严格抓起的一条仍在末段接触奶酪盒；判据是两侧指垫接触、抬高>2cm，连续5条状态记录。有限128步观察下，不能称该组完全没有抓取动作。抓起也不等于最终放篮成功。

操作范围均首轮去噪t0..14、层0..8、当前视觉50行；未来200行和UND维持原生，动作16行维持同处全连接基底。后续7轮恢复原生推理。Z0是全连接、Zv是未来V置零、Zk是硬切边；allocation端点为 BF16(FP32(Z0)+(FP32(Zk)−FP32(Zv)))，零增量保留原Z0字节。它不是合法归一化softmax，可能增幅。只有同一实际站点内Q/K相同；后续各组样本、solver history和反馈会分岔。不能给出可加的因果百分比。

**支持什么：** 简单“去掉未来内容就足够救回”解释不符合两组数据；内容与重分配拆开都失败，合并切边仍有效。**还没解释什么：** 是否特定未来内容、整体幅度、下游交互导致差异；某个单head是否必要；能否跨位置修正目标。不能把未来向量直接叫奶酪动作记忆或牛奶身份。

12首轮预测／360forwards／38070官方attention dispatch完成，保存1620个实际站点Q/K/V、官方端点、实际pre-W字节、RNG和37层边界。物理12条／96请求／84新后续预测／2520forwards，共2880；8旧对照的完整129状态、5JSON、8PNG及64整模型记录精确重现。1536原生动作转换、77后续整噪声配对通过。

CPU分析7d791源核验全部1620个raw captures、每arm端点集合、实际FP32见证与一次BF16 cast（含signedzero）、真实物理summary/strict sealed来源。没有模型、官方attention、solver、物理或CUDA调用。全连接的270站点FP32 QK诊断和官方端点的相对RMS误差中位0.184%，最大0.406%；future权重均值26.94%／26.95%，200/387均匀基准51.68%。不是语义重要度。头图是操作造成的局部输出RMS，不是原生活跃度。各子图颜色尺度独立，对照全零会自动画成中间色。

**公开范围：** [future-value-mechanism](future-value-mechanism/)全部协议、12首轮整记录与首t0边界、每组第一站点t00-L00完整Q/K/V/官方端点；其余1608处raw PT和348份逐层PT保留A6000。全1620站点来源SHA及全部统计在[future-value-analysis](future-value-analysis/)。[future-value-execution](future-value-execution/)包含12录像、真实状态、96执行模型记录和控制；执行目录重复t0边界未导出。所有原始实验产物保留，没有删除。

已冻结实际代码：

- producer `0610b1d0320512ff4f5ce4163b52337ccb3ddf3ed1eb8d6c13f51b614097b911`
- adapter `e900f970e4049f6db1755274d68fa38366a9a29916354ccc8357090c44165a51`
- CPU analyzer `7d791c896896a132033574940c1479fb7ca6e0a33a4973936b8ff3023b601e34`

下一步只在这条读取通路内做head小组的局部因果干预，配自身和等强度随机对照；不凭热图命名脑区。本次目标仍未完成。


## 最新：第1层真实四头因果干预（2026-10-04）

先依据全部270处真实端点的几何比较，把候选缩到第1层（索引0），没有把其他层同编号head当成相同功能。G4=代码head16..19，两项局部差方向相似度约−0.96；G6=24..27，约+0.3。它们是数学候选，不是物体身份脑区。事先主预测 H_G6：只在首轮去噪t0..14、第1层的G6四头切未来→当前读取，应使195改抓牛奶、198保留抓牛奶。反向次预测为前9层切边但G6保留时，195失去救回。

| 条件 | V195／A195 | V198／A195 |
|---|---|---|
| 原生 | 奶酪68 | 牛奶64 |
| 全连接对照 | 奶酪70 | 牛奶64 |
| 前9层全切 | 牛奶61 | 牛奶64 |
| 只切第1层G4 | 奶酪70 | 牛奶64 |
| 只切第1层G6（主） | 奶酪69 | 牛奶64 |
| 前9层切但第1层G4保留 | 未满足严格抓起 | 牛奶64 |
| 前9层切但第1层G6保留 | 牛奶60 | 牛奶64 |
| G6同数值替换 | 奶酪70 | 牛奶64 |
| G6整体等范数随机方向 | 奶酪68 | 牛奶64 |

**主预测 rejected，反向次预测也不支持。** 没有把G4的不同结果替代失败的主预测。leave_G4／195虽失去严格抓起，牛奶发生最大9.024厘米平面位移并倒下，奶酪未动；它可能影响抓稳，而不是改选目标。两指持续接触、抬高>2cm、连续5条状态记录才算抓起；不等于最后入篮。

真实head轴来自pre-W的32×128；同QKV局部替换指定4头，其余头保留同处Z0，不切4096残差坐标。全部非原生组有同一union66全连接底座；t15..29撤head干预但底座保留，后续7个物理query恢复native。全／保留组覆盖层0..8，单组头／随机／自身替换仅层0。随机按4头整体Frobenius范数匹配，非各头独立匹配；独立CPU Generator、所有30处实际BF16相对范数误差通过事先10%解释门，没有重调或挑选轨迹。

18首轮预测540forwards、57090官方attention dispatch、2430真实sites（全部记录真实W投影、attention残差、MLP输出和层输出，无额外projection/MLP forward）。126后续native预测3780forwards，共4320；18×128=2304实际动作，144模型记录，119后续整噪声配对。8历史对照129states/5JSON/8PNG及64完整模型records exact。首轮全部30份实际kwargs/37层边界、两次BF16残差加法和实际层边界绑定通过producer/执行工程门；工程一致不等于因果机制。

公开：[270site真实几何分析](future-head-geometry/)、[18首轮原数组与预登记](future-head-groups/)、[18完整真实执行](future-head-execution/)。producer公开全部18q0 records、每组首t00-L00完整site与t00全部边界，所有2430site来源SHA；另外2412site PT与522boundary PT保留A6000。执行公开144全部records、18实际MP4/物理states/contact/controller/动作；q0重复boundary不再次导出。没有删原始产物。

冻结producer b261366b969a8f33980cbf4d450c381a18a64fa4b16be2e1dd95d6c7220ae898；物理adapter6767ee5bf166ebee8c0db7172472610311cf9611f88f72b04d6c10509fa3d064。已有局部因果线索，但这四头不足以解释根因，更没有实现15厘米及新位置稳定修正。


### 真实传播复核与分析运行问题

[完成统计](future-head-effects-matched-runtime/analysis.json)及[实际NPZ](future-head-effects-matched-runtime/actual-head-effects.npz)已核验全部2430 sites、540 boundaries、144执行记录、2304动作和119后续整噪声配对。该CPU分析没有模型、attention、solver、physics或CUDA调用。

V195／head_G6第1层当前50行，前15次去噪的跨组距离均值：pre-W 0.00920558、W输出0.03968275、attention残差0.03968753、MLP新增量0.00509991、完整层输出0.04004191。第四项是新增量，不是隐藏状态。第36层当前视觉状态0.03241203、动作状态0.01318290，最终XYZ 0.00228053，物理仍抓奶酪。这不支持“第1层MLP把改动擦掉”；也没有告诉我们这些差异代表哪种目标。不同状态的RMS不可相加，也不能当信息保留百分比。全9层切边最终XYZ 0.26697248并抓牛奶，不证明只放大G6就能救回。

最初CPU分析[c3d失败源码与failed](future-head-effects/)保留：Torch2.7随机方向求FP32范数与产生数据的2.14末位不同，精确见证门停止。未把分析失败当成模型失败。随后[原Torch2.14 CPU复核](future-head-random-norm-replay/)逐位验证全部30处私有随机状态、5个FP32见证、最终BF16、原double范数和完整来源SHA；2.7分析器绑定已精确复核的FP32标量位，再独立复算逐点运算与BF16字节。自身2.7 double范数差只作NPZ诊断，不偷偷放宽原强度判据。实际BF16相对范数误差最大0.000044253，全部通过事先0.1门；没有重跑18条模型或物理执行。修复输出使用新独占目录，原失败证据未覆盖。

冻结CPU复核源码`3b9bb679e4c4511c7fa9cedec87db9366afbd606c30239cbc20918deb3e29568`，修复分析源码`4754aa6ef31786956a39292db57ec002c050a351b79e3a5d1ba2f84e4a5c0a79`。全7份分析产物和全3份原版本复核产物SHA已验证导出。

下一假设（尚未执行）：九层切边可能只是让这一场景偏向牛奶，并没有恢复按语言选目标。固定场景、噪声和既定九层干预，只换牛奶／奶酪盒目标；原生、全连接与九层切边配两指令、两V，共12条件，先核对真实token变化。若切边不随语言换目标，就不能把现有救回叫作语言能力恢复。依旧分开验收接触、抓起、搬运，并保留所有条件；不从热图给部件命名。
