# Cosmos3 社区 LIBERO 运行时约定审计

核验日期：2026-10-03。只读核对远端源码、checkpoint 元数据和已保存输入；本次没有加载完整模型、运行 GPU 推理、安装依赖或修改模型库。以下 CPU 检查在 `CUDA_VISIBLE_DEVICES=""` 下完成。审计针对当前实现与公开 framework 的等价性；社区训练者的完整原始训练配置尚未取得，不能把公开 recipe 的每个参数自动当成这份 checkpoint 的训练事实。

固定版本：framework `cf5d68c00d97ccd2480a2320ed652b92dec63102`；Diffusers `fef717ffb01f407d2637584ed936c16db908587a`；社区模型 `fwd4xl/cosmos3-nano-policy-liberoall-5k` revision `64798337c642c53f9e22332554ae92f76a86cf04`。两份远端模型库的 `git status --short` 均为空。文中源码链接固定到上述提交，行号采用本地 `nl -ba` 的原文件行号。

## 结论

已确认当前 Diffusers 输入与公开官方 LIBERO server 存在三项具体差异：多出的 system message、像素缩放前后的 uint8 量化顺序、30 步 native 时间表中两处整数 timestep。JSON 补充 idle 字段后，对非空任务描述的 JSON 本体已与官方 server 一致。动作表示、域、输出裁切、反归一化和环境转换的基本约定已对上。

官方这条 action/WAM 生成路径的 UND 索引只包含文字；视觉经过 Wan VAE 进入 GEN，包含 clean first-frame tokens。缺少独立 Qwen vision tower 调用不是本次查到的错误。上述实现差异对动作、闭环成功率、语言敏感性造成多大影响仍需受控推理验证，不能由源码差异直接断言。

## 已验证事实

### 1. 当前真实输入包含额外 19 个 system tokens

已用本 checkpoint 的 `text_tokenizer` 解码远端已保存的：

`outputs/official-demos/milk-posttrain/gripper-context/deep-dive/semantic-transport/states.pt` → `inputs['center_milk_198'][0]['input_ids']`。

该输入为 140 tokens，`und_len=140`，开头包含视频生成 system message，随后才是 user 的动作 JSON。只保留同一 user JSON，使用公开 `tokenize_caption(..., use_system_prompt=False)` 的函数体和相同 checkpoint tokenizer，再按官方 packer 追加 EOS、start-of-generation，得到 121 tokens。**这 121 tokens 与保存输入的末尾 121 tokens 逐项完全相等**，即多出的部分确实是 19 个 system tokens，而非 JSON 或 tokenizer 内容差异。

公开 Nano recipe 的 `vlm_config.use_system_prompt=False`；LIBERO recipe 从其深拷贝后没有覆盖该字段。训练 `ActionTransformPipeline` 创建 `TextTokenizerTransform` 时不传 system 覆盖，后者也默认 False。官方推理使用模型配置里的同一开关。Diffusers 构造函数默认 `default_use_system_prompt=True`，保存的 `model_index.json` 和 `Cosmos3OmniPipeline.load_config(...)` 都没有该字段；当前 server/semantic 脚本未覆盖此开关。已保存 token 输入证实这不是仅从默认参数猜测的风险。[Nano 配置 L113–117](https://github.com/NVIDIA/cosmos-framework/blob/cf5d68c00d97ccd2480a2320ed652b92dec63102/cosmos_framework/configs/base/experiment/sft/models/nano_model_config.py#L113-L117)、[LIBERO 配置 L28–41](https://github.com/NVIDIA/cosmos-framework/blob/cf5d68c00d97ccd2480a2320ed652b92dec63102/cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_libero_nano.py#L28-L41)、[训练入口 L654–664](https://github.com/NVIDIA/cosmos-framework/blob/cf5d68c00d97ccd2480a2320ed652b92dec63102/cosmos_framework/data/generator/action/utils/transforms.py#L654-L664)、[tokenizer L64–112](https://github.com/NVIDIA/cosmos-framework/blob/cf5d68c00d97ccd2480a2320ed652b92dec63102/cosmos_framework/data/generator/augmentors/text_tokenizer.py#L64-L112)、[官方推理 L2507–2536](https://github.com/NVIDIA/cosmos-framework/blob/cf5d68c00d97ccd2480a2320ed652b92dec63102/cosmos_framework/model/generator/omni_mot_model.py#L2507-L2536)、[Diffusers L419–435、1188–1220](https://github.com/huggingface/diffusers/blob/fef717ffb01f407d2637584ed936c16db908587a/src/diffusers/pipelines/cosmos/pipeline_cosmos3_omni.py#L419-L435)。

最小推理对照参数是 `pipe(..., use_system_prompt=False)`，也可以加载时传 `default_use_system_prompt=False`。无需修改库或权重。删除 system 同时改变文字长度及后续 GEN 的位置偏移，因此需要保留原始噪声并另设 GEN 位置控制，才可区分提示内容和位置变化。这里“公开官方不开 system”已验证；社区训练者的私有原始实验是否覆盖过它仍未知。

官方 chat template 为仅 user → assistant generation prefix；官方 packer在其后加 EOS、`<|vision_start|>`，当前 Diffusers 的这两个尾标与其一致。[tokenize_caption L28–70](https://github.com/NVIDIA/cosmos-framework/blob/cf5d68c00d97ccd2480a2320ed652b92dec63102/cosmos_framework/model/generator/reasoner/qwen3_vl/utils.py#L28-L70)、[packer L175–219](https://github.com/NVIDIA/cosmos-framework/blob/cf5d68c00d97ccd2480a2320ed652b92dec63102/cosmos_framework/data/generator/sequence_packing/modalities.py#L175-L219)。

### 2. JSON 本体与官方 server 相同，训练布局描述还有一个区别

对当前 milk 非空提示，官方 `ActionPromptJsonFormatter` 与 Diffusers JSON 加 idle 补丁的 CPU 比较为字符串完全相同：framing 为 concat_view 通用句；时间 `0:00-0:01`、`idle_frame="0 out of 16."`、duration `0s`、fps `20.0`、resolution `192×320`、aspect ratio `16,9`。这些短时长字符串的舍入和截断来自官方实现，不应擅自“修正”。[官方 server L844–864](https://github.com/NVIDIA/cosmos-framework/blob/cf5d68c00d97ccd2480a2320ed652b92dec63102/cosmos_framework/scripts/action_policy_server_libero.py#L844-L864)、[formatter L97–138](https://github.com/NVIDIA/cosmos-framework/blob/cf5d68c00d97ccd2480a2320ed652b92dec63102/cosmos_framework/data/generator/action/utils/json_formatter.py#L97-L138)、[Diffusers L1092–1130](https://github.com/huggingface/diffusers/blob/fef717ffb01f407d2637584ed936c16db908587a/src/diffusers/pipelines/cosmos/pipeline_cosmos3_omni.py#L1092-L1130)。

公开训练 dataset 会额外加入“左半是第三人称、右半是腕相机”的布局句，formatter 会接到 framing 后。官方 LIBERO server 与当前 runner 都没有提供这句，因此这是**公开训练与官方 server 自身之间**的差异，不是当前 runner 独有的错误。是否适用于社区私有训练代码，以及是否有行为影响，尚未知。[dataset L291–295](https://github.com/NVIDIA/cosmos-framework/blob/cf5d68c00d97ccd2480a2320ed652b92dec63102/cosmos_framework/data/generator/action/datasets/libero_lerobot_dataset.py#L291-L295)、[formatter L172–190](https://github.com/NVIDIA/cosmos-framework/blob/cf5d68c00d97ccd2480a2320ed652b92dec63102/cosmos_framework/data/generator/action/utils/json_formatter.py#L172-L190)。

官方 formatter 对空 caption 直接返回空串；Diffusers 对空描述仍生成 JSON 元数据。此前 empty-description 对照因此是“JSON 内空描述”，不能称为“官方完整空提示”。[formatter L89–91](https://github.com/NVIDIA/cosmos-framework/blob/cf5d68c00d97ccd2480a2320ed652b92dec63102/cosmos_framework/data/generator/action/utils/json_formatter.py#L89-L91)。

### 3. 图像尺寸和 clean-frame 布局一致，像素不逐值相同

公开训练配置使用 17 帧视频配 16 个动作、concat_view、两视角各 256×256。WAM 计划为 clean latent frame `[0]`、无 action conditioning、action temporal offset 为 1；当前 Diffusers `policy` 也使用相同条件布局并联合预测未来视频和动作。concat 图以 agentview 左、wrist 右排序；官方评估默认旋转180度、10个 warmup 步，与当前 collector/rollout 一致。[训练参数 L202–219](https://github.com/NVIDIA/cosmos-framework/blob/cf5d68c00d97ccd2480a2320ed652b92dec63102/cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_libero_nano.py#L202-L219)、[sequence plan L329–407](https://github.com/NVIDIA/cosmos-framework/blob/cf5d68c00d97ccd2480a2320ed652b92dec63102/cosmos_framework/data/generator/action/utils/transforms.py#L329-L407)、[视角 L322–330](https://github.com/NVIDIA/cosmos-framework/blob/cf5d68c00d97ccd2480a2320ed652b92dec63102/cosmos_framework/data/generator/action/datasets/libero_lerobot_dataset.py#L322-L330)、[评估 L814–821](https://github.com/NVIDIA/cosmos-framework/blob/cf5d68c00d97ccd2480a2320ed652b92dec63102/cosmos_framework/simulation/libero/closed_loop_eval.py#L814-L821)。

两套预处理都把 512×256 图缩为 content 320×160，再在底部 reflection pad 到 canvas 320×192，`image_size=[192,320,160,320]`。但官方训练和 server 将 uint8 输入送进 torchvision bicubic，其输出仍 uint8，经 clamp/round 后才映射至 [-1,1]；Diffusers 先归一化为 float，再做 bicubic，无相同 clamp/round。已用实际 center input.png 作 CPU 对照：**归一化像素 max absolute error=0.09907293、RMSE=0.002248974**；Diffusers 最小像素为 -1.09907293，超出官方范围。量化和 bicubic overshoot 的差异已验证；其对 VAE/策略输出的影响未验证。[训练 uint8 L196–205](https://github.com/NVIDIA/cosmos-framework/blob/cf5d68c00d97ccd2480a2320ed652b92dec63102/cosmos_framework/data/generator/action/datasets/base_dataset.py#L196-L205)、[官方 resize L175–228](https://github.com/NVIDIA/cosmos-framework/blob/cf5d68c00d97ccd2480a2320ed652b92dec63102/cosmos_framework/data/generator/action/utils/transforms.py#L175-L228)、[官方 normalize/encode L4558–4563](https://github.com/NVIDIA/cosmos-framework/blob/cf5d68c00d97ccd2480a2320ed652b92dec63102/cosmos_framework/model/generator/omni_mot_model.py#L4558-L4563)、[Diffusers L704–752](https://github.com/huggingface/diffusers/blob/fef717ffb01f407d2637584ed936c16db908587a/src/diffusers/pipelines/cosmos/pipeline_cosmos3_omni.py#L704-L752)。

CPU 测试没有安装 torchvision：一项在现有 `.venv-chat` 的 torch2.11/torchvision 中执行官方 helper；另一项提取该安装中无依赖的 torchvision tensor-resize 函数体，在当前 torch2.14 上对比实际 Diffusers `_prepare_action_video_conditioning`。两项给出相同最大误差和 RMSE；后者消除了跨 torch 版本的数值混淆。

### 4. UND 视觉缺失不是公开 action 路径不匹配

公开 action forward 先嵌入 text，再通过 `vae2llm` 投影视觉；真正调用 attention packer 时，`packed_und_token_indexes=packed_seq.text_indexes`。因此 clean 图像 tokens 仍属于 GEN stream，UND 为文字。独立 Qwen vision tower 是 reasoner/VLM 图像输入路径的能力；checkpoint 目录含 `vision_encoder/` 不意味着 action generation 必须调用它。当前 Diffusers 的文字 UND + VAE GEN 与此基本结构一致。[网络 L701–720、752–782](https://github.com/NVIDIA/cosmos-framework/blob/cf5d68c00d97ccd2480a2320ed652b92dec63102/cosmos_framework/model/generator/mot/cosmos3_vfm_network.py#L701-L720)、[实际 packer L1402–1412](https://github.com/NVIDIA/cosmos-framework/blob/cf5d68c00d97ccd2480a2320ed652b92dec63102/cosmos_framework/model/generator/mot/cosmos3_vfm_network.py#L1402-L1412)。

官方可只编码 clean 像素前缀，当前 Diffusers 编码重复的17帧后保留首 latent frame；理论上依赖 VAE temporal causality，但两实现首 latent 的数值等价尚未直接测试。[官方 prefix L4721–4744](https://github.com/NVIDIA/cosmos-framework/blob/cf5d68c00d97ccd2480a2320ed652b92dec63102/cosmos_framework/model/generator/omni_mot_model.py#L4721-L4744)、[Diffusers L884–886](https://github.com/huggingface/diffusers/blob/fef717ffb01f407d2637584ed936c16db908587a/src/diffusers/pipelines/cosmos/pipeline_cosmos3_omni.py#L884-L886)。

### 5. 动作域、表示和缩放约定匹配

LIBERO 域5、raw10、model64，10维为 xyz3 + rot6d6 + gripper1；公开 dataset 直接采用储存的每帧控制 delta，把 axisangle 转 rot6d，不计算相对初始/当前绝对位姿，也没有额外 xyz 尺度变换。`quantile_rot` 使用 `global_raw.q01/q99`，反归一化是 `q01 + (a_norm+1)/2*(q99-q01)`，不额外裁剪归一化动作。当前 rollout 用相同统计和官方 `_framewise_action_to_delta`/`_remap_gripper`，公式一致。官方 server 的 range 下限1e-6与当前1e-8不会在现有 stats 触发差别：最小range为0.061484。[dataset L108–132、284–306](https://github.com/NVIDIA/cosmos-framework/blob/cf5d68c00d97ccd2480a2320ed652b92dec63102/cosmos_framework/data/generator/action/datasets/libero_lerobot_dataset.py#L284-L306)、[server L718–736、776–793](https://github.com/NVIDIA/cosmos-framework/blob/cf5d68c00d97ccd2480a2320ed652b92dec63102/cosmos_framework/scripts/action_policy_server_libero.py#L776-L793)、[环境转换 L580–602](https://github.com/NVIDIA/cosmos-framework/blob/cf5d68c00d97ccd2480a2320ed652b92dec63102/cosmos_framework/simulation/libero/closed_loop_eval.py#L580-L602)。

这份提交的 `_remap_gripper(...,'zero_one')` 实际代码为连续 `-clip(2g-1,-1,1)`，虽然其 docstring 宣称 binary sign；报告应按代码描述，不能写成二值化。当前直接复用同函数，因此该问题并非两 runner 的差异。[实际代码 L523–532](https://github.com/NVIDIA/cosmos-framework/blob/cf5d68c00d97ccd2480a2320ed652b92dec63102/cosmos_framework/simulation/libero/closed_loop_eval.py#L523-L532)。

### 6. Native scheduler 算法同源，时间表还差两处整数

官方 UniPC wrapper 创建 `FlowUniPCMultistepScheduler` 并调用 `set_timesteps(30,shift=1.)`；公开 Nano inference config 为1000 train timesteps、shift1、无 dynamic shift，server 默认30步/guidance1。当前使用同一 scheduler 源码，但 Diffusers `use_native_flow_schedule=True` 先构造 `np.linspace(0.999,0,31)[:-1]`，再显式传 sigmas。[配置 L80–85](https://github.com/NVIDIA/cosmos-framework/blob/cf5d68c00d97ccd2480a2320ed652b92dec63102/cosmos_framework/configs/base/experiment/sft/models/nano_model_config.py#L80-L85)、[wrapper L68–106](https://github.com/NVIDIA/cosmos-framework/blob/cf5d68c00d97ccd2480a2320ed652b92dec63102/cosmos_framework/model/generator/diffusion/samplers/unipc.py#L68-L106)、[Diffusers L1691–1704](https://github.com/huggingface/diffusers/blob/fef717ffb01f407d2637584ed936c16db908587a/src/diffusers/pipelines/cosmos/pipeline_cosmos3_omni.py#L1691-L1704)。

CPU 对比结果：sigmas 最大差5.9604645e-8；但 `set_timesteps` 转int64，**step10官方t666/当前t665；step20官方t333/当前t332**。这来自默认 sigma_max 的float32值0.999000012874与显式numpy值0.999的舍入路径，不能只比float32 sigmas判定全部时间条件一致。用 `v=0.1*x+t/1000` 的30步CPU dummy flow，最大过程差9.5248e-5，最终差6.4135e-5；这是采样器对照，不代表真实动作差的量级。[scheduler L99–121、179–203](https://github.com/NVIDIA/cosmos-framework/blob/cf5d68c00d97ccd2480a2320ed652b92dec63102/cosmos_framework/model/generator/diffusion/samplers/fm_solvers_unipc.py#L179-L203)。

完全同源时间表应采用官方默认 `set_timesteps(30,shift=1.)` 或从同一 scheduler.sigma_max/min 生成，而非理想化0.999。另一个等价性细节是官方合并 vision/action 状态做一次 step，当前为各模态各有独立 scheduler 副本；在关闭 thresholding、同时间表、同初值/velocity 时应逐元素一致，但尚未运行合并/分开对照。

### 7. 相同 seed 不等于相同初始噪声

官方 `arch_invariant_rand` 用 numpy.RandomState，每次调用为 vision、action 分别按同一 seed 重启；当前 Diffusers 用同一 torch.Generator 连续产生各模态 randn。跨框架数值对照必须显式注入相同 tensors，不能只设相同 seed。现有同 runner 配对实验的噪声一致性不受此事实自动否定。[官方 noise L2666–2676、2730–2742](https://github.com/NVIDIA/cosmos-framework/blob/cf5d68c00d97ccd2480a2320ed652b92dec63102/cosmos_framework/model/generator/omni_mot_model.py#L2666-L2676)、[numpy实现 L144–165](https://github.com/NVIDIA/cosmos-framework/blob/cf5d68c00d97ccd2480a2320ed652b92dec63102/cosmos_framework/utils/misc.py#L144-L165)、[Diffusers noise L950–986](https://github.com/huggingface/diffusers/blob/fef717ffb01f407d2637584ed936c16db908587a/src/diffusers/pipelines/cosmos/pipeline_cosmos3_omni.py#L950-L986)。

## Checkpoint 证据和限制

该模型为社区发布；作者 model card 声称采用JSON、LIBERO四suite、frame-wise relative rot6d10D、quantile_rot、20FPS/chunk16、迭代5250，LIBERO-10 200回合95.5%。这只是作者声明，本审计未重现成功率。模型卡的warmup2000/cycle10000与当前公开文档的warmup500/cycle16000不同，因此不能只依赖当前公开 TOML 还原历史私有训练。[作者模型卡](https://huggingface.co/fwd4xl/cosmos3-nano-policy-liberoall-5k)、[固定revision文件](https://huggingface.co/fwd4xl/cosmos3-nano-policy-liberoall-5k/blob/64798337c642c53f9e22332554ae92f76a86cf04/README.md)、[公开recipe](https://github.com/NVIDIA/cosmos-framework/blob/cf5d68c00d97ccd2480a2320ed652b92dec63102/docs/action_policy_libero_posttrain.md)。固定revision网页受HF访问条件限制，本次以现有本地 README L12–31及 download_complete.json 的revision记录核验；公开网页可读内容一致。

目录同时存在 framework 合并权重索引 `model.safetensors.index.json`（1165 keys）和 Diffusers transformer 索引（814 keys）；两者的 action head entries 指向同一已下载的 transformer shards。因此无需再下载另一份庞大权重才能尝试官方 loader。官方 server 明确支持 DCP 和本地HF/safetensors目录，但是否无需环境修复便可加载现有目录、两索引的全部映射是否逐数值正确，尚未实测。[server 支持说明 L13–32](https://github.com/NVIDIA/cosmos-framework/blob/cf5d68c00d97ccd2480a2320ed652b92dec63102/cosmos_framework/scripts/action_policy_server_libero.py#L13-L32)。本审计未读取凭证文件、连接任何外部存储或验证私有训练实验。

## 最小数值等价验收

1. **文字（已完成）**：同一非空JSON、同一checkpoint tokenizer；比较官方formatter/tokenize/packer与Diffusers完整input_ids、UND长度和尾标。JSON exact；当前140与官方121的差异仅额外19个system tokens。
2. **像素（已完成）**：同center PNG，比较canvas、content、uint8/float映射后的全部像素；shape metadata exact，像素误差如上。下一步用当前已加载VAE，对官方uint8预处理结果与当前float结果分别编码，报告clean首latent绝对误差/相对RMS；再用同一像素检查1帧与17帧编码的首latent。
3. **采样器（部分已完成）**：显式比较所有sigmas和整数timesteps，30步dummy flow查到两处t差1。将时间表统一后，使用同一小tensor和每步预先固定的velocity，比较官方合并state与当前模态独立scheduler，预期逐步exact或FP32舍入级误差。
4. **模型首步（尚未完成）**：对同一现有checkpoint，先统一无system、像素/VAE latent、时间表、domain、mask和RoPE；注入同一FP32未来vision/action noise。先逐字段比pack metadata，再只做一个t999 forward，比较各模态velocity及一轮scheduler输出。官方模型可顺序加载同一现有权重目录，避免同时驻留两份模型。若尚不能运行完整官方loader，先在当前已加载Diffusers模型里只更换官方formatter、uint8预处理和native时间表，做分项对照；这能够测试这些差异的行为影响，但不能冒称完整两框架等价。
5. **已有结果解释**：在上述最小输入差异澄清前，保留原实验结果为当前固定runner的条件敏感性证据。勿把语言不敏感、所有UND移除或某层patch行为直接解释为官方训练模型的完整实现结论；对照若改变结果，应按真实变化更新结论。

本次仅新增本审计文档；没有新临时文件或待删除产物，没有修改库、提交、推送或部署。实际GPU对照由主任务另行记录，本文件没有假定其结果。

## 后续实际核验（与上面的只读审计分开记录）

2026-10-03主任务完成[实验09](../experiments/09-system-prompt/README.md)：36组真实动作预测和5条新增执行，关闭额外system后仍抓错。另完成[实验11](../experiments/11-official-transformer/README.md)：同一模型输入交给官方原生网络，814键匹配、动作输入边界精确一致；最终readout相对RMS差0.858%，同一个官方head重放的velocity前10维差0.325%。两边PyTorch/注意力后端不同，仅首步，不是完整运行方式等价。这些结果替代上文“模型首步尚未完成”的进度状态；像素/VAE差异和完整采样的行为影响仍未排除。
