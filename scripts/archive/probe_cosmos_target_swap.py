"""Matched-noise instruction controls and a hard UND-to-GEN causal ablation.

Frozen offline action predictions; the ablation removes all UND metadata too.
Raw first/last hidden states and actual action-head inputs remain inspectable.
"""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
import numpy as np
import torch
from PIL import Image
from diffusers import Cosmos3OmniPipeline
from diffusers.pipelines.cosmos import pipeline_cosmos3_omni as cm
from diffusers.models.transformers.transformer_cosmos3 import dispatch_attention_fn, _rotate_half

ROOT = Path('/home/current/work/cosmos3')
DATA = ROOT/'outputs/official-demos/milk-posttrain/gripper-context'
BASE = DATA/'deep-dive/null-language'
POSITION_CONTROLS = '--position-controls' in sys.argv
OUT = DATA/'deep-dive/target-swap'
assert (OUT/'inputs/metadata.json').exists()
(OUT/'cases').mkdir(exist_ok=False)
started = time.perf_counter()
torch.set_grad_enabled(False)
cm._EMBODIMENT_TO_RAW_ACTION_DIM['libero'] = 10
pipe = Cosmos3OmniPipeline.from_pretrained(
    str(ROOT/'checkpoints/Cosmos3-Nano-Policy-LIBEROall-5k'), torch_dtype=torch.bfloat16,
    sound_tokenizer=None, enable_safety_checker=False, local_files_only=True)
pipe.to('cuda'); pipe.vae.enable_tiling()
model = pipe.transformer
original_builder = pipe._build_action_json_prompt
def caption_builder(*args, **kwargs):
    value = json.loads(original_builder(*args, **kwargs))
    value['actions'][0]['idle_frame'] = '0 out of 16.'
    return json.dumps(value)
pipe._build_action_json_prompt = caption_builder
spec = importlib.util.spec_from_file_location('null_native_flow',
    ROOT/'cosmos-framework/cosmos_framework/model/generator/diffusion/samplers/fm_solvers_unipc.py')
sampler = importlib.util.module_from_spec(spec); spec.loader.exec_module(sampler)
pipe.register_to_config(use_native_flow_schedule=True)
active = None
mode = 'native'
step = -1
query_idx = None
position_template = None
traces = {}
arrays = {}
cases = []
checks = []

def status(stage, **kwargs):
    value = dict(stage=stage, elapsed_s=time.perf_counter()-started, **kwargs)
    (OUT/'status.json').write_text(json.dumps(value)); print(json.dumps(value), flush=True)

def cpu(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, list): return [cpu(x) for x in value]
    if isinstance(value, tuple): return tuple(cpu(x) for x in value)
    if isinstance(value, dict): return {k:cpu(v) for k,v in value.items()}
    return value

def begin_forward(module, args, kwargs):
    global step, query_idx
    if active is None: return
    step += 1
    query_idx = kwargs['action_sequence_indexes'].flatten()-kwargs['und_len']
    rec = traces[active]
    if position_template is not None:
        if step == 0: rec['original_position_ids'] = cpu(kwargs['position_ids'])
        positions = kwargs['position_ids'].clone()
        und_len = kwargs['und_len']
        assert positions[...,und_len:].shape == position_template.shape
        positions[...,und_len:] = position_template
        kwargs['position_ids'] = positions
    rec['pre_step'].append(cpu(kwargs['action_tokens'][0]))
    if step == 0:
        rec['input'] = cpu(kwargs)
    if step in [0,29]: rec['hidden'][step] = []
    return args,kwargs

def first_layer(module, args):
    if active is not None and step in [0,29]:
        traces[active]['hidden'][step].append(cpu(args[1][query_idx]))

def after_layer(module, args, output):
    if active is not None and step in [0,29]:
        traces[active]['hidden'][step].append(cpu(output[1][query_idx]))

def before_head(module, args):
    if active is None: return
    rec = traces[active]
    rec['readout'].append(cpu(args[0]))
    if step == 0: rec['domain_ids'] = cpu(args[1])

def after_model(module, args, output):
    if active is not None: traces[active]['velocity'].append(cpu(output[2][0]))

def attention_intervention(module, args, output):
    if mode == 'native': return
    und, gen, (cu,su,cg,sg) = args
    q = module.norm_added_q(module.add_q_proj(gen).view(-1,module.num_attention_heads,module.head_dim))
    k = module.norm_added_k(module.add_k_proj(gen).view(-1,module.num_key_value_heads,module.head_dim))
    v = module.add_v_proj(gen).view(-1,module.num_key_value_heads,module.head_dim)
    q = q*cg[:,None]+_rotate_half(q)*sg[:,None]
    k = k*cg[:,None]+_rotate_half(k)*sg[:,None]
    if mode == 'recompute_full':
        ku = module.norm_k(module.to_k(und).view(-1,module.num_key_value_heads,module.head_dim))
        if module.k_norm_und_for_gen is not None: ku = module.k_norm_und_for_gen(ku)
        ku = ku*cu[:,None]+_rotate_half(ku)*su[:,None]
        vu = module.to_v(und).view(-1,module.num_key_value_heads,module.head_dim)
        k = torch.cat([ku,k]); v = torch.cat([vu,v])
    else: assert mode == 'no_und'
    value = dispatch_attention_fn(q[None],k[None],v[None],is_causal=False,enable_gqa=True,
        backend=module.processor._attention_backend,parallel_config=module.processor._parallel_config)
    return output[0],module.to_add_out(value.squeeze(0).flatten(-2,-1))

handles = [model.register_forward_pre_hook(begin_forward,with_kwargs=True),
    model.layers[0].register_forward_pre_hook(first_layer),
    model.action_proj_out.register_forward_pre_hook(before_head),
    model.register_forward_hook(after_model)]
for layer in model.layers:
    handles.append(layer.self_attn.register_forward_hook(attention_intervention))
    handles.append(layer.register_forward_hook(after_layer))

prompts = dict(close='close the gripper while keeping the robot arm stationary',
    open='open the gripper while keeping the robot arm stationary',empty='')
images = dict(A=DATA/'input_00.png',
    B=DATA/'deep-dive/known_close_image_close_instruction/query.png')

def run(scene,seed,condition,run_mode,prompt=None,image=None,align_to=None):
    global active,mode,step,position_template
    case = f'{scene}_{seed}_{condition}_{run_mode}'
    prompt = prompts[condition] if prompt is None else prompt
    image = images[scene] if image is None else image
    destination = OUT/'cases'/case; destination.mkdir(exist_ok=False)
    traces[case] = dict(hidden={},readout=[],pre_step=[],velocity=[])
    position_template = None if align_to is None else align_to['position_ids'][...,align_to['und_len']:].to('cuda')
    active,mode,step = case,run_mode,-1
    pipe.scheduler = sampler.FlowUniPCMultistepScheduler(num_train_timesteps=1000,shift=1.)
    sigma = float(pipe.scheduler.sigmas[0]) if len(pipe.scheduler.sigmas) else None
    condition_spec = cm.CosmosActionCondition(mode='policy',chunk_size=16,domain_name='libero',
        resolution_tier=256,image=Image.open(image).convert('RGB'),view_point='concat_view')
    try:
        result = pipe(prompt=prompt,action=condition_spec,fps=20,num_inference_steps=30,
            guidance_scale=1.,generator=torch.Generator(device='cuda').manual_seed(seed),output_type='latent')
    finally: active,mode,position_template = None,'native',None
    rec = traces[case]
    assert len(rec['velocity']) == 30
    for key in ['readout','pre_step','velocity']: rec[key] = torch.stack(rec[key])
    rec['hidden'] = torch.stack([torch.stack(rec['hidden'][s]) for s in [0,29]])
    rec['sigmas'] = pipe.scheduler.sigmas.detach().cpu().clone()
    assert rec['hidden'].shape == (2,37,16,4096)
    assert rec['readout'].shape == (30,16,4096)
    prediction = result.action[0].float().cpu().numpy()
    rec['output_actions'] = torch.from_numpy(prediction)
    rec['output_vision_latents'] = cpu(result.video)
    rec['image_path'],rec['prompt'] = str(image),prompt
    torch.save(rec,destination/'states.pt')
    arrays[case+'_actions'] = prediction
    item = dict(case=case,scene=scene,seed=seed,condition=condition,mode=run_mode,prompt=prompt,
        gripper_mean=float(prediction[:,9].mean()),close_steps=int((prediction[:,9]<0).sum()),
        und_len=int(rec['input']['und_len']),image=str(image))
    (destination/'result.json').write_text(json.dumps(item,indent=2)); cases.append(item)
    status('case_complete',completed=len(cases),**item)
    return case


prompts.update(milk='pick up the milk and place it in the basket',butter='pick up the butter and place it in the basket')
images=dict(swapped=OUT/'inputs/swapped/input.png')
for condition in ['milk','butter']:run('swapped',198,condition,'native')
for h in handles:h.remove()
np.savez_compressed(OUT/'arrays.npz',**arrays)
(OUT/'result.json').write_text(json.dumps(dict(cases=cases,scope='Milk/butter XY swap; fixed weights, same seed 198'),indent=2))
(OUT/'complete.json').write_text(json.dumps(dict(state='complete',cases=len(cases))))
