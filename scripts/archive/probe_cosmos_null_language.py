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
OUT = BASE/'position-controls' if POSITION_CONTROLS else BASE
if POSITION_CONTROLS: OUT.mkdir(exist_ok=False)
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

if POSITION_CONTROLS:
    def saved(case):
        return torch.load(BASE/'cases'/case/'states.pt',map_location='cpu',weights_only=True,mmap=True)
    baseline_result = json.loads((BASE/'result.json').read_text())
    definitions = {item['case']:item for item in baseline_result['cases']}
    tests = []
    for scene in ['A','B']:
        tests.extend([(scene,198,'empty','close'),(scene,198,'close','empty')])
    for seed in [195,198]:
        tests.extend([('goal',seed,'stove','drawer'),('goal',seed,'drawer','stove')])
    comparisons = []
    for scene,seed,condition,align_condition in tests:
        case = f'{scene}_{seed}_{condition}_native'
        donor_case = f'{scene}_{seed}_{align_condition}_native'
        original,donor = saved(case),saved(donor_case)
        definition = definitions[case]
        run(scene,seed,condition,'native',prompt=definition['prompt'],image=Path(definition['image']),align_to=donor['input'])
        rec = traces[case]
        inp = rec['input'];ref = original['input'];di = donor['input']
        assert torch.equal(inp['input_ids'],ref['input_ids'])
        assert torch.equal(inp['action_tokens'][0],ref['action_tokens'][0])
        assert torch.equal(inp['vision_tokens'][0],ref['vision_tokens'][0])
        assert torch.equal(inp['position_ids'][...,:inp['und_len']],ref['position_ids'][...,:ref['und_len']])
        assert torch.equal(inp['position_ids'][...,inp['und_len']:],di['position_ids'][...,di['und_len']:])
        item = dict(case=case,align_GEN_positions_to=donor_case,same_original_input_ids_image_noise=True,
            UND_positions_preserved=True,GEN_positions_exactly_match_donor=True)
        for label,other in [('original',original),('other_command',donor)]:
            delta = rec['output_actions'].float()-other['output_actions'].float()
            item[label+'_action_rmse'] = float(delta.square().mean().sqrt())
            item[label+'_xyz_rmse'] = float(delta[:,:3].square().mean().sqrt())
            item[label+'_gripper_mae'] = float(delta[:,9].abs().mean())
        comparisons.append(item)
    for handle in handles: handle.remove()
    summary = dict(cases=cases,comparisons=comparisons,
        scope='Change only GEN rotary position IDs to the other condition; UND positions, token IDs, images and initial action/vision noise stay exactly unchanged. Frozen full30 offline predictions. Goal instructions still differ in token count, which can influence text attention normalization; this control isolates GEN positional drift, not every conceivable language-format effect.',
        elapsed_s=time.perf_counter()-started)
    np.savez_compressed(OUT/'arrays.npz',**arrays)
    (OUT/'result.json').write_text(json.dumps(summary,indent=2))
    (OUT/'complete.json').write_text(json.dumps(dict(state='complete',elapsed_s=summary['elapsed_s']),indent=2))
    status('complete',case_count=len(cases))
    sys.exit(0)

for scene in ['A','B']:
    for seed in [195,196,198]:
        for condition in ['close','open','empty']:
            run(scene,seed,condition,'native')
        for condition in ['close','open']:
            run(scene,seed,condition,'no_und')

# Verify native attention is reproduced by the same recomputation code, then
# verify all-layer removal erases the sole changed close/open token's influence.
def maxdiff(a,b): return float((a.float()-b.float()).abs().max())
source = torch.load(DATA/'deep-dive/causal-trace/first_step_states.pt',map_location='cpu',weights_only=True,mmap=True)
actual = traces['A_198_close_native']['velocity'][0]
old = torch.load(DATA/'deep-dive/free_space_same_seed/no_demonstrations/internal_states.pt',
    map_location='cpu',weights_only=True,mmap=True)['query_action_velocity'][0]
assert torch.equal(actual,old)
checks.append(dict(test='old_first_velocity_reproduced',max_error=0.))
native_input = {k: v.to('cuda') if isinstance(v,torch.Tensor) else
    [x.to('cuda') if isinstance(x,torch.Tensor) else x for x in v] if isinstance(v,list) else v
    for k,v in traces['A_198_close_native']['input'].items()}
mode = 'recompute_full'
try: repeat = model(**native_input)[2][0].cpu()
finally: mode = 'native'
error = maxdiff(repeat,actual); assert error == 0,error
checks.append(dict(test='native_full_attention_recompute',max_error=error))
del source,old,native_input,repeat

for scene in ['A','B']:
    for seed in [195,196,198]:
        a,b = [traces[f'{scene}_{seed}_{c}_no_und'] for c in ['close','open']]
        action_error = maxdiff(a['output_actions'],b['output_actions'])
        vision_error = maxdiff(a['output_vision_latents'],b['output_vision_latents'])
        assert action_error == vision_error == 0,(scene,seed,action_error,vision_error)
        checks.append(dict(test='hard_no_und_close_open_exact_equality',scene=scene,seed=seed,
            action_max_error=action_error,vision_latent_max_error=vision_error))

# Native ambiguous-scene tasks: use exactly one actual rendered image for both
# task captions, rather than assuming flattened MuJoCo states imply identical scenes.
goal_metadata = json.loads((OUT/'goal-control/metadata.json').read_text())
goal_image = Path(goal_metadata['tasks'][0]['image_path'])
for seed in [195,198]:
    for condition,task in zip(['drawer','stove'],goal_metadata['tasks']):
        run('goal',seed,condition,'native',prompt=task['benchmark_language'],image=goal_image)
    run('goal',seed,'empty','native',prompt='',image=goal_image)

weight = model.action_proj_out.fc.weight[5].detach().view(4096,64).float().cpu()
bias = model.action_proj_out.bias.weight[5].detach().float().cpu()
head = model.action_proj_out
for case in traces:
    rec = traces[case]
    for s in [0,29]:
        replay = head(rec['readout'][s].to('cuda'),rec['domain_ids'].to('cuda')).cpu()
        error = maxdiff(replay,rec['velocity'][s]); assert error == 0,(case,s,error)
checks.append(dict(test='actual_action_head_replay_all_cases_first_last',max_error=0.))
np.savez_compressed(OUT/'action_head.npz',weight=weight.numpy(),bias=bias.numpy())

def rms(value): return float(value.float().square().mean().sqrt())
def compare(left,right,kind):
    a,b = traces[left],traces[right]
    d = a['output_actions']-b['output_actions']
    item = dict(left=left,right=right,kind=kind,scene=left.split('_')[0],seed=int(left.split('_')[1]),
        action_rmse=rms(d),xyz_rmse=rms(d[:,:3]),gripper_mae=float(d[:,9].abs().mean()))
    for label,s in [('first',0),('late',29)]:
        h = a['readout'][s].float()-b['readout'][s].float()
        item[label+'_readout_relative_rms'] = rms(h)/max(rms(a['readout'][s]),1e-20)
        projection = h@weight[:,9]
        item[label+'_gripper_head_delta_rms'] = rms(projection)
        item[label+'_actual_gripper_velocity_delta_rms'] = rms(a['velocity'][s,:,9]-b['velocity'][s,:,9])
        item[label+'_head_projection_cosine_abs_mean'] = float((projection.abs()/
            (h.norm(dim=-1)*weight[:,9].norm()).clamp_min(1e-20)).mean())
    return item

pairs = []
noise_controls = []
for scene in ['A','B']:
    for seed in [195,196,198]:
        for right,kind in [('open_native','command'),('empty_native','empty'),
                           ('close_no_und','hard_cut'),('open_no_und','no_und_equivalence')]:
            left = f'{scene}_{seed}_close_no_und' if kind=='no_und_equivalence' else f'{scene}_{seed}_close_native'
            pairs.append(compare(left,f'{scene}_{seed}_{right}',kind))
        a,b = [traces[f'{scene}_{seed}_{condition}_native']['input'] for condition in ['close','open']]
        assert torch.equal(a['position_ids'],b['position_ids'])
        assert torch.equal(a['action_tokens'][0],b['action_tokens'][0])
        assert torch.equal(a['vision_tokens'][0],b['vision_tokens'][0])
        changed = torch.nonzero(a['input_ids']!=b['input_ids']).flatten().tolist()
        assert changed == [64],changed
    for prompt in ['close','open','empty']:
        for seed_pair in [(195,196),(195,198),(196,198)]:
            d = traces[f'{scene}_{seed_pair[0]}_{prompt}_native']['output_actions']-traces[f'{scene}_{seed_pair[1]}_{prompt}_native']['output_actions']
            noise_controls.append(dict(scene=scene,prompt=prompt,seed_pair=list(seed_pair),
                action_rmse=rms(d),gripper_mae=float(d[:,9].abs().mean())))
for seed in [195,198]:
    pairs.append(compare(f'goal_{seed}_drawer_native',f'goal_{seed}_stove_native','command'))
    pairs.append(compare(f'goal_{seed}_drawer_native',f'goal_{seed}_empty_native','empty'))

a,b = traces['A_198_close_native'],traces['A_198_open_native']
hd = a['hidden'].float()-b['hidden'].float()
arrays['language_hidden_relative_rms'] = (hd.square().mean((-1,-2)).sqrt()/
    a['hidden'].float().square().mean((-1,-2)).sqrt().clamp_min(1e-20)).numpy()
rh = a['readout'][[0,29]].float()-b['readout'][[0,29]].float()
projection = rh@weight[:,9]
arrays['language_head_cosine'] = (projection/(rh.norm(dim=-1)*weight[:,9].norm()).clamp_min(1e-20)).numpy()
arrays['language_head_projection'] = projection.numpy()
arrays['language_actual_velocity_delta'] = (a['velocity'][[0,29],:,9]-b['velocity'][[0,29],:,9]).float().numpy()
for h in handles: h.remove()
summary = dict(cases=cases,pairs=pairs,noise_controls=noise_controls,checks=checks,
    seeds=[195,196,198],inference_steps=30,checkpoint_revision='64798337c642c53f9e22332554ae92f76a86cf04',
    head_gripper_bias=float(bias[9]),head_gripper_weight_norm=float(weight[:,9].norm()),
    goal_control=dict(metadata_path=str(OUT/'goal-control/metadata.json'),same_exact_image_used_for_both_tasks=True,
        collector_render_equality=goal_metadata['pixel_checks']['concat_equal']),
    scope='Frozen offline 16-action predictions. Native close/open differs by one token with matched images, positions and action/future-vision noise. Empty description changes token count and may change rotary positions. Hard no_und removes all UND keys/values from every GEN query at all 36 layers and all 30 denoising rounds, including JSON metadata; this is an artificial causal intervention, not a repaired policy. Three seeds are descriptive repeats, not a statistical population. Goal control uses one exact actual rendered image for both benchmark task captions. No new physical success claim.',
    elapsed_s=time.perf_counter()-started)
np.savez_compressed(OUT/'arrays.npz',**arrays)
(OUT/'result.json').write_text(json.dumps(summary,indent=2))
(OUT/'complete.json').write_text(json.dumps(dict(state='complete',elapsed_s=summary['elapsed_s']),indent=2))
status('complete',case_count=len(cases),check_count=len(checks))
