"""Causal UND K/V exchange through native GEN attention, preserving residuals.

The principal comparisons use one changed close/open token, identical scene,
positions, noise and first sigma. Larger gains are artificial mechanism probes.
"""
import importlib.util
import json
import os
import time
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import Cosmos3OmniPipeline
from diffusers.pipelines.cosmos import pipeline_cosmos3_omni as cm

ROOT=Path('/home/current/work/cosmos3')
DATA=ROOT/'outputs/official-demos/milk-posttrain/gripper-context'
OUT=DATA/'deep-dive/language-route'
OUT.mkdir(exist_ok=False)
start=time.perf_counter()
torch.set_grad_enabled(False)
cm._EMBODIMENT_TO_RAW_ACTION_DIM['libero']=10
pipe=Cosmos3OmniPipeline.from_pretrained(
    str(ROOT/'checkpoints/Cosmos3-Nano-Policy-LIBEROall-5k'),torch_dtype=torch.bfloat16,
    sound_tokenizer=None,enable_safety_checker=False,local_files_only=True)
pipe.to('cuda');pipe.vae.enable_tiling()
model=pipe.transformer
source=torch.load(DATA/'deep-dive/causal-trace/first_step_states.pt',map_location='cuda',weights_only=True)
inputs={'close':source['inputs']['failure'],'open':source['inputs']['open']}
del source
sigma=json.loads((DATA/'deep-dive/causal-trace/result.json').read_text())['sigma']
assert torch.equal(inputs['close']['vision_tokens'][0],inputs['open']['vision_tokens'][0])
assert torch.equal(inputs['close']['action_tokens'][0],inputs['open']['action_tokens'][0])
assert torch.equal(inputs['close']['position_ids'],inputs['open']['position_ids'])
assert torch.nonzero(inputs['close']['input_ids']!=inputs['open']['input_ids']).flatten().tolist()==[64]
records={};active=None;buffer={}

def status(stage,**kwargs):
    value=dict(stage=stage,elapsed_s=time.perf_counter()-start,**kwargs)
    (OUT/'status.json').write_text(json.dumps(value));print(json.dumps(value),flush=True)

def metric(output,case):
    velocity=output[2][0].float()
    clean=inputs[case]['action_tokens'][0].float()-sigma*velocity
    return dict(first_gripper_mean=float(clean[:,9].mean()),
        first_gripper_close_steps=int((clean[:,9]<0).sum()),
        clean=clean[:,:10].cpu().numpy(),velocity=velocity.cpu().numpy())

def rotate(t,cos,sin):
    half=t.shape[-1]//2
    return t*cos[:,None]+torch.cat([-t[...,half:],t[...,:half]],-1)*sin[:,None]

def before_attention(layer):
    def hook(module,args):
        if active is not None:buffer[layer]={'rotary':args[2]}
    return hook
def projection(layer,key):
    def hook(module,args,output):
        if active is not None:buffer[layer][key]=output.detach()
    return hook
def after_attention(layer):
    def hook(module,args,output):
        if active is None:return
        rec=records[active];kwargs=inputs[active]
        idx=kwargs['action_sequence_indexes'].flatten()-kwargs['und_len']
        b=buffer.pop(layer);rec['k'].append(b['raw_k'].clone());rec['v'].append(b['raw_v'].clone())
        cu,su,cg,sg=b['rotary']
        q=rotate(b['q'],cg,sg)[idx]
        ku=rotate(b['ku'],cu,su);kg=rotate(b['kg'],cg,sg)
        count=module.num_attention_heads//module.num_key_value_heads
        key=torch.cat([ku,kg],0).repeat_interleave(count,1)
        prob=torch.softmax(torch.einsum('qhd,khd->hqk',q.float(),key.float())/(module.head_dim**.5),-1)
        und_len=kwargs['und_len']
        rec['text_mass'].append(float(prob[:,:,:und_len].sum(-1).mean()))
        rec['word_mass'].append(float(prob[:,:,64].mean()) if active in ['close','open'] else float('nan'))
        vu=b['raw_v'].view(-1,module.num_key_value_heads,module.head_dim)
        vg=b['raw_vg'].view(-1,module.num_key_value_heads,module.head_dim)
        values=torch.cat([vu,vg],0).repeat_interleave(count,1)
        raw=torch.einsum('hqk,khd->qhd',prob,values.float()).flatten(-2)
        reconstructed=module.to_add_out(raw.to(values.dtype)).float()
        actual=output[1][idx].float()
        error=float((reconstructed-actual).square().mean().sqrt()/actual.square().mean().sqrt())
        assert error<.03,(active,layer,error)
        rec['reconstruction_error'].append(error)
        text_raw=torch.einsum('hqk,khd->qhd',prob[:,:,:und_len],values[:und_len].float()).flatten(-2)
        text_update=F.linear(text_raw.to(values.dtype),module.to_add_out.weight,bias=None)
        rec['text_update'].append(text_update.clone())
        rec['attention'].append(output[1][idx].clone())
    return hook
def after_mlp(layer):
    def hook(module,args,output):
        if active is not None:
            idx=inputs[active]['action_sequence_indexes'].flatten()-inputs[active]['und_len']
            records[active]['mlp'].append(output[idx].clone())
    return hook
def before_first(module,args):
    if active is not None:
        idx=inputs[active]['action_sequence_indexes'].flatten()-inputs[active]['und_len']
        records[active]['hidden'].append(args[1][idx].clone())
def after_layer(layer):
    def hook(module,args,output):
        if active is not None:
            idx=inputs[active]['action_sequence_indexes'].flatten()-inputs[active]['und_len']
            records[active]['hidden'].append(output[1][idx].clone())
    return hook
handles=[model.layers[0].register_forward_pre_hook(before_first)]
for l,layer in enumerate(model.layers):
    attn=layer.self_attn
    handles.append(attn.register_forward_pre_hook(before_attention(l)))
    for key,name in [('raw_k','to_k'),('raw_v','to_v'),('raw_vg','add_v_proj'),
                     ('q','norm_added_q'),('kg','norm_added_k'),
                     ('ku','k_norm_und_for_gen' if attn.k_norm_und_for_gen is not None else 'norm_k')]:
        handles.append(getattr(attn,name).register_forward_hook(projection(l,key)))
    handles.append(attn.register_forward_hook(after_attention(l)))
    handles.append(layer.mlp_moe_gen.register_forward_hook(after_mlp(l)))
    handles.append(layer.register_forward_hook(after_layer(l)))

def baseline(case):
    global active
    records[case]={key:[] for key in ['k','v','hidden','attention','mlp','text_update',
        'text_mass','word_mass','reconstruction_error']}
    active=case
    try:output=model(**inputs[case])
    finally:active=None
    result=metric(output,case)
    assert len(records[case]['hidden'])==37
    status('baseline',case=case,first_gripper_mean=result['first_gripper_mean'])
    return result
baselines={case:baseline(case) for case in ['close','open']}
for case,folder in [('close','free_space_same_seed'),('open','free_space_half_open_instruction')]:
    old=torch.load(DATA/'deep-dive'/folder/'no_demonstrations/internal_states.pt',
        weights_only=True,mmap=True,map_location='cpu')
    assert np.array_equal(baselines[case]['velocity'],old['query_action_velocity'][0].float().numpy())
checks=[dict(test='saved_first_velocity',max_error=0.)]

def exchange(recipient,donor,layers,which='KV',gain=1.,target='all_gen',random=False):
    swap_handles=[]
    idx=inputs[recipient]['action_sequence_indexes'].flatten()-inputs[recipient]['und_len']
    for l in layers:
        attn=model.layers[l].self_attn
        def hook(module,args,output,l=l):
            replacements=[]
            if which in ['K','KV']:
                replacements.append(module.to_k.register_forward_hook(
                    lambda m,a,o:records[donor]['k'][l]))
            if which in ['V','KV','zero_value']:
                replacements.append(module.to_v.register_forward_hook(
                    lambda m,a,o:torch.zeros_like(o) if which=='zero_value' else records[donor]['v'][l]))
            try:
                _,alternative=module.processor(module,*args)
            finally:
                for h in replacements:h.remove()
            native=output[1]
            if random:
                delta=alternative.float()-native.float()
                if target=='query_action':
                    delta=delta[idx]
                noise=torch.randn(delta.shape,device='cuda',dtype=torch.float32,
                    generator=torch.Generator(device='cuda').manual_seed(6050+l))
                noise*=delta.norm()/noise.norm()
                changed=(native[idx].float()+gain*noise).to(native.dtype) if target=='query_action' else (native.float()+gain*noise).to(native.dtype)
            else:
                changed=alternative if gain==1 else native if gain==0 else (native.float()+gain*(alternative.float()-native.float())).to(native.dtype)
                if target=='query_action':changed=changed[idx]
            if target=='query_action':
                modified=native.clone();modified.index_copy_(0,idx,changed)
            else:modified=changed
            return output[0],modified
        swap_handles.append(attn.register_forward_hook(hook))
    try:output=model(**inputs[recipient])
    finally:
        for h in swap_handles:h.remove()
    return metric(output,recipient)

for test,recipient,donor,gain in [('self','open','open',1.),('zero_gain','open','close',0.)]:
    r=exchange(recipient,donor,range(36),gain=gain)
    error=float(np.max(np.abs(r['velocity']-baselines[recipient]['velocity'])))
    assert error==0,(test,error)
    checks.append(dict(test=test,max_error=error))
all_swap=exchange('open','close',range(36))
error=float(np.max(np.abs(all_swap['velocity']-baselines['close']['velocity'])))
assert error==0,error
checks.append(dict(test='all_36_GEN_reads_close_KV',max_error=error))
status('controls_verified',all_KV_max_error=error)
arrays=dict(layers=np.arange(1,37))
single_gripper=np.zeros((3,36));single_rmse=np.zeros((3,36))
for i,which in enumerate(['K','V','KV']):
    for l in range(36):
        r=exchange('open','close',[l],which=which)
        single_gripper[i,l]=r['first_gripper_mean']-baselines['open']['first_gripper_mean']
        single_rmse[i,l]=np.sqrt(np.mean((r['clean']-baselines['open']['clean'])**2))
    status('single_path_complete',which=which,max_gripper_change=float(np.max(np.abs(single_gripper[i]))))
windows=[list(range(i,i+6)) for i in range(0,36,6)]+[list(range(36))]
labels=[f'{x[0]+1}–{x[-1]+1}' for x in windows]
gains=[0.,1.,2.,4.,8.,16.]
window_mean=np.zeros((2,7,6));window_rmse=np.zeros((2,7,6))
for t,target in enumerate(['all_gen','query_action']):
    for w,window in enumerate(windows):
        for g,gain in enumerate(gains):
            r=exchange('open','close',window,gain=gain,target=target)
            window_mean[t,w,g]=r['first_gripper_mean']
            window_rmse[t,w,g]=np.sqrt(np.mean((r['clean']-baselines['open']['clean'])**2))
    status('window_complete',target=target,min_gripper_mean=float(window_mean[t].min()))
for gain in [1.,4.,16.]:
    r=exchange('close','open',range(36),gain=gain)
    checks.append(dict(test='reverse_all_layers',gain=gain,first_gripper_mean=r['first_gripper_mean']))
    r=exchange('open','close',range(12,18),gain=gain,random=True)
    checks.append(dict(test='random_direction_same_update_norm',layers='13–18',gain=gain,
        first_gripper_mean=r['first_gripper_mean']))
for window in [list(range(36)),list(range(12,18))]:
    r=exchange('open','open',window,which='zero_value')
    checks.append(dict(test='zero_text_values_keep_text_keys',layers=[l+1 for l in window],
        first_gripper_mean=r['first_gripper_mean']))

def rms(x):return x.float().square().mean((-1,-2)).sqrt().cpu().numpy()
hc=torch.stack(records['close']['hidden']);ho=torch.stack(records['open']['hidden'])
ac=torch.stack(records['close']['attention']);ao=torch.stack(records['open']['attention'])
mc=torch.stack(records['close']['mlp']);mo=torch.stack(records['open']['mlp'])
delta_attn=ac.float()-ao.float();delta_mlp=mc.float()-mo.float()
cosine=(delta_attn.flatten(1)*delta_mlp.flatten(1)).sum(1)/(delta_attn.flatten(1).norm(dim=1)*delta_mlp.flatten(1).norm(dim=1)).clamp_min(1e-20)
arrays.update(single_gripper_change=single_gripper,single_action_rmse=single_rmse,
    window_labels=np.asarray(labels),gains=np.asarray(gains),window_gripper_mean=window_mean,
    window_action_rmse=window_rmse,
    text_attention_mass=np.asarray([records[x]['text_mass'] for x in ['close','open']]),
    word_attention_mass=np.asarray([records[x]['word_mass'] for x in ['close','open']]),
    delta_input_rms=rms(hc[:-1].float()-ho[:-1].float()),delta_attention_rms=rms(delta_attn),
    delta_mlp_rms=rms(delta_mlp),delta_output_rms=rms(hc[1:].float()-ho[1:].float()),
    attention_mlp_delta_cosine=cosine.cpu().numpy(),
    delta_text_contribution_rms=rms(torch.stack(records['close']['text_update']).float()-torch.stack(records['open']['text_update']).float()))

# The task-style comparison changes the target noun on an existing initial image.
# It is a compositional diagnostic, not a guarantee of in-distribution success.
spec=importlib.util.spec_from_file_location('language_native_flow',ROOT/'cosmos-framework/cosmos_framework/model/generator/diffusion/samplers/fm_solvers_unipc.py')
sampler=importlib.util.module_from_spec(spec);spec.loader.exec_module(sampler)
pipe.register_to_config(use_native_flow_schedule=True)
original_builder=pipe._build_action_json_prompt
def caption(*args,**kwargs):
    value=json.loads(original_builder(*args,**kwargs));value['actions'][0]['idle_frame']='0 out of 16.'
    return json.dumps(value)
pipe._build_action_json_prompt=caption
class StopFirstInput(Exception):pass
task_image=DATA.parent/'fwd-liberoall-5k/seed-195-put-in-control/input_00.png'
def task_run(case,prompt):
    pipe.scheduler=sampler.FlowUniPCMultistepScheduler(num_train_timesteps=1000,shift=1.)
    captured={}
    def intercept(module,args,kwargs):
        captured.update(kwargs);raise StopFirstInput()
    h=model.register_forward_pre_hook(intercept,with_kwargs=True)
    condition=cm.CosmosActionCondition(mode='policy',chunk_size=16,domain_name='libero',
        resolution_tier=256,image=Image.open(task_image).convert('RGB'),view_point='concat_view')
    try:
        pipe(prompt=prompt,action=condition,fps=20,num_inference_steps=30,guidance_scale=1.,
            generator=torch.Generator(device='cuda').manual_seed(198),output_type='latent')
    except StopFirstInput:pass
    finally:h.remove()
    inputs[case]=captured;first=baseline(case)
    pipe.scheduler=sampler.FlowUniPCMultistepScheduler(num_train_timesteps=1000,shift=1.)
    result=pipe(prompt=prompt,action=condition,fps=20,num_inference_steps=30,guidance_scale=1.,
        generator=torch.Generator(device='cuda').manual_seed(198),output_type='latent')
    arrays[case+'_full_actions']=result.action[0].float().numpy()
    del result
    return first
task_prompts=dict(task_milk='Pick the milk and place it in the basket',task_butter='Pick the butter and place it in the basket')
for case,prompt in task_prompts.items():baselines[case]=task_run(case,prompt)
aa,bb=inputs['task_milk'],inputs['task_butter']
assert aa['input_ids'].shape==bb['input_ids'].shape
changed=torch.nonzero(aa['input_ids']!=bb['input_ids']).flatten().tolist()
assert len(changed)==1,changed
for key in ['position_ids','vision_tokens','action_tokens']:
    left,right=aa[key],bb[key]
    assert torch.equal(left[0],right[0]) if isinstance(left,list) else torch.equal(left,right)
r=exchange('task_milk','task_butter',range(36))
error=float(np.max(np.abs(r['velocity']-baselines['task_butter']['velocity'])))
assert error==0,error
checks.append(dict(test='task_style_all_36_KV_exchange',max_error=error))
task_delta=arrays['task_milk_full_actions']-arrays['task_butter_full_actions']
task_control=dict(prompts=task_prompts,input_image=str(task_image),seed=198,
    changed_token_indexes=changed,same_images_positions_noise=True,
    first_clean_all10_rmse=float(np.sqrt(np.mean((baselines['task_milk']['clean']-baselines['task_butter']['clean'])**2))),
    full30_all10_rmse=float(np.sqrt(np.mean(task_delta**2))),
    full30_xyz_rmse=float(np.sqrt(np.mean(task_delta[:,:3]**2))),
    full30_gripper_mean_abs_difference=float(np.abs(task_delta[:,9]).mean()),
    scope='Existing milk task initial image with two official task-style descriptions. Same-image noun swap is a compositional control; the butter caption with this milk scene is not proven to be in-distribution. No new physical rollouts.')
for h in handles:h.remove()
summary=dict(baselines={case:{k:v for k,v in value.items() if k not in ['clean','velocity']} for case,value in baselines.items()},
    checks=checks,task_style_control=task_control,sigma=sigma,seed=198,
    first_step_reconstruction_max_relative_error=max(max(records[x]['reconstruction_error']) for x in records),
    max_single_layer_gripper_change=float(np.max(np.abs(single_gripper))),
    min_amplified_window_gripper_mean=float(window_mean.min()),
    scope='Same-image close/open differs by one actual token with same positions/noise. Native GEN attention rereads donor text K/V; original UND attention and residual paths are preserved. Primary interventions are first-step only. Gains above 1 and zero text values are artificial mechanism probes, not repaired weights or executed robot actions. Task-style full predictions are offline compositional controls.',
    elapsed_s=time.perf_counter()-start)
np.savez_compressed(OUT/'arrays.npz',**arrays)
(OUT/'result.json').write_text(json.dumps(summary,indent=2))
torch.save({case:{k:torch.stack(v).cpu() if k in ['hidden','attention','mlp','text_update','k','v'] else v
    for k,v in record.items()} for case,record in records.items()},OUT/'attention_states.pt')
(OUT/'complete.json').write_text(json.dumps(dict(state='complete',elapsed_s=time.perf_counter()-start),indent=2))
status('complete',**{k:v for k,v in summary.items() if k in ['max_single_layer_gripper_change','min_amplified_window_gripper_mean','task_style_control']})
