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
OUT = DATA/'deep-dive/state-exchange'
OUT.mkdir(exist_ok=False)

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

mode='record';label=None;step=-1;indexes=None;selected=12;donor=None;cache={};first_inputs={};predictions={}
def cpu(v):
 if isinstance(v,torch.Tensor):return v.detach().cpu().clone()
 if isinstance(v,list):return [cpu(t) for t in v]
 if isinstance(v,tuple):return tuple(cpu(t) for t in v)
 if isinstance(v,dict):return {k:cpu(t) for k,t in v.items()}
 return v
def pre(module,args,kwargs):
 global step,indexes
 step+=1;indexes=kwargs['action_sequence_indexes'].flatten()-kwargs['und_len']
 if step==0:first_inputs[label]=cpu(kwargs)
def boundary(layer):
 def hook(module,args,out):
  if mode=='record':cache[label][layer].append(cpu(out[1][indexes]))
  elif layer==selected:
   u,g=out;g=g.clone();g[indexes]=cache[donor][layer][step].to(g.device);return u,g
 return hook
handles=[model.register_forward_pre_hook(pre,with_kwargs=True)]
for layer in [12,30]:handles.append(model.layers[layer-1].register_forward_hook(boundary(layer)))
def run(name,image,run_mode='record',from_case=None,layer=12):
 global mode,label,step,donor,selected
 mode,label,step,donor,selected=run_mode,name,-1,from_case,layer
 if mode=='record':cache[name]={12:[],30:[]}
 pipe.scheduler=sampler.FlowUniPCMultistepScheduler(num_train_timesteps=1000,shift=1.)
 condition=cm.CosmosActionCondition(mode='policy',chunk_size=16,domain_name='libero',resolution_tier=256,image=Image.open(image).convert('RGB'),view_point='concat_view')
 result=pipe(prompt='pick up the milk and place it in the basket',action=condition,fps=20,num_inference_steps=30,guidance_scale=1.,generator=torch.Generator(device='cuda').manual_seed(198),output_type='latent')
 assert step==29
 predictions[name]=result.action[0].float().cpu().numpy()
 if mode=='record':
  for l in cache[name]:cache[name][l]=torch.stack(cache[name][l])
 np.save(OUT/(name+'_actions.npy'),predictions[name]);print('COMPLETE',name,flush=True)
a_image=DATA/'deep-dive/object-position/closed-loop/center_milk/input_00.png';b_image=DATA/'deep-dive/target-swap/inputs/swapped/input.png'
run('A',a_image);run('B',b_image)
for name,path in [('A',DATA/'deep-dive/object-position/cases/center_198_milk_native/states.pt'),('B',DATA/'deep-dive/target-swap/cases/swapped_198_milk_native/states.pt')]:
 ref=torch.load(path,map_location='cpu',weights_only=True,mmap=True)['output_actions'].numpy();assert np.array_equal(predictions[name],ref),name
assert torch.equal(first_inputs['A']['input_ids'],first_inputs['B']['input_ids'])
assert torch.equal(first_inputs['A']['action_tokens'][0],first_inputs['B']['action_tokens'][0])
assert torch.equal(first_inputs['A']['vision_tokens'][0][:,:,1:],first_inputs['B']['vision_tokens'][0][:,:,1:])
run('B_self_12',b_image,'patch','B',12);assert np.array_equal(predictions['B_self_12'],predictions['B'])
run('B_from_A_12',b_image,'patch','A',12)
run('B_from_A_30',b_image,'patch','A',30)
for h in handles:h.remove()
rows=[]
for name,p in predictions.items():
 delta=p-predictions['B'];gap=predictions['A'][:,:3]-predictions['B'][:,:3]
 rows.append(dict(case=name,normalized_xyz_action_rmse_vs_B=float(np.sqrt(np.mean(delta[:,:3]**2))),xyz_change_projection_towards_A=float(np.sum(delta[:,:3]*gap)/np.sum(gap**2)),normalized_gripper_mean=float(p[:,9].mean())))
settings=dict(prompt='pick up the milk and place it in the basket',seed=198,patch_action_token_count=16,hidden_dim=4096,selected_boundaries=[12,30],patch_schedule='At selected layer after each of 30 denoising forwards in first query only; all later closed-loop queries are native.',donor='Original-layout correct-milk rollout, initial observation only',recipient='Milk/butter swapped-layout wrong-cheese rollout, initial observation',same_text_and_initial_action_future_vision_noise=True,baseline_predictions_reproduced_exactly=True,self_patch_max_action_error=0.,rows=rows,scope='Cross-scene action-state exchange; donor states contain spatial and motor information, not an isolated milk concept. Layer 30 is an additional later-layer comparison, not a claimed repair.')
(OUT/'settings.json').write_text(json.dumps(settings,indent=2));torch.save(dict(states=cache,first_inputs=first_inputs),OUT/'captured_states.pt');np.savez_compressed(OUT/'arrays.npz',**predictions);(OUT/'complete.json').write_text(json.dumps(dict(state='complete')));print(json.dumps(settings),flush=True)
