"""Frozen first-step state exchange with matched noise and explicit null controls.

This measures causal dependence of a predicted Flow clean target. It does not
repair the checkpoint or claim that a patched robot completes a task.
"""
import importlib.util
import json
import os
import time
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
import numpy as np
import torch
from PIL import Image
from diffusers import Cosmos3OmniPipeline
from diffusers.pipelines.cosmos import pipeline_cosmos3_omni as cm

ROOT = Path('/home/current/work/cosmos3')
DATA = ROOT/'outputs/official-demos/milk-posttrain/gripper-context'
OUT = DATA/'deep-dive/object-position/causal'
OUT.mkdir(exist_ok=False)
started = time.perf_counter()
torch.set_grad_enabled(False)
cm._EMBODIMENT_TO_RAW_ACTION_DIM['libero'] = 10
pipe = Cosmos3OmniPipeline.from_pretrained(
    str(ROOT/'checkpoints/Cosmos3-Nano-Policy-LIBEROall-5k'),
    torch_dtype=torch.bfloat16, sound_tokenizer=None,
    enable_safety_checker=False, local_files_only=True)
pipe.to('cuda')
pipe.vae.enable_tiling()
original_builder = pipe._build_action_json_prompt
def caption_builder(*args, **kwargs):
    caption = json.loads(original_builder(*args, **kwargs))
    caption['actions'][0]['idle_frame'] = '0 out of 16.'
    return json.dumps(caption)
pipe._build_action_json_prompt = caption_builder
spec = importlib.util.spec_from_file_location('causal_native_flow',
    ROOT/'cosmos-framework/cosmos_framework/model/generator/diffusion/samplers/fm_solvers_unipc.py')
sampler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sampler)
pipe.register_to_config(use_native_flow_schedule=True)

class CapturedFirstInput(Exception):
    pass

def capture_input(image, prompt):
    captured = {}
    def stop(module, args, kwargs):
        assert not args
        captured.update(kwargs)
        raise CapturedFirstInput()
    handle = pipe.transformer.register_forward_pre_hook(stop, with_kwargs=True)
    pipe.scheduler = sampler.FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1.)
    try:
        condition = cm.CosmosActionCondition(mode='policy', chunk_size=16,
            domain_name='libero', resolution_tier=256,
            image=Image.open(image).convert('RGB'), view_point='concat_view')
        pipe(prompt=prompt, action=condition, fps=20, num_inference_steps=30,
            guidance_scale=1., generator=torch.Generator(device='cuda').manual_seed(198),
            output_type='latent')
    except CapturedFirstInput:
        pass
    finally:
        handle.remove()
    assert captured
    return captured, float(pipe.scheduler.sigmas[0])


inputs={}
for name in ['center','xp']:
 inputs[name],sigma=capture_input(DATA/'deep-dive/object-position/inputs'/name/'input.png','pick up the milk and place it in the basket')
def same(a,b):
 if isinstance(a,torch.Tensor):return torch.equal(a,b)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
 return a==b
assert all(same(inputs['center'][k],inputs['xp'][k]) for k in inputs['center'] if k!='vision_tokens')
assert torch.equal(inputs['center']['vision_tokens'][0][:,:,1:],inputs['xp']['vision_tokens'][0][:,:,1:])
model=pipe.transformer;states={};outputs={}
for name,kwargs in inputs.items():
 cache=[]
 def before(module,args):cache.append(tuple(t.detach().clone() for t in args[:2]))
 def after(module,args,out):cache.append(tuple(t.detach().clone() for t in out))
 handles=[model.layers[0].register_forward_pre_hook(before)]+[l.register_forward_hook(after) for l in model.layers]
 try:outputs[name]=model(**kwargs)[2][0].float()
 finally:
  for h in handles:h.remove()
 states[name]=cache
idx=inputs['center']['action_sequence_indexes'].flatten()-inputs['center']['und_len']
known=inputs['center']['vision_sequence_indexes'].flatten()[:50]-inputs['center']['und_len']
gap=(outputs['xp']-outputs['center'])[:,:3];den=float(gap.square().sum());assert den>0
rows=[]
for group,selection in [('known_vision',known),('action',idx),('all_gen',torch.arange(states['center'][0][1].shape[0],device='cuda'))]:
 for stage in [0,6,12,18,24,30,36]:
  def swap(pair,donor):
   u,g=pair;g=g.clone();g[selection]=states[donor][stage][1][selection];return u,g
  for donor in ['center','xp']:
   if stage==0:
    def hook(module,args):return (*swap(args[:2],donor),*args[2:])
    handle=model.layers[0].register_forward_pre_hook(hook)
   else:
    def hook(module,args,out):return swap(out,donor)
    handle=model.layers[stage-1].register_forward_hook(hook)
   try:out=model(**inputs['center'])[2][0].float()
   finally:handle.remove()
   delta=(out-outputs['center'])[:,:3]
   error=float((out-outputs['center']).abs().max())
   if donor=='center':assert error==0
   if donor=='xp' and group=='all_gen':assert float((out-outputs['xp']).abs().max())==0
   rows.append(dict(group=group,stage=stage,donor=donor,recovered_xyz_velocity_fraction=float((delta*gap).sum())/den,max_change=error))
result=dict(rows=rows,matched_nonvisual_inputs=True,matched_future_vision_noise=True,scope='First denoising step causal state exchange; not a repaired policy or rollout success.')
(OUT/'result.json').write_text(json.dumps(result,indent=2));torch.save(dict(inputs={n:{k:(v.cpu() if isinstance(v,torch.Tensor) else [t.cpu() if isinstance(t,torch.Tensor) else t for t in v] if isinstance(v,list) else v) for k,v in kw.items()} for n,kw in inputs.items()},states={n:[tuple(t.cpu() for t in pair) for pair in cache] for n,cache in states.items()},velocities={n:v.cpu() for n,v in outputs.items()}),OUT/'states.pt')
print(json.dumps(result),flush=True)
