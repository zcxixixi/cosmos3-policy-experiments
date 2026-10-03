"""Natural counterfactual language K/V at matched denoising windows, full scheduler."""
import json,os,importlib.util,time
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1'
import torch,numpy as np
from PIL import Image
from diffusers import Cosmos3OmniPipeline
from diffusers.pipelines.cosmos import pipeline_cosmos3_omni as cm
R=Path('/home/current/work/cosmos3');D=R/'outputs/official-demos/milk-posttrain/gripper-context/deep-dive';O=D/'language-timing';O.mkdir(exist_ok=False);t0=time.perf_counter();torch.set_grad_enabled(False);cm._EMBODIMENT_TO_RAW_ACTION_DIM['libero']=10
p=Cosmos3OmniPipeline.from_pretrained(str(R/'checkpoints/Cosmos3-Nano-Policy-LIBEROall-5k'),torch_dtype=torch.bfloat16,sound_tokenizer=None,enable_safety_checker=False,local_files_only=True);p.to('cuda');p.vae.enable_tiling();m=p.transformer
old=p._build_action_json_prompt
def builder(*a,**kw):
 x=json.loads(old(*a,**kw));x['actions'][0]['idle_frame']='0 out of 16.';return json.dumps(x)
p._build_action_json_prompt=builder
spec=importlib.util.spec_from_file_location('flow',R/'cosmos-framework/cosmos_framework/model/generator/diffusion/samplers/fm_solvers_unipc.py');s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s);p.register_to_config(use_native_flow_schedule=True)
x=torch.load(D/'semantic-transport/states.pt',weights_only=True,map_location='cpu',mmap=True)
# UND is causal text-only and must be invariant over all captured noise phases.
for n in ['center_milk_198','center_butter_198','far_milk_198','far_butter_198']:
 for st in [10,20,29]:
  assert all(torch.equal(a,b) for a,b in zip(x['traces'][n]['stages'][0]['text'],x['traces'][n]['stages'][st]['text'])),(n,st)
def gpu(v):
 if isinstance(v,torch.Tensor):return v.cuda()
 if isinstance(v,list):return [gpu(a) for a in v]
 if isinstance(v,dict):return {k:gpu(a) for k,a in v.items()}
 return v
kv={};hs=[]
for l,layer in enumerate(m.layers):
 for key,proj in [('k',layer.self_attn.to_k),('v',layer.self_attn.to_v)]:
  def save(mod,args,out,l=l,key=key):kv[(l,key)]=out.detach().clone()
  hs.append(proj.register_forward_hook(save))
m(**gpu(x['inputs']['center_butter_198'][0]))
for h in hs:h.remove()
# Shared text prompt makes donor K/V identical across physical scenes.
for l in range(36):assert torch.equal(x['traces']['center_butter_198']['stages'][0]['text'][l],x['traces']['far_butter_198']['stages'][0]['text'][l])
del x
step=-1;window=set();gain=1.;record=False;rows=[];arrays={}
def pre(mod,args,kw):
 global step
 step+=1
h0=m.register_forward_pre_hook(pre,with_kwargs=True)
def swap(l):
 def hook(mod,args,out):
  if step not in window:return
  hs=[]
  for key,proj in [('k',mod.to_k),('v',mod.to_v)]:
   def sub(mo,ar,o,l=l,key=key):return kv[(l,key)]
   hs.append(proj.register_forward_hook(sub))
  try:_,changed=mod.processor(mod,*args)
  finally:
   for h in hs:h.remove()
  return out[0],changed
 return hook
handles=[layer.self_attn.register_forward_hook(swap(l)) for l,layer in enumerate(m.layers)]
images={'center':D/'object-position/inputs/center/input.png','far':D/'milk-far/inputs/xp15/input.png'}
for scene,img in images.items():
 for seed in [195,196,198]:
  native=np.load(D/'semantic-transport'/(f'{scene}_milk_{seed}_actions.npy'));alternate=np.load(D/'semantic-transport'/(f'{scene}_butter_{seed}_actions.npy'));gap=alternate[:,:3]-native[:,:3];den=float(np.sum(gap**2));assert den>0
  for label,steps in [('self',[]),('early',range(0,10)),('middle',range(10,20)),('late',range(20,30)),('all',range(30))]:
   step=-1;window=set(steps);p.scheduler=s.FlowUniPCMultistepScheduler(num_train_timesteps=1000,shift=1.)
   c=cm.CosmosActionCondition(mode='policy',chunk_size=16,domain_name='libero',resolution_tier=256,image=Image.open(img).convert('RGB'),view_point='concat_view')
   result=p(prompt='pick up the milk and place it in the basket',action=c,fps=20,num_inference_steps=30,guidance_scale=1.,generator=torch.Generator(device='cuda').manual_seed(seed),output_type='latent');a=result.action[0].float().cpu().numpy();assert step==29
   if label=='self':assert np.array_equal(a,native)
   if label=='all':assert np.array_equal(a,alternate)
   delta=a[:,:3]-native[:,:3];row=dict(scene=scene,seed=seed,window=label,xyz_rmse=float(np.sqrt(np.mean(delta**2))),projection_towards_full_noun_change=float(np.sum(delta*gap))/den,native_noun_xyz_rmse=float(np.sqrt(np.mean(gap**2))))
   rows.append(row);arrays[f'{scene}_{seed}_{label}']=a;(O/'progress.json').write_text(json.dumps(dict(completed=len(rows),total=30)));print('[LANGUAGE-TIMING]',row,flush=True)
for h in handles+[h0]:h.remove()
(O/'result.json').write_text(json.dumps(dict(rows=rows,checks=dict(text_states_invariant_over_captured_noise_phases=True,donor_text_states_same_across_scenes=True,native_self_and_all_alternate_exact=True),scope='Frozen natural milk->butter text K/V swaps. No GEN replacement and original UND residuals remain. Full30 scheduler; 2 scenes x3 seeds x5 windows=30 full predictions. Phase effects can interact and do not add as probability percentages; actual task success not established.',elapsed_s=time.perf_counter()-t0),indent=2));np.savez_compressed(O/'arrays.npz',**arrays);(O/'complete.json').write_text(json.dumps(dict(state='complete')))
