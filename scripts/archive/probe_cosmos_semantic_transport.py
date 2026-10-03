"""Frozen token-carrier causal transport across text/GEN and all denoising phases.
Native matched inputs; no training. Path swaps preserve original UND outputs.
"""
import importlib.util,json,os,time
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1'
import torch,numpy as np
from PIL import Image
from diffusers import Cosmos3OmniPipeline
from diffusers.pipelines.cosmos import pipeline_cosmos3_omni as cm
R=Path('/home/current/work/cosmos3');D=R/'outputs/official-demos/milk-posttrain/gripper-context/deep-dive'
O=D/'semantic-transport';O.mkdir(exist_ok=True);assert not (O/'complete.json').exists();t0=time.perf_counter()
torch.set_grad_enabled(False);cm._EMBODIMENT_TO_RAW_ACTION_DIM['libero']=10
pipe=Cosmos3OmniPipeline.from_pretrained(str(R/'checkpoints/Cosmos3-Nano-Policy-LIBEROall-5k'),torch_dtype=torch.bfloat16,sound_tokenizer=None,enable_safety_checker=False,local_files_only=True);pipe.to('cuda');pipe.vae.enable_tiling();m=pipe.transformer
orig=pipe._build_action_json_prompt
def builder(*a,**kw):
 d=json.loads(orig(*a,**kw));d['actions'][0]['idle_frame']='0 out of 16.';return json.dumps(d)
pipe._build_action_json_prompt=builder
spec=importlib.util.spec_from_file_location('native_flow',R/'cosmos-framework/cosmos_framework/model/generator/diffusion/samplers/fm_solvers_unipc.py');s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s);pipe.register_to_config(use_native_flow_schedule=True)
active=None;step=-1;idx=None;kwargs_current=None;traces={};outarrays={};inputs={};buffer={};capture=False
SELECT=[0,10,20,29]
def cpu(v):
 if isinstance(v,torch.Tensor):return v.detach().cpu().clone()
 if isinstance(v,list):return [cpu(x) for x in v]
 if isinstance(v,tuple):return tuple(cpu(x) for x in v)
 if isinstance(v,dict):return {k:cpu(x) for k,x in v.items()}
 return v
def gpu(v):
 if isinstance(v,torch.Tensor):return v.cuda()
 if isinstance(v,list):return [gpu(x) for x in v]
 if isinstance(v,tuple):return tuple(gpu(x) for x in v)
 if isinstance(v,dict):return {k:gpu(x) for k,x in v.items()}
 return v
def start_forward(mod,args,kw):
 global step,idx,kwargs_current
 if active is None:return
 step+=1;idx=kw['action_sequence_indexes'].flatten()-kw['und_len'];kwargs_current=kw
 if capture and step in SELECT:
  inputs[active][step]=cpu(kw);traces[active]['stages'][step]=dict(action=[],text=[],attention=[],mlp=[])
def before_first(mod,args):
 if capture and step in SELECT:
  tr=traces[active]['stages'][step];tr['action'].append(cpu(args[1][idx]));tr['text'].append(cpu(args[0]))
def layer_end(mod,args,out):
 if capture and step in SELECT:
  tr=traces[active]['stages'][step];tr['action'].append(cpu(out[1][idx]));tr['text'].append(cpu(out[0]))
def attn_end(mod,args,out):
 if capture and step in SELECT:traces[active]['stages'][step]['attention'].append(cpu(out[1][idx]))
def mlp_end(mod,args,out):
 if capture and step in SELECT:traces[active]['stages'][step]['mlp'].append(cpu(out[idx]))
def head_before(mod,args):
 if active is not None:traces[active]['readout'].append(cpu(args[0]))
handles=[m.register_forward_pre_hook(start_forward,with_kwargs=True),m.layers[0].register_forward_pre_hook(before_first),m.action_proj_out.register_forward_pre_hook(head_before)]
for l in m.layers:handles += [l.register_forward_hook(layer_end),l.self_attn.register_forward_hook(attn_end),l.mlp_moe_gen.register_forward_hook(mlp_end)]
images={'center':D/'object-position/inputs/center/input.png','far':D/'milk-far/inputs/xp15/input.png'}
rows=[]
for seed in [195,196,198]:
 for scene in images:
  for noun in ['milk','butter']:
   active=f'{scene}_{noun}_{seed}';step=-1;capture=seed==198;inputs[active]={};traces[active]={'stages':{},'readout':[]}
   pipe.scheduler=s.FlowUniPCMultistepScheduler(num_train_timesteps=1000,shift=1.)
   c=cm.CosmosActionCondition(mode='policy',chunk_size=16,domain_name='libero',resolution_tier=256,image=Image.open(images[scene]).convert('RGB'),view_point='concat_view')
   result=pipe(prompt=f'pick up the {noun} and place it in the basket',action=c,fps=20,num_inference_steps=30,guidance_scale=1.,generator=torch.Generator(device='cuda').manual_seed(seed),output_type='latent')
   outarrays[active]=result.action[0].float().cpu().numpy();traces[active]['readout']=torch.stack(traces[active]['readout']);np.save(O/(active+'_actions.npy'),outarrays[active]);rows.append(dict(case=active,gripper_mean=float(outarrays[active][:,9].mean())))
   print('[SEMANTIC-TRANSPORT] baseline',active,flush=True)
active=None;capture=False
for h in handles:h.remove()
# True output-head row-space comparison at all 30 noise levels.
torch.save(dict(traces=traces,inputs=inputs,arrays=outarrays,rows=rows),O/'native_checkpoint.pt')
W=m.action_proj_out.fc.weight[5].float().view(4096,64)[:,:10].T;_,_,vh=torch.linalg.svd(W,full_matrices=False);basis=vh.T
contrasts=[];profile=[]
def relative(a,b):return float((b.float()-a.float()).square().mean().sqrt()/a.float().square().mean().sqrt().clamp_min(1e-20))
for seed in [195,196,198]:
 for kind,left,right in [('language_center',f'center_milk_{seed}',f'center_butter_{seed}'),('language_far',f'far_milk_{seed}',f'far_butter_{seed}'),('scene_milk',f'center_milk_{seed}',f'far_milk_{seed}')]:
  a,b=traces[left]['readout'].float(),traces[right]['readout'].float();delta=b-a;proj=delta.cuda()@basis
  rel=(delta.square().mean((1,2)).sqrt()/a.square().mean((1,2)).sqrt()).numpy();frac=(proj.square().sum((1,2))/delta.cuda().square().sum((1,2)).clamp_min(1e-20)).cpu().numpy()
  outarrays[kind+f'_{seed}_readout_relative']=rel;outarrays[kind+f'_{seed}_head_rowspace_energy']=frac
  contrasts.append(dict(kind=kind,seed=seed,final_xyz_rmse=float(np.sqrt(np.mean((outarrays[right][:,:3]-outarrays[left][:,:3])**2))),first_readout_relative=float(rel[0]),last_readout_relative=float(rel[-1]),first_head_rowspace_energy=float(frac[0]),last_head_rowspace_energy=float(frac[-1])))
  if seed==198:
   for st in SELECT:
    ta,tb=traces[left]['stages'][st],traces[right]['stages'][st]
    for l in range(37):
     profile.append(dict(kind=kind,step=st,boundary=l,action_relative=relative(ta['action'][l],tb['action'][l]),text_relative=relative(ta['text'][l],tb['text'][l])))
checks=[];pathrows=[]
for scene in images:
 name=f'{scene}_milk_198';donor=f'{scene}_butter_198'
 for st in SELECT:
  kw=gpu(inputs[name][st]);dk=gpu(inputs[donor][st]);changed=torch.nonzero(kw['input_ids']!=dk['input_ids']).flatten().tolist();assert len(changed)==1;pos=changed[0]
  for k in kw:
   if k in ['input_ids','vision_tokens','action_tokens']:continue
   if isinstance(kw[k],torch.Tensor):assert torch.equal(kw[k],dk[k]),k
  assert torch.equal(kw['position_ids'],dk['position_ids'])
  # Clamp every GEN input to recipient: isolate direct language path at this noise level.
  dk['vision_tokens']=kw['vision_tokens'];dk['action_tokens']=kw['action_tokens']
  kv={};recorder=[]
  for l,layer in enumerate(m.layers):
   for key,proj in [('k',layer.self_attn.to_k),('v',layer.self_attn.to_v)]:
    def store(mod,args,out,l=l,key=key):kv[(l,key)]=out.detach().clone()
    recorder.append(proj.register_forward_hook(store))
  alternate=m(**dk)[2][0].float()
  for h in recorder:h.remove()
  native=m(**kw)[2][0].float();gap=alternate-native;den=gap[:,:10].square().sum().clamp_min(1e-20)
  def swap(which,layers):
   hs=[]
   if which=='word':selection=torch.tensor([pos],device='cuda')
   elif which=='suffix':selection=torch.arange(pos+1,kw['und_len'],device='cuda')
   elif which=='prefix':selection=torch.arange(pos,device='cuda')
   else:selection=torch.arange(kw['und_len'],device='cuda')
   for l in layers:
    attn=m.layers[l].self_attn
    def hook(mod,args,out,l=l,selection=selection):
     repl=[]
     for key,proj in [('k',mod.to_k),('v',mod.to_v)]:
      def sub(mo,ar,o,key=key,l=l):
       x=o.clone();x[selection]=kv[(l,key)][selection];return x
      repl.append(proj.register_forward_hook(sub))
     try:_,g=mod.processor(mod,*args)
     finally:
      for h in repl:h.remove()
     return out[0],g
    hs.append(attn.register_forward_hook(hook))
   try:res=m(**kw)[2][0].float()
   finally:
    for h in hs:h.remove()
   return res
  for group in ['prefix','word','suffix','all']:
   for window in [list(range(36))]+([list(range(i,i+6)) for i in range(0,36,6)] if st in [0,29] and group in ['word','suffix'] else []):
    pred=swap(group,window);delta=pred-native
    err=float((pred-alternate).abs().max())
    if group=='prefix':assert float(delta.abs().max())==0
    if group=='all':assert err==0
    pathrows.append(dict(scene=scene,step=st,target_token=pos,group=group,layers=[l+1 for l in window],velocity_all10_rmse=float(delta[:,:10].square().mean().sqrt()),recovered_language_velocity_fraction=float((delta[:,:10]*gap[:,:10]).sum()/den),max_error_vs_clamped_alternate=err,clamped_language_velocity_rms=float(gap[:,:10].square().mean().sqrt())))
  checks.append(dict(scene=scene,step=st,matching_positions=True,prefix_swap_exact_zero=True,all_text_KV_recovers_clamped_alternate_exactly=True))
  print('[SEMANTIC-TRANSPORT] path',scene,st,flush=True)
  del kw,dk,kv
# Compare language deltas to MLP updates without treating anticorrelation as causation.
mlp=[]
for scene in images:
 a=traces[f'{scene}_milk_198'];b=traces[f'{scene}_butter_198']
 for st in SELECT:
  da=torch.stack(b['stages'][st]['attention']).float()-torch.stack(a['stages'][st]['attention']).float();dm=torch.stack(b['stages'][st]['mlp']).float()-torch.stack(a['stages'][st]['mlp']).float()
  cos=(da.flatten(1)*dm.flatten(1)).sum(1)/(da.flatten(1).norm(dim=1)*dm.flatten(1).norm(dim=1)).clamp_min(1e-20)
  for l in range(36):mlp.append(dict(scene=scene,step=st,layer=l+1,attention_delta_rms=float(da[l].square().mean().sqrt()),mlp_delta_rms=float(dm[l].square().mean().sqrt()),delta_cosine=float(cos[l])))
# Existing 198 native outputs must reproduce the actual collected diagnostics exactly.
refs={'center_milk_198':D/'state-exchange/A_actions.npy','far_milk_198':D/'milk-far/closed-loop/xp15_milk/chunk_00/normalized_actions.json'}
for n,p in refs.items():
 ref=np.load(p) if p.suffix=='.npy' else np.array(json.loads(p.read_text()),dtype=np.float32)
 assert np.array_equal(ref,outarrays[n]),n
checks.append(dict(saved_center_and_far_native_first_chunk_exact=True))
result=dict(baselines=rows,contrasts=contrasts,profile=profile,path_rows=pathrows,mlp=mlp,checks=checks,elapsed_s=time.perf_counter()-t0,scope='Frozen 12 full action predictions (2 scenes x2 nouns x3 seeds), all30 readouts. Token-carrier KV exchange at4 denoising phases clamps GEN inputs. Not a repaired model, not decoded grasp probabilities. Head row-space is geometry, not semantic identity. Suffix includes JSON metadata whose contextual states carry preceding text.')
(O/'result.json').write_text(json.dumps(result,indent=2));torch.save(dict(traces=traces,inputs=inputs),O/'states.pt');np.savez_compressed(O/'arrays.npz',**outarrays);(O/'complete.json').write_text(json.dumps(dict(state='complete',elapsed_s=result['elapsed_s'])));print('[SEMANTIC-TRANSPORT] complete',flush=True)
