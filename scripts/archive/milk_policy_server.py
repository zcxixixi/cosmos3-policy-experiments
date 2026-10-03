import os,json,time,traceback,importlib.util
os.environ['HF_HUB_OFFLINE']='1'
from pathlib import Path
from http.server import HTTPServer,BaseHTTPRequestHandler
import torch
from PIL import Image
from diffusers import Cosmos3OmniPipeline,UniPCMultistepScheduler
from diffusers.pipelines.cosmos import pipeline_cosmos3_omni as cm
from diffusers.utils import export_to_video
R=Path('/home/current/work/cosmos3');O=Path(os.environ.get('MILK_OUTPUT',str(R/'outputs/official-demos/milk-base-policy')))
# Explicit exploratory adapter: domain ID 5 exists; Diffusers omits variable-width LIBERO.
# Select the official LIBERO SFT 10D convention. This does NOT make base weights LIBERO-trained.
cm._EMBODIMENT_TO_RAW_ACTION_DIM['libero']=10
model_path=os.environ.get('MILK_MODEL_CHECKPOINT',str(R/'checkpoints/Cosmos3-Nano'))
pipe=Cosmos3OmniPipeline.from_pretrained(model_path,torch_dtype=torch.bfloat16,sound_tokenizer=None,enable_safety_checker=False,local_files_only=True)
print('LOADED MODEL',model_path,flush=True)
prompt_format=os.environ.get('MILK_PROMPT_FORMAT','diffusers')
assert prompt_format in ['diffusers','framework_json']
if prompt_format=='framework_json':
 # Match the official LIBERO server's active-manipulation JSON condition.
 # Diffusers supplies the same caption fields except idle_frame.
 original_prompt_builder=pipe._build_action_json_prompt
 def framework_prompt_builder(*a,**kw):
  caption=json.loads(original_prompt_builder(*a,**kw))
  caption['actions'][0]['idle_frame']='0 out of 16.'
  return json.dumps(caption)
 pipe._build_action_json_prompt=framework_prompt_builder
adaptation=os.environ.get('MILK_LORA')
if adaptation:
 ad=Path(adaptation)
 pipe.transformer.load_lora_adapter(str(ad/'lora'),prefix=None,weight_name='pytorch_lora_weights.safetensors',use_safetensors=True)
 patch=torch.load(ad/'heads.pt',map_location='cpu',weights_only=True)
 assert patch['domain']==5
 with torch.no_grad():
  pipe.transformer.action_proj_in.fc.weight[5].copy_(patch['in_w'])
  pipe.transformer.action_proj_in.bias.weight[5].copy_(patch['in_b'])
  pipe.transformer.action_proj_out.fc.weight[5].copy_(patch['out_w'])
  pipe.transformer.action_proj_out.bias.weight[5].copy_(patch['out_b'])
  pipe.transformer.action_modality_embed.copy_(patch['action_embed'])
 print('LOADED JOINT TASK ADAPTATION',adaptation,flush=True)
head_path=os.environ.get('MILK_ACTION_HEAD')
if head_path:
 patch=torch.load(head_path,map_location='cpu',weights_only=True)
 assert patch['domain']==5 and patch['weight'].shape==(4096,10)
 with torch.no_grad():
  pipe.transformer.action_proj_out.fc.weight[5].view(4096,64)[:,:10].copy_(patch['weight'])
  pipe.transformer.action_proj_out.bias.weight[5,:10].copy_(patch['bias'])
 print('LOADED DOMAIN-5 OUTPUT HEAD',head_path,'offline val MSE',patch['val_loss'],flush=True)
base_scheduler_config=dict(pipe.scheduler.config)
base_native_flow_schedule=pipe.config.use_native_flow_schedule
default_sampler=os.environ.get('MILK_SAMPLER','legacy')
default_flow_shift=float(os.environ.get('MILK_FLOW_SHIFT','1' if default_sampler=='native' else '10'))
def make_scheduler(kind,shift):
 if kind=='legacy':return UniPCMultistepScheduler.from_config(base_scheduler_config,flow_shift=shift)
 assert kind=='native',kind
 source=R/'cosmos-framework/cosmos_framework/model/generator/diffusion/samplers/fm_solvers_unipc.py'
 spec=importlib.util.spec_from_file_location('milk_native_flow_unipc',source)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 scheduler=module.FlowUniPCMultistepScheduler(num_train_timesteps=1000,shift=1.)
 scheduler.register_to_config(shift=shift)
 return scheduler
pipe.scheduler=make_scheduler(default_sampler,default_flow_shift)
pipe.register_to_config(use_native_flow_schedule=True if default_sampler=='native' else base_native_flow_schedule)
residency=os.environ.get('MILK_RESIDENCY','offload')
assert residency in ['offload','resident']
if residency=='resident':pipe.to('cuda')
else:pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()
profile=os.environ.get('MILK_PROFILE','0')=='1'
timing={}
if profile:
 def measure_method(obj,name,label):
  original=getattr(obj,name)
  def measured(*args,**kwargs):
   torch.cuda.synchronize();start=time.perf_counter()
   value=original(*args,**kwargs)
   torch.cuda.synchronize();timing[label]=timing.get(label,0.)+time.perf_counter()-start
   return value
  setattr(obj,name,measured)
 measure_method(pipe,'_encode_video','image_encode_s')
 measure_method(pipe.transformer,'forward','transformer_s')
 measure_method(pipe.vae,'decode','video_decode_s')
 measure_method(pipe.video_processor,'postprocess_video','video_postprocess_s')
class H(BaseHTTPRequestHandler):
 def do_GET(self):
  self.send_response(200);self.end_headers();self.wfile.write(b'ready')
 def do_POST(self):
  try:
   req=json.loads(self.rfile.read(int(self.headers['Content-Length'])));q=int(req['chunk']);run=Path(req.get('output',str(O)));p=run/f'chunk_{q:02d}';p.mkdir(exist_ok=True)
   im=Image.open(run/f'input_{q:02d}.png').convert('RGB');cond=cm.CosmosActionCondition(mode='policy',chunk_size=16,domain_name='libero',resolution_tier=int(req.get('tier',256)),image=im,view_point=req.get('view_point','concat_view'))
   flow_shift=float(req.get('flow_shift',default_flow_shift))
   sampler=req.get('sampler',default_sampler)
   pipe.scheduler=make_scheduler(sampler,flow_shift)
   pipe.register_to_config(use_native_flow_schedule=True if sampler=='native' else base_native_flow_schedule)
   print('INFER',q,'tier',cond.resolution_tier,'input',im.size,flush=True);timing.clear();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
   out=pipe(prompt=req['prompt'],action=cond,fps=20,num_inference_steps=30,guidance_scale=1.,generator=torch.Generator(device='cuda').manual_seed(int(req.get('seed',195+q))))
   a=out.action[0].float().cpu().numpy();assert a.shape==(16,10)
   (p/'normalized_actions.json').write_text(json.dumps(a.tolist()))
   export_start=time.perf_counter();export_to_video(out.video,str(p/'prediction.mp4'),fps=20);timing['video_export_s']=time.perf_counter()-export_start
   out.video[-1].save(p/'predicted_last.png');(p/'metadata.json').write_text(json.dumps({'tier':cond.resolution_tier,'sampler':sampler,'flow_shift':flow_shift,'prompt_format':prompt_format,'residency':residency,'timing':dict(timing),'gpu_peak_gib':torch.cuda.max_memory_allocated()/2**30,'sigmas':pipe.scheduler.sigmas.tolist(),'timesteps':pipe.scheduler.timesteps.tolist(),'input_size':im.size,'output_size':out.video[0].size,'elapsed_s':time.perf_counter()-started,'action_shape':list(a.shape)}))
   b=json.dumps({'actions':a.tolist()}).encode();self.send_response(200);self.end_headers();self.wfile.write(b)
  except Exception as e:
   traceback.print_exc();self.send_response(500);self.end_headers();self.wfile.write(str(e).encode())
print('MILK POLICY READY',flush=True)
HTTPServer(('127.0.0.1',8921),H).serve_forever()
