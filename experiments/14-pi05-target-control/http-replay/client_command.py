import datetime,hashlib,importlib.metadata,json,os,signal,subprocess,time,urllib.error,urllib.request
from pathlib import Path
import numpy as np
root=Path('/home/current/work/cosmos3')
base=root/'outputs/official-demos/milk-posttrain/gripper-context/deep-dive/pi05-target-control'
out=base/'http-replay'
server=json.loads((root/'work/pi05_http_replay_server_process.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(path,value):
 with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
with urllib.request.urlopen('http://127.0.0.1:8923/health',timeout=5) as r:health=json.load(r)
assert health['state']=='ready' and health['requests_completed']==0
assert health['script_sha256']==server['script_sha256']
assert not out.exists()
source=base/'closed-loop/center_milk_seed198/chunk_00/simulator_inputs.npz'
assert sha(source)=='28862f0117842851c7dca730e1041e74c372f0e24ddadf87d5e23a68f2bb14d8'
out.mkdir()
chunk=out/'center_milk_seed198/chunk_00'
chunk.mkdir(parents=True)
with (chunk/'simulator_inputs.npz').open('xb') as f:f.write(source.read_bytes())
assert sha(chunk/'simulator_inputs.npz')==sha(source)
paths=['src/openpi/models/model.py','src/openpi/models/gemma.py','src/openpi/models/siglip.py','src/openpi/models/lora.py','src/openpi/shared/nnx_utils.py','src/openpi/training/config.py','src/openpi/training/sharding.py']
packages={}
for name in ['jax','jaxlib','flax','orbax-checkpoint','numpy','torch','sentencepiece','openpi-client']:
 try:packages[name]=importlib.metadata.version(name)
 except importlib.metadata.PackageNotFoundError:packages[name]=None
checkpoint=Path(health['checkpoint'])
provenance=dict(server_record=server,server_health_before_request=health,actual_server_jax_config=health['actual_jax_config'],packages=packages,
 additional_sources={p:sha(Path('/home/current/work/openpi-demo/openpi')/p) for p in paths},
 checkpoint_metadata_sha256=sha(checkpoint/'params/_METADATA'),checkpoint_manifest_sha256=sha(checkpoint/'params/manifest.ocdbt'),
 norm_stats_sha256=sha(checkpoint/'assets/physical-intelligence/libero/norm_stats.json'),tokenizer_sha256=sha(Path('/home/current/.cache/openpi/big_vision/paligemma_tokenizer.model')),
 source_input=str(source),source_input_sha256=sha(source),new_input_sha256=sha(chunk/'simulator_inputs.npz'),
 simulator_started=False,egl_started=False,action_execution=False,warmup=False,repeat=False,requests_requested=1)
save(out/'provenance.json',provenance)
request=dict(scene='center',noun='milk',query=0,seed=198,inputs_sha256=sha(chunk/'simulator_inputs.npz'))
save(out/'request.json',request)
req=urllib.request.Request('http://127.0.0.1:8923/infer',data=json.dumps(request).encode(),headers={'Content-Type':'application/json'},method='POST')
status=None;body=b'';request_error=None
try:
 try:
  with urllib.request.urlopen(req,timeout=60) as r:status=r.status;body=r.read()
 except urllib.error.HTTPError as exc:status=exc.code;body=exc.read()
 except Exception as exc:request_error=str(exc)
 with (out/'http_response_body.json').open('xb') as f:f.write(body)
 save(out/'http_response.json',dict(status=status,body=json.loads(body) if body else None,error=request_error))
 print('[PI05-HTTP-REPLAY] response',status,flush=True)
finally:
 pid=server['pid'];proc=Path(f'/proc/{pid}')
 if proc.exists():
  assert proc.joinpath('cmdline').read_bytes().split(b'\0')[2].decode()==server['command'][2]
  os.kill(pid,signal.SIGTERM)
  deadline=time.monotonic()+15
  while proc.exists() and time.monotonic()<deadline:time.sleep(.25)
 assert not proc.exists(),'Own server did not stop after SIGTERM'
 save(out/'shutdown.json',dict(server_pid=pid,server_process_absent=True,
  gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip(),
  compute_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()))
assert request_error is None,request_error
with np.load(chunk/'model_arrays.npz',allow_pickle=False) as f:a={k:f[k].copy() for k in f.files}
reference=base/'predictions/center_milk_198/arrays.npz'
with np.load(reference,allow_pickle=False) as f:b={k:f[k].copy() for k in f.files}
for key in b:
 if key!='actions_native_7':
  assert np.array_equal(a[key],b[key]) and a[key].dtype==b[key].dtype,key
assert np.array_equal(a['actions_native_7'],np.asarray(json.loads((chunk/'actions.json').read_text())))
delta=a['actions_native_7']-b['actions_native_7']
comparison=dict(exact=bool(np.array_equal(a['actions_native_7'],b['actions_native_7'])),
 max_abs=float(np.abs(delta).max()),rmse=float(np.sqrt(np.mean(delta**2))),
 per_dim_max_abs=np.abs(delta).max(axis=0).tolist(),per_dim_rmse=np.sqrt(np.mean(delta**2,axis=0)).tolist(),
 gripper_sign_equal=bool(np.array_equal(np.sign(a['actions_native_7'][:,6]),np.sign(b['actions_native_7'][:,6]))),
 gripper_sign_changed_steps=np.flatnonzero(np.sign(a['actions_native_7'][:,6])!=np.sign(b['actions_native_7'][:,6])).tolist(),
 native_units='LIBERO controller commands, XYZ not meters')
save(out/'result.json',dict(http_status=status,comparison=comparison,all_nonaction_arrays_and_dtype_exact=True,
 actual_arrays_sha256=sha(chunk/'model_arrays.npz'),reference_arrays_sha256=sha(reference),
 old_failed_server_actual_output_unavailable=True,scope='One actual HTTP handler request without simulator; no task success claim'))
save(out/'complete.json',dict(state='complete',requests=1,http_status=status,guard_passed=status==200,simulator_execution=False))
print('[PI05-HTTP-REPLAY] COMPLETE',json.dumps(comparison),flush=True)
