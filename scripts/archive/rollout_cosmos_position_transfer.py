"""Matched physical scene counterfactuals; only milk freejoint position changes."""
import os,sys,json
from pathlib import Path
os.environ['MUJOCO_GL']='egl'
ROOT=Path('/home/current/work/cosmos3')
sys.path.insert(0,'/home/current/work/openpi-demo/cosmos-policy');sys.path.insert(0,str(ROOT/'cosmos-framework'))
import numpy as np
from PIL import Image
from libero.libero import benchmark
from cosmos_policy.experiments.robot.libero.libero_utils import get_libero_env
OUT=ROOT/'outputs/official-demos/milk-posttrain/gripper-context'
def make_env():
 suite=benchmark.get_benchmark_dict()['libero_object']();env,_=get_libero_env(suite.get_task(7),'cosmos',resolution=256);env.reset();obs=env.set_init_state(suite.get_task_init_states(7)[0])
 for _ in range(10):obs,_,_,_=env.step([0.,0.,0.,0.,0.,0.,-1.])
 return env,obs
def view(obs):return np.concatenate([obs['agentview_image'][::-1,::-1],obs['robot0_eye_in_hand_image'][::-1,::-1]],axis=1).copy()

import imageio.v2 as imageio
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cosmos_framework.simulation.libero.closed_loop_eval import _framewise_action_to_delta,_remap_gripper
D=OUT/'deep-dive/position-transfer';SOURCE=OUT/'deep-dive/target-swap/inputs/swapped'
stats=json.loads((ROOT/'cosmos-framework/cosmos_framework/data/generator/action/normalizer_stats/libero_native_frame_wise_relative_rot6d.json').read_text())['global_raw'];lo=np.array(stats['q01']);hi=np.array(stats['q99']);scale=np.maximum(hi-lo,1e-8)/2;offset=(hi+lo)/2

import requests
requests.get('http://127.0.0.1:8921/health',timeout=5).raise_for_status()
C=D/'closed-loop';C.mkdir(exist_ok=False)
env,obs=make_env();en=env.env;summary=[]
def measure():
 return dict(eef_xyz=obs['robot0_eef_pos'].tolist(),objects={n:en.sim.data.get_joint_qpos(o.joints[0])[:3].tolist() for n,o in en.objects_dict.items()},grasped=[n for n,o in en.objects_dict.items() if en._check_grasp(en.robots[0].gripper,o)],gripper_qpos=obs['robot0_gripper_qpos'].tolist(),milk_in_basket=bool(en._check_success()))
try:
 for case in ['B_from_A_xp_30']:
  run=C/case;run.mkdir();env.reset();obs=env.set_init_state(np.load(SOURCE/'state.npy'));assert np.array_equal(view(obs),np.array(Image.open(SOURCE/'input.png')))
  records=[measure()];actions=[];prompt='pick up the milk and place it in the basket'
  with imageio.get_writer(str(run/'actual.mp4'),fps=20,codec='libx264',macro_block_size=1) as w:
   w.append_data(view(obs))
   for q in range(8):
    Image.fromarray(view(obs)).save(run/f'input_{q:02d}.png')
    if q==0:
     norm=np.load(D/(case+'_actions.npy'));chunk=run/'chunk_00';chunk.mkdir();(chunk/'normalized_actions.json').write_text(json.dumps(norm.tolist()));(chunk/'metadata.json').write_text(json.dumps(dict(source='Frozen offline matched-input state-exchange inference',case=case,seed=198)))
    else:
     response=requests.post('http://127.0.0.1:8921',json=dict(chunk=q,seed=198+q,prompt=prompt,output=str(run),tier=256,sampler='native',flow_shift=1,view_point='concat_view'),timeout=300);response.raise_for_status();norm=np.array(response.json()['actions'])
    native=np.array([_remap_gripper(_framewise_action_to_delta(a,'6d').tolist(),'zero_one') for a in norm*scale+offset])
    for command in native:
     obs,_,_,_=env.step(np.clip(command,-1,1).tolist());actions.append(command.tolist());records.append(measure());w.append_data(view(obs))
    (C/'status.json').write_text(json.dumps(dict(case=case,chunk=q+1,total_chunks=8,completed_cases=len(summary))));print(case,q+1,records[-1]['grasped'],flush=True)
  (run/'trajectory.json').write_text(json.dumps(records));(run/'actions.json').write_text(json.dumps(actions))
  item=dict(case=case,prompt=prompt,steps=128,ever_grasped=sorted(set(n for r in records for n in r['grasped'])),min_eef_milk_distance_m=min(float(np.linalg.norm(np.array(r['eef_xyz'])-r['objects']['milk_1'])) for r in records),final=records[-1],max_milk_xy_displacement_m=max(float(np.linalg.norm(np.array(r['objects']['milk_1'])[:2]-np.array(records[0]['objects']['milk_1'])[:2])) for r in records))
  summary.append(item);(C/'result.json').write_text(json.dumps(dict(cases=summary,scope='128-action closed-loop diagnostic, three state-exchange rollouts; patch only first action chunk, not a success-rate benchmark'),indent=2))
finally:env.close()
(C/'complete.json').write_text(json.dumps(dict(state='complete',cases=len(summary))))
