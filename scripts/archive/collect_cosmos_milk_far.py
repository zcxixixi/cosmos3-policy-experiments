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
DEST=OUT/'deep-dive/milk-far/inputs'
DEST.mkdir(parents=True,exist_ok=False)
env,obs=make_env();en=env.env
base=np.load(OUT/'deep-dive/object-position/inputs/center/state.npy');obs=env.set_init_state(base);milk=en.objects_dict['milk_1'];joint=milk.joints[0]
m0=en.sim.data.get_joint_qpos(joint).copy();rq=en.sim.data.qpos[en.robots[0]._ref_joint_pos_indexes].copy()
meta=dict(scenes={},task='pick up the milk and place it in the basket',shift_m=.15)
try:
 for name,shift in [('xp15',[.15,0])]:
  env.set_init_state(base);m=m0.copy();m[:2]+=shift;en.sim.data.set_joint_qpos(joint,m);en.sim.forward()
  state=en.sim.get_state().flatten().copy();obs=env.set_init_state(state)
  assert np.array_equal(en.sim.data.qpos[en.robots[0]._ref_joint_pos_indexes],rq)
  d=DEST/name;d.mkdir();np.save(d/'state.npy',state);Image.fromarray(view(obs)).save(d/'input.png')
  collision=[n for n,o in en.objects_dict.items() if n!='milk_1' and en.check_contact(milk.contact_geoms,o.contact_geoms)]
  assert not collision,(name,collision)
  meta['scenes'][name]=dict(milk_xyz=m[:3].tolist(),eef_xyz=obs['robot0_eef_pos'].tolist(),shift_xy=shift,other_object_contacts=collision)
 (DEST/'metadata.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta))
finally:env.close()
