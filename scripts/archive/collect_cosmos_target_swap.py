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

D=OUT/'deep-dive/target-swap';P=D/'inputs/swapped';P.mkdir(parents=True,exist_ok=False)
env,obs=make_env();en=env.env
try:
 base=np.load(OUT/'deep-dive/object-position/inputs/center/state.npy');obs=env.set_init_state(base)
 before=en.sim.data.qpos.copy();names=['milk_1','butter_1'];poses={n:en.sim.data.get_joint_qpos(en.objects_dict[n].joints[0]).copy() for n in names};changed=[]
 for n,other in [('milk_1','butter_1'),('butter_1','milk_1')]:
  q=poses[n].copy();q[:2]=poses[other][:2];joint=en.objects_dict[n].joints[0];en.sim.data.set_joint_qpos(joint,q)
 en.sim.forward();state=en.sim.get_state().flatten().copy();obs=env.set_init_state(state)
 changed=np.flatnonzero(en.sim.data.qpos!=before).tolist();assert len(changed)==4,changed
 contacts={n:[m for m,o in en.objects_dict.items() if m!=n and en.check_contact(en.objects_dict[n].contact_geoms,o.contact_geoms)] for n in names};assert all(not c for c in contacts.values()),contacts
 np.save(P/'state.npy',state);Image.fromarray(view(obs)).save(P/'input.png')
 (D/'inputs/metadata.json').write_text(json.dumps(dict(scenes=['swapped'],changed_qpos_indices=changed,intervention='Swap only milk and butter XY positions; preserve each object height and orientation, robot state and other objects.',before={n:q.tolist() for n,q in poses.items()},after={n:en.sim.data.get_joint_qpos(en.objects_dict[n].joints[0]).tolist() for n in names},contacts=contacts),indent=2));print('COLLECTED',changed,contacts)
finally:env.close()
