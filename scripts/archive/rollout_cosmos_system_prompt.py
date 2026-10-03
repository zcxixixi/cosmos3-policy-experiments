"""All-query system-off rollouts with legacy pixels, actions and scheduler intact."""
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ['MUJOCO_GL'] = 'egl'
ROOT = Path('/home/current/work/cosmos3')
sys.path.insert(0, '/home/current/work/openpi-demo/cosmos-policy')
sys.path.insert(0, str(ROOT/'cosmos-framework'))
import imageio.v2 as imageio
import numpy as np
import requests
from PIL import Image
from libero.libero import benchmark
from cosmos_policy.experiments.robot.libero.libero_utils import get_libero_env
from cosmos_framework.simulation.libero.closed_loop_eval import _framewise_action_to_delta, _remap_gripper

DATA = ROOT/'outputs/official-demos/milk-posttrain/gripper-context/deep-dive'
OUT = DATA/'runtime-system-prompt'
assert (OUT/'probe_complete.json').exists()
assert requests.get('http://127.0.0.1:8921/health', timeout=5).text == 'ready'
DEST = OUT/'closed-loop'
DEST.mkdir(exist_ok=False)
refs = {
    'center': DATA/'object-position/inputs/center',
    'far': DATA/'milk-far/inputs/xp15',
}
initial = {s: np.load(d/'state.npy') for s, d in refs.items()}
delta = initial['far']-initial['center']
assert np.flatnonzero(delta).tolist() == [10]
assert abs(delta[10]-.15) < 1e-15
probe = np.load(OUT/'arrays.npz')
stats = json.loads((ROOT/'cosmos-framework/cosmos_framework/data/generator/action/normalizer_stats/libero_native_frame_wise_relative_rot6d.json').read_text())['global_raw']
lo, hi = np.array(stats['q01']), np.array(stats['q99'])
scale, offset = np.maximum(hi-lo, 1e-8)/2, (hi+lo)/2
suite = benchmark.get_benchmark_dict()['libero_object']()
env, _ = get_libero_env(suite.get_task(7), 'cosmos', resolution=256)
env.reset()
obs = env.set_init_state(suite.get_task_init_states(7)[0])
for _ in range(10):
    obs, _, _, _ = env.step([0., 0., 0., 0., 0., 0., -1.])
en = env.env
results = []


def view():
    return np.concatenate([obs['agentview_image'][::-1, ::-1],
                           obs['robot0_eye_in_hand_image'][::-1, ::-1]], axis=1).copy()


def measure():
    return dict(eef_xyz=obs['robot0_eef_pos'].tolist(),
        objects={n: en.sim.data.get_joint_qpos(o.joints[0])[:3].tolist()
                 for n, o in en.objects_dict.items()},
        grasped=[n for n, o in en.objects_dict.items()
                 if en._check_grasp(en.robots[0].gripper, o)],
        gripper_qpos=obs['robot0_gripper_qpos'].tolist(),
        milk_in_basket=bool(en._check_success()))


def quantify(records):
    objects = {}
    for name in records[0]['objects']:
        pos = np.array([r['objects'][name] for r in records])
        lift = pos[:, 2]-pos[0, 2]
        contact = [i for i, r in enumerate(records) if name in r['grasped']]
        raised = [i for i in range(len(records)-4)
                  if np.all(lift[i:i+5] >= .02) and any(name in r['grasped'] for r in records[:i+1])]
        objects[name] = dict(first_grasp_step=contact[0] if contact else None,
            grasp_frames=len(contact), first_sustained_2cm_lift_step=raised[0] if raised else None,
            max_lift_cm=float(lift.max()*100), final_lift_cm=float(lift[-1]*100),
            max_xy_displacement_cm=float(np.linalg.norm(pos[:, :2]-pos[0, :2], axis=1).max()*100))
    return objects


try:
    # Three on trajectories already exist and their first actions are exactly replayed
    # by this probe; do not silently duplicate them or call them new trials.
    tests = [('center', 'milk', 'off'), ('center', 'butter', 'off'),
             ('far', 'milk', 'off'), ('far', 'butter', 'off'), ('far', 'butter', 'on')]
    for scene, noun, mode in tests:
        name = f'{scene}_{noun}_{mode}'
        run = DEST/name
        run.mkdir()
        env.reset()
        obs = env.set_init_state(initial[scene])
        assert np.array_equal(view(), np.array(Image.open(refs[scene]/'input.png')))
        np.save(run/'initial_state.npy', initial[scene])
        records = [measure()]
        actions, clipped = [], []
        contract = []
        with imageio.get_writer(str(run/'actual.mp4'), fps=20, codec='libx264', macro_block_size=1) as writer:
            writer.append_data(view())
            for q in range(8):
                Image.fromarray(view()).save(run/f'input_{q:02d}.png')
                reply = requests.post('http://127.0.0.1:8921', json=dict(
                    scene=scene, noun=noun, mode=mode, chunk=q, seed=198+q,
                    output=str(run)), timeout=300)
                reply.raise_for_status()
                value = reply.json()
                assert value['use_system_prompt'] == (mode == 'on')
                norm = np.array(value['actions'])
                if q == 0:
                    assert np.array_equal(norm, probe[f'{scene}_{noun}_198_{mode}'])
                contract.append(dict(query=q, seed=198+q, system_on=value['use_system_prompt'],
                                     und_len=value['und_len']))
                native = np.array([_remap_gripper(_framewise_action_to_delta(a, '6d').tolist(), 'zero_one')
                                   for a in norm*scale+offset])
                for command in native:
                    sent = np.clip(command, -1, 1)
                    obs, _, _, _ = env.step(sent.tolist())
                    actions.append(command.tolist())
                    clipped.append(sent.tolist())
                    records.append(measure())
                    writer.append_data(view())
                (run/'trajectory.json').write_text(json.dumps(records))
                (run/'actions.json').write_text(json.dumps(actions))
                (run/'executed_actions.json').write_text(json.dumps(clipped))
                (run/'query_contract.json').write_text(json.dumps(contract, indent=2))
                (DEST/'status.json').write_text(json.dumps(dict(
                    case=name, completed_queries=q+1, total_queries=8, completed_cases=len(results))))
                print('[SYSTEM-PROMPT-ROLLOUT]', name, q+1, records[-1]['grasped'], flush=True)
        item = dict(case=name, scene=scene, noun=noun, system_on=mode == 'on',
            prompt=f'pick up the {noun} and place it in the basket', steps=128,
            pixel_initial_condition_exact=True, first_chunk_matches_offline_probe_exact=True,
            initial_state_sha256=hashlib.sha256(initial[scene].tobytes()).hexdigest(),
            per_object=quantify(records), final=records[-1],
            scope='Target contact/lift diagnostic. milk_in_basket is the original milk task checker, not a butter-task success checker.')
        results.append(item)
        (DEST/'result.json').write_text(json.dumps(dict(cases=results,
            scope='Five new 128-step closed loops, matched initial states and seed 198+query. Explicit system setting on every query; existing on controls are reused, not counted as new trials.'), indent=2))
finally:
    env.close()
(DEST/'complete.json').write_text(json.dumps(dict(state='complete', cases=len(results))))
print('[SYSTEM-PROMPT-ROLLOUT] COMPLETE', flush=True)
