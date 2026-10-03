"""Hold milk fixed; exchange cream-cheese/butter positions in center/far scenes.

Use system-off for every native policy query, with otherwise legacy settings.
This does not separate visual appearance from physical geometry.
"""
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
OUT = DATA/'identity-position'
OUT.mkdir(exist_ok=False)
assert (DATA/'runtime-system-prompt/closed-loop/complete.json').exists()
assert requests.get('http://127.0.0.1:8921/health', timeout=5).text == 'ready'
refs = dict(center=DATA/'object-position/inputs/center', far=DATA/'milk-far/inputs/xp15')
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
controls = []


def view():
    return np.concatenate([obs['agentview_image'][::-1, ::-1],
                           obs['robot0_eye_in_hand_image'][::-1, ::-1]], axis=1).copy()


def measure():
    return dict(eef_xyz=obs['robot0_eef_pos'].tolist(),
        objects={n: en.sim.data.get_joint_qpos(o.joints[0])[:3].tolist()
                 for n, o in en.objects_dict.items()},
        grasped=[n for n, o in en.objects_dict.items()
                 if en._check_grasp(en.robots[0].gripper, o)],
        milk_in_basket=bool(en._check_success()))


def pair_collisions():
    geom_owner = {}
    for name, obj in en.objects_dict.items():
        for geom in obj.contact_geoms:
            geom_owner[en.sim.model.geom_name2id(geom)] = name
    hits = []
    for contact in en.sim.data.contact[:en.sim.data.ncon]:
        a, b = geom_owner.get(contact.geom1), geom_owner.get(contact.geom2)
        if a is not None and b is not None and a != b:
            hits.append([a, b])
    return hits


def quantify(records):
    values = {}
    for name in records[0]['objects']:
        pos = np.array([r['objects'][name] for r in records])
        lift = pos[:, 2]-pos[0, 2]
        contact = [i for i, r in enumerate(records) if name in r['grasped']]
        values[name] = dict(first_grasp_step=contact[0] if contact else None,
            grasp_frames=len(contact), max_lift_cm=float(lift.max()*100),
            max_xy_displacement_cm=float(np.linalg.norm(pos[:, :2]-pos[0, :2], axis=1).max()*100))
    return values


try:
    for base_scene in ['far', 'center']:
        original = np.load(refs[base_scene]/'state.npy')
        env.reset()
        obs = env.set_init_state(original)
        assert np.array_equal(view(), np.array(Image.open(refs[base_scene]/'input.png')))
        before = measure()
        cream = en.objects_dict['cream_cheese_1']
        butter = en.objects_dict['butter_1']
        cq = en.sim.data.get_joint_qpos(cream.joints[0]).copy()
        bq = en.sim.data.get_joint_qpos(butter.joints[0]).copy()
        cn, bn = cq.copy(), bq.copy()
        cn[:2], bn[:2] = bq[:2], cq[:2]
        en.sim.data.set_joint_qpos(cream.joints[0], cn)
        en.sim.data.set_joint_qpos(butter.joints[0], bn)
        en.sim.forward()
        swapped = en.sim.get_state().flatten()
        changed = np.flatnonzero(swapped-original).tolist()
        assert len(changed) == 4, changed
        assert np.array_equal(en.sim.data.get_joint_qpos(cream.joints[0])[2:], cq[2:])
        assert np.array_equal(en.sim.data.get_joint_qpos(butter.joints[0])[2:], bq[2:])
        collisions = pair_collisions()
        assert collisions == [], collisions
        name = base_scene+'_cream_butter_swap_milk_off'
        run = OUT/name
        run.mkdir()
        np.save(run/'initial_state.npy', swapped)
        env.reset()
        obs = env.set_init_state(swapped)
        after = measure()
        assert np.array_equal(before['eef_xyz'], after['eef_xyz'])
        for n in before['objects']:
            if n not in ['cream_cheese_1', 'butter_1']:
                assert np.array_equal(before['objects'][n], after['objects'][n]), n
        control = dict(case=name, changed_state_indexes=changed,
            state_delta=(swapped-original).tolist(), before=before, after=after,
            interobject_collision_pairs=collisions, own_z_and_orientation_preserved=True,
            milk_and_robot_and_other_objects_exact=True)
        controls.append(control)
        (OUT/'controls.json').write_text(json.dumps(controls, indent=2))
        records = [after]
        actions = []
        contract = []
        with imageio.get_writer(str(run/'actual.mp4'), fps=20, codec='libx264', macro_block_size=1) as writer:
            writer.append_data(view())
            for q in range(8):
                Image.fromarray(view()).save(run/f'input_{q:02d}.png')
                reply = requests.post('http://127.0.0.1:8921', json=dict(
                    scene=base_scene+'_cream_butter_swap', noun='milk', mode='off',
                    chunk=q, seed=198+q, output=str(run)), timeout=300)
                reply.raise_for_status()
                value = reply.json()
                assert value['use_system_prompt'] is False
                contract.append(dict(query=q, seed=198+q, system_on=False, und_len=value['und_len']))
                norm = np.array(value['actions'])
                native = np.array([_remap_gripper(_framewise_action_to_delta(a, '6d').tolist(), 'zero_one')
                                   for a in norm*scale+offset])
                for command in native:
                    sent = np.clip(command, -1, 1)
                    obs, _, _, _ = env.step(sent.tolist())
                    actions.append(sent.tolist())
                    records.append(measure())
                    writer.append_data(view())
                (run/'trajectory.json').write_text(json.dumps(records))
                (run/'executed_actions.json').write_text(json.dumps(actions))
                (run/'query_contract.json').write_text(json.dumps(contract, indent=2))
                (OUT/'status.json').write_text(json.dumps(dict(case=name, completed_queries=q+1,
                    total_queries=8, completed_cases=len(results))))
                print('[IDENTITY-POSITION]', name, q+1, records[-1]['grasped'], flush=True)
        result = dict(case=name, steps=128, scene=base_scene, noun='milk', system_on=False,
            per_object=quantify(records), final=records[-1])
        results.append(result)
        (OUT/'result.json').write_text(json.dumps(dict(cases=results,
            scope='Two native system-off 128-step rollouts. Hold milk fixed, exchange cream/butter XY only. Distinguishes tracking cream from occupancy at its old slot in these scenes; not training-memory proof or appearance/geometry separation.'), indent=2))
finally:
    env.close()
(OUT/'complete.json').write_text(json.dumps(dict(state='complete', cases=len(results))))
print('[IDENTITY-POSITION] COMPLETE', flush=True)
