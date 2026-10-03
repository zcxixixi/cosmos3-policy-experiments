"""Repeat four matched identity/position scenes with two additional noise seeds."""
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
BASE = DATA/'identity-position'
assert (BASE/'complete.json').exists()
assert requests.get('http://127.0.0.1:8921/health', timeout=5).text == 'ready'
OUT = BASE/'repeats'
OUT.mkdir(exist_ok=False)
sources = {
    'center_original': DATA/'runtime-system-prompt/closed-loop/center_milk_off',
    'far_original': DATA/'runtime-system-prompt/closed-loop/far_milk_off',
    'center_swapped': BASE/'center_cream_butter_swap_milk_off',
    'far_swapped': BASE/'far_cream_butter_swap_milk_off',
}
reference = np.load(DATA/'runtime-system-prompt/arrays.npz')
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
        objects={n: en.sim.data.get_joint_qpos(o.joints[0])[:3].tolist() for n, o in en.objects_dict.items()},
        grasped=[n for n, o in en.objects_dict.items() if en._check_grasp(en.robots[0].gripper, o)],
        milk_in_basket=bool(en._check_success()))


def quantify(records):
    result = {}
    for n in records[0]['objects']:
        xyz = np.array([r['objects'][n] for r in records])
        contact = [i for i, r in enumerate(records) if n in r['grasped']]
        result[n] = dict(first_grasp_step=contact[0] if contact else None, grasp_frames=len(contact),
                         max_lift_cm=float((xyz[:, 2]-xyz[0, 2]).max()*100))
    return result


try:
    for seed in [195, 196]:
        for scene, source in sources.items():
            name = f'{scene}_milk_off_seed{seed}'
            run = OUT/name
            run.mkdir()
            env.reset()
            state = np.load(source/'initial_state.npy')
            obs = env.set_init_state(state)
            assert np.array_equal(view(), np.array(Image.open(source/'input_00.png')))
            records, actions, contract = [measure()], [], []
            with imageio.get_writer(str(run/'actual.mp4'), fps=20, codec='libx264', macro_block_size=1) as writer:
                writer.append_data(view())
                for q in range(8):
                    Image.fromarray(view()).save(run/f'input_{q:02d}.png')
                    reply = requests.post('http://127.0.0.1:8921', json=dict(
                        scene=scene, noun='milk', mode='off', chunk=q, seed=seed+q, output=str(run)), timeout=300)
                    reply.raise_for_status()
                    value = reply.json()
                    assert value['use_system_prompt'] is False
                    norm = np.array(value['actions'])
                    if q == 0 and scene.endswith('original'):
                        assert np.array_equal(norm, reference[f'{scene.split("_")[0]}_milk_{seed}_off'])
                    contract.append(dict(query=q, seed=seed+q, system_on=False, und_len=value['und_len']))
                    native = np.array([_remap_gripper(_framewise_action_to_delta(a, '6d').tolist(), 'zero_one')
                                       for a in norm*scale+offset])
                    for command in native:
                        sent = np.clip(command, -1, 1)
                        obs, _, _, _ = env.step(sent.tolist())
                        records.append(measure())
                        actions.append(sent.tolist())
                        writer.append_data(view())
                    (run/'trajectory.json').write_text(json.dumps(records))
                    (run/'executed_actions.json').write_text(json.dumps(actions))
                    (run/'query_contract.json').write_text(json.dumps(contract, indent=2))
                    (OUT/'status.json').write_text(json.dumps(dict(case=name, completed_queries=q+1,
                        total_queries=8, completed_cases=len(results), total_cases=8)))
                    print('[IDENTITY-REPEAT]', name, q+1, records[-1]['grasped'], flush=True)
            results.append(dict(case=name, scene=scene, seed=seed, steps=128, per_object=quantify(records),
                source_initial_state=str(source/'initial_state.npy'), initial_pixels_exact=True,
                system_off_all_queries=True, final=records[-1]))
            (OUT/'result.json').write_text(json.dumps(dict(cases=results,
                scope='Two additional noise seeds at the SAME four simulator snapshots. Not independent tasks or initial placements.'), indent=2))
finally:
    env.close()
(OUT/'complete.json').write_text(json.dumps(dict(state='complete', cases=len(results))))
print('[IDENTITY-REPEAT] COMPLETE', flush=True)
