"""Execute real attention-route interventions, refreshing caches at every query."""
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['MUJOCO_GL'] = 'egl'
ROOT = Path('/home/current/work/cosmos3')
sys.path.insert(0, '/home/current/work/openpi-demo/cosmos-policy')
sys.path.insert(0, str(ROOT/'cosmos-framework'))
import imageio.v2 as imageio
import numpy as np
from PIL import Image
import requests
from libero.libero import benchmark
from cosmos_policy.experiments.robot.libero.libero_utils import get_libero_env
from cosmos_framework.simulation.libero.closed_loop_eval import _framewise_action_to_delta, _remap_gripper
from rollout_cosmos_pixel_schedule import quantify

DATA = ROOT/'outputs/official-demos/milk-posttrain/gripper-context/deep-dive'
OUT = Path(json.loads((ROOT/'work/language_route_isolation_process.json').read_text())['output'])
DEST = OUT/'closed-loop'
URL = 'http://127.0.0.1:8922'
REFS = {'center': DATA/'object-position/inputs/center', 'far': DATA/'milk-far/inputs/xp15'}
TESTS = [(scene, mode) for scene in ['center', 'far'] for mode in ['direct', 'indirect', 'joint']]
SCOPE = ('Six 128-step closed loops using recipient milk and donor butter at the SAME current observation. '
         'Natural-gain text attention routes and recipient-baseline visual K/V clamps. '
         'One paired seed per initial condition, two fixed snapshots. No benchmark success estimate; '
         'hybrid interventions are not normal policy inputs and do not imply additive mediation fractions.')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    done = json.loads((OUT/'offline_complete.json').read_text())
    assert done['state'] == 'complete' and done['scenes'] == ['center', 'far']
    assert not (OUT/'failed.json').exists() and not (OUT/'server_failed.json').exists()
    health = requests.get(URL, timeout=5)
    health.raise_for_status()
    assert health.text == 'ready'
    assert not DEST.exists()
    stats_path = ROOT/'cosmos-framework/cosmos_framework/data/generator/action/normalizer_stats/libero_native_frame_wise_relative_rot6d.json'
    stats = json.loads(stats_path.read_text())['global_raw']
    lo, hi = np.asarray(stats['q01']), np.asarray(stats['q99'])
    assert lo.shape == hi.shape == (10,) and np.all(hi-lo > 1e-6)
    scale, offset = (hi-lo)/2, (hi+lo)/2
    references = np.load(OUT/'actions.npz')
    initial = {scene: np.load(folder/'state.npy') for scene, folder in REFS.items()}
    DEST.mkdir(exist_ok=False)
    write_json(DEST/'provenance.json', dict(scope=SCOPE, script_sha256=sha(Path(__file__)),
        quantify_helper_sha256=sha(ROOT/'work/rollout_cosmos_pixel_schedule.py'),
        stats_sha256=sha(stats_path), offline_provenance=json.loads((OUT/'provenance.json').read_text()),
        query_seed_rule='198 + query_index', cache_rule='Recompute R and D on this query image before patching.',
        action_conversion='Unchanged official quantile inverse, rot6d to axisangle, continuous gripper remap, clip [-1,1].'))
    suite = benchmark.get_benchmark_dict()['libero_object']()
    env, _ = get_libero_env(suite.get_task(7), 'cosmos', resolution=256)
    results = []
    try:
        env.reset()
        obs = env.set_init_state(suite.get_task_init_states(7)[0])
        for _ in range(10):
            obs, _, _, _ = env.step([0., 0., 0., 0., 0., 0., -1.])
        en = env.env
        gripper = en.robots[0].gripper

        def view():
            return np.concatenate([obs['agentview_image'][::-1, ::-1],
                                   obs['robot0_eye_in_hand_image'][::-1, ::-1]], axis=1).copy()

        def measure(step):
            contacts = {}
            for name, obj in en.objects_dict.items():
                left = bool(en.check_contact(gripper.important_geoms['left_fingerpad'], obj.contact_geoms))
                right = bool(en.check_contact(gripper.important_geoms['right_fingerpad'], obj.contact_geoms))
                assert (left and right) == bool(en._check_grasp(gripper, obj))
                contacts[name] = dict(left=left, right=right, both=left and right)
            return dict(step=step, eef_xyz=obs['robot0_eef_pos'].tolist(),
                objects={name: en.sim.data.get_joint_qpos(obj.joints[0])[:3].tolist()
                         for name, obj in en.objects_dict.items()},
                grasped=[name for name, c in contacts.items() if c['both']],
                finger_contacts=contacts, gripper_qpos=obs['robot0_gripper_qpos'].tolist(),
                milk_in_basket=bool(en._check_success()))

        for scene, mode in TESTS:
            name = f'{scene}_milk_to_butter_{mode}'
            run = DEST/name
            run.mkdir(exist_ok=False)
            frames = run/'frames'
            frames.mkdir()
            env.reset()
            obs = env.set_init_state(initial[scene])
            assert np.array_equal(view(), np.asarray(Image.open(REFS[scene]/'input.png').convert('RGB')))
            np.save(run/'initial_state.npy', initial[scene])
            records = [measure(0)]
            actions, sent_actions, normalized, denormalized, contracts = [], [], [], [], []
            Image.fromarray(view()).save(frames/'frame_0000.png')
            with imageio.get_writer(str(run/'actual.mp4'), fps=20, codec='libx264', macro_block_size=1) as writer:
                writer.append_data(view())
                for query in range(8):
                    Image.fromarray(view()).save(run/f'input_{query:02d}.png')
                    request = dict(scene=scene, noun='milk', mode=mode, seed=198+query,
                                   chunk=query, output=str(run))
                    response = requests.post(URL, json=request, timeout=300)
                    response.raise_for_status()
                    value = response.json()
                    assert value['mode'] == mode and value['recipient'] == 'milk' and value['donor'] == 'butter'
                    assert value['pixels'] == value['schedule'] == 'official' and value['use_system_prompt'] is False
                    assert value['seed'] == 198+query and value['rebuilt_caches_for_current_observation']
                    assert value['current_observation_sha256'] == sha(run/f'input_{query:02d}.png')
                    assert value['timesteps'][10] == 666 and value['timesteps'][20] == 333
                    norm = np.asarray(value['actions'])
                    assert norm.shape == (16, 10) and np.isfinite(norm).all()
                    saved = np.asarray(json.loads((run/f'chunk_{query:02d}/patched/normalized_actions.json').read_text()))
                    assert np.array_equal(norm, saved)
                    if query == 0:
                        assert np.array_equal(norm, references[name]), ('offline_exact', name)
                    chunk_report = json.loads((run/f'chunk_{query:02d}/result.json').read_text())
                    assert chunk_report['routing']['complete'] and chunk_report['routing']['gain'] == 1
                    assert chunk_report['current_observation_sha256'] == value['current_observation_sha256']
                    contracts.append(dict(query=query, request=request, response={k:v for k,v in value.items() if k != 'actions'},
                                          first_chunk_offline_exact=True if query == 0 else None,
                                          patch_report=chunk_report['routing']))
                    raw = norm*scale+offset
                    native = np.asarray([_remap_gripper(_framewise_action_to_delta(a, '6d').tolist(), 'zero_one') for a in raw])
                    for index, command in enumerate(native):
                        sent = np.clip(command, -1, 1)
                        obs, _, _, _ = env.step(sent.tolist())
                        actions.append(command.tolist())
                        sent_actions.append(sent.tolist())
                        normalized.append(norm[index].tolist())
                        denormalized.append(raw[index].tolist())
                        records.append(measure(len(actions)))
                        frame = view()
                        Image.fromarray(frame).save(frames/f'frame_{len(actions):04d}.png')
                        writer.append_data(frame)
                    for file, content in [('trajectory.json', records), ('actions.json', actions),
                                          ('executed_actions.json', sent_actions), ('normalized_actions.json', normalized),
                                          ('denormalized_actions.json', denormalized), ('query_contract.json', contracts)]:
                        write_json(run/file, content)
                    write_json(DEST/'status.json', dict(case=name, completed_queries=query+1,
                        completed_cases=len(results), total_cases=len(TESTS), completed_steps=len(actions)))
                    print('[ROUTE-ROLLOUT]', name, query+1, records[-1]['grasped'], flush=True)
            assert len(records) == 129 and len(actions) == 128 and len(contracts) == 8
            per_object, selected = quantify(records)
            results.append(dict(case=name, scene=scene, mode=mode, selected_objects=selected,
                                per_object=per_object, final=records[-1], steps=128, scope=SCOPE))
            write_json(DEST/'result.json', dict(cases=results, scope=SCOPE))
    except Exception as exc:
        write_json(DEST/'failed.json', dict(error=str(exc), completed_cases=len(results)))
        raise
    finally:
        env.close()
    write_json(DEST/'complete.json', dict(state='complete', cases=len(TESTS), steps_per_case=128))
    print('[ROUTE-ROLLOUT] COMPLETE', flush=True)


if __name__ == '__main__':
    main()
