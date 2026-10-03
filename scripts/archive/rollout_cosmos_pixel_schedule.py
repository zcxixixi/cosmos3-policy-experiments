"""Four new closed loops using checked official pixels and integer timesteps.

Run only after the pixel/schedule offline probe has completed and its HTTP
server is ready. Preserve both fixed snapshots and all previous experiments.
"""
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
OUT = DATA/'runtime-pixel-schedule'
DEST = OUT/'closed-loop'
URL = 'http://127.0.0.1:8921'
TESTS = [('center', 'milk'), ('center', 'butter'), ('far', 'milk'), ('far', 'butter')]
REFS = {'center': DATA/'object-position/inputs/center', 'far': DATA/'milk-far/inputs/xp15'}
STATS = ROOT/'cosmos-framework/cosmos_framework/data/generator/action/normalizer_stats/libero_native_frame_wise_relative_rot6d.json'
SELECTION_CRITERION = 'Simultaneous left/right fingerpad contact AND >2cm lift for at least five consecutive simulator records.'
SCOPE = ('Four new 128-step executions at two fixed simulator snapshots, with milk/butter instructions. '
         'Official uint8 pixel path and default integer timesteps on the existing Diffusers VAE/model. '
         'Not full official-runner equivalence or a benchmark success-rate estimate. '
         'milk_in_basket is the original milk-task checker and does not measure butter-task success.')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def quantify(records):
    """Require contact and lift at every record in the same five-record window."""
    per_object, selected = {}, []
    for name in records[0]['objects']:
        xyz = np.asarray([record['objects'][name] for record in records])
        lift = xyz[:, 2] - xyz[0, 2]
        both = np.asarray([record['finger_contacts'][name]['both'] for record in records], dtype=bool)
        contact_steps = np.flatnonzero(both)
        strict = both & (lift > .02)
        first = next((i for i in range(len(strict)-4) if np.all(strict[i:i+5])), None)
        per_object[name] = dict(
            first_contact_frame=int(contact_steps[0]) if contact_steps.size else None,
            contact_frames=int(contact_steps.size),
            first_2cm_for_5frames=first,
            max_lift_cm=float(lift.max()*100),
            final_lift_cm=float(lift[-1]*100),
            max_xy_displacement_cm=float(np.linalg.norm(xyz[:, :2]-xyz[0, :2], axis=1).max()*100))
        if first is not None:
            selected.append(name)
    return per_object, selected


def main():
    complete = json.loads((OUT/'probe_complete.json').read_text())
    assert complete == {'cases': 16, 'state': 'complete'}, complete
    assert not (OUT/'failed.json').exists(), 'The offline/HTTP probe recorded a failure.'
    provenance = json.loads((OUT/'provenance.json').read_text())
    assert provenance['script_sha256'] == digest(ROOT/'work/probe_cosmos_pixel_schedule.py')
    assert provenance['pixel_helper_sha256'] == digest(ROOT/'work/check_cosmos_official_pixels_cpu.py')
    health = requests.get(URL+'/health', timeout=5)
    health.raise_for_status()
    assert health.text == 'ready', health.text
    assert not DEST.exists(), f'Refusing to overwrite existing output: {DEST}'

    initial = {scene: np.load(directory/'state.npy') for scene, directory in REFS.items()}
    delta = initial['far'] - initial['center']
    assert np.flatnonzero(delta).tolist() == [10]
    assert abs(delta[10]-.15) < 1e-15
    references, offline_metadata = {}, {}
    with np.load(OUT/'actions.npz') as probe:
        for scene, noun in TESTS:
            key = f'{scene}_{noun}_198_official_official'
            references[scene, noun] = probe[key].copy()
            metadata = json.loads((OUT/'cases'/key/'metadata.json').read_text())
            assert metadata['case'] == key
            assert metadata['scene'] == scene and metadata['noun'] == noun and metadata['seed'] == 198
            assert metadata['pixels'] == metadata['schedule'] == 'official'
            assert metadata['use_system_prompt'] is False
            assert len(metadata['timesteps']) == 30
            assert all(type(t) is int for t in metadata['timesteps'])
            assert references[scene, noun].shape == (16, 10)
            offline_metadata[scene, noun] = metadata

    stats = json.loads(STATS.read_text())['global_raw']
    lo, hi = np.asarray(stats['q01']), np.asarray(stats['q99'])
    assert lo.shape == hi.shape == (10,)
    assert np.all(hi-lo > 1e-6)  # Existing 1e-8 and official 1e-6 floors have identical effect here.
    scale, offset = np.maximum(hi-lo, 1e-8)/2, (hi+lo)/2
    DEST.mkdir(exist_ok=False)
    write_json(DEST/'provenance.json', dict(
        scope=SCOPE, selection_criterion=SELECTION_CRITERION,
        script_sha256=digest(Path(__file__)), offline_probe_provenance=provenance,
        offline_actions_sha256=digest(OUT/'actions.npz'), stats_sha256=digest(STATS),
        pixels='official', schedule='official', use_system_prompt=False,
        noise_seed_base=198, query_seed_rule='198 + query_index',
        action_conversion='Same quantile affine inverse and official _framewise_action_to_delta(6d), _remap_gripper(zero_one) as previous rollouts.',
        xyz_command_units='LIBERO controller commands; controller XYZ values are not meters.',
        clipping='np.clip(command, -1, 1) immediately before env.step.',
        library_edits=False, frame_record_rule='Initial frame plus one frame after every simulator step.'))

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
                both = left and right
                assert both == bool(en._check_grasp(gripper, obj)), name
                contacts[name] = dict(left=left, right=right, both=both)
            return dict(step=step, eef_xyz=obs['robot0_eef_pos'].tolist(),
                objects={name: en.sim.data.get_joint_qpos(obj.joints[0])[:3].tolist()
                         for name, obj in en.objects_dict.items()},
                grasped=[name for name, contact in contacts.items() if contact['both']],
                finger_contacts=contacts, gripper_qpos=obs['robot0_gripper_qpos'].tolist(),
                milk_in_basket=bool(en._check_success()))

        for scene, noun in TESTS:
            name = f'{scene}_{noun}_official_official_seed198'
            run = DEST/name
            run.mkdir(exist_ok=False)
            frames = run/'frames'
            frames.mkdir(exist_ok=False)
            env.reset()
            obs = env.set_init_state(initial[scene])
            expected_pixels = np.asarray(Image.open(REFS[scene]/'input.png').convert('RGB'))
            assert np.array_equal(view(), expected_pixels), ('initial_pixels', name)
            np.save(run/'initial_state.npy', initial[scene])
            records = [measure(0)]
            actions, clipped, normalized, denormalized, contract = [], [], [], [], []
            Image.fromarray(view()).save(frames/'frame_0000.png')
            with imageio.get_writer(str(run/'actual.mp4'), fps=20, codec='libx264', macro_block_size=1) as writer:
                writer.append_data(view())
                for query in range(8):
                    Image.fromarray(view()).save(run/f'input_{query:02d}.png')
                    request = dict(pixels='official', schedule='official', scene=scene, noun=noun,
                                   seed=198+query, chunk=query, output=str(run))
                    reply = requests.post(URL, json=request, timeout=300)
                    reply.raise_for_status()
                    value = reply.json()
                    assert value['pixels'] == value['schedule'] == 'official'
                    assert value['use_system_prompt'] is False
                    assert value['und_len'] == offline_metadata[scene, noun]['und_len']
                    assert value['timesteps'] == offline_metadata[scene, noun]['timesteps']
                    assert all(type(t) is int for t in value['timesteps'])
                    metadata = json.loads((run/f'chunk_{query:02d}'/'metadata.json').read_text())
                    expected_case = f'{scene}_{noun}_{198+query}_official_official'
                    assert metadata['case'] == expected_case
                    assert metadata['scene'] == scene and metadata['noun'] == noun
                    assert metadata['seed'] == 198+query
                    for key in ['pixels', 'schedule', 'use_system_prompt', 'und_len', 'timesteps']:
                        assert metadata[key] == value[key], (name, query, key)
                    norm = np.asarray(value['actions'])
                    assert norm.shape == (16, 10) and np.isfinite(norm).all()
                    saved_norm = np.asarray(json.loads((run/f'chunk_{query:02d}'/'normalized_actions.json').read_text()))
                    assert np.array_equal(norm, saved_norm), ('saved_actions', name, query)
                    q0_exact = bool(np.array_equal(norm, references[scene, noun])) if query == 0 else None
                    if query == 0:
                        assert q0_exact, ('first_chunk_offline_exact', name)
                    contract.append(dict(query=query, seed=198+query, request=request,
                        pixels=value['pixels'], schedule=value['schedule'], use_system_prompt=value['use_system_prompt'],
                        system_on=False, und_len=value['und_len'], timesteps=value['timesteps'],
                        timesteps_match_offline_exact=True, first_chunk_matches_offline_probe_exact=q0_exact,
                        input_png_sha256=digest(run/f'input_{query:02d}.png'),
                        response_matches_saved_chunk_exact=True, server_metadata=metadata))
                    write_json(run/'query_contract.json', contract)
                    raw = norm*scale+offset
                    native = np.asarray([_remap_gripper(_framewise_action_to_delta(a, '6d').tolist(), 'zero_one')
                                         for a in raw])
                    assert native.shape == (16, 7) and np.isfinite(native).all()
                    for index, command in enumerate(native):
                        sent = np.clip(command, -1, 1)
                        obs, _, _, _ = env.step(sent.tolist())
                        actions.append(command.tolist())
                        clipped.append(sent.tolist())
                        normalized.append(norm[index].tolist())
                        denormalized.append(raw[index].tolist())
                        records.append(measure(len(actions)))
                        frame = view()
                        Image.fromarray(frame).save(frames/f'frame_{len(actions):04d}.png')
                        writer.append_data(frame)
                    write_json(run/'trajectory.json', records)
                    write_json(run/'actions.json', actions)
                    write_json(run/'executed_actions.json', clipped)
                    write_json(run/'normalized_actions.json', normalized)
                    write_json(run/'denormalized_actions.json', denormalized)
                    write_json(DEST/'status.json', dict(case=name, completed_queries=query+1,
                        total_queries=8, completed_steps=len(actions), completed_cases=len(results), total_cases=4))
                    print('[PIXEL-SCHEDULE-ROLLOUT]', name, query+1, records[-1]['grasped'], flush=True)
            assert len(records) == 129 and len(actions) == len(clipped) == len(normalized) == len(denormalized) == 128
            assert len(contract) == 8 and len(list(frames.glob('frame_*.png'))) == 129
            per_object, selected = quantify(records)
            results.append(dict(case=name, scene=scene, noun=noun, noise_seed_base=198,
                prompt=f'pick up the {noun} and place it in the basket', pixels='official', schedule='official',
                use_system_prompt=False, steps=128, state_records=129, frame_png_count=129,
                initial_pixels_exact=True, first_chunk_matches_offline_probe_exact=True,
                timesteps_match_offline_all_queries_exact=True,
                initial_state_source=str(REFS[scene]/'state.npy'),
                initial_state_sha256=digest(run/'initial_state.npy'),
                source_initial_state_sha256=digest(REFS[scene]/'state.npy'),
                source_input_png_sha256=digest(REFS[scene]/'input.png'),
                selected_objects=selected, selection_criterion=SELECTION_CRITERION,
                per_object=per_object, final=records[-1], scope=SCOPE))
            write_json(DEST/'result.json', dict(cases=results, scope=SCOPE, selection_criterion=SELECTION_CRITERION))
    except Exception as exc:
        write_json(DEST/'failed.json', dict(error=str(exc), completed_cases=len(results)))
        raise
    finally:
        env.close()
    assert len(results) == 4
    write_json(DEST/'complete.json', dict(state='complete', cases=4, steps_per_case=128,
                                        state_records_per_case=129, queries_per_case=8))
    print('[PIXEL-SCHEDULE-ROLLOUT] COMPLETE', flush=True)


if __name__ == '__main__':
    main()
