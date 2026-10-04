"""Collect matched warmed milk scenes, then execute explicitly selected trials.

The existing center snapshot is already warmed. No env.step is permitted in
collection or restoration. The server owns model records under chunk_XX; this
script owns simulator inputs, actual actions, contacts, states and videos.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np


ROOT = Path('/home/current/work/cosmos3')
SOURCE = ROOT / 'outputs/official-demos/milk-posttrain/gripper-context/deep-dive/object-position/inputs/center'
STATS = ROOT / 'cosmos-framework/cosmos_framework/data/generator/action/normalizer_stats/libero_native_frame_wise_relative_rot6d.json'
PROMPT = 'pick up the milk and place it in the basket'
SHIFTS = {'x00': 0, 'x06': 6, 'x15': 15}
SEEDS = (195, 196, 198)
TRIALS = {f'{scene}_seed{seed}': (scene, seed) for scene in SHIFTS for seed in SEEDS}
URL = 'http://127.0.0.1:8925'
CRITERION = 'Simultaneous left/right fingerpad contact AND >2cm lift in each of five consecutive control-step records.'
SCOPE = 'Nine matched 128-action milk trials at X+0/6/15cm, three paired noise seeds, existing warmed snapshot and fixed current model. No benchmark success-rate claim.'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def capture(env, obs, collector):
    """Goal collector's fields, with the actual Object-state shape preserved."""
    en, sim = env.env, env.sim
    state = np.array(env.get_sim_state(), copy=True)
    assert state.ndim == 1 and state.dtype == np.float64 and np.isfinite(state).all()
    fixtures, fixture_bodies = {}, {name: obj.root_body for name, obj in en.fixtures_dict.items()}
    for name, body in fixture_bodies.items():
        body_id = sim.model.body_name2id(body)
        fixtures[name + '__body_pos'] = sim.model.body_pos[body_id].copy()
        fixtures[name + '__body_quat'] = sim.model.body_quat[body_id].copy()
    ids = [sim.model.camera_name2id(name) for name in collector.CAMERAS]
    cameras = {field: np.array(getattr(sim.model, field)[ids], copy=True)
               for field in collector.CAMERA_FIELDS}
    cameras['world_xpos'] = sim.data.cam_xpos[ids].copy()
    cameras['world_xmat'] = sim.data.cam_xmat[ids].copy()
    objects, object_bodies = {}, {}
    for name, obj in en.objects_dict.items():
        object_bodies[name] = obj.root_body
        body_id = sim.model.body_name2id(obj.root_body)
        objects[name + '__world_pos'] = sim.data.body_xpos[body_id].copy()
        objects[name + '__world_quat_wxyz'] = sim.data.body_xquat[body_id].copy()
        for joint in obj.joints:
            objects[name + '__' + joint + '__qpos'] = np.array(sim.data.get_joint_qpos(joint), copy=True)
            objects[name + '__' + joint + '__qvel'] = np.array(sim.data.get_joint_qvel(joint), copy=True)
    observations = collector.copy_observations(obs)
    images = {name: observations[name + '_image'][::-1, ::-1].copy() for name in collector.CAMERAS}
    images['input'] = np.concatenate([images[name] for name in collector.CAMERAS], axis=1)
    for category in (fixtures, cameras, objects):
        for key, value in category.items():
            assert value.dtype.kind in 'buifc' and np.isfinite(value).all(), key
    return dict(state={'sim_state': state}, fixtures=fixtures, cameras=cameras,
                objects=objects, observations=observations, images=images,
                fixture_bodies=fixture_bodies, object_bodies=object_bodies)


def load_capture(folder, metadata, collector):
    from PIL import Image
    result = dict(state={'sim_state': np.load(folder / 'state.npy', allow_pickle=False)},
                  fixture_bodies=metadata['fixture_bodies'], object_bodies=metadata['object_bodies'])
    for category in ('fixtures', 'cameras', 'observations', 'objects'):
        with np.load(folder / f'{category}.npz', allow_pickle=False) as archive:
            result[category] = {key: archive[key].copy() for key in archive.files}
    result['images'] = {name: np.asarray(Image.open(folder / f'{name}.png').convert('RGB')).copy()
                        for name in (*collector.CAMERAS, 'input')}
    manifest = json.loads((folder / 'arrays.json').read_text())
    for category in ('state', 'fixtures', 'cameras', 'observations', 'objects', 'images'):
        assert set(result[category]) == set(manifest[category]), category
        for key, value in result[category].items():
            assert collector.array_record(value) == manifest[category][key], (folder.name, category, key)
    return result


def reset_controller(en, rollout, expected=None):
    controller = en.robots[0].controller
    controller.update(force=True)
    controller.reset_goal()
    values = rollout.controller_state(en)
    checks = dict(goal_pos_equals_ee_pos=rollout.array_exact(values['goal_pos'], values['ee_pos']),
                  goal_ori_equals_ee_ori=rollout.array_exact(values['goal_ori'], values['ee_ori_mat']),
                  interpolator_pos_none=bool(values['interpolator_pos_none']),
                  interpolator_ori_none=bool(values['interpolator_ori_none']))
    if expected is not None:
        assert values.keys() == expected.keys()
        checks.update({f'paired_{key}_exact': rollout.array_exact(value, expected[key])
                       for key, value in values.items()})
    assert all(checks.values()), checks
    return values, checks


def contacts(en):
    owners = {}
    for kind, objects in (('object', en.objects_dict), ('fixture', en.fixtures_dict)):
        for name, obj in objects.items():
            for geom in obj.contact_geoms:
                owners[en.sim.model.geom_name2id(geom)] = (kind, name)
    robot = en.robots[0]
    for part in (robot.robot_model, robot.gripper):
        for geom in part.contact_geoms:
            owners[en.sim.model.geom_name2id(geom)] = ('robot', geom)
    rows, pairs = [], set()
    for contact in en.sim.data.contact[:en.sim.data.ncon]:
        a, b = int(contact.geom1), int(contact.geom2)
        left, right = owners.get(a, ('other', str(en.sim.model.geom_id2name(a)))), owners.get(b, ('other', str(en.sim.model.geom_id2name(b))))
        rows.append(dict(geom1=a, geom2=b, owner1=list(left), owner2=list(right),
                         distance_m=float(contact.dist), position=np.asarray(contact.pos).tolist()))
        if left[0] == right[0] == 'object' and left[1] != right[1]:
            pairs.add(tuple(sorted((left[1], right[1]))))
    return rows, [list(pair) for pair in sorted(pairs)]


def initial_contact_checks(en):
    rows, pairs = contacts(en)
    assert not pairs, ('Initial inter-object contact', pairs)
    milk_robot = [row for row in rows if any(owner == ['object', 'milk_1'] for owner in (row['owner1'], row['owner2']))
                  and any(owner[0] == 'robot' for owner in (row['owner1'], row['owner2']))]
    assert not milk_robot, ('Initial milk/robot contact', milk_robot)
    return dict(interobject_pairs=pairs, milk_robot_contacts=milk_robot, all_contacts=rows)


def make_env():
    from libero.libero import benchmark
    from cosmos_policy.experiments.robot.libero.libero_utils import get_libero_env
    suite = benchmark.get_benchmark_dict()['libero_object'](task_order_index=0)
    env, description = get_libero_env(suite.get_task(7), 'cosmos', resolution=256)
    assert description == PROMPT, description
    return env, suite


def collect_inputs(output, collector, rollout):
    from PIL import Image
    output.mkdir(parents=True, exist_ok=True)
    destination = output / 'inputs'
    destination.mkdir(exist_ok=False)
    env = None
    try:
        base = np.load(SOURCE / 'state.npy', allow_pickle=False)
        original = np.asarray(Image.open(SOURCE / 'input.png').convert('RGB'))
        env, suite = make_env()
        env.reset()
        obs = env.set_init_state(base)
        reference = capture(env, obs, collector)
        assert rollout.array_exact(reference['state']['sim_state'], base)
        assert np.array_equal(reference['images']['input'], original), 'Original center PNG pixels changed'
        metadata = dict(suite='libero_object', task_order_index=0, task_index=7, init_index=0,
            prompt=PROMPT, source_state=str(SOURCE / 'state.npy'), source_state_sha256=digest(SOURCE / 'state.npy'),
            source_png=str(SOURCE / 'input.png'), source_png_sha256=digest(SOURCE / 'input.png'),
            source_is_already_warmed=True, additional_warmup_steps=0, environment_seed=0,
            state_shape=list(base.shape), state_dtype=str(base.dtype),
            fixture_bodies=reference['fixture_bodies'], object_bodies=reference['object_bodies'],
            camera_order=list(collector.CAMERAS), resolution_per_camera=[256, 256],
            image_rule='Rotate each RGB by 180 degrees; concatenate agentview then wrist.',
            script_sha256=digest(Path(__file__)), collector_sha256=digest(Path(collector.__file__)),
            controller_helper_sha256=digest(Path(rollout.__file__)), shifts_cm=SHIFTS, seeds=list(SEEDS),
            selection_criterion=CRITERION, controller_rule='update(force=True); reset_goal(); preserve initial_joint; compare all captured controller/gripper fields exactly.',
            scenes={})
        expected_controller = None
        for scene, shift_cm in SHIFTS.items():
            env.reset()
            obs = collector.restore(env, reference)
            before = capture(env, obs, collector)
            base_checks = collector.compare(reference, before)
            assert all(row['exact'] for row in base_checks), 'Reset did not restore center exactly'
            en = env.env
            milk = en.objects_dict['milk_1']
            assert len(milk.joints) == 1
            joint = milk.joints[0]
            original_qpos = np.array(en.sim.data.get_joint_qpos(joint), copy=True)
            assert original_qpos.shape == (7,)
            address = en.sim.model.get_joint_qpos_addr(joint)
            start = int(address[0] if isinstance(address, tuple) else address)
            target = original_qpos.copy()
            target[0] += shift_cm / 100
            en.sim.data.set_joint_qpos(joint, target)
            en.sim.forward()
            shifted = np.array(env.get_sim_state(), copy=True)
            changed = np.flatnonzero(shifted != base).tolist()
            assert changed == ([] if shift_cm == 0 else [1 + start]), (scene, changed, start)
            assert np.array_equal(np.delete(shifted, 1 + start), np.delete(base, 1 + start))
            assert np.array_equal(en.sim.data.get_joint_qpos(joint), target)
            obs = env.set_init_state(shifted)
            controller, controller_checks = reset_controller(env.env, rollout, expected_controller)
            if expected_controller is None:
                expected_controller = controller
            after = capture(env, obs, collector)
            assert rollout.array_exact(after['state']['sim_state'], shifted)
            contact_checks = initial_contact_checks(env.env)
            folder = destination / scene
            collector.save_capture(folder, after)
            np.savez_compressed(folder / 'controller.npz', **controller)
            write_json(folder / 'center_restore_checks.json', base_checks)
            write_json(folder / 'controller_checks.json', controller_checks)
            write_json(folder / 'initial_contact_checks.json', contact_checks)
            metadata['scenes'][scene] = dict(shift_cm=shift_cm, milk_joint=joint, milk_qpos_address=start,
                changed_state_indexes=changed, milk_qpos=target.tolist(),
                input_png_sha256=digest(folder / 'input.png'), state_sha256=digest(folder / 'state.npy'),
                arrays_sha256=digest(folder / 'arrays.json'), controller_sha256=digest(folder / 'controller.npz'))
            # A second reset checks the complete shifted input and controller without stepping.
            env.reset()
            restored_obs = collector.restore(env, after)
            _, paired_controller = reset_controller(env.env, rollout, controller)
            restored = capture(env, restored_obs, collector)
            restore_checks = collector.compare(after, restored)
            write_json(folder / 'restore_checks.json', restore_checks)
            write_json(folder / 'restored_controller_checks.json', paired_controller)
            assert all(row['exact'] for row in restore_checks), (scene, 'Shifted restore changed')
            assert np.array_equal(restored['images']['input'], after['images']['input'])
            write_json(destination / 'metadata.json', metadata)
        assert np.array_equal(np.asarray(Image.open(destination / 'x00/input.png').convert('RGB')), original)
        write_json(destination / 'complete.json', dict(state='matched_inputs_verified', scenes=list(SHIFTS),
            shifts_cm=list(SHIFTS.values()), additional_warmup_steps=0, policy_executed=False,
            source_center_pixels_exact=True, metadata_sha256=digest(destination / 'metadata.json')))
    except Exception as exc:
        write_json(destination / 'failed.json', dict(type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc()))
        raise
    finally:
        if env is not None:
            env.close()


def measure(env, obs, step, rollout):
    en = env.env
    gripper = en.robots[0].gripper
    fingers, object_qpos, object_qvel = {}, {}, {}
    for name, obj in en.objects_dict.items():
        left = bool(en.check_contact(gripper.important_geoms['left_fingerpad'], obj.contact_geoms))
        right = bool(en.check_contact(gripper.important_geoms['right_fingerpad'], obj.contact_geoms))
        assert (left and right) == bool(en._check_grasp(gripper, obj)), name
        fingers[name] = dict(left=left, right=right, both=left and right)
        assert len(obj.joints) == 1, name
        object_qpos[name] = np.asarray(en.sim.data.get_joint_qpos(obj.joints[0])).tolist()
        object_qvel[name] = np.asarray(en.sim.data.get_joint_qvel(obj.joints[0])).tolist()
    contact_rows, pairs = contacts(en)
    values = dict(step=step, sim_time=float(en.sim.data.time), eef_xyz=obs['robot0_eef_pos'].tolist(),
        objects={name: value[:3] for name, value in object_qpos.items()},
        object_qpos=object_qpos, object_qvel=object_qvel, finger_contacts=fingers,
        grasped=[name for name, item in fingers.items() if item['both']],
        object_contact_pairs=pairs, all_contacts=contact_rows, milk_in_basket=bool(en._check_success()),
        observations={name: np.asarray(value).tolist() for name, value in obs.items() if not name.endswith('_image')},
        controller={name: value.tolist() for name, value in rollout.controller_state(en).items()})
    return values


def summarize_trial(trial, scene, seed, records, rollout):
    per_object, selected = rollout.quantify(records)
    first = min(((item['first_2cm_for_5frames'], name) for name, item in per_object.items()
                 if item['first_2cm_for_5frames'] is not None), default=None)
    return dict(trial=trial, scene=scene, shift_cm=SHIFTS[scene], noise_seed_base=seed,
        seed_rule='noise_seed_base + query_index', prompt=PROMPT, steps=128, queries=8,
        state_records=129, selected_objects=selected, first_selected_object=None if first is None else first[1],
        first_selection_step=None if first is None else first[0], per_object=per_object,
        final=records[-1], initial_controller_paired_exact=True, additional_warmup_steps=0,
        selection_criterion=CRITERION, scope=SCOPE)


def execute_trials(output, selected_trials, collector, rollout):
    import imageio.v2 as imageio
    from PIL import Image
    import requests
    from cosmos_framework.simulation.libero.closed_loop_eval import _framewise_action_to_delta, _remap_gripper

    inputs = output / 'inputs'
    complete = json.loads((inputs / 'complete.json').read_text())
    assert complete['state'] == 'matched_inputs_verified' and complete['additional_warmup_steps'] == 0
    assert not (inputs / 'failed.json').exists()
    assert complete['metadata_sha256'] == digest(inputs / 'metadata.json')
    metadata = json.loads((inputs / 'metadata.json').read_text())
    assert metadata['script_sha256'] == digest(Path(__file__)), 'Collection script changed'
    assert metadata['collector_sha256'] == digest(Path(collector.__file__))
    assert metadata['controller_helper_sha256'] == digest(Path(rollout.__file__))
    assert metadata['prompt'] == PROMPT and metadata['shifts_cm'] == SHIFTS
    references, controllers = {}, {}
    for scene in SHIFTS:
        folder = inputs / scene
        references[scene] = load_capture(folder, metadata, collector)
        assert metadata['scenes'][scene]['arrays_sha256'] == digest(folder / 'arrays.json')
        assert metadata['scenes'][scene]['controller_sha256'] == digest(folder / 'controller.npz')
        with np.load(folder / 'controller.npz', allow_pickle=False) as archive:
            controllers[scene] = {key: archive[key].copy() for key in archive.files}
    health = requests.get(URL + '/health', timeout=10)
    health.raise_for_status()
    assert health.text == 'ready'
    stats = json.loads(STATS.read_text())['global_raw']
    lo, hi = np.asarray(stats['q01']), np.asarray(stats['q99'])
    assert lo.shape == hi.shape == (10,) and np.all(hi - lo > 1e-6)
    scale, offset = np.maximum(hi - lo, 1e-8) / 2, (hi + lo) / 2
    destination = output / 'closed-loop'
    provenance = dict(script_sha256=digest(Path(__file__)), inputs_metadata_sha256=digest(inputs / 'metadata.json'),
        stats_sha256=digest(STATS), prompt=PROMPT, pixels='official', schedule='official',
        use_system_prompt=False, queries_per_trial=8, actions_per_query=16, steps_per_trial=128,
        seeds=list(SEEDS), shifts_cm=SHIFTS, additional_warmup_steps=0, selection_criterion=CRITERION,
        action_conversion='Quantile affine inverse; _framewise_action_to_delta(6d); _remap_gripper(zero_one); clip [-1,1].',
        scope=SCOPE)
    if destination.exists():
        assert not (destination / 'failed.json').exists(), 'Prior failed batch requires review; no automatic retry'
        assert json.loads((destination / 'provenance.json').read_text()) == provenance
    else:
        destination.mkdir(exist_ok=False)
        write_json(destination / 'provenance.json', provenance)
    for trial in selected_trials:
        assert not (destination / trial).exists(), f'Refusing existing trial: {trial}'
    results = []
    if (destination / 'summary.json').exists():
        results = json.loads((destination / 'summary.json').read_text())['cases']
    env, _ = make_env()
    trial, schedule_reference, seen_noise = None, None, set()
    try:
        for trial in selected_trials:
            scene, seed = TRIALS[trial]
            run = destination / trial
            run.mkdir(exist_ok=False)
            env.reset()
            obs = collector.restore(env, references[scene])
            controller, controller_checks = reset_controller(env.env, rollout, controllers[scene])
            np.savez_compressed(run / 'initial_controller.npz', **controller)
            initial = capture(env, obs, collector)
            checks = collector.compare(references[scene], initial)
            write_json(run / 'initial_checks.json', checks)
            write_json(run / 'initial_controller_checks.json', controller_checks)
            assert all(row['exact'] for row in checks), ('Initial restoration', trial)
            write_json(run / 'initial_contact_checks.json', initial_contact_checks(env.env))
            records = [measure(env, obs, 0, rollout)]
            states = [np.array(env.get_sim_state(), copy=True)]
            actions, executed, normalized, denormalized, contracts = [], [], [], [], []
            with imageio.get_writer(str(run / 'actual.mp4'), fps=20, codec='libx264', macro_block_size=1) as writer:
                writer.append_data(initial['images']['input'])
                for query in range(8):
                    pixels = np.concatenate([obs[name + '_image'][::-1, ::-1] for name in collector.CAMERAS], axis=1).copy()
                    image_path = run / f'input_{query:02d}.png'
                    Image.fromarray(pixels).save(image_path)
                    if query == 0:
                        assert image_path.read_bytes() == (inputs / scene / 'input.png').read_bytes(), 'Initial PNG bytes changed'
                    request = dict(trial=trial, query=query, seed=seed + query, prompt=PROMPT)
                    reply = requests.post(URL + '/predict', json=request, timeout=600)
                    reply.raise_for_status()
                    value = reply.json()
                    meta = value['metadata']
                    chunk = run / f'chunk_{query:02d}'
                    saved_meta = json.loads((chunk / 'metadata.json').read_text())
                    assert meta == saved_meta
                    for key, expected in dict(trial=trial, query=query, shift_cm=SHIFTS[scene], seed=seed + query,
                        prompt=PROMPT, pixels='official', schedule='official', use_system_prompt=False,
                        domain_name='libero', domain_id=5, raw_action_dim=10, chunk_size=16,
                        num_inference_steps=30, model_calls=30, prepare_calls=1).items():
                        assert meta[key] == expected, (trial, query, key, meta.get(key), expected)
                    assert meta['input_png_sha256'] == digest(image_path)
                    schedule = meta['timesteps']
                    assert len(schedule) == 30 and all(type(item) is int for item in schedule)
                    assert schedule[10] == 666 and schedule[20] == 333
                    if schedule_reference is None:
                        schedule_reference = schedule
                    assert schedule == schedule_reference
                    assert type(value['paired_noise_checked_exact']) is bool
                    if (seed, query) in seen_noise:
                        assert value['paired_noise_checked_exact'] is True
                    seen_noise.add((seed, query))
                    norm = np.asarray(value['actions'])
                    saved = np.asarray(json.loads((chunk / 'normalized_actions.json').read_text()))
                    assert norm.shape == (16, 10) and np.isfinite(norm).all() and np.array_equal(norm, saved)
                    assert (chunk / 'states.pt').is_file(), 'Model arrays missing'
                    contracts.append(dict(request=request, server_metadata=meta,
                        paired_noise_checked_exact=value['paired_noise_checked_exact'], response_saved_actions_exact=True))
                    raw = norm * scale + offset
                    native = np.asarray([_remap_gripper(_framewise_action_to_delta(item, '6d').tolist(), 'zero_one') for item in raw])
                    assert native.shape == (16, 7) and np.isfinite(native).all()
                    for index, command in enumerate(native):
                        sent = np.clip(command, -1, 1)
                        obs, _, done, _ = env.step(sent.tolist())
                        actions.append(command.tolist())
                        executed.append(sent.tolist())
                        normalized.append(norm[index].tolist())
                        denormalized.append(raw[index].tolist())
                        record = measure(env, obs, len(actions), rollout)
                        record['environment_done'] = bool(done)
                        records.append(record)
                        states.append(np.array(env.get_sim_state(), copy=True))
                        writer.append_data(np.concatenate([obs[name + '_image'][::-1, ::-1] for name in collector.CAMERAS], axis=1).copy())
                    for name, content in (('trajectory.json', records), ('actions.json', actions),
                        ('executed_actions.json', executed), ('normalized_actions.json', normalized),
                        ('denormalized_actions.json', denormalized), ('query_contract.json', contracts)):
                        write_json(run / name, content)
                    np.save(run / 'sim_states.npy', np.stack(states))
                    status = dict(trial=trial, completed_queries=query + 1, completed_steps=len(actions),
                                  completed_cases=len(results), requested_trials=selected_trials)
                    write_json(destination / 'status.json', status)
                    write_json(output / 'status.json', dict(stage='closed_loop', **status))
                    print('[MILK-SHIFT]', trial, query + 1, records[-1]['grasped'], flush=True)
            assert len(records) == len(states) == 129
            assert len(actions) == len(executed) == len(normalized) == len(denormalized) == 128
            assert len(contracts) == len(list(run.glob('input_*.png'))) == 8
            summary = summarize_trial(trial, scene, seed, records, rollout)
            write_json(run / 'summary.json', summary)
            results.append(summary)
            batch_summary = dict(cases=results, scope=SCOPE, selection_criterion=CRITERION,
                                 completed_trials=[item['trial'] for item in results], default_trials=list(TRIALS))
            write_json(destination / 'summary.json', batch_summary)
            write_json(output / 'summary.json', batch_summary)
    except Exception as exc:
        failure = dict(trial=trial, type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc(),
                       completed_trials=[item['trial'] for item in results])
        write_json(destination / 'failed.json', failure)
        write_json(output / 'simulator_failed.json', failure)
        raise
    finally:
        env.close()
    completed = dict(state='complete' if len(results) == len(TRIALS) else 'selected_trials_complete',
        cases=len(results), completed_trials=[item['trial'] for item in results], steps_per_case=128,
        queries_per_case=8, state_records_per_case=129, additional_warmup_steps=0)
    write_json(destination / 'complete.json', completed)
    write_json(output / 'simulator_complete.json', completed)
    print('[MILK-SHIFT] COMPLETE', json.dumps(completed), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--collect-only', action='store_true')
    parser.add_argument('--trials', nargs='+', choices=list(TRIALS), help='Explicit subset of the nine trials; no overwrite or retry.')
    args = parser.parse_args()
    assert not (args.collect_only and args.trials), '--trials applies only to execution'
    selected = list(TRIALS) if args.trials is None else args.trials
    assert len(selected) == len(set(selected)), 'Duplicate trials'
    os.environ.setdefault('MUJOCO_GL', 'egl')
    sys.path.insert(0, '/home/current/work/openpi-demo/cosmos-policy')
    sys.path.insert(0, str(ROOT / 'cosmos-framework'))
    collector = load_file('milk_shift_goal_collector_helpers', ROOT / 'work/collect_cosmos_goal_pair.py')
    rollout = load_file('milk_shift_goal_rollout_helpers', ROOT / 'work/rollout_cosmos_goal_pair.py')
    if args.collect_only:
        collect_inputs(args.output, collector, rollout)
    else:
        execute_trials(args.output, selected, collector, rollout)


if __name__ == '__main__':
    main()
