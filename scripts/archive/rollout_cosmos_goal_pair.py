"""Execute two normal 128-action Cosmos Goal trials at the same verified input.

No warmup or extra arm commands: restore the already warmed common snapshot,
reset the controller goal, verify all 63 captured items and three PNG byte
streams, then execute eight paired-seed chunks per native instruction.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np

ROOT = Path('/home/current/work/cosmos3')
INPUT = ROOT / 'outputs/official-demos/goal-pair-input'
OUT = ROOT / 'outputs/official-demos/goal-pair-policy'
DEST = OUT / 'closed-loop'
URL = 'http://127.0.0.1:8923'
STATS = ROOT / 'cosmos-framework/cosmos_framework/data/generator/action/normalizer_stats/libero_native_frame_wise_relative_rot6d.json'
SELECTION_CRITERION = 'Simultaneous left/right fingerpad contact AND >2cm lift for at least five consecutive simulator records.'
SCOPE = ('Two native Goal instructions, normal Cosmos inference, one restored Goal4/init0 snapshot. '
         'Target grasp and each task\'s native cabinet predicate are separate outcomes. '
         'A single shared initial state/seed is not a benchmark success rate or a latent mechanism test.')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_reference(metadata):
    from PIL import Image

    folder = INPUT / 'reference'
    result = dict(state={'sim_state': np.load(folder / 'state.npy', allow_pickle=False)},
                  fixture_bodies=metadata['fixture_bodies'], object_bodies=metadata['object_bodies'])
    for category in ('fixtures', 'cameras', 'observations', 'objects'):
        with np.load(folder / f'{category}.npz', allow_pickle=False) as archive:
            result[category] = {key: archive[key].copy() for key in archive.files}
    result['images'] = {key: np.asarray(Image.open(folder / f'{key}.png').convert('RGB')).copy()
                        for key in ('agentview', 'robot0_eye_in_hand', 'input')}
    return result


def controller_state(en):
    robot = en.robots[0]
    controller = robot.controller
    values = dict(goal_pos=np.array(controller.goal_pos, copy=True),
                  goal_ori=np.array(controller.goal_ori, copy=True),
                  ee_pos=np.array(controller.ee_pos, copy=True),
                  ee_ori_mat=np.array(controller.ee_ori_mat, copy=True),
                  initial_joint=np.array(controller.initial_joint, copy=True),
                  kp=np.array(controller.kp, copy=True), kd=np.array(controller.kd, copy=True),
                  interpolator_pos_none=np.array(controller.interpolator_pos is None),
                  interpolator_ori_none=np.array(controller.interpolator_ori is None),
                  gripper_current_action=np.array(robot.gripper.current_action, copy=True))
    for name, value in values.items():
        assert value.dtype.kind in 'bfiu' and np.isfinite(value).all(), name
    return values


def array_exact(a, b):
    return (a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)
            and a.tobytes(order='C') == b.tobytes(order='C'))


def quantify(records):
    """Same contact/lift criterion as the checked previous closed loops."""
    per_object, selected = {}, []
    for name in records[0]['objects']:
        xyz = np.asarray([record['objects'][name] for record in records])
        lift = xyz[:, 2] - xyz[0, 2]
        both = np.asarray([record['finger_contacts'][name]['both'] for record in records], dtype=bool)
        contact_steps = np.flatnonzero(both)
        strict = both & (lift > .02)
        first = next((index for index in range(len(strict) - 4)
                      if np.all(strict[index:index + 5])), None)
        per_object[name] = dict(first_contact_frame=int(contact_steps[0]) if contact_steps.size else None,
            contact_frames=int(contact_steps.size), first_2cm_for_5frames=first,
            max_lift_cm=float(lift.max() * 100), final_lift_cm=float(lift[-1] * 100),
            max_xy_displacement_cm=float(np.linalg.norm(xyz[:, :2] - xyz[0, :2], axis=1).max() * 100))
        if first is not None:
            selected.append(name)
    return per_object, selected


def main():
    # This script owns only this absent child directory, never the input or baseline.
    DEST.mkdir(exist_ok=False)
    env, results = None, []
    try:
        os.environ.setdefault('MUJOCO_GL', 'egl')
        os.environ['HF_HUB_OFFLINE'] = '1'
        sys.path.insert(0, '/home/current/work/openpi-demo/cosmos-policy')
        sys.path.insert(0, str(ROOT / 'cosmos-framework'))
        import imageio.v2 as imageio
        import requests
        from PIL import Image
        from libero.libero import benchmark
        from libero.libero.envs import bddl_utils
        from cosmos_policy.experiments.robot.libero.libero_utils import get_libero_env
        from cosmos_framework.simulation.libero.closed_loop_eval import _framewise_action_to_delta, _remap_gripper

        collector = load_file('goal_pair_collector', ROOT / 'work/collect_cosmos_goal_pair.py')
        service = load_file('goal_pair_service_contract', ROOT / 'work/serve_cosmos_goal_pair.py')
        metadata = service.validate_input()
        reference = load_reference(metadata)
        manifest = json.loads((INPUT / 'reference/arrays.json').read_text())
        reference_checks = []
        for category in ('state', 'fixtures', 'cameras', 'observations', 'objects', 'images'):
            assert set(reference[category]) == set(manifest[category]), category
            for key, array in reference[category].items():
                reference_checks.append(dict(category=category, array=key,
                    exact=collector.array_record(array) == manifest[category][key]))
        write_json(DEST / 'reference_manifest_checks.json', reference_checks)
        assert len(reference_checks) == 61 and all(row['exact'] for row in reference_checks)
        assert not (OUT / 'failed.json').exists()
        baseline_complete = json.loads((OUT / 'baseline_complete.json').read_text())
        assert baseline_complete['state'] == 'complete' and baseline_complete['exact_repeat_controls'] == 2
        provenance = json.loads((OUT / 'provenance.json').read_text())
        assert provenance['script_sha256'] == digest(ROOT / 'work/serve_cosmos_goal_pair.py')
        assert provenance['shared_input_metadata_sha256'] == digest(INPUT / 'metadata.json')
        assert provenance['shared_input_png_sha256'] == digest(INPUT / 'reference/input.png')
        health = requests.get(URL + '/health', timeout=5)
        health.raise_for_status()
        assert health.text == 'ready'
        stats = json.loads(STATS.read_text())['global_raw']
        lo, hi = np.asarray(stats['q01']), np.asarray(stats['q99'])
        assert lo.shape == hi.shape == (10,) and np.all(hi - lo > 1e-6)
        scale, offset = np.maximum(hi - lo, 1e-8) / 2, (hi + lo) / 2
        suite = benchmark.get_benchmark_dict()['libero_goal'](task_order_index=0)
        native_goals = {}
        definitions = []
        for task, (index, name, language, target) in service.TASKS.items():
            installed = suite.get_task(index)
            assert installed.name == name and installed.language == language, task
            path = Path(suite.get_task_bddl_file_path(index))
            row = next(row for row in metadata['tasks'] if row['index'] == index)
            assert digest(path) == row['bddl_sha256'], task
            definitions.append(collector.bddl_fields(path.read_text()))
            native_goals[task] = bddl_utils.robosuite_parse_problem(str(path))['goal_state']
            assert native_goals[task] == [['on', target, 'wooden_cabinet_1_top_side']], native_goals[task]
        for key in metadata['physical_bddl_fields_equal']:
            assert definitions[0][key] == definitions[1][key], key
        write_json(DEST / 'provenance.json', dict(script_sha256=digest(Path(__file__)),
            server_provenance=provenance, collector_sha256=digest(Path(collector.__file__)),
            stats_sha256=digest(STATS), native_goal_states=native_goals,
            pixels='official', schedule='official', use_system_prompt=False,
            noise_seed_base=198, query_seed_rule='198 + query_index', environment_bddl_task_index=4,
            warmup_steps_in_this_rollout=0, controller_rule='After restoring: update(force=True); reset_goal(). Compare initial goal_pos/goal_ori, initial_joint, kp/kd and gripper.current_action by dtype/shape/bytes across trials; require both interpolators None. initial_joint stays unchanged for the official nullspace controller. No additional env.step.',
            action_conversion='Existing quantile affine inverse; official _framewise_action_to_delta(6d); _remap_gripper(zero_one); clip seven controller commands to [-1,1] before env.step.',
            xyz_command_units='Dimensionless OSC controller inputs, not meters.',
            native_checker_rule='Evaluate each parsed native goal_state with the same environment _eval_predicate. Cabinet On does not require gripper release. Ignore bowl env done for fixed 128 actions.',
            selection_criterion=SELECTION_CRITERION, scope=SCOPE))
        env, description = get_libero_env(suite.get_task(4), 'cosmos', resolution=256)
        assert description == service.TASKS['bowl'][2]
        initial_controller = None
        for task, (index, name, language, target) in service.TASKS.items():
            run = DEST / task
            run.mkdir(exist_ok=False)
            frames = run / 'frames'
            frames.mkdir(exist_ok=False)
            env.reset()
            obs = collector.restore(env, reference)
            en = env.env
            en.robots[0].controller.update(force=True)
            en.robots[0].controller.reset_goal()
            current_controller = controller_state(en)
            np.savez_compressed(run / 'initial_controller.npz', **current_controller)
            restored = collector.capture(env, obs)
            collector.save_capture(run / 'initial_capture', restored)
            checks = collector.compare(reference, restored)
            write_json(run / 'initial_checks.json', checks)
            assert len(checks) == 63 and all(row['exact'] for row in checks), 'Restored 63-item input differs'
            png_checks = {filename: (run / 'initial_capture' / filename).read_bytes()
                          == (INPUT / 'reference' / filename).read_bytes()
                          for filename in ('agentview.png', 'robot0_eye_in_hand.png', 'input.png')}
            write_json(run / 'initial_png_checks.json', png_checks)
            assert all(png_checks.values()), 'Initial PNG bytes differ'
            controller_checks = dict(goal_pos_equals_ee_pos=array_exact(current_controller['goal_pos'], current_controller['ee_pos']),
                goal_ori_equals_ee_ori=array_exact(current_controller['goal_ori'], current_controller['ee_ori_mat']),
                interpolator_pos_is_none=bool(current_controller['interpolator_pos_none']),
                interpolator_ori_is_none=bool(current_controller['interpolator_ori_none']))
            is_controller_reference = initial_controller is None
            if is_controller_reference:
                initial_controller = current_controller
            else:
                controller_checks.update({f'paired_{key}_exact': array_exact(array, initial_controller[key])
                                          for key, array in current_controller.items()})
            write_json(run / 'initial_controller_checks.json', dict(
                role='first_trial_reference' if is_controller_reference else 'second_trial_paired_check',
                paired_check_performed=not is_controller_reference, checks=controller_checks))
            assert all(controller_checks.values()), 'Controller or gripper initial memory differs'
            gripper = en.robots[0].gripper

            def view():
                return np.concatenate([obs['agentview_image'][::-1, ::-1],
                                       obs['robot0_eye_in_hand_image'][::-1, ::-1]], axis=1).copy()

            def measure(step):
                contacts = {}
                for object_name, obj in en.objects_dict.items():
                    left = bool(en.check_contact(gripper.important_geoms['left_fingerpad'], obj.contact_geoms))
                    right = bool(en.check_contact(gripper.important_geoms['right_fingerpad'], obj.contact_geoms))
                    both = left and right
                    assert both == bool(en._check_grasp(gripper, obj)), object_name
                    contacts[object_name] = dict(left=left, right=right, both=both)
                goals = {key: bool(all(en._eval_predicate(predicate) for predicate in states))
                         for key, states in native_goals.items()}
                assert goals['bowl'] == bool(en._check_success()), 'Native bowl checker disagreement'
                return dict(step=step, objects={key: en.sim.data.get_joint_qpos(obj.joints[0])[:3].tolist()
                            for key, obj in en.objects_dict.items()}, finger_contacts=contacts,
                    grasped=[key for key, value in contacts.items() if value['both']], native_goal_checkers=goals,
                    robot_observations={key: np.asarray(value).tolist() for key, value in obs.items()
                                        if key.startswith('robot0_') and not key.endswith('_image')},
                    controller={key: value.tolist() for key, value in controller_state(en).items()})

            records, states = [measure(0)], [np.array(env.get_sim_state(), copy=True)]
            actions, clipped, normalized, denormalized, contracts = [], [], [], [], []
            Image.fromarray(view()).save(frames / 'frame_0000.png')
            with imageio.get_writer(str(run / 'actual.mp4'), fps=20, codec='libx264', macro_block_size=1) as writer:
                writer.append_data(view())
                for query in range(8):
                    Image.fromarray(view()).save(run / f'input_{query:02d}.png')
                    request = dict(task=task, query=query, seed=198 + query, prompt=language)
                    reply = requests.post(URL + '/predict', json=request, timeout=300)
                    reply.raise_for_status()
                    value = reply.json()
                    meta = value['metadata']
                    saved_meta = json.loads((run / f'chunk_{query:02d}/metadata.json').read_text())
                    assert meta == saved_meta, 'Returned/saved metadata differs'
                    assert meta['task'] == task and meta['task_index'] == index and meta['prompt'] == language
                    assert meta['seed'] == 198 + query and meta['pixels'] == meta['schedule'] == 'official'
                    assert meta['use_system_prompt'] is False and meta['domain_id'] == 5
                    assert meta['input_png_sha256'] == digest(run / f'input_{query:02d}.png')
                    baseline_meta = json.loads((OUT / f'baseline/{task}/q0/metadata.json').read_text())
                    assert meta['timesteps'] == baseline_meta['timesteps'] and meta['und_len'] == baseline_meta['und_len']
                    norm = np.asarray(value['actions'])
                    saved = np.asarray(json.loads((run / f'chunk_{query:02d}/normalized_actions.json').read_text()))
                    assert norm.shape == (16, 10) and np.isfinite(norm).all() and np.array_equal(norm, saved)
                    if query == 0:
                        baseline_actions = np.asarray(json.loads((OUT / f'baseline/{task}/q0/normalized_actions.json').read_text()))
                        assert value['q0_all_saved_tensors_exact'] is True and np.array_equal(norm, baseline_actions)
                    if task == 'wine':
                        assert value['paired_noise_checked_exact'] is True
                    contracts.append(dict(request=request, server_metadata=meta,
                        q0_all_saved_tensors_exact=value['q0_all_saved_tensors_exact'],
                        paired_noise_checked_exact=value['paired_noise_checked_exact'], response_saved_actions_exact=True))
                    write_json(run / 'query_contract.json', contracts)
                    raw = norm * scale + offset
                    native = np.asarray([_remap_gripper(_framewise_action_to_delta(array, '6d').tolist(), 'zero_one')
                                         for array in raw])
                    assert native.shape == (16, 7) and np.isfinite(native).all()
                    for row, command in enumerate(native):
                        sent = np.clip(command, -1, 1)
                        obs, _, done, _ = env.step(sent.tolist())
                        actions.append(command.tolist())
                        clipped.append(sent.tolist())
                        normalized.append(norm[row].tolist())
                        denormalized.append(raw[row].tolist())
                        record = measure(len(actions))
                        record['environment_bowl_done'] = bool(done)
                        records.append(record)
                        states.append(np.array(env.get_sim_state(), copy=True))
                        frame = view()
                        Image.fromarray(frame).save(frames / f'frame_{len(actions):04d}.png')
                        writer.append_data(frame)
                    write_json(run / 'trajectory.json', records)
                    write_json(run / 'actions.json', actions)
                    write_json(run / 'executed_actions.json', clipped)
                    write_json(run / 'normalized_actions.json', normalized)
                    write_json(run / 'denormalized_actions.json', denormalized)
                    np.save(run / 'sim_states.npy', np.stack(states))
                    write_json(DEST / 'status.json', dict(task=task, completed_queries=query + 1,
                        completed_steps=len(actions), completed_cases=len(results), total_cases=2))
                    print('[GOAL-PAIR-ROLLOUT]', task, query + 1, records[-1]['grasped'], flush=True)
            assert len(records) == len(states) == 129 and len(actions) == len(clipped) == len(normalized) == len(denormalized) == 128
            assert len(contracts) == 8 and len(list(frames.glob('frame_*.png'))) == 129
            per_object, selected = quantify(records)
            first_selection = min((value['first_2cm_for_5frames'], key) for key, value in per_object.items()
                                  if value['first_2cm_for_5frames'] is not None) if selected else None
            results.append(dict(task=task, task_index=index, task_name=name, prompt=language,
                target_object=target, selected_objects=selected, target_selected=target in selected,
                wrong_selected_objects=[key for key in selected if key != target],
                first_qualifying_object=first_selection[1] if first_selection else None,
                first_qualifying_frame=first_selection[0] if first_selection else None,
                final_native_target_goal=records[-1]['native_goal_checkers'][task],
                native_target_goal_ever=any(record['native_goal_checkers'][task] for record in records),
                per_object=per_object, final=records[-1], steps=128, state_records=129,
                initial_63_checks_exact=True, initial_3_png_bytes_exact=True,
                initial_controller_reset_goal_exact=True,
                initial_controller_role='first_trial_reference' if is_controller_reference else 'second_trial_paired_check',
                initial_controller_paired_checked_exact=None if is_controller_reference else True,
                q0_baseline_all_saved_tensors_exact=True,
                seed_rule='198 + query_index', selection_criterion=SELECTION_CRITERION, scope=SCOPE))
            write_json(DEST / 'result.json', dict(cases=results, scope=SCOPE,
                initial_controller_pair_checked_exact=True if len(results) == 2 else None,
                selection_criterion=SELECTION_CRITERION, native_goal_states=native_goals))
        assert len(results) == 2 and not (OUT / 'failed.json').exists()
        write_json(DEST / 'complete.json', dict(state='complete', cases=2, steps_per_case=128,
            state_records_per_case=129, queries_per_case=8, initial_controller_pair_checked_exact=True,
            controller_reference_trial='bowl', controller_paired_trial='wine',
            native_target_checks='Distinct native bowl/wine cabinet predicates.'))
    except Exception as exc:
        write_json(DEST / 'failed.json', dict(error=str(exc), traceback=traceback.format_exc(),
                                            completed_cases=len(results)))
        raise
    finally:
        if env is not None:
            env.close()
    print('[GOAL-PAIR-ROLLOUT] complete', flush=True)


if __name__ == '__main__':
    main()
