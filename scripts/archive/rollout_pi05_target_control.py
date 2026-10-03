"""Execute four real MuJoCo loops with native pi0.5 actions, not generated video.

Official observation preparation, horizon10/replan5, 128 simulator steps. All
snapshots, current inputs, model inputs, noise, tokens and actions are retained.
"""
import argparse
import ast
import inspect
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image

import probe_pi05_target_control as probe


BASE = probe.DATA/'pi05-target-control'
HORIZON, REPLAN, STEPS = 10, 5, 128
QUERIES = (STEPS + REPLAN - 1)//REPLAN
TESTS = [('center', 'milk'), ('center', 'butter'), ('far', 'milk'), ('far', 'butter')]
SELECTION = 'Both-finger contact AND >2cm lift for five consecutive simulator records.'
SCOPE = ('Four fixed-initial-state 128-step executions. Native pi05_libero inputs/actions, '
         'prediction horizon10 and official example replan5. Not a benchmark success rate. '
         'milk_in_basket is the milk-task checker, not butter-task success.')


def function_from_source(path, name, namespace):
    fn = next(node for node in ast.parse(path.read_text()).body
              if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=BASE/'closed-loop')
    parser.add_argument('--inputs-dir', type=Path, default=BASE/'inputs')
    parser.add_argument('--predictions-dir', type=Path, default=BASE/'predictions')
    parser.add_argument('--openpi-root', type=Path, default=probe.OPENPI)
    parser.add_argument('--url', default='http://127.0.0.1:8923')
    args = parser.parse_args()
    os.environ['MUJOCO_GL'] = 'egl'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import imageio.v2 as imageio
    import requests
    sys.path.insert(0, '/home/current/work/openpi-demo/cosmos-policy')
    from libero.libero import benchmark
    from cosmos_policy.experiments.robot.libero.libero_utils import get_libero_env

    collection = json.loads((args.inputs_dir/'complete.json').read_text())
    offline = json.loads((args.predictions_dir/'complete.json').read_text())
    assert collection['state'] == offline['state'] == 'complete' and offline['cases'] == 12
    assert probe.digest(Path(probe.__file__)) == collection['collector_sha256'] == offline['script_sha256']
    assert probe.digest(args.inputs_dir/'complete.json') == offline['input_collection_sha256']
    health_reply = requests.get(args.url+'/health', timeout=10)
    health_reply.raise_for_status()
    health = health_reply.json()
    assert health['state'] == 'ready' and health['model_config'] == 'pi05_libero'
    assert health['prediction_horizon'] == HORIZON and health['execution_replan'] == REPLAN
    assert health['queries_per_case'] == QUERIES and health['steps_per_case'] == STEPS
    assert health['num_denoising_steps'] == 10 and health['action_dim'] == 7
    assert health['probe_sha256'] == probe.digest(probe.__file__)
    assert health['script_sha256'] == probe.digest(Path(__file__).with_name('serve_pi05_target_control.py'))
    assert health['offline_complete_sha256'] == probe.digest(args.predictions_dir/'complete.json')
    images_source = args.openpi_root/'packages/openpi-client/src/openpi_client/image_tools.py'
    caller_source = args.openpi_root/'examples/libero/main.py'
    assert probe.digest(images_source) == collection['image_tools_sha256']
    assert probe.digest(caller_source) == collection['quaternion_function_source_sha256']
    image_tools = probe.load_file('pi05_rollout_image_tools', images_source)
    quat2axisangle = function_from_source(caller_source, '_quat2axisangle', {'np': np, 'math': math})
    quantify_source = probe.ROOT/'work/rollout_cosmos_pixel_schedule.py'
    quantify = function_from_source(quantify_source, 'quantify', {'np': np})
    sources, references = {}, {}
    for scene in probe.SCENES:
        assert probe.digest(args.inputs_dir/scene/'inputs.npz') == collection['scenes'][scene]['inputs_sha256']
        with np.load(args.inputs_dir/scene/'inputs.npz', allow_pickle=False) as file:
            sources[scene] = {key: file[key].copy() for key in file.files}
        for noun in ['milk', 'butter']:
            with np.load(args.predictions_dir/f'{scene}_{noun}_198/arrays.npz', allow_pickle=False) as file:
                references[scene, noun] = {key: file[key].copy() for key in file.files}
    output = probe.fresh_directory(args.output_dir)
    probe.write_json(output/'provenance.json', dict(scope=SCOPE, selection_criterion=SELECTION,
        script_sha256=probe.digest(__file__), probe_sha256=probe.digest(probe.__file__), server=health,
        collection_complete_sha256=probe.digest(args.inputs_dir/'complete.json'),
        offline_complete_sha256=probe.digest(args.predictions_dir/'complete.json'),
        official_caller_sha256=probe.digest(caller_source), image_tools_sha256=probe.digest(images_source),
        strict_quantify_source=str(quantify_source), strict_quantify_sha256=probe.digest(quantify_source),
        prediction_horizon=HORIZON, execution_replan=REPLAN, queries_per_case=QUERIES,
        steps_per_case=STEPS, num_denoising_steps=10, noise_seed_base=198,
        query_seed_rule='198 + query_index', action_conversion='identity: raw native7 sent directly as official caller.',
        external_clipping=False, gripper_remapping=False, xyz_units='Native LIBERO controller commands, not meters.',
        controller_behavior='Existing OSC scale_action clips arm input; PandaGripper uses sign, -1 open and +1 close.',
        frame_record_rule='Initial frame plus one frame after each actual simulator step.', library_edits=False))
    suite = benchmark.get_benchmark_dict()['libero_object']()
    env, _ = get_libero_env(suite.get_task(7), 'cosmos', resolution=256)
    results = []
    try:
        env.reset()
        obs = env.set_init_state(suite.get_task_init_states(7)[0])
        for _ in range(10):
            obs, _, _, _ = env.step([0., 0., 0., 0., 0., 0., -1.])
        en = env.env
        controller = en.robots[0].controller
        assert controller.name == 'OSC_POSE' and en.action_dim == 7
        assert type(en.robots[0].gripper).__name__ == 'PandaGripper'
        import robosuite.controllers.base_controller as controller_module
        import robosuite.models.grippers.panda_gripper as gripper_module
        probe.write_json(output/'simulator_contract.json', dict(
            control_freq=en.control_freq, action_dim=en.action_dim, controller_name=controller.name,
            controller_input_min=np.asarray(controller.input_min).tolist(),
            controller_input_max=np.asarray(controller.input_max).tolist(),
            controller_output_min=np.asarray(controller.output_min).tolist(),
            controller_output_max=np.asarray(controller.output_max).tolist(),
            controller_source_sha256=probe.digest(inspect.getfile(controller_module)),
            gripper_source_sha256=probe.digest(inspect.getfile(gripper_module)),
            native7_sent_without_external_clip_or_gripper_remap=True))

        def view():
            return np.concatenate([obs['agentview_image'][::-1, ::-1],
                                   obs['robot0_eye_in_hand_image'][::-1, ::-1]], axis=1).copy()

        def state8():
            quat = np.asarray(obs['robot0_eef_quat']).copy()
            value = np.concatenate((obs['robot0_eef_pos'], quat2axisangle(quat.copy()), obs['robot0_gripper_qpos']))
            assert value.shape == (8,) and np.isfinite(value).all()
            return value, quat

        def measure(step):
            gripper = en.robots[0].gripper
            contacts = {}
            for name, obj in en.objects_dict.items():
                left = bool(en.check_contact(gripper.important_geoms['left_fingerpad'], obj.contact_geoms))
                right = bool(en.check_contact(gripper.important_geoms['right_fingerpad'], obj.contact_geoms))
                both = left and right
                assert both == bool(en._check_grasp(gripper, obj)), name
                contacts[name] = dict(left=left, right=right, both=both)
            return dict(step=step, eef_xyz=obs['robot0_eef_pos'].tolist(),
                eef_quat_xyzw=obs['robot0_eef_quat'].tolist(),
                objects={name: en.sim.data.get_joint_qpos(obj.joints[0])[:3].tolist() for name, obj in en.objects_dict.items()},
                grasped=[name for name, contact in contacts.items() if contact['both']], finger_contacts=contacts,
                gripper_qpos=obs['robot0_gripper_qpos'].tolist(), milk_in_basket=bool(en._check_success()))

        for scene, noun in TESTS:
            name = f'{scene}_{noun}_seed198'
            run = output/name
            run.mkdir()
            frames = run/'frames'
            frames.mkdir()
            env.reset()
            obs = env.set_init_state(sources[scene]['snapshot'])
            assert np.array_equal(env.get_sim_state(), sources[scene]['snapshot'])
            expected = np.asarray(Image.open(args.inputs_dir/scene/'reference_concat.png').convert('RGB'))
            assert np.array_equal(view(), expected), ('initial_pixels', name)
            current_state, quat = state8()
            assert np.array_equal(current_state, sources[scene]['state']) and np.array_equal(quat, sources[scene]['eef_quat_xyzw'])
            np.save(run/'initial_state.npy', sources[scene]['snapshot'])
            records, actions, contracts, identities = [measure(0)], [], [], []
            Image.fromarray(view()).save(frames/'frame_0000.png')
            with imageio.get_writer(str(run/'actual.mp4'), fps=20, codec='libx264', macro_block_size=1) as writer:
                writer.append_data(view())
                for query in range(QUERIES):
                    chunk = run/f'chunk_{query:02d}'
                    chunk.mkdir()
                    frame = view()
                    Image.fromarray(frame).save(chunk/'input.png')
                    current_state, quat = state8()
                    base = image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(frame[:, :256]), 224, 224))
                    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(frame[:, 256:]), 224, 224))
                    noise = np.random.default_rng(198+query).standard_normal((HORIZON, 32), dtype=np.float32)
                    snapshot = env.get_sim_state().copy()
                    np.savez_compressed(chunk/'simulator_inputs.npz', state=current_state, eef_quat_xyzw=quat,
                        eef_xyz=obs['robot0_eef_pos'], gripper_qpos=obs['robot0_gripper_qpos'], snapshot=snapshot,
                        base_rgb_224=base, wrist_rgb_224=wrist, action_noise=noise)
                    Image.fromarray(base).save(chunk/'base_224.png')
                    Image.fromarray(wrist).save(chunk/'wrist_224.png')
                    if query == 0:
                        assert np.array_equal(base, sources[scene]['base_rgb_224']) and np.array_equal(wrist, sources[scene]['wrist_rgb_224'])
                    request = dict(scene=scene, noun=noun, query=query, seed=198+query,
                                   inputs_sha256=probe.digest(chunk/'simulator_inputs.npz'))
                    reply = requests.post(args.url+'/infer', json=request, timeout=300)
                    reply.raise_for_status()
                    response = reply.json()
                    native = np.asarray(response['actions'])
                    assert native.shape == (HORIZON, 7) and np.isfinite(native).all()
                    metadata = json.loads((chunk/'metadata.json').read_text())
                    assert metadata == response['metadata']
                    assert metadata['case'] == name and metadata['query'] == query and metadata['seed'] == 198+query
                    assert metadata['prediction_horizon'] == HORIZON and metadata['execution_replan'] == REPLAN
                    assert metadata['num_denoising_steps'] == 10 and metadata['native_action_conversion'] == 'identity'
                    assert metadata['input_sha256'] == request['inputs_sha256']
                    assert metadata['noise_sha256'] == probe.array_digest(noise)
                    with np.load(chunk/'model_arrays.npz', allow_pickle=False) as file:
                        assert np.array_equal(native, file['actions_native_7'])
                        assert np.array_equal(current_state, file['observation_state'])
                        assert np.array_equal(noise, file['action_noise'])
                        assert np.array_equal(base, file['base_rgb_224']) and np.array_equal(wrist, file['wrist_rgb_224'])
                    assert np.array_equal(native, np.asarray(json.loads((chunk/'actions.json').read_text())))
                    if query == 0:
                        assert metadata['first_full_chunk_matches_offline_exact'] is True
                        assert np.array_equal(native, references[scene, noun]['actions_native_7'])
                    executed_count = min(REPLAN, STEPS-len(actions))
                    contracts.append(dict(query=query, seed=198+query, simulator_step_before=len(actions),
                        predicted_actions=HORIZON, executed_actions=executed_count, request=request,
                        input_png_sha256=probe.digest(chunk/'input.png'), response_matches_saved_chunk_exact=True,
                        first_full_chunk_matches_offline_exact=metadata['first_full_chunk_matches_offline_exact'], server_metadata=metadata))
                    probe.write_json(run/'query_contract.json', contracts)
                    for index in range(executed_count):
                        command = native[index].copy()
                        sent = command.tolist()  # Official openpi caller: no external clip/remap.
                        assert np.array_equal(np.asarray(sent), command)
                        obs, _, done, _ = env.step(sent)
                        actions.append(sent)
                        identities.append(dict(step=len(actions), query=query, prediction_index=index,
                            raw_native7=command.tolist(), sent_native7=sent, conversion='identity',
                            external_clip=False, gripper_remap=False, original_task_done=bool(done)))
                        records.append(measure(len(actions)))
                        actual = view()
                        Image.fromarray(actual).save(frames/f'frame_{len(actions):04d}.png')
                        writer.append_data(actual)
                    probe.write_json(run/'trajectory.json', records)
                    probe.write_json(run/'actions.json', actions)
                    probe.write_json(run/'executed_actions.json', actions)
                    probe.write_json(run/'action_conversion.json', identities)
                    probe.write_json(output/'status.json', dict(case=name, completed_queries=query+1,
                        total_queries=QUERIES, completed_steps=len(actions), completed_cases=len(results), total_cases=4))
                    print('[PI05-ROLLOUT]', name, query+1, records[-1]['grasped'], flush=True)
            assert len(records) == 129 and len(actions) == len(identities) == STEPS
            assert len(contracts) == QUERIES and contracts[-1]['executed_actions'] == 3
            assert len(list(frames.glob('frame_*.png'))) == 129
            per_object, selected = quantify(records)
            result = dict(case=name, scene=scene, noun=noun, prompt=f'pick up the {noun} and place it in the basket',
                noise_seed_base=198, steps=STEPS, prediction_horizon=HORIZON, execution_replan=REPLAN,
                state_records=129, frame_png_count=129, queries=QUERIES, initial_pixels_state_exact=True,
                first_full_chunk_matches_offline_exact=True, native_action_conversion='identity',
                selected_objects=selected, selection_criterion=SELECTION, per_object=per_object,
                initial_state_sha256=probe.digest(run/'initial_state.npy'), final=records[-1], scope=SCOPE)
            results.append(result)
            probe.write_json(run/'complete.json', result)
            probe.write_json(output/'result.json', dict(cases=results, scope=SCOPE, selection_criterion=SELECTION))
    except Exception as exc:
        probe.write_json(output/'failed.json', dict(error=str(exc), completed_cases=len(results)))
        raise
    finally:
        env.close()
    assert len(results) == 4
    probe.write_json(output/'complete.json', dict(state='complete', cases=4, steps_per_case=STEPS,
        state_records_per_case=129, prediction_horizon=HORIZON, execution_replan=REPLAN, queries_per_case=QUERIES))
    print('[PI05-ROLLOUT] COMPLETE', flush=True)


if __name__ == '__main__':
    main()
