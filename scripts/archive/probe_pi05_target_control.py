"""Prepare CPU observations, then explicitly run paired pi0.5 first-chunk probes.

This file does nothing on import. ``collect`` restores existing MuJoCo snapshots
without a renderer and reuses their previously verified camera PNGs. ``predict``
loads the existing local openpi checkpoint only when explicitly invoked. It saves
12 predictions, not closed-loop executions or a benchmark success rate.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image


ROOT = Path('/home/current/work/cosmos3')
DATA = ROOT/'outputs/official-demos/milk-posttrain/gripper-context/deep-dive'
OPENPI = Path('/home/current/work/openpi-demo/openpi')
CHECKPOINT = Path('/home/current/work/openpi-demo/checkpoints/pi05_libero')
SCENES = {'center': 'object-position/inputs/center', 'far': 'milk-far/inputs/xp15'}
SCOPE = ('Two existing snapshots, milk/butter instructions, paired action noise. '
         'Native pi05_libero observations and native 7D controller actions. '
         'First-chunk predictions only; no execution, target selection, success '
         'rate, or cross-model latent equivalence is established by this probe.')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n')


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fresh_directory(path):
    path = Path(path).expanduser().resolve()
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite existing outputs: {path}')
    path.mkdir(parents=True)
    return path


def collect(args):
    # The simulator is used only for proprioception; no camera context is created.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    config_file = args.libero_config.expanduser().resolve()/'config.yaml'
    if not config_file.is_file():
        raise FileNotFoundError(f'Existing LIBERO config required: {config_file}')
    os.environ['LIBERO_CONFIG_PATH'] = str(config_file.parent)
    import yaml
    config = yaml.safe_load(config_file.read_text())
    for key in ['assets', 'bddl_files', 'init_states']:
        if not Path(config[key]).is_dir():
            raise FileNotFoundError(f'Existing LIBERO {key} required: {config[key]}')
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv

    image_source = args.openpi_root/'packages/openpi-client/src/openpi_client/image_tools.py'
    quat_source = args.openpi_root/'examples/libero/main.py'
    image_tools = load_file('pi05_probe_image_tools', image_source)
    fn = next(node for node in ast.parse(quat_source.read_text()).body
              if isinstance(node, ast.FunctionDef) and node.name == '_quat2axisangle')
    namespace = {'np': np, 'math': math}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(quat_source), 'exec'), namespace)
    quat2axisangle = namespace['_quat2axisangle']
    snapshots = {scene: np.load(args.reference_root/path/'state.npy', allow_pickle=False)
                 for scene, path in SCENES.items()}
    delta = snapshots['far']-snapshots['center']
    assert np.flatnonzero(delta).tolist() == [10]
    assert abs(float(delta[10])-.15) < 1e-15
    suite = benchmark.get_benchmark_dict()['libero_object']()
    task = suite.get_task(7)
    assert 'milk' in task.language.lower() and 'basket' in task.language.lower()
    bddl = Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
    env = ControlEnv(bddl_file_name=str(bddl), use_camera_obs=False,
                     has_renderer=False, has_offscreen_renderer=False,
                     camera_heights=256, camera_widths=256)
    manifest = {}
    try:
        env.seed(0)
        output = fresh_directory(args.output_dir)
        for scene, relative in SCENES.items():
            env.reset()
            obs = env.set_init_state(snapshots[scene])
            assert np.array_equal(env.get_sim_state(), snapshots[scene])
            reference = args.reference_root/relative
            # These are actual cameras saved for this exact snapshot, already
            # rotated 180 degrees as in the official openpi LIBERO caller.
            concat = np.asarray(Image.open(reference/'input.png').convert('RGB'))
            assert concat.shape == (256, 512, 3) and concat.dtype == np.uint8
            recorded_dir = args.reference_root/'runtime-system-prompt/closed-loop'/f'{scene}_milk_off'
            assert np.array_equal(np.load(recorded_dir/'initial_state.npy'), snapshots[scene])
            assert np.array_equal(concat, np.asarray(Image.open(recorded_dir/'input_00.png').convert('RGB')))
            recorded = json.loads((recorded_dir/'trajectory.json').read_text())[0]
            objects = {name: env.env.sim.data.get_joint_qpos(obj.joints[0])[:3].copy()
                       for name, obj in env.env.objects_dict.items()}
            differences = [float(np.max(np.abs(np.asarray(obs['robot0_eef_pos'])-recorded['eef_xyz']))),
                           float(np.max(np.abs(np.asarray(obs['robot0_gripper_qpos'])-recorded['gripper_qpos'])))]
            differences.extend(float(np.max(np.abs(position-recorded['objects'][name])))
                               for name, position in objects.items())
            assert max(differences) <= 1e-12, (scene, differences)
            quat = np.asarray(obs['robot0_eef_quat']).copy()
            state = np.concatenate((obs['robot0_eef_pos'], quat2axisangle(quat.copy()),
                                    obs['robot0_gripper_qpos']))
            assert state.shape == (8,) and np.isfinite(state).all()
            base = image_tools.resize_with_pad(np.ascontiguousarray(concat[:, :256]), 224, 224)
            wrist = image_tools.resize_with_pad(np.ascontiguousarray(concat[:, 256:]), 224, 224)
            directory = output/scene
            directory.mkdir()
            np.savez_compressed(directory/'inputs.npz', snapshot=snapshots[scene], state=state,
                                eef_quat_xyzw=quat, eef_xyz=obs['robot0_eef_pos'],
                                gripper_qpos=obs['robot0_gripper_qpos'],
                                base_rgb_224=base, wrist_rgb_224=wrist)
            Image.fromarray(concat).save(directory/'reference_concat.png')
            Image.fromarray(base).save(directory/'base_224.png')
            Image.fromarray(wrist).save(directory/'wrist_224.png')
            metadata = dict(scene=scene, task_name=task.name, task_language=task.language,
                snapshot_source=str(reference/'state.npy'), snapshot_sha256=digest(reference/'state.npy'),
                image_source=str(reference/'input.png'), image_source_sha256=digest(reference/'input.png'),
                images_are_reused_actual_frames=True, new_camera_render=False,
                restored_snapshot_exact=True, reference_proprio_max_abs=max(differences),
                state_order='eef_xyz, official quaternion-to-axis-angle(xyzw), two gripper qpos',
                observation_state=state.tolist(), observation_state_dtype=str(state.dtype),
                image_shapes=[list(base.shape), list(wrist.shape)],
                inputs_sha256=digest(directory/'inputs.npz'))
            write_json(directory/'metadata.json', metadata)
            manifest[scene] = metadata
        with np.load(output/'center/inputs.npz') as a, np.load(output/'far/inputs.npz') as b:
            assert np.array_equal(a['state'], b['state']), 'Moving only milk must preserve robot proprioception'
        write_json(output/'complete.json', dict(state='complete', scenes=manifest,
            collection='CPU MuJoCo state restoration without renderer; existing verified actual RGB frames.',
            collector_sha256=digest(__file__), image_tools_sha256=digest(image_source),
            quaternion_function_source_sha256=digest(quat_source), bddl_sha256=digest(bddl),
            libero_config=str(config_file), library_edits=False, weight_downloads=False))
    finally:
        env.close()


def flatten_arrays(value, prefix=''):
    result = {}
    for key, item in value.items():
        name = prefix+key
        if isinstance(item, dict):
            result.update(flatten_arrays(item, name+'__'))
        else:
            array = np.asarray(item)
            if array.dtype.kind in 'biuf':
                result[name] = array.copy()
    return result


def predict(args):
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    inputs_dir = args.inputs_dir.expanduser().resolve()
    inputs_meta = json.loads((inputs_dir/'complete.json').read_text())
    assert inputs_meta['state'] == 'complete' and set(inputs_meta['scenes']) == set(SCENES)
    checkpoint = args.checkpoint.expanduser().resolve()
    cache_dir = args.openpi_cache.expanduser().resolve()
    tokenizer_path = cache_dir/'big_vision/paligemma_tokenizer.model'
    # Native openpi resolves this already cached tokenizer; fail before loading
    # weights rather than allowing a missing tokenizer to trigger a download.
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f'Existing tokenizer required: {tokenizer_path}')
    os.environ['OPENPI_DATA_HOME'] = str(cache_dir)
    for path in [checkpoint/'params/_METADATA', checkpoint/'params/manifest.ocdbt',
                 checkpoint/'assets/physical-intelligence/libero/norm_stats.json']:
        if not path.is_file():
            raise FileNotFoundError(f'Local checkpoint component required: {path}')
    sys.path.insert(0, str(args.openpi_root/'src'))
    import jax
    import sentencepiece
    from openpi.policies import policy_config
    from openpi.training import config
    if not any(device.platform == 'gpu' for device in jax.devices()):
        raise RuntimeError('predict requires the existing GPU openpi environment; collect is the CPU entry')
    cfg = config.get_config('pi05_libero')
    assert cfg.model.pi05 and cfg.model.action_horizon == 10 and cfg.model.action_dim == 32
    assert cfg.model.discrete_state_input is False
    output = fresh_directory(args.output_dir)
    started = time.perf_counter()
    print('[PI05-TARGET] loading existing local pi05_libero', flush=True)
    policy = policy_config.create_trained_policy(cfg, str(checkpoint),
                                                sample_kwargs={'num_steps': args.num_steps})
    tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(tokenizer_path))
    original_transform = policy._input_transform
    captured = {}

    def observe_transform(value):
        transformed = original_transform(value)
        captured.clear()
        captured.update(flatten_arrays(transformed))
        return transformed

    policy._input_transform = observe_transform  # Capture only; return the original values unchanged.
    noises = {seed: np.random.default_rng(seed).standard_normal((10, 32), dtype=np.float32)
              for seed in args.seeds}
    results, controls, paired = {}, {}, {}
    try:
        for scene in SCENES:
            if digest(inputs_dir/scene/'inputs.npz') != inputs_meta['scenes'][scene]['inputs_sha256']:
                raise ValueError(f'Collected inputs changed: {scene}')
            with np.load(inputs_dir/scene/'inputs.npz', allow_pickle=False) as file:
                source = {key: file[key].copy() for key in file.files}
            assert source['state'].shape == (8,)
            for seed in args.seeds:
                noise = noises[seed]
                for noun in ['milk', 'butter']:
                    case = f'{scene}_{noun}_{seed}'
                    prompt = f'pick up the {noun} and place it in the basket'
                    obs = {'observation/image': source['base_rgb_224'].copy(),
                           'observation/wrist_image': source['wrist_rgb_224'].copy(),
                           'observation/state': source['state'].copy(), 'prompt': prompt}
                    before_noise = noise.copy()
                    prediction = policy.infer(obs, noise=noise.copy())
                    actions = np.asarray(prediction['actions']).copy()
                    transformed = {key: value.copy() for key, value in captured.items()}
                    assert np.array_equal(noise, before_noise)
                    assert actions.shape == (10, 7) and np.isfinite(actions).all()
                    assert 'tokenized_prompt' in transformed and 'tokenized_prompt_mask' in transformed
                    if not results:
                        replay = np.asarray(policy.infer(obs, noise=noise.copy())['actions'])
                        assert np.array_equal(actions, replay), 'Identical observations/noise failed exact replay'
                        controls['same_input_noise_replay_exact'] = True
                        assert all(np.array_equal(value, captured[key]) for key, value in transformed.items())
                    directory = output/case
                    directory.mkdir()
                    np.savez_compressed(directory/'arrays.npz', actions_native_7=actions,
                        action_noise=noise.copy(), observation_state=source['state'],
                        eef_quat_xyzw=source['eef_quat_xyzw'], snapshot=source['snapshot'],
                        base_rgb_224=source['base_rgb_224'], wrist_rgb_224=source['wrist_rgb_224'],
                        **{'transformed__'+key: value for key, value in transformed.items()})
                    write_json(directory/'actions.json', actions.tolist())
                    mask = transformed['tokenized_prompt_mask'].astype(bool)
                    tokens = transformed['tokenized_prompt']
                    metadata = dict(case=case, scene=scene, noun=noun, prompt=prompt, seed=seed,
                        action_shape=list(actions.shape), action_dtype=str(actions.dtype),
                        action_units='Native LIBERO controller commands; XYZ values are not meters.',
                        noise_shape=list(noise.shape), noise_dtype=str(noise.dtype),
                        noise_sha256=array_digest(noise), state_sha256=array_digest(source['state']),
                        token_ids=tokens.tolist(), token_mask=mask.tolist(),
                        decoded_active_tokens=tokenizer.decode(tokens[mask].tolist()),
                        inference=prediction.get('policy_timing'),
                        model_config='pi05_libero', checkpoint=str(checkpoint),
                        num_denoising_steps=args.num_steps, cosmos_action_conversion=False)
                    write_json(directory/'metadata.json', metadata)
                    results[case] = metadata
                    if noun == 'milk':
                        paired[scene, seed] = (actions, transformed)
                    else:
                        first, first_transform = paired.pop((scene, seed))
                        for key in first_transform:
                            if key not in ['tokenized_prompt', 'tokenized_prompt_mask']:
                                assert np.array_equal(first_transform[key], transformed[key]), (case, key)
                        controls[scene+'_'+str(seed)] = dict(images_state_noise_paired_exact=True,
                            token_changed_indices=np.flatnonzero(first_transform['tokenized_prompt'] != tokens).tolist(),
                            token_mask_equal=bool(np.array_equal(first_transform['tokenized_prompt_mask'], mask)),
                            native_xyz_rmse=float(np.sqrt(np.mean((first[:, :3]-actions[:, :3])**2))),
                            native_action7_rmse=float(np.sqrt(np.mean((first-actions)**2))))
                    write_json(output/'result.json', dict(scope=SCOPE, cases=results, controls=controls,
                                                         elapsed_s=time.perf_counter()-started))
                    print('[PI05-TARGET] '+case, flush=True)
        provenance = {relative: digest(args.openpi_root/relative) for relative in [
            'src/openpi/policies/policy_config.py', 'src/openpi/policies/policy.py',
            'src/openpi/policies/libero_policy.py', 'src/openpi/models/pi0.py',
            'src/openpi/models/pi0_config.py', 'src/openpi/models/tokenizer.py', 'src/openpi/transforms.py']}
        write_json(output/'complete.json', dict(state='complete', cases=len(results),
            seeds=args.seeds, action_horizon=10, padded_action_dim=32, native_action_dim=7,
            input_collection_sha256=digest(inputs_dir/'complete.json'), script_sha256=digest(__file__),
            checkpoint=str(checkpoint), norm_stats_sha256=digest(checkpoint/'assets/physical-intelligence/libero/norm_stats.json'),
            tokenizer_sha256=digest(tokenizer_path), sources=provenance,
            library_edits=False, weight_downloads=False, scope=SCOPE))
    finally:
        policy._input_transform = original_transform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    collection = commands.add_parser('collect', help='CPU snapshot/proprio extraction; reuse verified RGB')
    collection.add_argument('--output-dir', type=Path, required=True)
    collection.add_argument('--reference-root', type=Path, default=DATA)
    collection.add_argument('--openpi-root', type=Path, default=OPENPI)
    collection.add_argument('--libero-config', type=Path, default=Path('/home/current/.libero'))
    collection.set_defaults(function=collect)
    prediction = commands.add_parser('predict', help='Explicit GPU run of existing local weights')
    prediction.add_argument('--inputs-dir', type=Path, required=True)
    prediction.add_argument('--output-dir', type=Path, required=True)
    prediction.add_argument('--openpi-root', type=Path, default=OPENPI)
    prediction.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    prediction.add_argument('--openpi-cache', type=Path, default=Path('/home/current/.cache/openpi'))
    prediction.add_argument('--seeds', type=int, nargs='+', default=[195, 196, 198])
    prediction.add_argument('--num-steps', type=int, default=10)
    prediction.set_defaults(function=predict)
    args = parser.parse_args()
    if args.command == 'predict' and (args.num_steps < 1 or len(set(args.seeds)) != len(args.seeds)):
        parser.error('num-steps must be positive and seeds must be distinct')
    args.function(args)


if __name__ == '__main__':
    main()
