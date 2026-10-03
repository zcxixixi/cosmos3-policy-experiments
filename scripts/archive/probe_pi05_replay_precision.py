"""Quantify same-input pi0.5 output differences across a fresh process.

Four official predictions: cold198, repeat198, warm195, then198 again. Actual
arrays are saved before comparisons. Nonexact outputs remain evidence, not a
reason to loosen a guard or resume a failed rollout. No simulator execution.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

import probe_pi05_target_control as probe


BASE = probe.DATA/'pi05-target-control'
ORDER = [('cold198', 198), ('repeat198', 198), ('warm195', 195), ('after_warm198', 198)]
SCOPE = ('Same-input first-chunk numerical replay, not a grasping test. '
         'Differences do not by themselves establish a model bug or their cause.')


def compare(a, b):
    assert a.shape == b.shape == (10, 7)
    delta = b-a
    return dict(exact=bool(np.array_equal(a, b)), max_abs=float(np.abs(delta).max()),
        rmse=float(np.sqrt(np.mean(delta**2))), per_dim_max_abs=np.abs(delta).max(axis=0).tolist(),
        per_dim_rmse=np.sqrt(np.mean(delta**2, axis=0)).tolist(),
        xyz_rmse=float(np.sqrt(np.mean(delta[:, :3]**2))),
        gripper_sign_equal=bool(np.array_equal(np.sign(a[:, 6]), np.sign(b[:, 6]))),
        gripper_sign_changed_steps=np.flatnonzero(np.sign(a[:, 6]) != np.sign(b[:, 6])).tolist(),
        units='Native LIBERO controller commands; XYZ values are not meters.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=BASE/'replay-precision')
    parser.add_argument('--failed-chunk', type=Path, default=BASE/'closed-loop/center_milk_seed198/chunk_00')
    parser.add_argument('--predictions-dir', type=Path, default=BASE/'predictions')
    parser.add_argument('--openpi-root', type=Path, default=probe.OPENPI)
    parser.add_argument('--checkpoint', type=Path, default=probe.CHECKPOINT)
    parser.add_argument('--openpi-cache', type=Path, default=Path('/home/current/.cache/openpi'))
    args = parser.parse_args()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    failed_input = args.failed_chunk/'simulator_inputs.npz'
    failure = json.loads((args.failed_chunk/'failed.json').read_text())
    assert failure['error'] == 'q0 full 10-action chunk differs from offline seed198: center_milk_seed198'
    offline = json.loads((args.predictions_dir/'complete.json').read_text())
    assert offline['state'] == 'complete' and offline['cases'] == 12
    assert probe.digest(probe.__file__) == offline['script_sha256']
    assert str(args.checkpoint.expanduser().resolve()) == offline['checkpoint']
    for relative, sha in offline['sources'].items():
        assert probe.digest(args.openpi_root/relative) == sha, relative
    tokenizer_path = args.openpi_cache.expanduser().resolve()/'big_vision/paligemma_tokenizer.model'
    assert tokenizer_path.is_file() and probe.digest(tokenizer_path) == offline['tokenizer_sha256']
    norm_path = args.checkpoint/'assets/physical-intelligence/libero/norm_stats.json'
    assert norm_path.is_file() and probe.digest(norm_path) == offline['norm_stats_sha256']
    for path in ['params/_METADATA', 'params/manifest.ocdbt']:
        assert (args.checkpoint/path).is_file()
    references = {}
    for seed in [195, 198]:
        with np.load(args.predictions_dir/f'center_milk_{seed}/arrays.npz', allow_pickle=False) as file:
            references[seed] = {key: file[key].copy() for key in file.files}
        metadata = json.loads((args.predictions_dir/f'center_milk_{seed}/metadata.json').read_text())
        assert metadata['num_denoising_steps'] == 10
        assert metadata['prompt'] == 'pick up the milk and place it in the basket'
    with np.load(failed_input, allow_pickle=False) as file:
        source = {key: file[key].copy() for key in file.files}
    mapping = {'observation_state': 'state', 'snapshot': 'snapshot', 'eef_quat_xyzw': 'eef_quat_xyzw',
               'base_rgb_224': 'base_rgb_224', 'wrist_rgb_224': 'wrist_rgb_224'}
    for seed in [195, 198]:
        for key, input_key in mapping.items():
            assert np.array_equal(references[seed][key], source[input_key]), (seed, key)
        assert np.array_equal(references[seed]['action_noise'],
            np.random.default_rng(seed).standard_normal((10, 32), dtype=np.float32))
    assert np.array_equal(source['action_noise'], references[198]['action_noise'])
    for key in references[198]:
        if key.startswith('transformed__'):
            assert np.array_equal(references[195][key], references[198][key]), key
    output = probe.fresh_directory(args.output_dir)
    os.environ['OPENPI_DATA_HOME'] = str(args.openpi_cache.expanduser().resolve())
    sys.path.insert(0, str(args.openpi_root/'src'))
    import jax
    import sentencepiece
    from openpi.policies import policy_config
    from openpi.training import config
    assert any(device.platform == 'gpu' for device in jax.devices())
    cfg = config.get_config('pi05_libero')
    assert cfg.model.pi05 and cfg.model.action_horizon == 10 and cfg.model.action_dim == 32
    assert cfg.model.discrete_state_input is False
    packages = {}
    for name in ['jax', 'jaxlib', 'flax', 'orbax-checkpoint', 'numpy', 'torch', 'sentencepiece', 'openpi-client']:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    environment = dict(python=sys.version, executable=sys.executable,
        executable_sha256=probe.digest(sys.executable), packages=packages,
        actual_jax_config={key: jax.config.values.get(key) for key in [
            'jax_enable_x64', 'jax_default_matmul_precision', 'jax_compilation_cache_dir',
            'jax_enable_compilation_cache', 'jax_platforms', 'jax_numpy_rank_promotion',
            'jax_default_prng_impl']},
        selected_environment={key: os.environ.get(key) for key in [
            'CUDA_VISIBLE_DEVICES', 'XLA_FLAGS', 'XLA_PYTHON_CLIENT_PREALLOCATE',
            'JAX_COMPILATION_CACHE_DIR', 'JAX_ENABLE_X64', 'JAX_DEFAULT_MATMUL_PRECISION',
            'JAX_LOG_COMPILES', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'OPENPI_DATA_HOME']},
        gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'], text=True).strip(),
        checkpoint_metadata_sha256=probe.digest(args.checkpoint/'params/_METADATA'),
        checkpoint_manifest_sha256=probe.digest(args.checkpoint/'params/manifest.ocdbt'),
        norm_stats_sha256=probe.digest(norm_path), tokenizer_sha256=probe.digest(tokenizer_path),
        official_sources=offline['sources'],
        additional_sources={path: probe.digest(args.openpi_root/path) for path in [
            'src/openpi/models/model.py', 'src/openpi/models/gemma.py',
            'src/openpi/models/siglip.py', 'src/openpi/models/lora.py',
            'src/openpi/shared/nnx_utils.py', 'src/openpi/training/config.py',
            'src/openpi/training/sharding.py']})
    env_sha = hashlib.sha256(json.dumps(environment, sort_keys=True).encode()).hexdigest()
    provenance = dict(scope=SCOPE, script_sha256=probe.digest(__file__), probe_sha256=probe.digest(probe.__file__),
        environment_sha256=env_sha, environment=environment,
        failed_input_sha256=probe.digest(failed_input), failed_json_sha256=probe.digest(args.failed_chunk/'failed.json'),
        offline_complete_sha256=probe.digest(args.predictions_dir/'complete.json'),
        offline_arrays_sha256={str(seed): probe.digest(args.predictions_dir/f'center_milk_{seed}/arrays.npz') for seed in references},
        model_config='pi05_libero', checkpoint=str(args.checkpoint), num_denoising_steps=10,
        order=ORDER, actual_failed_server_output_unavailable=True, library_edits=False, weight_downloads=False)
    probe.write_json(output/'provenance.json', provenance)
    print('[PI05-PRECISION] loading existing local pi05_libero', flush=True)
    policy = policy_config.create_trained_policy(cfg, str(args.checkpoint), sample_kwargs={'num_steps': 10})
    tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(tokenizer_path))
    original_input, original_output = policy._input_transform, policy._output_transform
    captured_input, captured_output = {}, {}

    def record_input(value):
        transformed = original_input(value)
        captured_input.clear()
        captured_input.update(probe.flatten_arrays(transformed))
        return transformed

    def record_output(value):
        captured_output.clear()
        captured_output.update(probe.flatten_arrays(value))
        return original_output(value)

    policy._input_transform, policy._output_transform = record_input, record_output
    predictions, model_outputs, cases = {}, {}, {}
    started = time.perf_counter()
    try:
        for name, seed in ORDER:
            noise = source['action_noise'].copy() if seed == 198 else references[195]['action_noise'].copy()
            obs = {'observation/image': source['base_rgb_224'].copy(),
                'observation/wrist_image': source['wrist_rgb_224'].copy(),
                'observation/state': source['state'].copy(), 'prompt': 'pick up the milk and place it in the basket'}
            rng_before = policy._rng  # Immutable JAX key; only copied to CPU after prediction.
            prediction = policy.infer(obs, noise=noise.copy())
            actions = np.asarray(prediction['actions']).copy()
            transformed = {key: value.copy() for key, value in captured_input.items()}
            model_output = {key: value.copy() for key, value in captured_output.items()}
            directory = output/name
            directory.mkdir()
            # Persist the actual result before comparison, including nonexact output.
            np.savez_compressed(directory/'arrays.npz', actions_native_7=actions,
                action_noise=noise, observation_state=source['state'], snapshot=source['snapshot'],
                eef_quat_xyzw=source['eef_quat_xyzw'], base_rgb_224=source['base_rgb_224'],
                wrist_rgb_224=source['wrist_rgb_224'],
                rng_before=np.asarray(jax.random.key_data(rng_before)),
                rng_after=np.asarray(jax.random.key_data(policy._rng)),
                **{'transformed__'+key: value for key, value in transformed.items()},
                **{'model_output__'+key: value for key, value in model_output.items()})
            probe.write_json(directory/'actions.json', actions.tolist())
            assert actions.shape == (10, 7) and np.isfinite(actions).all()
            for key, value in transformed.items():
                assert np.array_equal(references[seed]['transformed__'+key], value), (name, key)
            mask = transformed['tokenized_prompt_mask'].astype(bool)
            metadata = dict(name=name, seed=seed, actual_arrays_sha256=probe.digest(directory/'arrays.npz'),
                raw_inputs_exact_offline=True, transformed_inputs_exact_offline=True,
                noise_exact_offline=True, decoded_tokens=tokenizer.decode(transformed['tokenized_prompt'][mask].tolist()),
                action_dtype=str(actions.dtype), model_output_action_dtype=str(model_output['actions'].dtype),
                vs_existing_offline=compare(references[seed]['actions_native_7'], actions))
            probe.write_json(directory/'metadata.json', metadata)
            predictions[name], model_outputs[name], cases[name] = actions, model_output['actions'], metadata
            probe.write_json(output/'status.json', dict(completed_cases=len(cases), total_cases=4, last_case=name))
            print('[PI05-PRECISION]', name, metadata['vs_existing_offline'], flush=True)
        pairs = {}
        for left, right in [('cold198', 'repeat198'), ('cold198', 'after_warm198'), ('repeat198', 'after_warm198')]:
            item = compare(predictions[left], predictions[right])
            delta = model_outputs[right]-model_outputs[left]
            item['normalized_model_output_exact'] = bool(np.array_equal(model_outputs[left], model_outputs[right]))
            item['normalized_model_output_max_abs'] = float(np.abs(delta).max())
            item['normalized_model_output_rmse'] = float(np.sqrt(np.mean(delta**2)))
            pairs[left+'_vs_'+right] = item
        probe.write_json(output/'result.json', dict(scope=SCOPE, cases=cases, comparisons=pairs,
            environment_sha256=env_sha, elapsed_s=time.perf_counter()-started,
            note='No tolerance was used to accept nonexact outputs. Cause remains unestablished.'))
        probe.write_json(output/'complete.json', dict(state='complete', cases=4,
            script_sha256=probe.digest(__file__), environment_sha256=env_sha, simulator_execution=False))
    except Exception as exc:
        probe.write_json(output/'failed.json', dict(error=str(exc), completed_cases=len(cases)))
        raise
    finally:
        policy._input_transform, policy._output_transform = original_input, original_output


if __name__ == '__main__':
    main()
