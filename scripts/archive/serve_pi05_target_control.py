"""Serve the existing official pi05_libero policy for four controlled rollouts.

Explicit execution loads local weights. Importing this file does not load a model.
Every request uses saved actual observations and explicit paired FP32 noise.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

import probe_pi05_target_control as probe


BASE = probe.DATA/'pi05-target-control'
HORIZON, REPLAN, STEPS = 10, 5, 128
QUERIES = (STEPS + REPLAN - 1)//REPLAN


def exact_array(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return bool(a.shape == b.shape and a.dtype == b.dtype and
                np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes())


def compare_actions(a, b):
    assert a.shape == b.shape == (HORIZON, 7)
    delta = b-a
    return dict(exact=exact_array(a, b), max_abs=float(np.abs(delta).max()),
        rmse=float(np.sqrt(np.mean(delta**2))), per_dim_max_abs=np.abs(delta).max(axis=0).tolist(),
        per_dim_rmse=np.sqrt(np.mean(delta**2, axis=0)).tolist(),
        xyz_rmse=float(np.sqrt(np.mean(delta[:, :3]**2))),
        gripper_sign_equal=bool(np.array_equal(np.sign(a[:, 6]), np.sign(b[:, 6]))),
        gripper_sign_changed_steps=np.flatnonzero(np.sign(a[:, 6]) != np.sign(b[:, 6])).tolist(),
        units='Native LIBERO controller commands; XYZ values are not meters.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8923)
    parser.add_argument('--output-root', type=Path, default=BASE/'closed-loop-service-paired')
    parser.add_argument('--inputs-dir', type=Path, default=BASE/'inputs')
    parser.add_argument('--predictions-dir', type=Path, default=BASE/'predictions')
    parser.add_argument('--openpi-root', type=Path, default=probe.OPENPI)
    parser.add_argument('--checkpoint', type=Path, default=probe.CHECKPOINT)
    parser.add_argument('--openpi-cache', type=Path, default=Path('/home/current/.cache/openpi'))
    args = parser.parse_args()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    args.output_root = args.output_root.expanduser().resolve()
    assert not args.output_root.exists(), 'New rollout output root must not exist at server startup'
    collection = json.loads((args.inputs_dir/'complete.json').read_text())
    offline = json.loads((args.predictions_dir/'complete.json').read_text())
    assert collection['state'] == offline['state'] == 'complete'
    assert offline['cases'] == 12 and offline['native_action_dim'] == 7
    assert probe.digest(Path(probe.__file__)) == collection['collector_sha256'] == offline['script_sha256']
    assert probe.digest(args.inputs_dir/'complete.json') == offline['input_collection_sha256']
    assert str(args.checkpoint.expanduser().resolve()) == offline['checkpoint']
    for relative, sha in offline['sources'].items():
        assert probe.digest(args.openpi_root/relative) == sha, relative
    tokenizer_path = args.openpi_cache.expanduser().resolve()/'big_vision/paligemma_tokenizer.model'
    assert tokenizer_path.is_file() and probe.digest(tokenizer_path) == offline['tokenizer_sha256']
    norm_path = args.checkpoint/'assets/physical-intelligence/libero/norm_stats.json'
    assert norm_path.is_file() and probe.digest(norm_path) == offline['norm_stats_sha256']
    for path in ['params/_METADATA', 'params/manifest.ocdbt']:
        assert (args.checkpoint/path).is_file()
    os.environ['OPENPI_DATA_HOME'] = str(args.openpi_cache.expanduser().resolve())
    references, reference_metadata = {}, {}
    for scene in probe.SCENES:
        assert probe.digest(args.inputs_dir/scene/'inputs.npz') == collection['scenes'][scene]['inputs_sha256']
        for noun in ['milk', 'butter']:
            case = f'{scene}_{noun}_198'
            with np.load(args.predictions_dir/case/'arrays.npz', allow_pickle=False) as file:
                references[scene, noun] = {key: file[key].copy() for key in file.files}
            metadata = json.loads((args.predictions_dir/case/'metadata.json').read_text())
            assert metadata['num_denoising_steps'] == 10 and metadata['cosmos_action_conversion'] is False
            assert metadata['prompt'] == f'pick up the {noun} and place it in the basket'
            reference_metadata[scene, noun] = metadata
    captured, state = {}, {'policy': None, 'failed': False, 'requests': 0, 'inferences': 0}

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, code, value):
            body = json.dumps(value, allow_nan=False).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != '/health':
                self.send_json(404, {'error': 'Unknown endpoint'})
                return
            self.send_json(200, dict(state='failed' if state['failed'] else 'ready',
                model_config='pi05_libero', checkpoint=str(args.checkpoint),
                prediction_horizon=HORIZON, execution_replan=REPLAN, num_denoising_steps=10,
                queries_per_case=QUERIES, steps_per_case=STEPS, action_dim=7,
                script_sha256=probe.digest(__file__), probe_sha256=probe.digest(probe.__file__),
                offline_complete_sha256=probe.digest(args.predictions_dir/'complete.json'),
                requests_completed=state['requests'], actual_jax_config=state['actual_jax_config'],
                inferences_completed=state['inferences'], runtime_provenance=state['runtime_provenance'],
                q0_acceptance='Bitwise actual vs canonical replay within this loaded service; old offline difference is measured.',
                library_edits=False, weight_downloads=False))

        def do_POST(self):
            chunk = None
            try:
                assert self.path == '/infer' and not state['failed']
                request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                scene, noun, query = request['scene'], request['noun'], request['query']
                assert scene in probe.SCENES and noun in ['milk', 'butter']
                assert type(query) is int and 0 <= query < QUERIES
                seed = 198 + query
                assert request['seed'] == seed
                case = f'{scene}_{noun}_seed198'
                chunk = args.output_root/case/f'chunk_{query:02d}'
                input_path = chunk/'simulator_inputs.npz'
                assert input_path.is_file() and probe.digest(input_path) == request['inputs_sha256']
                for name in ['model_arrays.npz', 'actions.json', 'metadata.json', 'failed.json',
                             'canonical_arrays.npz', 'canonical_actions.json', 'q0_pairing.json']:
                    assert not (chunk/name).exists(), f'Refusing existing model output: {chunk/name}'
                with np.load(input_path, allow_pickle=False) as file:
                    source = {key: file[key].copy() for key in file.files}
                assert source['state'].shape == (8,) and np.isfinite(source['state']).all()
                for name in ['base_rgb_224', 'wrist_rgb_224']:
                    assert source[name].shape == (224, 224, 3) and source[name].dtype == np.uint8
                noise = source['action_noise']
                assert noise.shape == (HORIZON, 32) and noise.dtype == np.float32
                assert np.array_equal(noise, np.random.default_rng(seed).standard_normal((HORIZON, 32), dtype=np.float32))
                prompt = f'pick up the {noun} and place it in the basket'
                obs = {'observation/image': source['base_rgb_224'].copy(),
                    'observation/wrist_image': source['wrist_rgb_224'].copy(),
                    'observation/state': source['state'].copy(), 'prompt': prompt}
                started = time.perf_counter()
                rng_before_actual = state['policy']._rng
                prediction = state['policy'].infer(obs, noise=noise.copy())
                state['inferences'] += 1
                rng_after_actual = state['policy']._rng
                actions = np.asarray(prediction['actions']).copy()
                transformed = {key: value.copy() for key, value in captured.items()}
                assert actions.shape == (HORIZON, 7) and np.isfinite(actions).all()
                np.savez_compressed(chunk/'model_arrays.npz', actions_native_7=actions,
                    action_noise=noise, observation_state=source['state'], snapshot=source['snapshot'],
                    eef_quat_xyzw=source['eef_quat_xyzw'], base_rgb_224=source['base_rgb_224'],
                    wrist_rgb_224=source['wrist_rgb_224'],
                    rng_before=np.asarray(jax.random.key_data(rng_before_actual)),
                    rng_after=np.asarray(jax.random.key_data(rng_after_actual)),
                    **{'transformed__'+key: value for key, value in transformed.items()})
                probe.write_json(chunk/'actions.json', actions.tolist())
                first_exact, first_service_exact, historical_comparison, pairing = None, None, None, None
                if query == 0:
                    reference = references[scene, noun]
                    mapping = {'observation_state': 'state', 'action_noise': 'action_noise',
                        'snapshot': 'snapshot', 'eef_quat_xyzw': 'eef_quat_xyzw',
                        'base_rgb_224': 'base_rgb_224', 'wrist_rgb_224': 'wrist_rgb_224'}
                    for key, input_key in mapping.items():
                        assert exact_array(reference[key], source[input_key]), (case, key)
                    reference_transformed = {key[len('transformed__'):]: value for key, value in reference.items()
                                             if key.startswith('transformed__')}
                    assert set(transformed) == set(reference_transformed), (case, 'transform_keys')
                    for key, value in transformed.items():
                        assert exact_array(reference_transformed[key], value), (case, key)
                    historical_comparison = compare_actions(reference['actions_native_7'], actions)
                    first_exact = historical_comparison['exact']
                    canonical_prompt = reference_metadata[scene, noun]['prompt']
                    assert prompt == canonical_prompt
                    canonical_obs = {'observation/image': reference['base_rgb_224'].copy(),
                        'observation/wrist_image': reference['wrist_rgb_224'].copy(),
                        'observation/state': reference['observation_state'].copy(), 'prompt': canonical_prompt}
                    # The first actual output/transforms are already copied and saved. This call clears captured.
                    rng_before_canonical = state['policy']._rng
                    canonical_prediction = state['policy'].infer(canonical_obs, noise=reference['action_noise'].copy())
                    state['inferences'] += 1
                    rng_after_canonical = state['policy']._rng
                    canonical_actions = np.asarray(canonical_prediction['actions']).copy()
                    canonical_transformed = {key: value.copy() for key, value in captured.items()}
                    np.savez_compressed(chunk/'canonical_arrays.npz', actions_native_7=canonical_actions,
                        action_noise=reference['action_noise'], observation_state=reference['observation_state'],
                        snapshot=reference['snapshot'], eef_quat_xyzw=reference['eef_quat_xyzw'],
                        base_rgb_224=reference['base_rgb_224'], wrist_rgb_224=reference['wrist_rgb_224'],
                        rng_before=np.asarray(jax.random.key_data(rng_before_canonical)),
                        rng_after=np.asarray(jax.random.key_data(rng_after_canonical)),
                        **{'transformed__'+key: value for key, value in canonical_transformed.items()})
                    probe.write_json(chunk/'canonical_actions.json', canonical_actions.tolist())
                    assert canonical_actions.shape == (HORIZON, 7) and np.isfinite(canonical_actions).all()
                    canonical_keys_exact = set(canonical_transformed) == set(reference_transformed)
                    canonical_old_transform_exact = canonical_keys_exact and all(
                        exact_array(canonical_transformed[key], value) for key, value in reference_transformed.items())
                    canonical_actual_transform_exact = canonical_keys_exact and all(
                        exact_array(canonical_transformed[key], value) for key, value in transformed.items())
                    first_service_exact = exact_array(actions, canonical_actions)
                    pairing = dict(actual_raw_and_transformed_exact_old_baseline=True,
                        canonical_prompt=canonical_prompt, actual_prompt=prompt,
                        canonical_transformed_exact_old_baseline=canonical_old_transform_exact,
                        canonical_transformed_exact_actual=canonical_actual_transform_exact,
                        first_within_service_replay_exact=first_service_exact,
                        comparison_rule='Equal shape, dtype and C-order bytes; no tolerance.',
                        historical_cross_process_comparison=historical_comparison,
                        canonical_vs_old_offline=compare_actions(reference['actions_native_7'], canonical_actions),
                        actual_vs_canonical=compare_actions(actions, canonical_actions),
                        actual_arrays_sha256=probe.digest(chunk/'model_arrays.npz'),
                        canonical_arrays_sha256=probe.digest(chunk/'canonical_arrays.npz'),
                        old_arrays_sha256=probe.digest(args.predictions_dir/f'{scene}_{noun}_198/arrays.npz'),
                        old_metadata_sha256=probe.digest(args.predictions_dir/f'{scene}_{noun}_198/metadata.json'),
                        policy_infer_calls_for_query=2, extra_q0_infer=True,
                        rng_before_actual=np.asarray(jax.random.key_data(rng_before_actual)).tolist(),
                        rng_after_actual=np.asarray(jax.random.key_data(rng_after_actual)).tolist(),
                        rng_before_canonical=np.asarray(jax.random.key_data(rng_before_canonical)).tolist(),
                        rng_after_canonical=np.asarray(jax.random.key_data(rng_after_canonical)).tolist(),
                        rng_note='Canonical infer splits the policy RNG once more. Both calls receive explicit saved noise. '
                                 'This extra call changes later RNG state; no claim of behavior neutrality or harmless historical drift.',
                        returned_and_executed_output='First actual prediction, never canonical.')
                    probe.write_json(chunk/'q0_pairing.json', pairing)
                    assert canonical_old_transform_exact and canonical_actual_transform_exact, (case, 'canonical_transform')
                    assert first_service_exact, f'q0 actual vs canonical service replay is not bitwise exact: {case}'
                tokens, mask = transformed['tokenized_prompt'], transformed['tokenized_prompt_mask'].astype(bool)
                metadata = dict(case=case, scene=scene, noun=noun, query=query, seed=seed,
                    prompt=prompt, input_sha256=probe.digest(input_path), model_arrays_sha256=probe.digest(chunk/'model_arrays.npz'),
                    noise_sha256=probe.array_digest(noise), token_ids=tokens.tolist(), token_mask=mask.tolist(),
                    decoded_active_tokens=state['tokenizer'].decode(tokens[mask].tolist()),
                    action_shape=list(actions.shape), action_dtype=str(actions.dtype),
                    prediction_horizon=HORIZON, execution_replan=REPLAN, num_denoising_steps=10,
                    first_full_chunk_matches_offline_exact=first_exact,
                    historical_cross_process_comparison=historical_comparison,
                    first_within_service_replay_exact=first_service_exact,
                    policy_infer_calls_for_query=2 if query == 0 else 1,
                    extra_q0_infer=query == 0,
                    q0_pairing_sha256=probe.digest(chunk/'q0_pairing.json') if query == 0 else None,
                    elapsed_s=time.perf_counter()-started, native_action_conversion='identity',
                    model_config='pi05_libero', checkpoint=str(args.checkpoint))
                probe.write_json(chunk/'metadata.json', metadata)
                state['requests'] += 1
                self.send_json(200, dict(actions=actions.tolist(), metadata=metadata))
                print('[PI05-SERVER]', case, query, flush=True)
            except Exception as exc:
                state['failed'] = True
                error = dict(error=str(exc), traceback=traceback.format_exc(), requests_completed=state['requests'],
                             inferences_completed=state['inferences'])
                if chunk is not None and chunk.is_dir() and not (chunk/'failed.json').exists():
                    probe.write_json(chunk/'failed.json', error)
                print(error['traceback'], flush=True)
                self.send_json(500, error)

    # Bind before loading weights so a port conflict cannot waste a model load.
    server = HTTPServer(('127.0.0.1', args.port), Handler)
    try:
        sys.path.insert(0, str(args.openpi_root/'src'))
        import jax
        import sentencepiece
        from openpi.policies import policy_config
        from openpi.training import config
        assert any(device.platform == 'gpu' for device in jax.devices())
        cfg = config.get_config('pi05_libero')
        assert cfg.model.pi05 and cfg.model.action_horizon == HORIZON and cfg.model.action_dim == 32
        assert cfg.model.discrete_state_input is False
        print('[PI05-SERVER] loading existing local pi05_libero', flush=True)
        policy = policy_config.create_trained_policy(cfg, str(args.checkpoint), sample_kwargs={'num_steps': 10})
        state['actual_jax_config'] = {key: jax.config.values.get(key) for key in [
            'jax_enable_x64', 'jax_default_matmul_precision', 'jax_compilation_cache_dir',
            'jax_enable_compilation_cache', 'jax_platforms', 'jax_numpy_rank_promotion',
            'jax_default_prng_impl']}
        packages = {}
        for name in ['jax', 'jaxlib', 'flax', 'orbax-checkpoint', 'numpy', 'sentencepiece', 'openpi-client']:
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        runtime = dict(python=sys.version, executable=sys.executable, packages=packages,
            actual_jax_config=state['actual_jax_config'],
            selected_environment={key: os.environ.get(key) for key in [
                'CUDA_VISIBLE_DEVICES', 'XLA_FLAGS', 'XLA_PYTHON_CLIENT_PREALLOCATE',
                'JAX_COMPILATION_CACHE_DIR', 'JAX_ENABLE_X64', 'JAX_DEFAULT_MATMUL_PRECISION',
                'JAX_LOG_COMPILES', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'OPENPI_DATA_HOME']},
            official_sources=offline['sources'],
            additional_sources={path: probe.digest(args.openpi_root/path) for path in [
                'src/openpi/models/model.py', 'src/openpi/models/gemma.py', 'src/openpi/models/siglip.py',
                'src/openpi/models/lora.py', 'src/openpi/shared/nnx_utils.py',
                'src/openpi/training/config.py', 'src/openpi/training/sharding.py']},
            checkpoint_metadata_sha256=probe.digest(args.checkpoint/'params/_METADATA'),
            checkpoint_manifest_sha256=probe.digest(args.checkpoint/'params/manifest.ocdbt'),
            norm_stats_sha256=probe.digest(norm_path), tokenizer_sha256=probe.digest(tokenizer_path))
        state['runtime_provenance'] = dict(environment=runtime,
            environment_sha256=hashlib.sha256(json.dumps(runtime, sort_keys=True).encode()).hexdigest())
        original_transform = policy._input_transform

        def observe(value):
            transformed = original_transform(value)
            captured.clear()
            captured.update(probe.flatten_arrays(transformed))
            return transformed

        policy._input_transform = observe
        state['policy'] = policy
        state['tokenizer'] = sentencepiece.SentencePieceProcessor(model_file=str(tokenizer_path))
        print('[PI05-SERVER] READY', args.port, flush=True)
        try:
            server.serve_forever()
        finally:
            policy._input_transform = original_transform
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
