"""Serve the existing official pi05_libero policy for four controlled rollouts.

Explicit execution loads local weights. Importing this file does not load a model.
Every request uses saved actual observations and explicit paired FP32 noise.
"""
import argparse
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8923)
    parser.add_argument('--output-root', type=Path, default=BASE/'closed-loop')
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
    references = {}
    for scene in probe.SCENES:
        assert probe.digest(args.inputs_dir/scene/'inputs.npz') == collection['scenes'][scene]['inputs_sha256']
        for noun in ['milk', 'butter']:
            case = f'{scene}_{noun}_198'
            with np.load(args.predictions_dir/case/'arrays.npz', allow_pickle=False) as file:
                references[scene, noun] = {key: file[key].copy() for key in file.files}
            metadata = json.loads((args.predictions_dir/case/'metadata.json').read_text())
            assert metadata['num_denoising_steps'] == 10 and metadata['cosmos_action_conversion'] is False
    captured, state = {}, {'policy': None, 'failed': False, 'requests': 0}

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
                for name in ['model_arrays.npz', 'actions.json', 'metadata.json', 'failed.json']:
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
                prediction = state['policy'].infer(obs, noise=noise.copy())
                actions = np.asarray(prediction['actions']).copy()
                transformed = {key: value.copy() for key, value in captured.items()}
                assert actions.shape == (HORIZON, 7) and np.isfinite(actions).all()
                np.savez_compressed(chunk/'model_arrays.npz', actions_native_7=actions,
                    action_noise=noise, observation_state=source['state'], snapshot=source['snapshot'],
                    eef_quat_xyzw=source['eef_quat_xyzw'], base_rgb_224=source['base_rgb_224'],
                    wrist_rgb_224=source['wrist_rgb_224'],
                    **{'transformed__'+key: value for key, value in transformed.items()})
                probe.write_json(chunk/'actions.json', actions.tolist())
                first_exact = None
                if query == 0:
                    reference = references[scene, noun]
                    mapping = {'observation_state': 'state', 'action_noise': 'action_noise',
                        'snapshot': 'snapshot', 'eef_quat_xyzw': 'eef_quat_xyzw',
                        'base_rgb_224': 'base_rgb_224', 'wrist_rgb_224': 'wrist_rgb_224'}
                    for key, input_key in mapping.items():
                        assert np.array_equal(reference[key], source[input_key]), (case, key)
                    for key, value in transformed.items():
                        assert np.array_equal(reference['transformed__'+key], value), (case, key)
                    first_exact = bool(np.array_equal(actions, reference['actions_native_7']))
                    assert first_exact, f'q0 full 10-action chunk differs from offline seed198: {case}'
                tokens, mask = transformed['tokenized_prompt'], transformed['tokenized_prompt_mask'].astype(bool)
                metadata = dict(case=case, scene=scene, noun=noun, query=query, seed=seed,
                    prompt=prompt, input_sha256=probe.digest(input_path), model_arrays_sha256=probe.digest(chunk/'model_arrays.npz'),
                    noise_sha256=probe.array_digest(noise), token_ids=tokens.tolist(), token_mask=mask.tolist(),
                    decoded_active_tokens=state['tokenizer'].decode(tokens[mask].tolist()),
                    action_shape=list(actions.shape), action_dtype=str(actions.dtype),
                    prediction_horizon=HORIZON, execution_replan=REPLAN, num_denoising_steps=10,
                    first_full_chunk_matches_offline_exact=first_exact,
                    elapsed_s=time.perf_counter()-started, native_action_conversion='identity',
                    model_config='pi05_libero', checkpoint=str(args.checkpoint))
                probe.write_json(chunk/'metadata.json', metadata)
                state['requests'] += 1
                self.send_json(200, dict(actions=actions.tolist(), metadata=metadata))
                print('[PI05-SERVER]', case, query, flush=True)
            except Exception as exc:
                state['failed'] = True
                error = dict(error=str(exc), traceback=traceback.format_exc(), requests_completed=state['requests'])
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
