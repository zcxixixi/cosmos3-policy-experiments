"""Test saved native full-GEN attention masks; never call a model or physics.

Use --output BASE. Only create absent BASE/current-mask-dispatch-output.
For each saved seed195/196/198 L17/t29 input, call the original official
dispatch exactly five times: native, all-allowed bool mask, and current
queries excluding future keys, action keys, or both. Keep complete shapes,
strides and native kwargs. Record numerical discrepancies without repairing
them. These are attention arithmetic results, not propagated model outputs.
"""

import argparse
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback


ROOT = Path('/home/current/work/cosmos3')
SEEDS = (195, 196, 198)
READER_SHA = 'a71604115ee797cef97662f29c9f76e536d84f7e5515d1a725526c0a0d08d2eb'
READER_COMPLETE_SHA = 'd657c7f560eba0bf8e96f0a49d208b23b87e5dbe16b313cd8b42b96c70892913'
READER_RESULTS_SHA = '19dca4a76998e8b9212c3c286ef17f354525bf8a97c758656c18831b805128e8'
DISPATCH_SHA = '93a57f07a77f8710233458ce8adc300e3eeaa57cb09a46a94cee02121d38ed88'
ARMS = ('native', 'all_allowed', 'cut_future', 'cut_action', 'cut_both')
SCOPE = ('Saved L17/t29 full-GEN official attention arithmetic only. Fifteen '
         'attention dispatches, zero model forwards, zero solver steps and zero '
         'physics. No latent identity decoding, downstream action effect or '
         'grasp conclusion. Noncurrent GEN rows and saved UND outputs are '
         'retained in merged arrays; no merged array is sent into a model.')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def raw(value, torch):
    return value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def exact(left, right, torch):
    return left.shape == right.shape and left.dtype == right.dtype and raw(left, torch) == raw(right, torch)


def meta(value, torch):
    return dict(shape=list(value.shape), dtype=str(value.dtype), sha256=hashlib.sha256(raw(value, torch)).hexdigest())


def discrepancy(left, right, torch):
    delta = left.detach().cpu().double() - right.detach().cpu().double()
    return dict(byte_exact=exact(left, right, torch), delta_l2=float(delta.norm()),
                delta_maxabs=float(delta.abs().max()), changed_values=int(torch.count_nonzero(delta)))


def audit_sources(baseline):
    folder = baseline / 'target-reader-inputs'
    assert not (folder / 'failed.json').exists()
    complete, results, provenance = (read(folder / (name + '.json')) for name in ('complete', 'results', 'provenance'))
    assert sha(folder / 'complete.json') == READER_COMPLETE_SHA
    assert sha(folder / 'results.json') == READER_RESULTS_SHA
    assert complete['state'] == results['state'] == 'complete'
    for name in ('results', 'provenance', 'protocol', 'sources'):
        assert complete[name + '_sha256'] == sha(folder / (name + '.json'))
    assert complete['script_sha256'] == provenance['script_sha256'] == READER_SHA
    assert sha(Path(__file__).resolve().parent / 'probe_cosmos_milk_target_reader_inputs.py') == READER_SHA
    assert complete['native_model_forwards'] == 90 and complete['extra_official_full_GEN_attention_replays'] == 3
    assert complete['all_original_entire_q0_records_bit_exact'] is True
    assert complete['all_full_GEN_native_dispatch_replays_bit_exact'] is True
    assert complete['original_dispatch_symbol_restored'] is complete['all_owned_hooks_removed'] is True
    assert provenance['dispatch_source_sha256'] == DISPATCH_SHA
    assert sha(Path(provenance['dispatch_source'])) == DISPATCH_SHA
    rows = {row['seed']: row for row in results['cases']}
    assert set(rows) == set(SEEDS)
    checked = {}
    names = ('dispatch-t29.pt', 'dispatch-t29.json', 'token-indexes.pt',
             'token-indexes.json', 'observer-report.json')
    for seed in SEEDS:
        row = rows[seed]
        assert row['original_entire_q0_record_bit_exact'] is row['official_full_GEN_replay_bit_exact'] is True
        assert Path(row['folder']).resolve() == (folder / f'seed{seed}').resolve()
        checked[seed] = {name: sha(folder / f'seed{seed}' / name) for name in names}
        assert all(checked[seed][name] == row['files_sha256'][name] for name in names)
    return folder, provenance, dict(reader_documents_sha256={name: sha(folder / (name + '.json'))
        for name in ('complete', 'results', 'provenance', 'protocol', 'sources')},
        reader_case_files_rehashed=checked,
        historical_whole_q0_exact_gate='Anchored completed producer; no new whole-q0 tensor replay in this script.')


def indexes_and_masks(indexes, q, k, torch):
    assert indexes['und_len'] == 121 and indexes['gen_len'] == 266 and indexes['full_joint_length'] == 387
    current = indexes['current_vision_GEN_rows'].flatten().long()
    text = indexes['text_global_key_indexes'].flatten().long()
    current_keys = indexes['current_vision_global_key_indexes'].flatten().long()
    future = indexes['future_vision_global_key_indexes'].flatten().long()
    action = indexes['action_global_indexes'].flatten().long()
    assert tuple(q.shape) == (1, 266, 32, 128) and tuple(k.shape) == (1, 387, 8, 128)
    assert [item.numel() for item in (text, current_keys, future, action)] == [121, 50, 200, 16]
    assert current.numel() == 50 and torch.equal(current + 121, current_keys)
    assert torch.equal(torch.cat((text, current_keys, future, action)).sort().values, torch.arange(387))
    assert int(current.min()) >= 0 and int(current.max()) < 266
    assert torch.equal(indexes['action_GEN_query_rows'].flatten().long() + 121, action)
    noncurrent = torch.tensor([i for i in range(266) if i not in set(current.tolist())], dtype=torch.long)
    masks = {}
    for arm, blocked in (('all_allowed', torch.empty(0, dtype=torch.long)),
                         ('cut_future', future), ('cut_action', action), ('cut_both', torch.cat((future, action)))):
        mask = torch.ones((1, 1, 266, 387), dtype=torch.bool)
        if blocked.numel():
            mask[0, 0, current[:, None], blocked[None, :]] = False
        assert int((~mask).sum()) == 50 * blocked.numel()
        assert bool(mask[0, 0, noncurrent].all()) and bool(mask[0, 0, :, text].all())
        assert bool(mask[0, 0, :, current_keys].all()) and bool(mask.any(dim=-1).all())
        masks[arm] = mask
    return current, noncurrent, masks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Completed baseline experiment root')
    baseline = parser.parse_args().output.resolve()
    source, reader_provenance, source_audit = audit_sources(baseline)
    out = baseline / 'current-mask-dispatch-output'
    out.mkdir(exist_ok=False)
    completed, calls = [], 0
    started = time.perf_counter()
    protocol = dict(seeds=list(SEEDS), layer_zero_based=17, denoising_step=29, arms=list(ARMS),
        planned_official_dispatch_calls=15, model_forwards=0, solver_steps=0, physics_calls=0,
        mask_shape=[1, 1, 266, 387], mask_dtype='torch.bool', bool_true_means_allowed=True,
        noncurrent_queries_preserved_by_native_output_row_selection=True,
        numerical_error_policy='Native replay must be byte exact. All-allowed discrepancies are saved, never residual-corrected. Cut signal exceeds sham floor only if its current-row L2 is >10 times the all-allowed current-row discrepancy and nonzero.',
        scope=SCOPE)
    write(out / 'protocol.json', protocol)
    try:
        os.environ['HF_HUB_OFFLINE'] = '1'
        sys.path.insert(0, str(ROOT / 'diffusers/src'))
        import torch
        dispatch_module = importlib.import_module('diffusers.models.attention_dispatch')
        dispatch = dispatch_module.dispatch_attention_fn
        dispatch_file = Path(inspect.getfile(dispatch))
        assert sha(dispatch_file) == DISPATCH_SHA
        assert hashlib.sha256(inspect.getsource(dispatch).encode()).hexdigest() == reader_provenance['dispatch_function_source_sha256']
        registry = dispatch_module._AttentionBackendRegistry
        backend_name, backend_fn = registry.get_active_backend()
        supported = registry._supported_arg_names[backend_name]
        assert 'attn_mask' in supported, 'Actual registry backend would silently discard attn_mask'
        assert backend_name.value != '_native_flash', 'Explicit native flash implementation refuses a mask'
        settings = dict(torch_version=torch.__version__, cuda_version=torch.version.cuda,
            sdp_enabled_flags={name: getattr(torch.backends.cuda, name)() for name in
                ('flash_sdp_enabled', 'mem_efficient_sdp_enabled', 'math_sdp_enabled', 'cudnn_sdp_enabled')},
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            matmul_allow_bf16_reduced_precision_reduction=getattr(torch.backends.cuda.matmul, 'allow_bf16_reduced_precision_reduction', None),
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            float32_matmul_precision=torch.get_float32_matmul_precision())
        old_settings = reader_provenance['backend_settings']
        for key, value in settings.items():
            assert old_settings[key] == value, ('Saved native arithmetic settings', key)
        provenance = dict(script_sha256=sha(Path(__file__)), protocol_sha256=sha(out / 'protocol.json'),
            source_audit=source_audit, dispatch_source=str(dispatch_file), dispatch_source_sha256=sha(dispatch_file),
            dispatch_signature=str(inspect.signature(dispatch)),
            registry_backend=backend_name.value, registry_backend_function=backend_fn.__qualname__,
            registry_backend_function_source_sha256=hashlib.sha256(inspect.getsource(backend_fn).encode()).hexdigest(),
            registry_supported_arguments=sorted(supported), attn_mask_supported=True, settings=settings,
            backend_evidence_limit='Current registry selection is recorded; no kernel profiler, and historical backend=None alone did not record the registry selection.',
            library_edits=False, model_initialized=False, model_forwards=0, solver_steps=0, physics_calls=0, scope=SCOPE)
        write(out / 'provenance.json', provenance)
        for seed in SEEDS:
            print('[CURRENT-MASK-DISPATCH] ' + json.dumps(dict(state='start_seed', seed=seed, dispatch_calls=calls)), flush=True)
            folder = out / f'seed{seed}'
            folder.mkdir(exist_ok=False)
            data = torch.load(source / f'seed{seed}/dispatch-t29.pt', map_location='cpu', weights_only=True, mmap=True)
            saved_meta = read(source / f'seed{seed}/dispatch-t29.json')
            indexes = torch.load(source / f'seed{seed}/token-indexes.pt', map_location='cpu', weights_only=True, mmap=True)
            gen = data['gen']
            assert gen['kwargs'] == dict(is_causal=False, enable_gqa=True, backend_repr='None', parallel_config_repr='None')
            assert gen['kwargs'] == saved_meta['gen']['kwargs']
            args = []
            for key, arg_meta in zip(('q', 'k', 'v'), gen['argument_metadata']):
                value = gen[key]
                assert meta(value, torch) == saved_meta['gen'][key]
                assert list(value.shape) == arg_meta['shape'] and str(value.dtype) == arg_meta['dtype']
                assert arg_meta['device'].startswith('cuda') and value.dtype == torch.bfloat16
                actual = torch.empty_strided(value.shape, tuple(arg_meta['stride']), dtype=value.dtype, device=arg_meta['device'])
                actual.copy_(value)
                assert list(actual.stride()) == arg_meta['stride'] and str(actual.device) == arg_meta['device']
                assert exact(actual, value, torch)
                args.append(actual)
            assert meta(gen['output'], torch) == saved_meta['gen']['output']
            assert exact(gen['output'], data['replay_output'], torch)
            assert exact(gen['output'].squeeze(0).flatten(-2, -1), data['pre_W'], torch)
            current, noncurrent, masks = indexes_and_masks(indexes, args[0], args[1], torch)
            torch.save(dict(q=gen['q'], k=gen['k'], v=gen['v'], masks=masks,
                current_GEN_rows=current, noncurrent_GEN_rows=noncurrent,
                original_argument_metadata=gen['argument_metadata']), folder / 'inputs.pt')
            outputs, metrics = {}, {}
            cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state(args[0].device)
            native_kwargs = dict(is_causal=False, enable_gqa=True, backend=None, parallel_config=None)
            with torch.no_grad():
                for arm in ARMS:
                    kwargs = dict(native_kwargs)
                    if arm != 'native':
                        kwargs['attn_mask'] = masks[arm].to(args[0].device)
                    calls += 1
                    output = dispatch(*args, **kwargs)
                    assert output.shape == gen['output'].shape and output.dtype == gen['output'].dtype
                    assert bool(torch.isfinite(output).all())
                    cpu_output = output.detach().cpu().clone()
                    if arm == 'native':
                        assert exact(cpu_output, gen['output'], torch), 'Native official FULL GEN replay not byte exact'
                    merged = gen['output'].clone()
                    merged[:, current] = cpu_output[:, current]
                    assert exact(merged[:, noncurrent], gen['output'][:, noncurrent], torch)
                    outputs[arm] = dict(official_full_output=cpu_output, merged_full_GEN_output=merged,
                        selected_current_output=cpu_output[:, current], retained_UND_output=data['causal']['output'])
                    metrics[arm] = dict(current_vs_native=discrepancy(cpu_output[:, current], gen['output'][:, current], torch),
                        official_full_vs_native=discrepancy(cpu_output, gen['output'], torch),
                        merged_noncurrent_native_byte_exact=True, retained_UND_native_byte_exact=True,
                        mask=None if arm == 'native' else dict(**meta(masks[arm], torch), device=str(kwargs['attn_mask'].device),
                            forbidden_query_key_pairs=int((~masks[arm]).sum())))
                    assert exact(torch.get_rng_state(), cpu_rng, torch) and exact(torch.cuda.get_rng_state(args[0].device), cuda_rng, torch)
                    for actual, key in zip(args, ('q', 'k', 'v')):
                        assert exact(actual, gen[key], torch), 'Official dispatch changed saved Q/K/V'
                    torch.save(outputs, folder / 'outputs.pt')
                    write(folder / (arm + '.json'), metrics[arm])
            floor = metrics['all_allowed']['current_vs_native']['delta_l2']
            for arm in ARMS[2:]:
                cell = metrics[arm]
                cell['current_vs_all_allowed'] = discrepancy(outputs[arm]['selected_current_output'], outputs['all_allowed']['selected_current_output'], torch)
                signal = cell['current_vs_native']['delta_l2']
                cell['all_allowed_numeric_floor_l2'] = floor
                cell['signal_exceeds_ten_times_sham_floor'] = signal > 0 and signal > 10 * floor
                cell['sham_floor_to_native_cut_signal_ratio'] = floor / signal if signal > 0 else None
            report = dict(seed=seed, calls=5, all_allowed_current_byte_exact=metrics['all_allowed']['current_vs_native']['byte_exact'],
                numerical_discrepancy_preserved_without_correction=True, metrics=metrics,
                Q_K_V_bytes_and_global_CPU_CUDA_RNG_unchanged=True,
                merged_noncurrent_and_saved_UND_retained_byte_exact=True, scope=SCOPE)
            write(folder / 'report.json', report)
            completed.append(dict(seed=seed, dispatch_calls=5, report=report,
                files_sha256={name: sha(folder / name) for name in ('inputs.pt', 'outputs.pt', 'report.json') + tuple(arm + '.json' for arm in ARMS)}))
            print('[CURRENT-MASK-DISPATCH] ' + json.dumps(dict(state='finished_seed', seed=seed, dispatch_calls=calls,
                all_allowed_current_byte_exact=report['all_allowed_current_byte_exact'])), flush=True)
            del args, outputs, data
        assert calls == 15 and len(completed) == 3
        write(out / 'results.json', dict(state='complete', official_dispatch_calls=calls, cases=completed,
            all_native_full_GEN_replays_byte_exact=True, model_forwards=0, physics_calls=0, scope=SCOPE))
        write(out / 'complete.json', dict(state='complete', official_dispatch_calls=15, model_forwards=0, physics_calls=0,
            results_sha256=sha(out / 'results.json'), provenance_sha256=sha(out / 'provenance.json'),
            protocol_sha256=sha(out / 'protocol.json'), script_sha256=sha(Path(__file__)), scope=SCOPE))
        print('[CURRENT-MASK-DISPATCH] ' + json.dumps(dict(state='complete', calls=calls, elapsed_s=time.perf_counter() - started)), flush=True)
    except Exception as exc:
        write(out / 'failed.json', dict(state='failed', error=str(exc), traceback=traceback.format_exc(),
            completed_seeds=[row['seed'] for row in completed], actual_dispatch_calls_attempted=calls,
            model_forwards=0, physics_calls=0, scope=SCOPE))
        raise


if __name__ == '__main__':
    main()
