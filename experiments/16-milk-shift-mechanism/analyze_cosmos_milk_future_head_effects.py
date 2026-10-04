"""CPU audit of eighteen completed head interventions and their actual propagation.

Require complete b261 producer and 6767 execution; create only absent
BASE/future-head-effects-matched-runtime. Preserve the failed c3d output;
use producer-version CPU-exact norm witnesses to bridge Torch2.14 to2.7.
Audit all2430 real sites,540 boundary files,
144 model records,2304 official action conversions and119 noise pairs.
Never instantiate a model, dispatch attention, step a solver or run physics.
Cross-arm MLP differences include different inputs; they are not a same-input
causal overwrite test. Head intervention ends after t14; union66 all-allowed
background continues through t29. Later queries are native.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
import traceback


ROOT = Path('/home/current/work/cosmos3')
PROBE_SHA = 'b261366b969a8f33980cbf4d450c381a18a64fa4b16be2e1dd95d6c7220ae898'
EXECUTION_SHA = '6767ee5bf166ebee8c0db7172472610311cf9611f88f72b04d6c10509fa3d064'
CPU_HELPER_SHA = 'ca88ff9c7ee62beef9851fdfaea1c9fddd6e33c0f99058ebee16f26a28a4af9c'
NORM_REPLAY_SHA = '3b9bb679e4c4511c7fa9cedec87db9366afbd606c30239cbc20918deb3e29568'
FAILED_ANALYZER_SHA = 'c3d707e45545ffaf0c89d230fd25b507e2110782ac25e344716f03ec47391f27'
ARMS = ('native', 'all_allowed', 'full_hardmask', 'head_G4', 'head_G6', 'leave_G4',
        'leave_G6', 'arithmetic_sham_G6', 'random_G6')
PLAN = tuple((f'V{v}_A195_{arm}', v, arm) for arm in ARMS for v in (195, 198))
PROP = ('block_input_current50', 'to_add_out_current50', 'attention_residual_current50',
        'mlp_moe_gen_current50', 'block_output_current50')
STAGES = ('preW', 'W_attention', 'attention_residual', 'MLP_output', 'block_output')
FLOOR = 1e-12
SCOPE = ('Fixed X12, V195/V198 and A195. Real32x128 pre-W heads, not residual slices or brain regions. '
         'Within-site head changes use the same live QKV; cross-arm propagation uses each arm dynamic trajectory. '
         'MLP output difference contains changed input and cannot establish causal MLP overwriting. '
         'L0 head pulse is active t0..14; common union66 all-allowed background persists t0..29. '
         't15..29 saved velocity differences describe persistence of an altered trajectory, not an attractor. '
         'Raw RMS is not physical distance, probability, intention or an additive causal contribution. '
         'All18 physical outcomes are reported; only both head_G6 only-milk cases test the registered primary. '
         'No identity decoder, single-head function, cross-position repair or universal necessity claim.')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def finite(value, shape, dtype, torch):
    assert value.shape == shape and value.dtype == dtype and value.device.type == 'cpu'
    assert bool(torch.isfinite(value).all())


def cosine(a, b):
    denominator = float(a.norm()) * float(b.norm())
    return float('nan') if denominator <= FLOOR else float((a * b).sum()) / denominator


def rms(value):
    return float(value.double().square().mean().sqrt())


def seal(base):
    probe_path = ROOT / 'work/probe_cosmos_milk_future_head_groups.py'
    execution_path = ROOT / 'work/run_cosmos_milk_future_head_execution.py'
    cpu_path = ROOT / 'work/analyze_cosmos_milk_future_current_layer_groups.py'
    assert sha(probe_path) == PROBE_SHA and sha(execution_path) == EXECUTION_SHA and sha(cpu_path) == CPU_HELPER_SHA
    probe, adapter, cpu = (load(name, path) for name, path in
        (('head_effects_producer', probe_path), ('head_effects_execution', execution_path), ('head_effects_cpu', cpu_path)))
    _, frozen, cross, helper, value, execution = adapter.prepared(base)
    assert adapter.PLAN == probe.PLAN == PLAN and probe.ARMS == ARMS
    assert frozen['producer_script_sha256'] == PROBE_SHA and frozen['script_sha256'] == EXECUTION_SHA
    producer = base / 'future-head-groups'
    for path in (producer / 'failed.json', execution / 'wrapper_failed.json', execution / 'server/failed.json', execution / 'closed-loop/failed.json'):
        assert not path.exists(), str(path)
    complete, controls, primary = (read(execution / (name + '.json')) for name in ('complete', 'controls', 'primary-prediction'))
    assert complete['state'] == 'complete' and complete['script_sha256'] == EXECUTION_SHA
    assert complete['trial_order'] == [x[0] for x in PLAN]
    assert complete['physical_trials'] == complete['saved_q0_predictions'] == 18
    assert complete['native_predictions'] == 126 and complete['fresh_model_forwards'] == 3780
    assert complete['producer_q0_forwards_separate'] == 540 and complete['combined_two_stage_model_forwards'] == 4320
    assert complete['actual_later_whole_noise_pairs'] == 119 and complete['actual_physical_actions'] == 2304
    for key in ('eight_entire129_state_5JSON_8PNG_and_all8_modelrecords_controls_exact',
                'all18_own_cached_q0_action_and_strict_window_gates_passed', 'all18_executed_without_score_selection'):
        assert complete[key] is True
    for name, expected in complete['files_sha256'].items():
        assert sha(execution / name) == expected
    server = read(execution / 'server/complete.json')
    assert server['state'] == 'complete' and server['queries'] == 144
    assert server['saved_q0_predictions'] == 18 and server['native_predictions'] == 126
    assert server['actual_later_whole_noise_pairs'] == 119 and server['extra_q0_model_forwards'] == server['q0_whole_noise_pair_checks'] == 0
    assert server['eight_controls_all8_entire_records_and_PNGs_exact'] is server['final_model_hooks_none'] is True
    for name, expected in server['files_sha256'].items():
        assert sha(execution / 'server' / name) == expected
    summary = read(execution / 'summary.json')
    assert summary == read(execution / 'closed-loop/summary.json')
    assert summary['completed_trials'] == [x[0] for x in PLAN] and [r['trial'] for r in summary['cases']] == [x[0] for x in PLAN]
    assert set(controls['actual_case_files_sha256']) == {x[0] for x in PLAN}
    for label, files in controls['actual_case_files_sha256'].items():
        for name, expected in files.items():
            assert sha(execution / 'closed-loop' / label / name) == expected
    assert controls['primary_prediction_sha256'] == sha(execution / 'primary-prediction.json')
    return probe, adapter, cpu, frozen, cross, helper, value, producer, execution, complete, controls, primary, summary


def norm_replay_seal(base, expected_complete_sha):
    assert len(expected_complete_sha) == 64 and all(c in '0123456789abcdef' for c in expected_complete_sha)
    source = ROOT / 'work/replay_cosmos_milk_random_norms.py'
    folder = base / 'future-head-random-norm-replay'
    assert sha(source) == NORM_REPLAY_SHA and sha(folder / 'complete.json') == expected_complete_sha
    assert not (folder / 'failed.json').exists()
    complete, provenance, norms = (read(folder / (name + '.json')) for name in ('complete', 'provenance', 'norms'))
    assert complete['state'] == norms['state'] == 'complete' and complete['script_sha256'] == provenance['script_sha256'] == NORM_REPLAY_SHA
    assert complete['producer_script_sha256'] == provenance['producer_script_sha256'] == PROBE_SHA
    assert complete['producer_complete_sha256'] == sha(base / 'future-head-groups/complete.json')
    assert complete['sites_checked'] == norms['sites_checked'] == 30
    assert complete['torch_version'] == provenance['torch_version'] == norms['torch_version'] == '2.14.0+cu130'
    assert complete['all30_FP32_pointwise_witnesses_and_final_BF16_bytes_exact'] is complete['all30_original_double_norm_reports_exact'] is True
    for key in ('model_calls', 'official_attention_dispatch_calls', 'solver_calls', 'physics_calls'):
        assert complete[key] == norms[key] == 0
    assert complete['CUDA_initialized'] is norms['CUDA_initialized'] is False
    for name, expected in complete['files_sha256'].items():
        assert sha(folder / name) == expected
    for name, expected in provenance['producer_documents_sha256'].items():
        assert sha(base / 'future-head-groups' / (name + '.json')) == expected
    assert provenance['private_CPU_generator_only'] is provenance['global_CPU_RNG_bytes_unchanged'] is True
    failed, old_source = base / 'future-head-effects/failed.json', base / 'future-head-effects/analysis-source.py'
    assert read(failed)['state'] == 'failed' and sha(old_source) == FAILED_ANALYZER_SHA
    assert complete['failed_analysis_source_sha256'] == provenance['failed_analysis_source_sha256'] == FAILED_ANALYZER_SHA
    assert complete['failed_analysis_artifact_sha256'] == provenance['failed_analysis_artifact_sha256'] == sha(failed)
    assert [(r['case'], r['step'], r['layer_zero_based']) for r in norms['sites']] == [
        (f'V{v}_A195_random_G6', t, 0) for v in (195, 198) for t in range(15)]
    return {(r['case'], r['step']): r for r in norms['sites']}, dict(source_sha256=NORM_REPLAY_SHA,
        complete_sha256=expected_complete_sha, files_sha256=complete['files_sha256'], torch_version=norms['torch_version'],
        failed_analysis_source_sha256=FAILED_ANALYZER_SHA, failed_analysis_artifact_sha256=sha(failed),
        scope='Norm scalars verified by all30 producer-version exact replays; current plotter independently verifies the same raw inputs and pointwise/BF16 bytes.')


def physical_audit(context, torch, np):
    probe, adapter, cpu, frozen, cross, helper, value, producer, execution, complete, controls, primary, summary = context
    normal = frozen['threshold_contract']['frozen_files']
    base = load('head_effects_normal_pure_checks', Path(normal['normal_runtime']['path']))
    runtime = object.__new__(base.NormalRuntime)  # Only pure byte-check methods, never __init__.
    runtime.torch = torch
    scan = load('head_effects_record_checks', Path(frozen['metrics_path']))
    components = load('head_effects_layout', Path(frozen['components_path']))
    rollout = load('head_effects_controller_checks', Path(normal['controller_helper']['path']))
    forward, gripper, converter_sources = cpu.converters(torch)
    stats = read(Path(normal['normalizer_stats']['path']))['global_raw']
    lo, hi = np.asarray(stats['q01']), np.asarray(stats['q99'])
    assert lo.shape == hi.shape == (10,) and np.all(hi - lo > 1e-6)
    scale, offset = np.maximum(hi - lo, 1e-8) / 2, (hi + lo) / 2
    initial = np.load(execution / 'inputs/x12/state.npy', allow_pickle=False)
    with np.load(execution / 'inputs/x12/controller.npz', allow_pickle=False) as source:
        initial_controller = {key: source[key].copy() for key in source.files}
    originals, native_pair = {}, {}
    for v in (195, 198):
        source = frozen['producer_source_cases'][str(v)]
        originals[v] = cpu.checked(Path(source['q0_folder']) / 'states.pt', source['q0_files_current_sha256']['states.pt'], torch)
        source = frozen['native_q0_source_pair'][str(v)]
        native_pair[v] = cpu.checked(Path(source['folder']) / 'states.pt', source['files_sha256']['states.pt'], torch)
    arrays = dict(normalized_actions=np.zeros((18, 128, 10)), raw_actions=np.zeros((18, 128, 10)),
        native_commands=np.zeros((18, 128, 7)), executed_commands=np.zeros((18, 128, 7)))
    records, rows, sources, later, pairs, whole, objects = {}, [], [], {}, 0, [], None
    for ci, (label, v, arm) in enumerate(PLAN):
        folder = execution / 'closed-loop' / label
        hashes, rule = controls['actual_case_files_sha256'][label], frozen['trials'][label]
        trajectory, result, strict = (read(folder / (name + '.json')) for name in ('trajectory', 'summary', 'strict-windows'))
        assert result == summary['cases'][ci] and result['trial'] == label
        assert len(trajectory) == 129 and [r['step'] for r in trajectory] == list(range(129))
        simstates = np.load(folder / 'sim_states.npy', allow_pickle=False)
        assert simstates.shape == (129, initial.size) and simstates.dtype == initial.dtype and np.isfinite(simstates).all()
        assert cpu.npexact(simstates[0], initial, np)
        assert strict == helper.strict_windows(trajectory, result, np)
        quantified, selected = rollout.quantify(trajectory)
        assert quantified == result['per_object'] and selected == result['selected_objects']
        if objects is None:
            objects = tuple(trajectory[0]['objects'])
            arrays['strict_lift_m'] = np.zeros((18, 129, len(objects)))
            arrays['strict_both_contacts'] = np.zeros((18, 129, len(objects)), dtype=bool)
            arrays['strict_contact_and_lift'] = np.zeros((18, 129, len(objects)), dtype=bool)
        assert tuple(trajectory[0]['objects']) == objects
        firsts = []
        for oi, name in enumerate(objects):
            xyz = np.asarray([r['objects'][name] for r in trajectory])
            contact = np.asarray([[r['finger_contacts'][name]['left'], r['finger_contacts'][name]['right']] for r in trajectory], dtype=bool)
            assert xyz.shape == (129, 3) and np.isfinite(xyz).all()
            assert contact.all(axis=1).tolist() == [r['finger_contacts'][name]['both'] for r in trajectory]
            lift = xyz[:, 2] - xyz[0, 2]
            valid = contact.all(axis=1) & (lift > .02)
            starts = [i for i in range(125) if bool(valid[i:i + 5].all())]
            assert starts == strict['objects'][name]['five_record_window_start_steps']
            assert strict['objects'][name]['first_window'] == (list(range(starts[0], starts[0] + 5)) if starts else None)
            assert result['per_object'][name]['first_2cm_for_5frames'] == (starts[0] if starts else None)
            if starts:
                firsts.append(starts[0])
            arrays['strict_lift_m'][ci, :, oi], arrays['strict_both_contacts'][ci, :, oi] = lift, contact.all(axis=1)
            arrays['strict_contact_and_lift'][ci, :, oi] = valid
        assert result['first_selection_step'] == (min(firsts) if firsts else None)
        typed = {}
        for step, name in ((0, 'initial_controller.npz'), (16, 'controller-step16.npz')):
            with np.load(folder / name, allow_pickle=False) as archive:
                typed[step] = {key: archive[key].copy() for key in archive.files}
            assert set(typed[step]) == set(initial_controller)
            for key, tensor in typed[step].items():
                assert tensor.dtype.kind in 'bfiu' and np.isfinite(tensor).all()
                assert cpu.npexact(np.asarray(trajectory[step]['controller'][key], dtype=tensor.dtype), tensor, np)
                if step == 0:
                    assert cpu.npexact(tensor, initial_controller[key], np)
        for r in trajectory:
            template = typed[0 if r['step'] == 0 else 16]
            assert set(r['controller']) == set(template)
            for key, data in r['controller'].items():
                array = np.asarray(data, dtype=template[key].dtype)
                assert array.shape == template[key].shape and np.isfinite(array).all()
        contracts = read(folder / 'query_contract.json')
        assert len(contracts) == 8
        query_sources = []
        for q in range(8):
            chunk, name = folder / f'chunk_{q:02d}', f'chunk_{q:02d}/states.pt'
            record = cpu.checked(chunk / 'states.pt', hashes[name], torch)
            scan.validate_record(record, torch)
            metadata = read(chunk / 'metadata.json')
            assert metadata == contracts[q]['server_metadata']
            assert contracts[q]['request'] == dict(trial=label, query=q, seed=195 + q, prompt=probe.PROMPT)
            assert contracts[q]['paired_noise_checked_exact'] is (q > 0 and ci > 0)
            assert contracts[q]['response_saved_actions_exact'] is True
            assert metadata['seed'] == 195 + q and metadata['model_calls'] == 30 and metadata['prepare_calls'] == 1
            assert metadata['saved_q0_prediction'] is (q == 0) and metadata['current_execution_model_calls'] == (0 if q == 0 else 30)
            assert metadata['whole_prior_control'] is rule['whole_reference_control']
            assert metadata['input_png_sha256'] == sha(folder / f'input_{q:02d}.png')
            assert metadata['component_intervention_in_this_execution'] is metadata['attention_mask_intervention_in_this_execution'] is False
            assert metadata['head_intervention_in_this_execution'] is metadata['initial_noise_replacement_in_this_execution'] is False
            assert len(record['pure_noise']) == 2
            for tensor, shape in zip(record['pure_noise'], ((1, 48, 5, 10, 20), (16, 64))):
                finite(tensor, shape, torch.float32, torch)
            assert cpu.exact(record['timesteps'], originals[195]['timesteps'], torch) and cpu.exact(record['sigmas'], originals[195]['sigmas'], torch)
            finite(record['action_states'], (31, 1, 16, 64), torch.float32, torch)
            assert torch.count_nonzero(record['action_states'][..., 10:]) == 0
            if q == 0:
                source = frozen['q0_sources'][label]
                original = cpu.checked(Path(source['folder']) / 'states.pt', source['files_sha256']['states.pt'], torch)
                assert cpu.exact(record, original, torch)
                witness = cpu.checked(chunk / 'noise-audit.pt', hashes['chunk_00/noise-audit.pt'], torch)
                helper.verify_cached(record, witness, originals, native_pair, arm, v, runtime, scan)
                records[label] = record
            elif q in later:
                assert cpu.exact(record['pure_noise'], later[q], torch)
                pairs += 1
            else:
                later[q] = record['pure_noise']
            if rule['whole_reference_control']:
                ref = frozen['whole_physical_references'][label]
                old = cpu.checked(Path(ref['folder']) / name, ref['files_current_sha256'][name], torch)
                assert cpu.exact(record, old, torch)
                assert (folder / f'input_{q:02d}.png').read_bytes() == (Path(ref['folder']) / f'input_{q:02d}.png').read_bytes()
            norm = np.asarray(read(chunk / 'normalized_actions.json'))
            assert norm.shape == (16, 10) and norm.dtype == np.float64 and np.isfinite(norm).all()
            assert cpu.npexact(norm, np.asarray(record['actions'].tolist()), np)
            raw = norm * scale + offset
            command = np.asarray([gripper(forward(row, '6d').tolist(), 'zero_one') for row in raw])
            assert command.shape == (16, 7) and command.dtype == np.float64 and np.isfinite(command).all()
            section = slice(q * 16, (q + 1) * 16)
            for key, data in (('normalized_actions', norm), ('raw_actions', raw), ('native_commands', command), ('executed_commands', np.clip(command, -1, 1))):
                arrays[key][ci, section] = data
            query_sources.append(dict(query=q, states_sha256=hashes[name], whole_reference=rule['whole_reference_control'],
                pure_noise_raw_hashes=[cpu.tensor_meta(tensor, torch) for tensor in record['pure_noise']]))
        for key, name in (('normalized_actions', 'normalized_actions.json'), ('raw_actions', 'denormalized_actions.json'),
                          ('native_commands', 'actions.json'), ('executed_commands', 'executed_actions.json')):
            assert cpu.npexact(np.asarray(read(folder / name)), arrays[key][ci], np)
        if rule['whole_reference_control']:
            ref = frozen['whole_physical_references'][label]
            old = Path(ref['folder'])
            for name, expected in ref['files_current_sha256'].items():
                assert sha(old / name) == expected
            assert cpu.npexact(simstates, np.load(old / 'sim_states.npy', allow_pickle=False), np)
            for name in (*adapter.PHYSICAL, *[f'input_{q:02d}.png' for q in range(8)]):
                assert (folder / name).read_bytes() == (old / name).read_bytes()
            for step, name in ((0, 'initial_controller.npz'), (16, 'controller-step16.npz')):
                with np.load(old / name, allow_pickle=False) as saved:
                    assert set(typed[step]) == set(saved.files) and all(cpu.npexact(a, saved[k], np) for k, a in typed[step].items())
            whole.append(label)
        rows.append(dict(case=label, arm=arm, vision_noise=v, selected_objects=result['selected_objects'],
            first_selection_step=result['first_selection_step'], strict_window_starts={n: strict['objects'][n]['five_record_window_start_steps'] for n in objects},
            whole_reference_control=rule['whole_reference_control'], physical_files_sha256=hashes, query_sources=query_sources))
        print('[HEAD-EFFECTS] physical audited ' + label, flush=True)
    assert pairs == 119 and len(whole) == 8
    for label, v, arm in PLAN:
        assert cpu.exact(records[label]['pure_noise'], records[f'V{v}_A195_native']['pure_noise'], torch)
        assert cpu.exact(records[label]['action_states'][0], records[PLAN[0][0]]['action_states'][0], torch)
    primary_labels = [f'V{v}_A195_head_G6' for v in (195, 198)]
    supported = all(rows[[p[0] for p in PLAN].index(label)]['selected_objects'] == ['milk_1'] for label in primary_labels)
    assert primary['state'] == 'evaluated_after_all18_complete' and primary['primary_arm'] == 'head_G6'
    assert primary['both_head_G6_strict_only_milk'] is supported and primary['conditional_sufficiency_supported'] is supported
    assert primary['prediction'] == complete['primary_prediction'] == ('supported' if supported else 'rejected')
    assert primary['group_specific_causal_claim_established'] is complete['group_specific_causal_claim_established'] is False
    arrays['object_names'] = np.asarray(objects)
    return records, arrays, rows, dict(actual_records=144, actual_converted_actions=2304,
        actual_later_whole_two_draw_pairs=119, eight_whole_reference_controls=whole, converter_sources=converter_sources,
        primary_prediction=primary['prediction']), runtime, components, scan


def site_audit(context, records, torch, np, runtime, components, scan, replay):
    probe, adapter, cpu, frozen, cross, helper, value, producer, execution, complete, controls, primary, summary = context
    arrays = dict(within_site_head_delta_RMS=np.zeros((18, 15, 9, 32)),
        within_site_head_reference_RMS=np.zeros((18, 15, 9, 32)),
        cross_arm_stage_delta_RMS=np.zeros((18, 15, 9, 5)),
        MLP_delta_vs_attention_residual_delta_cosine=np.full((18, 15, 9), np.nan),
        MLP_delta_vs_W_attention_delta_cosine=np.full((18, 15, 9), np.nan),
        BF16_addition_delta_discrepancy_RMS=np.zeros((18, 15, 9, 2)),
        current_boundary_delta_RMS=np.zeros((18, 30, 37)), action_boundary_delta_RMS=np.zeros((18, 30, 37)),
        q0_velocity_XYZ_delta_RMS=np.zeros((18, 30)), q0_final_XYZ_delta_RMS=np.zeros(18),
        random_actual_relative_norm_error=np.full((18, 15), np.nan),
        random_plotter_vs_producer_double_norm_difference=np.full((18, 15, 2), np.nan))
    sources, first_qkv, direction_by_step = [], {}, {}
    results = read(producer / 'results.json')
    for ci, ((label, v, arm), row) in enumerate(zip(PLAN, results['cases'])):
        assert row['case'] == label and row['arm'] == arm
        folder, own_sham = producer / label, producer / f'V{v}_A195_all_allowed'
        record, reference_record = records[label], records[f'V{v}_A195_all_allowed']
        index = cpu.checked(folder / 'indexes-and-masks.pt', row['files_sha256']['indexes-and-masks.pt'], torch)
        expected_index = components._layout(record['model_input'])
        assert cpu.exact(index['indexes']['action_rows'], expected_index['action_rows'], torch)
        assert cpu.exact(index['indexes']['current_rows'], torch.arange(50), torch)
        assert cpu.exact(index['indexes']['future_keys'], torch.arange(171, 371), torch)
        manifest = read(folder / 'site-captures.json')
        assert len(manifest['files']) == 135 and manifest['propagation_counts'] == dict(projection=135, residual=135, MLP=135, completed=135)
        site_boundary = None
        for item in manifest['files']:
            t, layer = item['step'], item['layer_zero_based']
            if layer == 0:
                boundary_item = row['boundary_files'][t]
                site_boundary = cpu.checked(folder / boundary_item['file'], boundary_item['sha256'], torch)
            d = cpu.checked(folder / item['file'], item['sha256'], torch)
            assert (d['step'], d['layer_zero_based'], d['case_arm']) == (t, layer, arm)
            assert d['event']['step'] == t and d['event']['layer_zero_based'] == layer
            active = probe.arm_window(arm) is not None and layer <= probe.arm_window(arm)['layers'][1]
            heads = probe.selected_heads(arm, layer) if active else []
            retained = [h for h in range(32) if h not in heads]
            assert d['selected_query_heads'] == heads and d['event']['selected_query_heads'] == heads
            assert cpu.exact(d['retained_query_heads'], torch.tensor(retained, dtype=torch.long), torch)
            assert cpu.exact(d['current_rows'], index['indexes']['current_rows'], torch)
            assert cpu.exact(d['future_keys'], index['indexes']['future_keys'], torch)
            q, k, val = d['Qcurrent50'], d['Kall'], d['Vall']
            finite(q, (1, 50, 32, 128), torch.bfloat16, torch)
            for tensor in (k, val):
                finite(tensor, (1, 387, 8, 128), torch.bfloat16, torch)
            if t == layer == 0:
                if v not in first_qkv:
                    first_qkv[v] = tuple(a.clone() for a in (q, k, val))
                assert all(cpu.exact(a, b, torch) for a, b in zip((q, k, val), first_qkv[v]))
            z = d['endpoints']
            expected = {'Znative', 'Zreal'} | ({'Z0'} if arm != 'native' else set())
            if active and arm not in ('native', 'all_allowed'):
                expected.add('Zclone' if arm == 'arithmetic_sham_G6' else 'Zk')
            assert set(z) == {'Znative', 'Z0', 'Zv', 'Zk', 'Zreal', 'Zclone'}
            assert {name for name, tensor in z.items() if tensor is not None} == expected
            assert d['missing_endpoints'] == item['missing_endpoints'] == [name for name, tensor in z.items() if tensor is None]
            for name, tensor in z.items():
                if tensor is not None:
                    finite(tensor, (1, 50, 32, 128), torch.bfloat16, torch)
                    meta = cpu.tensor_meta(tensor, torch)
                    assert all(d['endpoint_metadata'][name][key] == val for key, val in meta.items())
                else:
                    assert d['endpoint_metadata'][name] is None
            baseline = z['Znative'] if arm == 'native' else z['Z0']
            if arm == 'native' or not active or arm == 'arithmetic_sham_G6':
                assert cpu.exact(z['Zreal'], baseline, torch)
            assert cpu.exact(z['Zreal'][:, :, retained], baseline[:, :, retained], torch)
            if active and arm not in ('native', 'all_allowed', 'arithmetic_sham_G6', 'random_G6'):
                assert cpu.exact(z['Zreal'][:, :, heads], z['Zk'][:, :, heads], torch)
            if active and arm == 'arithmetic_sham_G6':
                assert cpu.exact(z['Zclone'], z['Z0'], torch)
                original = z['Z0'][:, :, heads]
                intended = original.float() + (z['Zclone'][:, :, heads].float() - original.float())
                assert cpu.exact(d['FP32arithmetic_sham_selected'], intended, torch)
                assert cpu.exact(z['Zreal'][:, :, heads], original, torch)
            assert d['FP32allocation'] is d['FP32arithmetic_sham'] is None
            assert (d['FP32arithmetic_sham_selected'] is not None) is (active and arm == 'arithmetic_sham_G6')
            if active and arm == 'random_G6':
                verified = replay[(label, t)]
                assert verified['source_file'] == item['file'] and verified['source_sha256'] == item['sha256']
                assert verified['all_FP32_witnesses_and_final_BF16_bytes_exact'] is verified['original_double_norm_report_exact'] is True
                for key, expected_meta in verified['source_raw_tensor_hashes'].items():
                    assert cpu.tensor_meta(z[key] if key in z else d[key], torch) == expected_meta
                before, after = d['random_generator_before'], d['random_generator_after']
                generator = torch.Generator(device='cpu').manual_seed(901000 + t * 36)
                assert cpu.exact(before, generator.get_state(), torch)
                direction = torch.randn((1, 50, 4, 128), dtype=torch.float32, generator=generator)
                assert cpu.exact(direction, d['FP32random_direction'], torch) and cpu.exact(after, generator.get_state(), torch)
                natural = z['Zk'][:, :, heads].float() - baseline[:, :, heads].float()
                raw_norm = torch.tensor(verified['FP32_raw_norm'], dtype=torch.float32)
                target = torch.tensor(verified['FP32_target_norm'], dtype=torch.float32)
                assert int(raw_norm.view(torch.int32)) == verified['FP32_raw_norm_int32_bits']
                assert int(target.view(torch.int32)) == verified['FP32_target_norm_int32_bits']
                assert bool(torch.isfinite(raw_norm)) and float(raw_norm) > 0 and bool(torch.isfinite(target)) and float(target) >= 0
                unit = direction / raw_norm
                increment = unit * target
                candidate = baseline[:, :, heads].float() + increment
                for key, tensor in (('FP32random_Dk_selected', natural), ('FP32random_unit_direction', unit),
                                    ('FP32random_delta_selected', increment), ('FP32random_selected', candidate)):
                    assert cpu.exact(d[key], tensor, torch)
                actual = torch.where(increment == 0, baseline[:, :, heads], candidate.to(torch.bfloat16))
                assert cpu.exact(actual, z['Zreal'][:, :, heads], torch)
                target_norm = verified['actual_BF16_target_double_L2']
                actual_norm = verified['actual_BF16_random_double_L2']
                arrays['random_plotter_vs_producer_double_norm_difference'][ci, t] = [
                    float((z['Zk'][:, :, heads].double() - baseline[:, :, heads].double()).norm()) - target_norm,
                    float((actual.double() - baseline[:, :, heads].double()).norm()) - actual_norm]
                error = 0.0 if target_norm == actual_norm == 0 else abs(actual_norm / target_norm - 1)
                witness = d['random_control']
                assert witness == verified['original_random_control_report'] and witness['target_Dk_FP32_l2'] == float(target)
                assert witness['target_Dk_actual_BF16_l2'] == target_norm and witness['random_actual_BF16_l2'] == actual_norm
                assert witness['actual_BF16_relative_norm_error'] == error
                assert witness['actual_BF16_equal_strength_within_tolerance'] is (error <= .10)
                assert witness == d['event']['random_control'] == manifest['random_control']['checks'][t]
                arrays['random_actual_relative_norm_error'][ci, t] = error
                if t not in direction_by_step:
                    direction_by_step[t] = direction.clone()
                assert cpu.exact(direction, direction_by_step[t], torch)
            else:
                assert all(d[key] is None for key in ('FP32random_Dk_selected', 'FP32random_direction', 'FP32random_unit_direction',
                    'FP32random_delta_selected', 'FP32random_selected', 'random_generator_before', 'random_generator_after', 'random_control'))
            for device in ('cpu', 'cuda'):
                assert cpu.exact(d['RNG_before'][device], d['RNG_after'][device], torch)
            assert d['actual_QKV_bytes_unchanged'] is d['actual_pre_W_current_bytes_equal_Zreal'] is True
            assert d['original_full_projection_calls_at_site'] == 1
            delta = z['Zreal'].double() - baseline.double()
            arrays['within_site_head_delta_RMS'][ci, t, layer] = delta.square().mean(dim=(0, 1, 3)).sqrt().numpy()
            arrays['within_site_head_reference_RMS'][ci, t, layer] = baseline.double().square().mean(dim=(0, 1, 3)).sqrt().numpy()
            assert [w['query_head'] for w in d['headwise_effective_vs_native_or_Z0']] == list(range(32))
            for h, w in enumerate(d['headwise_effective_vs_native_or_Z0']):
                assert w['kv_head'] == h // 4 and w['delta_l2'] == float(delta[:, :, h].norm())
                assert w['changed_values'] == int(torch.count_nonzero(delta[:, :, h]))
            p = d['propagation']
            assert set(p) == set(PROP) and d['actual_BF16_residual_chain_byte_exact'] is d['propagation_observed_forward_only'] is True
            for tensor in p.values():
                finite(tensor, (50, 4096), torch.bfloat16, torch)
            assert cpu.exact(p[PROP[0]], site_boundary['current_hidden'][layer], torch)
            assert cpu.exact(p[PROP[4]], site_boundary['current_hidden'][layer + 1], torch)
            assert d['event'] == site_boundary['site_events'][layer]
            assert cpu.exact(p[PROP[0]] + p[PROP[1]], p[PROP[2]], torch)
            assert cpu.exact(p[PROP[2]] + p[PROP[3]], p[PROP[4]], torch)
            sham = d if arm == 'all_allowed' else torch.load(own_sham / item['file'], map_location='cpu', weights_only=True, mmap=True)
            ds = [z['Zreal'].double() - sham['endpoints']['Zreal'].double()]
            ds.extend(p[key].double() - sham['propagation'][key].double() for key in (PROP[1], PROP[2], PROP[3], PROP[4]))
            arrays['cross_arm_stage_delta_RMS'][ci, t, layer] = [rms(a) for a in ds]
            arrays['MLP_delta_vs_attention_residual_delta_cosine'][ci, t, layer] = cosine(ds[3], ds[2])
            arrays['MLP_delta_vs_W_attention_delta_cosine'][ci, t, layer] = cosine(ds[3], ds[1])
            input_delta = p[PROP[0]].double() - sham['propagation'][PROP[0]].double()
            arrays['BF16_addition_delta_discrepancy_RMS'][ci, t, layer] = [rms(ds[2] - input_delta - ds[1]), rms(ds[4] - ds[2] - ds[3])]
            sources.append(dict(case=label, step=t, layer_zero_based=layer, file=item['file'], sha256=item['sha256'],
                actual_endpoint_raw_hashes={name: None if tensor is None else cpu.tensor_meta(tensor, torch) for name, tensor in z.items()},
                actual_propagation_raw_hashes={name: cpu.tensor_meta(tensor, torch) for name, tensor in p.items()}))
        for t, item in enumerate(row['boundary_files']):
            data = cpu.checked(folder / item['file'], item['sha256'], torch)
            assert data['step'] == t and data['case_arm'] == arm and data['selected_window'] == probe.arm_window(arm)
            kwargs = data['actual_model_kwargs']
            assert cpu.exact(kwargs['action_tokens'][0], record['action_states'][t, 0].to(torch.bfloat16), torch)
            assert cpu.exact(data['actual_model_action_timesteps'], kwargs['action_timesteps'], torch)
            assert cpu.exact(data['solver_sigma_from_same_frozen_schedule'], record['sigmas'][t], torch)
            assert cpu.exact(kwargs['vision_tokens'][0][0, :, 0], record['model_input']['vision_tokens'][0][0, :, 0], torch)
            assert cpu.exact(data['actual_pack'], dict(layout=components._layout(kwargs), metadata={key: kwargs.get(key) for key in components._STRUCTURAL_KEYS}), torch)
            assert cpu.exact(data['model_output'][2][0], record['action_velocity'][t], torch)
            if t == 0:
                assert cpu.exact(kwargs, record['model_input'], torch)
            sham = data if arm == 'all_allowed' else torch.load(own_sham / item['file'], map_location='cpu', weights_only=True, mmap=True)
            for key, count in (('current_hidden', 50), ('action_hidden', 16)):
                finite(data[key], (37, count, 4096), torch.bfloat16, torch)
                metric = 'current_boundary_delta_RMS' if count == 50 else 'action_boundary_delta_RMS'
                arrays[metric][ci, t] = (data[key].double() - sham[key].double()).square().mean((1, 2)).sqrt().numpy()
            assert len(data['site_events']) == 36
            for layer, event in enumerate(data['site_events']):
                active = probe.arm_window(arm) is not None and t <= 14 and layer <= probe.arm_window(arm)['layers'][1]
                heads = probe.selected_heads(arm, layer) if active else []
                assert event['step'] == t and event['layer_zero_based'] == layer and event['arm'] == arm
                assert event['selected_window_cut_active'] is active and event['selected_query_heads'] == heads
                assert event['extra_all_allowed_calls'] == int(arm != 'native')
                assert event['extra_cut_calls'] == int(active and arm != 'arithmetic_sham_G6')
                assert event['extra_clone_sham_calls'] == int(active and arm == 'arithmetic_sham_G6')
                assert event['to_add_out_actual_calls'] == 1 and event['substituted_query_rows'] == (0 if arm == 'native' else 66)
                assert event['original_QKV_bytes_preserved'] is event['RNG_bytes_preserved'] is True
                assert event['future200_native_bytes_preserved'] is event['UND_dispatch_return_unmodified'] is True
            if t == 0:
                assert cpu.exact(data['current_hidden'][0], sham['current_hidden'][0], torch)
                assert cpu.exact(data['action_hidden'][0], sham['action_hidden'][0], torch)
                assert cpu.exact(kwargs, sham['actual_model_kwargs'], torch)
            arrays['q0_velocity_XYZ_delta_RMS'][ci, t] = rms(record['action_velocity'][t, :, :3].double() - reference_record['action_velocity'][t, :, :3].double())
        arrays['q0_final_XYZ_delta_RMS'][ci] = rms(record['actions'][:, :3].double() - reference_record['actions'][:, :3].double())
        print('[HEAD-EFFECTS] sites and boundaries audited ' + label, flush=True)
    assert len(sources) == 2430 and len(direction_by_step) == 15
    return arrays, sources


def plots(out, arrays, np, plt):
    chosen = ('full_hardmask', 'head_G4', 'head_G6', 'leave_G4', 'leave_G6', 'arithmetic_sham_G6', 'random_G6')
    maximum = float(arrays['within_site_head_delta_RMS'].max())
    fig, axes = plt.subplots(len(chosen), 2, figsize=(12, 16), constrained_layout=True)
    for row, arm in enumerate(chosen):
        for col, v in enumerate((195, 198)):
            ci = [x[0] for x in PLAN].index(f'V{v}_A195_{arm}')
            im = axes[row, col].imshow(arrays['within_site_head_delta_RMS'][ci, :, 0].T, origin='lower', aspect='auto', vmin=0, vmax=maximum or 1)
            axes[row, col].set(title=f'L0 {arm}, V{v}', xlabel='Denoising step (0..14)', ylabel='Actual query head (0..31)')
    fig.colorbar(im, ax=axes, label='Within-site current50 pre-W change RMS')
    fig.savefig(out / 'actual-L0-head-changes.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, v in zip(axes, (195, 198)):
        for arm in chosen:
            ci = [x[0] for x in PLAN].index(f'V{v}_A195_{arm}')
            ax.plot(range(5), arrays['cross_arm_stage_delta_RMS'][ci, :, 0].mean(0), marker='o', label=arm)
        ax.set(title=f'L0 current50 / V{v}', ylabel='Mean of 15 actual site RMS vs own all-allowed trajectory', xticks=range(5), xticklabels=STAGES)
        ax.tick_params(axis='x', rotation=22)
        ax.legend(fontsize=8)
    fig.suptitle('Cross-arm propagation: changed MLP input is included; stage distances are not additive')
    fig.savefig(out / 'actual-propagation-stages.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, v in zip(axes, (195, 198)):
        for arm in chosen:
            ci = [x[0] for x in PLAN].index(f'V{v}_A195_{arm}')
            ax.plot(range(30), arrays['q0_velocity_XYZ_delta_RMS'][ci], label=arm)
        ax.axvline(14.5, color='black', ls='--', lw=1)
        ax.set(title=f'Actual q0 velocity / V{v}', xlabel='Denoising step', ylabel='Raw XYZ velocity RMS vs own all-allowed')
        ax.legend(fontsize=8)
    fig.suptitle('After t14 head intervention stops; all-allowed background remains through t29')
    fig.savefig(out / 'actual-withdrawal-velocity.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--analysis-stage', choices=('future-head-effects-matched-runtime',), default='future-head-effects-matched-runtime')
    parser.add_argument('--random-norms-complete-sha', required=True, help='SHA256 of completed producer-version CPU replay complete.json')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    context = seal(args.output.resolve())
    replay, replay_provenance = norm_replay_seal(args.output.resolve(), args.random_norms_complete_sha)
    out = args.output.resolve() / args.analysis_stage
    out.mkdir(exist_ok=False)
    started = time.perf_counter()
    try:
        import torch
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        torch.set_num_threads(1)
        assert not torch.cuda.is_initialized()
        records, physical_arrays, rows, audit, runtime, components, scan = physical_audit(context, torch, np)
        arrays, site_sources = site_audit(context, records, torch, np, runtime, components, scan, replay)
        arrays.update(physical_arrays, case_names=np.asarray([x[0] for x in PLAN]), propagation_stage_names=np.asarray(STAGES))
        for ci, row in enumerate(rows):
            row.update(q0_final_normalized_XYZ_RMS_vs_own_sham=float(arrays['q0_final_XYZ_delta_RMS'][ci]),
                q0_velocity_XYZ_RMS_vs_own_sham_t14=float(arrays['q0_velocity_XYZ_delta_RMS'][ci, 14]),
                q0_velocity_XYZ_RMS_vs_own_sham_t15=float(arrays['q0_velocity_XYZ_delta_RMS'][ci, 15]),
                q0_velocity_XYZ_RMS_vs_own_sham_t29=float(arrays['q0_velocity_XYZ_delta_RMS'][ci, 29]),
                L0_current_cross_arm_stage_mean_RMS=arrays['cross_arm_stage_delta_RMS'][ci, :, 0].mean(0).tolist(),
                random_actual_BF16_relative_norm_error_max=None if row['arm'] != 'random_G6' else float(arrays['random_actual_relative_norm_error'][ci].max()))
        np.savez_compressed(out / 'actual-head-effects.npz', **arrays)
        assert (out / 'actual-head-effects.npz').stat().st_size < 100_000_000
        with np.load(out / 'actual-head-effects.npz', allow_pickle=False) as saved:
            assert set(saved.files) == set(arrays)
            for key, actual in arrays.items():
                assert saved[key].dtype == actual.dtype and saved[key].shape == actual.shape and saved[key].tobytes(order='C') == actual.tobytes(order='C')
        plots(out, arrays, np, plt)
        producer, execution = context[7], context[8]
        provenance = dict(script_sha256=sha(Path(__file__)), producer_script_sha256=PROBE_SHA, execution_script_sha256=EXECUTION_SHA,
            CPU_helper_script_sha256=CPU_HELPER_SHA,
            producer_version_random_norm_replay=replay_provenance, plotter_torch_version=str(torch.__version__),
            producer_documents_sha256={name: sha(producer / (name + '.json')) for name in ('complete', 'results', 'provenance', 'protocol', 'sources', 'primary-prediction')},
            execution_complete_sha256=sha(execution / 'complete.json'), execution_documents_sha256=context[9]['files_sha256'],
            actual_site_files_and_tensor_hashes=site_sources, actual_physical_files_sha256=context[10]['actual_case_files_sha256'],
            scope=SCOPE)
        write(out / 'provenance.json', provenance)
        assert not torch.cuda.is_initialized()
        write(out / 'analysis.json', dict(state='complete', cases=rows, audit=audit, sites_checked=2430,
            boundaries_checked=540, actual_BF16_residual_chain_bytes_exact=True, actual_head_and_random_witnesses_bytes_exact=True,
            NPZ_all_keys_dtype_shape_Cbytes_exact=True, full_outputs_scope='Full model tuple is saved/hash-bound; action payload checked against all30 recorded velocity. Eight historical controls are whole-record exact.',
            MLP_delta_cosine_scope='Descriptive cross-arm output deltas with different MLP inputs; no same-input causal overwrite test.',
            zero_norm_cosines='NaN with 1e-12 denominator floor; no large relative percentages reported.',
            random_norm_scope='Torch2.14 replay must reproduce all30 original FP32 witnesses, BF16 endpoints and original double L2 exactly. Plotter uses those sealed scalars and independently repeats RNG/pointwise/BF16 bytes; its own reduction differences are diagnostics only.',
            failed_analysis_preserved=replay_provenance, plotter_torch_version=str(torch.__version__),
            source=provenance['producer_documents_sha256'], primary_physical_prediction=context[11],
            model_calls=0, official_attention_dispatch_calls=0, solver_calls=0, physics_calls=0, CUDA_initialized=False, scope=SCOPE))
        names = ('actual-head-effects.npz', 'actual-L0-head-changes.png', 'actual-propagation-stages.png',
                 'actual-withdrawal-velocity.png', 'analysis.json', 'provenance.json')
        write(out / 'complete.json', dict(state='complete', script_sha256=sha(Path(__file__)),
            producer_complete_sha256=sha(producer / 'complete.json'), execution_complete_sha256=sha(execution / 'complete.json'),
            files_sha256={name: sha(out / name) for name in names}, sites_checked=2430, boundaries_checked=540,
            actual_records=144, actual_converted_actions=2304, actual_later_whole_two_draw_pairs=119,
            model_calls=0, official_attention_dispatch_calls=0, solver_calls=0, physics_calls=0, CUDA_initialized=False,
            elapsed_s=time.perf_counter() - started))
        print('[HEAD-EFFECTS] ' + json.dumps(dict(state='complete', cases=rows, elapsed_s=time.perf_counter() - started)), flush=True)
    except Exception as exc:
        write(out / 'failed.json', dict(state='failed', error=str(exc), traceback=traceback.format_exc(), model_calls=0, physics_calls=0))
        raise


if __name__ == '__main__':
    main()
