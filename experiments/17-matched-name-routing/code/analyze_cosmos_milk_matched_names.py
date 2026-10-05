"""CPU audit of all24 completed matched-name predictions and physical trials.

Create only absent BASE/matched-names-analysis. Read real packed tensors,
720 sparse decoder/dispatch records, 240 native language suffixes, 192 model
records and 3072 physical commands. Reconstruct t0 attention in Float64 only
as a diagnostic; this is neither the executed BF16 weights nor an intervention.
No model construction, forward, dispatch, solver, environment or physics call.
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
PROBE_SHA = '0446978830ac36978396fdad5b09df238b0fd6b3b425aaa351a9b40262f94df0'
EXECUTION_SHA = '6b5ea13b8ce888aecac5cdb42b1a23ae9fec81972edf51f241f1e4429008217c'
CPU_SHA = 'ca88ff9c7ee62beef9851fdfaea1c9fddd6e33c0f99058ebee16f26a28a4af9c'
LANGUAGE_SHA = '74ac42db450a1a7cd308c837bf75049cd2dcae858a4869dab1f0f60450c69b13'
STAGE = 'matched-names-analysis'
SCENES, GOALS, VISIONS = ('x12', 'x15'), ('milk_box', 'cream_cheese'), (195, 198)
ARMS = ('native', 'all_allowed', 'full_hardmask')
GROUPS = ('UND_before_target', 'target2', 'UND_after_target', 'current50', 'future200', 'action16')
SCOPE = ('Matched milk box/cream cheese are legitimate equal-length names, not a repair of the original '
         'milk instruction at15cm. Only four X15 native strict outcomes test the preregistered natural '
         'donor hypothesis. Eight numerical shams separately constrain cut attribution. Real Q/Z changes '
         'are paired within the same step/scene/arm/noise or goal; later steps contain dynamic feedback. '
         'UND causal-suffix equality narrows the source of differences only when actual bytes agree; it '
         'does not prove object grounding, one functional head, or a language/motor brain region. '
         'Offline Float64 softmax is derived from actual post-RoPE QK, not recorded backend weights. '
         'Hard masks redistribute weight; hidden RMS is not probability, centimetres or information.')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(data, indent=2, allow_nan=False) + '\n')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def label(scene, goal, v, arm):
    return f'{scene}_{goal}_V{v}_A195_{arm}'


def rms_delta(a, b, np):
    # Conversion precedes subtraction; signed BF16 values are not subtracted in BF16.
    return np.sqrt(np.mean((a.double().numpy() - b.double().numpy()) ** 2, axis=-1, dtype=np.float64))


def seal(base):
    pp, ep = ROOT / 'work/probe_cosmos_milk_matched_names.py', ROOT / 'work/run_cosmos_milk_matched_names.py'
    cp, lp = ROOT / 'work/analyze_cosmos_milk_future_current_layer_groups.py', ROOT / 'work/probe_cosmos_milk_language_selectivity.py'
    assert sha(pp) == PROBE_SHA and sha(ep) == EXECUTION_SHA and sha(cp) == CPU_SHA and sha(lp) == LANGUAGE_SHA
    adapter, cpu, language = load('matched_analysis_adapter', ep), load('matched_analysis_cpu', cp), load('matched_analysis_language', lp)
    # prepared() rejects an already completed execution. Use its same contract,
    # then audit the completed provenance instead of pretending it is unfinished.
    execution = base / adapter.STAGE
    assert read(execution / 'complete.json')['state'] == 'complete', 'All24 physical trials must finish first'
    frozen, probe, simulator, helper, cross = adapter.contract(base)
    producer = base / probe.STAGE
    for path in (producer / 'failed.json', execution / 'wrapper_failed.json', execution / 'simulator_failed.json',
                 execution / 'server/failed.json', execution / 'closed-loop/failed.json'):
        assert not path.exists(), str(path)
    assert read(execution / 'provenance.json') == frozen
    assert read(execution / 'prepared.json') == dict(state='prepared', provenance_sha256=sha(execution / 'provenance.json'))
    assert read(execution / 'primary-preregistered.json') == dict(state='preregistered', **probe.PRIMARY)
    pc, results = read(producer / 'complete.json'), read(producer / 'results.json')
    ec, controls, summary, comparisons = (read(execution / (name + '.json')) for name in ('complete', 'controls', 'summary', 'comparisons'))
    assert ec['script_sha256'] == EXECUTION_SHA and ec['physical_trials'] == ec['saved_q0_predictions'] == 24
    assert ec['native_predictions'] == 168 and ec['fresh_model_forwards'] == 5040 and ec['producer_q0_forwards'] == 720
    assert ec['combined_two_stage_model_forwards'] == 5760 and ec['actual_later_whole_noise_pairs'] == 161
    assert ec['actual_physical_actions'] == 3072 and ec['trial_order'] == frozen['trial_order']
    for key in ('six_entire129_state_5JSON_8PNG_all8_modelrecords_and_controller0_16_controls_exact',
                'all24_cached_q0_action_and_strict_window_gates_passed', 'all24_executed_without_score_selection'):
        assert ec[key] is True
    for name, expected in ec['files_sha256'].items():
        assert sha(execution / name) == expected
    assert summary == read(execution / 'closed-loop/summary.json') and summary['completed_trials'] == frozen['trial_order']
    assert [r['trial'] for r in summary['cases']] == [r['case'] for r in comparisons['cases']] == frozen['trial_order']
    assert comparisons['state'] == 'all24_complete' and set(controls['actual_case_files_sha256']) == set(frozen['trial_order'])
    for case, files in controls['actual_case_files_sha256'].items():
        for name, expected in files.items():
            assert sha(execution / 'closed-loop' / case / name) == expected
    server = read(execution / 'server/complete.json')
    assert server['state'] == 'complete' and server['queries'] == 192 and server['saved_q0_predictions'] == 24
    assert server['native_predictions'] == 168 and server['fresh_model_forwards'] == 5040 and server['actual_later_whole_noise_pairs'] == 161
    assert server['six_X12_cream_cheese_controls_all8_entire_records_and_PNGs_exact'] is server['final_model_hooks_none'] is True
    for name, expected in server['files_sha256'].items():
        assert sha(execution / 'server' / name) == expected
    assert not (execution / 'inputs/metadata.json').exists()
    for scene, item in frozen['scenes'].items():
        assert (execution / 'inputs' / scene).is_symlink() and (execution / 'inputs' / scene).resolve() == Path(item['folder']).resolve()
    return dict(base=base, producer=producer, execution=execution, adapter=adapter, probe=probe, cpu=cpu, language=language,
        frozen=frozen, simulator=simulator, helper=helper, cross=cross, pc=pc, ec=ec, rows=results['cases'], controls=controls,
        comparisons=comparisons, summary=summary)


def physical_audit(c, torch, np):
    cpu, frozen, execution = c['cpu'], c['frozen'], c['execution']
    files = frozen['threshold_contract']['frozen_files']
    base = load('matched_analysis_byte_methods', Path(files['normal_runtime']['path']))
    runtime = object.__new__(base.NormalRuntime)  # cpu/require_exact only: no __init__, pipeline or model.
    runtime.torch = torch
    scan = load('matched_analysis_record_checks', Path(frozen['metrics_path']))
    components = load('matched_analysis_pack_layout', Path(frozen['components_path']))
    rollout = load('matched_analysis_controller', Path(files['controller_helper']['path']))
    forward, gripper, converter_sources = cpu.converters(torch)
    stats = read(Path(files['normalizer_stats']['path']))['global_raw']
    lo, hi = np.asarray(stats['q01']), np.asarray(stats['q99'])
    assert lo.shape == hi.shape == (10,) and np.all(hi - lo > 1e-6)
    scale, offset = np.maximum(hi - lo, 1e-8) / 2, (hi + lo) / 2
    arrays = {key: np.empty((24, 128, width), np.float64) for key, width in
              (('normalized_actions', 10), ('raw_actions', 10), ('native_commands', 7), ('executed_commands', 7))}
    q0, physical, later, whole, pairs, objects, schedule = {}, [], {}, [], 0, None, None
    for ci, row in enumerate(c['rows']):
        case, scene, goal = row['case'], row['scene'], row['goal']
        rule, folder = frozen['trials'][case], execution / 'closed-loop' / case
        hashes = c['controls']['actual_case_files_sha256'][case]
        inputs = execution / 'inputs' / scene
        trajectory, result, strict = (read(folder / (name + '.json')) for name in ('trajectory', 'summary', 'strict-windows'))
        assert result == c['summary']['cases'][ci] and result['policy_goal'] == goal
        assert len(trajectory) == 129 and [r['step'] for r in trajectory] == list(range(129))
        initial, state = np.load(inputs / 'state.npy', allow_pickle=False), np.load(folder / 'sim_states.npy', allow_pickle=False)
        assert state.shape == (129, initial.size) and state.dtype == initial.dtype and np.isfinite(state).all()
        assert cpu.npexact(state[0], initial, np) and (folder / 'input_00.png').read_bytes() == (inputs / 'input.png').read_bytes()
        assert all(r['exact'] for r in read(folder / 'initial_checks.json'))
        assert all(x is True for x in read(folder / 'initial_controller_checks.json').values())
        assert read(folder / 'initial_contact_checks.json') == read(inputs / 'initial_contact_checks.json')
        assert strict == c['helper'].strict_windows(trajectory, result, np)
        quantified, selected = rollout.quantify(trajectory)
        assert quantified == result['per_object'] and selected == result['selected_objects']
        if objects is None:
            objects = tuple(trajectory[0]['objects'])
            arrays['strict_lift_m'] = np.empty((24, 129, len(objects)), np.float64)
            arrays['strict_both_contacts'] = np.empty((24, 129, len(objects)), bool)
            arrays['strict_contact_and_lift'] = np.empty((24, 129, len(objects)), bool)
        assert tuple(trajectory[0]['objects']) == objects
        firsts, found = [], []
        for oi, name in enumerate(objects):
            xyz = np.asarray([r['objects'][name] for r in trajectory])
            contact = np.asarray([[r['finger_contacts'][name]['left'], r['finger_contacts'][name]['right']] for r in trajectory], bool)
            assert xyz.shape == (129, 3) and np.isfinite(xyz).all()
            assert contact.all(1).tolist() == [r['finger_contacts'][name]['both'] for r in trajectory]
            lift, valid = xyz[:, 2] - xyz[0, 2], contact.all(1) & ((xyz[:, 2] - xyz[0, 2]) > .02)
            starts = [i for i in range(125) if bool(valid[i:i + 5].all())]
            assert starts == strict['objects'][name]['five_record_window_start_steps']
            assert strict['objects'][name]['first_window'] == (list(range(starts[0], starts[0] + 5)) if starts else None)
            assert quantified[name]['first_2cm_for_5frames'] == (starts[0] if starts else None)
            if starts:
                found.append(name)
                firsts.append(starts[0])
            arrays['strict_lift_m'][ci, :, oi], arrays['strict_both_contacts'][ci, :, oi], arrays['strict_contact_and_lift'][ci, :, oi] = lift, contact.all(1), valid
        assert found == selected and result['first_selection_step'] == (min(firsts) if firsts else None)
        with np.load(inputs / 'controller.npz', allow_pickle=False) as archive:
            original_controller = {key: archive[key].copy() for key in archive.files}
        typed = {}
        for step, name in ((0, 'initial_controller.npz'), (16, 'controller-step16.npz')):
            with np.load(folder / name, allow_pickle=False) as archive:
                typed[step] = {key: archive[key].copy() for key in archive.files}
            assert set(typed[step]) == set(original_controller)
            for key, value in typed[step].items():
                assert np.isfinite(value).all() and cpu.npexact(value, np.asarray(trajectory[step]['controller'][key], dtype=value.dtype), np)
                if step == 0:
                    assert cpu.npexact(value, original_controller[key], np)
        contracts = read(folder / 'query_contract.json')
        assert len(contracts) == 8
        for query in range(8):
            name, chunk = f'chunk_{query:02d}/states.pt', folder / f'chunk_{query:02d}'
            record = cpu.checked(chunk / 'states.pt', hashes[name], torch)
            scan.validate_record(record, torch)
            if schedule is None:
                schedule = {key: record[key] for key in ('timesteps', 'sigmas')}
            for key, value in schedule.items():
                runtime.require_exact(record[key], value, '192 complete schedules/' + key)
            meta, contract = read(chunk / 'metadata.json'), contracts[query]
            assert meta == contract['server_metadata'] and contract['request'] == dict(trial=case, query=query, seed=195 + query, prompt=rule['policy_prompt'])
            assert meta['prompt'] == c['probe'].PROMPTS[goal] and meta['seed'] == 195 + query and meta['task_index'] == 7
            for key, value in dict(pixels='official', schedule='official', use_system_prompt=False,
                domain_name='libero', domain_id=5, raw_action_dim=10, chunk_size=16, num_inference_steps=30).items():
                assert meta[key] == value
            assert meta['component_intervention_in_this_execution'] is meta['attention_mask_intervention_in_this_execution'] is False
            assert meta['input_png_sha256'] == sha(folder / f'input_{query:02d}.png')
            assert meta['saved_q0_prediction'] is (query == 0) and meta['current_execution_model_calls'] == (0 if query == 0 else 30)
            assert meta['model_calls'] == 30 and meta['prepare_calls'] == 1 and contract['response_saved_actions_exact'] is True
            assert contract['paired_noise_checked_exact'] is (query > 0 and ci > 0)
            assert torch.count_nonzero(record['action_states'][..., 10:]) == 0
            if query == 0:
                source = frozen['q0_sources'][case]
                runtime.require_exact(record, cpu.checked(Path(source['folder']) / 'states.pt', source['files_sha256']['states.pt'], torch), 'whole cached q0')
                q0[case] = record
            elif query in later:
                runtime.require_exact(record['pure_noise'], later[query], 'whole later two-draw pair')
                pairs += 1
            else:
                later[query] = record['pure_noise']
            if rule['whole_reference_control']:
                ref = frozen['whole_physical_references'][case]
                runtime.require_exact(record, cpu.checked(Path(ref['folder']) / name, ref['files_current_sha256'][name], torch), 'six canonical all8 whole records')
                assert (folder / f'input_{query:02d}.png').read_bytes() == (Path(ref['folder']) / f'input_{query:02d}.png').read_bytes()
            normed = np.asarray(read(chunk / 'normalized_actions.json'))
            assert normed.shape == (16, 10) and normed.dtype == np.float64 and np.isfinite(normed).all()
            assert cpu.npexact(normed, np.asarray(record['actions'].tolist()), np)
            raw = normed * scale + offset
            command = np.asarray([gripper(forward(item, '6d').tolist(), 'zero_one') for item in raw])
            assert command.shape == (16, 7) and command.dtype == np.float64 and np.isfinite(command).all()
            section = slice(query * 16, (query + 1) * 16)
            for key, value in (('normalized_actions', normed), ('raw_actions', raw), ('native_commands', command), ('executed_commands', np.clip(command, -1, 1))):
                arrays[key][ci, section] = value
        for key, name in (('normalized_actions', 'normalized_actions.json'), ('raw_actions', 'denormalized_actions.json'), ('native_commands', 'actions.json'), ('executed_commands', 'executed_actions.json')):
            assert cpu.npexact(np.asarray(read(folder / name)), arrays[key][ci], np)
        if rule['whole_reference_control']:
            ref = frozen['whole_physical_references'][case]
            assert cpu.npexact(state, np.load(Path(ref['folder']) / 'sim_states.npy', allow_pickle=False), np)
            for name in ('trajectory.json', 'actions.json', 'executed_actions.json', 'normalized_actions.json', 'denormalized_actions.json', *[f'input_{q:02d}.png' for q in range(8)]):
                assert (folder / name).read_bytes() == (Path(ref['folder']) / name).read_bytes()
            for step, name in ((0, 'initial_controller.npz'), (16, 'controller-step16.npz')):
                with np.load(Path(ref['folder']) / name, allow_pickle=False) as archive:
                    assert set(typed[step]) == set(archive.files) and all(cpu.npexact(value, archive[key], np) for key, value in typed[step].items())
            whole.append(case)
        a, b = (np.asarray(trajectory[0]['objects'][name]) for name in ('milk_1', 'cream_cheese_1'))
        direction = a - b
        length = float(np.linalg.norm(direction))
        assert length > 1e-6
        direction /= length
        displacement = np.asarray(trajectory[16]['eef_xyz']) - np.asarray(trajectory[0]['eef_xyz'])
        actual = c['comparisons']['cases'][ci]
        assert actual['selected_objects'] == selected and actual['per_object'] == quantified
        assert actual['strict_requested_target_selected'] is (selected == [c['probe'].TARGETS[goal]])
        assert actual['first_selection_window_start'] == result['first_selection_step'] and actual['strict_windows_sha256'] == hashes['strict-windows.json']
        assert actual['eef_q0_displacement_m'] == displacement.tolist() and actual['initial_cheese_to_milk_unit_direction'] == direction.tolist()
        assert actual['initial_object_separation_m'] == length and actual['q0_eef_projection_toward_milk_from_cheese_m'] == float(np.dot(displacement, direction))
        physical.append(dict(case=case, scene=scene, goal=goal, V=row['vision_noise_source_seed'], arm=row['arm'], selected_objects=selected,
            requested_target=c['probe'].TARGETS[goal], requested_target_only=selected == [c['probe'].TARGETS[goal]], first_selection_step=result['first_selection_step'],
            per_object=quantified, q0_eef_projection_toward_milk_from_cheese_m=actual['q0_eef_projection_toward_milk_from_cheese_m']))
        print('[MATCHED-ANALYSIS] physical ' + case, flush=True)
    assert pairs == 161 and len(whole) == 6 and len(q0) == 24
    arrays['object_names'] = np.asarray(objects)
    return q0, arrays, physical, runtime, components, scan, dict(actual_model_records=192, actual_converted_actions=3072,
        actual_later_whole_two_draw_pairs=pairs, six_canonical_whole_controls=whole, converter_sources=converter_sources)


def offline_attention(dispatch, np):
    """Actual post-RoPE Q/K -> FP64 estimate. No official dispatch or model call."""
    q, k = dispatch['Qaction16'][0].double().numpy(), dispatch['K'][0].double().numpy()
    assert q.shape == (16, 32, 128) and k.shape == (388, 8, 128)
    # Four consecutive Q heads share each actual KV head, as enable_gqa specifies.
    expanded = np.repeat(k, 4, axis=1)
    logits = np.einsum('ahd,khd->ahk', q, expanded, optimize=True) / np.sqrt(128.)
    logits -= logits.max(-1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(-1, keepdims=True)
    s, u = dispatch['target_span']['target_positions'][0], dispatch['indexes']['und_len']
    groups = (range(s), range(s, s + 2), range(s + 2, u), range(u, u + 50), range(u + 50, u + 250), range(u + 250, u + 266))
    assert [i for group in groups for i in group] == list(range(388)) and u == 122
    mass = np.stack([weights[..., list(group)].sum(-1) for group in groups], -1)
    # Ordinary FP64 summation of388 nonnegative terms is not bit exact.
    tolerance = 388 * np.finfo(np.float64).eps
    assert np.isfinite(weights).all() and np.isfinite(mass).all() and np.allclose(mass.sum(-1), 1., rtol=0, atol=tolerance)
    return mass, [len(group) for group in groups]


def internal_audit(c, records, runtime, components, scan, torch, np):
    cpu, language, producer = c['cpu'], c['language'], c['producer']
    original = {}
    for v in VISIONS:
        source = c['frozen']['producer_source_cases'][str(v)]
        original[v] = cpu.checked(Path(source['q0_folder']) / 'states.pt', source['q0_files_current_sha256']['states.pt'], torch)
    arrays = dict(b36_action_actual_BF16_uint16=np.empty((24, 30, 16, 4096), np.uint16),
        t0_offline_FP64_keygroup_mass=np.empty((24, 16, 32, 6), np.float64))
    captures, boundaries, suffixes, sources, native_infos = {}, {}, {}, [], {}
    counts = {key: 0 for key in c['probe'].expected_counts()}
    for ci, row in enumerate(c['rows']):
        case, folder = row['case'], producer / row['case']
        native = records[label(row['scene'], row['goal'], row['vision_noise_source_seed'], 'native')]
        audit = cpu.checked(folder / 'noise-audit.pt', row['files_sha256']['noise-audit.pt'], torch)
        c['adapter'].verify_saved(records[case], audit, folder, native, original, row['vision_noise_source_seed'], runtime, components, language)
        report, lc = read(folder / 'observer-report.json'), cpu.checked(folder / 'language-contract.pt', row['files_sha256']['language-contract.pt'], torch)
        assert report['counts'] == row['actual_dispatch_counts'] == c['probe'].expected_counts(row['arm'])
        for key in counts:
            counts[key] += report['counts'][key]
        assert lc['scene'] == row['scene'] and lc['goal'] == row['goal'] and lc['case'] == case
        assert lc['language']['target_token_ids'] == c['probe'].TARGET_IDS[row['goal']]
        runtime.require_exact(lc['first_model_kwargs'], records[case]['model_input'], 'actual language contract first input')
        observed = c['probe'].target_span(records[case]['model_input'], row['goal'],
            lc['language']['prompt_token_ids'], language, runtime)
        runtime.require_exact(observed, lc['language'], 'Actual packed eleven-token instruction and two target IDs')
        index_data = cpu.checked(folder / 'indexes-and-masks.pt', row['files_sha256']['indexes-and-masks.pt'], torch)
        index = index_data['indexes']
        assert index['und_len'] == 122 and index['gen_len'] == 266 and index['full_joint_length'] == 388
        for key, expected in dict(current_rows=range(50), future_rows=range(50, 250), action_rows=range(250, 266),
            text_keys=range(122), current_keys=range(122, 172), future_keys=range(172, 372), action_keys=range(372, 388)).items():
            runtime.require_exact(index[key], torch.tensor(list(expected)), 'Independent actual token index/' + key)
        runtime.require_exact(index_data['masks'], language.dynamic_masks(index, torch), 'Independent bool masks from actual index mapping')
        if row['arm'] == 'native':
            native_infos[(row['scene'], row['goal'], row['vision_noise_source_seed'])] = lc
        calls, dispatches = [], []
        for step, (bi, di) in enumerate(zip(row['boundary_files'], row['dispatch_files'])):
            data, dispatch = cpu.checked(folder / bi['file'], bi['sha256'], torch), cpu.checked(folder / di['file'], di['sha256'], torch)
            assert data['scene'] == dispatch['scene'] == row['scene'] and data['goal'] == dispatch['goal'] == row['goal']
            assert data['case_arm'] == dispatch['arm'] == row['arm'] and data['step'] == dispatch['step'] == step
            runtime.require_exact(data['actual_pack'], lc['actual_pack'][step], 'All30 actual recorded structural pack')
            runtime.require_exact(dispatch['indexes'], index, 'All30 captured QKV actual index map')
            runtime.require_exact(data['actual_model_action_timesteps'], data['actual_pack']['metadata']['action_timesteps'], 'Actual model timestep, not inferred from rounded scheduler integers')
            action = data['action_hidden'][36] if step == 0 else data['action_hidden']
            bits = action.contiguous().view(torch.int16).numpy().view(np.uint16)
            arrays['b36_action_actual_BF16_uint16'][ci, step] = bits
            assert bits.tobytes(order='C') == cpu.raw(action, torch)
            assert len(data['site_events']) == 36
            for layer, event in enumerate(data['site_events']):
                active = row['arm'] == 'full_hardmask' and step <= 14 and layer <= 8
                assert event['step'] == step and event['layer_zero_based'] == layer and event['selected_window_cut_active'] is active
                assert event['extra_all_allowed_calls'] == (0 if row['arm'] == 'native' else 1) and event['extra_cut_calls'] == int(active)
                assert event['to_add_out_actual_calls'] == 1 and event['actual_dispatch_return_flatten_equals_pre_W_byte_exact'] is True
                if active:
                    assert event['all_unblocked_GEN_rows_vs_same_input_all_allowed_byte_exact'] is True
            if step == 0:
                boundaries[case] = data
                mass, sizes = offline_attention(dispatch, np)
                arrays['t0_offline_FP64_keygroup_mass'][ci] = mass
                arrays['t0_offline_keygroup_counts'] = np.asarray(sizes, np.int64)
            calls.append(dict(Q=dispatch['Qaction16'], Znative=dispatch['Znative_action16'],
                Zreturned=dispatch['Znative_action16'] if row['arm'] == 'native' else dispatch['Zmerged_action16']))
            if row['arm'] == 'native':
                for prefix in ('K', 'V'):
                    assert dispatch[prefix + 'causal_suffix'] is not None
                suffixes[(row['goal'], row['scene'], row['vision_noise_source_seed'], step)] = dict(
                    K=dispatch['Kcausal_suffix'], V=dispatch['Vcausal_suffix'], indexes=dispatch['causal_suffix_key_indexes'])
            dispatches.append(dict(step=step, file=di['file'], file_sha256=di['sha256'],
                sparse_raw=dispatch['sparse_metadata'], target_positions=lc['language']['target_positions']))
        captures[case] = calls
        sources.append(dict(case=case, files_sha256=row['files_sha256'], boundary_files=row['boundary_files'], dispatch_files=dispatches))
        print('[MATCHED-ANALYSIS] internal ' + case, flush=True)
    assert counts == c['probe'].expected_counts() and len(suffixes) == 240 and len(boundaries) == 24
    assert c['probe'].matched_contract(native_infos, runtime) == read(producer / 'matched-language-contract.json'), 'Recompute eight native raw matched-name packing gates'
    arrays['t0_offline_FP64_mass_partition_error'] = arrays['t0_offline_FP64_keygroup_mass'].sum(-1) - 1.
    return arrays, captures, boundaries, suffixes, sources


def suffix_statistics(suffixes, torch, np, cpu):
    rows, conclusions = [], []
    order = [(scene, v, step) for scene in SCENES for v in VISIONS for step in range(30)]
    arrays = dict(suffix_comparison_order=np.asarray(order, dtype='U8'))
    for goal in GOALS:
        raw = []
        reference_item = suffixes[(goal, SCENES[0], VISIONS[0], 0)]
        for scene, v, step in order:
            item = suffixes[(goal, scene, v, step)]
            raw.append((cpu.raw(item['K'], torch), cpu.raw(item['V'], torch)))
        matrix = np.asarray([[a == b for b in raw] for a in raw], bool)
        arrays[goal + '_same_name_suffix_allpair_KV_byte_equal'] = matrix
        for prefix in ('K', 'V'):
            reference_rows = [cpu.raw(row, torch) for row in reference_item[prefix][0]]
            arrays[f'{goal}_same_name_{prefix}_suffix_row_byte_equal_to_first_reference'] = np.asarray([
                [cpu.raw(row, torch) == ref for row, ref in zip(suffixes[(goal, scene, v, step)][prefix][0], reference_rows)]
                for scene, v, step in order], bool)
        for scene, v in ((s, v) for s in SCENES for v in VISIONS):
            reference = suffixes[(goal, scene, v, 0)]
            for step in range(30):
                item = suffixes[(goal, scene, v, step)]
                assert cpu.exact(item['indexes'], reference['indexes'], torch)
                rows.append(dict(goal=goal, scene=scene, V=v, step=step,
                    K_vs_same_case_t0_byte_equal=cpu.exact(item['K'], reference['K'], torch),
                    V_vs_same_case_t0_byte_equal=cpu.exact(item['V'], reference['V'], torch)))
        conclusions.append(dict(goal=goal, all120_same_name_suffix_KV_arrays_byte_identical=bool(matrix.all()),
            unique_K_byte_patterns=len({a for a, _ in raw}), unique_V_byte_patterns=len({b for _, b in raw}),
            compared_axes='X12/X15, V195/V198, all t0..29; equality is measured, not assumed.'))
    reference = next(iter(suffixes.values()))['indexes']
    arrays['actual_causal_suffix_key_indexes'] = reference.numpy()
    for item in suffixes.values():
        assert cpu.exact(item['indexes'], reference, torch), 'Matched-name suffix coordinates must be comparable'
    for scene in SCENES:
        for v in VISIONS:
            for prefix in ('K', 'V'):
                # Rowwise all30 name differences, with actual KV grouping retained.
                values, equal = [], []
                for step in range(30):
                    a, b = (suffixes[(goal, scene, v, step)][prefix] for goal in GOALS)
                    values.append(rms_delta(a[0], b[0], np))
                    equal.append([cpu.raw(x, torch) == cpu.raw(y, torch) for x, y in zip(a[0], b[0])])
                arrays[f'suffix_{prefix}_language_RMS_{scene}_V{v}'] = np.stack(values)
                arrays[f'suffix_{prefix}_language_row_byte_equal_{scene}_V{v}'] = np.asarray(equal, bool)
    return arrays, dict(same_name_complete_arrays=conclusions, same_case_time_checks=rows,
        inference='If same-name suffix K/V are exact, this captured L35 supply did not change with scene/noise/time. Reader, other unverified supply or downstream changes remain possible; before-target UND raw is saved only at t0. No object-identity or causal-head conclusion.')


def statistics(c, captures, boundaries, np):
    arrays, rows = {}, []
    for arm in ARMS:
        for scene in SCENES:
            for v in VISIONS:
                a, b = (captures[label(scene, goal, v, arm)] for goal in GOALS)
                for key in ('Q', 'Znative', 'Zreturned'):
                    arrays[f'head_{key}_language_RMS_{scene}_V{v}_{arm}'] = np.stack([rms_delta(x[key][0], y[key][0], np) for x, y in zip(a, b)])
                for group in ('und', 'current', 'action'):
                    key = group + '_hidden'
                    values = np.stack([rms_delta(boundaries[label(scene, GOALS[0], v, arm)][key][i], boundaries[label(scene, GOALS[1], v, arm)][key][i], np) for i in range(37)])
                    arrays[f't0_{group}_language_per_token_RMS_{scene}_V{v}_{arm}'] = values
                    arrays[f't0_{group}_language_RMS_{scene}_V{v}_{arm}'] = np.sqrt(np.mean(values * values, -1))
        for goal in GOALS:
            for v in VISIONS:
                a, b = (captures[label(scene, goal, v, arm)] for scene in SCENES)
                for key in ('Q', 'Znative', 'Zreturned'):
                    arrays[f'head_{key}_scene_RMS_{goal}_V{v}_{arm}'] = np.stack([rms_delta(x[key][0], y[key][0], np) for x, y in zip(a, b)])
                for group in ('und', 'current', 'action'):
                    key = group + '_hidden'
                    values = np.stack([rms_delta(boundaries[label(SCENES[0], goal, v, arm)][key][i], boundaries[label(SCENES[1], goal, v, arm)][key][i], np) for i in range(37)])
                    arrays[f't0_{group}_scene_per_token_RMS_{goal}_V{v}_{arm}'] = values
                    arrays[f't0_{group}_scene_RMS_{goal}_V{v}_{arm}'] = np.sqrt(np.mean(values * values, -1))
    for row in c['rows']:
        case, scene, goal, v = row['case'], row['scene'], row['goal'], row['vision_noise_source_seed']
        ci = [r['case'] for r in c['rows']].index(case)
        # Actual normalized final q0 positions are descriptive, not solver replays.
        rows.append(dict(case=case, t0_offline_target2_mass_mean=float(c['_mass'][ci, :, :, 1].mean()),
            language_action_b36_RMS=float(arrays[f't0_action_language_RMS_{scene}_V{v}_{row["arm"]}'][36]),
            scene_action_b36_RMS=float(arrays[f't0_action_scene_RMS_{goal}_V{v}_{row["arm"]}'][36])))
    return arrays, rows


def primary(c, physical):
    by_case = {r['case']: r for r in physical}
    cases = [by_case[name] for name in c['probe'].PRIMARY['cases']]
    verdict = 'supported' if all(r['requested_target_only'] for r in cases) else 'rejected'
    sham = []
    for scene in SCENES:
        for goal in GOALS:
            for v in VISIONS:
                a, b = (by_case[label(scene, goal, v, arm)] for arm in ('native', 'all_allowed'))
                sham.append(dict(scene=scene, goal=goal, V=v, native=a['selected_objects'], all_allowed=b['selected_objects'], equal=a['selected_objects'] == b['selected_objects']))
    gate = all(r['equal'] for r in sham)
    assert c['ec']['primary_prediction'] == c['comparisons']['primary_prediction'] == verdict
    assert c['ec']['numerical_sham_classification_gate_passed'] is gate
    assert c['ec']['all_four_X15_native_cases_strict_only_requested_target'] is (verdict == 'supported')
    saved = read(c['execution'] / 'primary-prediction.json')
    assert saved['prediction'] == verdict and saved['numerical_sham_classification_gate_passed'] is gate
    return dict(prediction=verdict, four_X15_native_cases=cases, numerical_sham_is_separate_cut_attribution_diagnostic=True,
        numerical_sham_classification_gate_passed=gate, all_eight_sham_comparisons=sham,
        all24_actual_physics_complete=True, original_milk_instruction_at15cm_repaired=False)


def plots(out, c, arrays, physical, np, plt):
    fig, ax = plt.subplots(figsize=(12, 11), constrained_layout=True)
    cell = [[r['scene'], str(r['V']), r['goal'].replace('_', ' '), r['arm'], ', '.join(r['selected_objects']) or 'none', str(r['first_selection_step'])] for r in physical]
    ax.axis('off')
    table = ax.table(cellText=cell, colLabels=['Scene', 'V', 'Requested', 'Q0 arm', 'Strict object(s)', 'First window'], loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.6)
    ax.set_title('All24 real executions: bilateral contact + lift >2cm for five records\nOnly four X15 native rows test the registered donor prediction')
    fig.savefig(out / 'strict-target-table.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    for ax, (scene, goal, v) in zip(axes.flat, ((s, g, v) for s in SCENES for g in GOALS for v in VISIONS)):
        ci = [r['case'] for r in c['rows']].index(label(scene, goal, v, 'native'))
        im = ax.imshow(arrays['t0_offline_FP64_keygroup_mass'][ci, :, :, 1], origin='lower', aspect='auto', vmin=0)
        ax.set(title=f'{scene} {goal} V{v}', xlabel='Actual Q head (0..31)', ylabel='Action query (0..15)')
        fig.colorbar(im, ax=ax, label='Offline FP64 target2 mass; 2 of388 keys')
    fig.suptitle('L35 t0 native target-word routing estimate; not captured backend weights or causal importance')
    fig.savefig(out / 'L35-offline-target-reading.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
    for ri, group in enumerate(('und', 'current', 'action')):
        for scene in SCENES:
            for v in VISIONS:
                axes[ri, 0].plot(arrays[f't0_{group}_language_RMS_{scene}_V{v}_native'], label=f'{scene} V{v}')
        for goal in GOALS:
            for v in VISIONS:
                axes[ri, 1].plot(arrays[f't0_{group}_scene_RMS_{goal}_V{v}_native'], label=f'{goal} V{v}')
        for col, title in enumerate(('milk box - cream cheese', 'X12 - X15')):
            axes[ri, col].set(title=f'Native t0 {group}: {title}', xlabel='Actual block boundary (0..36)', ylabel='Absolute hidden RMS')
            axes[ri, col].legend(fontsize=7)
    fig.suptitle('Matched contrasts at t0; masked-arm curves and all30 actual head changes are in NPZ')
    fig.savefig(out / 't0-language-versus-scene.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    c = seal(args.output.resolve())  # No partial-data analysis or placeholder execution.
    out = args.output.resolve() / STAGE
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
        records, arrays, physical, runtime, components, scan, physical_counts = physical_audit(c, torch, np)
        internal, captures, boundaries, suffixes, sources = internal_audit(c, records, runtime, components, scan, torch, np)
        arrays.update(internal)
        suffix_arrays, suffix = suffix_statistics(suffixes, torch, np, c['cpu'])
        arrays.update(suffix_arrays)
        c['_mass'] = arrays['t0_offline_FP64_keygroup_mass']
        contrasts, descriptions = statistics(c, captures, boundaries, np)
        arrays.update(contrasts)
        result = primary(c, physical)
        arrays['case_order'] = np.asarray([r['case'] for r in c['rows']])
        arrays['offline_keygroup_names'] = np.asarray(GROUPS)
        np.savez_compressed(out / 'arrays.npz', **arrays)
        with np.load(out / 'arrays.npz', allow_pickle=False) as saved:
            assert set(saved.files) == set(arrays)
            for key, value in arrays.items():
                reread = saved[key]
                assert reread.dtype == value.dtype and reread.shape == value.shape and reread.tobytes(order='C') == value.tobytes(order='C')
            exported_b36 = saved['b36_action_actual_BF16_uint16']
            for ci, row in enumerate(c['rows']):
                for step, entry in enumerate(row['boundary_files']):
                    data = c['cpu'].checked(c['producer'] / row['case'] / entry['file'], entry['sha256'], torch)
                    actual = data['action_hidden'][36] if step == 0 else data['action_hidden']
                    assert exported_b36[ci, step].tobytes(order='C') == c['cpu'].raw(actual, torch)
        assert (out / 'arrays.npz').stat().st_size < 256 * 1024 ** 2
        plots(out, c, arrays, physical, np, plt)
        provenance = dict(script_sha256=sha(Path(__file__)), producer_script_sha256=PROBE_SHA, execution_script_sha256=EXECUTION_SHA,
            CPU_helper_sha256=CPU_SHA, language_helper_sha256=LANGUAGE_SHA,
            producer_documents_sha256={name: sha(c['producer'] / (name + '.json')) for name in ('complete', 'results', 'protocol', 'provenance', 'sources', 'primary-prediction', 'matched-language-contract')},
            execution_documents_sha256={name: sha(c['execution'] / (name + '.json')) for name in ('complete', 'controls', 'comparisons', 'summary', 'provenance')},
            actual_case_sources=sources, torch_version=str(torch.__version__), numpy_version=str(np.__version__), scope=SCOPE)
        write(out / 'provenance.json', provenance)
        write(out / 'analysis.json', dict(state='complete', primary=result, physical_cases=physical, physical_counts=physical_counts,
            L35_actual_native_causal_suffix=suffix, actual_contrasts=descriptions,
            arrays=dict(raw_b36='BF16 bits in uint16, not integer-valued activity', head_RMS='[30,16,32], reduce128 channels only',
                head_endpoints='Q=actual action query; Znative=actual original dispatch output (masked arms do not send it to W); Zreturned=actual action rows returned to W, Zmerged for masked arms.',
                t0_hidden_RMS='[37,tokens] and pooled token/channel RMS, no probability interpretation',
                offline_mass='[24,16,32,6], actual Q/K -> stable FP64 softmax; not actual BF16 backend attention'),
            actual_boundary_files=720, actual_dispatch_files=720, actual_native_suffix_files=240,
            model_calls=0, official_dispatch_calls=0, solver_calls=0, physics_calls=0, CUDA_initialized=torch.cuda.is_initialized(), scope=SCOPE))
        files = ('arrays.npz', 'analysis.json', 'provenance.json', 'strict-target-table.png', 'L35-offline-target-reading.png', 't0-language-versus-scene.png')
        assert not torch.cuda.is_initialized()
        write(out / 'complete.json', dict(state='complete', script_sha256=sha(Path(__file__)), primary_prediction=result['prediction'],
            physical_counts=physical_counts, actual_boundary_files=720, actual_dispatch_files=720, actual_native_suffix_files=240,
            files_sha256={name: sha(out / name) for name in files}, elapsed_s=time.perf_counter() - started,
            model_calls=0, official_dispatch_calls=0, solver_calls=0, physics_calls=0, CUDA_initialized=False, scope=SCOPE))
        print('[MATCHED-ANALYSIS] complete ' + str(out), flush=True)
    except Exception as exc:
        write(out / 'failed.json', dict(state='failed', error=str(exc), traceback=traceback.format_exc(), script_sha256=sha(Path(__file__))))
        raise


if __name__ == '__main__':
    main()
