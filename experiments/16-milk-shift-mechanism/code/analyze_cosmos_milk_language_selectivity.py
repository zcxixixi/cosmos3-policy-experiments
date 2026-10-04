"""CPU-only audit of completed language x frozen-nine-layer interventions.

Require all eighteen q0 predictions and physical executions. Create only an
absent BASE/future-language-selectivity-analysis. The registered primary is
t0 action boundary36, milk versus cheese, separately for V195 and V198.
Later states are descriptive total effects of dynamically different inputs.
Canonical cream cheese is auxiliary. No model, attention, solver or physics
execution. Array distances are not object probabilities or brain regions.
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
PROBE_SHA = '74ac42db450a1a7cd308c837bf75049cd2dcae858a4869dab1f0f60450c69b13'
EXECUTION_SHA = '3c42a3a7137ba2dffc78699046198acbb1d924e229fdaf92bf516afe11530437'
CPU_HELPER_SHA = 'ca88ff9c7ee62beef9851fdfaea1c9fddd6e33c0f99058ebee16f26a28a4af9c'
ARMS = ('native', 'all_allowed', 'full_hardmask')
GOALS = ('milk', 'cheese', 'cream_cheese')
VISIONS = (195, 198)
CURRENT_STEPS = (0, 14, 15, 29)
PROP = ('block_input_current50', 'to_add_out_current50', 'attention_residual_current50',
        'mlp_moe_gen_current50', 'block_output_current50')
FLOOR = 1e-12
SCOPE = ('Fixed X12, A195 and two actual V noise sources. Main instructions milk/cheese '
         'must have equal length and exactly one different target token. Canonical cream cheese '
         'is auxiliary, with different text length. Primary t0 action boundary36 precedes final norm; '
         'later denoising and closed-loop differences contain dynamically changed inputs. '
         'All masked q0 arms share union66 all-allowed numerical background through t29; '
         'the future-to-current cut is restricted to t0..14 and L0..8. Hard masks also redistribute '
         'remaining attention weights. Common hidden displacement does not imply fixed world position, '
         'unused semantics, successful grasp, language/motor double dissociation or a functional brain region. '
         'Cross-arm MLP differences contain changed input and are not a same-input overwrite test. '
         'No new inference, attention replay, solver update, environment or physics.')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def finite(value, shape, dtype, torch):
    assert tuple(value.shape) == tuple(shape) and value.dtype == dtype and value.device.type == 'cpu'
    assert bool(torch.isfinite(value).all())


def norm(value, np):
    return float(np.sqrt(np.sum(value * value, dtype=np.float64)))


def geometry(a, b, np):
    """Per leading index; reduce only the last token/channel axes."""
    an = np.sqrt(np.sum(a * a, axis=(-2, -1), dtype=np.float64))
    bn = np.sqrt(np.sum(b * b, axis=(-2, -1), dtype=np.float64))
    dot = np.sum(a * b, axis=(-2, -1), dtype=np.float64)
    cosine = np.full_like(an, np.nan)
    np.divide(dot, an * bn, out=cosine, where=(an > FLOOR) & (bn > FLOOR))
    return an, bn, cosine


def finite_json(value):
    if isinstance(value, float):
        return value if value == value and abs(value) != float('inf') else None
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [finite_json(item) for item in value]
    return value


def verdict(rows):
    if any(row['valid'] and not row['thresholds_pass'] for row in rows):
        return 'rejected'
    if all(row['valid'] and row['thresholds_pass'] for row in rows):
        return 'supported'
    return 'insufficient'


def seal(base):
    for pin in (PROBE_SHA, EXECUTION_SHA):
        assert len(pin) == 64 and all(c in '0123456789abcdef' for c in pin), 'Source not frozen; analysis forbidden'
    pp = ROOT / 'work/probe_cosmos_milk_language_selectivity.py'
    ep = ROOT / 'work/run_cosmos_milk_language_selectivity.py'
    cp = ROOT / 'work/analyze_cosmos_milk_future_current_layer_groups.py'
    assert sha(pp) == PROBE_SHA and sha(ep) == EXECUTION_SHA and sha(cp) == CPU_HELPER_SHA
    adapter, cpu = load('language_analysis_execution', ep), load('language_analysis_cpu_checks', cp)
    frozen, probe, simulator, helper, cross, execution = adapter.prepared(base)
    producer = base / probe.STAGE
    for path in (producer / 'failed.json', execution / 'wrapper_failed.json', execution / 'simulator_failed.json',
                 execution / 'server/failed.json', execution / 'closed-loop/failed.json'):
        assert not path.exists(), str(path)
    pc, results, protocol = (read(producer / (name + '.json')) for name in ('complete', 'results', 'protocol'))
    assert pc['state'] == results['state'] == 'complete' and pc['script_sha256'] == PROBE_SHA
    assert pc['fresh_q0_predictions'] == 18 and pc['fresh_model_forwards'] == 540
    assert pc['actual_selected_site_captures'] == 2430 and pc['total_official_dispatch_calls'] == 52650
    assert results['official_dispatch_counts'] == pc['official_dispatch_counts'] == probe.expected_counts()
    counts = pc['official_dispatch_counts']
    assert counts['native_UND_dispatch'] == counts['native_GEN_dispatch'] == 19440
    assert counts['all_allowed_dispatch'] == 12960 and counts['cut_dispatch'] == 810
    for name in ('results', 'protocol', 'provenance', 'sources', 'primary-prediction', 'primary-language-contract'):
        assert pc[name.replace('-', '_') + '_sha256'] == sha(producer / (name + '.json'))
    assert protocol['primary_hypothesis'] == probe.PRIMARY
    assert probe.PRIMARY['cosine_min'] == .95 and probe.PRIMARY['relative_delta_difference_max'] == .10
    assert probe.PRIMARY['numerical_background_multiplier'] == 10
    assert pc['full_current_boundary_and_kwargs_tuple_raw_steps'] == list(CURRENT_STEPS)
    assert pc['full_UND_boundary_raw_steps'] == [0]
    rows = results['cases']
    assert len(rows) == 18 and len({row['case'] for row in rows}) == 18
    assert {(r['arm'], r['goal'], r['vision_noise_source_seed']) for r in rows} == {
        (a, g, v) for a in ARMS for g in GOALS for v in VISIONS}
    labels = [r['case'] for r in rows]
    assert labels == frozen['trial_order'] == protocol['trial_order']
    ec, controls, summary, comparisons = (read(execution / (name + '.json')) for name in
        ('complete', 'controls', 'summary', 'comparisons'))
    assert ec['state'] == 'complete' and ec['script_sha256'] == EXECUTION_SHA and ec['trial_order'] == labels
    assert ec['physical_trials'] == ec['saved_q0_predictions'] == 18 and ec['native_predictions'] == 126
    assert ec['fresh_model_forwards'] == 3780 and ec['producer_q0_forwards'] == 540
    assert ec['combined_two_stage_model_forwards'] == 4320 and ec['actual_later_whole_noise_pairs'] == 119
    assert ec['actual_physical_actions'] == 2304 and ec['internal_primary_evaluated'] is False
    for key in ('six_entire129_state_5JSON_8PNG_and_all8_modelrecords_controls_exact',
                'all18_cached_q0_action_and_strict_window_gates_passed', 'all18_executed_without_score_selection'):
        assert ec[key] is True
    for name, expected in ec['files_sha256'].items():
        assert sha(execution / name) == expected
    assert summary == read(execution / 'closed-loop/summary.json') and summary['completed_trials'] == labels
    assert [r['trial'] for r in summary['cases']] == [r['case'] for r in comparisons['cases']] == labels
    assert comparisons['state'] == 'all18_complete' and comparisons['internal_primary_requires_actual_array_analysis'] is True
    assert set(controls['actual_case_files_sha256']) == set(labels)
    for label, files in controls['actual_case_files_sha256'].items():
        for name, expected in files.items():
            assert sha(execution / 'closed-loop' / label / name) == expected
    server = read(execution / 'server/complete.json')
    assert server['state'] == 'complete' and server['queries'] == 144 and server['saved_q0_predictions'] == 18
    assert server['native_predictions'] == 126 and server['fresh_model_forwards'] == 3780
    assert server['actual_later_whole_noise_pairs'] == 119 and server['final_model_hooks_none'] is True
    for name, expected in server['files_sha256'].items():
        assert sha(execution / 'server' / name) == expected
    return dict(base=base, producer=producer, execution=execution, probe=probe, adapter=adapter, cpu=cpu,
        frozen=frozen, simulator=simulator, helper=helper, cross=cross, pc=pc, ec=ec, controls=controls,
        rows=rows, labels=labels, summary=summary, comparisons=comparisons, protocol=protocol)


def physical_audit(c, torch, np):
    cpu, frozen, execution, helper = c['cpu'], c['frozen'], c['execution'], c['helper']
    files = frozen['threshold_contract']['frozen_files']
    base = load('language_analysis_byte_runtime', Path(files['normal_runtime']['path']))
    runtime = object.__new__(base.NormalRuntime)  # Pure byte-check methods only; never runtime initialization.
    runtime.torch = torch
    scan = load('language_analysis_record_checks', Path(frozen['metrics_path']))
    components = load('language_analysis_pack_layout', Path(frozen['components_path']))
    rollout = load('language_analysis_controller_checks', Path(files['controller_helper']['path']))
    forward, gripper, converter_sources = cpu.converters(torch)
    stats = read(Path(files['normalizer_stats']['path']))['global_raw']
    lo, hi = np.asarray(stats['q01']), np.asarray(stats['q99'])
    assert lo.shape == hi.shape == (10,) and np.all(hi - lo > 1e-6)
    scale, offset = np.maximum(hi - lo, 1e-8) / 2, (hi + lo) / 2
    initial = np.load(execution / 'inputs/x12/state.npy', allow_pickle=False)
    with np.load(execution / 'inputs/x12/controller.npz', allow_pickle=False) as source:
        initial_controller = {key: source[key].copy() for key in source.files}
    arrays = dict(normalized_actions=np.zeros((18, 128, 10)), raw_actions=np.zeros((18, 128, 10)),
                  native_commands=np.zeros((18, 128, 7)), executed_commands=np.zeros((18, 128, 7)))
    records, physical, later, objects, whole, pairs, schedule = {}, [], {}, None, [], 0, None
    for ci, row in enumerate(c['rows']):
        label, arm, goal, v = row['case'], row['arm'], row['goal'], row['vision_noise_source_seed']
        rule, folder = frozen['trials'][label], execution / 'closed-loop' / label
        hashes = c['controls']['actual_case_files_sha256'][label]
        trajectory, result, strict = (read(folder / (name + '.json')) for name in ('trajectory', 'summary', 'strict-windows'))
        assert result == c['summary']['cases'][ci] and result['trial'] == label and result['policy_goal'] == goal
        assert len(trajectory) == 129 and [r['step'] for r in trajectory] == list(range(129))
        simstates = np.load(folder / 'sim_states.npy', allow_pickle=False)
        assert simstates.shape == (129, initial.size) and simstates.dtype == initial.dtype and np.isfinite(simstates).all()
        assert cpu.npexact(simstates[0], initial, np)
        assert (folder / 'input_00.png').read_bytes() == (execution / 'inputs/x12/input.png').read_bytes()
        assert all(r['exact'] for r in read(folder / 'initial_checks.json'))
        assert all(x is True for x in read(folder / 'initial_controller_checks.json').values())
        assert read(folder / 'initial_contact_checks.json') == read(execution / 'inputs/x12/initial_contact_checks.json')
        assert strict == helper.strict_windows(trajectory, result, np)
        quantified, selected = rollout.quantify(trajectory)
        assert quantified == result['per_object'] and selected == result['selected_objects']
        if objects is None:
            objects = tuple(trajectory[0]['objects'])
            arrays.update(strict_lift_m=np.zeros((18, 129, len(objects))),
                strict_both_contacts=np.zeros((18, 129, len(objects)), dtype=bool),
                strict_contact_and_lift=np.zeros((18, 129, len(objects)), dtype=bool))
        assert tuple(trajectory[0]['objects']) == objects
        firsts, actual_selected = [], []
        for oi, name in enumerate(objects):
            xyz = np.asarray([r['objects'][name] for r in trajectory])
            contact = np.asarray([[r['finger_contacts'][name]['left'], r['finger_contacts'][name]['right']] for r in trajectory], dtype=bool)
            assert xyz.shape == (129, 3) and np.isfinite(xyz).all()
            assert contact.all(1).tolist() == [r['finger_contacts'][name]['both'] for r in trajectory]
            lift = xyz[:, 2] - xyz[0, 2]
            valid = contact.all(1) & (lift > .02)
            starts = [i for i in range(125) if bool(valid[i:i + 5].all())]
            assert starts == strict['objects'][name]['five_record_window_start_steps']
            assert strict['objects'][name]['first_window'] == (list(range(starts[0], starts[0] + 5)) if starts else None)
            assert result['per_object'][name]['first_2cm_for_5frames'] == (starts[0] if starts else None)
            firsts.extend(starts[:1])
            if starts:
                actual_selected.append(name)
            arrays['strict_lift_m'][ci, :, oi] = lift
            arrays['strict_both_contacts'][ci, :, oi] = contact.all(1)
            arrays['strict_contact_and_lift'][ci, :, oi] = valid
        assert result['first_selection_step'] == (min(firsts) if firsts else None)
        assert actual_selected == selected
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
        contracts = read(folder / 'query_contract.json')
        assert len(contracts) == 8
        query_sources = []
        for q in range(8):
            name, chunk = f'chunk_{q:02d}/states.pt', folder / f'chunk_{q:02d}'
            record = cpu.checked(chunk / 'states.pt', hashes[name], torch)
            scan.validate_record(record, torch)
            if schedule is None:
                schedule = {key: record[key] for key in ('timesteps', 'sigmas')}
            for key, value in schedule.items():
                assert cpu.exact(record[key], value, torch)
            metadata = read(chunk / 'metadata.json')
            assert metadata == contracts[q]['server_metadata']
            assert contracts[q]['request'] == dict(trial=label, query=q, seed=195 + q, prompt=rule['policy_prompt'])
            assert contracts[q]['paired_noise_checked_exact'] is (q > 0 and ci > 0)
            assert metadata['seed'] == 195 + q and metadata['prompt'] == rule['policy_prompt']
            assert metadata['input_png_sha256'] == sha(folder / f'input_{q:02d}.png')
            assert metadata['saved_q0_prediction'] is (q == 0)
            assert metadata['current_execution_model_calls'] == (0 if q == 0 else 30)
            assert metadata['model_calls'] == 30 and metadata['prepare_calls'] == 1
            assert contracts[q]['response_saved_actions_exact'] is True
            assert metadata['component_intervention_in_this_execution'] is metadata['attention_mask_intervention_in_this_execution'] is False
            assert len(record['pure_noise']) == 2
            for tensor, shape in zip(record['pure_noise'], ((1, 48, 5, 10, 20), (16, 64))):
                finite(tensor, shape, torch.float32, torch)
            finite(record['action_states'], (31, 1, 16, 64), torch.float32, torch)
            finite(record['readout'], (30, 16, 4096), torch.bfloat16, torch)
            finite(record['action_velocity'], (30, 16, 64), torch.bfloat16, torch)
            assert torch.count_nonzero(record['action_states'][..., 10:]) == 0
            if q == 0:
                source = frozen['q0_sources'][label]
                original = cpu.checked(Path(source['folder']) / 'states.pt', source['files_sha256']['states.pt'], torch)
                assert cpu.exact(record, original, torch)
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
            normed = np.asarray(read(chunk / 'normalized_actions.json'))
            assert normed.shape == (16, 10) and normed.dtype == np.float64 and np.isfinite(normed).all()
            assert cpu.npexact(normed, np.asarray(record['actions'].tolist()), np)
            raw = normed * scale + offset
            command = np.asarray([gripper(forward(r, '6d').tolist(), 'zero_one') for r in raw])
            assert command.shape == (16, 7) and command.dtype == np.float64 and np.isfinite(command).all()
            section = slice(q * 16, (q + 1) * 16)
            for key, data in (('normalized_actions', normed), ('raw_actions', raw), ('native_commands', command),
                              ('executed_commands', np.clip(command, -1, 1))):
                arrays[key][ci, section] = data
            query_sources.append(dict(query=q, states_sha256=hashes[name],
                pure_noise_raw_hashes=[cpu.tensor_meta(t, torch) for t in record['pure_noise']]))
        for key, name in (('normalized_actions', 'normalized_actions.json'), ('raw_actions', 'denormalized_actions.json'),
                          ('native_commands', 'actions.json'), ('executed_commands', 'executed_actions.json')):
            assert cpu.npexact(np.asarray(read(folder / name)), arrays[key][ci], np)
        if rule['whole_reference_control']:
            ref, old = frozen['whole_physical_references'][label], Path(frozen['whole_physical_references'][label]['folder'])
            for name, expected in ref['files_current_sha256'].items():
                assert sha(old / name) == expected
            assert cpu.npexact(simstates, np.load(old / 'sim_states.npy', allow_pickle=False), np)
            for name in ('trajectory.json', 'actions.json', 'executed_actions.json', 'normalized_actions.json',
                         'denormalized_actions.json', *[f'input_{q:02d}.png' for q in range(8)]):
                assert (folder / name).read_bytes() == (old / name).read_bytes()
            for step, name in ((0, 'initial_controller.npz'), (16, 'controller-step16.npz')):
                with np.load(old / name, allow_pickle=False) as archive:
                    assert set(typed[step]) == set(archive.files) and all(cpu.npexact(a, archive[k], np) for k, a in typed[step].items())
            whole.append(label)
        milk0, cheese0 = (np.asarray(trajectory[0]['objects'][name]) for name in ('milk_1', 'cream_cheese_1'))
        direction = milk0 - cheese0
        length = float(np.linalg.norm(direction))
        assert length > 1e-6
        direction /= length
        displacement = np.asarray(trajectory[16]['eef_xyz']) - np.asarray(trajectory[0]['eef_xyz'])
        comparison = c['comparisons']['cases'][ci]
        assert comparison['selected_objects'] == selected and comparison['per_object'] == quantified
        assert comparison['strict_requested_target_selected'] is (selected == [frozen['targets'][goal]])
        assert comparison['first_selection_window_start'] == result['first_selection_step']
        assert comparison['strict_windows_sha256'] == hashes['strict-windows.json']
        assert comparison['eef_q0_displacement_m'] == displacement.tolist()
        assert comparison['initial_cheese_to_milk_unit_direction'] == direction.tolist()
        assert comparison['initial_object_separation_m'] == length
        assert comparison['q0_eef_projection_toward_milk_from_cheese_m'] == float(np.dot(displacement, direction))
        physical.append(dict(case=label, arm=arm, goal=goal, V=v, selected_objects=selected,
            strict_requested_target_selected=selected == [frozen['targets'][goal]],
            first_selection_step=result['first_selection_step'], per_object=quantified,
            q0_eef_projection_toward_milk_from_cheese_m=comparison['q0_eef_projection_toward_milk_from_cheese_m'],
            query_sources=query_sources, whole_milk_reference_control=rule['whole_reference_control']))
        print('[LANGUAGE-ANALYSIS] physical audited ' + label, flush=True)
    assert pairs == 119 and len(whole) == 6
    reference = records[c['rows'][0]['case']]
    for record in records.values():
        assert cpu.exact(record['timesteps'], reference['timesteps'], torch)
        assert cpu.exact(record['sigmas'], reference['sigmas'], torch)
    arrays['object_names'] = np.asarray(objects)
    return records, arrays, physical, dict(actual_records=144, actual_converted_actions=2304,
        actual_later_whole_two_draw_pairs=119, six_whole_milk_reference_controls=whole,
        converter_sources=converter_sources), runtime, components, scan


def contrast_stats(native_m, native_c, allowed_m, allowed_c, cut_m, cut_c, np):
    dm, dc = cut_m - allowed_m, cut_c - allowed_c
    D, Dcut = allowed_m - allowed_c, cut_m - cut_c
    I, B = dm - dc, (dm + dc) / 2
    a, b, cos = geometry(dm, dc, np)
    ref_m, ref_c, _ = geometry(allowed_m, allowed_c, np)
    Dn, Dcutn, Dcos = geometry(D, Dcut, np)
    In = np.sqrt(np.sum(I * I, axis=(-2, -1), dtype=np.float64))
    E = np.maximum(np.sqrt(np.sum((native_m - allowed_m) ** 2, axis=(-2, -1), dtype=np.float64)),
                   np.sqrt(np.sum((native_c - allowed_c) ** 2, axis=(-2, -1), dtype=np.float64)))
    r, gain, rel_m, rel_c, rel_I = (np.full_like(a, np.nan) for _ in range(5))
    np.divide(In, a + b, out=r, where=(a + b) > FLOOR)
    np.divide(a, ref_m, out=rel_m, where=ref_m > FLOOR)
    np.divide(b, ref_c, out=rel_c, where=ref_c > FLOOR)
    np.divide(In, Dn, out=rel_I, where=Dn > FLOOR)
    np.divide(np.sum(Dcut * D, axis=(-2, -1), dtype=np.float64), Dn * Dn, out=gain, where=Dn > FLOOR)
    residual = Dcut - np.nan_to_num(gain)[..., None, None] * D
    residual_norm = np.sqrt(np.sum(residual * residual, axis=(-2, -1), dtype=np.float64))
    residual_norm = np.where(Dn > FLOOR, residual_norm, np.nan)
    valid = (a > FLOOR) & (b > FLOOR) & (Dn > FLOOR) & (Dn > 10 * E)
    assert np.allclose(I, Dcut - D, rtol=0, atol=1e-12), 'Difference-in-differences identity'
    metrics = dict(delta_m_norm=a, delta_c_norm=b, delta_cosine=cos, interaction_norm=In,
        common_shift_norm=np.sqrt(np.sum(B * B, axis=(-2, -1), dtype=np.float64)),
        interaction_ratio=r, language_D_norm=Dn, numerical_E_norm=E,
        allowed_m_reference_norm=ref_m, allowed_c_reference_norm=ref_c,
        delta_m_relative_to_allowed_m_norm=rel_m, delta_c_relative_to_allowed_c_norm=rel_c,
        interaction_relative_to_language_D_norm=rel_I, D_cut_norm=Dcutn, D_cut_vs_D_cosine=Dcos,
        D_cut_parallel_gain=gain, D_cut_orthogonal_residual_norm=residual_norm, valid=valid,
        thresholds_pass=(cos >= .95) & (r <= .10))
    vectors = dict(delta_m=dm, delta_c=dc, D=D, D_cut=Dcut, I=I, B=B)
    return metrics, vectors


def token_input_checks(c, records, torch):
    cpu = c['cpu']
    mapping = {(r['arm'], r['goal'], r['vision_noise_source_seed']): r['case'] for r in c['rows']}
    checks = []
    for v in VISIONS:
        for arm in ARMS:
            m, ch = (records[mapping[arm, goal, v]] for goal in ('milk', 'cheese'))
            a, b = m['model_input'], ch['model_input']
            ids_m, ids_c = a['input_ids'], b['input_ids']
            assert ids_m.dtype == ids_c.dtype == torch.long and ids_m.shape == ids_c.shape
            assert ids_m.numel() == int(a['und_len']) == int(b['und_len'])
            changed = torch.nonzero(ids_m.flatten() != ids_c.flatten()).flatten()
            assert changed.numel() == 1, 'Main instructions require one actual changed token'
            position = int(changed[0])
            assert cpu.exact({k: val for k, val in a.items() if k != 'input_ids'},
                             {k: val for k, val in b.items() if k != 'input_ids'}, torch)
            for key in ('pure_noise', 'timesteps', 'sigmas'):
                assert cpu.exact(m[key], ch[key], torch)
            assert cpu.exact(m['action_states'][0], ch['action_states'][0], torch)
            checks.append(dict(V=v, arm=arm, changed_packed_token_index=position,
                milk_token_id=int(ids_m.flatten()[position]), cheese_token_id=int(ids_c.flatten()[position]),
                und_len=int(a['und_len']), input_ids_raw_hashes=[cpu.tensor_meta(x, torch) for x in (ids_m, ids_c)],
                all_other_actual_initial_kwargs_and_both_noise_draws_exact=True))
    assert len({(x['changed_packed_token_index'], x['milk_token_id'], x['cheese_token_id'], x['und_len']) for x in checks}) == 1
    return checks


def noise_checks(record, audit, sources, v, cpu, torch):
    assert len(audit['calls']) == len(record['pure_noise']) == 2
    for i, call in enumerate(audit['calls']):
        seed = v if i == 0 else 195
        assert call['ordinal'] == i and call['modality'] == ('vision' if i == 0 else 'action')
        assert call['returned_source_seed'] == seed and call['original_generated_seed'] == 195
        assert cpu.exact(call['actual_returned'], record['pure_noise'][i], torch)
        assert cpu.exact(call['actual_returned'], sources[seed]['pure_noise'][i], torch)
        assert cpu.exact(call['original_generated_discarded'], sources[195]['pure_noise'][i], torch)
        before, after = call['generator_state_before'], call['generator_state_after']
        assert before.dtype == after.dtype == torch.uint8 and before.shape == after.shape
        assert not cpu.exact(before, after, torch), 'Both original draws must consume real RNG state'
        if i:
            assert cpu.exact(before, audit['calls'][i - 1]['generator_state_after'], torch)


def internal_audit(c, records, runtime, components, scan, torch, np):
    cpu, probe, producer = c['cpu'], c['probe'], c['producer']
    original = {}
    for v in VISIONS:
        source = c['frozen']['producer_source_cases'][str(v)]
        original[v] = cpu.checked(Path(source['q0_folder']) / 'states.pt', source['q0_files_current_sha256']['states.pt'], torch)
    mapping = {(r['arm'], r['goal'], r['vision_noise_source_seed']): r['case'] for r in c['rows']}
    boundaries, sites, UND, language, packs = {}, [], {}, {}, {}
    actual_prompt_ids = read(producer / 'provenance.json')['actual_prompt_token_ids']
    for row in c['rows']:
        label, goal, v, arm = row['case'], row['goal'], row['vision_noise_source_seed'], row['arm']
        folder, record = producer / label, records[label]
        for name, expected in row['files_sha256'].items():
            assert sha(folder / name) == expected
        audit = cpu.checked(folder / 'noise-audit.pt', row['files_sha256']['noise-audit.pt'], torch)
        packs[label] = audit['model_steps']
        noise_checks(record, audit, original, v, cpu, torch)
        native = records[mapping['native', goal, v]]
        for key in ('prepared_latents_and_masks', 'model_input'):
            assert cpu.exact(record[key], native[key], torch), 'Own goal/V native preparation and first input'
        assert cpu.exact(record['model_input'], audit['initial_model_kwargs'], torch)
        assert cpu.exact(record['action_states'][0], original[195]['action_states'][0], torch)
        assert cpu.exact(record['timesteps'], original[195]['timesteps'], torch)
        assert cpu.exact(record['sigmas'], original[195]['sigmas'], torch)
        info = cpu.checked(folder / 'language-contract.pt', row['files_sha256']['language-contract.pt'], torch)
        contract = info['language']
        assert info['case'] == label and info['goal'] == goal
        assert info['native_reference_case'] == mapping['native', goal, v]
        assert cpu.exact(info['actual_pack'], audit['model_steps'], torch)
        assert cpu.exact(info['first_model_kwargs'], record['model_input'], torch)
        assert cpu.exact(info['original_solver_initial_action_FP32'], record['action_states'][0], torch)
        assert contract['goal'] == goal and contract['prompt'] == c['frozen']['prompts'][goal]
        kwargs0 = record['model_input']
        assert cpu.exact(contract['input_ids'], kwargs0['input_ids'], torch)
        assert cpu.exact(contract['text_indexes'], kwargs0['text_indexes'], torch)
        assert cpu.exact(contract['position_ids'], kwargs0['position_ids'], torch)
        packed, prompt_ids = contract['input_ids'].flatten().tolist(), contract['prompt_token_ids']
        assert prompt_ids == actual_prompt_ids[goal]
        start = probe.unique_span(packed, prompt_ids, 'analysis actual prompt/' + goal)
        target = probe.unique_span(prompt_ids, probe.TARGET_IDS[goal], 'analysis target/' + goal)
        assert contract['prompt_span'] == [start, start + len(prompt_ids)]
        assert contract['target_positions'] == list(range(start + target, start + target + len(probe.TARGET_IDS[goal])))
        assert contract['target_token_ids'] == probe.TARGET_IDS[goal] and len(prompt_ids) == probe.PROMPT_LENGTHS[goal]
        finite(info['actual_clean_current'], (48, 10, 20), torch.bfloat16, torch)
        assert cpu.exact(info['actual_clean_current'], kwargs0['vision_tokens'][0][0, :, 0], torch)
        clean_meta = cpu.tensor_meta(info['actual_clean_current'], torch)
        prepared = record['prepared_latents_and_masks']
        assert len(prepared) == 12
        for i, value in enumerate(prepared):
            source_v = v if i == 0 else 195
            assert cpu.exact(value, original[source_v]['prepared_latents_and_masks'][i], torch)
        assert cpu.exact(prepared[0][0, :, 0].to(torch.bfloat16), info['actual_clean_current'], torch)
        assert torch.count_nonzero(prepared[2][:, 10:]) == 0
        assert cpu.exact(kwargs0['vision_tokens'], original[v]['model_input']['vision_tokens'], torch)
        assert cpu.exact(kwargs0['action_tokens'], original[195]['model_input']['action_tokens'], torch)
        language[label] = contract
        index = cpu.checked(folder / 'indexes-and-masks.pt', row['files_sha256']['indexes-and-masks.pt'], torch)
        u = int(kwargs0['und_len'])
        expected_index = dict(und_len=u, gen_len=266, full_joint_length=u + 266,
            current_rows=torch.arange(50), action_rows=torch.arange(250, 266),
            union_rows=torch.cat((torch.arange(50), torch.arange(250, 266))),
            future_rows=torch.arange(50, 250), future_keys=torch.arange(u + 50, u + 250),
            current_keys=torch.arange(u, u + 50), action_keys=torch.arange(u + 250, u + 266), text_keys=torch.arange(u))
        for key, value in expected_index.items():
            assert cpu.exact(index['indexes'][key], value, torch)
        masks = probe.dynamic_masks(expected_index, torch)
        assert cpu.exact(index['masks'], masks, torch)
        items = row['boundary_files']
        assert len(items) == 30 and len(audit['model_steps']) == 30
        boundaries[label] = items
        by_step = {}
        for t, item in enumerate(items):
            assert item['step'] == t and item['file'] == f'boundaries-t{t:02d}.pt'
            data = cpu.checked(folder / item['file'], item['sha256'], torch)
            assert data['step'] == t and data['case_arm'] == arm
            finite(data['action_hidden'], (37, 16, 4096), torch.bfloat16, torch)
            assert cpu.exact(data['actual_model_action_tokens'], record['action_states'][t, 0].to(torch.bfloat16), torch)
            assert cpu.exact(data['actual_model_action_output'], record['action_velocity'][t], torch)
            assert torch.count_nonzero(data['actual_model_action_tokens'][:, 10:]) == 0
            assert cpu.exact(data['solver_sigma_from_same_frozen_schedule'], record['sigmas'][t], torch)
            assert cpu.exact(data['actual_pack'], audit['model_steps'][t], torch)
            assert data['actual_current_frame_sha256'] == clean_meta['sha256']
            assert all(data['actual_current_frame_metadata'][key] == value for key, value in clean_meta.items())
            metadata = data['actual_pack']['metadata']
            assert cpu.exact(data['actual_pack']['layout'], components._layout(kwargs0), torch)
            for key in components._STRUCTURAL_KEYS:
                if key not in components._TIMESTEP_KEYS:
                    assert cpu.exact(metadata[key], kwargs0.get(key), torch)
            assert cpu.exact(data['actual_model_action_timesteps'], metadata['action_timesteps'], torch)
            if t in CURRENT_STEPS:
                kw = data['actual_model_kwargs']
                assert cpu.exact(data['actual_pack'], dict(layout=components._layout(kw),
                    metadata={key: kw.get(key) for key in components._STRUCTURAL_KEYS}), torch)
                assert cpu.exact(kw['action_tokens'][0], data['actual_model_action_tokens'], torch)
                assert cpu.exact(kw['vision_tokens'][0][0, :, 0], kwargs0['vision_tokens'][0][0, :, 0], torch)
                assert cpu.exact(data['model_output'][2][0], data['actual_model_action_output'], torch)
                assert probe.tree_sha(kw, runtime) == data['actual_model_kwargs_sha256']
                assert probe.tree_sha(data['model_output'], runtime) == data['model_output_sha256']
                assert probe.tree_meta(kw, runtime) == data['actual_model_kwargs_metadata']
                assert probe.tree_meta(data['model_output'], runtime) == data['model_output_metadata']
                finite(data['current_hidden'], (37, 50, 4096), torch.bfloat16, torch)
            else:
                assert data['actual_model_kwargs'] is data['model_output'] is data['current_hidden'] is None
                for key in ('actual_model_kwargs_sha256', 'model_output_sha256'):
                    assert len(data[key]) == 64
            if t == 0:
                assert cpu.exact(data['actual_model_kwargs'], kwargs0, torch)
                finite(data['und_hidden'], (37, u, 4096), torch.bfloat16, torch)
                if goal in UND:
                    assert cpu.exact(data['und_hidden'], UND[goal], torch)
                else:
                    UND[goal] = data['und_hidden'].clone()
                sham_row = next(r for r in c['rows'] if (r['arm'], r['goal'], r['vision_noise_source_seed']) == ('all_allowed', goal, v))
                sham_item = sham_row['boundary_files'][0]
                sham = cpu.checked(producer / sham_row['case'] / sham_item['file'], sham_item['sha256'], torch)
                for key in ('current_hidden', 'action_hidden'):
                    assert cpu.exact(data[key][0], sham[key][0], torch)
            else:
                assert data['und_hidden'] is None
            assert len(data['site_events']) == 36
            assert item['full_kwargs_and_tuple_saved'] is (t in CURRENT_STEPS)
            assert item['current_shape'] == ([37, 50, 4096] if t in CURRENT_STEPS else None)
            assert item['und_shape'] == ([37, u, 4096] if t == 0 else None)
            assert item['action_shape'] == [37, 16, 4096]
            assert item['actual_model_kwargs_sha256'] == data['actual_model_kwargs_sha256']
            assert item['model_output_sha256'] == data['model_output_sha256']
            assert all(data['action_hidden_metadata'][key] == value for key, value in cpu.tensor_meta(data['action_hidden'], torch).items())
            if t in CURRENT_STEPS:
                assert all(data['current_hidden_metadata'][key] == value for key, value in cpu.tensor_meta(data['current_hidden'], torch).items())
            if t == 0:
                assert all(data['und_hidden_metadata'][key] == value for key, value in cpu.tensor_meta(data['und_hidden'], torch).items())
            if goal == 'milk':
                ref = c['frozen']['whole_physical_references'][label]
                name = 'chunk_00/' + item['file']
                old = cpu.checked(Path(ref['folder']) / name, ref['files_current_sha256'][name], torch)
                assert cpu.exact(data['action_hidden'], old['action_hidden'], torch)
                assert cpu.exact(data['actual_pack'], old['actual_pack'], torch)
                assert probe.tree_sha(old['actual_model_kwargs'], runtime) == data['actual_model_kwargs_sha256']
                assert probe.tree_sha(old['model_output'], runtime) == data['model_output_sha256']
                assert all(data['current_hidden_metadata'][key] == value for key, value in cpu.tensor_meta(old['current_hidden'], torch).items())
                if t in CURRENT_STEPS:
                    assert cpu.exact(data['current_hidden'], old['current_hidden'], torch)
                    assert cpu.exact(data['actual_model_kwargs'], old['actual_model_kwargs'], torch)
                    assert cpu.exact(data['model_output'], old['model_output'], torch)
            for layer, event in enumerate(data['site_events']):
                active = arm == 'full_hardmask' and t <= 14 and layer <= 8
                assert event['step'] == t and event['layer_zero_based'] == layer and event['arm'] == arm
                assert event['selected_window_cut_active'] is active
                assert event['extra_all_allowed_calls'] == int(arm != 'native') and event['extra_cut_calls'] == int(active)
                assert event['to_add_out_actual_calls'] == 1 and event['substituted_query_rows'] == (0 if arm == 'native' else 66)
            if t < 15:
                by_step[t] = data
        manifest = read(folder / 'site-captures.json')
        assert manifest['state'] == 'complete' and manifest['sites'] == len(manifest['files']) == 135
        assert [(i['step'], i['layer_zero_based']) for i in manifest['files']] == [(t, l) for t in range(15) for l in range(9)]
        for item in manifest['files']:
            t, layer = item['step'], item['layer_zero_based']
            path = folder / item['file']
            assert path.stat().st_size == item['bytes']
            d = cpu.checked(path, item['sha256'], torch)
            assert (d['step'], d['layer_zero_based'], d['case_arm']) == (t, layer, arm)
            assert cpu.exact(d['current_rows'], expected_index['current_rows'], torch)
            assert cpu.exact(d['future_keys'], expected_index['future_keys'], torch)
            z = d['endpoints']
            assert set(z) == {'Znative', 'Z0', 'Zv', 'Zk', 'Zreal', 'Zclone'}
            executed = {'Znative', 'Zreal'} | ({'Z0'} if arm != 'native' else set()) | ({'Zk'} if arm == 'full_hardmask' else set())
            raw_keys = executed if t == 0 else {'Zreal'}
            assert {key for key, value in z.items() if value is not None} == raw_keys
            assert set(d['executed_endpoints']) == executed
            assert set(d['omitted_executed_endpoint_raw']) == executed - raw_keys
            for key, value in z.items():
                if value is not None:
                    finite(value, (1, 50, 32, 128), torch.bfloat16, torch)
                    raw_meta = cpu.tensor_meta(value, torch)
                    assert all(d['live_endpoint_metadata'][key][k] == val for k, val in raw_meta.items())
            if t == 0:
                finite(d['Qcurrent50'], (1, 50, 32, 128), torch.bfloat16, torch)
                assert all(d['live_Qcurrent50_metadata'][key] == value for key, value in cpu.tensor_meta(d['Qcurrent50'], torch).items())
                for key in ('Kall', 'Vall'):
                    finite(d[key], (1, u + 266, 8, 128), torch.bfloat16, torch)
                    qi = 1 if key == 'Kall' else 2
                    assert all(d['live_QKV_metadata'][qi][k] == value for k, value in cpu.tensor_meta(d[key], torch).items())
                reference = z['Znative'] if arm == 'native' else z['Z0'] if arm == 'all_allowed' else z['Zk']
                assert cpu.exact(z['Zreal'], reference, torch)
            else:
                assert d['Qcurrent50'] is d['Kall'] is d['Vall'] is None
            assert d['event'] == by_step[t]['site_events'][layer]
            for device in ('cpu', 'cuda'):
                assert cpu.exact(d['RNG_before'][device], d['RNG_after'][device], torch)
            p = d['propagation']
            assert set(p) == set(PROP)
            for value in p.values():
                finite(value, (50, 4096), torch.bfloat16, torch)
            assert cpu.exact(p[PROP[0]] + p[PROP[1]], p[PROP[2]], torch)
            assert cpu.exact(p[PROP[2]] + p[PROP[3]], p[PROP[4]], torch)
            if t in CURRENT_STEPS:
                assert cpu.exact(p[PROP[0]], by_step[t]['current_hidden'][layer], torch)
                assert cpu.exact(p[PROP[4]], by_step[t]['current_hidden'][layer + 1], torch)
            sites.append(dict(case=label, step=t, layer_zero_based=layer, file=item['file'], sha256=item['sha256'],
                raw_endpoints={key: None if value is None else cpu.tensor_meta(value, torch) for key, value in z.items()},
                actual_propagation_raw_hashes={key: cpu.tensor_meta(value, torch) for key, value in p.items()}))
        print('[LANGUAGE-ANALYSIS] internal arrays audited ' + label, flush=True)
    checks = token_input_checks(c, records, torch)
    for check in checks:
        for goal in ('milk', 'cheese'):
            contract = language[mapping[check['arm'], goal, check['V']]]
            assert contract['target_positions'] == [check['changed_packed_token_index']]
    primary_contract = read(producer / 'primary-language-contract.json')
    assert primary_contract['state'] == 'exact' and primary_contract['primary_goals'] == ['milk', 'cheese']
    for vi, v in enumerate(VISIONS):
        item = primary_contract['cases'][vi]
        check = next(x for x in checks if x['V'] == v and x['arm'] == 'native')
        assert item['vision_noise_source_seed'] == v
        assert item['changed_actual_input_id_positions'] == [check['changed_packed_token_index']]
        assert item['actual_und_len'] == check['und_len']
        assert item['actual_sequence_length'] == check['und_len'] + 266
        assert item['milk_target_ids'] == [check['milk_token_id']] and item['cheese_target_ids'] == [check['cheese_token_id']]
        # These are observed structural tensors; even late dynamic vision/action
        # samples may differ, their packing and sigma remain paired.
        for arm in ARMS:
            left, right = (mapping[arm, goal, v] for goal in ('milk', 'cheese'))
            for t in range(30):
                lm, rm = packs[left][t], packs[right][t]
                assert cpu.exact(lm['layout'], rm['layout'], torch)
                assert cpu.exact({k: val for k, val in lm['metadata'].items() if k != 'input_ids'},
                                 {k: val for k, val in rm['metadata'].items() if k != 'input_ids'}, torch)
    assert len(sites) == 2430
    return boundaries, sites, dict(main_actual_single_token_checks=checks,
        actual_action_boundaries=540, complete_current_and_kwargs_and_model_outputs=72,
        raw_UND_boundaries=18, actual_BF16_residual_chains=2430,
        six_milk_all30_actual_old_full_arrays_or_live_hash_controls_checked=True,
        new_goal_native_first_observation_not_independent_repeat=True,
        sparse_scope='Action and pack/input-output arrays all30; full current/kwargs/return only0,14,15,29; UND only0. Other full trees have live SHA only, not independently reconstructed.')


def statistics(c, boundary_sources, records, torch, np):
    """Only saved arrays; fixed primary, no layer/time winner search."""
    mapping = {(r['arm'], r['goal'], r['vision_noise_source_seed']): r['case'] for r in c['rows']}
    cpu = c['cpu']

    def boundary(arm, goal, v, t):
        label = mapping[arm, goal, v]
        item = boundary_sources[label][t]
        return cpu.checked(c['producer'] / label / item['file'], item['sha256'], torch)

    def six(v, t, key, auxiliary=False):
        target = 'cream_cheese' if auxiliary else 'cheese'
        return [boundary(arm, goal, v, t)[key].double().numpy()
                for arm in ARMS for goal in ('milk', target)]

    arrays = dict(case_names=np.asarray(c['labels']), V_sources=np.asarray(VISIONS),
        layer_boundaries=np.arange(37), denoising_steps=np.arange(30), current_saved_steps=np.asarray(CURRENT_STEPS),
        t0_action_L36_actual_FP32=np.zeros((18, 16, 4096), dtype=np.float32),
        q0_velocity_XYZ_delta_vs_own_allallowed_RMS=np.zeros((18, 30)),
        q0_final_XYZ_delta_vs_own_allallowed_RMS=np.zeros(18))
    main, auxiliary, head_chain = [], [], []
    for ci, row in enumerate(c['rows']):
        label, arm, goal, v = row['case'], row['arm'], row['goal'], row['vision_noise_source_seed']
        arrays['t0_action_L36_actual_FP32'][ci] = boundary(arm, goal, v, 0)['action_hidden'][36].float().numpy()
        ref = records[mapping['all_allowed', goal, v]]
        delta = records[label]['action_velocity'][:, :, :3].double() - ref['action_velocity'][:, :, :3].double()
        arrays['q0_velocity_XYZ_delta_vs_own_allallowed_RMS'][ci] = delta.square().mean((1, 2)).sqrt().numpy()
        delta = records[label]['actions'][:, :3].double() - ref['actions'][:, :3].double()
        arrays['q0_final_XYZ_delta_vs_own_allallowed_RMS'][ci] = float(delta.square().mean().sqrt())
    for vi, v in enumerate(VISIONS):
        primary_metrics, primary_vectors = None, None
        for scope, key, steps in (('action', 'action_hidden', range(30)),
                                  ('current', 'current_hidden', CURRENT_STEPS),
                                  ('UND', 'und_hidden', (0,))):
            metrics_by_time, tokens_by_time = [], []
            for t in steps:
                values = six(v, t, key)
                metrics, vectors = contrast_stats(*values, np)
                token_metrics, _ = contrast_stats(*(x[..., None, :] for x in values), np)
                metrics_by_time.append(metrics)
                tokens_by_time.append(token_metrics)
                if scope == 'action' and t == 0:
                    primary_metrics = {name: float(value[36]) for name, value in metrics.items()
                        if value.dtype.kind != 'b'}
                    primary_metrics.update(valid=bool(metrics['valid'][36]), thresholds_pass=bool(metrics['thresholds_pass'][36]))
                    primary_vectors = {name: value[36] for name, value in vectors.items()}
                    for name, vector in primary_vectors.items():
                        arrays[f'primary_V{v}_{name}'] = vector
            for name in metrics_by_time[0]:
                arrays[f'{scope}_V{v}_{name}'] = np.stack([x[name] for x in metrics_by_time])
                arrays[f'{scope}_V{v}_per_token_{name}'] = np.stack([x[name] for x in tokens_by_time])
        assert primary_metrics is not None and primary_vectors is not None
        main.append(dict(V=v, position=dict(step=0, field='action_hidden', boundary=36, shape=[16, 4096]),
            **primary_metrics, norm_ratio_delta_m_over_delta_c=primary_metrics['delta_m_norm'] / primary_metrics['delta_c_norm']
            if primary_metrics['delta_c_norm'] > FLOOR else None,
            zero_or_numerical_gate_limit='Invalid when either delta norm <=1e-12, D<=1e-12, or D<=10E.'))
        aux, _ = contrast_stats(*six(v, 0, 'action_hidden', auxiliary=True), np)
        for name, data in aux.items():
            arrays[f'auxiliary_milk_vs_cream_cheese_action_t0_V{v}_{name}'] = data
        auxiliary.append(dict(V=v, status='auxiliary_different_length_not_primary',
            action_L36={name: bool(data[36]) if data.dtype.kind == 'b' else float(data[36]) for name, data in aux.items()}))
        for name, key, section in (('readout_after_actual_norm', 'readout', slice(None)),
                                   ('raw_velocity_all64_includes_padding', 'action_velocity', slice(None)),
                                   ('raw_velocity_actual10', 'action_velocity', slice(0, 10)),
                                   ('raw_velocity_XYZ', 'action_velocity', slice(0, 3)),
                                   ('raw_velocity_rotation6D', 'action_velocity', slice(3, 9)),
                                   ('raw_velocity_gripper', 'action_velocity', slice(9, 10))):
            values = [records[mapping[arm, goal, v]][key][0, :, section].double().numpy()
                      for arm in ARMS for goal in ('milk', 'cheese')]
            metrics, vectors = contrast_stats(*values, np)
            for field, value in metrics.items():
                arrays[f't0_{name}_V{v}_{field}'] = value
            for field, value in vectors.items():
                arrays[f't0_{name}_V{v}_{field}'] = value
            head_chain.append(dict(V=v, stage=name,
                status='descriptive_actual_outputs_not_primary_or_same_input_mechanistic_suppression',
                metrics={field: bool(value) if value.dtype.kind == 'b' else float(value) for field, value in metrics.items()}))
        print('[LANGUAGE-ANALYSIS] statistics V' + str(v), flush=True)
    primary = dict(name=c['probe'].PRIMARY['name'], research_note_alias='H_fixed_bias', prediction=verdict(main), actual_cases=main,
        fixed_position='q0 t0 action boundary36, full16x4096 hidden before final norm',
        thresholds=dict(delta_cosine_min=.95, interaction_ratio_max=.10, D_over_E_strict_min=10),
        zero_norm_floor=FLOOR, rule='Any valid failure: rejected; both valid pass: supported; otherwise insufficient.',
        thresholds_not_independent='r<=.10 implies cosine>=.98 and max/min delta norm<=1.222222 in exact arithmetic.',
        no_population_significance_or_alternative_primary=True)
    return arrays, finite_json(primary), finite_json(auxiliary), finite_json(head_chain)


def plots(out, arrays, np, plt):
    # Each scope and V retains its own panel. All coordinates are true block
    # boundaries; future boundaries were not saved and are not fabricated.
    scopes, keys = ('UND', 'current', 'action'), ('language_D_norm', 'D_cut_norm', 'interaction_norm')
    fig, axes = plt.subplots(6, 3, figsize=(15, 18), constrained_layout=True)
    for si, scope in enumerate(scopes):
        for vi, v in enumerate(VISIONS):
            ri = si * 2 + vi
            data = [arrays[f'{scope}_V{v}_{key}'][0] for key in keys]
            top = max(float(x.max()) for x in data) or 1
            for ki, key in enumerate(keys):
                ax = axes[ri, ki]
                im = ax.imshow(data[ki][None, :], aspect='auto', origin='lower', vmin=0, vmax=top,
                               extent=(-.5, 36.5, -.5, .5))
                ax.set(title=f't0 {scope} V{v}: {key}', xlabel='Actual block boundary (0..36)', yticks=[])
                fig.colorbar(im, ax=ax, label='Absolute hidden L2; no probability interpretation')
    fig.savefig(out / 't0-language-and-interaction.png', dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for vi, v in enumerate(VISIONS):
        for ki, key in enumerate(keys):
            data = arrays[f'action_V{v}_per_token_{key}'][0].T
            ax = axes[vi, ki]
            im = ax.imshow(data, origin='lower', aspect='auto', extent=(-.5, 36.5, -.5, 15.5))
            ax.set(title=f't0 action V{v}: {key}', xlabel='Actual block boundary (0..36)', ylabel='Action token (0..15)')
            fig.colorbar(im, ax=ax, label='Absolute per-token hidden L2')
    fig.savefig(out / 't0-action-token-contrasts.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for vi, v in enumerate(VISIONS):
        for ki, key in enumerate(('language_D_norm', 'interaction_norm')):
            ax = axes[vi, ki]
            data = arrays[f'action_V{v}_{key}']
            im = ax.imshow(data, origin='lower', aspect='auto', extent=(-.5, 36.5, -.5, 29.5))
            ax.set(title=f'action V{v}: {key}; later = dynamic total effect',
                   xlabel='Actual block boundary (0..36)', ylabel='Denoising step (0..29)')
            fig.colorbar(im, ax=ax, label='Absolute hidden L2')
    fig.savefig(out / 'later-action-total-effects.png', dpi=150)
    plt.close(fig)


def linear_head_statistics(weight, bias, arrays, np):
    """Float64 diagnostic of the actual affine head, never a replay gate."""
    assert weight.shape == (4096, 64) and bias.shape == (64,)
    assert np.isfinite(weight).all() and np.isfinite(bias).all()
    U, singular, _ = np.linalg.svd(weight[:, :10], full_matrices=False)
    tolerance = max(4096, 10) * np.finfo(np.float64).eps * float(singular[0])
    basis = U[:, singular > tolerance]
    arrays['actual_head_first10_singular_values_Float64'] = singular
    rows = []
    for v in VISIONS:
        for name in ('D', 'I'):
            vector = arrays[f't0_readout_after_actual_norm_V{v}_{name}']
            projected = (vector @ basis) @ basis.T
            remainder = vector - projected
            total = float(np.sum(vector * vector, dtype=np.float64))
            visible = float(np.sum(projected * projected, dtype=np.float64))
            null = float(np.sum(remainder * remainder, dtype=np.float64))
            predicted = vector @ weight[:, :10]
            actual = arrays[f't0_raw_velocity_actual10_V{v}_{name}']
            error = norm(predicted - actual, np)
            rows.append(dict(V=v, vector=name, actual_head_first10_rank=int(basis.shape[1]),
                readout_total_squared_L2=total, projected_squared_L2=visible, null_squared_L2=null,
                projected_energy_fraction=visible / total if total > FLOOR ** 2 else None,
                null_energy_fraction=null / total if total > FLOOR ** 2 else None,
                decomposition_squared_L2_discrepancy=visible + null - total,
                Float64_linear_prediction_vs_actual_BF16_velocity_difference_L2=error,
                actual_BF16_velocity_vector_L2=norm(actual, np),
                scope='Descriptive SVD of actual parameter map to ten outputs; not semantics or a BF16 kernel equivalence gate.'))
    return finite_json(rows)


def head_artifact(c, arrays, torch, np):
    path = c['producer'] / 'output-head-actual.pt'
    expected = c['pc']['output_head_actual_sha256']
    assert read(c['producer'] / 'provenance.json')['output_head_actual_sha256'] == expected
    head = c['cpu'].checked(path, expected, torch)
    assert head['domain_id'] == 5 and head['source_transformer_sha256'] == c['frozen']['actual_transformer_sha256']
    finite(head['weight'], (4096, 64), torch.bfloat16, torch)
    finite(head['bias'], (64,), torch.bfloat16, torch)
    assert isinstance(head['norm_state'], dict) and head['norm_state']
    for value in head['norm_state'].values():
        assert isinstance(value, torch.Tensor) and value.device.type == 'cpu' and bool(torch.isfinite(value).all())
    arrays['actual_domain5_head_weight_FP32'] = head['weight'].float().numpy()
    arrays['actual_domain5_head_bias_FP32'] = head['bias'].float().numpy()
    statistics = linear_head_statistics(head['weight'].double().numpy(), head['bias'].double().numpy(), arrays, np)
    source = dict(file=path.name, sha256=expected, domain_id=5, actual_input_output_sizes=[4096, 64],
        raw_weight=c['cpu'].tensor_meta(head['weight'], torch), raw_bias=c['cpu'].tensor_meta(head['bias'], torch),
        norm_state={name: c['cpu'].tensor_meta(value, torch) for name, value in head['norm_state'].items()},
        source_transformer_sha256=head['source_transformer_sha256'], capture=head['capture'])
    return statistics, source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    context = seal(args.output.resolve())  # Complete authority before any output or plot.
    out = args.output.resolve() / 'future-language-selectivity-analysis'
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
        records, physical_arrays, physical, physical_counts, runtime, components, scan = physical_audit(context, torch, np)
        boundary_sources, site_sources, input_checks = internal_audit(context, records, runtime, components, scan, torch, np)
        arrays, primary, auxiliary, head_chain = statistics(context, boundary_sources, records, torch, np)
        head_subspace, head_source = head_artifact(context, arrays, torch, np)
        arrays.update(physical_arrays)
        np.savez_compressed(out / 'arrays.npz', **arrays)
        with np.load(out / 'arrays.npz', allow_pickle=False) as reread:
            assert set(reread.files) == set(arrays)
            for key, actual in arrays.items():
                saved = reread[key]
                assert saved.dtype == actual.dtype and saved.shape == actual.shape and saved.tobytes(order='C') == actual.tobytes(order='C')
            for ci, row in enumerate(context['rows']):
                item = boundary_sources[row['case']][0]
                data = context['cpu'].checked(context['producer'] / row['case'] / item['file'], item['sha256'], torch)
                actual = data['action_hidden'][36].float().numpy()
                exported = reread['t0_action_L36_actual_FP32'][ci]
                assert exported.dtype == actual.dtype and exported.shape == actual.shape
                assert exported.tobytes(order='C') == actual.tobytes(order='C'), 'Primary export actual BF16-to-FP32 bytes'
        assert (out / 'arrays.npz').stat().st_size < 100 * 1024 ** 2
        plots(out, arrays, np, plt)
        provenance = dict(script_sha256=sha(Path(__file__)), producer_script_sha256=PROBE_SHA,
            execution_script_sha256=EXECUTION_SHA, CPU_helper_sha256=CPU_HELPER_SHA,
            producer_documents_sha256={name: sha(context['producer'] / (name + '.json')) for name in
                ('complete', 'results', 'protocol', 'provenance', 'sources', 'primary-prediction', 'primary-language-contract')},
            execution_documents_sha256={name: sha(context['execution'] / (name + '.json')) for name in
                ('complete', 'controls', 'comparisons', 'summary', 'provenance')},
            actual_boundary_sources=boundary_sources, actual_site_sources=site_sources,
            actual_output_head_source=head_source,
            torch_version=str(torch.__version__), numpy_version=str(np.__version__),
            measurement_dtype='Float64; raw primary endpoints are lossless BF16-to-FP32 exports', scope=SCOPE)
        write(out / 'provenance.json', provenance)
        write(out / 'analysis.json', finite_json(dict(state='complete', primary=primary,
            canonical_auxiliary=auxiliary, actual_t0_output_head_chain=head_chain,
            actual_linear_head_subspace=head_subspace,
            physical_cases=physical, physical_counts=physical_counts,
            actual_input_and_scope_checks=input_checks, model_calls=0, official_dispatch_calls=0,
            solver_calls=0, physics_calls=0, CUDA_initialized=torch.cuda.is_initialized(), scope=SCOPE)))
        files = ('arrays.npz', 'analysis.json', 'provenance.json', 't0-language-and-interaction.png',
                 't0-action-token-contrasts.png', 'later-action-total-effects.png')
        assert not torch.cuda.is_initialized()
        write(out / 'complete.json', dict(state='complete', script_sha256=sha(Path(__file__)),
            primary_prediction=primary['prediction'], actual_boundary_files=540, actual_site_files=2430,
            physical_counts=physical_counts, files_sha256={name: sha(out / name) for name in files},
            model_calls=0, official_dispatch_calls=0, solver_calls=0, physics_calls=0,
            CUDA_initialized=False, elapsed_s=time.perf_counter() - started, scope=SCOPE))
        print('[LANGUAGE-ANALYSIS] complete ' + str(out), flush=True)
    except Exception as exc:
        write(out / 'failed.json', dict(state='failed', error=str(exc), traceback=traceback.format_exc(),
            script_sha256=sha(Path(__file__)), elapsed_s=time.perf_counter() - started))
        raise


if __name__ == '__main__':
    main()
