"""Verify and plot the 24 completed correction-persistence predictions on CPU.

Use --output BASE. Open actual states/session tensors with map_location=cpu;
never instantiate an inference model. Check the original completion/hash chain,
all nine entire-record controls, actual first-reset PP arrays, native inputs,
and every actual intervention/random delta. Non-action arrays were stored as
raw byte hashes by the probe; verify those sealed equality reports rather than
claiming to reconstruct arrays that were not stored.

Create only an absent correction-persistence/figures directory, containing one
three-seed figure and plot.json. Percent is signed final normalized XYZ output
projection onto each seed's own paired D-minus-R axis. It is not a probability,
milk selection or grasp success. Display a common scale containing 0 and 5%.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path


ROOT = Path('/home/current/work/cosmos3')
SEEDS = (195, 196, 198)
SHOWN = ('native_R', 'P', 'P_single_reset', 'P_persistent_gain050',
         'P_persistent_gain100', 'P_persistent_gain200', 'P_random_normmatched')
CONTROLS = ('native_R', 'P', 'R_persistent_self')
FILES = ('states.pt', 'actualsession.pt', 'actualsession.json', 'metadata.json',
         'normalized_actions.json', 'metrics.json', 'norms.json')
GAP_ABS_FLOOR, GAP_RELATIVE_FLOOR, NORM_TOL = 1e-6, 1e-4, 1e-3
FORMULA = ('sum((A-R)*(D-R))/sum((D-R)^2), on final 16x3 normalized action XYZ; '
           'plotted percent = 100*fraction. Each seed has its own paired R=x15 '
           'and D=x06 axis. 0 is R and 100 is D along that axis. Orthogonal '
           'differences are not represented; negative and >100 values are valid. '
           'Percent is not milk probability, target selection or grasp success.')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def exact(a, b, torch, label):
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        assert isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor), label
        assert a.device.type == b.device.type == 'cpu' and a.dtype == b.dtype and a.shape == b.shape, label
        assert torch.equal(a, b), label
        assert torch.equal(a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8)), label + '/bytes'
    elif isinstance(a, dict) or isinstance(b, dict):
        assert isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys(), label
        for key in a:
            exact(a[key], b[key], torch, f'{label}/{key}')
    elif isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b), label
        for index, (left, right) in enumerate(zip(a, b)):
            exact(left, right, torch, f'{label}/{index}')
    else:
        assert a == b, label


def tensor_meta(value, torch):
    assert value.device.type == 'cpu'
    payload = value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return dict(shape=list(value.shape), dtype=str(value.dtype), sha256=hashlib.sha256(payload).hexdigest())


def jsonified(value, torch):
    if isinstance(value, torch.Tensor):
        return tensor_meta(value, torch)
    if isinstance(value, dict):
        return {str(key): jsonified(item, torch) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonified(item, torch) for item in value]
    return value


def close(actual, expected, label):
    assert type(actual) in (int, float) and type(expected) in (int, float), label
    assert math.isfinite(actual) and math.isfinite(expected), label
    assert math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-10), (label, actual, expected)


def load(path, torch):
    return torch.load(path, map_location='cpu', weights_only=True, mmap=True)


def record_contract(record, reference, torch, label):
    for key in ('pure_noise', 'timesteps', 'sigmas', 'model_input', 'prepared_latents_and_masks'):
        exact(record[key], reference[key], torch, label + '/native_input/' + key)
    assert len(record['pure_noise']) == 2
    assert all(value.dtype == torch.float32 and torch.isfinite(value).all() for value in record['pure_noise'])
    assert record['pure_noise'][0].ndim == 5 and record['pure_noise'][1].shape == (16, 64)
    prepared = record['prepared_latents_and_masks']
    assert len(prepared) == 12 and prepared[0].dtype == prepared[2].dtype == torch.float32
    assert prepared[1] is None and prepared[8].tolist() == [5] and prepared[10] == 10
    assert prepared[2].shape == (16, 64) and torch.count_nonzero(prepared[2][:, 10:]) == 0
    assert torch.count_nonzero(prepared[7]) == 0
    for key, shape, dtype in (('actions', (16, 10), torch.float32),
                             ('readout', (30, 16, 4096), torch.bfloat16),
                             ('action_velocity', (30, 16, 64), torch.bfloat16),
                             ('action_states', (31, 1, 16, 64), torch.float32)):
        value = record[key]
        assert value.device.type == 'cpu' and tuple(value.shape) == shape and value.dtype == dtype, (label, key)
        assert torch.isfinite(value).all(), (label, key)
    assert record['timesteps'].numel() == 30 and record['sigmas'].numel() >= 30
    assert torch.isfinite(record['sigmas']).all()


def final_metric(r, d, a, torch):
    r, d, a = (value[:, :3].double().reshape(-1) for value in (r, d, a))
    gap, delta = d - r, a - r
    gap_rms = float(gap.square().mean().sqrt())
    floor = max(GAP_ABS_FLOOR, GAP_RELATIVE_FLOOR * max(float(r.square().mean().sqrt()), float(d.square().mean().sqrt())))
    assert gap_rms > floor, 'Final paired XYZ gap too small; refuse a forced projection figure'
    fraction = float((delta * gap).sum() / gap.square().sum())
    return dict(signed_projection_fraction=fraction, rmse_to_recipient=float(delta.square().mean().sqrt()),
        rmse_to_donor=float((a - d).square().mean().sqrt()), donor_recipient_gap_rmse=gap_rms,
        gap_floor=floor, gap_meaningful=True)


def check_metric(actual, recorded, label):
    assert recorded['gap_meaningful'] is True, label
    for key, value in actual.items():
        if type(value) is bool:
            assert recorded[key] is value, label + '/' + key
        else:
            close(value, recorded[key], label + '/' + key)


def source_chain(baseline):
    root = baseline / 'correction-persistence'
    assert not (root / 'failed.json').exists()
    complete, results, protocol, provenance = (read_json(root / (name + '.json'))
                                             for name in ('complete', 'results', 'protocol', 'provenance'))
    assert complete['state'] == results['state'] == 'complete'
    assert complete['q0_predictions'] == results['predictions'] == protocol['predictions'] == 24
    assert complete['exact_full_record_controls'] == 9
    for key in ('all_two_draw_noise_schedule_prepared_contracts_exact', 'all_site_calls_30_and_nonaction_rows_exact',
                'first_reset_actual_input_and_PP_increment_exact', 'actual_random_norms_matched_per_step'):
        assert complete[key] is True, key
    for name in ('results', 'protocol', 'provenance'):
        assert complete[name + '_sha256'] == sha(root / (name + '.json')), name + '/original completion SHA'
    assert results['protocol_sha256'] == provenance['protocol_sha256'] == complete['protocol_sha256']
    probe = Path(__file__).with_name('probe_cosmos_milk_correction_persistence.py')
    assert complete['script_sha256'] == provenance['script_sha256'] == sha(probe)
    assert provenance['library_edits'] is provenance['checkpoint_edits'] is provenance['closed_loop_execution'] is False
    assert protocol['seeds'] == list(SEEDS) and protocol['actual_bf16_norm_relative_tolerance'] == NORM_TOL
    assert results['physical_selection_predeclared'] is protocol['physical_candidates_selected_before_scores'] is True
    assert results['future_physical_candidates'] == protocol['future_physical_candidates'] == [
        dict(seed=198, arm=name) for name in ('native_R', 'P', 'P_persistent_gain100', 'P_random_normmatched')]
    definitions = {arm['name']: arm for arm in protocol['arms']}
    assert len(definitions) == 8 and set(definitions) == set(SHOWN) | {'R_persistent_self'}
    assert protocol['execution_order'] == list(definitions)
    assert protocol['pulse'] == dict(layer_zero_based=17, component='attention', step=20, gain=1., donor='paired x06')
    assert protocol['candidate']['layer_zero_based'] == 3 and protocol['candidate']['component'] == 'mlp'
    source_paths = {
        'normal_runtime': ROOT / 'work/serve_cosmos_goal_pair.py',
        'factory': ROOT / 'work/probe_cosmos_language_routes.py',
        'components': ROOT / 'work/cosmos_component_interventions.py',
        'pulse_metric_source': ROOT / 'work/probe_cosmos_milk_component_pulses.py',
        'feedback_script': ROOT / 'work/probe_cosmos_milk_feedback_inputs.py',
        'scheduler': ROOT / 'cosmos-framework/cosmos_framework/model/generator/diffusion/samplers/fm_solvers_unipc.py',
        'actual_transformer': Path(provenance['actual_transformer_source']),
    }
    for key in ('normal_runtime', 'factory', 'components'):
        assert sha(source_paths[key]) == provenance['source_sha256'][key]
    for key in ('pulse_metric_source', 'feedback_script', 'scheduler', 'actual_transformer'):
        assert sha(source_paths[key]) == provenance[key + '_sha256']
    transformer = source_paths['actual_transformer']
    assert transformer.parts[-4:] == ('diffusers', 'models', 'transformers', 'transformer_cosmos3.py'), 'Unverified package source layout'
    pipeline = transformer.parents[2] / 'pipelines/cosmos/pipeline_cosmos3_omni.py'
    assert sha(pipeline) == provenance['actual_pipeline_source_sha256']
    source_paths['actual_pipeline'] = pipeline
    feedback = baseline / 'feedback-inputs'
    assert not (feedback / 'failed.json').exists()
    for name in ('complete', 'provenance', 'summary'):
        assert sha(feedback / (name + '.json')) == provenance['feedback_' + name + '_sha256']
    f_complete, f_provenance, f_summary = (read_json(feedback / (name + '.json')) for name in ('complete', 'provenance', 'summary'))
    assert f_complete['state'] == 'complete' and f_complete['seeds'] == list(SEEDS)
    assert f_complete['q0_predictions'] == f_complete['native_replays_exact'] == 6
    assert f_complete['direct_model_forwards'] == 12 and f_complete['components_per_forward'] == 72
    assert f_complete['script_sha256'] == f_provenance['script_sha256'] == provenance['feedback_script_sha256']
    assert f_complete['summary_sha256'] == provenance['feedback_summary_sha256']
    assert f_summary['all_native_R_and_P_t21_replays_bit_exact'] is f_summary['entire_t21_model_output_including_vision_exact'] is True
    assert f_summary['current_condition_all_seeds_exact'] is True
    assert f_provenance['baseline_summary_sha256'] == sha(baseline / 'summary.json')
    assert f_provenance['pulse_contract_sha256'] == provenance['pulse_metric_source_sha256']
    for key in ('normal_runtime', 'factory', 'components'):
        assert f_provenance['source_sha256'][key] == provenance['source_sha256'][key]
    assert f_provenance['scheduler_sha256'] == provenance['scheduler_sha256']
    assert f_provenance['actual_transformer_sha256'] == provenance['actual_transformer_sha256']
    assert f_provenance['actual_pipeline_source_sha256'] == provenance['actual_pipeline_source_sha256']
    cases = results['cases']
    expected = {f'seed{seed}_{arm}' for seed in SEEDS for arm in definitions}
    assert len(cases) == 24 and {row['run'] for row in cases} == set(results['completed']) == expected
    rows = {row['run']: row for row in cases}
    controls = results['controls']
    assert len(controls) == 9 and {row['run'] for row in controls} == {f'seed{seed}_{arm}' for seed in SEEDS for arm in CONTROLS}
    for control in controls:
        assert control['all_saved_fields_bit_exact'] is True
        row = rows[control['run']]
        assert control['states_sha256'] == row['files_sha256']['states.pt']
        assert control['actualsession_sha256'] == row['files_sha256']['actualsession.pt']
    return root, results, protocol, provenance, definitions, rows, dict(
        complete_sha256=sha(root / 'complete.json'), original_results_sha256=complete['results_sha256'],
        original_protocol_sha256=complete['protocol_sha256'], original_provenance_sha256=complete['provenance_sha256'],
        frozen_probe_sha256=complete['script_sha256'], source_files={key: dict(path=str(path), sha256=sha(path)) for key, path in source_paths.items()}, runs=[])


def seed_sources(baseline, root, seed, rows, torch):
    folder = root / f'seed{seed}'
    source_path = folder / 'sources.json'
    for row in rows.values():
        if row['seed'] == seed:
            assert row['sources_sha256'] == sha(source_path)
    sources = read_json(source_path)
    contract = read_json(Path(sources['feedback']['feedback_input_contract']['path']))
    assert contract['seed'] == seed and contract['both_pure_noise_draws_exact'] is True
    assert contract['complete_schedule_and_30step_metadata_exact'] is True
    records, caches = {}, {}
    for role, scene in (('R', 'x15'), ('D', 'x06')):
        trial = baseline / 'closed-loop' / f'{scene}_seed{seed}'
        assert sources[role]['trial'] == trial.name
        assert Path(sources[role]['image']).resolve() == (trial / 'input_00.png').resolve()
        assert sources[role]['image_sha256'] == sha(trial / 'input_00.png')
        for name in ('states', 'components'):
            assert sources[role][name + '_sha256'] == sha(trial / 'chunk_00' / (name + '.pt'))
            assert contract['sources'][role][name + '_sha256'] == sources[role][name + '_sha256']
        records[role] = load(trial / 'chunk_00/states.pt', torch)
        caches[role] = load(trial / 'chunk_00/components.pt', torch)
        record_contract(records[role], records[role], torch, f'seed{seed}/source{role}')
        assert caches[role]['complete'] is True and caches[role]['n_steps'] == len(caches[role]['steps']) == 30
        assert caches[role]['n_layers'] == 36 and caches[role]['component_call_count'] == caches[role]['captured_component_count'] == 2160
        assert caches[role]['processor_recompute_calls'] == 0
        assert set(caches[role]['frames']) == {(step, layer, component) for step in range(30)
            for layer in range(36) for component in ('attention', 'mlp')}
        exact(caches[role]['first_model_kwargs'], records[role]['model_input'], torch, role + '/source first kwargs')
    r, d = records['R'], records['D']
    for key in ('pure_noise', 'timesteps', 'sigmas'):
        exact(r[key], d[key], torch, 'paired/' + key)
    exact(caches['R']['steps'], caches['D']['steps'], torch, 'paired/all30 pack metadata')
    exact(r['model_input']['action_tokens'], d['model_input']['action_tokens'], torch, 'paired initial action noise')
    for entry in sources['feedback'].values():
        assert entry['sha256'] == sha(Path(entry['path']))
    feedback = sources['feedback']
    native_r = load(Path(feedback['feedback_native_R_states']['path']), torch)
    p = load(Path(feedback['feedback_P_states']['path']), torch)
    exact(native_r, r, torch, f'seed{seed}/feedback_native_R full record')
    record_contract(p, r, torch, f'seed{seed}/sourceP')
    r_inputs = load(Path(feedback['feedback_R_inputs']['path']), torch)
    p_inputs = load(Path(feedback['feedback_P_inputs']['path']), torch)
    assert set(r_inputs) == set(p_inputs) == {20, 21}
    exact(r_inputs[20], p_inputs[20], torch, 'saved original pulse input20')
    pp_path = Path(feedback['feedback_PP_components']['path'])
    pp_meta = read_json(Path(feedback['feedback_PP_metadata']['path']))
    assert pp_meta['components_sha256'] == sha(pp_path) and pp_meta['original_q0_entire_model_output_exact'] is True
    assert pp_meta['arrays_sha256'] == sha(pp_path.with_name('arrays.npz'))
    assert pp_meta['model_output_sha256'] == sha(pp_path.with_name('model_output.pt'))
    pp = load(pp_path, torch)
    assert pp['complete'] is True and pp['n_steps'] == len(pp['steps']) == pp['expected_steps'] == 1
    assert pp['n_layers'] == 36 and pp['captured_component_count'] == pp['component_call_count'] == 72
    exact(pp['first_model_kwargs'], p_inputs[21], torch, 'actual PP first model kwargs')
    exact(pp['steps'][0], caches['R']['steps'][21], torch, 'actual PP metadata at21')
    pair_path = folder / 'pair-contract.pt'
    pair = load(pair_path, torch)
    for key, reference in (('recipient_pure_noise', r['pure_noise']), ('donor_pure_noise', d['pure_noise']),
                           ('timesteps', r['timesteps']), ('sigmas', r['sigmas']),
                           ('recipient_steps', caches['R']['steps']), ('donor_steps', caches['D']['steps']),
                           ('R_actual_input21', r_inputs[21]), ('P_actual_input21', p_inputs[21]),
                           ('PP_actual_L3_MLP_increment', pp['frames'][(0, 3, 'mlp')])):
        exact(pair[key], reference, torch, f'seed{seed}/pair_contract/' + key)
    return r, d, p, caches, r_inputs[21], p_inputs[21], pp['frames'][(0, 3, 'mlp')], dict(
        sources_sha256=sha(source_path), pair_contract_current_sha256=sha(pair_path),
        pair_contract_hash_note='Not historically sealed in results; all stored values checked against sealed source records and source inputs.',
        frozen_sources=sources)


def checked_run(root, seed, arm, row, definition, r, d, p, caches, r21, p21, pp_frame, protocol, torch):
    directory = root / f'seed{seed}/runs' / arm
    assert row['run'] == f'seed{seed}_{arm}' and row['seed'] == seed and row['arm'] == arm
    assert Path(row['artifacts']).resolve() == directory.resolve()
    assert set(row['files_sha256']) == set(FILES)
    for name in FILES:
        assert row['files_sha256'][name] == sha(directory / name), row['run'] + '/' + name
    assert row['all_native_contract_fields_exact'] is True and row['normalized_action_shape'] == [16, 10]
    assert row['pulse'] == definition['pulse'] and row['reset_steps'] == definition['reset_steps'] and row['gain'] == definition['gain']
    record, report = load(directory / 'states.pt', torch), load(directory / 'actualsession.pt', torch)
    record_contract(record, r, torch, row['run'])
    metadata = read_json(directory / 'metadata.json')
    assert metadata['task'] == 'milk' and metadata['seed'] == seed
    assert metadata['prompt'] == 'pick up the milk and place it in the basket'
    assert metadata['model_calls'] == 30 and metadata['prepare_calls'] == 1
    assert metadata['input_png_sha256'] == sha(Path(metadata['image_path']))
    assert metadata['timesteps'] == record['timesteps'].tolist()
    assert record['actions'].tolist() == read_json(directory / 'normalized_actions.json')
    assert row['final_normalized_xyz'] == record['actions'][:, :3].tolist()
    assert jsonified(report, torch) == read_json(directory / 'actualsession.json'), 'JSON must describe the actual saved session tensors'
    actual_definition = dict(definition, reset_steps=tuple(definition['reset_steps']))
    exact(report['arm'], actual_definition, torch, row['run'] + '/actual arm')
    assert report['complete'] is True and report['seed'] == seed
    assert report['n_model_forwards'] == report['finished_forwards'] == report['attention_site_calls'] == report['mlp_site_calls'] == 30
    assert report['explicit_processor_recomputations'] == 0
    assert report['step21_actual_kwargs_exact'] is report['step21_actual_MLP_before_exact'] is report['nonaction_rows_bit_exact'] is True
    exact(report['first_model_kwargs'], r['model_input'], torch, row['run'] + '/first entire model kwargs')
    exact(report['steps'], caches['R']['steps'], torch, row['run'] + '/all actual pack/sigma metadata')
    exact(report['actual_step21_kwargs'], p21 if definition['pulse'] else r21, torch, row['run'] + '/actual first reset21 input')
    expected_events = {(step, 3, 'mlp') for step in definition['reset_steps']}
    if definition['pulse']:
        expected_events.add((20, 17, 'attention'))
    assert set(report['events']) == expected_events
    assert report['intervention_count'] == report['expected_intervention_count'] == row['intervention_count'] == len(expected_events)
    actual_norms, norm_rows = {}, []
    for key, event in sorted(report['events'].items()):
        step, layer, component = key
        assert (event['step'], event['layer'], event['component']) == key
        rows = caches['R']['steps'][step]['layout']['action_rows']
        exact(event['action_rows'], rows, torch, row['run'] + '/event action rows')
        before, after, delta = event['before'], event['after'], event['actual_delta']
        assert before.dtype == after.dtype == torch.bfloat16 and before.shape == after.shape == (16, 4096)
        assert delta.dtype == torch.float64 and delta.shape == (16, 4096)
        assert all(torch.isfinite(value).all() for value in (before, after, delta))
        exact(after.double() - before.double(), delta, torch, row['run'] + '/actual event delta')
        actual = float(delta.norm())
        close(actual, event['actual_delta_l2'], row['run'] + '/actual delta norm')
        assert tensor_meta(before, torch) == event['before_meta'] and tensor_meta(after, torch) == event['after_meta']
        assert event['exact_no_change'] is (event['before_meta'] == event['after_meta'])
        gen_len = caches['R']['steps'][step]['layout']['gen_len']
        assert event['nonaction_rows_count'] == gen_len - 16 and event['nonaction_rows_bit_exact'] is True
        assert event['nonaction_before'] == event['nonaction_after']
        assert event['nonaction_before']['shape'] == [gen_len - 16, 4096]
        assert event['nonaction_before']['dtype'] == 'torch.bfloat16' and len(event['nonaction_before']['sha256']) == 64
        assert event['full_GEN_before']['shape'] == event['full_GEN_after']['shape'] == [gen_len, 4096]
        if event['exact_no_change']:
            assert event['full_GEN_before'] == event['full_GEN_after']
        if component == 'attention':
            assert event['mode'] == 'natural_D_pulse' and event['gain'] == 1. and event['random_calibration'] is None
            exact(before, caches['R']['frames'][key], torch, row['run'] + '/actual pulse before R')
            exact(after, caches['D']['frames'][key], torch, row['run'] + '/actual pulse after D')
            assert event['source_cache_meta'] == tensor_meta(caches['D']['frames'][key], torch)
        else:
            if step == 21:
                exact(before, pp_frame if definition['pulse'] else caches['R']['frames'][key], torch, row['run'] + '/actual first MLP before PP/R')
            if definition['mode'] == 'blend_R':
                assert event['mode'] == 'current_plus_gain_times_R_minus_current_in_native_bf16'
                assert event['gain'] == definition['gain'] and event['random_calibration'] is None
                source = caches['R']['frames'][key]
                assert event['source_cache_meta'] == tensor_meta(source, torch)
                if definition['gain'] == 1:
                    exact(after, source, torch, row['run'] + '/gain1 actual source cache exact')
                else:
                    exact(after, before + definition['gain'] * (source - before), torch, row['run'] + '/actual native BF16 blend')
                if arm == 'R_persistent_self':
                    exact(before, source, torch, row['run'] + '/actual self before R')
                    assert actual == 0 and event['exact_no_change'] is True
            elif definition['mode'] == 'random':
                assert event['mode'] == 'actual_norm_matched_random' and event['gain'] is None and event['source_cache_meta'] is None
                calibration = event['random_calibration']
                assert calibration['cpu_generator_seed'] == protocol['random_cpu_seeds'][str(seed)][str(step)]
                exact(delta, calibration['actual_quantized_delta'], torch, row['run'] + '/random actual quantized delta')
                close(actual, calibration['actual_quantized_delta_l2'], row['run'] + '/random calibration norm')
                close(actual, calibration['actual_hook_delta_l2'], row['run'] + '/random hook norm')
                target = calibration['target_actual_delta_l2']
                assert math.isfinite(target) and target >= 0
                error = 0. if target == 0 else abs(actual - target) / target
                close(error, calibration['relative_norm_error'], row['run'] + '/random calibration error')
                close(error, calibration['actual_hook_relative_error'], row['run'] + '/random actual hook error')
                assert calibration['relative_norm_tolerance'] == NORM_TOL and error <= NORM_TOL
                if target == 0:
                    exact(before, after, torch, row['run'] + '/zero target strict noop')
                    assert actual == 0 and calibration['zero_target_noop'] is True
                else:
                    assert calibration['zero_target_noop'] is False
            actual_norms[step] = actual
        calibration = event['random_calibration']
        norm_rows.append(dict(step=step, layer=layer, component=component, mode=event['mode'], gain=event['gain'],
            actual_delta_l2=event['actual_delta_l2'], exact_no_change=event['exact_no_change'], nonaction_rows_bit_exact=True,
            random_target_actual_delta_l2=None if calibration is None else calibration['target_actual_delta_l2'],
            random_actual_delta_l2=None if calibration is None else calibration['actual_hook_delta_l2'],
            random_relative_error=None if calibration is None else calibration['actual_hook_relative_error']))
    norms = read_json(directory / 'norms.json')
    assert norms['norm_relative_tolerance'] == NORM_TOL and norms['events'] == norm_rows
    for key in ('readout', 'action_velocity'):
        exact(record[key][:20], r[key][:20], torch, row['run'] + '/native before attn20/' + key)
    exact(record['action_states'][:21], r['action_states'][:21], torch, row['run'] + '/native solver before attn20')
    if definition['pulse']:
        for key in ('readout', 'action_velocity'):
            exact(record[key][:21], p[key][:21], torch, row['run'] + '/P before first MLP21/' + key)
        exact(record['action_states'][:22], p['action_states'][:22], torch, row['run'] + '/P solver before first MLP21')
    if arm in CONTROLS:
        exact(record, p if arm == 'P' else r, torch, row['run'] + '/ENTIRE control record including future latents')
    metric = final_metric(r['actions'], d['actions'], record['actions'], torch)
    check_metric(metric, row['final_normalized_action_xyz'], row['run'] + '/results actual final XYZ')
    metrics = read_json(directory / 'metrics.json')
    check_metric(metric, metrics['final_normalized_action_xyz'], row['run'] + '/metrics actual final XYZ')
    assert len(metrics['per_denoising_step']) == 30
    for step, saved in enumerate(metrics['per_denoising_step']):
        assert saved['step'] == step and saved['timestep'] == int(record['timesteps'][step])
        close(float(record['sigmas'][step]), saved['sigma'], row['run'] + '/actual sigma/' + str(step))
    if arm in ('native_R', 'R_persistent_self'):
        assert metric['signed_projection_fraction'] == metric['rmse_to_recipient'] == 0
    return record, report, metric, actual_norms, dict(run=row['run'], files_sha256=row['files_sha256'],
        entire_control_record_bit_exact=arm in CONTROLS, actual_first_input_and_PP_gate_checked=True,
        actual_before_after_delta_arrays_checked=True, sealed_nonaction_raw_hash_equalities_checked=True,
        final_xyz_recomputed=metric)


def checked_data(baseline, torch):
    root, results, protocol, provenance, definitions, rows, sources = source_chain(baseline)
    values, raw, norm_table = {}, {}, []
    for seed in SEEDS:
        r, d, p, caches, r21, p21, pp_frame, source = seed_sources(baseline, root, seed, rows, torch)
        seed_reports, seed_norms, seed_values = {}, {}, {}
        source['sigma_values'] = r['sigmas'].tolist()
        source['timesteps'] = r['timesteps'].tolist()
        raw[str(seed)] = dict(R_normalized_xyz=r['actions'][:, :3].tolist(), D_normalized_xyz=d['actions'][:, :3].tolist(), arms={})
        for arm in protocol['execution_order']:
            row = rows[f'seed{seed}_{arm}']
            record, report, metric, norms, evidence = checked_run(root, seed, arm, row, definitions[arm],
                r, d, p, caches, r21, p21, pp_frame, protocol, torch)
            seed_reports[arm], seed_norms[arm], seed_values[arm] = report, norms, metric
            raw[str(seed)]['arms'][arm] = dict(final_normalized_xyz=record['actions'][:, :3].tolist(), recomputed_metric=metric)
            sources['runs'].append(evidence)
        targets_path = root / f'seed{seed}/random-target-norms.json'
        targets = read_json(targets_path)
        assert targets['reference_arm'] == 'P_persistent_gain100'
        assert targets['reference_actualsession_sha256'] == rows[f'seed{seed}_P_persistent_gain100']['files_sha256']['actualsession.pt']
        assert set(targets['actual_bf16_delta_l2_by_step']) == {str(step) for step in range(21, 30)}
        norm_checks = []
        for step in range(21, 30):
            natural, actual = seed_norms['P_persistent_gain100'][step], seed_norms['P_random_normmatched'][step]
            close(natural, targets['actual_bf16_delta_l2_by_step'][str(step)], f'seed{seed}/actual natural norm{step}')
            event = seed_reports['P_random_normmatched']['events'][(step, 3, 'mlp')]
            close(natural, event['random_calibration']['target_actual_delta_l2'], f'seed{seed}/actual random target{step}')
            error = 0. if natural == 0 else abs(actual - natural) / natural
            assert (actual == 0 if natural == 0 else error <= NORM_TOL)
            norm_checks.append(dict(step=step, actual_gain1_delta_l2=natural, actual_random_delta_l2=actual, relative_norm_error=error))
        norm_table.append(dict(seed=seed, exact_whole_record_controls=3, total_whole_record_controls=3,
            max_relative_norm_error=max(row['relative_norm_error'] for row in norm_checks), actual_step_norms=norm_checks))
        source['random_targets_current_sha256'] = sha(targets_path)
        source['random_targets_hash_note'] = 'Not historically sealed as a standalone file; all target values checked against historically sealed actual gain1/random session tensors.'
        sources[str(seed)] = source
        values[seed] = seed_values
    assert len(sources['runs']) == 24
    return root, values, raw, norm_table, sources, protocol


def render(path, values, norm_table, np):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    percentages = np.asarray([[100 * values[seed][arm]['signed_projection_fraction'] for arm in SHOWN] for seed in SEEDS])
    assert np.isfinite(percentages).all()
    lower, upper = min(0., float(percentages.min())), max(5., float(percentages.max()))
    span = upper - lower
    limits = (lower - .10 * span if lower < 0 else 0., upper + .13 * span)
    figure, axes = plt.subplots(1, 3, figsize=(14.2, 6.4), sharey=True)
    labels = ('Native\nR', 'Pulse\nP', 'Once\n1×', 'Persist\n0.5×', 'Persist\n1×', 'Persist\n2×', 'Random')
    colors = ('#888888', '#D55E00', '#56B4E9', '#009E73', '#0072B2', '#332288', '#bbbbbb')
    try:
        for axis, seed, percent in zip(axes, SEEDS, percentages):
            bars = axis.bar(np.arange(len(SHOWN)), percent, color=colors, width=.76)
            axis.set_title(f'Paired seed {seed}', fontsize=12)
            axis.set_xticks(np.arange(len(SHOWN)), labels, fontsize=8)
            axis.set_ylim(*limits)
            axis.axhline(0, color='#333333', linewidth=.9)
            axis.axhline(5, color='#777777', linestyle=':', linewidth=.8)
            axis.grid(axis='y', alpha=.15)
            axis.set_axisbelow(True)
            axis.spines[['top', 'right']].set_visible(False)
            for bar, value in zip(bars, percent):
                axis.annotate(f'{value:+.3f}%', (bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 4 if value >= 0 else -4), textcoords='offset points',
                    ha='center', va='bottom' if value >= 0 else 'top', fontsize=8)
        axes[0].set_ylabel('Final normalized XYZ projection on paired D − R axis (%)')
        figure.suptitle('Short versus sustained L3 MLP replacement after one L17 attention pulse', fontsize=14, y=.97)
        figure.subplots_adjust(left=.065, right=.99, top=.86, bottom=.39, wspace=.12)
        table_axis = figure.add_axes([.23, .19, .54, .12])
        table_axis.axis('off')
        table = table_axis.table(cellText=[[str(row['seed']), f'{100 * row["max_relative_norm_error"]:.5f}% ≤ 0.1%', '3 / 3 exact'] for row in norm_table],
            colLabels=['Seed', 'Max actual random norm error over 9 steps', 'Entire-record controls'],
            cellLoc='center', loc='center', colWidths=[.14, .59, .27])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.35)
        figure.text(.065, .12, 'Each seed uses its own paired x15 R / x06 D final XYZ axis. Shared display scale includes 0 and 5%; 100% is D along that axis.', fontsize=9)
        figure.text(.065, .085, 'Pulse: L17 attention at step20 once. Reset: L3 MLP at step21 once, or steps21–29. Random matches gain1 actual BF16 norm at each step.', fontsize=9)
        figure.text(.065, .05, 'All 9 entire-record controls are exact, including future latents; native R and no-pulse persistent self have exactly zero change.', fontsize=9)
        figure.text(.065, .015, 'Fixed-input q0 outputs only. Percent is not milk probability or grasp success; continued changes may be inherited by the multistep solver.', fontsize=9)
        figure.savefig(path, dpi=200, facecolor='white')
        return dict(arms=list(SHOWN), percentages_by_seed={str(seed): percentages[index].tolist() for index, seed in enumerate(SEEDS)},
                    common_y_limits_percent=list(limits), scale_anchors_percent=[0., 5.])
    finally:
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Completed original baseline root')
    baseline = parser.parse_args().output.resolve()
    import torch
    import numpy as np
    assert not torch.cuda.is_initialized(), 'CPU verification must not initialize CUDA'
    root, values, raw, norm_table, sources, protocol = checked_data(baseline, torch)
    assert not torch.cuda.is_initialized()
    out = root / 'figures'
    out.mkdir(exist_ok=False)
    image = out / 'correction-persistence.png'
    plotted = render(image, values, norm_table, np)
    payload = dict(state='complete', figure_count=1, seeds=list(SEEDS), predictions_verified=24,
        exact_entire_record_controls=9, formula=FORMULA, plotted_values=plotted, actual_raw_values=raw,
        actual_random_norm_checks=norm_table, source_hashes=sources, protocol_physical_candidates=protocol['future_physical_candidates'],
        verification=dict(all24_saved_final_action_arrays_recomputed=True, all24_final_projection_and_RMSE_JSON_checked=True,
            all24_actual_session_before_after_delta_norms_recomputed=True, all_first_reset_actual_inputs_and_PP_arrays_exact=True,
            all9_entire_records_controls_exact=True, all_actual_sigmas_and_noise_prepared_contracts_exact=True,
            sealed_nonaction_raw_hash_equality_reports_checked=True, no_gpu_or_model_instantiation=True),
        nonaction_evidence_limit='The original probe stored complete non-action row byte hashes and equality flags, not their arrays. This plotter checks those sealed reports; it does not claim independent non-action array reconstruction.',
        limitations=['Each seed has its own D-minus-R direction; percentages share units, not one common geometric axis.',
                     'Final XYZ projection omits orthogonal differences and does not identify an object or grasp outcome.',
                     'The 4 physical candidates remain the preregistered seed198 native_R/P/gain1/random arms.'],
        plot_script_sha256=sha(Path(__file__)), image_sha256=sha(image), scope='Completed fixed-input q0 outputs only; no new inference or simulator execution.')
    (out / 'plot.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(state='complete', image=str(image), plot_json=str(out / 'plot.json'),
                         image_sha256=payload['image_sha256'], predictions_verified=24, exact_controls=9)), flush=True)


if __name__ == '__main__':
    main()
