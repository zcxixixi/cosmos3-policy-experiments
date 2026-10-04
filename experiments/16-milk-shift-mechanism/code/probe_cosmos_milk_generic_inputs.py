"""Continue only six generic forwards after the preserved 18-forward failure.

The original units probe and its failed.json remain untouched. Recheck the
surviving actual arrays, existing SHA relationships and entire native/self/
whole/joint controls. Freeze the same three CPU input draws after actual
rounding norm/angle feedback, without model-output selection. Keep the old
S128 and t21/L3 site. New before/after parameter hashes cover this process;
there is no measured post-18 parameter hash from the failed process.
"""

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import time
import traceback


ROOT = Path('/home/current/work/cosmos3')
OLD_SHA = '6818a6a3c3ae2f665040f2196c00055fc9487b7775642601e7ca7e9e69c95a6a'
BUILDER_SHA = '15cd6e671014c69dc1db4b6a06b6b86a9c1ba615f2a96c08918a8e8f3690b842'
SEED, STEP, LAYER, WIDTH = 198, 21, 3, 12288
INPUT_SEEDS = (5219801, 5219802, 5219803)
SUPPORT_SEEDS = (3219801, 3219802, 3219803)
COEFFICIENT_SEEDS = (4219801, 4219802, 4219803)
SITE_KEYS = {'mlp_input', 'gate_pre', 'g', 'u', 'down_input', 'down_output', 'mlp_output'}
SCOPE = ('Six continuation model forwards, fixed t21/L3 and previously frozen S128. '
         'Generic controls use their own native-to-RR raw Flow XYZ velocity denominator; '
         'different denominators cannot establish semantic success. Original phase1 remains '
         'failed with eighteen surviving forwards. New parameter hashes do not measure its missing post-18 state.')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value, exclusive=False):
    with path.open('x' if exclusive else 'w') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expected_phase1():
    return ['native_RR', 'native_PR', 'self_all12288', 'swap_all12288',
            'natural032', 'natural128', 'natural512', 'reverse128',
            'factor128_PP', 'factor128_RP', 'factor128_PR', 'factor128_RR',
            *[f'random_support128_seed{seed}' for seed in SUPPORT_SEEDS],
            *[f'random_coefficients128_seed{seed}' for seed in COEFFICIENT_SEEDS]]


def read_actual(folder, runtime, correction, scan, old):
    """Check existing producer SHA fields; seal previously unanchored files now."""
    import numpy as np
    torch = runtime.torch
    metadata = read(folder / 'metadata.json')
    assert metadata['label'] == folder.name and metadata['actual_model_calls'] == 1
    assert metadata['seed'] == SEED and metadata['original_denoising_forward'] == STEP
    assert metadata['actual_boundaries'] == 37 and metadata['component_calls'] == 72
    assert metadata['component_cache_complete'] is metadata['all_owned_hooks_removed'] is True
    assert metadata['legacy_whole_output_patch_count'] == 0
    for name, expected in metadata['files_sha256'].items():
        assert sha(folder / name) == expected, str(folder / name)
    assert sha(folder / 'unit_site.pt') == metadata['unit_site_sha256']
    cache = torch.load(folder / 'components.pt', map_location='cpu', weights_only=True, mmap=True)
    tensors = torch.load(folder / 'actual_head_boundaries.pt', map_location='cpu', weights_only=True, mmap=True)
    actual = dict(cache=cache, hidden=tensors['action_boundary_hidden'], readout=tensors['readout'],
        velocity=tensors['action_velocity'], model_output=torch.load(folder / 'model_output.pt', map_location='cpu', weights_only=True, mmap=True),
        unit=torch.load(folder / 'unit_site.pt', map_location='cpu', weights_only=True, mmap=True), folder=folder)
    correction.require_finite(actual, torch)
    for key, shape, meta_key in (('hidden', (37, 16, 4096), 'boundary'), ('readout', (16, 4096), 'readout'), ('velocity', (16, 64), 'velocity')):
        assert actual[key].shape == shape and actual[key].dtype == torch.bfloat16
        runtime.require_exact(scan.tensor_meta(actual[key], torch), metadata[meta_key], 'surviving_head_and_boundary_metadata/' + key)
    report = actual['unit']
    assert report['complete'] is True and set(report['native_module_calls']) == SITE_KEYS
    assert all(value == 1 for value in report['native_module_calls'].values())
    runtime.require_exact(report['native_module_calls'], metadata['unit_hook_native_calls'], 'surviving_actual_site_counts')
    assert metadata['unit_target_intervention_count'] == len(report['branch_events']) + len(report['direct_z_events'])
    assert metadata['extra_direct_down_forward_calibration_calls'] == report['calibration']['direct_down_forward_calls']
    rows = report['action_rows']
    for key, value in report['site'].items():
        if isinstance(value, torch.Tensor) and value.ndim == 2:
            width = 4096 if key in ('x', 'down') else WIDTH
            assert value.shape == (cache['steps'][0]['layout']['gen_len'], width) and value.dtype == torch.bfloat16
    runtime.require_exact(rows.tolist(), metadata['action_rows'], 'surviving_actual_rows_metadata')
    runtime.require_exact(rows, cache['steps'][0]['layout']['action_rows'], 'surviving_actual_rows_cache')
    runtime.require_exact(report['site']['down'][rows], cache['frames'][(0, LAYER, 'mlp')], 'surviving_actual_down_component')
    runtime.require_exact(report['site']['z_down_input'], report['site']['g_effective'] * report['site']['u_effective'], 'surviving_actual_product')
    runtime.require_exact(report['site']['z_native_branch_product'], report['site']['g_before'] * report['site']['u_before'], 'surviving_native_branch_product')
    runtime.require_exact(read(folder / 'unit_site.json'), old.json_tensors(report, scan, torch), 'surviving_actual_unit_json')
    with np.load(folder / 'unit_arrays.npz', allow_pickle=False) as arrays:
        expected_keys = {key for key, value in report['site'].items() if isinstance(value, torch.Tensor) and value.ndim == 2}
        assert set(arrays.files) == expected_keys
        for key in expected_keys:
            assert np.array_equal(arrays[key], report['site'][key][rows].float().numpy()), 'Actual PT/NPZ equality/' + key
            assert arrays[key].dtype == np.float32 and np.isfinite(arrays[key]).all()
    with np.load(folder / 'arrays.npz', allow_pickle=False) as arrays:
        for key, value in (('action_boundary_hidden', actual['hidden']), ('readout', actual['readout']), ('action_velocity', actual['velocity'])):
            assert np.array_equal(arrays[key], value.float().numpy()), 'Actual output PT/NPZ equality/' + key
    files = ('metadata.json', 'components.pt', 'model_output.pt', 'actual_head_boundaries.pt', 'arrays.npz',
             'unit_site.pt', 'unit_site.json', 'unit_arrays.npz')
    if (folder / 'metrics.json').exists():
        files = (*files, 'metrics.json')
    manifest = dict(run=folder.name, files_sha256={name: sha(folder / name) for name in files},
        producer_existing_sha_fields_checked=True, PT_BF16_to_FP32_NPZ_exact=True,
        seal_scope='These whole-case file hashes are first observed and sealed by continuation; phase1 never wrote results/complete.')
    return actual, manifest


def validate_footprint(actual, native, source, indices, runtime):
    torch = runtime.torch
    report, baseline = actual['unit'], native['unit']
    site, old_site, rows = report['site'], baseline['site'], report['action_rows']
    runtime.require_exact(rows, baseline['action_rows'], 'same_native_action_rows')
    nonaction = torch.ones(site['z_landed'].shape[0], dtype=torch.bool)
    nonaction[rows] = False
    selected = torch.tensor(indices, dtype=torch.long)
    unused = torch.ones(WIDTH, dtype=torch.bool)
    unused[selected] = False
    for key in ('x', 'gate_pre', 'g_before', 'u_before', 'z_native_branch_product'):
        runtime.require_exact(site[key], old_site[key], 'surviving_native_context/' + key)
    for key in ('g', 'u'):
        effective = old_site[key + '_before'].clone()
        if report['mode'] in ('factor_joint', 'factor_' + key):
            effective[rows[:, None], selected[None, :]] = source['unit']['site'][key + '_before'][rows][:, selected]
        runtime.require_exact(site[key + '_effective'], effective, 'surviving_actual_branch_after/' + key)
    runtime.require_exact(site['z_landed'][nonaction], old_site['z_down_input'][nonaction], 'surviving_nonaction_z')
    runtime.require_exact(site['z_landed'][rows][:, unused], old_site['z_down_input'][rows][:, unused], 'surviving_unselected_z')
    runtime.require_exact(site['down'][nonaction], old_site['down'][nonaction], 'surviving_nonaction_down')
    if report['mode'] in ('z_swap', 'z_self'):
        expected = native if report['mode'] == 'z_self' else source
        runtime.require_exact(site['z_down_input'], old_site['z_down_input'], 'surviving_actual_z_before')
        runtime.require_exact(site['z_landed'][rows][:, selected], expected['unit']['site']['z_down_input'][rows][:, selected], 'surviving_actual_requested_z_after')
    delta_norm = float((site['down'][rows].double() - old_site['down'][rows].double()).norm())
    assert delta_norm == site['actual_down_delta_l2']
    if report['mode'].startswith('random_'):
        calibration = report['calibration']
        target = calibration['target_actual_down_delta_l2']
        error = abs(delta_norm - target) / target
        assert error <= 1e-3 and error == calibration['actual_model_relative_error']
        assert calibration['actual_model_down_delta_l2'] == delta_norm
        assert calibration['native_reconstruction_full_GEN_bit_exact'] is True
        assert calibration['direct_down_forward_calls'] == len(calibration['tested_gains']) <= 80
        assert 0 < calibration['selected_gain'] <= calibration['maximum_tested_gain'] <= 128


def checked_input(rr, pr, seed, builder, runtime, scan):
    torch = runtime.torch
    ar, ap = rr['kwargs']['action_tokens'][0], pr['kwargs']['action_tokens'][0]
    changed, audit = builder.build_input(ar.clone(), ap.clone(), seed, torch)
    assert changed.shape == ar.shape == (16, 64) and changed.dtype == ar.dtype and changed.device.type == 'cpu'
    assert torch.isfinite(changed).all() and audit['cpu_generator_seed'] == seed
    runtime.require_exact(changed[:, 3:], ar[:, 3:], 'generic_unchanged_other61')
    natural = ap.double() - ar.double()
    delta, xyz = changed[:, :3].double() - ar[:, :3].double(), natural[:, :3]
    target, norm = float(xyz.norm()), float(delta.norm())
    error = abs(norm - target) / target
    cosine = float((delta * xyz).sum() / (norm * target))
    assert target > 0 and norm > 0 and error <= 1e-3 and abs(cosine) <= 1e-3
    runtime.require_exact(audit['actual_action_tokens'], changed, 'builder_actual_action_audit')
    runtime.require_exact(audit['actual_xyz_delta'], delta, 'builder_actual_delta_audit')
    runtime.require_exact(audit['natural_full_delta'], natural, 'builder_actual_natural_delta_audit')
    assert audit['actual_norm_relative_error'] == error and audit['actual_cosine_with_natural_xyz'] == cosine
    assert audit['norm_tolerance'] == audit['orthogonal_cosine_tolerance'] == 1e-3
    assert audit['selection_uses_model_outputs'] is False
    assert 1 <= audit['calibration_iterations'] <= audit['maximum_iterations'] == 10000
    generator = torch.Generator(device='cpu').manual_seed(seed)
    raw = torch.randn((16, 3), generator=generator, dtype=torch.float64, device='cpu')
    runtime.require_exact(raw, audit['raw_direction'], 'same_preregistered_CPU_random_draw')
    assert len(audit['calibration_trace']) == audit['calibration_iterations']
    history = audit['actual_landed_xyz_history']
    proposals = audit['desired_delta_history']
    assert history.shape == proposals.shape == (audit['calibration_iterations'], 16, 3)
    runtime.require_exact(history, (ar[:, :3].double()[None] + proposals).to(ar.dtype), 'builder_all_actual_rounding_history')
    for item in audit['calibration_trace'][:-1]:
        assert item['actual_norm_relative_error'] > 1e-3 or abs(item['actual_cosine']) > 1e-3, 'Builder must choose first passing input'
    kwargs = dict(rr['kwargs'])
    kwargs['action_tokens'] = [changed]
    for key in kwargs:
        if key != 'action_tokens':
            runtime.require_exact(kwargs[key], rr['kwargs'][key], 'generic_all_other_kwargs_RR/' + key)
    reference = dict(rr)
    reference['kwargs'] = kwargs
    reference['cache'] = dict(rr['cache'], first_model_kwargs=kwargs)
    audit.update(independent_actual_input_gate=True, natural_xyz_fraction_of_all64_delta_l2=target / float(natural.norm()),
        norm_and_cosine_recomputed_from_actual_dtypes=True, all_other61_and_nonaction_kwargs_RR_exact=True)
    return reference, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Baseline containing the preserved failed mlp-units phase1')
    baseline = parser.parse_args().output.resolve()
    source = ROOT / 'work'
    old_path, builder_path = source / 'probe_cosmos_milk_mlp_units.py', source / 'cosmos_quantized_orthogonal_input.py'
    assert sha(old_path) == OLD_SHA and sha(builder_path) == BUILDER_SHA
    old, builder = load('generic_frozen_units_probe', old_path), load('generic_frozen_CPU_input_builder', builder_path)
    paths = dict(normal_runtime=source / 'serve_cosmos_goal_pair.py', factory=source / 'probe_cosmos_language_routes.py',
        components=source / 'cosmos_component_interventions.py', pulse=source / 'probe_cosmos_milk_component_pulses.py',
        feedback=source / 'probe_cosmos_milk_feedback_inputs.py', correction=source / 'probe_cosmos_milk_action_correction_sites.py')
    scan, correction = load('generic_frozen_metrics', paths['pulse']), load('generic_frozen_correction', paths['correction'])
    scan.validate_baseline(baseline)
    phase1 = baseline / 'mlp-units'
    assert not (phase1 / 'complete.json').exists() and not (phase1 / 'results.json').exists()
    assert not (phase1 / 'parameters-after.json').exists()
    failed, provenance = read(phase1 / 'failed.json'), read(phase1 / 'provenance.json')
    expected = expected_phase1()
    assert failed['state'] == 'failed' and failed['error'] == 'Actual generic input norm/orthogonality gate failed'
    assert failed['completed'] == expected and len(expected) == 18
    assert all(not (phase1 / f'generic_{kind}_seed{seed}').exists() for seed in INPUT_SEEDS for kind in ('native', 'restore128'))
    assert provenance['script_sha256'] == OLD_SHA
    for key, path in paths.items():
        assert sha(path) == provenance['sources'][key]['sha256']
    for path, expected_sha in provenance['input_json_sha256'].items():
        assert sha(Path(path)) == expected_sha
    assert provenance['protocol_sha256'] == sha(phase1 / 'protocol.json')
    assert provenance['input_contract_sha256'] == sha(phase1 / 'input-contract.json')
    assert provenance['full_parameter_before_sha256'] == sha(phase1 / 'parameters-before.json')
    selection, frozen = read(phase1 / 'selection.json'), read(phase1 / 'selection-frozen.json')
    assert frozen['selection_sha256'] == sha(phase1 / 'selection.json')
    assert frozen['protocol_sha256'] == sha(phase1 / 'protocol.json')
    assert frozen['before_first_intervention_output_score'] is True
    assert selection['selection_arrays_sha256'] == sha(phase1 / 'selection-arrays.pt')
    indices = selection['nested_indices']['128']
    assert len(indices) == len(set(indices)) == 128
    assert set(selection['nested_indices']['32']).issubset(indices) and set(indices).issubset(selection['nested_indices']['512'])
    out = baseline / 'mlp-generic-inputs'
    out.mkdir(exist_ok=False)
    write(out / 'protocol.json', dict(expected_new_model_forwards=6, original_surviving_model_forwards=18,
        original_phase_state='failed', original_complete_and_post18_parameter_hash_absent=True,
        original_probe_sha256=OLD_SHA, constructor_sha256=BUILDER_SHA, seeds=list(INPUT_SEEDS),
        fixed_site=dict(layer_zero_based=LAYER, denoising_step=STEP, units=indices),
        selection_sha256=sha(phase1 / 'selection.json'), norm_tolerance=1e-3, orthogonal_cosine_tolerance=1e-3,
        all_three_inputs_frozen_before_any_new_model_output=True, input_selection_uses_model_outputs=False,
        generic_axis='Each Gi native raw XYZ -> RR raw XYZ; separate denominator for each Gi.',
        old_phase_case_manifest_scope='No old results/complete was written. Surviving files are first observed and sealed here, while their existing producer SHA relationships are checked.',
        parameter_hash_scope='New process before matches old before; new process after equals its before. No measurement of old process post18.', scope=SCOPE), exclusive=True)
    started, completed, hooks = time.perf_counter(), [], None

    def progress(stage, **fields):
        value = dict(stage=stage, new_actual_model_forwards=len(completed), expected_new_model_forwards=6,
            old_surviving_model_forwards=18, completed=completed, elapsed_s=time.perf_counter() - started, **fields)
        write(out / 'progress.json', value)
        print('[GENERIC-CONTINUATION] ' + json.dumps(value, allow_nan=False), flush=True)

    try:
        os.environ['HF_HUB_OFFLINE'] = '1'
        base, factory = load('generic_native_runtime', paths['normal_runtime']), load('generic_native_factory', paths['factory'])
        factory.OUT = out
        site_prov = read(baseline / 'action-correction-sites/provenance.json')
        assert sha(factory.SCHEDULER) == site_prov['scheduler_sha256']
        progress('loading_model')
        runtime = base.NormalRuntime(factory)
        torch, model = runtime.torch, runtime.pipe.transformer
        assert not model.training and not torch.is_grad_enabled() and not model.is_cache_enabled
        assert torch.__version__ == provenance['actual_torch_version']
        assert sha(Path(inspect.getfile(type(model)))) == provenance['actual_transformer_sha256']
        assert sha(Path(inspect.getfile(type(runtime.pipe)))) == provenance['actual_pipeline_source_sha256']
        module = load('generic_readonly_components', paths['components'])
        hooks = module.CosmosComponentInterventions(model)
        refs = {label: correction.load_reference(baseline / f'feedback-inputs/seed{SEED}/replays' / label, hooks, runtime, scan, label) for label in ('RR', 'PR')}
        rr, pr = refs['RR'], refs['PR']
        for label, reference in refs.items():
            assert sha(reference['folder'] / 'metadata.json') == site_prov['reference_metadata_sha256'][label]
        original_contract = read(phase1 / 'input-contract.json')
        assert original_contract['both_actual_pure_noise_draws_exact'] is original_contract['all_30_schedule_pack_metadata_exact'] is True
        raw_records, raw_caches = {}, {}
        for role, scene in (('R', 'x15'), ('D', 'x06')):
            folder = baseline / 'closed-loop' / f'{scene}_seed{SEED}' / 'chunk_00'
            for name in ('states', 'components'):
                assert sha(folder / (name + '.pt')) == original_contract['raw_sources'][role][name + '_sha256']
            raw_records[role] = torch.load(folder / 'states.pt', map_location='cpu', weights_only=True, mmap=True)
            raw_caches[role] = torch.load(folder / 'components.pt', map_location='cpu', weights_only=True, mmap=True)
            scan.validate_record(raw_records[role], torch)
            scan.validate_cache(raw_caches[role], raw_records[role], hooks, runtime, role)
        for key in ('pure_noise', 'timesteps', 'sigmas'):
            runtime.require_exact(raw_records['R'][key], raw_records['D'][key], 'continuation_original_two_draw_full_schedule/' + key)
        runtime.require_exact(raw_caches['R']['steps'], raw_caches['D']['steps'], 'continuation_original_complete_30step_metadata')
        runtime.require_exact(rr['velocity'], raw_records['R']['action_velocity'][STEP], 'continuation_RR_original_t21_velocity')
        before_time = time.perf_counter()
        progress('hash_new_process_parameter_bytes_before')
        parameters_before = old.parameter_hashes(model, torch)
        before_elapsed = time.perf_counter() - before_time
        assert parameters_before == read(phase1 / 'parameters-before.json')
        write(out / 'parameters-before.json', parameters_before, exclusive=True)
        progress('independent_readonly_phase1_array_and_control_checks')
        surviving, manifests, old_metrics = {}, [], []
        for label in expected:
            surviving[label], manifest = read_actual(phase1 / label, runtime, correction, scan, old)
            correction.validate_single_cache(surviving[label]['cache'], hooks, runtime, label)
            manifests.append(manifest)
        native_r, native_p = surviving['native_RR'], surviving['native_PR']
        for label, reference in (('native_RR', rr), ('native_PR', pr), ('self_all12288', pr), ('factor128_PP', pr)):
            correction.exact_reference_result(surviving[label], reference, runtime, 'phase1_independent_ENTIRE/' + label)
        rows = native_r['unit']['action_rows']
        runtime.require_exact(native_r['unit']['site']['down'][rows], rr['cache']['frames'][(0, LAYER, 'mlp')], 'native_RR_actual_down_source')
        for context, capture in (('RR', native_r), ('PR', native_p)):
            assert selection['native_unit_capture_sha256'][context] == sha(capture['folder'] / 'unit_site.pt')
        upstream_results = read(baseline / 'action-correction-sites/results.json')
        whole_row = next(row for row in upstream_results['cases'] if row['run'] == 'layer03_mlp')
        assert whole_row['metadata_sha256'] == provenance['whole_metadata_sha256']
        whole = old.load_whole(baseline / 'action-correction-sites/runs/layer03_mlp', whole_row, correction, runtime)
        correction.exact_reference_result(surviving['swap_all12288'], whole, runtime, 'phase1_independent_ENTIRE_whole_swap')
        runtime.require_exact(surviving['swap_all12288']['unit']['site']['down'][rows], native_r['unit']['site']['down'][rows], 'phase1_full_units_actual_down_RR')
        runtime.require_exact(surviving['factor128_RR']['unit']['site']['z_landed'], surviving['natural128']['unit']['site']['z_landed'], 'phase1_factor_joint_actual_z')
        correction.exact_reference_result(surviving['factor128_RR'], surviving['natural128'], runtime, 'phase1_factor_joint_ENTIRE')
        for label in expected[2:]:
            actual = surviving[label]
            reverse = label == 'reverse128'
            native, source_capture = (native_r, native_p) if reverse else (native_p, native_r)
            selected = actual['unit']['selected_indices'].tolist()
            if label in ('self_all12288', 'swap_all12288'):
                runtime.require_exact(selected, list(range(WIDTH)), 'frozen_all12288_indices')
            elif label.startswith('natural'):
                runtime.require_exact(selected, selection['nested_indices'][str(int(label[7:]))], 'frozen_nested_indices')
            elif label.startswith('random_support'):
                runtime.require_exact(selected, selection['random_support_sets'][label.split('seed')[1]]['indices'], 'frozen_random_support_indices')
            else:
                runtime.require_exact(selected, indices, 'frozen_S128_indices')
            reference = rr if reverse else pr
            for key in ('first_model_kwargs', 'steps', 'site_specs'):
                runtime.require_exact(actual['cache'][key], reference['cache'][key], 'surviving_actual_input_contract/' + label + '/' + key)
            validate_footprint(actual, native, source_capture, selected, runtime)
            metric = old.metric_row(actual, rr if reverse else pr, pr if reverse else rr, scan, label, 'RR_reverse' if reverse else 'PR')
            runtime.require_exact(metric, read(actual['folder'] / 'metrics.json'), 'phase1_saved_metric_recomputed/' + label)
            old_metrics.append(metric)
        natural_down_norm = surviving['natural128']['unit']['site']['actual_down_delta_l2']
        for label in expected[12:]:
            assert surviving[label]['unit']['calibration']['target_actual_down_delta_l2'] == natural_down_norm
        whole_metric = scan.direction_metrics(pr['velocity'][:, :3], rr['velocity'][:, :3], whole['velocity'][:, :3])
        assert whole_metric['gap_meaningful'] is True
        assert whole_metric['signed_projection_fraction'] == read(phase1 / 'protocol.json')['existing_whole_actual_restoration_fraction']
        selected_arrays = torch.load(phase1 / 'selection-arrays.pt', map_location='cpu', weights_only=True, mmap=True)
        runtime.require_exact(selected_arrays['actual_down_W_columns'], runtime.cpu(model.layers[LAYER].mlp_moe_gen.down_proj.weight)[:, selected_arrays['indices']], 'frozen_actual_W_columns_against_reloaded_weights')
        raw_old_before_sha = sha(phase1 / 'parameters-before.json')
        phase1_validation = dict(original_state='failed', original_model_forwards=18, original_error=failed['error'],
            original_failed_sha256=sha(phase1 / 'failed.json'), original_probe_sha256=OLD_SHA,
            original_complete_absent=True, original_parameter_after_absent=True,
            original_post18_parameter_unchanged_verdict=None,
            entire_native_self_whole_and_joint_controls_independently_exact=True, actual_PT_NPZ_arrays_and_metrics_recomputed=True,
            original_actual_two_noise_draws_and_complete_30step_contract_rechecked=True,
            phase1_files_first_observed_and_sealed_at_continuation=True, files=manifests,
            existing_source_provenance_sha256=sha(phase1 / 'provenance.json'), selection_sha256=sha(phase1 / 'selection.json'),
            original_parameter_before_sha256=raw_old_before_sha, scope=SCOPE)
        write(out / 'phase1-validation.json', phase1_validation, exclusive=True)
        inputs = {}
        progress('freeze_all_three_actual_inputs_before_new_forwards')
        for seed in INPUT_SEEDS:
            reference, audit = checked_input(rr, pr, seed, builder, runtime, scan)
            folder = out / f'input_seed{seed}'
            folder.mkdir(exist_ok=False)
            torch.save(audit, folder / 'audit.pt')
            write(folder / 'audit.json', old.json_tensors(audit, scan, torch), exclusive=True)
            inputs[seed] = (reference, audit)
        input_manifest = {str(seed): dict(audit_sha256=sha(out / f'input_seed{seed}/audit.pt'),
            audit_json_sha256=sha(out / f'input_seed{seed}/audit.json'), iterations=audit['calibration_iterations'],
            actual_norm_relative_error=audit['actual_norm_relative_error'], actual_cosine=audit['actual_cosine_with_natural_xyz'])
            for seed, (_, audit) in inputs.items()}
        write(out / 'inputs-frozen.json', dict(inputs=input_manifest, seeds=list(INPUT_SEEDS),
            builder_sha256=BUILDER_SHA, before_any_new_model_output=True, selection_uses_model_outputs=False), exclusive=True)
        frozen_inputs_sha = sha(out / 'inputs-frozen.json')
        write(out / 'provenance.json', dict(script_sha256=sha(Path(__file__)), original_probe_sha256=OLD_SHA,
            input_builder_sha256=BUILDER_SHA, original_phase1_provenance_sha256=sha(phase1 / 'provenance.json'),
            phase1_validation_sha256=sha(out / 'phase1-validation.json'), protocol_sha256=sha(out / 'protocol.json'),
            inputs_frozen_sha256=frozen_inputs_sha, current_parameter_before_sha256=sha(out / 'parameters-before.json'),
            current_parameter_before_matches_old_before=True, old_post18_parameter_state_measured=False,
            source_sha256={key: sha(path) for key, path in paths.items()},
            actual_transformer_sha256=provenance['actual_transformer_sha256'], actual_pipeline_sha256=provenance['actual_pipeline_source_sha256'],
            scope=SCOPE), exclusive=True)
        cases = []
        for seed in INPUT_SEEDS:
            assert sha(out / 'inputs-frozen.json') == frozen_inputs_sha and sha(phase1 / 'selection.json') == frozen['selection_sha256']
            reference, audit = inputs[seed]
            native_label, patch_label = f'generic_native_seed{seed}', f'generic_restore128_seed{seed}'
            progress('generic_native_single_forward', run=native_label)
            native = old.run(runtime, hooks, correction, scan, reference, native_label, out / native_label)
            completed.append(native_label)
            native_metric = old.metric_row(native, native, rr, scan, native_label, 'RR_plus_orthogonal_XYZ')
            native_metric.update(input_audit_sha256=input_manifest[str(seed)]['audit_sha256'], own_axis=True)
            cases.append(old.seal_case(native, native_metric, runtime, scan))
            progress('generic_S128_single_forward', run=patch_label)
            altered = old.run(runtime, hooks, correction, scan, reference, patch_label, out / patch_label,
                native=native['unit']['site'], source=native_r['unit']['site'], indices=indices, mode='z_swap')
            completed.append(patch_label)
            metric = old.metric_row(altered, native, rr, scan, patch_label, 'RR_plus_orthogonal_XYZ')
            natural_gap = rr['velocity'][:, :3].double() - pr['velocity'][:, :3].double()
            generic_gap = rr['velocity'][:, :3].double() - native['velocity'][:, :3].double()
            metric.update(input_audit_sha256=input_manifest[str(seed)]['audit_sha256'], native_run=native_label,
                own_axis=True, denominator='RR XYZ velocity minus this Gi native XYZ velocity',
                generic_gap_cosine_with_natural_PR_gap=None if float(generic_gap.norm()) == 0 else float((generic_gap * natural_gap).sum() / (generic_gap.norm() * natural_gap.norm())),
                relative_original_PR_axis=scan.direction_metrics(pr['velocity'][:, :3], rr['velocity'][:, :3], altered['velocity'][:, :3]),
                interpretation='Own-axis restoration cannot by itself classify semantic or target-specific computation.')
            cases.append(old.seal_case(altered, metric, runtime, scan))
        assert len(completed) == len(cases) == 6
        progress('hash_new_process_parameter_bytes_after')
        after_time = time.perf_counter()
        parameters_after = old.parameter_hashes(model, torch)
        after_elapsed = time.perf_counter() - after_time
        assert parameters_after == parameters_before
        write(out / 'parameters-after.json', parameters_after, exclusive=True)
        old_by_run = {metric['run']: metric for metric in old_metrics}
        whole_score = read(phase1 / 'protocol.json')['existing_whole_actual_restoration_fraction']
        r128 = old_by_run['natural128']['restoration_xyz']['signed_projection_fraction']
        gate = old_by_run['factor128_RP']['restoration_xyz']['signed_projection_fraction']
        up = old_by_run['factor128_PR']['restoration_xyz']['signed_projection_fraction']
        assessable = all(value is not None for value in (r128, gate, up))
        predictions = dict(whole_restoration=whole_score, half_whole_threshold=.5 * whole_score,
            natural128_restoration=r128, gate_only_restoration=gate, up_only_restoration=up,
            natural128_at_least_half_whole=None if r128 is None else r128 >= .5 * whole_score,
            natural_gate_stronger_than_up=None if gate is None or up is None else gate > up,
            verdict='inconclusive_gap_floor' if not assessable else 'supported' if r128 >= .5 * whole_score and gate > up else 'rejected',
            provenance='Computed from independently verified surviving phase1 arrays; generic outputs do not select units or redefine thresholds.')
        extra_down_calls = sum(actual['unit']['calibration']['direct_down_forward_calls'] for actual in surviving.values())
        write(out / 'results.json', dict(state='complete', phase1_state='failed', phase2_state='complete',
            original_surviving_model_forwards=18, continuation_model_forwards=6, combined_actual_model_forwards=24,
            old_phase1_extra_direct_down_calls=extra_down_calls, continuation_extra_direct_down_calls=0,
            original_phase1_case_hash_scope='First observed at continuation, not an original completed results manifest.',
            original_post18_parameter_state_measured=False, current_before_equals_old_before=True, current_after_equals_current_before=True,
            original_post18_parameter_unchanged_verdict=None,
            parameter_hash_files_sha256={name: sha(out / name) for name in ('parameters-before.json', 'parameters-after.json')},
            current_parameter_hash_elapsed_s=dict(before=before_elapsed, after=after_elapsed),
            phase1_metrics=old_metrics, generic_cases=cases, preregistered_predictions=predictions,
            phase1_validation_sha256=sha(out / 'phase1-validation.json'), inputs_frozen_sha256=frozen_inputs_sha,
            selection_sha256=frozen['selection_sha256'], scope=SCOPE), exclusive=True)
        write(out / 'complete.json', dict(state='complete', phase1_state='failed', phase2_state='complete',
            original_surviving_model_forwards=18, continuation_model_forwards=6, combined_actual_model_forwards=24,
            original_post18_parameter_state_measured=False, only_new_process_parameter_bytes_unchanged=True,
            original_post18_parameter_unchanged_verdict=None,
            all_inputs_frozen_before_new_outputs=True, all_actual_input_norm_and_angle_gates_passed=True,
            all_generic_actual_before_selected_after_and_nonaction_rows_exact=True,
            results_sha256=sha(out / 'results.json'), provenance_sha256=sha(out / 'provenance.json'),
            protocol_sha256=sha(out / 'protocol.json'), inputs_frozen_sha256=frozen_inputs_sha,
            phase1_validation_sha256=sha(out / 'phase1-validation.json'), script_sha256=sha(Path(__file__)), scope=SCOPE), exclusive=True)
        progress('complete', original_state='failed', combined_actual_model_forwards=24, predictions=predictions['verdict'])
    except Exception as exc:
        write(out / 'failed.json', dict(state='failed', error=str(exc), traceback=traceback.format_exc(),
            completed=completed, original_phase1_state='failed', original_surviving_model_forwards=18,
            elapsed_s=time.perf_counter() - started, scope=SCOPE))
        raise
    finally:
        if hooks is not None:
            hooks.reset()


if __name__ == '__main__':
    main()
