"""Audit actual CPU tensors and plot the interrupted 18+6 units experiment.

Single entry point for the existing Torch/NumPy/Matplotlib CPU environment.
No model creation, inference, simulator, network request or CUDA context.
The original mlp-units phase remains failed; mlp-generic-inputs completed six
continuation forwards. Their combined 24 forwards are never described as an
uninterrupted completed original phase. Parameter-after evidence covers only
the new six-forward process. Figures are written to an absent new directory.
"""

import argparse
import hashlib
import html
import importlib.util
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace


OLD_SHA = '6818a6a3c3ae2f665040f2196c00055fc9487b7775642601e7ca7e9e69c95a6a'
CONTINUATION_SHA = '614434e2e52cf4340c0533adb16d46c62f01c1b4a46918679b13aae626564a66'
BUILDER_SHA = '15cd6e671014c69dc1db4b6a06b6b86a9c1ba615f2a96c08918a8e8f3690b842'
FIRST_THREE_UNITS = (8015, 11140, 5008)
INPUT_SEEDS = (5219801, 5219802, 5219803)
SUPPORT_SEEDS = (3219801, 3219802, 3219803)
COEFFICIENT_SEEDS = (4219801, 4219802, 4219803)
SCORE = ('Natural/factorial/random: 100*dot(Vtrial-V_PR,V_RR-V_PR)/||V_RR-V_PR||^2, '
         'over sixteen raw Flow XYZ velocity rows at fixed denoising t21. Each generic Gi: '
         '100*dot(VGi_patch-VGi_native,V_RR-VGi_native)/||V_RR-VGi_native||^2. '
         'Generic denominators differ. Values are axis projections, not identities, '
         'probabilities, physical corrections, grasp rates or additive unit contributions.')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text())


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CPUValues:
    """Only tensor comparisons; there is intentionally no pipeline/model API."""

    def __init__(self, torch):
        self.torch = torch

    def cpu(self, value):
        if isinstance(value, self.torch.Tensor):
            assert value.device.type == 'cpu'
            return value.detach().clone()
        if isinstance(value, dict):
            return {key: self.cpu(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return type(value)(self.cpu(item) for item in value)
        return value

    def require_exact(self, a, b, label):
        torch = self.torch
        if isinstance(a, torch.Tensor):
            assert isinstance(b, torch.Tensor) and a.device.type == b.device.type == 'cpu', label
            assert a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b), label
            assert torch.equal(a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8)), label
        elif isinstance(a, dict):
            assert isinstance(b, dict) and a.keys() == b.keys(), label
            for key in a:
                self.require_exact(a[key], b[key], label + '/' + str(key))
        elif isinstance(a, (list, tuple)):
            assert type(a) is type(b) and len(a) == len(b), label
            for index, (left, right) in enumerate(zip(a, b)):
                self.require_exact(left, right, label + '/' + str(index))
        else:
            assert a == b, label


def translated_input_path(baseline, path):
    parts = Path(path).parts
    for anchor in ('feedback-inputs', 'action-correction-sites'):
        if anchor in parts:
            return baseline.joinpath(*parts[parts.index(anchor):])
    raise ValueError('Unexpected upstream provenance input: ' + path)


def percent(metric):
    value = metric['restoration_xyz']['signed_projection_fraction']
    return None if value is None else 100 * value


def require_export_bytes(folder, actual):
    """Check literal FP32 export bits, including the sign of zero."""
    import numpy as np
    rows = actual['unit']['action_rows']
    site = actual['unit']['site']
    unit_exports = {key: value[rows].float().numpy() for key, value in site.items()
                    if hasattr(value, 'ndim') and value.ndim == 2}
    head_exports = dict(action_boundary_hidden=actual['hidden'].float().numpy(),
                        readout=actual['readout'].float().numpy(), action_velocity=actual['velocity'].float().numpy())
    for filename, expected in (('unit_arrays.npz', unit_exports), ('arrays.npz', head_exports)):
        with np.load(folder / filename, allow_pickle=False) as arrays:
            assert set(arrays.files) == set(expected), str(folder / filename)
            for key, value in expected.items():
                assert arrays[key].dtype == value.dtype == np.float32 and arrays[key].shape == value.shape
                assert arrays[key].tobytes(order='C') == value.tobytes(order='C'), f'Actual BF16-to-FP32 export bytes/{folder.name}/{key}'


def audit(baseline, torch):
    import numpy as np
    runtime = CPUValues(torch)
    source = Path(__file__).resolve().parent
    old_path, continuation_path, builder_path = (source / name for name in
        ('probe_cosmos_milk_mlp_units.py', 'probe_cosmos_milk_generic_inputs.py', 'cosmos_quantized_orthogonal_input.py'))
    assert sha(old_path) == OLD_SHA and sha(continuation_path) == CONTINUATION_SHA and sha(builder_path) == BUILDER_SHA
    old, continuation, builder = load('plot_frozen_units', old_path), load('plot_frozen_continuation', continuation_path), load('plot_frozen_input_builder', builder_path)
    scan = load('plot_frozen_direction_metrics', source / 'probe_cosmos_milk_component_pulses.py')
    correction = load('plot_frozen_record_comparisons', source / 'probe_cosmos_milk_action_correction_sites.py')
    original, root = baseline / 'mlp-units', baseline / 'mlp-generic-inputs'
    assert not (root / 'failed.json').exists()
    assert not (original / 'complete.json').exists() and not (original / 'results.json').exists() and not (original / 'parameters-after.json').exists()
    complete, results, provenance, validation = (read(root / name) for name in ('complete.json', 'results.json', 'provenance.json', 'phase1-validation.json'))
    assert complete['state'] == results['state'] == 'complete'
    assert complete['phase1_state'] == results['phase1_state'] == 'failed'
    assert complete['phase2_state'] == results['phase2_state'] == 'complete'
    for key, expected in (('original_surviving_model_forwards', 18), ('continuation_model_forwards', 6), ('combined_actual_model_forwards', 24)):
        assert complete[key] == results[key] == expected
    for key, filename in (('results_sha256', 'results.json'), ('provenance_sha256', 'provenance.json'),
                          ('protocol_sha256', 'protocol.json'), ('inputs_frozen_sha256', 'inputs-frozen.json'),
                          ('phase1_validation_sha256', 'phase1-validation.json')):
        assert complete[key] == sha(root / filename)
    assert complete['script_sha256'] == provenance['script_sha256'] == CONTINUATION_SHA
    for key in ('only_new_process_parameter_bytes_unchanged', 'all_inputs_frozen_before_new_outputs',
                'all_actual_input_norm_and_angle_gates_passed', 'all_generic_actual_before_selected_after_and_nonaction_rows_exact'):
        assert complete[key] is True
    assert complete['original_post18_parameter_state_measured'] is False
    assert complete['original_post18_parameter_unchanged_verdict'] is None
    assert validation['original_state'] == 'failed' and validation['original_model_forwards'] == 18
    assert validation['phase1_files_first_observed_and_sealed_at_continuation'] is True
    for key in ('entire_native_self_whole_and_joint_controls_independently_exact', 'actual_PT_NPZ_arrays_and_metrics_recomputed',
                'original_actual_two_noise_draws_and_complete_30step_contract_rechecked'):
        assert validation[key] is True
    assert validation['original_complete_absent'] is validation['original_parameter_after_absent'] is True
    assert validation['original_post18_parameter_unchanged_verdict'] is None
    assert validation['original_failed_sha256'] == sha(original / 'failed.json')
    failed = read(original / 'failed.json')
    assert failed['state'] == 'failed' and failed['completed'] == continuation.expected_phase1()
    assert len(validation['files']) == 18 and [row['run'] for row in validation['files']] == continuation.expected_phase1()
    old_prov = read(original / 'provenance.json')
    assert provenance['original_phase1_provenance_sha256'] == validation['existing_source_provenance_sha256'] == sha(original / 'provenance.json')
    assert old_prov['script_sha256'] == OLD_SHA and provenance['input_builder_sha256'] == BUILDER_SHA
    for key, expected in provenance['source_sha256'].items():
        path = source / Path(old_prov['sources'][key]['path']).name
        assert sha(path) == expected == old_prov['sources'][key]['sha256']
    upstream_sources = {}
    for name, expected in old_prov['input_json_sha256'].items():
        path = translated_input_path(baseline, name)
        assert sha(path) == expected
        upstream_sources[str(path)] = expected
    assert old_prov['protocol_sha256'] == sha(original / 'protocol.json')
    assert old_prov['input_contract_sha256'] == sha(original / 'input-contract.json')
    assert old_prov['full_parameter_before_sha256'] == sha(original / 'parameters-before.json')
    parameters = {name: read(root / name) for name in ('parameters-before.json', 'parameters-after.json')}
    for name in parameters:
        assert results['parameter_hash_files_sha256'][name] == sha(root / name)
    assert parameters['parameters-before.json'] == parameters['parameters-after.json'] == read(original / 'parameters-before.json')
    parameter_hash = parameters['parameters-before.json']
    digest = hashlib.sha256()
    for name, value in sorted(parameter_hash['parameters'].items()):
        digest.update(json.dumps([name, value], sort_keys=True).encode())
    assert digest.hexdigest() == parameter_hash['aggregate_sha256']
    assert parameter_hash['parameter_count'] == len(parameter_hash['parameters'])
    selection, frozen = read(original / 'selection.json'), read(original / 'selection-frozen.json')
    assert frozen['selection_sha256'] == results['selection_sha256'] == validation['selection_sha256'] == sha(original / 'selection.json')
    assert frozen['protocol_sha256'] == sha(original / 'protocol.json') and frozen['before_first_intervention_output_score'] is True
    assert selection['entire_transformer_parameter_bytes_sha256'] == parameter_hash['aggregate_sha256']
    assert selection['selection_arrays_sha256'] == sha(original / 'selection-arrays.pt')
    selected = selection['nested_indices']['128']
    assert len(selected) == len(set(selected)) == 128 and tuple(selected[:3]) == FIRST_THREE_UNITS
    weights = torch.load(original / 'selection-arrays.pt', map_location='cpu', weights_only=True, mmap=True)
    assert weights['actual_down_W_columns'].shape == (4096, len(weights['indices']))
    assert weights['actual_down_W_columns'].dtype == torch.bfloat16
    down_parameters = [value for name, value in parameter_hash['parameters'].items() if name == 'layers.3.mlp_moe_gen.down_proj.weight']
    assert len(down_parameters) == 1
    for key in ('shape', 'dtype', 'sha256'):
        assert selection['actual_full_down_weight'][key] == down_parameters[0][key]
    assert np.array_equal(weights['proxy_signed_scores'].numpy(), np.asarray(selection['proxy_signed_scores']))
    assert selection['ranking'] == sorted(range(12288), key=lambda index: (-selection['proxy_signed_scores'][index], index))
    records, manifests = {}, {}
    for row in validation['files']:
        folder = original / row['run']
        for name, expected in row['files_sha256'].items():
            assert sha(folder / name) == expected
        records[row['run']], manifests[row['run']] = continuation.read_actual(folder, runtime, correction, scan, old)
        require_export_bytes(folder, records[row['run']])
        print('[UNITS-PLOT] checked actual phase1 ' + row['run'], flush=True)
    native_r, native_p = records['native_RR'], records['native_PR']
    rows = native_r['unit']['action_rows']
    runtime.require_exact(rows, native_p['unit']['action_rows'], 'plot_native_rows_align')
    specs = SimpleNamespace(layers=list(range(36)), site_specs=native_r['cache']['site_specs'])
    for label, actual in records.items():
        correction.validate_single_cache(actual['cache'], specs, runtime, label)
    refs = {label: correction.load_reference(baseline / 'feedback-inputs/seed198/replays' / label, specs, runtime, scan, label) for label in ('RR', 'PR')}
    rr, pr = refs['RR'], refs['PR']
    site_prov = read(baseline / 'action-correction-sites/provenance.json')
    for label, ref in refs.items():
        assert sha(ref['folder'] / 'metadata.json') == site_prov['reference_metadata_sha256'][label]
    for label, ref in (('native_RR', rr), ('native_PR', pr), ('self_all12288', pr), ('factor128_PP', pr)):
        correction.exact_reference_result(records[label], ref, runtime, 'plot_independent_entire/' + label)
    upstream_results = read(baseline / 'action-correction-sites/results.json')
    whole_row = next(row for row in upstream_results['cases'] if row['run'] == 'layer03_mlp')
    assert whole_row['metadata_sha256'] == old_prov['whole_metadata_sha256']
    whole = old.load_whole(baseline / 'action-correction-sites/runs/layer03_mlp', whole_row, correction, runtime)
    correction.exact_reference_result(records['swap_all12288'], whole, runtime, 'plot_independent_entire_whole')
    correction.exact_reference_result(records['factor128_RR'], records['natural128'], runtime, 'plot_independent_entire_joint')
    runtime.require_exact(records['factor128_RR']['unit']['site']['z_landed'], records['natural128']['unit']['site']['z_landed'], 'plot_actual_joint_z')
    old_metrics = {row['run']: row for row in results['phase1_metrics']}
    assert len(old_metrics) == 16 and set(old_metrics) == set(continuation.expected_phase1()[2:])
    for label, saved in old_metrics.items():
        reverse = label == 'reverse128'
        actual = records[label]
        native, endpoint = (native_r, native_p) if reverse else (native_p, native_r)
        actual_indices = actual['unit']['selected_indices'].tolist()
        if label in ('self_all12288', 'swap_all12288'):
            expected_indices = list(range(12288))
        elif label.startswith('natural'):
            expected_indices = selection['nested_indices'][str(int(label[7:]))]
        elif label.startswith('random_support'):
            expected_indices = selection['random_support_sets'][label.split('seed')[1]]['indices']
        else:
            expected_indices = selected
        runtime.require_exact(actual_indices, expected_indices, 'plot_frozen_unit_groups')
        for key in ('first_model_kwargs', 'steps', 'site_specs'):
            runtime.require_exact(actual['cache'][key], (rr if reverse else pr)['cache'][key], 'plot_exact_old_context/' + key)
        continuation.validate_footprint(actual, native, endpoint, actual['unit']['selected_indices'].tolist(), runtime)
        metric = old.metric_row(actual, rr if reverse else pr, pr if reverse else rr, scan, label, 'RR_reverse' if reverse else 'PR')
        runtime.require_exact(metric, saved, 'plot_phase1_raw_metric_recomputed')
        runtime.require_exact(metric, read(original / label / 'metrics.json'), 'plot_phase1_metric_artifact')
    natural_norm = records['natural128']['unit']['site']['actual_down_delta_l2']
    extra_calls = 0
    for label in continuation.expected_phase1():
        info = records[label]['unit']['calibration']
        extra_calls += info['direct_down_forward_calls']
        if label.startswith('random_'):
            assert info['target_actual_down_delta_l2'] == natural_norm
    assert results['old_phase1_extra_direct_down_calls'] == extra_calls and results['continuation_extra_direct_down_calls'] == 0
    inputs = read(root / 'inputs-frozen.json')
    assert inputs['builder_sha256'] == BUILDER_SHA and inputs['seeds'] == list(INPUT_SEEDS)
    assert inputs['before_any_new_model_output'] is True and inputs['selection_uses_model_outputs'] is False
    cases = {row['run']: row for row in results['generic_cases']}
    expected_generic = [f'generic_{kind}_seed{seed}' for seed in INPUT_SEEDS for kind in ('native', 'restore128')]
    assert len(cases) == 6 and set(cases) == set(expected_generic)
    generic = []
    for seed in INPUT_SEEDS:
        folder = root / f'input_seed{seed}'
        info = inputs['inputs'][str(seed)]
        assert info['audit_sha256'] == sha(folder / 'audit.pt') and info['audit_json_sha256'] == sha(folder / 'audit.json')
        audit_values = torch.load(folder / 'audit.pt', map_location='cpu', weights_only=True, mmap=True)
        runtime.require_exact(read(folder / 'audit.json'), old.json_tensors(audit_values, scan, torch), 'plot_actual_input_audit_json')
        reference, recomputed_audit = continuation.checked_input(rr, pr, seed, builder, runtime, scan)
        runtime.require_exact(audit_values, recomputed_audit, 'plot_reproduced_CPU_first_passing_input')
        native_label, patch_label = f'generic_native_seed{seed}', f'generic_restore128_seed{seed}'
        for label in (native_label, patch_label):
            saved = cases[label]
            for name, expected in saved['files_sha256'].items():
                assert sha(root / label / name) == expected
            actual, _ = continuation.read_actual(root / label, runtime, correction, scan, old)
            require_export_bytes(root / label, actual)
            correction.validate_single_cache(actual['cache'], specs, runtime, label)
            runtime.require_exact(actual['cache']['first_model_kwargs'], reference['kwargs'], 'plot_generic_actual_input')
            records[label] = actual
            runtime.require_exact(read(root / label / 'metrics.json'), {key: value for key, value in saved.items() if key not in ('files_sha256', 'artifacts')}, 'plot_generic_metric_file')
        native, altered = records[native_label], records[patch_label]
        runtime.require_exact(altered['unit']['selected_indices'].tolist(), selected, 'plot_generic_frozen_S128')
        continuation.validate_footprint(altered, native, native_r, selected, runtime)
        for label, actual in ((native_label, native), (patch_label, altered)):
            core = old.metric_row(actual, native, rr, scan, label, 'RR_plus_orthogonal_XYZ')
            for key, value in core.items():
                runtime.require_exact(value, cases[label][key], 'plot_generic_actual_own_axis/' + key)
        metric = cases[patch_label]
        gap = rr['velocity'][:, :3].double() - native['velocity'][:, :3].double()
        natural_gap = rr['velocity'][:, :3].double() - pr['velocity'][:, :3].double()
        cosine = None if float(gap.norm()) == 0 else float((gap * natural_gap).sum() / (gap.norm() * natural_gap.norm()))
        assert cosine == metric['generic_gap_cosine_with_natural_PR_gap']
        generic.append(dict(seed=seed, percent=percent(metric), restoration=metric['restoration_xyz'],
            actual_input_norm_relative_error=audit_values['actual_norm_relative_error'], actual_input_cosine=audit_values['actual_cosine_with_natural_xyz'],
            input_first_passing_iteration=audit_values['calibration_iterations'], gap_cosine_with_natural_PR_gap=cosine,
            own_native_raw_xyz=native['velocity'][:, :3].tolist(), patched_raw_xyz=altered['velocity'][:, :3].tolist(),
            endpoint_RR_raw_xyz=rr['velocity'][:, :3].tolist()))
    predictions = results['preregistered_predictions']
    whole_score = old_metrics['swap_all12288']['restoration_xyz']['signed_projection_fraction']
    assert whole_score == predictions['whole_restoration'] == read(original / 'protocol.json')['existing_whole_actual_restoration_fraction']
    r128, gate, up = (old_metrics[label]['restoration_xyz']['signed_projection_fraction'] for label in ('natural128', 'factor128_RP', 'factor128_PR'))
    assert predictions['natural128_restoration'] == r128 and predictions['gate_only_restoration'] == gate and predictions['up_only_restoration'] == up
    sparse_supported = None if r128 is None else r128 >= .5 * whole_score
    gate_supported = None if gate is None or up is None else gate > up
    assert predictions['natural128_at_least_half_whole'] is sparse_supported and predictions['natural_gate_stronger_than_up'] is gate_supported
    assert predictions['half_whole_threshold'] == .5 * whole_score
    verdict = 'inconclusive_gap_floor' if sparse_supported is None or gate_supported is None else 'supported' if sparse_supported and gate_supported else 'rejected'
    assert predictions['verdict'] == verdict
    delta_z = (native_r['unit']['site']['z_down_input'][rows][:, selected].double()
               - native_p['unit']['site']['z_down_input'][rows][:, selected].double()).T.numpy()
    changed_hidden = records['natural128']['hidden']
    original_hidden = native_p['hidden']
    runtime.require_exact(changed_hidden[:4], original_hidden[:4], 'propagation_boundaries0_through3_byte_exact')
    runtime.require_exact(records['natural128']['cache']['frames'][(0, 3, 'attention')], native_p['cache']['frames'][(0, 3, 'attention')], 'propagation_target_layer_attention_unchanged')
    hidden_delta = changed_hidden.double() - original_hidden.double()
    assert float(hidden_delta[4].norm()) > 0
    hidden_diff_rms = hidden_delta.square().mean(-1).sqrt()
    original_rms = original_hidden.double().square().mean(-1).sqrt()
    relative_percent = [[None if float(reference) == 0 else float(100 * difference / reference)
                         for difference, reference in zip(differences, references)]
                        for differences, references in zip(hidden_diff_rms, original_rms)]
    down_delta = (records['natural128']['unit']['site']['down'][rows].double()
                  - native_p['unit']['site']['down'][rows].double())
    after_residual_delta = hidden_delta[4]
    propagation = dict(reference='Actual native_PR', intervention='Actual natural128 z replacement',
        boundary_convention='0 before layer0; boundary4 after layer3 native residual addition;36 after layer35.',
        predicted_action_token_indices=list(range(16)), actual_GEN_rows=rows.tolist(),
        boundary_indices=list(range(37)), difference_rms=hidden_diff_rms.tolist(), reference_PR_rms=original_rms.tolist(),
        relative_difference_rms_percent=relative_percent, zero_PR_rms_policy='Null relative value if reference RMS is zero.',
        boundaries0_through3_byte_exact=True, boundary4_nonzero=True,
        sources={label: dict(actual_head_boundaries_sha256=sha(original / label / 'actual_head_boundaries.pt'),
                            unit_site_sha256=sha(original / label / 'unit_site.pt')) for label in ('native_PR', 'natural128')},
        residual_landing=dict(actual_down_delta_l2=float(down_delta.norm()), actual_boundary4_delta_l2=float(after_residual_delta.norm()),
            actual_difference_after_residual_minus_down_l2=float((after_residual_delta - down_delta).norm()),
            actual_down_delta_per_token_l2=down_delta.norm(dim=1).tolist(), actual_boundary4_delta_per_token_l2=after_residual_delta.norm(dim=1).tolist(),
            note='Observed BF16 output difference after native residual addition versus observed down increment difference; no residual tensor is reconstructed.'),
        interpretation='Actual intervention-relative hidden-state differences while later native layers run freely; no RR-PR semantic projection, growth mechanism or error-root-cause claim.')
    examples = []
    for rank, unit in enumerate(selected[:3], 1):
        column = weights['actual_down_W_columns'][:, weights['indices'].index(unit)]
        example = dict(proxy_rank=rank, unit_zero_based=unit, predicted_action_token_index=0,
            actual_GEN_row=int(rows[0]), original_denoising_forward=21, layer_zero_based=3,
            W_column=scan.tensor_meta(column, torch), W_column_first8_values=column[:8].float().tolist())
        for context, actual in (('RR', native_r), ('PR', native_p)):
            values = actual['unit']['site']
            example[context] = {key: float(values[name][rows[0], unit]) for key, name in
                (('actual_postSiLU_gate_g', 'g_before'), ('actual_up_u', 'u_before'), ('actual_product_z', 'z_down_input'), ('actual_gate_pre', 'gate_pre'))}
        examples.append(example)
    sources = dict(plot_code_sha256=sha(Path(__file__)), units_probe_sha256=OLD_SHA, continuation_sha256=CONTINUATION_SHA,
        input_builder_sha256=BUILDER_SHA, current_metrics_code_sha256=sha(source / 'probe_cosmos_milk_component_pulses.py'),
        phase1_validation_sha256=sha(root / 'phase1-validation.json'), phase2_json_sha256={name: sha(root / name) for name in
            ('complete.json', 'results.json', 'provenance.json', 'protocol.json', 'inputs-frozen.json', 'parameters-before.json', 'parameters-after.json')},
        original_failed_sha256=sha(original / 'failed.json'), original_provenance_sha256=sha(original / 'provenance.json'),
        selection_sha256=sha(original / 'selection.json'), selection_arrays_sha256=sha(original / 'selection-arrays.pt'),
        whole_parameter_bytes_aggregate_sha256=parameter_hash['aggregate_sha256'], actual_full_down_weight=selection['actual_full_down_weight'],
        upstream_json_sha256=upstream_sources, original_case_current_hashes=validation['files'], continuation_case_hashes={key: row['files_sha256'] for key, row in cases.items()})
    values = dict(predictions=predictions, sparse_percent={str(size): percent(old_metrics[label]) for size, label in
        ((32, 'natural032'), (128, 'natural128'), (512, 'natural512'), (12288, 'swap_all12288'))},
        branch_percent={cell: percent(old_metrics['factor128_' + cell]) for cell in ('PP', 'RP', 'PR', 'RR')},
        branch_actual_down_delta_l2={cell: old_metrics['factor128_' + cell]['actual_down_delta_l2'] for cell in ('PP', 'RP', 'PR', 'RR')},
        branch_input_amplitudes={cell: old_metrics['factor128_' + cell]['branch_input_amplitudes'] for cell in ('PP', 'RP', 'PR', 'RR')},
        random_support_percent=[percent(old_metrics[f'random_support128_seed{seed}']) for seed in SUPPORT_SEEDS],
        same_support_random_coefficients_percent=[percent(old_metrics[f'random_coefficients128_seed{seed}']) for seed in COEFFICIENT_SEEDS],
        natural128_actual_down_delta_l2=natural_norm, random_controls={label: old_metrics[label] for label in old_metrics if label.startswith('random_')},
        generic=generic, selected128_units_zero_based=selected, first_three_rank_actual_action_token0_examples=examples,
        real_selected128_delta_z=delta_z.tolist(), propagation=propagation, original_extra_direct_down_calls=extra_calls)
    return root, values, sources, delta_z


def label_value(axis, x, value):
    if value is None:
        axis.text(x, 0, 'gap floor', ha='center', va='bottom', fontsize=8)
    else:
        axis.annotate(f'{value:+.2f}%', (x, value), xytext=(0, 5 if value >= 0 else -5),
                      textcoords='offset points', ha='center', va='bottom' if value >= 0 else 'top', fontsize=9)


def render_main(path, values):
    import numpy as np
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(13.6, 10.2))
    sparse, branch, random, generic_axis = axes.flat
    sizes = [32, 128, 512, 12288]
    scores = [values['sparse_percent'][str(size)] for size in sizes]
    sparse.plot(sizes, [np.nan if value is None else value for value in scores], marker='o', color='#0072B2', linewidth=1.7)
    sparse.set_xscale('log', base=2)
    sparse.set_xticks(sizes, [str(size) for size in sizes])
    threshold = 100 * values['predictions']['half_whole_threshold']
    sparse.axhline(threshold, linestyle='--', color='#D55E00', label=f'Half-whole threshold {threshold:.2f}%')
    sparse.legend(fontsize=9, loc='upper left')
    sparse.set_xlabel('Frozen nested selected-unit count (log spacing)')
    sparse.set_title('A  Natural donor z replacement; independent sizes')
    for x, value in zip(sizes, scores):
        label_value(sparse, x, value)
    cells = ('PP', 'RP', 'PR', 'RR')
    branch_values = [values['branch_percent'][cell] for cell in cells]
    branch.bar(np.arange(4), [np.nan if value is None else value for value in branch_values], color=['#999999', '#D55E00', '#0072B2', '#CC79A7'])
    branch.set_xticks(np.arange(4), ['Self\ngP × uP', 'Gate only\ngR × uP', 'Up only\ngP × uR', 'Joint\ngR × uR'])
    branch.set_title('B  S128 branch exchange, not norm equalized')
    for x, value, cell in zip(range(4), branch_values, cells):
        label_value(branch, x, value)
        branch.text(x, .02, f'ΔM L2\n{values["branch_actual_down_delta_l2"][cell]:.3f}', transform=branch.get_xaxis_transform(), ha='center', fontsize=8)
    groups = [[values['sparse_percent']['128']], values['random_support_percent'], values['same_support_random_coefficients_percent']]
    for x, points, color in zip(range(3), groups, ('#D55E00', '#777777', '#0072B2')):
        offsets = [0] if len(points) == 1 else [-.15, 0, .15]
        for offset, value in zip(offsets, points):
            if value is not None:
                random.scatter(x + offset, value, color=color, s=54, zorder=3)
            label_value(random, x + offset, value)
    random.set_xticks(range(3), ['Natural S128', 'Random support\n128 units (3 draws)', 'Same S128\nrandom coefficients (3 draws)'])
    random.set_title('C  Actual BF16 down-output ΔL2 matched to natural S128')
    generic_values = [row['percent'] for row in values['generic']]
    generic_axis.bar(range(3), [np.nan if value is None else value for value in generic_values], color='#009E73', width=.65)
    generic_axis.set_xticks(range(3), [f'Gi {i+1}\ninput seed {row["seed"]}' for i, row in enumerate(values['generic'])])
    generic_axis.set_title('D  S128 restore under unrelated XYZ inputs: own axes')
    for x, value, row in zip(range(3), generic_values, values['generic']):
        label_value(generic_axis, x, value)
        generic_axis.text(x, .025, f'own gap RMS\n{row["restoration"]["donor_recipient_gap_rmse"]:.5f}', transform=generic_axis.get_xaxis_transform(), ha='center', fontsize=8)
    generic_axis.set_xlabel('Each bar uses a different Gi-native → RR denominator')
    for axis in (sparse, branch, random):
        axis.set_ylabel('Signed PR → RR raw XYZ velocity projection (%)')
    generic_axis.set_ylabel('Signed own-axis raw XYZ restoration (%)')
    for axis in axes.flat:
        axis.axhline(0, color='#555555', linewidth=.9)
        axis.grid(axis='y', alpha=.2)
        axis.set_axisbelow(True)
        lower, upper = axis.get_ylim()
        axis.set_ylim(min(0, lower) - .07 * max(upper - lower, 1), upper + .17 * max(upper - lower, 1))
    predictions = values['predictions']
    verdict_a = 'INCONCLUSIVE' if predictions['natural128_at_least_half_whole'] is None else 'SUPPORTED' if predictions['natural128_at_least_half_whole'] else 'REJECTED'
    verdict_b = 'INCONCLUSIVE' if predictions['natural_gate_stronger_than_up'] is None else 'SUPPORTED' if predictions['natural_gate_stronger_than_up'] else 'REJECTED'
    figure.suptitle(f'Fixed t21 MLP layer4 (zero-based L3) — interrupted experiment: 18 + 6 forwards\nPreregistered: S128 ≥ half whole {verdict_a}; natural gate-only > up-only {verdict_b}', fontsize=14, fontweight='bold', y=.98)
    figure.subplots_adjust(left=.08, right=.98, top=.86, bottom=.15, hspace=.52, wspace=.27)
    figure.text(.08, .095, 'R = RR replay; P = PR action-only replay with R vision. g is actual post-SiLU gate, u is actual up branch. Percentages are not probabilities.', fontsize=9)
    figure.text(.08, .055, 'Generic bars have separate denominators; their heights do not establish identity or target specificity. Independent unit-size effects are not additive shares.', fontsize=9)
    figure.text(.08, .015, f'Original phase1 remains failed. {values["original_extra_direct_down_calls"]} extra actual down.forward calibration calls are separate from 24 model forwards; only continuation parameters have an after hash.', fontsize=9)
    figure.savefig(path, dpi=175)
    plt.close(figure)


def render_heatmap(path, delta, units):
    import numpy as np
    import matplotlib.pyplot as plt
    maximum = float(np.abs(delta).max())
    assert delta.shape == (128, 16) and np.isfinite(delta).all() and maximum > 0
    figure, axis = plt.subplots(figsize=(10, 20))
    plot = axis.imshow(delta, cmap='RdBu_r', vmin=-maximum, vmax=maximum, aspect='auto', interpolation='nearest')
    axis.set_xticks(range(16), [str(index) for index in range(16)])
    axis.set_yticks(range(128), [f'{rank+1}: {unit}' for rank, unit in enumerate(units)], fontsize=6.3)
    axis.set_xlabel('Predicted action-token index, not denoising step')
    axis.set_ylabel('Frozen proxy rank: real zero-based unit index')
    axis.set_title('Actual selected-unit Δz = RR z − PR z at down input\nLayer4 (zero-based L3), t21; BF16 samples converted to FP32, no unit normalization', fontsize=12)
    figure.colorbar(plot, ax=axis, fraction=.035, pad=.02, label='Raw actual activation difference')
    figure.subplots_adjust(left=.16, right=.91, top=.965, bottom=.04)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def render_propagation(path, values):
    import numpy as np
    import matplotlib.pyplot as plt
    relative = np.asarray([[np.nan if value is None else value for value in row]
                           for row in values['relative_difference_rms_percent']], dtype=float)
    assert relative.shape == (37, 16)
    figure, axis = plt.subplots(figsize=(11, 5.2))
    plot = axis.imshow(relative.T, aspect='auto', interpolation='nearest', cmap='viridis', vmin=0)
    axis.axvline(3.5, color='#D55E00', linewidth=1.5, linestyle='--')
    axis.set_xticks([0, 3, 4, 8, 12, 16, 20, 24, 28, 32, 36])
    axis.set_yticks(range(16), [str(index) for index in range(16)], fontsize=8)
    axis.set_xlabel('Actual decoder boundary (0 input;4 after L3 intervention;36 output)')
    axis.set_ylabel('Predicted action-token index')
    axis.set_title('Observed downstream difference from native PR after S128 replacement\nBoundary0–3 byte-exact; boundary4 begins a measured difference; subsequent native layers run freely', fontsize=11)
    figure.colorbar(plot, ax=axis, fraction=.035, pad=.02, label='100 × difference RMS / native PR RMS (%)')
    figure.subplots_adjust(left=.075, right=.92, top=.83, bottom=.18)
    landing = values['residual_landing']
    figure.text(.075, .07, f'Actual down ΔL2={landing["actual_down_delta_l2"]:.6g}; after residual boundary4 ΔL2={landing["actual_boundary4_delta_l2"]:.6g}; difference between these deltas ΔL2={landing["actual_difference_after_residual_minus_down_l2"]:.6g}.', fontsize=9)
    figure.text(.075, .025, 'This is intervention-relative hidden-state RMS, not an identity axis, growth mechanism or physical correction. BF16 residual landing can change the observed delta.', fontsize=9)
    figure.savefig(path, dpi=175)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Completed baseline with failed mlp-units and completed mlp-generic-inputs')
    baseline = parser.parse_args().output.resolve()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    assert not torch.cuda.is_initialized()
    root, values, sources, delta = audit(baseline, torch)
    assert not torch.cuda.is_initialized()
    import matplotlib
    matplotlib.use('Agg')
    destination = root / 'figures'
    destination.mkdir(exist_ok=False)
    main_path, heatmap_path = destination / 'milk-mlp-units.png', destination / 'milk-mlp-unit-values.png'
    propagation_path = destination / 'milk-mlp-unit-propagation.png'
    render_main(main_path, values)
    render_heatmap(heatmap_path, delta, values['selected128_units_zero_based'])
    render_propagation(propagation_path, values['propagation'])
    details = destination / 'unit-details.html'
    examples = values['first_three_rank_actual_action_token0_examples']
    rows = ''.join('<tr>' + ''.join(f'<td>{html.escape(str(value))}</td>' for value in
        (example['proxy_rank'], example['unit_zero_based'], context, sample['actual_postSiLU_gate_g'], sample['actual_up_u'], sample['actual_product_z'])) + '</tr>'
        for example in examples for context, sample in ((name, example[name]) for name in ('RR', 'PR')))
    details.write_text('<!doctype html><meta charset="utf-8"><title>Actual MLP unit evidence</title>'
        '<style>body{font:16px system-ui;margin:24px;max-width:1100px}img{max-width:100%}td,th{padding:8px;border-bottom:1px solid #ddd}summary{cursor:pointer;margin:20px 0}</style>'
        '<h1>Actual fixed-t21 layer4 unit evidence (zero-based L3)</h1><p>Interrupted run: 18 saved forwards plus 6 continuation. No identity or probability claim.</p>'
        '<img src="milk-mlp-units.png" alt="Four panels of actual raw velocity-axis controls">'
        '<details><summary>Open actual S128 activation heatmap (128 units × 16 action tokens)</summary><img src="milk-mlp-unit-values.png" alt="Actual RR minus PR activation differences"></details>'
        '<details><summary>Open observed downstream differences (37 decoder boundaries × 16 action tokens)</summary><img src="milk-mlp-unit-propagation.png" alt="Observed differences relative to native PR after S128 replacement"></details>'
        '<details><summary>Open first three ranked units, predicted action token 0</summary><p>All examples are denoising t21, not denoising step0. g/u/z are actual BF16 hook samples.</p>'
        '<table><thead><tr><th>Rank</th><th>Unit</th><th>Context</th><th>Actual g</th><th>Actual u</th><th>Actual z</th></tr></thead><tbody>' + rows + '</tbody></table></details>\n')
    manifest = dict(state='complete', plot_script_sha256=sha(Path(__file__)), sources=sources, actual_values=values,
        score_definition=SCORE, figure_count=3, main_panel_count=4, figures={path.name: sha(path) for path in (main_path, heatmap_path, propagation_path)},
        expandable_details=dict(path=str(details), sha256=sha(details)), actual_PT_to_NPZ_bytes_and_scores_independently_checked=True,
        entire_native_self_whole_joint_records_independently_checked=True, CUDA_initialized=False, model_calls=0,
        original_phase1_state='failed', original_surviving_model_forwards=18, continuation_model_forwards=6,
        combined_actual_model_forwards=24, original_post18_parameter_unchanged_verdict=None,
        parameter_hash_scope='Offline checked complete producer bytehash JSON equality only for the continuation process; its before matches the old before. The old post18 state was never measured.',
        phase1_file_hash_scope='First-observed continuation manifest, checked against surviving actual PT tensors and existing producer SHA relationships; no original completed phase1 manifest is asserted.',
        example_step_convention='Action token0 at denoising t21; actual GEN row is recorded per example.',
        interpretation='Two preregistered raw-velocity predictions and generic controls; no semantic identity, probability, physical correction or additive causal partition.')
    with (destination / 'plot.json').open('x') as stream:
        stream.write(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    assert not torch.cuda.is_initialized()
    print('[UNITS-PLOT] ' + json.dumps(dict(state='complete', figures=str(destination), predictions=values['predictions']['verdict'],
        combined_actual_model_forwards=24, original_phase1_state='failed', extra_direct_down_calls=values['original_extra_direct_down_calls'])), flush=True)


if __name__ == '__main__':
    main()
