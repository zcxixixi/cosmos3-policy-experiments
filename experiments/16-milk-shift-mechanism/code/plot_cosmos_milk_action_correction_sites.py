"""Render one checked, two-panel fixed-t21 component intervention figure.

Read only a complete action-correction-sites experiment. Verify the original
results SHA, per-case metadata and all recorded artifact SHAs, exact-control
reports and actual bf16 array hashes. Recompute the raw XYZ restoration from
saved velocities. No Torch, model, simulation, SSH or GPU is used.

Left: 36 layers x attention/MLP, one RR increment into fixed PR inputs.
Right: selected natural replacement vs three actual bf16 norm-matched random
directions. Reverse uses RR inputs and its own RR->PR reference; annotate it
separately. No positive site means no localization and no selected comparison.
Create only an absent BASE/action-correction-sites/figures directory.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


ROOT = Path('/home/current/work/cosmos3')
SEED, STEP = 198, 21
COMPONENTS = ('attention', 'mlp')
RANDOM_SEEDS = (21019801, 21019802, 21019803)
FORMULA = ('Restoration = dot(patchedPR - PR, RR - PR) / ||RR - PR||^2, '
           'on channels 0:3 of the actual 16x64 raw Flow action velocity; '
           'plotted percent = 100 * restoration. RR and PR have identical R vision, '
           'text, structure and sigma; only their actual action samples differ. '
           'This is instantaneous correction restoration, not grasp probability or milk selection. '
           'Reverse uses dot(patchedRR - RR, PR - RR) / ||PR - RR||^2 and is not a forward bar.')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def close(actual, expected, label):
    assert type(actual) in (float, int) and type(expected) in (float, int)
    assert math.isfinite(actual) and math.isfinite(expected)
    assert math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-10), (label, actual, expected)


def bf16_metadata(array, np):
    """Recover exact bf16 bytes from lossless float32 export without Torch."""
    assert sys.byteorder == 'little', 'Require the original host byte order for tensor byte hashes'
    value = np.ascontiguousarray(array)
    assert value.dtype == np.float32 and np.isfinite(value).all()
    bits = value.view(np.uint32)
    assert not np.any(bits & 0xFFFF), 'Export is not a lossless bf16 -> float32 array'
    payload = (bits >> 16).astype('<u2').tobytes()
    return dict(shape=list(value.shape), dtype='torch.bfloat16', sha256=hashlib.sha256(payload).hexdigest())


def load_arrays(directory, metadata, np, with_boundary_hash=True):
    with np.load(directory / 'arrays.npz', allow_pickle=False) as source:
        arrays = {key: source[key].copy() for key in ('action_boundary_hidden', 'readout', 'action_velocity')}
    for key, shape, meta_key in (('action_boundary_hidden', (37, 16, 4096), 'boundary'),
                                ('readout', (16, 4096), 'readout'), ('action_velocity', (16, 64), 'velocity')):
        assert arrays[key].shape == shape, (directory, key)
        actual_meta = bf16_metadata(arrays[key], np)
        if key != 'action_boundary_hidden' or with_boundary_hash:
            assert actual_meta == metadata[meta_key], (directory, key, 'actual bf16 byte hash')
    return arrays


def verify_run(directory, row, np, control=False):
    metadata_path = directory / 'metadata.json'
    metadata_hash = sha(metadata_path)
    if not control:
        assert row['metadata_sha256'] == metadata_hash, f'{row["run"]}/historical metadata SHA'
        assert Path(row['artifacts']).resolve() == directory.resolve(), 'Unexpected artifact path'
    metadata = read_json(metadata_path)
    assert metadata['label'] == row['run'] and metadata['seed'] == SEED
    assert metadata['original_denoising_forward'] == STEP
    assert metadata['actual_model_calls'] == 1 and metadata['actual_boundaries'] == 37
    assert metadata['component_cache_complete'] is True and metadata['component_calls'] == 72
    assert metadata['full_actual_model_kwargs_and_sigma_unchanged'] is True
    assert metadata['all_owned_hooks_removed'] is True
    rows = metadata['action_rows']
    assert len(rows) == len(set(rows)) == 16 and all(type(value) is int and value >= 0 for value in rows)
    expected_files = {'components.pt', 'model_output.pt', 'actual_head_boundaries.pt', 'arrays.npz'}
    assert set(metadata['files_sha256']) == expected_files
    for name, digest in metadata['files_sha256'].items():
        assert digest == sha(directory / name), f'{row["run"]}/{name}: recorded artifact SHA'
    native = control and row['run'] in ('native_RR', 'native_PR')
    assert metadata['target_intervention_count'] == (0 if native else 1)
    if control:
        assert row['full_return_head_boundaries_all_components_bit_exact'] is True
        assert metadata['full_native_return_head_boundaries_components_bit_exact'] is True
    if not native:
        assert metadata['component_event_sha256'] == sha(directory / 'component_event.pt')
        event = metadata['component_event']
        for key in ('nonaction_rows_bit_exact', 'before_equals_recipient_cache', 'after_equals_requested_source'):
            assert event[key] is True, (row['run'], key)
        assert event['local_forward'] == 0 and event['original_denoising_forward'] == STEP
        assert event['component'] in COMPONENTS and 0 <= event['layer_zero_based'] < 36
        assert event['nonaction_rows_count'] > 0
        for key, dtype in (('before', 'torch.bfloat16'), ('after', 'torch.bfloat16'), ('actual_delta', 'torch.float64')):
            assert event[key]['shape'] == [16, 4096] and event[key]['dtype'] == dtype
            assert len(event[key]['sha256']) == 64
        if control:
            assert event['exact_no_change'] is True and event['actual_delta_l2'] == 0
            assert event['before'] == event['after']
            assert event['component'] == row['component'] and event['layer_zero_based'] == row['layer_zero_based']
        else:
            assert event['component'] == row['component'] and event['layer_zero_based'] == row['layer_zero_based']
            close(event['actual_delta_l2'], row['actual_increment_delta_l2'], row['run'] + '/actual delta norm')
    arrays = load_arrays(directory, metadata, np)
    evidence = dict(run=row['run'], metadata_sha256=metadata_hash,
        metadata_hash_historically_linked_to_results=not control,
        files_sha256=metadata['files_sha256'], actual_bf16_head_and_boundary_hashes_checked=True,
        event_sha256=metadata.get('component_event_sha256'))
    return metadata, arrays, evidence


def restoration(source, endpoint, altered, recorded, np, label):
    r, d, p = (value[:, :3].astype(np.float64).reshape(-1) for value in (source, endpoint, altered))
    gap, delta = d - r, p - r
    gap_rms = float(np.sqrt(np.mean(gap * gap)))
    floor = max(1e-6, 1e-4 * max(float(np.sqrt(np.mean(r * r))), float(np.sqrt(np.mean(d * d)))))
    assert gap_rms > floor and recorded['gap_meaningful'] is True, 'Invalid XYZ gap: refuse score'
    close(gap_rms, recorded['donor_recipient_gap_rmse'], label + '/gap RMS')
    close(floor, recorded['gap_floor'], label + '/gap floor')
    fraction = float(np.dot(delta, gap) / np.dot(gap, gap))
    close(fraction, recorded['signed_projection_fraction'], label + '/restoration')
    close(float(np.sqrt(np.mean(delta * delta))), recorded['rmse_to_recipient'], label + '/raw delta RMS')
    return fraction


def checked_data(baseline, np):
    root = baseline / 'action-correction-sites'
    assert not (root / 'failed.json').exists()
    complete, results, provenance = (read_json(root / (name + '.json')) for name in ('complete', 'results', 'provenance'))
    assert complete['state'] == results['state'] == 'complete'
    assert complete['results_sha256'] == sha(root / 'results.json'), 'Require original complete -> results SHA'
    assert complete['seed'] == results['seed'] == SEED
    assert complete['original_denoising_forward'] == results['original_denoising_forward'] == STEP
    assert complete['q0_predictions'] == 0 and complete['scan_forwards'] == results['scan_forwards'] == 72
    assert complete['exact_control_forwards'] == results['control_forwards'] == 4
    for key in ('full_return_head_boundaries_component_controls_bit_exact',
                'all_actual_inputs_full_sigma_unchanged', 'all_target_before_after_and_nonaction_rows_exact'):
        assert complete[key] is True
    probe = ROOT / 'work/probe_cosmos_milk_action_correction_sites.py'
    assert complete['script_sha256'] == provenance['script_sha256'] == sha(probe)
    assert provenance['seed'] == SEED and provenance['original_denoising_forward'] == STEP
    assert provenance['library_edits'] is False and provenance['q0_predictions'] == 0
    assert provenance['random_seeds'] == list(RANDOM_SEEDS)
    assert provenance['random_actual_norm_relative_tolerance'] <= 1e-3
    contract = read_json(root / 'input-contract.json')
    assert results['input_contract'] == contract and contract['seed'] == SEED
    for key in ('actual_RR_vs_PR_only_action_tokens_differ', 'vision_text_structure_full_sigma_exact',
                'both_actual_pure_noise_draws_exact', 'all_30_schedule_pack_metadata_exact',
                'native_R_full_saved_record_exact', 'RR_full_return_matches_native_t21'):
        assert contract[key] is True
    controls = results['controls']
    assert len(controls) == 4 and {row['run'] for row in controls} == {
        'native_RR', 'native_PR', 'PR_self_attention', 'PR_self_mlp'}
    evidence, native_arrays = [], {}
    for row in controls:
        metadata, arrays, record = verify_run(root / 'controls' / row['run'], row, np, control=True)
        assert metadata['action_rows'] == contract['action_rows']
        evidence.append(record)
        native_arrays[row['run']] = arrays
    # Independently check all saved head/boundary arrays against the original
    # RR/PR replay files. Full transformer/PT equality remains the source run's
    # explicit exact report; this CPU plotter does not unpickle tensors.
    reference_evidence = {}
    for label in ('RR', 'PR'):
        folder = baseline / 'feedback-inputs' / f'seed{SEED}' / 'replays' / label
        metadata_path = folder / 'metadata.json'
        assert provenance['reference_metadata_sha256'][label] == sha(metadata_path)
        metadata = read_json(metadata_path)
        for key, name in (('arrays_sha256', 'arrays.npz'), ('components_sha256', 'components.pt'),
                          ('model_output_sha256', 'model_output.pt')):
            assert metadata[key] == sha(folder / name)
        expected = load_arrays(folder, metadata, np, with_boundary_hash=False)
        for key in expected:
            assert expected[key].tobytes() == native_arrays['native_' + label][key].tobytes(), (label, key, 'native arrays exact')
        reference_evidence[label] = dict(metadata_sha256=sha(metadata_path), arrays_sha256=metadata['arrays_sha256'],
            components_sha256=metadata['components_sha256'], model_output_sha256=metadata['model_output_sha256'])
    for label in ('PR_self_attention', 'PR_self_mlp'):
        for key in native_arrays[label]:
            assert native_arrays[label][key].tobytes() == native_arrays['native_PR'][key].tobytes(), (label, key, 'self arrays exact')
    rr, pr = native_arrays['native_RR']['action_velocity'], native_arrays['native_PR']['action_velocity']
    cases = results['cases']
    assert len(cases) == 72
    assert {(row['layer_zero_based'], row['component']) for row in cases} == {
        (layer, component) for layer in range(36) for component in COMPONENTS}
    values, velocities = {}, {}
    for row in cases:
        layer, component = row['layer_zero_based'], row['component']
        assert row['run'] == f'layer{layer:02d}_{component}'
        assert row['input_context'] == 'PR' and row['intervention'] == 'RR_increment_into_PR'
        metadata, arrays, record = verify_run(root / 'runs' / row['run'], row, np)
        assert metadata['recipient_context'] == f'seed{SEED}_PR' and metadata['action_rows'] == contract['action_rows']
        fraction = restoration(pr, rr, arrays['action_velocity'], row['restoration_xyz'], np, row['run'])
        values[(layer, component)] = fraction
        velocities[row['run']] = arrays['action_velocity'][:, :3].tolist()
        evidence.append(record)
    ranked = sorted(cases, key=lambda row: values[(row['layer_zero_based'], row['component'])], reverse=True)
    assert results['restoration_ranking'] == ranked
    positive = [row for row in ranked if values[(row['layer_zero_based'], row['component'])] > 0 and row['actual_increment_delta_l2'] > 0]
    selected = positive[0] if positive else None
    assert results['selected_site'] == selected and complete['selected_positive_site'] is (selected is not None)
    random, reverse = [], None
    followups = results['followups']
    if selected is None:
        assert not followups and results['followup_forwards'] == 0
        assert complete['single_model_forwards'] == 76
        assert complete['reverse_control_forwards'] == complete['random_control_forwards'] == 0
    else:
        assert len(followups) == results['followup_forwards'] == 4 and complete['single_model_forwards'] == 80
        assert complete['reverse_control_forwards'] == 1 and complete['random_control_forwards'] == 3
        layer, component = selected['layer_zero_based'], selected['component']
        random = sorted([row for row in followups if row['intervention'] == 'actual_bf16_norm_matched_random'],
                        key=lambda row: row['random_control']['cpu_generator_seed'])
        reversed_rows = [row for row in followups if row['intervention'] == 'PR_increment_into_RR']
        assert len(random) == 3 and len(reversed_rows) == 1
        assert [row['random_control']['cpu_generator_seed'] for row in random] == list(RANDOM_SEEDS)
        reverse = reversed_rows[0]
        assert reverse['run'] == f'reverse_RR_layer{layer:02d}_{component}' and reverse['input_context'] == 'RR'
        for row in followups:
            assert row['layer_zero_based'] == layer and row['component'] == component
            directory = root / 'followups' / row['run']
            metadata, arrays, record = verify_run(directory, row, np)
            assert metadata['action_rows'] == contract['action_rows']
            backwards = row is reverse
            assert metadata['recipient_context'] == f'seed{SEED}_' + ('RR' if backwards else 'PR')
            fraction = restoration(rr if backwards else pr, pr if backwards else rr,
                                   arrays['action_velocity'], row['restoration_xyz'], np, row['run'])
            velocities[row['run']] = arrays['action_velocity'][:, :3].tolist()
            row['recomputed_restoration_fraction'] = fraction
            if not backwards:
                assert row['input_context'] == 'PR'
                info = row['random_control']
                assert info == read_json(directory / 'random_control.json')
                assert row['random_control_sha256'] == sha(directory / 'random_control.pt')
                close(info['actual_natural_delta_l2'], selected['actual_increment_delta_l2'], row['run'] + '/natural norm')
                close(info['actual_random_delta_l2'], row['actual_increment_delta_l2'], row['run'] + '/actual norm')
                error = abs(info['actual_random_delta_l2'] - selected['actual_increment_delta_l2']) / selected['actual_increment_delta_l2']
                close(error, info['relative_norm_error'], row['run'] + '/rounded norm error')
                assert error <= info['relative_norm_tolerance'] <= 1e-3
                record['random_control_sha256'] = row['random_control_sha256']
            evidence.append(record)
    assert len(results['completed']) == complete['single_model_forwards']
    assert len(set(results['completed'])) == len(results['completed'])
    assert set(results['completed']) == {row['run'] for row in controls + cases + followups}
    sources = dict(complete_sha256=sha(root / 'complete.json'), original_results_sha256=complete['results_sha256'],
        provenance_sha256=sha(root / 'provenance.json'), input_contract_sha256=sha(root / 'input-contract.json'),
        original_probe_script_sha256=complete['script_sha256'], reference_replays=reference_evidence, runs=evidence,
        control_metadata_hash_note='The original results did not store control metadata SHAs. These four hashes are first observed by this plotter; exact control reports, recorded artifact SHAs and original head/boundary arrays were checked, without asserting a historical seal for those metadata files.')
    return root, results, values, selected, random, reverse, rr, pr, velocities, sources


def render(path, values, selected, random, reverse, np):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    matrix = np.asarray([[100 * values[(layer, component)] for component in COMPONENTS] for layer in range(36)])
    limit = max(1., float(np.max(np.abs(matrix))))
    figure, axes = plt.subplots(1, 2, figsize=(12.6, 10.8), gridspec_kw={'width_ratios': [1., 1.3]})
    try:
        image = axes[0].imshow(matrix, cmap='RdBu_r', vmin=-limit, vmax=limit, aspect='auto')
        axes[0].set_xticks([0, 1], ['Attention', 'MLP'])
        axes[0].set_yticks(np.arange(36), [str(layer) for layer in range(36)], fontsize=8)
        axes[0].set_ylabel('Decoder layer (zero based)')
        axes[0].set_title('A  One RR increment into fixed PR inputs', loc='left', fontsize=11)
        for layer in range(36):
            for index in range(2):
                value = matrix[layer, index]
                axes[0].text(index, layer, f'{value:.2f}', ha='center', va='center', fontsize=7,
                             color='white' if abs(value) > .55 * limit else '#202020')
        colorbar = figure.colorbar(image, ax=axes[0], fraction=.045, pad=.035)
        colorbar.set_label('Restoration of RR - PR raw XYZ velocity gap (%)', fontsize=9)
        axes[1].set_title('B  Natural vs. actual norm-matched random', loc='left', fontsize=11)
        if selected is None:
            axes[1].axis('off')
            axes[1].text(.5, .6, 'No positive restoration site.\nNo local site selected.\nReverse/random follow-ups were not run.',
                         ha='center', va='center', transform=axes[1].transAxes, fontsize=12)
            comparison = []
            reverse_note = 'No positive restoration site; the heatmap still shows all 72 tested interventions.'
        else:
            layer, component = selected['layer_zero_based'], selected['component']
            axes[0].add_patch(Rectangle((COMPONENTS.index(component) - .5, layer - .5), 1, 1,
                                       fill=False, edgecolor='#101010', linewidth=2))
            natural = 100 * values[(layer, component)]
            heights = [natural] + [100 * row['recomputed_restoration_fraction'] for row in random]
            labels = [f'Natural RR\nL{layer} {component.upper()}'] + [f'Random {index + 1}' for index in range(3)]
            bars = axes[1].bar(np.arange(4), heights, color=['#0072B2', '#999999', '#999999', '#999999'], width=.68)
            axes[1].axhline(0, color='#333333', linewidth=.8)
            axes[1].set_xticks(np.arange(4), labels, fontsize=9)
            axes[1].set_ylabel('Restoration of the same RR - PR XYZ gap (%)')
            axes[1].spines[['top', 'right']].set_visible(False)
            axes[1].grid(axis='y', alpha=.2)
            axes[1].set_axisbelow(True)
            spread = max(1., max(heights) - min(heights))
            axes[1].set_ylim(min(0., min(heights)) - .08 * spread, max(0., max(heights)) + .18 * spread)
            for bar, value in zip(bars, heights):
                axes[1].annotate(f'{value:.3f}%', (bar.get_x() + bar.get_width() / 2, value),
                                 xytext=(0, 5 if value >= 0 else -5), textcoords='offset points',
                                 ha='center', va='bottom' if value >= 0 else 'top', fontsize=10)
            norm = selected['actual_increment_delta_l2']
            errors = [100 * row['random_control']['relative_norm_error'] for row in random]
            axes[1].text(.03, .98, f'Natural actual BF16 increment delta L2: {norm:.5g}\n'
                         + 'Random actual norm errors: ' + ', '.join(f'{value:.4f}%' for value in errors),
                         transform=axes[1].transAxes, ha='left', va='top', fontsize=9)
            comparison = [dict(run=selected['run'], kind='natural', restoration_percent=natural,
                               actual_increment_delta_l2=norm)] + [dict(run=row['run'], kind='random',
                restoration_percent=100 * row['recomputed_restoration_fraction'],
                actual_increment_delta_l2=row['actual_increment_delta_l2'], random_control=row['random_control']) for row in random]
            reverse_note = (f'Reverse RR <- PR at L{layer} {component.upper()}: '
                            f'{100 * reverse["recomputed_restoration_fraction"]:.3f}% toward PR. '
                            'Different RR -> PR reference; excluded from the forward bars.')
        figure.suptitle('Fixed t21 causal increment replacement | seed 198', fontsize=15, y=.97)
        figure.text(.075, .125, 'Restoration = dot(patchedPR - PR, RR - PR) / ||RR - PR||^2, on 16 raw XYZ velocity rows.', fontsize=10)
        figure.text(.075, .097, 'RR/PR differ only in the action input; vision, text, structure and sigma remain fixed. No scheduler update.', fontsize=9)
        figure.text(.075, .069, reverse_note, fontsize=9)
        figure.text(.075, .041, 'Single seed and diffusion step; selected using these same interventions. Percent is not milk selection or grasp success.', fontsize=9)
        figure.subplots_adjust(left=.075, right=.98, top=.91, bottom=.21, wspace=.42)
        figure.savefig(path, dpi=220, bbox_inches='tight', facecolor='white')
        return matrix.tolist(), comparison
    finally:
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Completed experiment/baseline root')
    baseline = parser.parse_args().output.resolve()
    import numpy as np
    root, results, values, selected, random, reverse, rr, pr, velocities, sources = checked_data(baseline, np)
    out = root / 'figures'
    out.mkdir(exist_ok=False)
    image_path = out / 'action-correction-sites.png'
    matrix, comparison = render(image_path, values, selected, random, reverse, np)
    payload = dict(state='complete', seed=SEED, original_denoising_forward=STEP, formula=FORMULA,
        plotted_values=dict(layers_zero_based=list(range(36)), components=list(COMPONENTS),
                           restoration_percent_matrix=matrix, selected_comparison=comparison,
                           reverse_separate_reference=None if reverse is None else dict(run=reverse['run'],
                               restoration_percent=100 * reverse['recomputed_restoration_fraction'],
                               axis='RR -> PR raw XYZ velocity')),
        selected_site=None if selected is None else dict(layer_zero_based=selected['layer_zero_based'],
            component=selected['component'], restoration_fraction=values[(selected['layer_zero_based'], selected['component'])]),
        raw_metric=dict(RR_velocity_xyz=rr[:, :3].tolist(), PR_velocity_xyz=pr[:, :3].tolist(),
                        intervention_velocity_xyz=velocities, gap_abs_floor=1e-6, gap_relative_floor=1e-4,
                        sigma=results['input_contract']['sigma'], timestep=results['input_contract']['timestep']),
        source_hashes=sources, plot_script_sha256=sha(Path(__file__)), image_sha256=sha(image_path),
        verification=dict(original_results_SHA_verified=True, all_four_full_return_boundary_component_exact_reports=True,
                          all_case_artifact_and_event_SHA_verified=True, actual_bf16_head_boundary_array_hashes_verified=True,
                          all_restoration_metrics_recomputed_from_raw_arrays=True, no_partial_scan_table_used=True),
        limitations=results['limitations'], scope=results['scope'])
    (out / 'plot.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(state='complete', image=str(image_path), plot_json=str(out / 'plot.json'),
                          selected_site=payload['selected_site'], image_sha256=payload['image_sha256']), allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
