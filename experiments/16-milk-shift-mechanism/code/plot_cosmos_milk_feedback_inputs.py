"""Plot the completed three-seed fixed-t21 action/future-vision interchange.

Read the actual saved 2x2 replay arrays and checked JSON. Recompute only the
four arithmetic effects and their fixed-t21 XYZ projections to verify the
saved scores. No Torch, model calls, video rendering or simulation. --output
is the experiment16/baseline root; figures must not already exist.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


SEEDS = (195, 196, 198)
CELLS = ('RR', 'PP', 'PR', 'RP')
EFFECTS = ('action_only', 'future_vision_only', 'interaction', 'total_pulse')
FORMULAS = dict(action_only='PR-RR', future_vision_only='RP-RR',
                interaction='PP-PR-RP+RR', total_pulse='PP-RR')
SCORE_DEFINITION = (
    'Within each seed, g = actual donor_t21_XYZ_velocity - actual recipient_t21_XYZ_velocity, '
    'over all sixteen action rows. Percent = 100*dot(effect_XYZ,g)/dot(g,g). '
    'All four effects in that seed use this same fixed-t21 reference; the reference differs across seeds. '
    'Zero means no projection on this axis; 100 means a full baseline D-R velocity difference along this axis. '
    'These percentages are not probabilities, identities, grasp results or final action scores.'
)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def file_source(path):
    return dict(path=str(path), sha256=sha(path))


def checked_data(baseline):
    root = baseline / 'feedback-inputs'
    assert not (root / 'failed.json').exists()
    files = {key: root / f'{key}.json' for key in ('complete', 'summary', 'provenance')}
    complete, summary, provenance = (read_json(files[key]) for key in files)
    assert complete['state'] == 'complete' and complete['seeds'] == list(SEEDS)
    assert (complete['q0_predictions'], complete['native_replays_exact'], complete['direct_model_forwards'],
            complete['boundaries_per_forward'], complete['components_per_forward']) == (6, 6, 12, 37, 72)
    assert complete['summary_sha256'] == sha(files['summary'])
    assert complete['script_sha256'] == provenance['script_sha256']
    for key in ('current_condition_all_seeds_exact', 'all_native_R_and_P_t21_replays_bit_exact',
                'entire_t21_model_output_including_vision_exact'):
        assert summary[key] is True
    assert provenance['seeds'] == list(SEEDS) and provenance['recipient'] == 'x15' and provenance['donor'] == 'x06'
    assert (provenance['pulse_layer_zero_based'], provenance['pulse_step'], provenance['fixed_replay_step'],
            provenance['pulse_component'], provenance['pulse_gain'], provenance['library_edits']) == (17, 20, 21, 'attention', 1, False)
    source_folder = Path(__file__).resolve().parent
    source_files = dict(probe=source_folder / 'probe_cosmos_milk_feedback_inputs.py',
        normal_runtime=source_folder / 'serve_cosmos_goal_pair.py', factory=source_folder / 'probe_cosmos_language_routes.py',
        components=source_folder / 'cosmos_component_interventions.py',
        pulse_contract=source_folder / 'probe_cosmos_milk_component_pulses.py',
        actual_transformer=source_folder.parent / 'network-source/cosmos3/transformer_cosmos3.py')
    assert sha(source_files['probe']) == complete['script_sha256']
    assert sha(source_files['pulse_contract']) == provenance['pulse_contract_sha256']
    assert sha(source_files['actual_transformer']) == provenance['actual_transformer_sha256']
    server_path = baseline / 'server/provenance.json'
    server = read_json(server_path)
    for key in ('normal_runtime', 'factory', 'components'):
        assert sha(source_files[key]) == provenance['source_sha256'][key] == server[key + '_sha256']
    assert provenance['scheduler_sha256'] == server['scheduler_sha256']
    assert provenance['baseline_summary_sha256'] == sha(baseline / 'summary.json')
    assert provenance['coarse_results_sha256'] == sha(baseline / 'component-pulses/results.json')
    assert [row['seed'] for row in summary['seeds']] == list(SEEDS)
    sources = dict(json={key: file_source(path) for key, path in files.items()},
        code={key: file_source(path) for key, path in source_files.items()},
        baseline_server_provenance=file_source(server_path),
        baseline_summary=file_source(baseline / 'summary.json'),
        coarse_results=file_source(baseline / 'component-pulses/results.json'), seeds=[])
    plotted = []
    for row in summary['seeds']:
        seed = row['seed']
        folder = root / f'seed{seed}'
        assert row == read_json(folder / 'summary.json')
        contract_path = folder / 'input-contract.json'
        contract = read_json(contract_path)
        assert contract['seed'] == seed
        for key in ('both_pure_noise_draws_exact', 'complete_schedule_and_30step_metadata_exact',
                    'native_R_all_record_fields_exact', 'other_t21_kwargs_all_exact'):
            assert contract[key] is True
        assert contract['seed198_P_all_coarse_record_fields_exact'] is (seed == 198)
        vision = contract['vision']
        assert vision['verified_squeezed_order'] == 'C,T,H,W'
        assert vision['current_condition_frames'] == [0] and vision['future_noisy_frames'] == [1, 2, 3, 4]
        assert vision['current_condition_bit_exact'] is vision['packed_current_condition_bit_exact'] is True
        assert vision['runtime_vision_shape'] == [1, 48, 5, 10, 20] and vision['runtime_patch_grid'] == [5, 5, 10]
        seed_sources = dict(seed=seed, summary=file_source(folder / 'summary.json'),
                            input_contract=file_source(contract_path), baseline={}, cells={})
        for role, scene in (('R', 'x15'), ('D', 'x06')):
            source = contract['sources'][role]
            assert source['trial'] == f'{scene}_seed{seed}'
            trial = baseline / 'closed-loop' / source['trial']
            assert source['image_sha256'] == sha(trial / 'input_00.png')
            assert source['states_sha256'] == sha(trial / 'chunk_00/states.pt')
            seed_sources['baseline'][role] = dict(image=file_source(trial / 'input_00.png'),
                states=file_source(trial / 'chunk_00/states.pt'), original_component_cache_sha256=source['components_sha256'])
        velocities = {}
        for cell in CELLS:
            destination = folder / 'replays' / cell
            metadata_path, arrays_path = destination / 'metadata.json', destination / 'arrays.npz'
            metadata = read_json(metadata_path)
            native = cell in ('RR', 'PP')
            assert metadata['label'] == f'seed{seed}_{cell}' and metadata['original_denoising_forward'] == 21
            assert metadata['original_q0_readout_and_velocity_exact'] is native
            assert metadata['original_q0_entire_model_output_exact'] is native
            assert metadata['component_cache_complete'] is True and metadata['all_observation_hooks_removed'] is True
            assert (metadata['actual_boundaries'], metadata['component_forward_count'], metadata['component_calls']) == (37, 1, 72)
            assert metadata['action_rows'] == list(range(250, 266))
            assert metadata['timestep_values'] == [row['timestep']] * 16
            assert metadata['arrays_sha256'] == sha(arrays_path)
            assert metadata['velocity']['shape'] == [16, 64] and metadata['readout']['shape'] == [16, 4096]
            with np.load(arrays_path, allow_pickle=False) as arrays:
                assert set(arrays.files) == {'action_boundary_hidden', 'readout', 'action_velocity'}
                assert arrays['action_boundary_hidden'].shape == (37, 16, 4096) and arrays['readout'].shape == (16, 4096)
                velocities[cell] = arrays['action_velocity'].astype(np.float64)
            assert velocities[cell].shape == (16, 64) and np.isfinite(velocities[cell]).all()
            seed_sources['cells'][cell] = dict(metadata=file_source(metadata_path), arrays=file_source(arrays_path),
                producer_actual_entire_output_exact=metadata['original_q0_entire_model_output_exact'],
                original_full_model_output_sha256=metadata['model_output_sha256'],
                original_component_cache_sha256=metadata['components_sha256'])
        v = velocities
        effects = dict(action_only=v['PR'] - v['RR'], future_vision_only=v['RP'] - v['RR'],
                       interaction=v['PP'] - v['PR'] - v['RP'] + v['RR'], total_pulse=v['PP'] - v['RR'])
        error = float(np.max(np.abs(effects['total_pulse'] - effects['action_only'] - effects['future_vision_only'] - effects['interaction'])))
        assert error < 1e-12 and row['decomposition_max_abs_error'] < 1e-12
        assert error == row['decomposition_max_abs_error']
        assert {key: row['effect_formulas'][key] for key in EFFECTS} == FORMULAS
        factorial_path = folder / 'factorial-arrays.npz'
        with np.load(factorial_path, allow_pickle=False) as arrays:
            for cell in CELLS:
                assert np.array_equal(arrays[cell + '_velocity'], v[cell])
            for key in EFFECTS:
                assert np.array_equal(arrays[key + '_effect'], effects[key])
            recipient = arrays['recipient_velocity'].astype(np.float64)
            donor = arrays['donor_velocity'].astype(np.float64)
        assert recipient.shape == donor.shape == (16, 3)
        assert np.isfinite(recipient).all() and np.isfinite(donor).all()
        assert np.array_equal(recipient, v['RR'][:, :3])
        gap = donor - recipient
        denominator = float(np.sum(gap * gap))
        assert denominator > 0 and math.isfinite(denominator)
        percentages = {}
        for key in EFFECTS:
            metric = row['effects'][key]['velocity_xyz']
            assert metric['gap_meaningful'] is True
            fraction = float(np.sum(effects[key][:, :3] * gap) / denominator)
            saved = metric['signed_projection_fraction']
            assert math.isfinite(saved) and math.isclose(fraction, saved, rel_tol=1e-12, abs_tol=1e-14)
            assert math.isclose(math.sqrt(denominator / 48), metric['donor_recipient_gap_rmse'], rel_tol=1e-12)
            assert math.isclose(float(np.linalg.norm(effects[key])), row['effects'][key]['raw_all64_delta_l2'], rel_tol=1e-12)
            percentages[key] = 100 * saved
        assert math.isclose(percentages['total_pulse'], sum(percentages[key] for key in EFFECTS[:3]), abs_tol=1e-12)
        seed_sources['factorial_arrays'] = file_source(factorial_path)
        sources['seeds'].append(seed_sources)
        plotted.append(dict(seed=seed, timestep=row['timestep'], sigma=row['sigma'],
            percent=percentages, decomposition_max_abs_error=error, denominator_sum_squares=denominator,
            recipient_t21_velocity_xyz=recipient.tolist(), donor_t21_velocity_xyz=donor.tolist(),
            effect_velocity_xyz={key: effects[key][:, :3].tolist() for key in EFFECTS}))
    assert len({(row['timestep'], row['sigma']) for row in plotted}) == 1
    return root, plotted, sources


def render(path, rows):
    figure, axes = plt.subplots(1, 3, figsize=(13.4, 6.4), sharey=True)
    labels = ('Action\nonly', 'Future vision\nonly', 'Interaction', 'Total')
    colors = ('#D55E00', '#0072B2', '#CC79A7', '#444444')
    values = [value for row in rows for value in row['percent'].values()]
    lower, upper = min(0, min(values)), max(0, max(values))
    span = max(upper - lower, 1)
    for axis, row in zip(axes, rows):
        scores = [row['percent'][key] for key in EFFECTS]
        axis.bar(np.arange(4), scores, color=colors, width=.67)
        axis.axhline(0, color='#555555', linewidth=1)
        axis.grid(axis='y', alpha=.23)
        axis.set_axisbelow(True)
        axis.set_xticks(np.arange(4), labels, fontsize=10)
        axis.set_title(f'Seed {row["seed"]}', fontsize=13)
        axis.set_ylim(lower - .17 * span, upper + .17 * span)
        for i, value in enumerate(scores):
            axis.text(i, value - .30 if value < 0 else value + .25, f'{value:+.3f}%',
                      ha='center', va='top' if value < 0 else 'bottom', fontsize=10)
    axes[0].set_ylabel('Signed projection on the fixed t21 D-R\nXYZ velocity difference (%)', fontsize=11)
    figure.suptitle('Input interchange: fixed t21 XYZ velocity effects across three seeds\nL18 attention donor pulse at t20; shared positive/negative scale, axis-only percentages', fontsize=14, y=.97)
    figure.subplots_adjust(left=.085, right=.98, top=.80, bottom=.27, wspace=.13)
    figure.text(.085, .18, f'All four bars within a seed share its actual t21 reference (timestep {rows[0]["timestep"]}, sigma {rows[0]["sigma"]:.6f}); references differ across seeds.', fontsize=9)
    figure.text(.085, .135, 'R = X+15 cm; P = R with the t20 donor pulse; D = X+6 cm. Cells name action first, future vision second.', fontsize=9)
    figure.text(.085, .09, 'Action: PR-RR; future vision: RP-RR; interaction: PP-PR-RP+RR; total: PP-RR. Actual 2x2 decomposition error < 1e-12.', fontsize=9)
    figure.text(.085, .045, 'Fixed t21 model forwards preserve the current visual condition; these velocity-axis effects are not probabilities or actual grasp outcomes.', fontsize=9)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Completed experiment16/baseline root containing feedback-inputs')
    args = parser.parse_args()
    root, rows, sources = checked_data(args.output.resolve())
    destination = root / 'figures'
    destination.mkdir(exist_ok=False)
    path = destination / 'fixed_t21_input_interchange.png'
    render(path, rows)
    manifest = dict(state='complete', plot_script_sha256=sha(Path(__file__)), sources=sources,
        figure_count=1, panel_count=3, effect_order=list(EFFECTS), effect_formulas=FORMULAS,
        score_definition=SCORE_DEFINITION, fixed_denoising_index=21, pulse=dict(layer_zero_based=17, step=20, component='attention'),
        actual_values=rows, actual_2x2_decomposition_checked=True, complete_summary_and_cells_sha_checked=True,
        producer_RR_PP_full_model_output_gate_checked=True,
        full_output_gate_scope='Actual entire model return, including vision, was compared bit-exact by the checked producer; the saved gates and original output hashes are preserved here. Plotting hashes the available actual replay NPZ arrays, without loading the remote full-output PT bundles.',
        factorial_arrays_hash_scope='Hash recorded at plotting time; producer did not seal this aggregate NPZ. All four cell velocities and plotted effects were checked against originally hashed cell NPZ arrays, and scores against the sealed summary.',
        figure=file_source(path), interpretation='Conditional action/future-vision latent effects on fixed-t21 raw Flow velocity. No solver update between cells, no final-action comparison or closed-loop outcome claim.')
    (destination / 'plot.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    print('[FEEDBACK-PLOT] ' + json.dumps(dict(state='complete', figure=str(path),
        values=[dict(seed=row['seed'], percent=row['percent']) for row in rows])), flush=True)


if __name__ == '__main__':
    main()
