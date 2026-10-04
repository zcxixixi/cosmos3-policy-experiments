"""Plot two figures from a completed 56-pulse milk q0 scan's saved JSON scores.

--output is the baseline root. Require the scan's completion/hash/count and
four exact controls before creating its absent component-pulses/figures child.
No Torch, model, MuJoCo or raw-video processing; no network-array reanalysis.
Projection percentages are positions on one continuous donor-minus-recipient
XYZ axis. They are NOT success rates, probabilities, object identity scores,
or evidence of an attractor. Later state changes may be solver inheritance.
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
from matplotlib.colors import TwoSlopeNorm


LAYERS = (5, 11, 17, 23, 29, 31, 35)
COMPONENTS = ('attention', 'mlp')
PULSE_STEPS = (0, 10, 20, 29)
CONTROL_RUNS = {
    'recipient_observer_free_repeat', 'donor_observer_free_repeat',
    'self_layer17_attention_step00', 'self_layer17_mlp_step00',
}
SCORE_DEFINITION = (
    'fraction = sum((P-R)*(D-R))/sum((D-R)^2), using saved continuous normalized '
    'XYZ values over the sixteen action rows; plotted percent = 100*fraction. '
    'R is X+15cm recipient baseline, D is paired X+6cm donor baseline. '
    '0 is recipient on this axis; 100 is donor on the same axis. '
    'Orthogonal differences are not represented. Negative and >100 values are '
    'valid axis projections, not probabilities or success rates. Small reference '
    'gaps were marked null by the scan; null cells/curve samples remain missing.'
)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def projection(metric, label):
    score = metric['signed_projection_fraction']
    assert type(metric['gap_meaningful']) is bool, label
    if score is None:
        assert metric['gap_meaningful'] is False, f'{label}: unexplained missing projection'
        return np.nan
    assert metric['gap_meaningful'] is True, f'{label}: projection despite small gap'
    assert type(score) in (int, float) and math.isfinite(score), label
    return float(score)


def load_checked_scores(baseline):
    scan = baseline / 'component-pulses'
    assert not (scan / 'failed.json').exists(), 'Failed scan must be reviewed'
    complete_path, results_path, validation_path = (scan / name for name in ('complete.json', 'results.json', 'validation.json'))
    complete = read_json(complete_path)
    assert complete['state'] == 'complete'
    assert complete['scan_predictions'] == 56 and complete['control_predictions'] == 4
    for key in ('all_noise_schedule_pack_checks_exact', 'all_repeat_and_selfpatch_fields_bit_exact',
                'single_component_pulse_per_scan_prediction'):
        assert complete[key] is True, key
    assert complete['results_sha256'] == sha(results_path), 'Completed results SHA mismatch'
    assert complete['validation_sha256'] == sha(validation_path), 'Completed validation SHA mismatch'
    results, validation = read_json(results_path), read_json(validation_path)
    assert results['state'] == 'complete'
    assert results['scan_predictions'] == 56 and results['control_predictions'] == 4
    assert results['paired_seed'] == complete['paired_seed']
    assert validation['state'] == 'all_exact_controls_passed'
    controls = validation['controls']
    assert len(controls) == 4 and {row['run'] for row in controls} == CONTROL_RUNS
    assert all(row['all_normal_runtime_fields_bit_exact'] is True for row in controls)
    assert results['controls'] == controls, 'Results/control ledger mismatch'
    for key in ('complete_nine_baseline_summaries', 'both_fp32_random_draws_exact',
                'action_noise_and_all_30_pack_metadata_exact', 'timesteps_and_sigmas_exact',
                'caches_complete_and_len_steps_30', 'first_kwargs_match_runtime'):
        assert validation[key] is True, key
    for row in controls:
        destination = scan / 'controls' / row['run']
        assert row['states_sha256'] == sha(destination / 'states.pt'), f'{row["run"]}: control artifact SHA'
        if 'patch_report_sha256' in row:
            assert row['patch_report_sha256'] == sha(destination / 'patch_report.pt'), f'{row["run"]}: self-patch artifact SHA'
    provenance_path = scan / 'provenance.json'
    provenance = read_json(provenance_path)
    assert provenance['script_sha256'] == complete['script_sha256']
    assert provenance['layers'] == list(LAYERS) and provenance['components'] == list(COMPONENTS)
    assert provenance['pulse_steps'] == list(PULSE_STEPS) and provenance['gain'] == 1
    assert provenance['library_edits'] is False and provenance['closed_loop_execution'] is False
    assert results['ranking_metric'] == 'final_normalized_action_xyz.signed_projection_fraction'
    cases = results['cases']
    expected = {(layer, component, step) for layer in LAYERS for component in COMPONENTS for step in PULSE_STEPS}
    assert len(cases) == 56 and {(row['layer'], row['component'], row['pulse_step']) for row in cases} == expected
    by_run = {}
    matrices = {component: np.full((len(LAYERS), len(PULSE_STEPS)), np.nan) for component in COMPONENTS}
    for row in cases:
        layer, component, step = row['layer'], row['component'], row['pulse_step']
        name = f'layer{layer:02d}_{component}_step{step:02d}'
        assert row['run'] == name and name not in by_run
        assert row['gain'] == 1 and row['intervention_count'] == 1
        assert row['recipient_trial'] == provenance['recipient_trial'] and row['donor_trial'] == provenance['donor_trial']
        score = projection(row['final_normalized_action_xyz'], name)
        matrices[component][LAYERS.index(layer), PULSE_STEPS.index(step)] = score * 100
        by_run[name] = row
        destination = scan / 'runs' / name
        for field, filename in (('states_sha256', 'states.pt'), ('patch_report_sha256', 'patch_report.pt'),
                                ('runtime_metadata_sha256', 'metadata.json')):
            assert row[field] == sha(destination / filename), f'{name}: {filename} SHA mismatch'
    # Verify the stored ranking of saved scalar scores, without recomputing any
    # latent/action metric or opening the network tensors.
    ranked = sorted((row for row in cases if row['final_normalized_action_xyz']['signed_projection_fraction'] is not None),
                    key=lambda row: row['final_normalized_action_xyz']['signed_projection_fraction'], reverse=True)
    assert results['ranked_runs'] == [row['run'] for row in ranked]
    assert results['top_three_candidates'] == ranked[:3]
    assert set(results['unrankable_runs']) == {row['run'] for row in cases if row not in ranked}
    top_path = scan / 'top_candidates.json'
    top_file = read_json(top_path)
    assert top_file['state'] == 'candidates_for_followup_only'
    assert top_file['candidates'] == results['top_three_candidates']
    sources = dict(complete=dict(path=str(complete_path), sha256=sha(complete_path)),
                   results=dict(path=str(results_path), sha256=complete['results_sha256']),
                   validation=dict(path=str(validation_path), sha256=complete['validation_sha256']),
                   provenance=dict(path=str(provenance_path), sha256=sha(provenance_path)),
                   top_candidates=dict(path=str(top_path), sha256=sha(top_path)),
                   scan_script_sha256=complete['script_sha256'], metrics=[])
    curves = []
    for rank, row in enumerate(results['top_three_candidates'], start=1):
        metrics_path = scan / 'runs' / row['run'] / 'metrics.json'
        metrics = read_json(metrics_path)
        assert metrics['final_normalized_action_xyz'] == row['final_normalized_action_xyz']
        steps = metrics['per_denoising_step']
        assert len(steps) == 30 and [value['step'] for value in steps] == list(range(30))
        samples = [projection(value['action_state_after_step_xyz'], f'{row["run"]}/state{value["step"]}') * 100
                   for value in steps]
        assert steps[row['pulse_step']]['timestep'] == row['pulse_timestep']
        assert steps[row['pulse_step']]['sigma'] == row['pulse_sigma']
        curves.append(dict(rank=rank, run=row['run'], layer=row['layer'], component=row['component'],
                           pulse_step=row['pulse_step'], percent=np.asarray(samples, dtype=float),
                           final_output_axis_percent=row['final_normalized_action_xyz']['signed_projection_fraction'] * 100))
        sources['metrics'].append(dict(run=row['run'], path=str(metrics_path), sha256=sha(metrics_path)))
    return scan, results, matrices, curves, sources


def heatmap(path, matrices, seed):
    available = np.concatenate([value[np.isfinite(value)] for value in matrices.values()])
    bound = max(1.0, float(np.max(np.abs(available)))) if available.size else 1.0
    norm = TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('#dddddd')
    figure, axes = plt.subplots(1, 2, figsize=(12.3, 6.2), sharey=True)
    for axis, component in zip(axes, COMPONENTS):
        values = matrices[component]
        image = axis.imshow(np.ma.masked_invalid(values), cmap=cmap, norm=norm, aspect='auto')
        axis.set_title('Attention increment' if component == 'attention' else 'MLP increment', fontsize=13)
        axis.set_xticks(np.arange(len(PULSE_STEPS)), [str(step) for step in PULSE_STEPS])
        axis.set_yticks(np.arange(len(LAYERS)), [str(layer) for layer in LAYERS])
        axis.set_xlabel('Single pulse at denoising forward (zero based)')
        for i in range(len(LAYERS)):
            for j in range(len(PULSE_STEPS)):
                value = values[i, j]
                label = 'gap too small' if not np.isfinite(value) else f'{value:+.1f}%'
                color = 'white' if np.isfinite(value) and abs(value) > bound * 0.58 else '#222222'
                axis.text(j, i, label, ha='center', va='center', fontsize=8 if not np.isfinite(value) else 10, color=color)
    axes[0].set_ylabel('Decoder layer (zero based)')
    figure.suptitle(f'Q0 local pulses: final XYZ axis projection (paired seed {seed})\nAxis only - not success or probability', fontsize=15, y=.97)
    figure.subplots_adjust(left=.07, right=.84, top=.80, bottom=.23, wspace=.16)
    color_axis = figure.add_axes([.88, .25, .025, .54])
    colorbar = figure.colorbar(image, cax=color_axis)
    colorbar.set_label('Signed XYZ axis projection (%)')
    figure.text(.07, .12, '0% = X+15 cm cheese-grasp baseline; 100% = X+6 cm milk-grasp baseline along the same XYZ axis.', fontsize=10)
    figure.text(.07, .08, 'These grasp labels come from baseline closed loops; the plotted values describe only q0 normalized action outputs.', fontsize=9)
    figure.text(.07, .04, 'Negative or >100% projections are valid. Gray cells have too little baseline gap. Neither color nor percentage is a success rate.', fontsize=9)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return dict(shared_colorbar=True, symmetric_color_limits_percent=[-bound, bound], missing_color='#dddddd')


def state_curves(path, curves, seed):
    figure, axis = plt.subplots(figsize=(12.3, 6.2))
    colors = ('#0072B2', '#D55E00', '#009E73')
    markers = ('o', 's', '^')
    x = np.arange(30)
    for curve, color, marker in zip(curves, colors, markers):
        label = f'#{curve["rank"]} L{curve["layer"]} {curve["component"]}, pulse s{curve["pulse_step"]}'
        axis.plot(x, curve['percent'], color=color, linewidth=2, marker=marker, markersize=3, label=label)
        step = curve['pulse_step']
        axis.axvline(step, color=color, linestyle=':', linewidth=1.4, alpha=.8)
        if np.isfinite(curve['percent'][step]):
            axis.scatter([step], [curve['percent'][step]], color=color, marker=marker, s=75, edgecolors='black', linewidths=.6, zorder=4)
    axis.axhline(0, color='#555555', linewidth=1, linestyle='--')
    axis.axhline(100, color='#777777', linewidth=1, linestyle='--')
    axis.set_xlim(-.4, 29.4)
    axis.set_xticks((0, 5, 10, 15, 20, 25, 29))
    axis.set_xlabel('Denoising forward just completed (zero based)')
    axis.set_ylabel('Action-state XYZ axis projection (%)\nAxis only; not success or probability')
    axis.grid(alpha=.22)
    if curves:
        axis.legend(loc='best', fontsize=10)
    else:
        axis.text(.5, .5, 'No rankable q0 candidate: final XYZ baseline gap was too small.',
                  transform=axis.transAxes, ha='center', va='center')
    figure.suptitle(f'Top q0 candidates: action state after each denoising step (seed {seed})\nSingle pulse marked; later forwards run natively', fontsize=14, y=.97)
    figure.subplots_adjust(left=.10, right=.98, top=.81, bottom=.24)
    figure.text(.10, .13, 'Each curve uses that step\'s own paired baseline XYZ axis; a small baseline gap appears as a missing sample.', fontsize=10)
    figure.text(.10, .09, 'Only one layer/component at one forward was replaced. After its marked pulse, subsequent forwards run without this intervention.', fontsize=9)
    figure.text(.10, .05, 'A continuing difference may be inherited by the multistep solver. This figure does not establish an attractor or a repaired grasp.', fontsize=9)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def serializable_scores(array):
    return [[float(value) if np.isfinite(value) else None for value in row] for row in array]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Completed nine-trial baseline root')
    args = parser.parse_args()
    baseline = args.output.resolve()
    scan, results, matrices, curves, sources = load_checked_scores(baseline)
    destination = scan / 'figures'
    destination.mkdir(exist_ok=False)
    heatmap_path, curves_path = destination / 'signed_xyz_pulses.png', destination / 'post_pulse_action_state_axis.png'
    style = heatmap(heatmap_path, matrices, results['paired_seed'])
    state_curves(curves_path, curves, results['paired_seed'])
    manifest = dict(state='complete', plot_script_sha256=sha(Path(__file__)), sources=sources,
        scan_predictions=56, exact_controls=4, figure_count=2, paired_seed=results['paired_seed'],
        score_definition=SCORE_DEFINITION,
        heatmap=dict(layers=list(LAYERS), pulse_steps=list(PULSE_STEPS),
                     matrices_percent={key: serializable_scores(value) for key, value in matrices.items()}, **style),
        state_curves=[dict(rank=value['rank'], run=value['run'], layer=value['layer'], component=value['component'],
                          pulse_step=value['pulse_step'], steps=list(range(30)),
                          percent=[float(item) if np.isfinite(item) else None for item in value['percent']],
                          final_output_axis_percent=value['final_output_axis_percent']) for value in curves],
        figures=[dict(path=str(path), sha256=sha(path)) for path in (heatmap_path, curves_path)],
        interpretation='Saved fixed-input q0 outputs only. Grasp/identity/attractor/root-cause claims require further causal and MuJoCo controls.',
        persistence_caveat='Later forwards have no component intervention; state differences can be inherited through the multistep solver.',
        curve_axis_caveat='Each step uses its own baseline XYZ direction and gap; fraction changes can reflect a changing denominator.',
        metrics_hash_boundary=('The scan completion hashes results/validation and the results hash each run\'s raw artifacts. '
                               'Per-step metrics JSON files had no completion-time SHA entry; this plot records their current SHA '
                               'and checks their final score against completed results. It does not reconstruct step scores from raw tensors.'),
        array_reanalysis=False, raw_video_rendering=False)
    (destination / 'plot.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    print('[COMPONENT-PULSE-PLOTS] ' + json.dumps(dict(state='complete', figures=str(destination), figure_count=2)), flush=True)


if __name__ == '__main__':
    main()
