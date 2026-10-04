"""Plot one two-panel figure from completed, checked attention-head JSON data.

--output is the experiment/baseline root containing attention-heads. Read
saved scores only; no Torch, network-array reanalysis, simulation or video.
Panel one shows 32 separate single-head XYZ-axis projections. Panel two shows
head23's natural donor pulse and its three actual bf16 norm-matched random
controls. The reverse trial has a D->R reference and is annotated separately;
forward/random bars have R->D references. Heads are zero based. Do not sum the
single-head scores to partition a whole-component effect.
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


TOP_HEAD = 23
RANDOM_SEEDS = (17002901, 17002902, 17002903)
SCORE_DEFINITION = (
    'fraction = dot(P-R,D-R)/dot(D-R,D-R), on sixteen continuous normalized XYZ action rows; percent = 100*fraction. '
    'Forward and random trials: R=x15_seed198, D=x06_seed198. Reverse trial: recipient=D and donor=R, '
    'so its axis is D->R and must not be plotted as an R->D effect. The score is an axis projection, '
    'not probability, identity, actual grasp outcome or a share of an additive whole-component effect.'
)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def score(row):
    metric = row['final_normalized_action_xyz']
    assert metric['gap_meaningful'] is True
    value = metric['signed_projection_fraction']
    assert type(value) in (int, float) and math.isfinite(value)
    return float(value) * 100


def validate_artifacts(directory, row):
    for key, filename in (('states_sha256', 'states.pt'), ('head_site_sha256', 'head_site.pt'),
                           ('metrics_sha256', 'metrics.json')):
        assert row[key] == sha(directory / filename), f'{row["run"]}/{filename}: original SHA mismatch'
    metrics = read_json(directory / 'metrics.json')
    assert metrics['final_normalized_action_xyz'] == row['final_normalized_action_xyz']
    assert metrics['altered_normalized_xyz'] == row['normalized_xyz']
    assert np.asarray(row['normalized_xyz']).shape == (16, 3)
    assert np.isfinite(np.asarray(row['normalized_xyz'], dtype=float)).all()


def checked_data(baseline):
    root = baseline / 'attention-heads'
    assert not (root / 'failed.json').exists()
    paths = {key: root / f'{key}.json' for key in ('complete', 'results', 'validation', 'provenance')}
    complete = read_json(paths['complete'])
    assert complete['state'] == 'complete'
    assert (complete['predictions'], complete['controls'], complete['single_heads'], complete['followups']) == (40, 4, 32, 4)
    for key in ('all32_pre_W_entire_record_equals_coarse_post_W',
                'all_actual_non_action_and_unselected_head_checks_exact', 'all_noise_schedule_pack_prefix_checks_exact'):
        assert complete[key] is True
    assert complete['results_sha256'] == sha(paths['results'])
    assert complete['validation_sha256'] == sha(paths['validation'])
    assert complete['captures_sha256'] == sha(root / 'actual_head_captures.pt')
    results, validation, provenance = (read_json(paths[key]) for key in ('results', 'validation', 'provenance'))
    assert results['state'] == 'complete' and results['predictions'] == 40
    assert results['single_head_predictions'] == 32 and results['followup_predictions'] == results['control_predictions'] == 4
    assert validation['state'] == 'all_four_exact_controls_passed'
    for key in ('two_fp32_noise_draws_exact', 'full_30step_metadata_exact', 'timesteps_sigmas_exact',
                'raw_prepared_action_noise_exact', 'all32_pre_W_entire_record_equals_coarse_post_W'):
        assert validation[key] is True
    controls = validation['controls']
    assert results['controls'] == controls and len(controls) == 4
    assert {row['run'] for row in controls} == {'capture_recipient', 'capture_donor', 'self_all32', 'donor_all32_coarse_equivalence'}
    for row in controls:
        if row['run'].startswith('capture_'):
            assert row['all_saved_baseline_fields_bit_exact'] is True and row['post_W_equals_native_attention_increment'] is True
        else:
            assert row['all_reference_record_fields_bit_exact'] is True
        destination = root / 'controls' / row['run']
        assert row['states_sha256'] == sha(destination / 'states.pt')
        assert row['head_site_sha256'] == sha(destination / 'head_site.pt')
    assert provenance['script_sha256'] == complete['script_sha256']
    assert provenance['sources']['direction_metrics']['sha256'] == complete['direction_metric_source_sha256']
    assert provenance['fixed_site'] == dict(layer_zero_based=17, layer_human=18, denoising_step=29,
                                          num_heads=32, head_dim=128, input_width=4096)
    assert provenance['recipient_trial'] == 'x15_seed198' and provenance['donor_trial'] == 'x06_seed198' and provenance['seed'] == 198
    cases = results['single_heads']
    assert len(cases) == 32 and {row['head_zero_based'] for row in cases} == set(range(32))
    by_head = {}
    for row in cases:
        head = row['head_zero_based']
        assert row['run'] == f'head{head:02d}' and row['head_human'] == head + 1
        assert row['mode'] == 'natural_donor' and row['recipient_role'] == 'recipient'
        assert row['layer'] == 17 and row['denoising_step'] == 29 and row['intervention_count'] == 1
        score(row)
        validate_artifacts(root / 'runs' / row['run'], row)
        by_head[head] = row
    ranked = sorted(cases, key=score, reverse=True)
    assert results['ranked_heads'] == [row['head_zero_based'] for row in ranked]
    assert results['top_three_heads'] == ranked[:3]
    assert ranked[0]['head_zero_based'] == TOP_HEAD, 'Measured strongest head differs: refuse mislabeled plot'
    followups = results['followups']
    assert len(followups) == 4
    reverse = [row for row in followups if row['mode'] == 'reverse_natural_donor']
    random = [row for row in followups if row['mode'] == 'actual_norm_matched_random']
    assert len(reverse) == 1 and len(random) == 3
    reverse = reverse[0]
    assert reverse['head_zero_based'] == TOP_HEAD and reverse['recipient_role'] == 'donor'
    assert reverse['run'] == f'reverse_head{TOP_HEAD:02d}'
    score(reverse)
    validate_artifacts(root / 'followups' / reverse['run'], reverse)
    random.sort(key=lambda row: row['random_control']['cpu_generator_seed'])
    assert [row['random_control']['cpu_generator_seed'] for row in random] == list(RANDOM_SEEDS)
    natural_norm = by_head[TOP_HEAD]['actual_selected_delta_l2']
    for row in random:
        assert row['head_zero_based'] == TOP_HEAD and row['recipient_role'] == 'recipient'
        info = row['random_control']
        assert info['head'] == TOP_HEAD
        assert math.isclose(info['target_actual_donor_delta_l2'], natural_norm, rel_tol=1e-12)
        assert math.isclose(info['actual_quantized_random_delta_l2'], row['actual_selected_delta_l2'], rel_tol=1e-12)
        error = abs(info['actual_quantized_random_delta_l2'] - natural_norm) / natural_norm
        assert math.isclose(error, info['relative_norm_error'], rel_tol=1e-10, abs_tol=1e-15)
        assert error <= info['relative_norm_tolerance'] <= 1e-3
        destination = root / 'followups' / row['run']
        validate_artifacts(destination, row)
        assert row['random_control_sha256'] == sha(destination / 'random_control.pt')
        assert info == read_json(destination / 'random_control.json')
        score(row)
    sources = {key: dict(path=str(path), sha256=sha(path)) for key, path in paths.items()}
    sources.update(head_captures=dict(path=str(root / 'actual_head_captures.pt'), sha256=complete['captures_sha256']),
                   original_probe_script_sha256=complete['script_sha256'],
                   direction_metric_source_sha256=complete['direction_metric_source_sha256'],
                   metrics=[dict(run=row['run'], sha256=row['metrics_sha256']) for row in cases + followups])
    return root, results, by_head, random, reverse, sources


def render(path, by_head, random, reverse):
    heads = np.arange(32)
    values = np.asarray([score(by_head[int(head)]) for head in heads])
    comparison = [by_head[TOP_HEAD], *random]
    colors = ['#0072B2' if head != TOP_HEAD else '#D55E00' for head in heads]
    figure, axes = plt.subplots(1, 2, figsize=(13.3, 6.5), gridspec_kw={'width_ratios': [1.9, 1]}, sharey=True)
    axes[0].bar(heads, values, color=colors, width=.8)
    axes[0].set_xticks(heads, [str(head) for head in heads], fontsize=8)
    axes[0].set_xlabel('Real attention head index (zero based)')
    axes[0].set_ylabel('Signed normalized XYZ-axis projection (%)\nR -> D, fixed first input only')
    axes[0].set_title('32 independent single-head donor pulses', fontsize=13)
    top_value = values[TOP_HEAD]
    axes[0].annotate(f'H{TOP_HEAD}: {top_value:.3f}%', xy=(TOP_HEAD, top_value),
                     xytext=(TOP_HEAD - 6, top_value + .20), arrowprops={'arrowstyle': '->', 'color': '#D55E00'}, fontsize=10)
    x = np.arange(4)
    second_values = [score(row) for row in comparison]
    axes[1].bar(x, second_values, color=['#D55E00', '#999999', '#999999', '#999999'], width=.65)
    axes[1].set_xticks(x, [f'H{TOP_HEAD}\ndonor', 'random\n1', 'random\n2', 'random\n3'])
    axes[1].set_title(f'H{TOP_HEAD}: natural vs actual norm-matched noise', fontsize=12)
    axes[1].set_xlabel('Same selected head; three independent CPU RNG seeds')
    for i, value in enumerate(second_values):
        axes[1].text(i, value + .07 if value >= 0 else value - .07, f'{value:.3f}%',
                     ha='center', va='bottom' if value >= 0 else 'top', fontsize=10)
    lower = min(0., float(values.min()), min(second_values))
    upper = max(float(values.max()), max(second_values), .1)
    span = max(upper - lower, .1)
    axes[0].set_ylim(lower - .12 * span, upper + .27 * span)
    for axis in axes:
        axis.axhline(0, color='#555555', linewidth=.9)
        axis.grid(axis='y', alpha=.22)
        axis.set_axisbelow(True)
    figure.suptitle('Real attention heads: fixed-input q0 action-output XYZ-axis effects\nLayer 18 (index 17), denoising index 29; percentages are not probabilities or grasp rates', fontsize=14, y=.97)
    figure.subplots_adjust(left=.075, right=.98, top=.80, bottom=.25, wspace=.22)
    figure.text(.075, .15, f'Reverse trial is not a forward bar: H{TOP_HEAD}, D -> R axis, {score(reverse):+.3f}%. Its recipient is X+6 cm, not X+15 cm.', fontsize=10)
    figure.text(.075, .11, 'Forward/random bars use R = X+15 cm baseline and D = X+6 cm baseline. Random norms are measured after bf16 quantization.', fontsize=9)
    figure.text(.075, .07, 'Single-head effects are separate interventions; adding them does not partition the whole attention effect.', fontsize=9)
    figure.text(.075, .03, 'These are normalized action-output projections. This figure reports no identity score or actual grasp outcome.', fontsize=9)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Completed experiment16/baseline root containing attention-heads')
    args = parser.parse_args()
    root, results, by_head, random, reverse, sources = checked_data(args.output.resolve())
    destination = root / 'figures'
    destination.mkdir(exist_ok=False)
    path = destination / 'head_xyz_and_random_controls.png'
    render(path, by_head, random, reverse)
    manifest = dict(state='complete', plot_script_sha256=sha(Path(__file__)), sources=sources,
        figure_count=1, panel_count=2, predictions=40, controls=4,
        score_definition=SCORE_DEFINITION, head_index_convention='zero based; H23 is the 24th head',
        single_head_percent=[score(by_head[head]) for head in range(32)],
        strongest_head=TOP_HEAD, natural=dict(percent=score(by_head[TOP_HEAD]),
            actual_delta_l2=by_head[TOP_HEAD]['actual_selected_delta_l2'], normalized_xyz=by_head[TOP_HEAD]['normalized_xyz']),
        random=[dict(percent=score(row), norm_check=row['random_control'], normalized_xyz=row['normalized_xyz']) for row in random],
        reverse=dict(percent=score(reverse), direction='D -> R', recipient='x06_seed198', donor='x15_seed198',
                     included_in_forward_bars=False, normalized_xyz=reverse['normalized_xyz']),
        baseline_normalized_xyz=results['baseline_normalized_xyz'],
        figure=dict(path=str(path), sha256=sha(path)), array_reanalysis=False,
        interpretation='Separate fixed-input output causal effects; single-head scores are not additive shares, identities, probabilities or actual grasp results.')
    (destination / 'plot.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    print('[HEAD-PLOT] ' + json.dumps(dict(state='complete', figure=str(path), strongest_head=TOP_HEAD,
                                         natural_percent=score(by_head[TOP_HEAD]), reverse_direction='D -> R',
                                         reverse_percent=score(reverse))), flush=True)


if __name__ == '__main__':
    main()
