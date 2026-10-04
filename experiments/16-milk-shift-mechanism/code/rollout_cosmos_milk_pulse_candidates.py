"""Prepare, then execute one control and three saved q0 pulse candidates.

Run --prepare-only before starting the 8926 server. Normal mode verifies that
preparation is unchanged and delegates all simulator operations to the exact
baseline rollout script. Inputs are a symlink to the immutable baseline.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys


SCOPE = ('Four actual 128-action executions at the same X+15cm input and paired seed: '
         'one native control and three ranked, saved query-zero single-component pulses. '
         'After the first 16 actions, queries 1-7 are native inference at current observations. '
         'Candidate ranking concerns continuous q0 action outputs, not a repaired grasp, '
         'target identity, attractor or localized mechanism. Further controls remain necessary.')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preparation(baseline):
    source = Path(__file__).with_name('rollout_cosmos_milk_shift_mechanism.py')
    pulse_source = Path(__file__).with_name('probe_cosmos_milk_component_pulses.py')
    scan = load_file('candidate_scan_readonly_helpers', pulse_source)
    baseline_summary, by_trial = scan.validate_baseline(baseline)
    inputs_meta = read_json(baseline / 'inputs/metadata.json')
    assert sha(source) == inputs_meta['script_sha256'], 'Exact baseline simulator source required'
    module = load_file('candidate_exact_baseline_simulator', source)
    pulses = baseline / 'component-pulses'
    assert not (pulses / 'failed.json').exists() and not (pulses / 'blocked.json').exists()
    complete = read_json(pulses / 'complete.json')
    assert complete['state'] == 'complete' and complete['scan_predictions'] == 56 and complete['control_predictions'] == 4
    for field in ('all_noise_schedule_pack_checks_exact', 'all_repeat_and_selfpatch_fields_bit_exact',
                  'single_component_pulse_per_scan_prediction'):
        assert complete[field] is True, field
    assert complete['script_sha256'] == sha(pulse_source)
    assert complete['results_sha256'] == sha(pulses / 'results.json')
    assert complete['validation_sha256'] == sha(pulses / 'validation.json')
    validation = read_json(pulses / 'validation.json')
    assert validation['state'] == 'all_exact_controls_passed' and len(validation['controls']) == 4
    results, top, pair = (read_json(pulses / name) for name in ('results.json', 'top_candidates.json', 'pair_selection.json'))
    assert results['state'] == 'complete' and len(results['cases']) == 56
    assert results['scan_predictions'] == 56 and results['control_predictions'] == 4
    assert top['state'] == 'candidates_for_followup_only' and len(top['candidates']) == 3
    assert top['candidates'] == results['top_three_candidates']
    assert top['ranking_metric'] == results['ranking_metric']
    assert pair['chosen'] is not None and pair['baseline_summary_sha256'] == sha(baseline / 'summary.json')
    seed = pair['chosen']['seed']
    assert seed in module.SEEDS and seed == complete['paired_seed'] == results['paired_seed']
    donor, recipient = f'x06_seed{seed}', f'x15_seed{seed}'
    assert pair['chosen']['donor_trial'] == donor and pair['chosen']['recipient_trial'] == recipient
    assert by_trial[donor]['selected_objects'] == ['milk_1'] and by_trial[recipient]['selected_objects'] == ['cream_cheese_1']
    candidates, sources, trial_map = [], {}, {'control': dict(scene='x15', seed=seed, candidate_rank=None, pulse_run=None)}
    previous_score = float('inf')
    for index, row in enumerate(top['candidates']):
        assert row in results['cases'] and row['run'] == results['ranked_runs'][index]
        assert row['donor_trial'] == donor and row['recipient_trial'] == recipient
        assert row['gain'] == 1 and row['intervention_count'] == 1
        score = row['final_normalized_action_xyz']['signed_projection_fraction']
        assert score is not None and math.isfinite(score) and score <= previous_score
        previous_score = score
        folder = pulses / 'runs' / row['run']
        assert Path(row['states_path']).resolve() == (folder / 'states.pt').resolve()
        for filename, field in (('states.pt', 'states_sha256'), ('patch_report.pt', 'patch_report_sha256'),
                                ('metadata.json', 'runtime_metadata_sha256')):
            assert sha(folder / filename) == row[field], (row['run'], field)
        for filename in ('patch_report.json', 'metrics.json', 'normalized_actions.json'):
            assert (folder / filename).is_file(), (row['run'], filename)
        candidate_id = f'candidate{index:02d}'
        candidates.append(dict(candidate_id=candidate_id, rank=index + 1, **row))
        trial_map[candidate_id] = dict(scene='x15', seed=seed, candidate_rank=index + 1, pulse_run=row['run'])
        sources[candidate_id] = dict(directory=str(folder), sha256={name: sha(folder / name) for name in
            ('states.pt', 'patch_report.pt', 'patch_report.json', 'metadata.json', 'metrics.json', 'normalized_actions.json')})
    selection = dict(state='prepared_for_actual_execution', baseline_root=str(baseline), chosen_seed=seed, seed=seed,
        donor_trial=donor, recipient_trial=recipient, trial_order=list(trial_map), trials=trial_map,
        ranking_metric=top['ranking_metric'], candidates=candidates, baseline_all_summaries=baseline_summary, scope=SCOPE)
    provenance = dict(wrapper_sha256=sha(Path(__file__)), exact_baseline_simulator_sha256=sha(source),
        pulse_scan_script_sha256=sha(pulse_source), baseline_root=str(baseline), chosen_seed=seed,
        baseline_sha256={name: sha(baseline / name) for name in
            ('summary.json', 'closed-loop/summary.json', 'closed-loop/complete.json', 'server/complete.json', 'inputs/metadata.json')},
        pulse_sha256={name: sha(pulses / name) for name in
            ('complete.json', 'top_candidates.json', 'results.json', 'validation.json', 'pair_selection.json', 'provenance.json')},
        candidate_sources=sources, url='http://127.0.0.1:8926', scope=SCOPE)
    module.TRIALS = {trial: ('x15', seed) for trial in trial_map}
    module.SEEDS = (seed,)
    module.URL = provenance['url']
    module.SCOPE = SCOPE
    return module, selection, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Completed baseline experiment root')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    baseline = args.output.resolve()
    module, selection, provenance = preparation(baseline)
    destination = baseline / 'candidate-execution'
    if args.prepare_only:
        destination.mkdir(exist_ok=False)
        (destination / 'inputs').symlink_to(baseline / 'inputs', target_is_directory=True)
        write_json(destination / 'selection.json', selection)
        write_json(destination / 'provenance.json', provenance)
        write_json(destination / 'prepared.json', dict(state='prepared', selection_sha256=sha(destination / 'selection.json'),
            provenance_sha256=sha(destination / 'provenance.json'), inputs_target=str(baseline / 'inputs'), scope=SCOPE))
        print('[PULSE-CANDIDATES] PREPARED', str(destination), flush=True)
        return
    assert destination.is_dir(), 'Run --prepare-only, start the 8926 server, then execute'
    assert (destination / 'inputs').is_symlink() and (destination / 'inputs').resolve() == baseline / 'inputs'
    assert read_json(destination / 'selection.json') == selection, 'Prepared selection changed'
    assert read_json(destination / 'provenance.json') == provenance, 'Prepared provenance or source changed'
    prepared = read_json(destination / 'prepared.json')
    assert prepared['state'] == 'prepared' and prepared['selection_sha256'] == sha(destination / 'selection.json')
    assert prepared['provenance_sha256'] == sha(destination / 'provenance.json')
    os.environ.setdefault('MUJOCO_GL', 'egl')
    sys.path.insert(0, '/home/current/work/openpi-demo/cosmos-policy')
    sys.path.insert(0, str(module.ROOT / 'cosmos-framework'))
    collector = module.load_file('candidate_baseline_collector', module.ROOT / 'work/collect_cosmos_goal_pair.py')
    rollout = module.load_file('candidate_baseline_rollout_helpers', module.ROOT / 'work/rollout_cosmos_goal_pair.py')
    module.execute_trials(destination, list(module.TRIALS), collector, rollout)


if __name__ == '__main__':
    main()
