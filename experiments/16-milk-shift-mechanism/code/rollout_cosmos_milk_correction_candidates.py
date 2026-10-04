"""Prepare and execute four preregistered seed198 saved-q0 candidates.

Use --output BASE --prepare-only before starting the port8928 server. Then
run without --prepare-only. The frozen baseline simulator owns physics,
strict dual-finger/lift selection, videos, 129 actual states and eight inputs.
Only the first sixteen actions come from correction-persistence. Queries1..7
use normal prediction on actual new images. No inference or simulator source
is edited, no extra physical warmup is added, and no weights are trained.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import traceback


SEED = 198
TRIALS = ('native_R', 'P', 'P_persistent_gain100', 'P_random_normmatched')
FILES = ('states.pt', 'metadata.json', 'normalized_actions.json', 'metrics.json',
         'actualsession.pt', 'actualsession.json', 'norms.json')
PORT = 8928
SCOPE = ('Four preregistered seed198 128-action executions from the unchanged X+15cm '
         'physical initial state. Query0 executes saved native_R/P/persistent gain1/random '
         'actions; gain100 means 100% replacement, numerical gain1. Queries1..7 are native '
         'predictions on actual current images. Strict dual-finger contact and >2cm lift '
         'for five consecutive records determines selected objects. No training or claim '
         'of a milk neuron, identity probability, attractor or localized root cause.')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def write_new_json(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preparation(baseline):
    """All source/data gates precede creation or use of the execution directory."""
    work = Path(__file__).resolve().parent
    scan_path = work / 'probe_cosmos_milk_component_pulses.py'
    scan = load_file('correction_execution_baseline_contract', scan_path)
    _, by_trial = scan.validate_baseline(baseline)
    assert by_trial['x15_seed198']['selected_objects'] == ['cream_cheese_1']
    inputs = read_json(baseline / 'inputs/metadata.json')
    simulator_path = work / 'rollout_cosmos_milk_shift_mechanism.py'
    assert sha(simulator_path) == inputs['script_sha256']
    module = load_file('correction_execution_frozen_simulator', simulator_path)
    assert Path(module.__file__).resolve() == simulator_path.resolve()
    server_provenance = read_json(baseline / 'server/provenance.json')
    simulator_provenance = read_json(baseline / 'closed-loop/provenance.json')
    code_paths = dict(server=work / 'serve_cosmos_milk_correction_candidates.py', rollout=Path(__file__),
        simulator=simulator_path, persistence=work / 'probe_cosmos_milk_correction_persistence.py',
        pulse_contract=scan_path, feedback=work / 'probe_cosmos_milk_feedback_inputs.py',
        normal_runtime=module.ROOT / 'work/serve_cosmos_goal_pair.py',
        factory=module.ROOT / 'work/probe_cosmos_language_routes.py',
        components=module.ROOT / 'work/cosmos_component_interventions.py',
        pixel_helper=module.ROOT / 'work/check_cosmos_official_pixels_cpu.py',
        collector=module.ROOT / 'work/collect_cosmos_goal_pair.py',
        controller_helper=module.ROOT / 'work/rollout_cosmos_goal_pair.py', stats=module.STATS)
    for key in ('normal_runtime', 'factory', 'components', 'pixel_helper'):
        assert sha(code_paths[key]) == server_provenance[key + '_sha256']
    for key in ('collector', 'controller_helper'):
        assert sha(code_paths[key]) == inputs[key + '_sha256']
    assert sha(code_paths['stats']) == simulator_provenance['stats_sha256']
    persistence = baseline / 'correction-persistence'
    assert not (persistence / 'failed.json').exists() and not (persistence / 'blocked.json').exists()
    complete, results, protocol, provenance = (read_json(persistence / name) for name in
        ('complete.json', 'results.json', 'protocol.json', 'provenance.json'))
    assert complete['state'] == 'complete'
    assert complete['q0_predictions'] == 24 and complete['exact_full_record_controls'] == 9
    for key in ('all_two_draw_noise_schedule_prepared_contracts_exact', 'all_site_calls_30_and_nonaction_rows_exact',
                'first_reset_actual_input_and_PP_increment_exact', 'actual_random_norms_matched_per_step'):
        assert complete[key] is True
    for key in ('results', 'protocol', 'provenance'):
        assert complete[key + '_sha256'] == sha(persistence / (key + '.json'))
    assert complete['script_sha256'] == provenance['script_sha256'] == sha(code_paths['persistence'])
    assert results['state'] == 'complete' and results['predictions'] == protocol['predictions'] == 24
    assert results['protocol_sha256'] == provenance['protocol_sha256'] == complete['protocol_sha256']
    assert results['physical_selection_predeclared'] is protocol['physical_candidates_selected_before_scores'] is True
    physical = [dict(seed=SEED, arm=arm) for arm in TRIALS]
    assert results['future_physical_candidates'] == protocol['future_physical_candidates'] == physical
    assert protocol['seeds'] == [195, 196, 198]
    assert protocol['pulse'] == dict(layer_zero_based=17, component='attention', step=20, gain=1., donor='paired x06')
    assert protocol['candidate']['layer_zero_based'] == 3 and protocol['candidate']['component'] == 'mlp'
    definitions = {arm['name']: arm for arm in protocol['arms']}
    assert len(definitions) == 8 and protocol['execution_order'] == list(definitions)
    for trial in TRIALS:
        arm = definitions[trial]
        assert arm['pulse'] is (trial != 'native_R')
        assert arm['reset_steps'] == (list(range(21, 30)) if trial in TRIALS[2:] else [])
        assert arm['gain'] == (1. if trial == 'P_persistent_gain100' else None)
        assert arm['mode'] == ('blend_R' if trial == 'P_persistent_gain100' else 'random' if trial == 'P_random_normmatched' else 'native')
    assert provenance['pulse_metric_source_sha256'] == sha(scan_path)
    assert provenance['feedback_script_sha256'] == sha(code_paths['feedback'])
    feedback = baseline / 'feedback-inputs'
    assert not (feedback / 'failed.json').exists()
    for key in ('complete', 'provenance', 'summary'):
        assert provenance['feedback_' + key + '_sha256'] == sha(feedback / (key + '.json'))
    feedback_complete, feedback_provenance = (read_json(feedback / name) for name in ('complete.json', 'provenance.json'))
    assert feedback_complete['state'] == 'complete' and feedback_complete['native_replays_exact'] == 6
    assert feedback_complete['summary_sha256'] == provenance['feedback_summary_sha256']
    for key in ('normal_runtime', 'factory', 'components'):
        assert provenance['source_sha256'][key] == feedback_provenance['source_sha256'][key] == sha(code_paths[key])
    assert provenance['scheduler_sha256'] == feedback_provenance['scheduler_sha256'] == server_provenance['scheduler_sha256']
    assert provenance['actual_transformer_sha256'] == feedback_provenance['actual_transformer_sha256']
    assert provenance['actual_pipeline_source_sha256'] == feedback_provenance['actual_pipeline_source_sha256']
    cases = results['cases']
    expected_runs = {f'seed{seed}_{arm}' for seed in (195, 196, 198) for arm in definitions}
    assert len(cases) == 24 and {row['run'] for row in cases} == set(results['completed']) == expected_runs
    rows = {row['run']: row for row in cases}
    for row in cases:
        assert row['run'] == f'seed{row["seed"]}_{row["arm"]}'
        assert row['all_native_contract_fields_exact'] is True and row['normalized_action_shape'] == [16, 10]
        directory = persistence / f'seed{row["seed"]}' / 'runs' / row['arm']
        assert Path(row['artifacts']).resolve() == directory.resolve()
        assert set(row['files_sha256']) == set(FILES)
        for name in FILES:
            assert row['files_sha256'][name] == sha(directory / name), f'{row["run"]}/{name} changed'
        assert row['sources_sha256'] == sha(persistence / f'seed{row["seed"]}' / 'sources.json')
    controls = results['controls']
    assert len(controls) == 9 and {row['run'] for row in controls} == {
        f'seed{seed}_{arm}' for seed in (195, 196, 198) for arm in ('native_R', 'P', 'R_persistent_self')}
    for control in controls:
        assert control['all_saved_fields_bit_exact'] is True
        case = rows[control['run']]
        assert control['states_sha256'] == case['files_sha256']['states.pt']
        assert control['actualsession_sha256'] == case['files_sha256']['actualsession.pt']
    folder = persistence / 'seed198'
    sources = read_json(folder / 'sources.json')
    for role, scene in (('R', 'x15'), ('D', 'x06')):
        source = sources[role]
        trial = baseline / 'closed-loop' / f'{scene}_seed{SEED}'
        assert source['trial'] == trial.name and source['image_sha256'] == sha(trial / 'input_00.png')
        for name in ('states', 'components'):
            assert source[name + '_sha256'] == sha(trial / 'chunk_00' / (name + '.pt'))
    for source in sources['feedback'].values():
        assert source['sha256'] == sha(Path(source['path']))
    targets_path = folder / 'random-target-norms.json'
    targets = read_json(targets_path)
    assert targets['reference_arm'] == 'P_persistent_gain100'
    assert targets['reference_actualsession_sha256'] == rows['seed198_P_persistent_gain100']['files_sha256']['actualsession.pt']
    assert set(targets['actual_bf16_delta_l2_by_step']) == {str(step) for step in range(21, 30)}
    chosen = {}
    for trial in TRIALS:
        row = rows[f'seed198_{trial}']
        arm = definitions[trial]
        assert row['pulse'] == arm['pulse'] and row['reset_steps'] == arm['reset_steps'] and row['gain'] == arm['gain']
        directory = folder / 'runs' / trial
        report = read_json(directory / 'actualsession.json')
        assert report['complete'] is True and report['arm'] == arm and report['seed'] == SEED
        assert report['n_model_forwards'] == report['finished_forwards'] == report['attention_site_calls'] == report['mlp_site_calls'] == 30
        assert report['explicit_processor_recomputations'] == 0 and report['nonaction_rows_bit_exact'] is True
        assert report['step21_actual_kwargs_exact'] is report['step21_actual_MLP_before_exact'] is True
        expected_events = {(20, 17, 'attention')} if arm['pulse'] else set()
        expected_events |= {(step, 3, 'mlp') for step in arm['reset_steps']}
        assert report['intervention_count'] == report['expected_intervention_count'] == row['intervention_count'] == len(expected_events)
        events = report['events'].values()
        assert {(event['step'], event['layer'], event['component']) for event in events} == expected_events
        norm_events = read_json(directory / 'norms.json')['events']
        assert len(norm_events) == len(expected_events)
        for event in report['events'].values():
            assert event['nonaction_rows_bit_exact'] is True and event['nonaction_before'] == event['nonaction_after']
            if trial == 'P_persistent_gain100' and event['component'] == 'mlp':
                assert event['actual_delta_l2'] == targets['actual_bf16_delta_l2_by_step'][str(event['step'])]
            if trial == 'P_random_normmatched' and event['component'] == 'mlp':
                step = event['step']
                target = targets['actual_bf16_delta_l2_by_step'][str(step)]
                actual = event['actual_delta_l2']
                calibration = event['random_calibration']
                assert math.isfinite(target) and target >= 0 and math.isfinite(actual) and actual >= 0
                assert calibration['cpu_generator_seed'] == protocol['random_cpu_seeds']['198'][str(step)]
                assert calibration['target_actual_delta_l2'] == target and calibration['actual_hook_delta_l2'] == actual
                error = 0. if target == 0 else abs(actual - target) / target
                assert actual == 0 if target == 0 else error <= protocol['actual_bf16_norm_relative_tolerance'] <= 1e-3
                assert math.isclose(error, calibration['actual_hook_relative_error'], abs_tol=1e-15)
        metadata = read_json(directory / 'metadata.json')
        assert metadata['seed'] == SEED and metadata['prompt'] == module.PROMPT
        assert metadata['model_calls'] == metadata['num_inference_steps'] == 30 and metadata['prepare_calls'] == 1
        assert metadata['input_png_sha256'] == sha(baseline / 'inputs/x15/input.png')
        actions = module.np.asarray(read_json(directory / 'normalized_actions.json'))
        assert actions.shape == (16, 10) and module.np.isfinite(actions).all()
        chosen[trial] = dict(scene='x15', seed=SEED, arm=arm, run=row['run'],
            directory=str(directory), files_sha256=row['files_sha256'])
    selection = dict(state='prepared_for_actual_execution', seed=SEED, trial_order=list(TRIALS), trials=chosen,
        q0_source='Each preregistered arm uses its completed correction-persistence states/actions.',
        later_queries='Queries1..7 native at actual new images; no component patches.', scope=SCOPE)
    frozen = dict(code={key: dict(path=str(path), sha256=sha(path)) for key, path in code_paths.items()},
        baseline={name: sha(baseline / name) for name in ('summary.json', 'closed-loop/complete.json',
            'closed-loop/provenance.json', 'server/complete.json', 'server/provenance.json', 'inputs/metadata.json')},
        persistence={name: sha(persistence / name) for name in ('complete.json', 'results.json', 'protocol.json', 'provenance.json')},
        chosen_sources=chosen, seed198_sources_sha256=sha(folder / 'sources.json'),
        seed198_random_targets_sha256=sha(targets_path), url=f'http://127.0.0.1:{PORT}',
        queries=32, saved_q0_predictions=4, native_new_image_predictions=28, scope=SCOPE)
    module.TRIALS = {trial: ('x15', SEED) for trial in TRIALS}
    module.SEEDS = (SEED,)
    module.URL, module.SCOPE = frozen['url'], SCOPE
    return module, selection, frozen


def verify_prepared(baseline):
    module, selection, provenance = preparation(baseline)
    out = baseline / 'correction-execution'
    assert out.is_dir() and not (out / 'wrapper_failed.json').exists()
    assert (out / 'inputs').is_symlink() and (out / 'inputs').resolve() == (baseline / 'inputs').resolve()
    assert read_json(out / 'selection.json') == selection
    assert read_json(out / 'provenance.json') == provenance
    prepared = read_json(out / 'prepared.json')
    assert prepared['state'] == 'prepared' and prepared['selection_sha256'] == sha(out / 'selection.json')
    assert prepared['provenance_sha256'] == sha(out / 'provenance.json')
    return module, selection, provenance


def native_control(baseline, out, np):
    reference, current = baseline / 'closed-loop/x15_seed198', out / 'closed-loop/native_R'
    with (reference / 'sim_states.npy').open('rb') as a, (current / 'sim_states.npy').open('rb') as b:
        old, new = np.load(a, allow_pickle=False), np.load(b, allow_pickle=False)
    assert old.shape == new.shape and old.shape[0] == 129 and old.dtype == new.dtype
    assert old.tobytes(order='C') == new.tobytes(order='C'), 'Native_R complete physical states differ'
    files = ['trajectory.json', 'actions.json', 'executed_actions.json', 'normalized_actions.json',
             'denormalized_actions.json', *[f'input_{query:02d}.png' for query in range(8)]]
    for name in files:
        assert (reference / name).read_bytes() == (current / name).read_bytes(), f'Native_R baseline bytes differ: {name}'
    return dict(state='exact', trial='native_R', baseline_trial='x15_seed198', physical_records=129,
        executed_actions=128, actual_input_images=8, all_sim_state_bytes_exact=True,
        all_action_and_trajectory_and_input_files_bytes_exact=True,
        files={name: dict(baseline_sha256=sha(reference / name), actual_sha256=sha(current / name))
               for name in ('sim_states.npy', *files)}, scope=SCOPE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Completed original baseline root')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    baseline = args.output.resolve()
    out = baseline / 'correction-execution'
    if args.prepare_only:
        _, selection, provenance = preparation(baseline)
        out.mkdir(exist_ok=False)
        (out / 'inputs').symlink_to(baseline / 'inputs', target_is_directory=True)
        write_new_json(out / 'selection.json', selection)
        write_new_json(out / 'provenance.json', provenance)
        write_new_json(out / 'prepared.json', dict(state='prepared', selection_sha256=sha(out / 'selection.json'),
            provenance_sha256=sha(out / 'provenance.json'), scope=SCOPE))
        print('[CORRECTION-EXECUTION] prepared: ' + str(out), flush=True)
        return
    module, _, _ = verify_prepared(baseline)
    assert not (out / 'closed-loop').exists(), 'No overwrite, partial reuse or automatic simulator retry'
    try:
        os.environ.setdefault('MUJOCO_GL', 'egl')
        sys.path.insert(0, '/home/current/work/openpi-demo/cosmos-policy')
        sys.path.insert(0, str(module.ROOT / 'cosmos-framework'))
        collector = module.load_file('correction_execution_collector', module.ROOT / 'work/collect_cosmos_goal_pair.py')
        rollout = module.load_file('correction_execution_controller', module.ROOT / 'work/rollout_cosmos_goal_pair.py')
        module.execute_trials(out, list(TRIALS), collector, rollout)
        result = read_json(out / 'closed-loop/complete.json')
        assert result['state'] == 'complete' and result['cases'] == 4 and result['completed_trials'] == list(TRIALS)
        assert result['steps_per_case'] == 128 and result['state_records_per_case'] == 129
        control = native_control(baseline, out, module.np)
        write_new_json(out / 'native-control.json', control)
        server = read_json(out / 'server/complete.json')
        assert server['state'] == 'complete' and server['queries'] == 32
        assert server['saved_q0_predictions'] == 4 and server['native_new_image_predictions'] == 28
        assert server['saved_native_R_entire_record_equals_baseline'] is server['all_saved_q0_contracts_exact'] is True
        write_new_json(out / 'complete.json', dict(state='complete', trials=list(TRIALS), seed=SEED,
            physical_trials=4, queries=32, saved_q0_predictions=4, native_new_image_predictions=28,
            native_R_full_129_state_and_actions_and_trajectory_exact=True,
            server_complete_sha256=sha(out / 'server/complete.json'),
            simulator_complete_sha256=sha(out / 'closed-loop/complete.json'),
            native_control_sha256=sha(out / 'native-control.json'), provenance_sha256=sha(out / 'provenance.json'), scope=SCOPE))
        print('[CORRECTION-EXECUTION] complete: ' + str(out), flush=True)
    except Exception as exc:
        if not (out / 'wrapper_failed.json').exists():
            write_new_json(out / 'wrapper_failed.json', dict(error=str(exc), traceback=traceback.format_exc(), scope=SCOPE))
        raise


if __name__ == '__main__':
    main()
