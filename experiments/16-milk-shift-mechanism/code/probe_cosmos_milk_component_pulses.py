"""Scan brief local donor pulses on the paired milk-position query-zero inputs.

--output is the completed nine-trial milk-shift BASELINE root. This script owns
only its absent component-pulses child. It changes no model-library file and
does not execute MuJoCo. Choose one seed with strict milk-only selection at
X+6cm and strict cheese-only selection at X+15cm (198, then 195, then 196).
If that pair does not exist, preserve blocked.json and exit normally.

After full-noise/schedule/cache alignment and exact repeat/self-patch controls,
scan seven zero-based layers x two component increments x four single denoising
forwards = 56 natural donor replacements. The other 29 forwards run natively.
Rank continuous final normalized XYZ outputs by their signed projection toward
the donor. These are causal effects on fixed-input q0 internal computation and
outputs, NOT evidence of which object will be grasped, an isolated identity
feature, an attractor, or the root cause of the later closed-loop switch.
Effects after a pulse may simply be inherited through the multistep solver.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time
import traceback


ROOT = Path('/home/current/work/cosmos3')
PROMPT = 'pick up the milk and place it in the basket'
SHIFTS = {'x00': 0, 'x06': 6, 'x15': 15}
SEEDS = (195, 196, 198)
SEED_PREFERENCE = (198, 195, 196)
LAYERS = (5, 11, 17, 23, 29, 31, 35)
COMPONENTS = ('attention', 'mlp')
PULSE_STEPS = (0, 10, 20, 29)
CRITERION = 'Simultaneous left/right fingerpad contact AND >2cm lift in each of five consecutive control-step records.'
GAP_ABS_FLOOR = 1e-6
GAP_RELATIVE_FLOOR = 1e-4
SCOPE = ('Fixed first observation/q0 local component-output causal scan only. '
         'No MuJoCo intervention rollout; no claim of grasp repair, target identity, '
         'attractor, or a localized root cause. Persistence may be solver inheritance.')


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def validate_baseline(out):
    """Read-only gates run before importing torch/loading any model."""
    expected = {f'{scene}_seed{seed}' for scene in SHIFTS for seed in SEEDS}
    for path in ('server/failed.json', 'closed-loop/failed.json', 'simulator_failed.json', 'inputs/failed.json'):
        assert not (out / path).exists(), f'Baseline failure must be reviewed: {path}'
    summary = read_json(out / 'summary.json')
    assert summary == read_json(out / 'closed-loop/summary.json'), 'Baseline summaries disagree'
    cases = summary['cases']
    assert len(cases) == 9 and {row['trial'] for row in cases} == expected, 'All nine baseline summaries required'
    assert summary['selection_criterion'] == CRITERION
    assert set(summary['completed_trials']) == expected
    complete = read_json(out / 'closed-loop/complete.json')
    assert complete['state'] == 'complete' and complete['cases'] == 9
    assert set(complete['completed_trials']) == expected and complete['steps_per_case'] == 128
    assert complete['queries_per_case'] == 8 and complete['state_records_per_case'] == 129
    server_complete = read_json(out / 'server/complete.json')
    assert server_complete['state'] == 'complete' and server_complete['predictions'] == 72
    assert server_complete['query_zero_component_captures'] == 9
    assert server_complete['observer_free_repeat_all_tensors_exact'] is True
    inputs_complete = read_json(out / 'inputs/complete.json')
    assert inputs_complete['state'] == 'matched_inputs_verified'
    assert inputs_complete['metadata_sha256'] == sha(out / 'inputs/metadata.json')
    assert inputs_complete['source_center_pixels_exact'] is True
    inputs_meta = read_json(out / 'inputs/metadata.json')
    assert inputs_meta['prompt'] == PROMPT and inputs_meta['shifts_cm'] == SHIFTS
    by_trial = {row['trial']: row for row in cases}
    for scene in SHIFTS:
        for seed in SEEDS:
            trial = f'{scene}_seed{seed}'
            row, run = by_trial[trial], out / 'closed-loop' / trial
            assert row == read_json(run / 'summary.json'), trial
            assert row['scene'] == scene and row['shift_cm'] == SHIFTS[scene]
            assert row['noise_seed_base'] == seed and row['prompt'] == PROMPT
            assert row['steps'] == 128 and row['queries'] == 8 and row['state_records'] == 129
            assert row['selection_criterion'] == CRITERION
            selected = {name for name, value in row['per_object'].items()
                        if value['first_2cm_for_5frames'] is not None}
            assert selected == set(row['selected_objects']), f'{trial}: strict summary mismatch'
            assert (run / 'input_00.png').read_bytes() == (out / 'inputs' / scene / 'input.png').read_bytes()
            for filename in ('states.pt', 'components.pt', 'metadata.json', 'normalized_actions.json'):
                assert (run / 'chunk_00' / filename).is_file(), (trial, filename)
    return summary, by_trial


def select_pair(by_trial):
    candidates = []
    for seed in SEED_PREFERENCE:
        donor, recipient = by_trial[f'x06_seed{seed}'], by_trial[f'x15_seed{seed}']
        matched = (set(donor['selected_objects']) == {'milk_1'}
                   and set(recipient['selected_objects']) == {'cream_cheese_1'})
        candidates.append(dict(seed=seed, donor_trial=donor['trial'], recipient_trial=recipient['trial'],
                               donor_strict_selected=donor['selected_objects'],
                               recipient_strict_selected=recipient['selected_objects'], eligible=matched))
    choice = next((row for row in candidates if row['eligible']), None)
    return choice, candidates


def tensor_meta(value, torch):
    flat_bytes = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                sha256=hashlib.sha256(flat_bytes).hexdigest())


def validate_record(record, torch):
    assert len(record['pure_noise']) == 2, 'Both actual vision/action random draws required'
    assert all(value.dtype == torch.float32 and torch.isfinite(value).all() for value in record['pure_noise'])
    assert record['pure_noise'][0].ndim == 5 and record['pure_noise'][1].shape == (16, 64)
    assert record['timesteps'].numel() == 30 and record['sigmas'].numel() >= 30
    for key, shape in (('readout', (30, 16, 4096)), ('action_velocity', (30, 16, 64)),
                       ('action_states', (31, 1, 16, 64)), ('actions', (16, 10))):
        assert tuple(record[key].shape) == shape and torch.isfinite(record[key]).all(), key
    assert record['action_states'].dtype == torch.float32
    assert record['model_input']['action_mse_loss_indexes'].numel() == 16


def validate_cache(cache, record, hooks, runtime, label):
    torch = runtime.torch
    assert cache['version'] == 1 and cache['complete'] is True and cache['n_layers'] == 36
    assert cache['n_steps'] == len(cache['steps']) == 30, f'{label}: truncated metadata'
    expected = {(s, layer, component) for s in range(30) for layer in range(36) for component in COMPONENTS}
    assert set(cache['frames']) == expected
    assert cache['captured_component_count'] == cache['component_call_count'] == len(expected)
    assert cache['processor_recompute_calls'] == 0
    runtime.require_exact(cache['site_specs'], hooks.site_specs, f'{label}/actual_model_sites')
    runtime.require_exact(cache['first_model_kwargs'], record['model_input'], f'{label}/actual_first_kwargs')
    rows = record['model_input']['action_mse_loss_indexes'].flatten() - int(record['model_input']['und_len'])
    assert rows.numel() == 16 and rows.unique().numel() == 16
    for step in range(30):
        runtime.require_exact(cache['steps'][step]['layout']['action_rows'], rows, f'{label}/action_rows/{step}')
    for key, frame in cache['frames'].items():
        assert tuple(frame.shape) == (16, 4096) and torch.isfinite(frame).all(), (label, key)
        assert frame.dtype == torch.bfloat16, (label, key, frame.dtype)
    return rows


def direction_metrics(recipient, donor, altered):
    """Continuous signed projection with absolute AND relative gap guards."""
    r, d, p = (value.detach().double().cpu().reshape(-1) for value in (recipient, donor, altered))
    assert r.shape == d.shape == p.shape
    gap, delta = d - r, p - r
    gap_rms = float(gap.square().mean().sqrt())
    delta_rms = float(delta.square().mean().sqrt())
    reference_rms = max(float(r.square().mean().sqrt()), float(d.square().mean().sqrt()))
    threshold = max(GAP_ABS_FLOOR, GAP_RELATIVE_FLOOR * reference_rms)
    meaningful = gap_rms > threshold
    numerator = float((delta * gap).sum())
    denominator = float(gap.square().sum())
    result = dict(rmse_to_recipient=delta_rms, rmse_to_donor=float((p - d).square().mean().sqrt()),
                  donor_recipient_gap_rmse=gap_rms, gap_floor=threshold, gap_meaningful=meaningful,
                  signed_projection_fraction=numerator / denominator if meaningful else None,
                  signed_projection_rms=numerator / math.sqrt(denominator * r.numel()) if meaningful else None,
                  delta_cosine_with_donor_direction=(numerator / math.sqrt(denominator * float(delta.square().sum()))
                                                    if meaningful and delta_rms > GAP_ABS_FLOOR else None))
    assert all(value is None or type(value) is bool or math.isfinite(value) for value in result.values())
    return result


def prediction_metrics(recipient, donor, altered):
    per_step = []
    for step in range(30):
        row = dict(step=step, timestep=int(recipient['timesteps'][step]), sigma=float(recipient['sigmas'][step]))
        for name, key, subset in (('readout', 'readout', None),
                                  ('action_velocity_all64', 'action_velocity', None),
                                  ('action_velocity_xyz', 'action_velocity', slice(0, 3))):
            values = [record[key][step] if subset is None else record[key][step, :, subset]
                      for record in (recipient, donor, altered)]
            row[name] = direction_metrics(*values)
        row['action_state_after_step_xyz'] = direction_metrics(
            recipient['action_states'][step + 1, 0, :, :3], donor['action_states'][step + 1, 0, :, :3],
            altered['action_states'][step + 1, 0, :, :3])
        per_step.append(row)
    return dict(final_normalized_action_xyz=direction_metrics(
                    recipient['actions'][:, :3], donor['actions'][:, :3], altered['actions'][:, :3]),
                final_normalized_action_all10=direction_metrics(recipient['actions'], donor['actions'], altered['actions']),
                per_denoising_step=per_step)


def report_metadata(report, torch):
    result = {key: value for key, value in report.items() if key not in ('changes', 'action_rows')}
    result['action_rows'] = report['action_rows'].tolist()
    result['changes'] = [dict(step=value['step'], layer=value['layer'], component=value['component'],
        shape=list(value['shape']), exact_no_change=value['exact_no_change'], delta_l2=value['delta_l2'],
        before=tensor_meta(value['before'], torch), after=tensor_meta(value['after'], torch))
        for _, value in sorted(report['changes'].items())]
    return result


def validate_run_contract(record, recipient, runtime, label):
    validate_record(record, runtime.torch)
    for key in ('pure_noise', 'timesteps', 'sigmas', 'model_input', 'prepared_latents_and_masks'):
        runtime.require_exact(record[key], recipient[key], f'{label}/native_input_contract/{key}')


def validate_patch_report(report, recipient_cache, donor_cache, rows, layer, component, step, runtime):
    assert report['complete'] is True and report['n_steps'] == report['expected_steps'] == 30
    assert report['intervention_count'] == 1 and report['processor_recompute_calls'] == 0
    assert report['layer'] == layer and report['component'] == component and report['steps'] == (step,)
    assert report['gain'] == 1 and report['mode'] == 'donor_blend'
    runtime.require_exact(report['action_rows'], rows, 'actual_sixteen_GEN_action_rows')
    key = (step, layer, component)
    assert set(report['changes']) == {key}
    event = report['changes'][key]
    assert tuple(event['shape']) == (16, 4096)
    runtime.require_exact(event['before'], recipient_cache['frames'][key], 'single_pulse_native_before')
    runtime.require_exact(event['after'], donor_cache['frames'][key], 'single_pulse_exact_donor_after')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Completed nine-trial baseline root')
    args = parser.parse_args()
    baseline = args.output.resolve()
    summary, by_trial = validate_baseline(baseline)
    out = baseline / 'component-pulses'
    out.mkdir(exist_ok=False)
    started, completed = time.perf_counter(), []
    choice, candidates = select_pair(by_trial)
    selection = dict(seed_preference=list(SEED_PREFERENCE), candidates=candidates, chosen=choice,
                     criterion=CRITERION, exclusive_object_selection_required=True,
                     baseline_summary_sha256=sha(baseline / 'summary.json'),
                     script_sha256=sha(Path(__file__)), scope=SCOPE)
    write_json(out / 'pair_selection.json', selection)
    if choice is None:
        write_json(out / 'blocked.json', dict(state='blocked', reason='No paired seed has strict milk-only x06 and cheese-only x15 selection.',
                                              **selection))
        print('[COMPONENT-PULSES] BLOCKED: no qualifying paired baseline seed; no model loaded.', flush=True)
        return
    runtime, hooks = None, None

    def progress(state, run=None, **extra):
        value = dict(state=state, completed_predictions=len(completed), total_predictions=60,
                     completed_runs=completed, current_run=run, elapsed_s=time.perf_counter() - started, **extra)
        write_json(out / 'progress.json', value)
        print('[COMPONENT-PULSES] ' + json.dumps(value, allow_nan=False), flush=True)

    try:
        os.environ['HF_HUB_OFFLINE'] = '1'
        paths = dict(normal_runtime=ROOT / 'work/serve_cosmos_goal_pair.py',
                     factory=ROOT / 'work/probe_cosmos_language_routes.py',
                     components=ROOT / 'work/cosmos_component_interventions.py')
        provenance = read_json(baseline / 'server/provenance.json')
        for key, field in (('normal_runtime', 'normal_runtime_sha256'), ('factory', 'factory_sha256'),
                           ('components', 'components_sha256')):
            assert sha(paths[key]) == provenance[field], f'Baseline {key} source changed'
        base = load_file('pulse_normal_runtime', paths['normal_runtime'])
        base.TASKS['milk'] = (7, 'pick_up_the_milk_and_place_it_in_the_basket', PROMPT, 'milk_1')
        factory = load_file('pulse_checked_factory', paths['factory'])
        factory.OUT = out
        assert sha(factory.SCHEDULER) == provenance['scheduler_sha256'], 'Baseline scheduler changed'
        progress('loading_model')
        runtime = base.NormalRuntime(factory)
        torch = runtime.torch
        module = load_file('pulse_component_hooks', paths['components'])
        hooks = module.CosmosComponentInterventions(runtime.pipe.transformer)
        assert len(hooks.layers) == 36 and not runtime.pipe.transformer.training
        import inspect
        transformer_source = Path(inspect.getfile(type(runtime.pipe.transformer)))
        write_json(out / 'provenance.json', dict(
            script_sha256=sha(Path(__file__)), source_sha256={key: sha(path) for key, path in paths.items()},
            actual_transformer_source=str(transformer_source), actual_transformer_sha256=sha(transformer_source),
            scheduler_sha256=sha(factory.SCHEDULER), baseline_server_provenance=provenance,
            baseline_summary_sha256=sha(baseline / 'summary.json'), prompt=PROMPT,
            seed=choice['seed'], donor_trial=choice['donor_trial'], recipient_trial=choice['recipient_trial'],
            layer_convention='zero based', denoising_step_convention='zero based model forward in q0',
            layers=list(LAYERS), components=list(COMPONENTS), pulse_steps=list(PULSE_STEPS), gain=1,
            patch_site='GEN action component increment before its residual addition; exactly 16 actual action rows',
            library_edits=False, closed_loop_execution=False, scan_predictions=56, control_predictions=4,
            projection_definition='sum((P-R)*(D-R))/sum((D-R)^2), final normalized action XYZ; 0 is recipient, 1 donor along this axis',
            gap_abs_floor=GAP_ABS_FLOOR, gap_relative_floor=GAP_RELATIVE_FLOOR, scope=SCOPE))
        records, caches, sources = {}, {}, {}
        for role, trial in (('recipient', choice['recipient_trial']), ('donor', choice['donor_trial'])):
            run = baseline / 'closed-loop' / trial
            state_path, cache_path = run / 'chunk_00/states.pt', run / 'chunk_00/components.pt'
            records[role] = torch.load(state_path, map_location='cpu', weights_only=True, mmap=True)
            caches[role] = torch.load(cache_path, map_location='cpu', weights_only=True, mmap=True)
            validate_record(records[role], torch)
            rows = validate_cache(caches[role], records[role], hooks, runtime, role)
            meta = read_json(run / 'chunk_00/metadata.json')
            assert meta['trial'] == trial and meta['query'] == 0 and meta['seed'] == choice['seed']
            assert meta['prompt'] == PROMPT and meta['model_calls'] == 30 and meta['component_capture'] is True
            assert meta['input_png_sha256'] == sha(run / 'input_00.png')
            runtime.require_exact(records[role]['actions'].tolist(), read_json(run / 'chunk_00/normalized_actions.json'), f'{role}/saved_actions')
            sources[role] = dict(trial=trial, image=str(run / 'input_00.png'), image_sha256=sha(run / 'input_00.png'),
                states_path=str(state_path), states_sha256=sha(state_path), cache_path=str(cache_path),
                cache_sha256=sha(cache_path), metadata_sha256=sha(run / 'chunk_00/metadata.json'),
                action_rows=rows.tolist(), component_frames=len(caches[role]['frames']),
                pure_noise=[tensor_meta(value, torch) for value in records[role]['pure_noise']],
                timesteps=tensor_meta(records[role]['timesteps'], torch), sigmas=tensor_meta(records[role]['sigmas'], torch))
        recipient, donor = records['recipient'], records['donor']
        r_cache, d_cache = caches['recipient'], caches['donor']
        for key in ('pure_noise', 'timesteps', 'sigmas'):
            runtime.require_exact(recipient[key], donor[key], f'paired_complete_{key}')
        runtime.require_exact(r_cache['steps'], d_cache['steps'], 'paired_complete_30step_pack_metadata')
        runtime.require_exact(recipient['model_input']['action_tokens'], donor['model_input']['action_tokens'], 'paired_prepared_action_noise')
        rows = r_cache['steps'][0]['layout']['action_rows']
        contract = dict(state='paired_sources_verified', sources=sources, complete_nine_baseline_summaries=True,
                        both_fp32_random_draws_exact=True, action_noise_and_all_30_pack_metadata_exact=True,
                        timesteps_and_sigmas_exact=True, caches_complete_and_len_steps_30=True,
                        first_kwargs_match_runtime=True, controls=[], scope=SCOPE)
        torch.save(dict(recipient_pure_noise=recipient['pure_noise'], donor_pure_noise=donor['pure_noise'],
                        timesteps=recipient['timesteps'], sigmas=recipient['sigmas'], action_rows=rows,
                        recipient_step_metadata=r_cache['steps'], donor_step_metadata=d_cache['steps']),
                   out / 'pair_contract.pt')
        write_json(out / 'validation.json', contract)
        (out / 'controls').mkdir(exist_ok=False)
        (out / 'runs').mkdir(exist_ok=False)

        # Remove component observers entirely, then demand equality of every
        # saved NormalRuntime field (including video, states and prepared input).
        for role in ('recipient', 'donor'):
            label, destination = f'{role}_observer_free_repeat', out / 'controls' / f'{role}_observer_free_repeat'
            hooks.reset()
            assert not hooks._handles
            progress('control_running', label)
            repeated, _ = runtime.predict('milk', Path(sources[role]['image']), choice['seed'], destination)
            runtime.require_exact(repeated, records[role], f'{role}/observer_free_full_record')
            completed.append(label)
            contract['controls'].append(dict(run=label, all_normal_runtime_fields_bit_exact=True,
                                             states_sha256=sha(destination / 'states.pt')))
            write_json(out / 'validation.json', contract)
            progress('control_passed', label)

        def patched_prediction(label, destination, layer, component, step, source_cache):
            progress('patch_running', label)
            hooks.begin_patch(r_cache, layer=layer, component=component, steps=[step], gain=1, donor_cache=source_cache)
            try:
                record, metadata = runtime.predict('milk', Path(sources['recipient']['image']), choice['seed'], destination)
            finally:
                report = hooks.end()
                if destination.exists():
                    torch.save(report, destination / 'patch_report.pt')
                    write_json(destination / 'patch_report.json', report_metadata(report, torch))
            validate_run_contract(record, recipient, runtime, label)
            validate_patch_report(report, r_cache, source_cache, rows, layer, component, step, runtime)
            for key in ('readout', 'action_velocity'):
                runtime.require_exact(record[key][:step], recipient[key][:step], f'{label}/native_before_pulse/{key}')
            runtime.require_exact(record['action_states'][:step + 1], recipient['action_states'][:step + 1],
                                  f'{label}/native_solver_states_before_pulse')
            return record, metadata, report

        for component in COMPONENTS:
            label = f'self_layer17_{component}_step00'
            destination = out / 'controls' / label
            record, _, report = patched_prediction(label, destination, 17, component, 0, r_cache)
            runtime.require_exact(record, recipient, f'{label}/full_record')
            assert report['changes'][(0, 17, component)]['exact_no_change'] is True
            completed.append(label)
            contract['controls'].append(dict(run=label, all_normal_runtime_fields_bit_exact=True,
                                             patch_report_sha256=sha(destination / 'patch_report.pt'),
                                             states_sha256=sha(destination / 'states.pt')))
            write_json(out / 'validation.json', contract)
            progress('control_passed', label)
        contract['state'] = 'all_exact_controls_passed'
        write_json(out / 'validation.json', contract)
        results = []
        for layer in LAYERS:
            for component in COMPONENTS:
                for step in PULSE_STEPS:
                    label = f'layer{layer:02d}_{component}_step{step:02d}'
                    destination = out / 'runs' / label
                    record, metadata, report = patched_prediction(label, destination, layer, component, step, d_cache)
                    metrics = prediction_metrics(recipient, donor, record)
                    write_json(destination / 'metrics.json', metrics)
                    row = dict(run=label, layer=layer, component=component, pulse_step=step,
                        pulse_timestep=int(recipient['timesteps'][step]), pulse_sigma=float(recipient['sigmas'][step]),
                        gain=1, donor_trial=choice['donor_trial'], recipient_trial=choice['recipient_trial'],
                        intervention_count=1, component_delta_l2=report['changes'][(step, layer, component)]['delta_l2'],
                        final_normalized_action_xyz=metrics['final_normalized_action_xyz'],
                        final_normalized_action_all10=metrics['final_normalized_action_all10'],
                        metrics_path=str(destination / 'metrics.json'), states_path=str(destination / 'states.pt'),
                        states_sha256=sha(destination / 'states.pt'), patch_report_sha256=sha(destination / 'patch_report.pt'),
                        runtime_metadata_sha256=sha(destination / 'metadata.json'), elapsed_s=metadata['elapsed_s'])
                    results.append(row)
                    completed.append(label)
                    write_json(out / 'results.json', dict(state='running', completed_scan_predictions=len(results),
                                                         expected_scan_predictions=56, cases=results, scope=SCOPE))
                    progress('patch_complete', label, completed_scan_predictions=len(results),
                             xyz_projection=metrics['final_normalized_action_xyz']['signed_projection_fraction'])
        assert len(results) == 56 and len(completed) == 60
        ranked = sorted((row for row in results
                         if row['final_normalized_action_xyz']['signed_projection_fraction'] is not None),
                        key=lambda row: row['final_normalized_action_xyz']['signed_projection_fraction'], reverse=True)
        final = dict(state='complete', paired_seed=choice['seed'], cases=results,
            ranking_metric='final_normalized_action_xyz.signed_projection_fraction', ranked_runs=[row['run'] for row in ranked],
            top_three_candidates=ranked[:3], unrankable_runs=[row['run'] for row in results if row not in ranked],
            controls=contract['controls'], scan_predictions=56, control_predictions=4,
            interpretation=('Positive signed XYZ projection means the fixed-input normalized action output moved toward '
                            'the donor along one continuous axis. This does not establish a correct grasp or pure identity. '
                            'If q0 baseline gaps are too small, projection is null; inspect later saved query inputs.'),
            next_validation=('Locate the first actual approach divergence, then test leading components on matched saved '
                             'inputs at that query with real MuJoCo execution, reverse replacements, equal-norm random '
                             'perturbations, independent seeds and moved-object layouts.'), scope=SCOPE)
        write_json(out / 'results.json', final)
        write_json(out / 'top_candidates.json', dict(state='candidates_for_followup_only',
            ranking_metric=final['ranking_metric'], candidates=ranked[:3], interpretation=final['interpretation'],
            next_validation=final['next_validation'], scope=SCOPE))
        write_json(out / 'complete.json', dict(state='complete', paired_seed=choice['seed'],
            scan_predictions=56, control_predictions=4, all_noise_schedule_pack_checks_exact=True,
            all_repeat_and_selfpatch_fields_bit_exact=True, single_component_pulse_per_scan_prediction=True,
            results_sha256=sha(out / 'results.json'), validation_sha256=sha(out / 'validation.json'),
            script_sha256=sha(Path(__file__)), scope=SCOPE))
        progress('complete', ranked_candidates=len(ranked))
    except Exception as exc:
        write_json(out / 'failed.json', dict(state='failed', type=type(exc).__name__, error=str(exc),
            traceback=traceback.format_exc(), completed_runs=completed, completed_predictions=len(completed),
            elapsed_s=time.perf_counter() - started, scope=SCOPE))
        raise
    finally:
        if hooks is not None:
            hooks.reset()


if __name__ == '__main__':
    main()
