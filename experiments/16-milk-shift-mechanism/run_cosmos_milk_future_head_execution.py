"""Execute all eighteen preregistered X12 query-head-group q0 cases.

Use --output BASE --mode prepare|server|simulate. Exclusively create absent
BASE/future-head-execution after the unique producer and prior value stage
freeze. Cached q0 then native actual feedback at seeds196..202: 144 requests,
18 cached,126 fresh predictions,3780 later forwards,119 whole-noise pairs;
producer540 plus later3780 equals4320. Execute every 128-action trajectory.

Reuse frozen e900 input checks, simulator execute_trials, NormalRuntime,
controller, converter and strict five-record contact/lift calculation.
Eight control cases must reproduce the accepted value-stage whole records,
all129 states,128 actions and eight images. The sole primary is both head_G6
cases selecting strictly only milk. G4,leave complements and independent
random controls are fully reported without replacing that prediction.
G6 means layer0 query heads24..27 at steps0..14. Leave-group conditions
retain those four layer0 heads within the accepted nine-layer full cut.
Conditional sufficiency and leave-group effects concern X12 only.
"""

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer


ROOT = Path('/home/current/work/cosmos3')
VALUE_SHA = 'e900f970e4049f6db1755274d68fa38366a9a29916354ccc8357090c44165a51'
PROBE_SHA = 'b261366b969a8f33980cbf4d450c381a18a64fa4b16be2e1dd95d6c7220ae898'
PORT = 8936
PROMPT = 'pick up the milk and place it in the basket'
ARMS = ('native', 'all_allowed', 'full_hardmask', 'head_G4', 'head_G6', 'leave_G4', 'leave_G6', 'arithmetic_sham_G6', 'random_G6')
PLAN = tuple((f'V{v}_A195_{arm}', v, arm) for arm in ARMS for v in (195, 198))
FULL_WINDOW = dict(steps=[0, 14], layers=[0, 8])
LOCAL_WINDOW = dict(steps=[0, 14], layers=[0, 0])
PRIMARY_CASES = ('V195_A195_head_G6', 'V198_A195_head_G6')
REFERENCE_ARMS = dict(native='native', all_allowed='all_allowed', full_hardmask='hardmask', arithmetic_sham_G6='all_allowed')
FILES = ('states.pt', 'normalized_actions.json', 'metadata.json', 'noise-audit.pt', 'noise-audit.json',
         'observer-report.json', 'indexes-and-masks.pt', 'control-checks.json', 'site-captures.json')
PHYSICAL = ('trajectory.json', 'actions.json', 'executed_actions.json', 'normalized_actions.json', 'denormalized_actions.json')
SCOPE = ('All eighteen fixed X12 trajectories using saved q0 followed by native actual feedback. '
         'Only both head_G6 strict only-milk cases test the primary; G4/leave/random results cannot replace it. '
         'G6 is layer0 query heads24..27 at steps0..14, not four heads throughout the network. '
         'Full and leave cuts span layers0..8; leave retains only the specified layer0 group. '
         'Sufficiency and leave-group effects are conditional on this numerical background and scene. '
         'A matching random effect does not support group-specific causation; a failed random strength '
         'gate prevents that interpretation. No single-head,identity,functional brain region, '
         'cross-position repair,training,additive causal percentage or universal necessity claim.')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


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


def arm_window(arm):
    if arm in ('native', 'all_allowed'):
        return None
    return FULL_WINDOW if arm in ('full_hardmask', 'leave_G4', 'leave_G6') else LOCAL_WINDOW


def contract(baseline):
    assert isinstance(PROBE_SHA, str) and len(PROBE_SHA) == 64 and all(c in '0123456789abcdef' for c in PROBE_SHA), 'Producer SHA PENDING; no output or execution is allowed.'
    value_path = ROOT / 'work/run_cosmos_milk_future_value_execution.py'
    probe_path = ROOT / 'work/probe_cosmos_milk_future_head_groups.py'
    assert sha(value_path) == VALUE_SHA and sha(probe_path) == PROBE_SHA
    value = load('head_execution_frozen_value_adapter', value_path)
    simulator, prior, cross, helper, old_out = value.prepared(baseline)
    old_complete = read(old_out / 'complete.json')
    assert old_complete['state'] == 'complete' and old_complete['script_sha256'] == VALUE_SHA
    for name, expected in old_complete['files_sha256'].items():
        assert sha(old_out / name) == expected
    old_controls, old_server = read(old_out / 'controls.json'), read(old_out / 'server/complete.json')
    for name, expected in old_server['files_sha256'].items():
        assert sha(old_out / 'server' / name) == expected
    assert old_complete['eight_entire129_state_5JSON_8PNG_and_all8_modelrecords_controls_exact'] is True
    probe = load('head_execution_frozen_head_producer', probe_path)
    context = probe.source_contract(baseline)
    assert probe.PLAN == PLAN and probe.ARMS == ARMS and tuple(probe.FILES) == FILES
    assert context['frozen_contract'] == prior['threshold_contract']
    producer = baseline / 'future-head-groups'
    assert context['out'] == producer and not (producer / 'failed.json').exists()
    docs = {name: read(producer / (name + '.json')) for name in ('complete', 'results', 'protocol', 'provenance', 'sources', 'primary-prediction')}
    complete, results, protocol, provenance = (docs[name] for name in ('complete', 'results', 'protocol', 'provenance'))
    assert complete['state'] == results['state'] == 'complete'
    assert complete['script_sha256'] == provenance['script_sha256'] == PROBE_SHA
    for name in ('results', 'protocol', 'provenance', 'sources', 'primary-prediction'):
        assert complete[name.replace('-', '_') + '_sha256'] == sha(producer / (name + '.json'))
    assert complete['fresh_q0_predictions'] == results['fresh_q0_predictions'] == 18
    assert complete['fresh_model_forwards'] == results['fresh_model_forwards'] == 540
    counts = probe.expected_counts()
    assert results['official_dispatch_counts'] == complete['official_dispatch_counts'] == protocol['expected_counts'] == counts
    assert counts['model_before'] == counts['model_after'] == 540
    assert [counts[k] for k in probe.DISPATCH_KEYS] == [19440, 19440, 17280, 30, 0, 900]
    assert sum(counts[k] for k in probe.DISPATCH_KEYS) == complete['total_official_dispatch_calls'] == results['total_official_dispatch_calls'] == 57090
    assert complete['original_randn_calls_consumed'] == complete['returned_source_draws_observed'] == 36
    assert complete['actual_selected_site_captures'] == 2430
    for key in ('two_native_and_two_allallowed_entire_accepted_value_records_and_boundaries_exact',
                'two_full_hardmask_entire_accepted_value_hardmask_records_and_boundaries_exact',
                'two_arithmetic_shams_entire_accepted_value_allallowed_records_and_boundaries_exact',
                'all16_masked_sites_same_union66_background', 'all540_actual_complete_model_kwargs_saved',
                'all2430_actual_BF16_projection_MLP_residual_chains_byte_exact',
                'random15_cross_V_independent_directions_byte_exact', 'random_control_actual_BF16_norm_checks_reported',
                'all_original_symbols_restored', 'all_owned_hooks_removed'):
        assert complete[key] is True
    assert complete['random_relative_norm_tolerance'] == probe.RANDOM_RELATIVE_NORM_TOLERANCE == 0.10
    assert complete['random_tolerance_is_interpretation_only'] is True
    assert complete['physics_calls'] == complete['later_query_predictions'] == 0
    assert complete['primary_physical_prediction_evaluated'] is results['primary_physical_prediction_evaluated'] is False
    assert protocol['trial_order'] == results['preregistered_physics_cases'] == [label for label, _, _ in PLAN]
    assert protocol['plan'] == [dict(case=label, vision_noise_source_seed=v, action_noise_source_seed=195,
        runtime_rng_seed=195, arm=arm, window=arm_window(arm), selected_query_heads=probe.selected_heads(arm),
        control_reference_case=probe.control_reference_case(v, arm)) for label, v, arm in PLAN]
    assert protocol['later_physics_preregistered'] == dict(trial_order=[label for label, _, _ in PLAN], cases=18,
        cached_q0=18, later_noise_base=195, q1_to_q7_seeds=list(range(196, 203)), later_predictions=126,
        later_model_forwards=3780, total_two_stage_model_forwards=4320, no_q0_score_or_rank_selection=True)
    assert protocol['prompt'] == PROMPT and protocol['shift_cm'] == 12 and protocol['image_sha256'] == sha(context['image'])
    assert context['image'].read_bytes() == (baseline / 'position-threshold/inputs/x12/input.png').read_bytes()
    assert probe.PRIMARY['primary_arm'] == 'head_G6' and probe.PRIMARY['primary_cases'] == list(PRIMARY_CASES)
    assert protocol['primary_hypothesis'] == probe.PRIMARY
    assert results['primary_prediction_sha256'] == sha(producer / 'primary-prediction.json')
    assert docs['primary-prediction'] == dict(status='not_evaluated_q0_only', **probe.PRIMARY, scope=probe.SCOPE)
    assert provenance['actual_transformer_sha256'] == prior['actual_transformer_sha256']
    assert provenance['actual_pipeline_source_sha256'] == prior['actual_pipeline_source_sha256']
    assert provenance['source_sha256'] == {key: sha(path) for key, path in context['paths'].items()}
    assert docs['sources'] == dict(source_evidence=context['source_evidence'],
        original_sources=context['original_noise_sources'], frozen_contract=context['frozen_contract'])
    np, original = simulator.np, prior['threshold_contract']
    stats = read(Path(original['frozen_files']['normalizer_stats']['path']))['global_raw']
    lo, hi = np.asarray(stats['q01']), np.asarray(stats['q99'])
    assert lo.shape == hi.shape == (10,) and np.all(hi - lo > 1e-6)
    scale, offset = np.maximum(hi - lo, 1e-8) / 2, (hi + lo) / 2
    assert [row['case'] for row in results['cases']] == [label for label, _, _ in PLAN]
    q0, rules, references = {}, {}, {}
    for row, (label, v, arm) in zip(results['cases'], PLAN):
        folder, window = producer / label, arm_window(arm)
        assert probe.arm_window(arm) == window
        assert row['arm'] == arm and row['window'] == window and row['vision_noise_source_seed'] == v
        assert row['action_noise_source_seed'] == row['runtime_rng_seed'] == 195 and row['fresh_model_forwards'] == 30
        assert set(row['files_sha256']) == set(FILES)
        for name, expected in row['files_sha256'].items():
            assert sha(folder / name) == expected
        report, metadata = read(folder / 'observer-report.json'), read(folder / 'metadata.json')
        assert report['state'] == 'complete' and report['case_arm'] == arm and report['selected_window'] == window
        assert report['counts'] == row['actual_dispatch_counts'] == probe.expected_counts(arm)
        assert report['indexes_and_masks_sha256'] == row['files_sha256']['indexes-and-masks.pt']
        assert report['site_capture_manifest_sha256'] == row['site_capture_manifest_sha256']
        for key in ('all_owned_hooks_removed', 'original_dispatch_symbol_restored', 'all30_actual_complete_model_kwargs_saved',
                    'all30_actual37_current_action_boundaries_finite', 'future200_and_UND_native_dispatch_returns_preserved'):
            assert report[key] is True
        assert report['original_full_266_row_projection_calls'] == 1080 and report['processor_recompute_calls'] == 0
        assert report['all_masked_sites_same_union66'] is (arm != 'native')
        assert report['all_actual_dispatch_flatten_pre_W_byte_exact'] is report['action16_same_input_Z0_at_all_masked_sites'] is True
        assert report['all135_actual_BF16_projection_MLP_residual_chains_byte_exact'] is True
        assert report['captured_sites'] == 135 and report['extra_projection_or_MLP_forward_calls'] == 0
        if arm == 'arithmetic_sham_G6':
            assert report['arithmetic_sham_all_selected_sites_byte_exact_Z0'] is True
        if probe.expected_counts(arm)['cut_dispatch']:
            assert report['hardmask_unblocked_queries_byte_exact_Z0'] is True
        assert len(row['boundary_files']) == 30 and report['boundary_files'] == row['boundary_files']
        for step, item in enumerate(row['boundary_files']):
            assert item['step'] == step and item['file'] == f'boundaries-t{step:02d}.pt'
            assert item['current_shape'] == [37, 50, 4096] and item['action_shape'] == [37, 16, 4096]
            assert sha(folder / item['file']) == item['sha256']
        manifest = read(folder / 'site-captures.json')
        assert row['site_capture_manifest_sha256'] == row['files_sha256']['site-captures.json']
        assert manifest['state'] == 'complete' and manifest['sites'] == len(manifest['files']) == 135
        assert manifest['random_control'] == row['random_control']
        assert [(s['step'], s['layer_zero_based']) for s in manifest['files']] == [(t, l) for t in range(15) for l in range(9)]
        for item in manifest['files']:
            path = folder / item['file']
            assert sha(path) == item['sha256'] and path.stat().st_size == item['bytes']
        assert metadata['case'] == label and metadata['arm'] == arm
        assert metadata['seed'] == metadata['action_noise_source_seed'] == metadata['runtime_rng_seed'] == 195
        assert metadata['vision_noise_source_seed'] == v and metadata['model_calls'] == 30 and metadata['prepare_calls'] == 1
        assert metadata['input_png_sha256'] == protocol['image_sha256']
        norm = np.asarray(read(folder / 'normalized_actions.json'))
        assert norm.shape == (16, 10) and norm.dtype == np.float64 and np.isfinite(norm).all()
        assert norm.tolist() == row['final_normalized_actions']
        assert report['random_control'] == row['random_control']
        random_control = row['random_control']
        assert random_control['relative_norm_tolerance'] == 0.10
        assert random_control['tolerance_is_interpretation_only'] is random_control['failure_does_not_recalibrate_or_select_cases'] is True
        assert random_control['sites'] == len(random_control['checks']) == (15 if arm == 'random_G6' else 0)
        if arm == 'random_G6':
            assert [(item['step'], item['layer_zero_based']) for item in random_control['checks']] == [(t, 0) for t in range(15)]
            for item in random_control['checks']:
                assert item['seed'] == 901000 + item['step'] * 36 and item['relative_norm_tolerance'] == 0.10
                assert item['finite'] is item['independent_CPU_generator'] is item['no_post_BF16_rescaling'] is True
                target, actual = item['target_Dk_actual_BF16_l2'], item['random_actual_BF16_l2']
                assert np.isfinite(target) and np.isfinite(actual) and target >= 0 and actual >= 0
                error = 0.0 if target == actual == 0 else abs(actual / target - 1.0)
                assert item['actual_BF16_relative_norm_error'] == error
                assert item['actual_BF16_norm_ratio'] == (None if target == 0 else actual / target)
                assert item['actual_BF16_equal_strength_within_tolerance'] is (error <= 0.10)
            assert random_control['actual_BF16_equal_strength_all_sites_within_tolerance'] is all(
                item['actual_BF16_equal_strength_within_tolerance'] for item in random_control['checks'])
        else:
            assert random_control['actual_BF16_equal_strength_all_sites_within_tolerance'] is None
        reference_case = f'V{v}_A195_{REFERENCE_ARMS[arm]}' if arm in REFERENCE_ARMS else None
        assert row['control_reference_case'] == reference_case
        assert row['existing_entire_control_record_exact'] is (reference_case is not None)
        if reference_case is not None:
            ref = old_out / 'closed-loop' / reference_case
            files = old_controls['actual_case_files_sha256'][reference_case]
            for name, expected in files.items():
                assert sha(ref / name) == expected
            references[label] = dict(folder=str(ref), reference_case=reference_case, files_current_sha256=files,
                controller_step16_path=str(ref / 'controller-step16.npz'), controller_step16_sha256=files['controller-step16.npz'])
        q0[label] = dict(folder=str(folder), files_sha256=row['files_sha256'], boundary_files=row['boundary_files'],
            normalized_actions=norm.tolist(), raw_actions=(norm * scale + offset).tolist(), arm=arm, selected_window=window,
            masked_attention=arm != 'native', vision_noise_source_seed=v, action_noise_source_seed=195,
            site_capture_manifest_sha256=row['site_capture_manifest_sha256'], random_control=row['random_control'])
        rules[label] = dict(arm=arm, selected_window=window, q0_vision_noise_source_seed=v, q0_action_noise_source_seed=195,
            whole_reference_control=reference_case is not None, whole_reference_case=reference_case,
            actual_query_seeds=list(range(195, 203)), later_noise_base=195, q0_whole_noise_pairing=False,
            seed_rule='Cached q0 has separate V/A sources and invocation195; q1..7=195+query.')
    frozen = dict(script_sha256=sha(Path(__file__)), value_execution_sha256=VALUE_SHA, producer_script_sha256=PROBE_SHA,
        threshold_contract=original, actual_transformer_sha256=prior['actual_transformer_sha256'],
        actual_pipeline_source_sha256=prior['actual_pipeline_source_sha256'], metrics_path=prior['metrics_path'], components_path=prior['components_path'],
        producer_source_cases=prior['producer_source_cases'], native_q0_source_pair=prior['native_q0_source_pair'],
        prior_value_complete_sha256=sha(old_out / 'complete.json'), prior_value_documents_sha256=old_complete['files_sha256'],
        producer_documents_sha256={name: sha(producer / (name + '.json')) for name in docs}, producer_protocol=protocol,
        source_evidence=context['source_evidence'], primary_hypothesis=probe.PRIMARY, primary_cases=list(PRIMARY_CASES),
        q0_files=list(FILES), q0_sources=q0, trials=rules, trial_order=[label for label, _, _ in PLAN], whole_physical_references=references,
        inputs_symlink_target=str(baseline / 'position-threshold/inputs'), url=f'http://127.0.0.1:{PORT}',
        saved_q0_predictions=18, requests=144, native_predictions=126, fresh_model_forwards=3780,
        producer_q0_forwards_separate=540, combined_two_stage_model_forwards=4320, actual_later_whole_noise_pairs_expected=119,
        q0_whole_noise_pair_checks=0, all18_execute_without_score_selection=True, scope=SCOPE)
    simulator.SHIFTS, simulator.SEEDS = dict(x00=0, x09=9, x12=12), (195,)
    simulator.TRIALS = {label: ('x12', cross.QuerySeed(195, 195)) for label, _, _ in PLAN}
    simulator.URL, simulator.SCOPE = frozen['url'], SCOPE
    return simulator, frozen, cross, helper, value


def prepared(baseline):
    simulator, frozen, cross, helper, value = contract(baseline)
    out = baseline / 'future-head-execution'
    assert out.is_dir() and not (out / 'wrapper_failed.json').exists()
    assert (out / 'inputs').is_symlink() and (out / 'inputs').resolve() == Path(frozen['inputs_symlink_target']).resolve()
    assert read(out / 'provenance.json') == frozen
    assert read(out / 'prepared.json') == dict(state='prepared', provenance_sha256=sha(out / 'provenance.json'))
    return simulator, frozen, cross, helper, value, out


def server(baseline):
    _, frozen, _, helper, value, out = prepared(baseline)
    destination = out / 'server'
    destination.mkdir(exist_ok=False)
    served, later_noise, reference_queries, pairs, cached, fresh = [], {}, [], 0, 0, 0
    http, failed, started = None, False, time.perf_counter()
    try:
        os.environ['HF_HUB_OFFLINE'] = '1'
        paths = frozen['threshold_contract']['frozen_files']
        base = load('head_execution_runtime', Path(paths['normal_runtime']['path']))
        base.TASKS['milk'] = (7, 'pick_up_the_milk_and_place_it_in_the_basket', PROMPT, 'milk_1')
        factory = load('head_execution_factory', Path(paths['factory']['path']))
        factory.OUT = destination
        assert sha(factory.SCHEDULER) == paths['scheduler']['sha256']
        runtime = base.NormalRuntime(factory)
        torch, model = runtime.torch, runtime.pipe.transformer
        assert not model.training and not model.is_cache_enabled and not torch.is_grad_enabled()
        assert sha(Path(inspect.getfile(type(model)))) == frozen['actual_transformer_sha256']
        assert sha(Path(inspect.getfile(type(runtime.pipe)))) == frozen['actual_pipeline_source_sha256']
        initial_hooks = value.hooks(model)
        assert all(not pre and not post for _, pre, post in initial_hooks)
        scan = load('head_execution_scan', Path(frozen['metrics_path']))
        components = load('head_execution_actual_layout', Path(frozen['components_path']))
        originals, native_pair, q0, checks = {}, {}, {}, []
        for v in (195, 198):
            source = frozen['producer_source_cases'][str(v)]
            folder = Path(source['q0_folder'])
            assert sha(folder / 'states.pt') == source['q0_files_current_sha256']['states.pt']
            originals[v] = torch.load(folder / 'states.pt', map_location='cpu', weights_only=True, mmap=True)
            source = frozen['native_q0_source_pair'][str(v)]
            folder = Path(source['folder'])
            assert sha(folder / 'states.pt') == source['files_sha256']['states.pt']
            native_pair[v] = torch.load(folder / 'states.pt', map_location='cpu', weights_only=True, mmap=True)
        for label, v, arm in PLAN:
            source, rule = frozen['q0_sources'][label], frozen['trials'][label]
            folder = Path(source['folder'])
            record = torch.load(folder / 'states.pt', map_location='cpu', weights_only=True, mmap=True)
            audit = torch.load(folder / 'noise-audit.pt', map_location='cpu', weights_only=True, mmap=True)
            scan.validate_record(record, torch)
            helper.verify_cached(record, audit, originals, native_pair, arm, v, runtime, scan)
            value.verify_full_inputs(record, audit, source, runtime, components)
            assert record['actions'].tolist() == source['normalized_actions']
            q0[label] = record
            if rule['whole_reference_control']:
                ref = frozen['whole_physical_references'][label]
                path = Path(ref['folder']) / 'chunk_00/states.pt'
                assert sha(path) == ref['files_current_sha256']['chunk_00/states.pt']
                runtime.require_exact(record, torch.load(path, map_location='cpu', weights_only=True, mmap=True), 'ENTIRE prior value-stage q0/' + label)
            checks.append(dict(case=label, arm=arm, actual_noise_preparation_full_kwargs_all30_solver_casts_exact=True,
                entire_prior_q0_exact=True if rule['whole_reference_control'] else None, reference_case=rule['whole_reference_case'],
                q0_model_forwards_in_this_execution=0, producer_files_sha256=source['files_sha256']))
        write(destination / 'saved-q0-controls.json', dict(state='exact', cases=checks, eight_entire_prior_q0_controls_exact=True,
            extra_q0_model_forwards=0, q0_whole_noise_pair_checks=0), exclusive=True)
        write(destination / 'provenance.json', dict(script_sha256=sha(Path(__file__)), execution_provenance_sha256=sha(out / 'provenance.json'),
            saved_q0_controls_sha256=sha(destination / 'saved-q0-controls.json'), component_hooks_installed=False,
            attention_dispatch_wrapper_installed=False, scope=SCOPE), exclusive=True)

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                assert self.path == '/health'
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'ready')

            def do_POST(self):
                nonlocal failed, pairs, cached, fresh
                try:
                    assert self.path == '/predict'
                    request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                    assert set(request) == {'trial', 'query', 'seed', 'prompt'}
                    label, query = request['trial'], request['query']
                    assert type(query) is int and 0 <= query < 8
                    assert label == frozen['trial_order'][len(served) // 8] and query == len(served) % 8 and (label, query) not in served
                    rule, source = frozen['trials'][label], frozen['q0_sources'][label]
                    seed = rule['actual_query_seeds'][query]
                    assert request['seed'] == seed and request['prompt'] == PROMPT
                    image = out / 'closed-loop' / label / f'input_{query:02d}.png'
                    chunk = image.parent / f'chunk_{query:02d}'
                    if query == 0:
                        assert image.read_bytes() == (out / 'inputs/x12/input.png').read_bytes()
                        chunk.mkdir(exist_ok=False)
                        folder = Path(source['folder'])
                        for name in FILES:
                            assert sha(folder / name) == source['files_sha256'][name]
                            if name != 'metadata.json':
                                shutil.copyfile(folder / name, chunk / name)
                        for item in source['boundary_files']:
                            assert sha(folder / item['file']) == item['sha256']
                            shutil.copyfile(folder / item['file'], chunk / item['file'])
                        record, metadata = q0[label], read(folder / 'metadata.json')
                        cached, paired = cached + 1, False
                    else:
                        record, metadata = runtime.predict('milk', image, seed, chunk)
                        fresh += 1
                        paired = query in later_noise
                        if paired:
                            runtime.require_exact(record['pure_noise'], later_noise[query], 'actual later whole two-draw pair/' + str(query))
                            pairs += 1
                        else:
                            later_noise[query] = record['pure_noise']
                    scan.validate_record(record, torch)
                    for key in ('timesteps', 'sigmas'):
                        runtime.require_exact(record[key], originals[195][key], 'actual complete schedule/' + key)
                    assert value.hooks(model) == initial_hooks
                    if rule['whole_reference_control']:
                        ref = frozen['whole_physical_references'][label]
                        folder, name = Path(ref['folder']), f'chunk_{query:02d}/states.pt'
                        assert sha(folder / name) == ref['files_current_sha256'][name]
                        assert image.read_bytes() == (folder / f'input_{query:02d}.png').read_bytes()
                        runtime.require_exact(record, torch.load(folder / name, map_location='cpu', weights_only=True, mmap=True), 'ENTIRE prior control query/' + label + '/' + str(query))
                        reference_queries.append(dict(case=label, query=query, reference_case=rule['whole_reference_case'],
                            entire_NormalRuntime_record_bytes_exact=True, actual_feedback_PNG_bytes_exact=True,
                            reference_states_sha256=sha(folder / name), current_states_sha256=sha(chunk / 'states.pt')))
                    assert metadata['seed'] == seed and metadata['model_calls'] == 30 and metadata['prepare_calls'] == 1
                    metadata.update(trial=label, query=query, shift_cm=12, image_path=str(image), input_png_sha256=sha(image),
                        saved_q0_prediction=query == 0, current_execution_model_calls=0 if query == 0 else 30,
                        source_record_model_calls=30 if query == 0 else None, saved_q0_attention_arm=rule['arm'],
                        saved_q0_window=rule['selected_window'], q0_vision_noise_source_seed=rule['q0_vision_noise_source_seed'],
                        q0_action_noise_source_seed=195, q0_runtime_rng_seed=195, later_noise_base=195,
                        actual_query_seeds=rule['actual_query_seeds'], seed_rule=rule['seed_rule'], whole_prior_control=rule['whole_reference_control'],
                        whole_prior_reference_case=rule['whole_reference_case'], component_intervention_in_this_execution=False,
                        attention_mask_intervention_in_this_execution=False, head_intervention_in_this_execution=False,
                        initial_noise_replacement_in_this_execution=False, paired_noise_checked_exact=paired,
                        saved_q0_source_states_sha256=source['files_sha256']['states.pt'] if query == 0 else None,
                        saved_q0_site_captures_source_folder=source['folder'] if query == 0 else None,
                        saved_q0_site_capture_manifest_sha256=source['site_capture_manifest_sha256'] if query == 0 else None,
                        site_capture_arrays_copied_to_execution=False)
                    write(chunk / 'metadata.json', metadata)
                    assert read(chunk / 'normalized_actions.json') == record['actions'].tolist()
                    served.append((label, query))
                    write(destination / 'progress.json', dict(queries=len(served), saved_q0_predictions=cached, native_predictions=fresh,
                        fresh_model_forwards=30 * fresh, actual_later_whole_noise_pairs=pairs, served_queries=served, elapsed_s=time.perf_counter() - started))
                    print('[FUTURE-HEAD-EXECUTION] ' + json.dumps(dict(requests=len(served), case=label, query=query, seed=seed)), flush=True)
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(dict(actions=record['actions'].tolist(), metadata=metadata, paired_noise_checked_exact=paired), allow_nan=False).encode())
                except Exception as exc:
                    failed = True
                    write(destination / 'failed.json', dict(error=str(exc), traceback=traceback.format_exc(), served_queries=served))
                    self.send_error(500, str(exc))

        http = HTTPServer(('127.0.0.1', PORT), Handler)
        http.timeout = 30
        write(destination / 'ready.json', dict(state='ready', url=frozen['url']), exclusive=True)
        while not failed and len(served) < 144:
            http.handle_request()
            assert not (out / 'closed-loop/failed.json').exists() and not (out / 'wrapper_failed.json').exists()
        assert not failed and len(served) == 144 and cached == 18 and fresh == 126 and pairs == 119
        assert [(r['case'], r['query']) for r in reference_queries] == [(label, q) for label, _, arm in PLAN if arm in REFERENCE_ARMS for q in range(8)]
        assert value.hooks(model) == initial_hooks
        write(destination / 'whole-reference-query-controls.json', dict(state='exact', queries=reference_queries,
            eight_controls_all8_entire_records_and_PNGs_exact=True), exclusive=True)
        names = ('provenance.json', 'saved-q0-controls.json', 'whole-reference-query-controls.json')
        write(destination / 'complete.json', dict(state='complete', queries=144, saved_q0_predictions=18, native_predictions=126,
            fresh_model_forwards=3780, actual_later_whole_noise_pairs=119, q0_whole_noise_pair_checks=0, extra_q0_model_forwards=0,
            eight_controls_all8_entire_records_and_PNGs_exact=True, final_model_hooks_none=True, component_hooks_installed=False,
            attention_dispatch_wrapper_installed=False, files_sha256={name: sha(destination / name) for name in names}, scope=SCOPE), exclusive=True)
    finally:
        if http is not None:
            http.server_close()


def simulate(baseline):
    simulator, frozen, cross, helper, _, out = prepared(baseline)
    assert not (out / 'closed-loop').exists(), 'No partial output retry'
    os.environ.setdefault('MUJOCO_GL', 'egl')
    sys.path.insert(0, '/home/current/work/openpi-demo/cosmos-policy')
    sys.path.insert(0, str(ROOT / 'cosmos-framework'))
    import requests
    from cosmos_framework.simulation.libero.closed_loop_eval import _framewise_action_to_delta, _remap_gripper
    collector = simulator.load_file('head_execution_collector', ROOT / 'work/collect_cosmos_goal_pair.py')
    rollout = simulator.load_file('head_execution_controller', ROOT / 'work/rollout_cosmos_goal_pair.py')
    np = simulator.np
    stats = read(Path(frozen['threshold_contract']['frozen_files']['normalizer_stats']['path']))['global_raw']
    lo, hi = np.asarray(stats['q01']), np.asarray(stats['q99'])
    scale, offset = np.maximum(hi - lo, 1e-8) / 2, (hi + lo) / 2
    initial = np.load(out / 'inputs/x12/state.npy', allow_pickle=False)
    original_post, original_measure, original_summary = requests.post, simulator.measure, simulator.summarize_trial
    active, whole_controls, entries, windows = None, [], [], []

    def measure(env, obs, step, controller):
        result = original_measure(env, obs, step, controller)
        if step == 16:
            arrays = controller.controller_state(env.env)
            np.savez_compressed(out / 'closed-loop' / active / 'controller-step16.npz', **arrays)
            if frozen['trials'][active]['whole_reference_control']:
                ref = frozen['whole_physical_references'][active]
                assert sha(Path(ref['controller_step16_path'])) == ref['controller_step16_sha256']
                with np.load(ref['controller_step16_path'], allow_pickle=False) as source:
                    assert set(arrays) == set(source.files) and all(rollout.array_exact(value, source[key]) for key, value in arrays.items())
        return result

    def post(url, *args, **kwargs):
        request = kwargs['json']
        assert url == frozen['url'] + '/predict' and request['trial'] == active
        query, rule = request['query'], frozen['trials'][active]
        assert request['seed'] == rule['actual_query_seeds'][query]
        if query == 1:
            folder, source = out / 'closed-loop' / active, frozen['q0_sources'][active]
            state = np.load(folder / 'sim_states.npy', allow_pickle=False)
            assert state.shape == (17, initial.size) and state.dtype == initial.dtype and rollout.array_exact(state[0], initial)
            norm, raw = np.asarray(source['normalized_actions']), np.asarray(source['raw_actions'])
            native = np.asarray([_remap_gripper(_framewise_action_to_delta(item, '6d').tolist(), 'zero_one') for item in raw])
            for name, value in {'normalized_actions.json': norm, 'denormalized_actions.json': raw, 'actions.json': native, 'executed_actions.json': np.clip(native, -1, 1)}.items():
                assert rollout.array_exact(np.asarray(read(folder / name)), value), ('own cached q0 execution', active, name)
            trajectory = read(folder / 'trajectory.json')
            assert len(trajectory) == 17 and [r['step'] for r in trajectory] == list(range(17))
            assert all(r['exact'] for r in read(folder / 'initial_checks.json'))
            assert all(x is True for x in read(folder / 'initial_controller_checks.json').values())
            assert read(folder / 'initial_contact_checks.json') == read(out / 'inputs/x12/initial_contact_checks.json')
            with np.load(folder / 'initial_controller.npz', allow_pickle=False) as current:
                with np.load(out / 'inputs/x12/controller.npz', allow_pickle=False) as source_controller:
                    assert set(current.files) == set(source_controller.files)
                    assert all(rollout.array_exact(current[key], source_controller[key]) for key in current.files)
            if rule['whole_reference_control']:
                ref = Path(frozen['whole_physical_references'][active]['folder'])
                assert rollout.array_exact(state, np.load(ref / 'sim_states.npy', allow_pickle=False)[:17])
                assert trajectory == read(ref / 'trajectory.json')[:17]
                assert (folder / 'input_01.png').read_bytes() == (ref / 'input_01.png').read_bytes()
            names = ('sim_states.npy', 'trajectory.json', 'input_01.png', 'initial_controller.npz', 'controller-step16.npz')
            gate = dict(case=active, saved_attention_arm=rule['arm'], selected_window=rule['selected_window'],
                all16_own_cached_q0_normalized_raw_native_clipped_actions_byte_exact=True, common_initial_sim_state_and_typed_controller_exact=True,
                actual_prefix_state_records=17, actual_prefix_files_sha256={name: sha(folder / name) for name in names},
                whole_prior_prefix_bytes_exact=True if rule['whole_reference_control'] else None, reference_case=rule['whole_reference_case'])
            write(folder / 'q1-entry-control.json', gate, exclusive=True)
            entries.append(gate)
        return original_post(url, *args, **kwargs)

    def summary(trial, scene, seed, records, controller):
        rule = frozen['trials'][trial]
        result = original_summary(trial, scene, int(seed), records, controller)
        result.update(saved_q0_attention_arm=rule['arm'], saved_q0_window=rule['selected_window'],
            q0_vision_noise_source_seed=rule['q0_vision_noise_source_seed'], q0_action_noise_source_seed=195,
            later_noise_base=195, noise_seed_base=195, actual_query_seeds=rule['actual_query_seeds'], seed_rule=rule['seed_rule'])
        return result

    requests.post, simulator.measure, simulator.summarize_trial = post, measure, summary
    try:
        for label, v, arm in PLAN:
            active = label
            simulator.execute_trials(out, [label], collector, rollout)
            folder = out / 'closed-loop' / label
            actual = read(folder / 'summary.json')
            state = np.load(folder / 'sim_states.npy', allow_pickle=False)
            assert state.shape == (129, initial.size) and state.dtype == initial.dtype and np.isfinite(state).all()
            norm = np.asarray(read(folder / 'normalized_actions.json'))
            assert norm.shape == (128, 10) and norm.dtype == np.float64 and np.isfinite(norm).all()
            assert rollout.array_exact(norm, np.concatenate([np.asarray(read(folder / f'chunk_{q:02d}/normalized_actions.json')) for q in range(8)]))
            raw = norm * scale + offset
            native = np.asarray([_remap_gripper(_framewise_action_to_delta(item, '6d').tolist(), 'zero_one') for item in raw])
            assert native.shape == (128, 7) and native.dtype == np.float64 and np.isfinite(native).all()
            for name, value in (('denormalized_actions.json', raw), ('actions.json', native), ('executed_actions.json', np.clip(native, -1, 1))):
                assert rollout.array_exact(np.asarray(read(folder / name)), value), ('actual all128 official conversion', label, name)
            strict = helper.strict_windows(read(folder / 'trajectory.json'), actual, np)
            write(folder / 'strict-windows.json', strict, exclusive=True)
            windows.append(dict(case=label, arm=arm, selected_objects=actual['selected_objects'], strict_windows_sha256=sha(folder / 'strict-windows.json')))
            if frozen['trials'][label]['whole_reference_control']:
                ref = frozen['whole_physical_references'][label]
                control = cross.native_control(out, label, ref, np)
                with np.load(folder / 'initial_controller.npz', allow_pickle=False) as current:
                    with np.load(Path(ref['folder']) / 'initial_controller.npz', allow_pickle=False) as old:
                        assert set(current.files) == set(old.files) and all(rollout.array_exact(current[key], old[key]) for key in current.files)
                control.update(arm=arm, reference_case=ref['reference_case'], typed_controller0_and16_exact=True)
                whole_controls.append(control)
        done, served = read(out / 'closed-loop/complete.json'), read(out / 'server/complete.json')
        assert done['state'] == served['state'] == 'complete'
        assert done['cases'] == 18 and done['completed_trials'] == [r[0] for r in PLAN]
        assert done['steps_per_case'] == 128 and done['state_records_per_case'] == 129
        assert len(whole_controls) == 8 and len(entries) == len(windows) == 18
        assert served['queries'] == 144 and served['saved_q0_predictions'] == 18 and served['native_predictions'] == 126
        assert served['fresh_model_forwards'] == 3780 and served['actual_later_whole_noise_pairs'] == 119
        assert served['q0_whole_noise_pair_checks'] == served['extra_q0_model_forwards'] == 0
        assert served['eight_controls_all8_entire_records_and_PNGs_exact'] is served['final_model_hooks_none'] is True
        for name, expected in served['files_sha256'].items():
            assert sha(out / 'server' / name) == expected
        comparisons = []
        for label, v, arm in PLAN:
            actual = read(out / 'closed-loop' / label / 'summary.json')
            sham_label = f'V{v}_A195_all_allowed'
            sham = read(out / 'closed-loop' / sham_label / 'summary.json')
            comparisons.append(dict(case=label, arm=arm, vision_noise_source_seed=v, same_V_sham=sham_label,
                selected_objects=actual['selected_objects'], sham_selected_objects=sham['selected_objects'],
                classification_changed_vs_sham=set(actual['selected_objects']) != set(sham['selected_objects']),
                only_milk=actual['selected_objects'] == ['milk_1'], first_selection_window_start=actual['first_selection_step'],
                per_object=actual['per_object']))
        by_case = {row['case']: row for row in comparisons}
        primary_rows = [by_case[label] for label in PRIMARY_CASES]
        supported = all(row['only_milk'] for row in primary_rows)
        necessity = []
        for v in (195, 198):
            full, leave, native = (by_case[f'V{v}_A195_{arm}'] for arm in ('full_hardmask', 'leave_G6', 'native'))
            necessity.append(dict(vision_noise_source_seed=v, full_case=full['case'], leave_G6_case=leave['case'],
                native_already_only_milk=native['only_milk'], full_only_milk=full['only_milk'], leave_G6_only_milk=leave['only_milk'],
                loss_of_full_only_milk=full['only_milk'] and not leave['only_milk'],
                preregistered_V195_loss_of_rescue_secondary=v == 195, V198_complete_report_not_primary=v == 198,
                conditional_on_full_cut_numerical_background=True))
        random_both_milk = all(by_case[f'V{v}_A195_random_G6']['only_milk'] for v in (195, 198))
        random_strength = [dict(case=f'V{v}_A195_random_G6', vision_noise_source_seed=v,
            **frozen['q0_sources'][f'V{v}_A195_random_G6']['random_control']) for v in (195, 198)]
        random_strength_passed = all(item['actual_BF16_equal_strength_all_sites_within_tolerance'] for item in random_strength)
        write(out / 'primary-prediction.json', dict(state='evaluated_after_all18_complete', registered_hypothesis=frozen['primary_hypothesis'],
            primary_arm='head_G6', cases=primary_rows, prediction='supported' if supported else 'rejected', both_head_G6_strict_only_milk=supported,
            G4_leave_and_random_cannot_replace_primary=True, all18_cases_executed=True, eight_whole_byte_controls_passed=True,
            conditional_sufficiency_supported=supported, conditional_leave_G6_reports=necessity,
            random_G6_both_strict_only_milk=random_both_milk, random_replicates_primary_outcome_do_not_claim_head_specificity=random_both_milk,
            random_G6_actual_BF16_strength_reports=random_strength, random_actual_strength_gate_passed=random_strength_passed,
            random_strength_failure_prevents_equal_strength_specificity_interpretation=not random_strength_passed,
            group_specific_causal_claim_established=False, identity_or_single_head_or_cross_position_claim=False, scope=SCOPE), exclusive=True)
        write(out / 'comparisons.json', dict(state='evaluated_after_all18_complete', cases=comparisons,
            conditional_leave_G6_reports=necessity, all_secondary_and_random_cases_reported=True,
            random_G6_actual_BF16_strength_reports=random_strength, random_actual_strength_gate_passed=random_strength_passed,
            no_q0_or_physics_ranking_or_selection=True, primary_prediction_sha256=sha(out / 'primary-prediction.json'), scope=SCOPE), exclusive=True)
        case_files = {}
        for label, _, _ in PLAN:
            folder = out / 'closed-loop' / label
            names = ('sim_states.npy', 'actual.mp4', 'initial_controller.npz', 'controller-step16.npz', 'summary.json', 'strict-windows.json',
                'q1-entry-control.json', 'query_contract.json', 'initial_checks.json', 'initial_controller_checks.json', 'initial_contact_checks.json',
                *PHYSICAL, *[f'input_{q:02d}.png' for q in range(8)],
                *[f'chunk_{q:02d}/{name}' for q in range(8) for name in ('states.pt', 'metadata.json', 'normalized_actions.json')],
                *[f'chunk_00/{name}' for name in FILES if name not in ('states.pt', 'metadata.json', 'normalized_actions.json')],
                *[f"chunk_00/{s['file']}" for s in frozen['q0_sources'][label]['boundary_files']])
            case_files[label] = {name: sha(folder / name) for name in names}
        write(out / 'controls.json', dict(eight_whole_prior_controls=whole_controls, q1_entry_controls=entries, strict_windows=windows,
            actual_case_files_sha256=case_files, comparisons_sha256=sha(out / 'comparisons.json'),
            primary_prediction_sha256=sha(out / 'primary-prediction.json')), exclusive=True)
        names = ('controls.json', 'comparisons.json', 'primary-prediction.json', 'provenance.json', 'prepared.json', 'server/complete.json', 'closed-loop/complete.json', 'summary.json')
        write(out / 'complete.json', dict(state='complete', script_sha256=sha(Path(__file__)), trial_order=[r[0] for r in PLAN],
            physical_trials=18, saved_q0_predictions=18, native_predictions=126, fresh_model_forwards=3780,
            producer_q0_forwards_separate=540, combined_two_stage_model_forwards=4320, actual_later_whole_noise_pairs=119,
            eight_entire129_state_5JSON_8PNG_and_all8_modelrecords_controls_exact=True, actual_physical_actions=2304,
            all18_own_cached_q0_action_and_strict_window_gates_passed=True, all18_executed_without_score_selection=True,
            primary_arm='head_G6', primary_prediction='supported' if supported else 'rejected', both_head_G6_strict_only_milk=supported,
            all_secondary_and_random_cases_reported=True, group_specific_causal_claim_established=False,
            files_sha256={name: sha(out / name) for name in names}, scope=SCOPE), exclusive=True)
    finally:
        requests.post, simulator.measure, simulator.summarize_trial = original_post, original_measure, original_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('prepare', 'server', 'simulate'), required=True)
    args = parser.parse_args()
    baseline, out = args.output.resolve(), args.output.resolve() / 'future-head-execution'
    if args.mode == 'prepare':
        _, frozen, _, _, _ = contract(baseline)
        out.mkdir(exist_ok=False)
        (out / 'inputs').symlink_to(frozen['inputs_symlink_target'], target_is_directory=True)
        write(out / 'provenance.json', frozen, exclusive=True)
        write(out / 'prepared.json', dict(state='prepared', provenance_sha256=sha(out / 'provenance.json')), exclusive=True)
        print('[FUTURE-HEAD-EXECUTION] prepared ' + str(out), flush=True)
        return
    try:
        server(baseline) if args.mode == 'server' else simulate(baseline)
    except Exception as exc:
        if out.is_dir() and not (out / 'wrapper_failed.json').exists():
            write(out / 'wrapper_failed.json', dict(mode=args.mode, error=str(exc), traceback=traceback.format_exc()), exclusive=True)
        raise


if __name__ == '__main__':
    main()
