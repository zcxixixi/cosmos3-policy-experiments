"""Execute twelve preregistered early-denoising nine-layer group cases.

Use --output BASE --mode prepare|server|simulate; create only an absent
BASE/future-current-layer-execution. Require the frozen completed layer-group producer and its older joint-window evidence.
Serve its twelve saved q0 actions with zero new q0 model calls. Queries1..7
use each actual feedback image at seeds196..202: 96 requests, 84 native
predictions, 2520 forwards, 77 exact whole two-draw noise pairs. Combined
with the producer this is 2880 forwards, with all twelve 128-step trials.

Reuse the frozen 985f simulator/cross contract, e0c cached-record validation
and strict-window calculation, controller and NormalRuntime. Both native
and both global all-allowed sham cases must match their respective complete
e0c physical references, including all eight whole model records. All eight
window cases execute in fixed order without q0 scoring or candidate selection.
The only primary prediction is early_l1_9: BOTH V cases select only milk.
Its failure is reported as rejected; the other three windows cannot replace
it. No new network intervention, cross-V closure requirement or training.
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
LEGACY_SHA = '985f10b079684689dcd9735f8f55159c97976eaef1ac33ff163011980f6ca9d8'
HELPER_SHA = 'e0c5073ee320bbc173eebf827126c9431043a3efaa960eb0a8ccff289cae30ed'
PROBE_SHA = 'b5af0c4c63dbeee37c81a783c8c7f1f73146a6c9c679c0130899756b7f835689'
PORT, LATER_BASE = 8933, 195
PROMPT = 'pick up the milk and place it in the basket'
WINDOWS = {'early_l1_9': dict(steps=[0, 14], layers=[0, 8]),
           'early_l10_18': dict(steps=[0, 14], layers=[9, 17]),
           'early_l19_27': dict(steps=[0, 14], layers=[18, 26]),
           'early_l28_36': dict(steps=[0, 14], layers=[27, 35])}
PLAN = tuple((f'V{v}_A195_{arm}', v, arm) for arm in ('native', 'all_allowed', *WINDOWS) for v in (195, 198))
FILES = ('states.pt', 'normalized_actions.json', 'metadata.json', 'noise-audit.pt', 'noise-audit.json',
         'observer-report.json', 'indexes-and-masks.pt', 'control-checks.json')
PHYSICAL = ('trajectory.json', 'actions.json', 'executed_actions.json', 'normalized_actions.json', 'denormalized_actions.json')
SCOPE = ('Twelve complete X+12cm physical trajectories executing the frozen window q0 actions, '
         'then native inference from actual feedback at later base195. The two native and two '
         'global all-allowed shams have separate complete prior references. All four nine-layer groups '
         'for both V realizations are executed; only early_l1_9 both selecting exactly milk '
         'tests the primary prediction. A hard mask also redistributes remaining attention. '
         'Higher-layer groups are same-site-count regional comparisons, not semantically unrelated controls; no score selection, additional internal intervention, training, semantic identity '
         'or general neural root-cause claim.')


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


def contract(baseline):
    legacy_path = ROOT / 'work/run_cosmos_milk_initial_noise_execution.py'
    helper_path = ROOT / 'work/run_cosmos_milk_future_read_execution.py'
    probe_path = ROOT / 'work/probe_cosmos_milk_future_current_layer_groups.py'
    assert sha(legacy_path) == LEGACY_SHA and sha(helper_path) == HELPER_SHA and sha(probe_path) == PROBE_SHA
    legacy = load('future_current_execution_legacy', legacy_path)
    simulator, prior, cross, _ = legacy.prepared(baseline)
    helper = load('future_current_execution_frozen_helpers', helper_path)
    probe = load('future_current_execution_frozen_windows', probe_path)
    _, _, _, original, original_sources, image, paths, evidence = probe.source_contract(baseline)
    assert original == prior['threshold_contract'] and probe.PLAN == PLAN and probe.WINDOWS == WINDOWS
    producer = baseline / 'future-current-layer-groups'
    assert producer.is_dir() and not (producer / 'failed.json').exists()
    complete, results, protocol, provenance, sources, primary = (read(producer / (name + '.json')) for name in
        ('complete', 'results', 'protocol', 'provenance', 'sources', 'primary-prediction'))
    assert complete['state'] == results['state'] == 'complete'
    assert complete['script_sha256'] == provenance['script_sha256'] == PROBE_SHA
    for name in ('results', 'protocol', 'provenance', 'sources'):
        assert complete[name + '_sha256'] == sha(producer / (name + '.json'))
    assert complete['primary_prediction_sha256'] == results['primary_prediction_sha256'] == sha(producer / 'primary-prediction.json')
    assert complete['fresh_q0_predictions'] == results['fresh_q0_predictions'] == 12
    assert complete['fresh_model_forwards'] == results['fresh_model_forwards'] == 360
    assert complete['total_official_dispatch_calls'] == results['total_official_dispatch_calls'] == 37800
    assert complete['original_dispatch_calls'] == 25920 and complete['extra_dispatch_calls'] == 11880
    assert complete['original_randn_calls_consumed'] == complete['returned_source_draws_observed'] == 24
    for key in ('two_native_entire_prior_records_exact', 'two_sham_entire_prior_records_exact',
                'all_four_control_all30_full_tuples_and_boundaries_exact', 'all8_windows_before_first_cut_own_sham_prefix_exact',
                'all_active_cut_sites_unblocked_GEN_vs_same_input_allallowed_exact',
                'all10_masked_full1080_sites_return_common_union66', 'all360_actual_complete_model_kwargs_saved',
                'all_original_symbols_restored', 'all_owned_hooks_removed'):
        assert complete[key] is True
    assert complete['physics_calls'] == complete['later_query_predictions'] == results['physics_calls'] == 0
    assert complete['primary_physical_prediction_evaluated'] is results['primary_physical_prediction_evaluated'] is False
    assert results['official_dispatch_counts'] == dict(model_before=360, model_after=360, native_UND_dispatch=12960,
        native_GEN_dispatch=12960, all_allowed_dispatch=10800, cut_dispatch=1080)
    expected = [dict(case=label, vision_noise_source_seed=v, action_noise_source_seed=195, runtime_rng_seed=195,
        arm=arm, window=WINDOWS.get(arm)) for label, v, arm in PLAN]
    assert protocol['plan'] == expected and protocol['trial_order'] == results['preregistered_physics_cases'] == [item[0] for item in PLAN]
    assert protocol['windows_zero_based_inclusive'] == WINDOWS and protocol['cut_sites_per_window_case'] == 135
    assert protocol['masked_sites_per_case'] == 1080 and protocol['primary_window'] == 'early_l1_9'
    primary_cases = ['V195_A195_early_l1_9', 'V198_A195_early_l1_9']
    assert protocol['primary_cases'] == primary['cases'] == primary_cases
    assert primary['status'] == 'not_evaluated_q0_only' and primary['both_selected_objects_must_equal'] == ['milk_1']
    assert primary['remaining_windows_reported_without_replacing_primary'] == list(WINDOWS)[1:]
    assert primary['all12_physical_cases_preregistered'] is True and primary['q0_numeric_rank_used'] is False
    assert protocol['primary_evaluation']['success_requires_both_V_only_milk'] is True
    assert protocol['primary_evaluation']['rejection_does_not_promote_another_window'] is True
    physical_plan = protocol['later_physics_preregistered']
    assert physical_plan['trial_order'] == [item[0] for item in PLAN] and physical_plan['cases'] == physical_plan['cached_q0'] == 12
    assert physical_plan['later_noise_base'] == 195 and physical_plan['q1_to_q7_seeds'] == list(range(196, 203))
    assert physical_plan['later_predictions'] == 84 and physical_plan['later_model_forwards'] == 2520
    assert physical_plan['total_two_stage_model_forwards'] == 2880 and physical_plan['no_q0_score_or_rank_selection'] is True
    assert physical_plan['native_and_allallowed_entire129_state_5JSON_8PNG_and_all8_query_records_must_equal_existing_references'] is True
    assert physical_plan['physical_control_references'] == evidence['physical_control_references']
    assert protocol['image_sha256'] == sha(image) and protocol['prompt'] == PROMPT and protocol['shift_cm'] == 12
    assert sources == dict(source_evidence=evidence, original_sources=original_sources, frozen_contract=original)
    assert provenance['actual_transformer_sha256'] == prior['actual_transformer_sha256']
    assert provenance['actual_pipeline_source_sha256'] == prior['actual_pipeline_source_sha256']
    assert provenance['source_sha256'] == {key: sha(path) for key, path in paths.items()}
    np = simulator.np
    stats = read(Path(original['frozen_files']['normalizer_stats']['path']))['global_raw']
    lo, hi = np.asarray(stats['q01']), np.asarray(stats['q99'])
    assert lo.shape == hi.shape == (10,) and np.all(hi - lo > 1e-6)
    scale, offset = np.maximum(hi - lo, 1e-8) / 2, (hi + lo) / 2
    assert [row['case'] for row in results['cases']] == [item[0] for item in PLAN]
    q0, rules, references = {}, {}, {}
    for row, (label, v, arm) in zip(results['cases'], PLAN):
        folder = producer / label
        window, controlled = WINDOWS.get(arm), arm in ('native', 'all_allowed')
        assert row['arm'] == arm and row['window'] == window and row['vision_noise_source_seed'] == v
        assert row['action_noise_source_seed'] == row['runtime_rng_seed'] == 195 and row['fresh_model_forwards'] == 30
        assert row['existing_entire_control_record_exact'] is (True if controlled else None)
        assert row['own_sham_pre_first_cut_prefix_exact'] is (True if window else None)
        assert row['physical_prediction_evaluated'] is False and set(row['files_sha256']) == set(FILES)
        for name, expected_sha in row['files_sha256'].items():
            assert sha(folder / name) == expected_sha
        report, meta, check = (read(folder / name) for name in ('observer-report.json', 'metadata.json', 'control-checks.json'))
        expected_counts = dict(model_before=30, model_after=30, block_before=1080, block_after=1080, pre_W=1080,
            native_UND_dispatch=1080, native_GEN_dispatch=1080, all_allowed_dispatch=0 if arm == 'native' else 1080,
            cut_dispatch=135 if window else 0, substituted_sites=0 if arm == 'native' else 1080)
        assert report['counts'] == row['actual_dispatch_counts'] == expected_counts
        assert report['state'] == 'complete' and report['case_arm'] == arm and report['selected_window'] == window
        for key in ('all_actual_dispatch_flatten_pre_W_byte_exact', 'future200_and_UND_native_dispatch_returns_preserved',
                    'all30_actual_complete_model_kwargs_saved', 'all_owned_hooks_removed', 'original_dispatch_symbol_restored'):
            assert report[key] is True
        assert report['original_full_266_row_projection_calls'] == 1080 and report['processor_recompute_calls'] == 0
        assert report['all_masked_sites_same_union66'] is (None if arm == 'native' else True)
        assert report['all_window_cut_unblocked_GEN_rows_vs_same_input_all_allowed_exact'] is (True if window else None)
        assert len(row['boundary_files']) == 30 and report['boundary_files'] == row['boundary_files']
        for step, item in enumerate(row['boundary_files']):
            assert item['step'] == step and item['file'] == f'boundaries-t{step:02d}.pt'
            assert item['current_shape'] == [37, 50, 4096] and item['action_shape'] == [37, 16, 4096]
            assert sha(folder / item['file']) == item['sha256']
        assert meta['case'] == label and meta['arm'] == arm and meta['window_zero_based_inclusive'] == window
        assert meta['seed'] == meta['action_noise_source_seed'] == meta['runtime_rng_seed'] == 195
        assert meta['vision_noise_source_seed'] == v and meta['model_calls'] == 30 and meta['prepare_calls'] == 1
        assert meta['numerical_masked_sham'] is (arm == 'all_allowed') and meta['native_noop'] is (arm == 'native')
        assert meta['common_union66_all_allowed_background'] is (arm != 'native')
        assert meta['window_cut_sites'] == expected_counts['cut_dispatch'] and meta['actual_masked_site_substitutions'] == expected_counts['substituted_sites']
        assert meta['input_png_sha256'] == protocol['image_sha256'] and meta['physical_prediction_evaluated'] is False
        if controlled:
            assert check['entire_existing_q0_record_byte_exact'] is check['all30_existing_full_model_output_tuples_exact'] is True
            assert check['all30_existing_current_action_37boundaries_exact'] is True
            assert check['comparison_folder'] == evidence['q0_controls'][label]['folder']
            reference = evidence['physical_control_references'][label]
            references[label] = dict(folder=reference['folder'], files_current_sha256=reference['files_sha256'],
                controller_step16_path=str(Path(reference['folder']) / 'controller-step16.npz'),
                controller_step16_sha256=reference['files_sha256']['controller-step16.npz'])
        else:
            assert check['state'] == 'exact' and check['first_cut_step'] == window['steps'][0] and check['first_cut_layer_zero_based'] == window['layers'][0]
            for key in ('earlier_complete_model_outputs_and_head_prefix_exact', 'all_actual_complete_kwargs_through_first_cut_step_exact',
                        'earlier_all37_and_first_cut_before_layer_current_action_boundaries_exact', 'solver_samples_through_first_cut_input_exact'):
                assert check[key] is True
            assert check['after_first_cut_equal_to_independent_sham_required'] is False
        normalized = np.asarray(read(folder / 'normalized_actions.json'))
        assert normalized.shape == (16, 10) and np.isfinite(normalized).all() and normalized.tolist() == row['final_normalized_actions']
        q0[label] = dict(folder=str(folder), files_sha256=row['files_sha256'], boundary_files=row['boundary_files'],
            normalized_actions=normalized.tolist(), raw_actions=(normalized * scale + offset).tolist(),
            arm=arm, selected_window=window, masked_attention=arm != 'native', vision_noise_source_seed=v, action_noise_source_seed=195,
            q0_masked_description=dict(selected_window_zero_based_inclusive=window, cut_sites=expected_counts['cut_dispatch'],
                actual_substituted_sites=expected_counts['substituted_sites'], common_union_query_rows=0 if arm == 'native' else 66,
                global_all_allowed_background=arm != 'native', future200_retained_native=True, UND_retained_native=True,
                full_GEN_output_projection_shape=[266, 4096], original_projection_calls=1080,
                cut_unblocked_query_gate_vs_same_input_all_allowed=window is not None, residual_compensation=False))
        rules[label] = dict(arm=arm, selected_window=window, q0_vision_noise_source_seed=v, q0_action_noise_source_seed=195,
            whole_reference_control=controlled, native_control=arm == 'native', numerical_sham=arm == 'all_allowed',
            primary_case=arm == 'early_l1_9', later_noise_base=195, actual_query_seeds=list(range(195, 203)),
            seed_rule='Saved q0 has separate V/A sources and invocation195; q1..7=195+query.', q0_whole_noise_pairing=False)
    frozen = dict(script_sha256=sha(Path(__file__)), legacy_execution_sha256=LEGACY_SHA, helper_execution_sha256=HELPER_SHA,
        producer_script_sha256=PROBE_SHA, threshold_contract=original, actual_transformer_sha256=prior['actual_transformer_sha256'],
        actual_pipeline_source_sha256=prior['actual_pipeline_source_sha256'], metrics_path=prior['metrics_path'],
        producer_source_cases=prior['producer_source_cases'], native_q0_source_pair=evidence['anchors']['source_pair'],
        whole_physical_references=references, source_evidence=evidence,
        producer_documents_sha256={name: sha(producer / (name + '.json')) for name in ('complete', 'results', 'provenance', 'protocol', 'sources', 'primary-prediction')},
        producer_protocol=protocol, q0_sources=q0, trials=rules, trial_order=[item[0] for item in PLAN],
        primary_window='early_l1_9', primary_cases=primary_cases, primary_selected_objects_exact=['milk_1'],
        primary_rejection_does_not_promote_other_windows=True, all12_execute_without_q0_score_selection=True,
        inputs_symlink_target=str(baseline / 'position-threshold/inputs'), url=f'http://127.0.0.1:{PORT}',
        saved_q0_predictions=12, requests=96, native_predictions=84, fresh_model_forwards=2520, producer_q0_forwards_separate=360,
        combined_two_stage_model_forwards=2880, actual_later_whole_noise_pairs_expected=77, q0_whole_noise_pair_checks=0, scope=SCOPE)
    simulator.SHIFTS, simulator.SEEDS = dict(x00=0, x09=9, x12=12), (195,)
    simulator.TRIALS = {label: ('x12', cross.QuerySeed(195, 195)) for label, _, _ in PLAN}
    simulator.URL, simulator.SCOPE = frozen['url'], SCOPE
    return simulator, frozen, cross, helper


def prepared(baseline):
    simulator, frozen, cross, helper = contract(baseline)
    out = baseline / 'future-current-layer-execution'
    assert out.is_dir() and not (out / 'wrapper_failed.json').exists()
    assert (out / 'inputs').is_symlink() and (out / 'inputs').resolve() == Path(frozen['inputs_symlink_target']).resolve()
    assert read(out / 'provenance.json') == frozen
    saved = read(out / 'prepared.json')
    assert saved['state'] == 'prepared' and saved['provenance_sha256'] == sha(out / 'provenance.json')
    return simulator, frozen, cross, helper, out


def server(baseline):
    _, frozen, _, helper, out = prepared(baseline)
    destination = out / 'server'
    destination.mkdir(exist_ok=False)
    served, later_noise, pairs, cached, fresh, reference_queries = [], {}, 0, 0, 0, []
    failed, http, started = False, None, time.perf_counter()
    try:
        os.environ['HF_HUB_OFFLINE'] = '1'
        paths = frozen['threshold_contract']['frozen_files']
        base = load('future_current_execution_normal_runtime', Path(paths['normal_runtime']['path']))
        base.TASKS['milk'] = (7, 'pick_up_the_milk_and_place_it_in_the_basket', PROMPT, 'milk_1')
        factory = load('future_current_execution_factory', Path(paths['factory']['path']))
        factory.OUT = destination
        assert sha(factory.SCHEDULER) == paths['scheduler']['sha256']
        runtime = base.NormalRuntime(factory)
        torch, model = runtime.torch, runtime.pipe.transformer
        assert not model.training and not model.is_cache_enabled and not torch.is_grad_enabled()
        assert sha(Path(inspect.getfile(type(model)))) == frozen['actual_transformer_sha256']
        assert sha(Path(inspect.getfile(type(runtime.pipe)))) == frozen['actual_pipeline_source_sha256']
        scan = load('future_current_execution_record_checks', Path(frozen['metrics_path']))
        originals, native_pair, q0, controls = {}, {}, {}, []
        for seed in (195, 198):
            source = frozen['producer_source_cases'][str(seed)]
            folder = Path(source['q0_folder'])
            assert sha(folder / 'states.pt') == source['q0_files_current_sha256']['states.pt']
            originals[seed] = torch.load(folder / 'states.pt', map_location='cpu', weights_only=True, mmap=True)
            source = frozen['native_q0_source_pair'][str(seed)]
            folder = Path(source['folder'])
            assert sha(folder / 'states.pt') == source['files_sha256']['states.pt']
            native_pair[seed] = torch.load(folder / 'states.pt', map_location='cpu', weights_only=True, mmap=True)
        for label, v, arm in PLAN:
            source, folder = frozen['q0_sources'][label], Path(frozen['q0_sources'][label]['folder'])
            actual = torch.load(folder / 'states.pt', map_location='cpu', weights_only=True, mmap=True)
            audit = torch.load(folder / 'noise-audit.pt', map_location='cpu', weights_only=True, mmap=True)
            scan.validate_record(actual, torch)
            helper.verify_cached(actual, audit, originals, native_pair, arm, v, runtime, scan)
            assert actual['actions'].tolist() == source['normalized_actions']
            q0[label] = actual
            if frozen['trials'][label]['whole_reference_control']:
                ref = frozen['whole_physical_references'][label]
                path = Path(ref['folder']) / 'chunk_00/states.pt'
                assert sha(path) == ref['files_current_sha256']['chunk_00/states.pt']
                runtime.require_exact(actual, torch.load(path, map_location='cpu', weights_only=True, mmap=True), 'ENTIRE own prior native or sham q0/' + label)
            controls.append(dict(case=label, arm=arm, window=source['selected_window'], actual_noise_preparation_initial_kwargs_and_schedule_exact=True,
                native_entire_fixed_A195_source_record_exact=True if arm == 'native' else None,
                entire_own_prior_q0_reference_exact=True if frozen['trials'][label]['whole_reference_control'] else None,
                q0_model_forwards_in_this_execution=0, producer_files_sha256=source['files_sha256'], saved_masked_description=source['q0_masked_description']))
        write(destination / 'saved-q0-controls.json', dict(state='exact', cases=controls, four_whole_prior_control_records_exact=True,
            q0_whole_noise_pair_checks=0, same_seed_does_not_imply_same_whole_V_A_pair=True, extra_q0_model_forwards=0), exclusive=True)
        write(destination / 'provenance.json', dict(script_sha256=sha(Path(__file__)), execution_provenance_sha256=sha(out / 'provenance.json'),
            saved_q0_controls_sha256=sha(destination / 'saved-q0-controls.json'), component_hooks_installed=False,
            attention_dispatch_wrapper_installed=False, saved_q0=12, native_later_predictions=84, expected_fresh_model_forwards=2520, scope=SCOPE), exclusive=True)

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
                    assert label in frozen['trials'] and type(query) is int and 0 <= query < 8
                    assert (label, query) not in served and all((label, i) in served for i in range(query))
                    assert label == frozen['trial_order'][len(served) // 8] and query == len(served) % 8, 'Fixed case/query order'
                    rule, source = frozen['trials'][label], frozen['q0_sources'][label]
                    seed = rule['actual_query_seeds'][query]
                    assert request['seed'] == seed and request['prompt'] == PROMPT
                    image = out / 'closed-loop' / label / f'input_{query:02d}.png'
                    chunk = image.parent / f'chunk_{query:02d}'
                    if query == 0:
                        assert image.read_bytes() == (out / 'inputs/x12/input.png').read_bytes()
                        folder = Path(source['folder'])
                        chunk.mkdir(exist_ok=False)
                        for name in FILES:
                            assert sha(folder / name) == source['files_sha256'][name]
                            if name != 'metadata.json':
                                shutil.copyfile(folder / name, chunk / name)
                        record, metadata = q0[label], read(folder / 'metadata.json')
                        cached, paired = cached + 1, False
                    else:
                        record, metadata = runtime.predict('milk', image, seed, chunk)
                        fresh += 1
                        paired = query in later_noise
                        if paired:
                            runtime.require_exact(record['pure_noise'], later_noise[query], 'actual later195 whole two-draw pair/' + str(query))
                            pairs += 1
                        else:
                            later_noise[query] = record['pure_noise']
                    scan.validate_record(record, torch)
                    for key in ('timesteps', 'sigmas'):
                        runtime.require_exact(record[key], originals[195][key], 'actual complete native schedule/' + key)
                    if rule['whole_reference_control']:
                        reference = frozen['whole_physical_references'][label]
                        folder, name = Path(reference['folder']), f'chunk_{query:02d}/states.pt'
                        assert image.read_bytes() == (folder / f'input_{query:02d}.png').read_bytes()
                        assert sha(folder / name) == reference['files_current_sha256'][name]
                        runtime.require_exact(record, torch.load(folder / name, map_location='cpu', weights_only=True, mmap=True), 'ENTIRE prior own native/sham query/' + label + '/' + str(query))
                        reference_queries.append(dict(case=label, query=query, entire_NormalRuntime_record_bytes_exact=True,
                            actual_feedback_PNG_bytes_exact=True, reference_states_sha256=sha(folder / name), current_states_sha256=sha(chunk / 'states.pt')))
                    assert metadata['seed'] == seed and metadata['model_calls'] == 30 and metadata['prepare_calls'] == 1
                    metadata.update(trial=label, query=query, shift_cm=12, image_path=str(image), input_png_sha256=sha(image),
                        saved_q0_prediction=query == 0, current_execution_model_calls=0 if query == 0 else 30,
                        source_record_model_calls=30 if query == 0 else None, saved_q0_attention_arm=rule['arm'],
                        saved_q0_was_masked=source['masked_attention'], saved_q0_window=rule['selected_window'],
                        q0_vision_noise_source_seed=rule['q0_vision_noise_source_seed'], q0_action_noise_source_seed=195,
                        later_noise_base=195, actual_query_seeds=rule['actual_query_seeds'], seed_rule=rule['seed_rule'],
                        whole_prior_control=rule['whole_reference_control'], component_intervention_in_this_execution=False,
                        attention_mask_intervention_in_this_execution=False, initial_noise_replacement_in_this_execution=False,
                        paired_noise_checked_exact=paired, saved_q0_source_states_sha256=source['files_sha256']['states.pt'] if query == 0 else None,
                        saved_q0_masked_description=source['q0_masked_description'] if query == 0 else None)
                    write(chunk / 'metadata.json', metadata)
                    assert read(chunk / 'normalized_actions.json') == record['actions'].tolist()
                    served.append((label, query))
                    write(destination / 'progress.json', dict(queries=len(served), saved_q0_predictions=cached, native_predictions=fresh,
                        fresh_model_forwards=30 * fresh, actual_later_whole_noise_pairs=pairs, served_queries=served, elapsed_s=time.perf_counter() - started))
                    print('[FUTURE-CURRENT-LAYER-EXECUTION] ' + json.dumps(dict(requests=len(served), case=label, query=query, seed=seed)), flush=True)
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
        while not failed and len(served) < 96:
            http.handle_request()
            assert not (out / 'closed-loop/failed.json').exists() and not (out / 'wrapper_failed.json').exists()
        assert not failed and len(served) == 96 and cached == 12 and fresh == 84 and pairs == 77
        assert [(item['case'], item['query']) for item in reference_queries] == [(label, q) for label, _, _ in PLAN[:4] for q in range(8)]
        write(destination / 'whole-reference-query-controls.json', dict(state='exact', queries=reference_queries,
            two_native_and_two_sham_all8_entire_records_and_PNGs_exact=True), exclusive=True)
        write(destination / 'complete.json', dict(state='complete', queries=96, saved_q0_predictions=12, native_predictions=84,
            fresh_model_forwards=2520, actual_later_whole_noise_pairs=77, q0_whole_noise_pair_checks=0, extra_q0_model_forwards=0,
            four_controls_all_eight_query_records_entire_exact=True, component_hooks_installed=False, attention_dispatch_wrapper_installed=False,
            files_sha256={name: sha(destination / name) for name in ('provenance.json', 'saved-q0-controls.json', 'whole-reference-query-controls.json')}, scope=SCOPE), exclusive=True)
    finally:
        if http is not None:
            http.server_close()


def simulate(baseline):
    simulator, frozen, cross, helper, out = prepared(baseline)
    assert not (out / 'closed-loop').exists(), 'No partial output retry'
    os.environ.setdefault('MUJOCO_GL', 'egl')
    sys.path.insert(0, '/home/current/work/openpi-demo/cosmos-policy')
    sys.path.insert(0, str(ROOT / 'cosmos-framework'))
    import requests
    from cosmos_framework.simulation.libero.closed_loop_eval import _framewise_action_to_delta, _remap_gripper
    collector = simulator.load_file('future_current_execution_collector', ROOT / 'work/collect_cosmos_goal_pair.py')
    rollout = simulator.load_file('future_current_execution_controller', ROOT / 'work/rollout_cosmos_goal_pair.py')
    np = simulator.np
    original_post, original_measure, original_summary = requests.post, simulator.measure, simulator.summarize_trial
    active, whole_controls, entries, windows, sham_controls = None, [], [], [], []

    def measure(env, obs, step, controller):
        result = original_measure(env, obs, step, controller)
        if step == 16:
            arrays = controller.controller_state(env.env)
            folder = out / 'closed-loop' / active
            np.savez_compressed(folder / 'controller-step16.npz', **arrays)
            if frozen['trials'][active]['whole_reference_control']:
                ref = frozen['whole_physical_references'][active]
                assert sha(Path(ref['controller_step16_path'])) == ref['controller_step16_sha256']
                with np.load(ref['controller_step16_path'], allow_pickle=False) as source:
                    assert set(arrays) == set(source.files)
                    assert all(rollout.array_exact(value, source[key]) for key, value in arrays.items())
        return result

    def post(url, *args, **kwargs):
        request = kwargs['json']
        assert url == frozen['url'] + '/predict' and request['trial'] == active
        query, rule = request['query'], frozen['trials'][active]
        assert request['seed'] == rule['actual_query_seeds'][query]
        if query == 1:
            folder, source = out / 'closed-loop' / active, frozen['q0_sources'][active]
            state = np.load(folder / 'sim_states.npy', allow_pickle=False)
            initial = np.load(out / 'inputs/x12/state.npy', allow_pickle=False)
            assert state.shape == (17, initial.size) and state.dtype == initial.dtype
            assert state[0].tobytes(order='C') == initial.tobytes(order='C')
            norm, raw = np.asarray(source['normalized_actions']), np.asarray(source['raw_actions'])
            native = np.asarray([_remap_gripper(_framewise_action_to_delta(item, '6d').tolist(), 'zero_one') for item in raw])
            for name, value in {'normalized_actions.json': norm, 'denormalized_actions.json': raw,
                                'actions.json': native, 'executed_actions.json': np.clip(native, -1, 1)}.items():
                assert rollout.array_exact(np.asarray(read(folder / name)), value), ('own cached q0 execution', active, name)
            trajectory = read(folder / 'trajectory.json')
            assert len(trajectory) == 17 and [item['step'] for item in trajectory] == list(range(17))
            if rule['whole_reference_control']:
                reference = Path(frozen['whole_physical_references'][active]['folder'])
                assert rollout.array_exact(state, np.load(reference / 'sim_states.npy', allow_pickle=False)[:17])
                assert trajectory == read(reference / 'trajectory.json')[:17]
                assert (folder / 'input_01.png').read_bytes() == (reference / 'input_01.png').read_bytes()
            gate = dict(case=active, saved_attention_arm=rule['arm'], selected_window=rule['selected_window'],
                all_16_own_cached_q0_normalized_raw_native_clipped_actions_byte_exact=True,
                common_initial_sim_state_bytes_exact=True, actual_prefix_state_records=17,
                actual_prefix_files_sha256={name: sha(folder / name) for name in ('sim_states.npy', 'trajectory.json', 'input_01.png', 'controller-step16.npz')},
                whole_prior_control_prefix_bytes_exact=True if rule['whole_reference_control'] else None,
                window_old_prefix_reference_used=False if not rule['whole_reference_control'] else None)
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
            if arm in WINDOWS:
                assert len(whole_controls) == 4
            # Each actual legacy call resets seed-only seen_noise. Q0 is unpaired;
            # the server separately checks all77 later whole returned draw pairs.
            simulator.execute_trials(out, [label], collector, rollout)
            folder = out / 'closed-loop' / label
            actual_summary = read(folder / 'summary.json')
            gate = helper.strict_windows(read(folder / 'trajectory.json'), actual_summary, np)
            write(folder / 'strict-windows.json', gate, exclusive=True)
            windows.append(dict(case=label, arm=arm, selected_window=WINDOWS.get(arm), selected_objects=actual_summary['selected_objects'],
                files_sha256={name: sha(folder / name) for name in ('strict-windows.json', 'summary.json', 'trajectory.json')}))
            if frozen['trials'][label]['whole_reference_control']:
                control = cross.native_control(out, label, frozen['whole_physical_references'][label], np)
                control.update(attention_arm=arm, respective_prior_reference_folder=frozen['whole_physical_references'][label]['folder'])
                whole_controls.append(control)
                write(out / (label + '-control.json'), control, exclusive=True)
            if arm == 'all_allowed':
                native_label = f'V{v}_A195_native'
                native_objects = read(out / 'closed-loop' / native_label / 'summary.json')['selected_objects']
                changed = set(native_objects) != set(actual_summary['selected_objects'])
                sham_controls.append(dict(case=label, native_reference_case=native_label, native_selected_objects=native_objects,
                    sham_selected_objects=actual_summary['selected_objects'], classification_changed=changed,
                    original_native_path_interpretation_gate_passed=not changed,
                    numerical_sham_is_native_noop=False, whole_prior_sham_physics_reference_exact=True))
        done, served = read(out / 'closed-loop/complete.json'), read(out / 'server/complete.json')
        assert done['state'] == served['state'] == 'complete'
        assert done['cases'] == 12 and done['completed_trials'] == [item[0] for item in PLAN]
        assert done['steps_per_case'] == 128 and done['state_records_per_case'] == 129
        assert len(whole_controls) == 4 and len(sham_controls) == 2 and len(entries) == len(windows) == 12
        assert served['queries'] == 96 and served['saved_q0_predictions'] == 12 and served['native_predictions'] == 84
        assert served['fresh_model_forwards'] == 2520 and served['actual_later_whole_noise_pairs'] == 77
        assert served['q0_whole_noise_pair_checks'] == served['extra_q0_model_forwards'] == 0
        assert served['four_controls_all_eight_query_records_entire_exact'] is True
        for name, expected_sha in served['files_sha256'].items():
            assert sha(out / 'server' / name) == expected_sha
        primary_results, comparisons = [], []
        for arm in WINDOWS:
            for v in (195, 198):
                label, sham = f'V{v}_A195_{arm}', f'V{v}_A195_all_allowed'
                actual, background = read(out / 'closed-loop' / label / 'summary.json'), read(out / 'closed-loop' / sham / 'summary.json')
                comparisons.append(dict(case=label, same_V_global_sham=sham, selected_objects=actual['selected_objects'],
                    sham_selected_objects=background['selected_objects'], classification_changed_vs_sham=set(actual['selected_objects']) != set(background['selected_objects']),
                    only_milk=actual['selected_objects'] == ['milk_1'], milk_included='milk_1' in actual['selected_objects'],
                    first_selection_step=actual['first_selection_step'], per_object=actual['per_object'], primary_case=arm == 'early_l1_9'))
                if arm == 'early_l1_9':
                    primary_results.append(dict(case=label, selected_objects=actual['selected_objects'], strict_only_milk=actual['selected_objects'] == ['milk_1'],
                        strict_windows_sha256=sha(out / 'closed-loop' / label / 'strict-windows.json')))
        supported = all(item['strict_only_milk'] for item in primary_results)
        assert [item['case'] for item in primary_results] == frozen['primary_cases']
        write(out / 'primary-prediction.json', dict(state='evaluated_after_all12_complete', primary_window='early_l1_9',
            criterion='Each of the two cases selected_objects == [milk_1] by both-finger contact and lift >.02m for five consecutive records.',
            cases=primary_results, prediction='supported' if supported else 'rejected', both_only_milk=supported,
            all12_cases_executed=True, other_windows_cannot_replace_primary=True,
            descriptive_milk_inclusion_or_max_lift_cannot_replace_primary=True, comparisons_to_same_V_sham=comparisons, scope=SCOPE), exclusive=True)
        case_files = {}
        for label, _, _ in PLAN:
            folder = out / 'closed-loop' / label
            names = ('sim_states.npy', 'actual.mp4', 'initial_controller.npz', 'controller-step16.npz', 'summary.json', 'strict-windows.json',
                'q1-entry-control.json', 'query_contract.json', *PHYSICAL, *[f'input_{q:02d}.png' for q in range(8)],
                *[f'chunk_{q:02d}/{name}' for q in range(8) for name in ('states.pt', 'metadata.json', 'normalized_actions.json')],
                *[f'chunk_00/{name}' for name in FILES if name not in ('states.pt', 'metadata.json', 'normalized_actions.json')])
            case_files[label] = {name: sha(folder / name) for name in names}
        write(out / 'controls.json', dict(native_and_sham_whole_controls=whole_controls, q1_entry_controls=entries, strict_windows=windows,
            sham_classification_controls=sham_controls, original_native_path_interpretation_gate_passed=all(not item['classification_changed'] for item in sham_controls),
            primary_prediction_sha256=sha(out / 'primary-prediction.json'), actual_case_files_sha256=case_files), exclusive=True)
        files = ('controls.json', 'primary-prediction.json', 'provenance.json', 'prepared.json', 'server/complete.json', 'closed-loop/complete.json', 'summary.json')
        write(out / 'complete.json', dict(state='complete', script_sha256=sha(Path(__file__)), trial_order=[item[0] for item in PLAN],
            physical_trials=12, saved_q0_predictions=12, native_predictions=84, fresh_model_forwards=2520,
            producer_q0_forwards_separate=360, combined_two_stage_model_forwards=2880, actual_later_whole_noise_pairs=77,
            four_native_and_sham_entire_129_state_5JSON_8PNG_and_8_modelrecords_exact=True,
            all12_own_cached_q0_action_execution_and_strict_window_gates_passed=True, all12_cases_executed_without_q0_selection=True,
            primary_prediction='supported' if supported else 'rejected', primary_both_only_milk=supported,
            numerical_sham_classification_gate_passed=all(not item['classification_changed'] for item in sham_controls),
            files_sha256={name: sha(out / name) for name in files}, scope=SCOPE), exclusive=True)
    finally:
        requests.post, simulator.measure, simulator.summarize_trial = original_post, original_measure, original_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('prepare', 'server', 'simulate'), required=True)
    args = parser.parse_args()
    baseline, out = args.output.resolve(), args.output.resolve() / 'future-current-layer-execution'
    if args.mode == 'prepare':
        _, frozen, _, _ = contract(baseline)
        out.mkdir(exist_ok=False)
        (out / 'inputs').symlink_to(frozen['inputs_symlink_target'], target_is_directory=True)
        write(out / 'provenance.json', frozen, exclusive=True)
        write(out / 'prepared.json', dict(state='prepared', provenance_sha256=sha(out / 'provenance.json')), exclusive=True)
        print('[FUTURE-CURRENT-LAYER-EXECUTION] prepared ' + str(out), flush=True)
        return
    try:
        server(baseline) if args.mode == 'server' else simulate(baseline)
    except Exception as exc:
        if out.is_dir() and not (out / 'wrapper_failed.json').exists():
            write(out / 'wrapper_failed.json', dict(mode=args.mode, error=str(exc), traceback=traceback.format_exc()), exclusive=True)
        raise


if __name__ == '__main__':
    main()
