"""Execute all twelve frozen X12 future-value q0 cases with native feedback.

Use --output BASE --mode prepare|server|simulate. Exclusively create
BASE/future-value-execution after the producer source and artifacts freeze.
No output is created while PROBE_SHA is pending. The twelve saved q0 records
execute sixteen actions each, followed by seven actual feedback predictions
at seeds196..202: 96 requests, 84 fresh predictions, 2520 fresh forwards and
77 whole two-draw noise pairs. Producer forwards360 are counted separately.

Reuse the accepted simulator execute_trials, robot controller, NormalRuntime
and e0c cached-input/strict-window helpers. Native/all_allowed/hardmask must
match the accepted first-nine-layer native/all_allowed/early_l1_9 complete
physical references. Arithmetic sham must match the accepted all_allowed
reference. Eight controls require whole records, images, all129 states and
all128 action conversions, not classification alone. Four decomposition
cases execute and are reported without selection. H_content only predicts
both value_zero cases strictly select milk; allocation_only cannot replace
this prediction. Effects are compared to
their same-V sham and separately across both V realizations; no additive
causal percentage, semantic anatomy or new network intervention is claimed.
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
ACCEPTED_SHA = 'e6d5abcb19d866f372d1373ba97a383f8093f1917859ff9714451f8e6138d40f'
POSITION_SHA = '24c237af1c137d4979c358b665f1906ae02acae464140b2a371f4e98789b76f2'
HELPER_SHA = 'e0c5073ee320bbc173eebf827126c9431043a3efaa960eb0a8ccff289cae30ed'
PROBE_SHA = '0610b1d0320512ff4f5ce4163b52337ccb3ddf3ed1eb8d6c13f51b614097b911'
PORT = 8935
PROMPT = 'pick up the milk and place it in the basket'
ARMS = ('native', 'all_allowed', 'arithmetic_sham', 'value_zero', 'allocation_only', 'hardmask')
PLAN = tuple((f'V{v}_A195_{arm}', v, arm) for arm in ARMS for v in (195, 198))
WINDOW = dict(steps=[0, 14], layers=[0, 8])
REFERENCE_ARMS = dict(native='native', all_allowed='all_allowed', arithmetic_sham='all_allowed', hardmask='early_l1_9')
PRIMARY_CASES = ('V195_A195_value_zero', 'V198_A195_value_zero')
FILES = ('states.pt', 'normalized_actions.json', 'metadata.json', 'noise-audit.pt', 'noise-audit.json',
         'observer-report.json', 'indexes-and-masks.pt', 'control-checks.json', 'site-captures.json')
PHYSICAL = ('trajectory.json', 'actions.json', 'executed_actions.json', 'normalized_actions.json', 'denormalized_actions.json')
SCOPE = ('Twelve fixed X12 future-value trajectories using saved q0 and native later feedback. '
         'Eight whole prior-reference controls and all four value/allocation cases execute. '
         'Compare each arm with its same-V sham; report both V realizations without score selection. '
         'Arithmetic and hard-mask effects are numerical decompositions, not additive causal '
         'percentages, semantic specialization, a learned brain region or a general root cause. '
         'No additional network intervention, source replacement, training or physics selection.')


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


def hooks(model):
    return [(name, tuple(module._forward_pre_hooks), tuple(module._forward_hooks))
            for name, module in model.named_modules()]


def contract(baseline):
    assert isinstance(PROBE_SHA, str) and len(PROBE_SHA) == 64 and all(c in '0123456789abcdef' for c in PROBE_SHA), 'Producer source is not frozen; no output may be created.'
    accepted_path = ROOT / 'work/run_cosmos_milk_future_current_layer_execution.py'
    position_path = ROOT / 'work/run_cosmos_milk_future_current_position_execution.py'
    helper_path = ROOT / 'work/run_cosmos_milk_future_read_execution.py'
    probe_path = ROOT / 'work/probe_cosmos_milk_future_value_mechanism.py'
    for path, expected in ((accepted_path, ACCEPTED_SHA), (position_path, POSITION_SHA), (helper_path, HELPER_SHA), (probe_path, PROBE_SHA)):
        assert sha(path) == expected
    accepted = load('value_execution_accepted_layer_stage', accepted_path)
    simulator, prior, cross, helper, accepted_out = accepted.prepared(baseline)
    old_complete = read(accepted_out / 'complete.json')
    assert old_complete['state'] == 'complete' and old_complete['script_sha256'] == ACCEPTED_SHA
    for name, expected in old_complete['files_sha256'].items():
        assert sha(accepted_out / name) == expected
    old_controls = read(accepted_out / 'controls.json')
    old_server = read(accepted_out / 'server/complete.json')
    for name, expected in old_server['files_sha256'].items():
        assert sha(accepted_out / 'server' / name) == expected
    probe = load('value_execution_frozen_producer', probe_path)
    context = probe.source_contract(baseline)
    assert isinstance(context, dict) and context['frozen_contract'] == prior['threshold_contract']
    assert probe.PLAN == PLAN and probe.ARMS == ARMS and tuple(probe.FILES) == FILES
    producer = baseline / 'future-value-mechanism'
    assert context['out'] == producer and producer.is_dir() and not (producer / 'failed.json').exists()
    docs = {name: read(producer / (name + '.json')) for name in ('complete', 'results', 'protocol', 'provenance', 'sources', 'primary-prediction')}
    complete, results, protocol, provenance = (docs[name] for name in ('complete', 'results', 'protocol', 'provenance'))
    assert complete['state'] == results['state'] == 'complete'
    assert complete['script_sha256'] == provenance['script_sha256'] == PROBE_SHA
    for name in ('results', 'protocol', 'provenance', 'sources', 'primary-prediction'):
        assert complete[name.replace('-', '_') + '_sha256'] == sha(producer / (name + '.json'))
    assert complete['fresh_q0_predictions'] == results['fresh_q0_predictions'] == 12
    assert complete['fresh_model_forwards'] == results['fresh_model_forwards'] == 360
    assert complete['total_official_dispatch_calls'] == results['total_official_dispatch_calls'] == 38070
    assert complete['original_randn_calls_consumed'] == complete['returned_source_draws_observed'] == 24
    counts = probe.expected_counts()
    assert results['official_dispatch_counts'] == complete['official_dispatch_counts'] == protocol['expected_counts'] == counts
    assert counts['model_before'] == counts['model_after'] == 360
    assert [counts[key] for key in probe.DISPATCH_KEYS] == [12960, 12960, 10800, 270, 540, 540]
    assert sum(counts[key] for key in probe.DISPATCH_KEYS) == 38070
    for key in ('two_native_and_two_allallowed_entire_accepted_records_and_boundaries_exact',
                'two_arithmetic_shams_entire_accepted_allallowed_records_and_boundaries_exact',
                'two_hardmask_entire_accepted_early_l1_9_records_and_boundaries_exact',
                'all10_masked_sites_same_union66_background', 'all360_actual_complete_model_kwargs_saved',
                'all_original_symbols_restored', 'all_owned_hooks_removed'):
        assert complete[key] is True
    assert complete['physics_calls'] == complete['later_query_predictions'] == 0
    assert complete['primary_physical_prediction_evaluated'] is results['primary_physical_prediction_evaluated'] is False
    assert protocol['trial_order'] == results['preregistered_physics_cases'] == [row[0] for row in PLAN]
    assert protocol['plan'] == [dict(case=label, vision_noise_source_seed=v, action_noise_source_seed=195, runtime_rng_seed=195,
        arm=arm, window=None if arm in ('native', 'all_allowed') else WINDOW) for label, v, arm in PLAN]
    assert protocol['primary_hypothesis'] == probe.PRIMARY and probe.PRIMARY['name'] == 'H_content'
    assert probe.PRIMARY['primary_arm'] == 'value_zero' and probe.PRIMARY['primary_cases'] == list(PRIMARY_CASES)
    assert docs['primary-prediction'] == dict(status='not_evaluated_q0_only', **probe.PRIMARY, scope=probe.SCOPE)
    assert results['primary_prediction_sha256'] == sha(producer / 'primary-prediction.json')
    physical_plan = protocol['later_physics_preregistered']
    assert physical_plan == dict(trial_order=[label for label, _, _ in PLAN], cases=12, cached_q0=12, later_noise_base=195,
        q1_to_q7_seeds=list(range(196, 203)), later_predictions=84, later_model_forwards=2520,
        total_two_stage_model_forwards=2880, no_q0_score_or_rank_selection=True)
    assert protocol['prompt'] == PROMPT and protocol['shift_cm'] == 12 and protocol['image_sha256'] == sha(context['image'])
    assert provenance['actual_transformer_sha256'] == prior['actual_transformer_sha256']
    assert provenance['actual_pipeline_source_sha256'] == prior['actual_pipeline_source_sha256']
    original = prior['threshold_contract']
    assert docs['sources'] == dict(source_evidence=context['source_evidence'], original_sources=context['original_noise_sources'], frozen_contract=original)
    assert provenance['source_sha256'] == {key: sha(path) for key, path in context['paths'].items()}
    assert context['image'].read_bytes() == (baseline / 'position-threshold/inputs/x12/input.png').read_bytes()
    stats = read(Path(original['frozen_files']['normalizer_stats']['path']))['global_raw']
    np = simulator.np
    lo, hi = np.asarray(stats['q01']), np.asarray(stats['q99'])
    assert lo.shape == hi.shape == (10,) and np.all(hi - lo > 1e-6)
    scale, offset = np.maximum(hi - lo, 1e-8) / 2, (hi + lo) / 2
    q0, rules, references = {}, {}, {}
    assert [row['case'] for row in results['cases']] == [row[0] for row in PLAN]
    for row, (label, v, arm) in zip(results['cases'], PLAN):
        folder, window = producer / label, None if arm in ('native', 'all_allowed') else WINDOW
        assert row['arm'] == arm and row['window'] == window and row['vision_noise_source_seed'] == v
        assert row['action_noise_source_seed'] == row['runtime_rng_seed'] == 195 and row['fresh_model_forwards'] == 30
        assert set(row['files_sha256']) == set(FILES)
        for name, expected in row['files_sha256'].items():
            assert sha(folder / name) == expected
        report, meta = read(folder / 'observer-report.json'), read(folder / 'metadata.json')
        assert report['state'] == 'complete' and report['case_arm'] == arm and report['selected_window'] == window
        assert report['counts'] == row['actual_dispatch_counts'] == probe.expected_counts(arm)
        assert report['counts']['model_before'] == report['counts']['model_after'] == 30
        assert report['counts']['native_UND_dispatch'] == report['counts']['native_GEN_dispatch'] == 1080
        assert report['counts']['all_allowed_dispatch'] == (0 if arm == 'native' else 1080)
        assert report['all_owned_hooks_removed'] is report['original_dispatch_symbol_restored'] is True
        for key in ('all_actual_dispatch_flatten_pre_W_byte_exact', 'future200_and_UND_native_dispatch_returns_preserved',
                    'action16_same_input_Z0_at_all_masked_sites', 'all30_actual_complete_model_kwargs_saved',
                    'all30_actual37_current_action_boundaries_finite'):
            assert report[key] is True
        assert report['original_full_266_row_projection_calls'] == 1080 and report['processor_recompute_calls'] == 0
        assert report['all_masked_sites_same_union66'] is (arm != 'native')
        assert len(row['boundary_files']) == 30 and report['boundary_files'] == row['boundary_files']
        for step, item in enumerate(row['boundary_files']):
            assert item['step'] == step and item['file'] == f'boundaries-t{step:02d}.pt'
            assert item['current_shape'] == [37, 50, 4096] and item['action_shape'] == [37, 16, 4096]
            assert sha(folder / item['file']) == item['sha256']
        assert row['site_capture_manifest_sha256'] == row['files_sha256']['site-captures.json']
        manifest = read(folder / 'site-captures.json')
        assert manifest['state'] == 'complete' and manifest['case_arm'] == arm and manifest['window'] == WINDOW
        assert manifest['sites'] == len(manifest['files']) == 135
        assert [(item['step'], item['layer_zero_based']) for item in manifest['files']] == [(t, l) for t in range(15) for l in range(9)]
        for item in manifest['files']:
            capture = folder / item['file']
            assert capture.stat().st_size == item['bytes'] < protocol['max_capture_file_bytes']
            assert sha(capture) == item['sha256']
        assert meta['case'] == label and meta['arm'] == arm and meta['model_calls'] == 30 and meta['prepare_calls'] == 1
        assert meta['seed'] == meta['action_noise_source_seed'] == meta['runtime_rng_seed'] == 195
        assert meta['vision_noise_source_seed'] == v and meta['input_png_sha256'] == protocol['image_sha256']
        normalized = np.asarray(read(folder / 'normalized_actions.json'))
        assert normalized.shape == (16, 10) and normalized.dtype == np.float64 and np.isfinite(normalized).all()
        assert normalized.tolist() == row['final_normalized_actions']
        reference_case = f'V{v}_A195_{REFERENCE_ARMS[arm]}' if arm in REFERENCE_ARMS else None
        assert row['control_reference_case'] == reference_case
        assert row['existing_entire_control_record_exact'] is (reference_case is not None)
        check = read(folder / 'control-checks.json')
        assert check['reference_case'] == reference_case
        if reference_case is not None:
            for key in ('entire_existing_q0_record_byte_exact', 'all30_existing_full_model_output_tuples_exact',
                        'all30_existing_current_action_37boundaries_exact'):
                assert check[key] is True
            assert check['comparison_folder'] == context['old_q0_references'][reference_case]['folder']
            ref_folder = accepted_out / 'closed-loop' / reference_case
            files = old_controls['actual_case_files_sha256'][reference_case]
            for name, expected in files.items():
                assert sha(ref_folder / name) == expected
            references[label] = dict(folder=str(ref_folder), reference_case=reference_case, files_current_sha256=files,
                controller_step16_path=str(ref_folder / 'controller-step16.npz'),
                controller_step16_sha256=files['controller-step16.npz'])
        else:
            assert check['state'] == 'exact' and check['first_cut_step'] == check['first_cut_layer_zero_based'] == 0
            for key in ('earlier_complete_model_outputs_and_head_prefix_exact', 'all_actual_complete_kwargs_through_first_cut_step_exact',
                        'earlier_all37_and_first_cut_before_layer_current_action_boundaries_exact', 'solver_samples_through_first_cut_input_exact'):
                assert check[key] is True
        q0[label] = dict(folder=str(folder), files_sha256=row['files_sha256'], boundary_files=row['boundary_files'],
            normalized_actions=normalized.tolist(), raw_actions=(normalized * scale + offset).tolist(), arm=arm,
            selected_window=window, masked_attention=arm != 'native', vision_noise_source_seed=v, action_noise_source_seed=195,
            site_capture_manifest_sha256=row['site_capture_manifest_sha256'])
        rules[label] = dict(arm=arm, selected_window=window, q0_vision_noise_source_seed=v, q0_action_noise_source_seed=195,
            whole_reference_control=reference_case is not None, whole_reference_case=reference_case,
            actual_query_seeds=list(range(195, 203)), later_noise_base=195, q0_whole_noise_pairing=False,
            seed_rule='Cached q0 has separate V/A sources and invocation195; q1..7=195+query.')
    frozen = dict(script_sha256=sha(Path(__file__)), accepted_execution_sha256=ACCEPTED_SHA, position_execution_sha256=POSITION_SHA,
        helper_execution_sha256=HELPER_SHA, producer_script_sha256=PROBE_SHA, threshold_contract=original,
        actual_transformer_sha256=prior['actual_transformer_sha256'], actual_pipeline_source_sha256=prior['actual_pipeline_source_sha256'],
        metrics_path=prior['metrics_path'], components_path=original['frozen_files']['components']['path'],
        producer_source_cases=prior['producer_source_cases'], native_q0_source_pair=prior['native_q0_source_pair'],
        accepted_complete_sha256=sha(accepted_out / 'complete.json'), accepted_documents_sha256=old_complete['files_sha256'],
        producer_documents_sha256={name: sha(producer / (name + '.json')) for name in docs},
        producer_protocol=protocol, source_evidence=context['source_evidence'], q0_files=list(FILES), q0_sources=q0,
        whole_physical_references=references, trials=rules, trial_order=[row[0] for row in PLAN],
        primary_hypothesis=probe.PRIMARY, primary_cases=list(PRIMARY_CASES), allocation_only_cannot_replace_primary=True,
        inputs_symlink_target=str(baseline / 'position-threshold/inputs'), url=f'http://127.0.0.1:{PORT}',
        saved_q0_predictions=12, requests=96, native_predictions=84, fresh_model_forwards=2520,
        producer_q0_forwards_separate=360, combined_two_stage_model_forwards=2880, actual_later_whole_noise_pairs_expected=77,
        q0_whole_noise_pair_checks=0, no_posthoc_case_selection=True, scope=SCOPE)
    simulator.SHIFTS, simulator.SEEDS = dict(x00=0, x09=9, x12=12), (195,)
    simulator.TRIALS = {label: ('x12', cross.QuerySeed(195, 195)) for label, _, _ in PLAN}
    simulator.URL, simulator.SCOPE = frozen['url'], SCOPE
    return simulator, frozen, cross, helper


def prepared(baseline):
    simulator, frozen, cross, helper = contract(baseline)
    out = baseline / 'future-value-execution'
    assert out.is_dir() and not (out / 'wrapper_failed.json').exists()
    assert (out / 'inputs').is_symlink() and (out / 'inputs').resolve() == Path(frozen['inputs_symlink_target']).resolve()
    assert read(out / 'provenance.json') == frozen
    saved = read(out / 'prepared.json')
    assert saved['state'] == 'prepared' and saved['provenance_sha256'] == sha(out / 'provenance.json')
    return simulator, frozen, cross, helper, out


def verify_full_inputs(record, audit, source, runtime, components):
    """Check saved actual model inputs on CPU without constructing another model."""
    torch, folder = runtime.torch, Path(source['folder'])
    clean = record['model_input']['vision_tokens'][0][0, :, 0]
    for step, item in enumerate(source['boundary_files']):
        data = torch.load(folder / item['file'], map_location='cpu', weights_only=True, mmap=True)
        kwargs = data['actual_model_kwargs']
        assert data['step'] == step and set(kwargs) == set(record['model_input'])
        expected = dict(layout=components._layout(kwargs), metadata={key: kwargs.get(key) for key in components._STRUCTURAL_KEYS})
        runtime.require_exact(data['actual_pack'], expected, 'saved complete actual pack/' + str(step))
        runtime.require_exact(data['actual_pack'], audit['model_steps'][step], 'inner observed pack/' + str(step))
        runtime.require_exact(kwargs['action_tokens'][0], record['action_states'][step, 0].to(torch.bfloat16), 'FP32 actual solver sample to BF16 kwargs/' + str(step))
        runtime.require_exact(kwargs['vision_tokens'][0][0, :, 0], clean, 'actual current clamp/' + str(step))
        runtime.require_exact(data['actual_model_action_timesteps'], kwargs['action_timesteps'], 'actual action timestep/' + str(step))
        runtime.require_exact(data['solver_sigma_from_same_frozen_schedule'], record['sigmas'][step], 'actual schedule sigma/' + str(step))
        assert type(data['model_output']) is tuple and len(data['model_output']) == 3
        runtime.require_exact(data['model_output'][2][0], record['action_velocity'][step], 'actual full tuple action head/' + str(step))
        for key, count in (('current_hidden', 50), ('action_hidden', 16)):
            value = data[key]
            assert value.shape == (37, count, 4096) and value.dtype == torch.bfloat16 and bool(torch.isfinite(value).all())
        assert torch.count_nonzero(kwargs['action_tokens'][0][:, 10:]) == 0
        if step == 0:
            runtime.require_exact(kwargs, record['model_input'], 'full first kwargs')


def server(baseline):
    _, frozen, _, helper, out = prepared(baseline)
    destination = out / 'server'
    destination.mkdir(exist_ok=False)
    served, later_noise, reference_queries, pairs, cached, fresh = [], {}, [], 0, 0, 0
    http, failed, started = None, False, time.perf_counter()
    try:
        os.environ['HF_HUB_OFFLINE'] = '1'
        paths = frozen['threshold_contract']['frozen_files']
        base = load('value_execution_normal_runtime', Path(paths['normal_runtime']['path']))
        base.TASKS['milk'] = (7, 'pick_up_the_milk_and_place_it_in_the_basket', PROMPT, 'milk_1')
        factory = load('value_execution_factory', Path(paths['factory']['path']))
        factory.OUT = destination
        assert sha(factory.SCHEDULER) == paths['scheduler']['sha256']
        runtime = base.NormalRuntime(factory)
        torch, model = runtime.torch, runtime.pipe.transformer
        assert not model.training and not model.is_cache_enabled and not torch.is_grad_enabled()
        assert sha(Path(inspect.getfile(type(model)))) == frozen['actual_transformer_sha256']
        assert sha(Path(inspect.getfile(type(runtime.pipe)))) == frozen['actual_pipeline_source_sha256']
        initial_hooks = hooks(model)
        assert all(not before and not after for _, before, after in initial_hooks)
        scan = load('value_execution_record_checks', Path(frozen['metrics_path']))
        components = load('value_execution_input_layout', Path(frozen['components_path']))
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
            verify_full_inputs(record, audit, source, runtime, components)
            assert record['actions'].tolist() == source['normalized_actions']
            q0[label] = record
            if rule['whole_reference_control']:
                ref = frozen['whole_physical_references'][label]
                path = Path(ref['folder']) / 'chunk_00/states.pt'
                assert sha(path) == ref['files_current_sha256']['chunk_00/states.pt']
                runtime.require_exact(record, torch.load(path, map_location='cpu', weights_only=True, mmap=True), 'ENTIRE prior control q0/' + label)
            checks.append(dict(case=label, arm=arm, actual_noise_preparation_initial_full_kwargs_schedule_and_all30_solver_casts_exact=True,
                entire_prior_q0_reference_exact=True if rule['whole_reference_control'] else None,
                reference_case=rule['whole_reference_case'], q0_model_forwards_in_this_execution=0, producer_files_sha256=source['files_sha256']))
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
                    assert label == frozen['trial_order'][len(served) // 8] and query == len(served) % 8
                    assert (label, query) not in served
                    rule, source = frozen['trials'][label], frozen['q0_sources'][label]
                    seed = rule['actual_query_seeds'][query]
                    assert request['seed'] == seed and request['prompt'] == PROMPT
                    image = out / 'closed-loop' / label / f'input_{query:02d}.png'
                    chunk = image.parent / f'chunk_{query:02d}'
                    if query == 0:
                        assert image.read_bytes() == (out / 'inputs/x12/input.png').read_bytes()
                        folder = Path(source['folder'])
                        chunk.mkdir(exist_ok=False)
                        for name in frozen['q0_files']:
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
                        runtime.require_exact(record[key], originals[195][key], 'actual complete native schedule/' + key)
                    assert hooks(model) == initial_hooks
                    if rule['whole_reference_control']:
                        ref = frozen['whole_physical_references'][label]
                        ref_folder, name = Path(ref['folder']), f'chunk_{query:02d}/states.pt'
                        assert sha(ref_folder / name) == ref['files_current_sha256'][name]
                        assert image.read_bytes() == (ref_folder / f'input_{query:02d}.png').read_bytes()
                        runtime.require_exact(record, torch.load(ref_folder / name, map_location='cpu', weights_only=True, mmap=True), 'ENTIRE prior query/' + label + '/' + str(query))
                        reference_queries.append(dict(case=label, query=query, reference_case=rule['whole_reference_case'],
                            entire_NormalRuntime_record_bytes_exact=True, actual_feedback_PNG_bytes_exact=True,
                            reference_states_sha256=sha(ref_folder / name), current_states_sha256=sha(chunk / 'states.pt')))
                    assert metadata['seed'] == seed and metadata['model_calls'] == 30 and metadata['prepare_calls'] == 1
                    metadata.update(trial=label, query=query, shift_cm=12, image_path=str(image), input_png_sha256=sha(image),
                        saved_q0_prediction=query == 0, current_execution_model_calls=0 if query == 0 else 30,
                        source_record_model_calls=30 if query == 0 else None, saved_q0_attention_arm=rule['arm'],
                        saved_q0_was_masked=source['masked_attention'], saved_q0_window=rule['selected_window'],
                        q0_vision_noise_source_seed=rule['q0_vision_noise_source_seed'], q0_action_noise_source_seed=195,
                        q0_runtime_rng_seed=195, later_noise_base=195, actual_query_seeds=rule['actual_query_seeds'], seed_rule=rule['seed_rule'],
                        whole_prior_control=rule['whole_reference_control'], whole_prior_reference_case=rule['whole_reference_case'],
                        component_intervention_in_this_execution=False, attention_mask_intervention_in_this_execution=False,
                        value_intervention_in_this_execution=False, initial_noise_replacement_in_this_execution=False,
                        paired_noise_checked_exact=paired, saved_q0_source_states_sha256=source['files_sha256']['states.pt'] if query == 0 else None,
                        saved_q0_site_captures_source_folder=source['folder'] if query == 0 else None,
                        saved_q0_site_capture_manifest_sha256=source['site_capture_manifest_sha256'] if query == 0 else None,
                        site_capture_arrays_copied_to_execution=False)
                    write(chunk / 'metadata.json', metadata)
                    assert read(chunk / 'normalized_actions.json') == record['actions'].tolist()
                    served.append((label, query))
                    write(destination / 'progress.json', dict(queries=len(served), saved_q0_predictions=cached, native_predictions=fresh,
                        fresh_model_forwards=30 * fresh, actual_later_whole_noise_pairs=pairs, served_queries=served, elapsed_s=time.perf_counter() - started))
                    print('[FUTURE-VALUE-EXECUTION] ' + json.dumps(dict(requests=len(served), case=label, query=query, seed=seed)), flush=True)
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
        assert [(row['case'], row['query']) for row in reference_queries] == [(label, q) for label, _, arm in PLAN if arm in REFERENCE_ARMS for q in range(8)]
        assert hooks(model) == initial_hooks
        write(destination / 'whole-reference-query-controls.json', dict(state='exact', queries=reference_queries,
            eight_controls_all8_entire_records_and_PNGs_exact=True), exclusive=True)
        names = ('provenance.json', 'saved-q0-controls.json', 'whole-reference-query-controls.json')
        write(destination / 'complete.json', dict(state='complete', queries=96, saved_q0_predictions=12, native_predictions=84,
            fresh_model_forwards=2520, actual_later_whole_noise_pairs=77, q0_whole_noise_pair_checks=0, extra_q0_model_forwards=0,
            eight_controls_all8_entire_records_and_PNGs_exact=True, final_model_hooks_none=True,
            component_hooks_installed=False, attention_dispatch_wrapper_installed=False,
            files_sha256={name: sha(destination / name) for name in names}, scope=SCOPE), exclusive=True)
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
    collector = simulator.load_file('value_execution_collector', ROOT / 'work/collect_cosmos_goal_pair.py')
    rollout = simulator.load_file('value_execution_controller', ROOT / 'work/rollout_cosmos_goal_pair.py')
    np = simulator.np
    stats = read(Path(frozen['threshold_contract']['frozen_files']['normalizer_stats']['path']))['global_raw']
    lo, hi = np.asarray(stats['q01']), np.asarray(stats['q99'])
    scale, offset = np.maximum(hi - lo, 1e-8) / 2, (hi + lo) / 2
    initial_state = np.load(out / 'inputs/x12/state.npy', allow_pickle=False)
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
            initial = np.load(out / 'inputs/x12/state.npy', allow_pickle=False)
            assert state.shape == (17, initial.size) and state.dtype == initial.dtype and rollout.array_exact(state[0], initial)
            norm, raw = np.asarray(source['normalized_actions']), np.asarray(source['raw_actions'])
            native = np.asarray([_remap_gripper(_framewise_action_to_delta(item, '6d').tolist(), 'zero_one') for item in raw])
            for name, value in {'normalized_actions.json': norm, 'denormalized_actions.json': raw, 'actions.json': native, 'executed_actions.json': np.clip(native, -1, 1)}.items():
                assert rollout.array_exact(np.asarray(read(folder / name)), value), ('own cached q0 execution', active, name)
            trajectory = read(folder / 'trajectory.json')
            assert len(trajectory) == 17 and [item['step'] for item in trajectory] == list(range(17))
            assert all(item['exact'] for item in read(folder / 'initial_checks.json'))
            assert all(value is True for value in read(folder / 'initial_controller_checks.json').values())
            assert read(folder / 'initial_contact_checks.json') == read(out / 'inputs/x12/initial_contact_checks.json')
            with np.load(folder / 'initial_controller.npz', allow_pickle=False) as actual:
                with np.load(out / 'inputs/x12/controller.npz', allow_pickle=False) as source_controller:
                    assert set(actual.files) == set(source_controller.files)
                    assert all(rollout.array_exact(actual[key], source_controller[key]) for key in actual.files)
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
            states = np.load(folder / 'sim_states.npy', allow_pickle=False)
            assert states.shape == (129, initial_state.size) and states.dtype == initial_state.dtype and np.isfinite(states).all()
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
        assert done['cases'] == 12 and done['completed_trials'] == [row[0] for row in PLAN]
        assert done['steps_per_case'] == 128 and done['state_records_per_case'] == 129
        assert len(whole_controls) == 8 and len(entries) == len(windows) == 12
        assert served['queries'] == 96 and served['saved_q0_predictions'] == 12 and served['native_predictions'] == 84
        assert served['fresh_model_forwards'] == 2520 and served['actual_later_whole_noise_pairs'] == 77
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
                only_milk=actual['selected_objects'] == ['milk_1'], sham_only_milk=sham['selected_objects'] == ['milk_1'],
                first_selection_window_start=actual['first_selection_step'], per_object=actual['per_object']))
        paired = {}
        for arm in ('value_zero', 'allocation_only'):
            rows = [row for row in comparisons if row['arm'] == arm]
            assert [row['vision_noise_source_seed'] for row in rows] == [195, 198]
            paired[arm] = dict(cases=[row['case'] for row in rows], both_only_milk=all(row['only_milk'] for row in rows),
                both_shams_only_milk=all(row['sham_only_milk'] for row in rows),
                both_classifications_changed_vs_own_sham=all(row['classification_changed_vs_sham'] for row in rows),
                direction_agreement_in_only_milk=int(rows[0]['only_milk']) - int(rows[0]['sham_only_milk']) == int(rows[1]['only_milk']) - int(rows[1]['sham_only_milk']),
                numerical_partition_is_not_additive_causal_percentage=True)
        primary_rows = [row for row in comparisons if row['case'] in PRIMARY_CASES]
        assert [row['case'] for row in primary_rows] == list(PRIMARY_CASES)
        sham_controls = []
        for v in (195, 198):
            native = read(out / 'closed-loop' / f'V{v}_A195_native' / 'summary.json')['selected_objects']
            sham = read(out / 'closed-loop' / f'V{v}_A195_all_allowed' / 'summary.json')['selected_objects']
            arithmetic = read(out / 'closed-loop' / f'V{v}_A195_arithmetic_sham' / 'summary.json')['selected_objects']
            sham_controls.append(dict(vision_noise_source_seed=v, native_selected_objects=native,
                all_allowed_selected_objects=sham, arithmetic_sham_selected_objects=arithmetic,
                same_V_classifications_preserved=all(set(value) == set(native) for value in (sham, arithmetic))))
        sham_gate = all(row['same_V_classifications_preserved'] for row in sham_controls)
        content_success = paired['value_zero']['both_only_milk']
        allocation_success = paired['allocation_only']['both_only_milk']
        hardmask_success = all(row['only_milk'] for row in comparisons if row['arm'] == 'hardmask')
        interpretation = ('both_splits_succeed_exclusive_attribution_not_established' if content_success and allocation_success
            else 'both_splits_fail_hardmask_succeeds_joint_or_interactive_effects_pending' if not content_success and not allocation_success and hardmask_success
            else 'split_outcomes_reported_without_exclusive_causal_attribution')
        write(out / 'primary-prediction.json', dict(state='evaluated_after_all12_complete', name='H_content', primary_arm='value_zero',
            registered_hypothesis=frozen['primary_hypothesis'], cases=primary_rows,
            prediction='supported' if content_success and sham_gate else 'rejected', both_value_zero_strict_only_milk=content_success,
            numerical_sham_classification_gate_passed=sham_gate, eight_whole_byte_controls_passed=True,
            allocation_only_cases_reported_without_replacing_primary=True, all12_cases_executed=True,
            decomposition_interpretation=interpretation, exclusive_content_attribution_established=False,
            functional_specialization_established=False, scope=SCOPE), exclusive=True)
        write(out / 'comparisons.json', dict(state='evaluated_after_all12_complete', cases=comparisons,
            both_V_decomposition_reports=paired, all_four_value_allocation_cases_reported=True,
            no_q0_or_physics_ranking_or_selection=True, same_V_sham_comparison_required=True,
            two_V_consistency_is_a_limit_not_a_root_cause_claim=True, decomposition_interpretation=interpretation,
            primary_prediction_sha256=sha(out / 'primary-prediction.json'), scope=SCOPE), exclusive=True)
        case_files = {}
        for label, _, _ in PLAN:
            folder = out / 'closed-loop' / label
            names = ('sim_states.npy', 'actual.mp4', 'initial_controller.npz', 'controller-step16.npz', 'summary.json', 'strict-windows.json',
                'q1-entry-control.json', 'query_contract.json', 'initial_checks.json', 'initial_controller_checks.json', 'initial_contact_checks.json',
                *PHYSICAL, *[f'input_{q:02d}.png' for q in range(8)],
                *[f'chunk_{q:02d}/{name}' for q in range(8) for name in ('states.pt', 'metadata.json', 'normalized_actions.json')],
                *[f'chunk_00/{name}' for name in FILES if name not in ('states.pt', 'metadata.json', 'normalized_actions.json')],
                *[f"chunk_00/{item['file']}" for item in frozen['q0_sources'][label]['boundary_files']])
            case_files[label] = {name: sha(folder / name) for name in names}
        write(out / 'controls.json', dict(eight_whole_prior_controls=whole_controls, q1_entry_controls=entries, strict_windows=windows,
            sham_classification_controls=sham_controls, numerical_sham_classification_gate_passed=sham_gate,
            actual_case_files_sha256=case_files, comparisons_sha256=sha(out / 'comparisons.json'),
            primary_prediction_sha256=sha(out / 'primary-prediction.json')), exclusive=True)
        names = ('controls.json', 'comparisons.json', 'primary-prediction.json', 'provenance.json', 'prepared.json', 'server/complete.json', 'closed-loop/complete.json', 'summary.json')
        write(out / 'complete.json', dict(state='complete', script_sha256=sha(Path(__file__)), trial_order=[row[0] for row in PLAN],
            physical_trials=12, saved_q0_predictions=12, native_predictions=84, fresh_model_forwards=2520,
            producer_q0_forwards_separate=360, combined_two_stage_model_forwards=2880, actual_later_whole_noise_pairs=77,
            eight_entire129_state_5JSON_8PNG_and_all8_modelrecords_controls_exact=True,
            actual_physical_actions=1536, all12_own_cached_q0_action_and_strict_window_gates_passed=True,
            all12_executed_without_score_selection=True, all_four_value_allocation_cases_reported=True,
            primary_hypothesis='H_content', primary_prediction='supported' if content_success and sham_gate else 'rejected',
            both_value_zero_strict_only_milk=content_success, numerical_sham_classification_gate_passed=sham_gate,
            numerical_partition_is_not_additive_causal_percentage=True, files_sha256={name: sha(out / name) for name in names}, scope=SCOPE), exclusive=True)
    finally:
        requests.post, simulator.measure, simulator.summarize_trial = original_post, original_measure, original_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('prepare', 'server', 'simulate'), required=True)
    args = parser.parse_args()
    baseline, out = args.output.resolve(), args.output.resolve() / 'future-value-execution'
    if args.mode == 'prepare':
        _, frozen, _, _ = contract(baseline)
        out.mkdir(exist_ok=False)
        (out / 'inputs').symlink_to(frozen['inputs_symlink_target'], target_is_directory=True)
        write(out / 'provenance.json', frozen, exclusive=True)
        write(out / 'prepared.json', dict(state='prepared', provenance_sha256=sha(out / 'provenance.json')), exclusive=True)
        print('[FUTURE-VALUE-EXECUTION] prepared ' + str(out), flush=True)
        return
    try:
        server(baseline) if args.mode == 'server' else simulate(baseline)
    except Exception as exc:
        if out.is_dir() and not (out / 'wrapper_failed.json').exists():
            write(out / 'wrapper_failed.json', dict(mode=args.mode, error=str(exc), traceback=traceback.format_exc()), exclusive=True)
        raise


if __name__ == '__main__':
    main()
