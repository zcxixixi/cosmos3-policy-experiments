"""Execute all ten preregistered future-read q0 cases, with native later feedback.

Use --output BASE --mode prepare|server|simulate. Exclusively create
BASE/future-read-execution. Require completed frozen c106 q0 evidence before
any stage. Serve ten saved q0 results without a model call; run queries1..7
from each actual new image at seeds196..202. Eighty replies / 70 fresh
predictions / 2100 forwards / ten actual 128-step physical trajectories.

Reuse the frozen 985f execution contract, simulator execute_trials, controller
and NormalRuntime. Two untouched native cases must equal the real fixed-A195
V195/A195 and V198/A195 references. All eight masked cases still execute if
a numerical sham changes the grasp classification; that failure is recorded
as an interpretation limit. Both-cut physical and later-model equality is a
hard engineering gate. No new internal hooks, training or candidate selection.
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
PROBE_SHA = 'c106bba687c6eaf0853ac9abe7aa2767f9464365162399c201e98b2369af7a4a'
PORT, LATER_BASE = 8931, 195
PROMPT = 'pick up the milk and place it in the basket'
ARMS = ('native', 'all_allowed', 'cut_future_current', 'cut_future_action', 'cut_future_both')
PLAN = tuple((f'V{v}_A195_{arm}', v, arm) for arm in ARMS for v in (195, 198))
FILES = ('states.pt', 'normalized_actions.json', 'metadata.json', 'noise-audit.pt', 'noise-audit.json',
         'observer-report.json', 'indexes-and-masks.pt')
PHYSICAL = ('trajectory.json', 'actions.json', 'executed_actions.json', 'normalized_actions.json', 'denormalized_actions.json')
BOTH = ('V195_A195_cut_future_both', 'V198_A195_cut_future_both')
SCOPE = ('Ten complete X+12cm physical trajectories executing the frozen ten saved '
         'future-read q0 predictions, then native feedback inference at later base195. '
         'Both numerical shams are retained and all cases execute in preregistered order. '
         'No additional network intervention or training. Sham classification changes '
         'invalidate original-native-path interpretation without changing the source '
         'pair or selecting another candidate. Equality gates concern actual arrays '
         'and physical execution, not semantic identity or a neural root cause.')


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
    probe_path = ROOT / 'work/probe_cosmos_milk_future_read_cuts.py'
    assert sha(legacy_path) == LEGACY_SHA and sha(probe_path) == PROBE_SHA
    legacy = load('future_read_execution_frozen_initial_execution', legacy_path)
    simulator, prior, cross, old_execution = legacy.prepared(baseline)
    old_complete = read(old_execution / 'complete.json')
    assert old_complete['state'] == 'complete' and old_complete['script_sha256'] == LEGACY_SHA
    for name, expected in old_complete['files_sha256'].items():
        assert sha(old_execution / name) == expected
    probe = load('future_read_execution_frozen_producer', probe_path)
    _, original, threshold_sources, image, _, current_anchors = probe.preflight(baseline)
    assert original == prior['threshold_contract']
    producer = baseline / 'future-read-cuts'
    assert producer.is_dir() and not (producer / 'failed.json').exists()
    complete, results, protocol, provenance, sources, closure = (read(producer / (name + '.json')) for name in
        ('complete', 'results', 'protocol', 'provenance', 'sources', 'closure'))
    assert complete['state'] == results['state'] == 'complete'
    assert complete['script_sha256'] == provenance['script_sha256'] == PROBE_SHA
    for name in ('results', 'protocol', 'provenance', 'sources', 'closure'):
        assert complete[name + '_sha256'] == sha(producer / (name + '.json'))
    assert complete['fresh_q0_predictions'] == results['fresh_q0_predictions'] == 10
    assert complete['fresh_model_forwards'] == results['fresh_model_forwards'] == 300
    assert complete['total_official_dispatch_calls'] == results['total_official_dispatch_calls'] == 36720
    assert complete['extra_official_dispatch_calls'] == 15120
    assert complete['original_randn_calls_consumed'] == complete['returned_source_draws_observed'] == 20
    for key in ('two_original_native_entire_q0_records_byte_exact', 'all_masked_actual_site_unblocked_queries_exact_vs_same_input_sham',
                'all_future200_and_UND_native_dispatch_rows_retained', 'both_engineering_action_and_boundary_equality_passed',
                'original_noise_and_dispatch_symbols_restored', 'all_owned_hooks_removed'):
        assert complete[key] is True
    assert complete['physics_calls'] == complete['later_query_predictions'] == 0
    assert results['official_dispatch_counts'] == dict(model_before=300, model_after=300, native_UND_dispatch=10800,
        native_GEN_dispatch=10800, all_allowed_dispatch=8640, cut_dispatch=6480)
    assert protocol['source_pair'] == ['V195_A195', 'V198_A195'] and protocol['action_noise_source_seed'] == 195
    expected_plan = [dict(case=name, vision_noise_source_seed=v, action_noise_source_seed=195, runtime_rng_seed=195, arm=arm) for name, v, arm in PLAN]
    assert protocol['plan'] == expected_plan and results['preregistered_physics_cases'] == [name for name, _, _ in PLAN]
    physical_plan = protocol['later_physics_preregistered']
    assert physical_plan['trial_order'] == [name for name, _, _ in PLAN]
    assert physical_plan['cases'] == physical_plan['cached_q0'] == 10
    assert physical_plan['later_noise_base'] == 195 and physical_plan['q1_to_q7_seeds'] == list(range(196, 203))
    assert physical_plan['later_predictions'] == 70 and physical_plan['later_model_forwards'] == 2100
    assert physical_plan['total_two_stage_model_forwards'] == 2400 and physical_plan['no_selection_after_q0_scores'] is True
    assert protocol['image_sha256'] == sha(image) and protocol['prompt'] == PROMPT and protocol['shift_cm'] == 12
    assert sources['anchors'] == current_anchors and sources['threshold_sources'] == threshold_sources and sources['frozen_contract'] == original
    assert provenance['actual_transformer_sha256'] == prior['actual_transformer_sha256']
    assert provenance['actual_pipeline_source_sha256'] == prior['actual_pipeline_source_sha256']
    for key, expected in provenance['source_sha256'].items():
        path = Path(original['frozen_files'][key]['path']) if key in original['frozen_files'] else ROOT / 'work' / ('probe_cosmos_milk_component_pulses.py' if key == 'metrics' else 'probe_cosmos_milk_target_reader_inputs.py')
        assert sha(path) == expected
    assert closure['state'] == 'checked' and closure['physics_executed'] is False
    assert all(item['byte_exact'] for item in closure['both_record_fields'].values())
    assert set(closure['both_record_fields']) == {'readout', 'action_velocity', 'action_states', 'actions'}
    assert len(closure['both_boundary_checks']) == 60 and all(item['byte_exact'] for item in closure['both_boundary_checks'])
    assert set(closure['single_first_layer_checks']) == {'cut_future_current', 'cut_future_action'}
    assert all(len(items) == 1 and items[0]['byte_exact'] for items in closure['single_first_layer_checks'].values())
    np = simulator.np
    stats = read(Path(original['frozen_files']['normalizer_stats']['path']))['global_raw']
    lo, hi = np.asarray(stats['q01']), np.asarray(stats['q99'])
    assert lo.shape == hi.shape == (10,) and np.all(hi - lo > 1e-6)
    scale, offset = np.maximum(hi - lo, 1e-8) / 2, (hi + lo) / 2
    assert [row['case'] for row in results['cases']] == [name for name, _, _ in PLAN]
    q0, rules = {}, {}
    for row, (label, v, arm) in zip(results['cases'], PLAN):
        folder = producer / label
        assert row['arm'] == arm and row['vision_noise_source_seed'] == v
        assert row['action_noise_source_seed'] == row['runtime_rng_seed'] == 195 and row['fresh_model_forwards'] == 30
        assert row['entire_native_q0_record_byte_exact'] is (True if arm == 'native' else None)
        assert row['all_noise_preparation_schedule_padding_and_dispatch_footprint_gates_passed'] is True
        assert set(row['files_sha256']) == set(FILES)
        for name, expected in row['files_sha256'].items():
            assert sha(folder / name) == expected
        report, meta = read(folder / 'observer-report.json'), read(folder / 'metadata.json')
        expected_counts = dict(model_before=30, model_after=30, block_before=1080, block_after=1080, pre_W=1080,
            native_UND_dispatch=1080, native_GEN_dispatch=1080, all_allowed_dispatch=0 if arm == 'native' else 1080,
            cut_dispatch=1080 if arm.startswith('cut_') else 0, substituted_sites=0 if arm == 'native' else 1080)
        assert report['counts'] == row['actual_dispatch_counts'] == expected_counts
        assert report['state'] == 'complete' and report['arm'] == arm
        assert report['all_actual_dispatch_return_pre_W_byte_exact'] is True and report['original_full_266_row_GEMM_calls'] == 1080
        assert report['same_union66_substitution_every_masked_site'] is (arm != 'native')
        assert report['original_dispatch_symbol_restored'] is report['all_owned_hooks_removed'] is True
        assert len(row['boundary_files']) == 30 and report['boundary_files'] == row['boundary_files']
        for step, item in enumerate(row['boundary_files']):
            assert item['step'] == step and item['file'] == f'boundaries-t{step:02d}.pt'
            assert item['current_shape'] == [37, 50, 4096] and item['action_shape'] == [37, 16, 4096]
            assert sha(folder / item['file']) == item['sha256']
        assert meta['case'] == label and meta['arm'] == arm and meta['seed'] == meta['action_noise_source_seed'] == 195
        assert meta['vision_noise_source_seed'] == v and meta['model_calls'] == 30 and meta['prepare_calls'] == 1
        assert meta['numerical_masked_sham'] is (arm == 'all_allowed') and meta['native_noop'] is (arm == 'native')
        assert meta['input_png_sha256'] == protocol['image_sha256']
        normalized = np.asarray(read(folder / 'normalized_actions.json'))
        assert normalized.shape == (16, 10) and np.isfinite(normalized).all() and normalized.tolist() == row['final_normalized_actions']
        q0[label] = dict(folder=str(folder), files_sha256=row['files_sha256'], boundary_files=row['boundary_files'],
            normalized_actions=normalized.tolist(), raw_actions=(normalized * scale + offset).tolist(),
            arm=arm, masked_attention=arm != 'native', vision_noise_source_seed=v, action_noise_source_seed=195,
            q0_masked_description=dict(actual_substituted_sites=expected_counts['substituted_sites'],
                common_union_query_rows=66 if arm != 'native' else 0, future200_retained_native=True, UND_retained_native=True,
                full_GEN_output_projection_shape=[266, 4096], original_projection_calls=1080,
                cut_unblocked_query_gate_vs_same_input_all_allowed=arm.startswith('cut_'), residual_compensation=False))
        rules[label] = dict(arm=arm, q0_vision_noise_source_seed=v, q0_action_noise_source_seed=195,
            native_control=arm == 'native', numerical_sham=arm == 'all_allowed', later_noise_base=195,
            actual_query_seeds=list(range(195, 203)), seed_rule='Saved q0 has separate V/A sources and invocation195; q1..7=195+query.',
            q0_whole_noise_pairing=False)
    native_refs = {}
    for label, v, arm in PLAN[:2]:
        source = dict(current_anchors['physical_references'][str(v)])
        folder = Path(source['folder'])
        source['controller_step16_path'] = str(folder / 'controller-step16.npz')
        source['controller_step16_sha256'] = sha(folder / 'controller-step16.npz')
        native_refs[label] = source
    frozen = dict(script_sha256=sha(Path(__file__)), legacy_execution_sha256=LEGACY_SHA, producer_script_sha256=PROBE_SHA,
        threshold_contract=original, actual_transformer_sha256=prior['actual_transformer_sha256'],
        actual_pipeline_source_sha256=prior['actual_pipeline_source_sha256'], metrics_path=prior['metrics_path'],
        producer_source_cases=prior['producer_source_cases'],
        native_q0_source_pair=current_anchors['source_pair'], native_physical_references=native_refs,
        prior_execution_complete_sha256=sha(old_execution / 'complete.json'), prior_execution_files_sha256=old_complete['files_sha256'],
        producer_documents_sha256={name: sha(producer / (name + '.json')) for name in ('complete', 'results', 'provenance', 'protocol', 'sources', 'closure')},
        producer_protocol=protocol, q0_sources=q0, trials=rules, trial_order=[name for name, _, _ in PLAN],
        inputs_symlink_target=str(baseline / 'position-threshold/inputs'), url=f'http://127.0.0.1:{PORT}',
        saved_q0_predictions=10, native_predictions=70, fresh_model_forwards=2100, producer_q0_forwards_separate=300,
        actual_later_whole_noise_pairs_expected=63, q0_whole_noise_pair_checks=0, scope=SCOPE)
    simulator.SHIFTS, simulator.SEEDS = dict(x00=0, x09=9, x12=12), (195,)
    simulator.TRIALS = {label: ('x12', cross.QuerySeed(195, 195)) for label, _, _ in PLAN}
    simulator.URL, simulator.SCOPE = frozen['url'], SCOPE
    return simulator, frozen, cross


def prepared(baseline):
    simulator, frozen, cross = contract(baseline)
    out = baseline / 'future-read-execution'
    assert out.is_dir() and not (out / 'wrapper_failed.json').exists()
    assert (out / 'inputs').is_symlink() and (out / 'inputs').resolve() == Path(frozen['inputs_symlink_target']).resolve()
    assert read(out / 'provenance.json') == frozen
    saved = read(out / 'prepared.json')
    assert saved['state'] == 'prepared' and saved['provenance_sha256'] == sha(out / 'provenance.json')
    return simulator, frozen, cross, out


def verify_cached(actual, audit, originals, native_pair, arm, v, runtime, scan):
    """Noise/preparation gates apply to all arms; whole-output exact only to native."""
    runtime.require_exact(actual['pure_noise'], [originals[v]['pure_noise'][0], originals[195]['pure_noise'][1]], 'cached selected raw V/A draw pair')
    if arm == 'native':
        runtime.require_exact(actual, native_pair[v], 'native ENTIRE fixed-A195 source q0')
    p = actual['prepared_latents_and_masks']
    assert len(p) == 12 and p[10] == 10
    runtime.require_exact(p[0], originals[v]['prepared_latents_and_masks'][0], 'cached prepared V source')
    runtime.require_exact(p[2], originals[195]['prepared_latents_and_masks'][2], 'cached prepared A195 source')
    for i in range(12):
        if i not in (0, 2):
            runtime.require_exact(p[i], originals[195]['prepared_latents_and_masks'][i], 'cached common preparation/' + str(i))
    kwargs = actual['model_input']
    runtime.require_exact(kwargs['vision_tokens'], originals[v]['model_input']['vision_tokens'], 'cached initial V kwargs')
    runtime.require_exact(kwargs['action_tokens'], originals[195]['model_input']['action_tokens'], 'cached initial A kwargs')
    for key in kwargs:
        if key not in ('vision_tokens', 'action_tokens'):
            runtime.require_exact(kwargs[key], originals[195]['model_input'][key], 'cached common initial kwargs/' + key)
    runtime.require_exact(audit['initial_model_kwargs'], kwargs, 'cached actual first-model observation')
    assert len(audit['calls']) == 2 and len(audit['model_steps']) == 30
    assert audit['original_symbol_restored'] is audit['observer_hooks_removed'] is audit['dispatch_symbol_restored'] is True
    for i, call in enumerate(audit['calls']):
        assert call['ordinal'] == i and call['original_generated_seed'] == 195
        assert call['returned_source_seed'] == (v if i == 0 else 195)
        runtime.require_exact(call['original_generated_discarded'], originals[195]['pure_noise'][i], 'actual original consumed A195 draw')
        runtime.require_exact(call['actual_returned'], actual['pure_noise'][i], 'actual returned/inner observed draw')
        for key, value in (('original_generated_metadata', call['original_generated_discarded']), ('returned_metadata', call['actual_returned'])):
            for field, expected in scan.tensor_meta(value, runtime.torch).items():
                assert call[key][field] == expected
    runtime.require_exact(audit['calls'][0]['generator_state_after'], audit['calls'][1]['generator_state_before'], 'actual RNG chain')
    for key in ('timesteps', 'sigmas'):
        runtime.require_exact(actual[key], originals[195][key], 'cached full schedule/' + key)
    assert runtime.torch.count_nonzero(actual['action_states'][..., 10:]) == 0


def server(baseline):
    _, frozen, _, out = prepared(baseline)
    destination = out / 'server'
    destination.mkdir(exist_ok=False)
    served, later_noise, pairs, cached, fresh, both_queries = [], {}, 0, 0, 0, []
    failed, http, started = False, None, time.perf_counter()
    try:
        os.environ['HF_HUB_OFFLINE'] = '1'
        paths = frozen['threshold_contract']['frozen_files']
        base = load('future_read_execution_normal_runtime', Path(paths['normal_runtime']['path']))
        base.TASKS['milk'] = (7, 'pick_up_the_milk_and_place_it_in_the_basket', PROMPT, 'milk_1')
        factory = load('future_read_execution_factory', Path(paths['factory']['path']))
        factory.OUT = destination
        assert sha(factory.SCHEDULER) == paths['scheduler']['sha256']
        runtime = base.NormalRuntime(factory)
        torch, model = runtime.torch, runtime.pipe.transformer
        assert not model.training and not model.is_cache_enabled and not torch.is_grad_enabled()
        assert sha(Path(inspect.getfile(type(model)))) == frozen['actual_transformer_sha256']
        assert sha(Path(inspect.getfile(type(runtime.pipe)))) == frozen['actual_pipeline_source_sha256']
        scan = load('future_read_execution_record_checks', Path(frozen['metrics_path']))
        originals, native_pair, q0 = {}, {}, {}
        for seed in (195, 198):
            source = frozen['producer_source_cases'][str(seed)]
            folder = Path(source['q0_folder'])
            assert sha(folder / 'states.pt') == source['q0_files_current_sha256']['states.pt']
            originals[seed] = torch.load(folder / 'states.pt', map_location='cpu', weights_only=True, mmap=True)
            source = frozen['native_q0_source_pair'][str(seed)]
            native_folder = Path(source['folder'])
            assert sha(native_folder / 'states.pt') == source['files_sha256']['states.pt']
            native_pair[seed] = torch.load(native_folder / 'states.pt', map_location='cpu', weights_only=True, mmap=True)
        controls = []
        for label, v, arm in PLAN:
            source, folder = frozen['q0_sources'][label], Path(frozen['q0_sources'][label]['folder'])
            actual = torch.load(folder / 'states.pt', map_location='cpu', weights_only=True, mmap=True)
            audit = torch.load(folder / 'noise-audit.pt', map_location='cpu', weights_only=True, mmap=True)
            scan.validate_record(actual, torch)
            verify_cached(actual, audit, originals, native_pair, arm, v, runtime, scan)
            assert actual['actions'].tolist() == source['normalized_actions']
            q0[label] = actual
            controls.append(dict(case=label, arm=arm, actual_noise_preparation_initial_kwargs_and_schedule_exact=True,
                native_entire_fixed_A195_source_record_exact=True if arm == 'native' else None,
                q0_model_forwards_in_this_execution=0, producer_files_sha256=source['files_sha256'],
                saved_masked_description=source['q0_masked_description']))
        runtime.require_exact(q0[BOTH[0]]['actions'], q0[BOTH[1]]['actions'], 'Both saved q0 all10 actions byte exact')
        write(destination / 'saved-q0-controls.json', dict(state='exact', cases=controls, both_q0_all10_actions_exact=True,
            q0_whole_noise_pair_checks=0, same_seed_does_not_imply_same_whole_V_A_pair=True, extra_q0_model_forwards=0), exclusive=True)
        write(destination / 'provenance.json', dict(script_sha256=sha(Path(__file__)), execution_provenance_sha256=sha(out / 'provenance.json'),
            saved_q0_controls_sha256=sha(destination / 'saved-q0-controls.json'), component_hooks_installed=False,
            attention_dispatch_wrapper_installed=False, saved_q0=10, native_later_predictions=70, expected_fresh_model_forwards=2100, scope=SCOPE), exclusive=True)

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
                    expected_case = frozen['trial_order'][len(served) // 8]
                    assert label == expected_case and query == len(served) % 8, 'Fixed trial/query order, no score selection'
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
                        cached += 1
                        paired = False
                    else:
                        if label == BOTH[1]:
                            assert image.read_bytes() == (out / 'closed-loop' / BOTH[0] / f'input_{query:02d}.png').read_bytes(), 'Both actual later feedback PNG differs'
                        record, metadata = runtime.predict('milk', image, seed, chunk)
                        fresh += 1
                        paired = query in later_noise
                        if paired:
                            runtime.require_exact(record['pure_noise'], later_noise[query], 'actual later195 two-draw pair/' + str(query))
                            pairs += 1
                        else:
                            later_noise[query] = record['pure_noise']
                        if label == BOTH[1]:
                            reference = out / 'closed-loop' / BOTH[0] / f'chunk_{query:02d}/states.pt'
                            runtime.require_exact(record, torch.load(reference, map_location='cpu', weights_only=True, mmap=True), 'Both ENTIRE later modelrecord/' + str(query))
                            both_queries.append(dict(query=query, actual_feedback_PNG_bytes_exact=True, entire_NormalRuntime_record_bytes_exact=True,
                                left_states_sha256=sha(reference), right_states_sha256=sha(chunk / 'states.pt')))
                    scan.validate_record(record, torch)
                    for key in ('timesteps', 'sigmas'):
                        runtime.require_exact(record[key], originals[195][key], 'complete actual native schedule/' + key)
                    if rule['native_control']:
                        reference = frozen['native_physical_references'][label]
                        folder = Path(reference['folder'])
                        assert image.read_bytes() == (folder / f'input_{query:02d}.png').read_bytes()
                        name = f'chunk_{query:02d}/states.pt'
                        assert sha(folder / name) == reference['files_current_sha256'][name]
                        runtime.require_exact(record, torch.load(folder / name, map_location='cpu', weights_only=True, mmap=True), 'native ENTIRE fixed-A195 reference query/' + label + '/' + str(query))
                    assert metadata['seed'] == seed and metadata['model_calls'] == 30 and metadata['prepare_calls'] == 1
                    metadata.update(trial=label, query=query, shift_cm=12, image_path=str(image), input_png_sha256=sha(image),
                        saved_q0_prediction=query == 0, current_execution_model_calls=0 if query == 0 else 30,
                        source_record_model_calls=30 if query == 0 else None, saved_q0_attention_arm=rule['arm'],
                        saved_q0_was_masked=source['masked_attention'], q0_vision_noise_source_seed=rule['q0_vision_noise_source_seed'], q0_action_noise_source_seed=195,
                        later_noise_base=195, actual_query_seeds=rule['actual_query_seeds'], seed_rule=rule['seed_rule'],
                        component_intervention_in_this_execution=False, attention_mask_intervention_in_this_execution=False,
                        initial_noise_replacement_in_this_execution=False, paired_noise_checked_exact=paired,
                        saved_q0_source_states_sha256=source['files_sha256']['states.pt'] if query == 0 else None,
                        saved_q0_masked_description=source['q0_masked_description'] if query == 0 else None)
                    write(chunk / 'metadata.json', metadata)
                    assert read(chunk / 'normalized_actions.json') == record['actions'].tolist()
                    served.append((label, query))
                    write(destination / 'progress.json', dict(queries=len(served), saved_q0_predictions=cached, native_predictions=fresh,
                        fresh_model_forwards=30 * fresh, actual_later_whole_noise_pairs=pairs, served_queries=served, elapsed_s=time.perf_counter() - started))
                    print('[FUTURE-READ-EXECUTION] ' + json.dumps(dict(requests=len(served), case=label, query=query, seed=seed)), flush=True)
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
        while not failed and len(served) < 80:
            http.handle_request()
            assert not (out / 'closed-loop/failed.json').exists() and not (out / 'wrapper_failed.json').exists()
        assert not failed and len(served) == 80 and cached == 10 and fresh == 70 and pairs == 63
        assert [item['query'] for item in both_queries] == list(range(1, 8))
        write(destination / 'both-later-record-controls.json', dict(state='exact', cases=list(BOTH), queries=both_queries,
            q0_whole_vision_record_equality_required=False, q1_to_q7_entire_records_and_PNGs_exact=True), exclusive=True)
        write(destination / 'complete.json', dict(state='complete', queries=80, saved_q0_predictions=10, native_predictions=70,
            fresh_model_forwards=2100, actual_later_whole_noise_pairs=63, q0_whole_noise_pair_checks=0, extra_q0_model_forwards=0,
            two_native_all_eight_query_records_entire_exact=True, both_seven_later_query_records_entire_exact=True,
            component_hooks_installed=False, attention_dispatch_wrapper_installed=False,
            files_sha256={name: sha(destination / name) for name in ('provenance.json', 'saved-q0-controls.json', 'both-later-record-controls.json')}, scope=SCOPE), exclusive=True)
    finally:
        if http is not None:
            http.server_close()


def strict_windows(records, summary, np):
    rows = {}
    for name in records[0]['objects']:
        lift = np.asarray([item['objects'][name][2] - records[0]['objects'][name][2] for item in records])
        both = np.asarray([item['finger_contacts'][name]['left'] and item['finger_contacts'][name]['right'] for item in records], dtype=bool)
        assert both.tolist() == [item['finger_contacts'][name]['both'] for item in records]
        strict = both & (lift > .02)
        starts = [i for i in range(125) if bool(strict[i:i + 5].all())]
        first = starts[0] if starts else None
        assert summary['per_object'][name]['first_2cm_for_5frames'] == first
        rows[name] = dict(lift_m=lift.tolist(), both_fingers=both.tolist(), strict_contact_and_lift=strict.tolist(),
            five_record_window_start_steps=starts, first_window=None if first is None else list(range(first, first + 5)))
    assert {name for name, item in rows.items() if item['first_window'] is not None} == set(summary['selected_objects'])
    return dict(records=129, criterion='Both fingerpad contacts and lift strictly >.02m for five consecutive records; window labels use the first record.', objects=rows)


def simulate(baseline):
    simulator, frozen, cross, out = prepared(baseline)
    assert not (out / 'closed-loop').exists(), 'No partial output retry'
    os.environ.setdefault('MUJOCO_GL', 'egl')
    sys.path.insert(0, '/home/current/work/openpi-demo/cosmos-policy')
    sys.path.insert(0, str(ROOT / 'cosmos-framework'))
    import requests
    from cosmos_framework.simulation.libero.closed_loop_eval import _framewise_action_to_delta, _remap_gripper
    collector = simulator.load_file('future_read_execution_collector', ROOT / 'work/collect_cosmos_goal_pair.py')
    rollout = simulator.load_file('future_read_execution_controller', ROOT / 'work/rollout_cosmos_goal_pair.py')
    np = simulator.np
    original_post, original_measure, original_summary = requests.post, simulator.measure, simulator.summarize_trial
    active, native_controls, entries, windows, sham_controls = None, [], [], [], []

    def measure(env, obs, step, helper):
        result = original_measure(env, obs, step, helper)
        if step == 16:
            arrays = helper.controller_state(env.env)
            folder = out / 'closed-loop' / active
            np.savez_compressed(folder / 'controller-step16.npz', **arrays)
            if frozen['trials'][active]['native_control']:
                ref = frozen['native_physical_references'][active]
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
                actual = np.asarray(read(folder / name))
                assert rollout.array_exact(actual, value), ('own cached q0 execution', active, name)
            trajectory = read(folder / 'trajectory.json')
            assert len(trajectory) == 17 and [item['step'] for item in trajectory] == list(range(17))
            if rule['native_control']:
                reference = Path(frozen['native_physical_references'][active]['folder'])
                assert rollout.array_exact(state, np.load(reference / 'sim_states.npy', allow_pickle=False)[:17])
                assert trajectory == read(reference / 'trajectory.json')[:17]
                assert (folder / 'input_01.png').read_bytes() == (reference / 'input_01.png').read_bytes()
            gate = dict(case=active, saved_attention_arm=rule['arm'], all_16_own_cached_q0_normalized_raw_native_clipped_actions_byte_exact=True,
                common_initial_sim_state_bytes_exact=True, actual_prefix_state_records=17,
                actual_prefix_files_sha256={name: sha(folder / name) for name in ('sim_states.npy', 'trajectory.json', 'input_01.png', 'controller-step16.npz')},
                native_old_prefix_bytes_exact=True if rule['native_control'] else None,
                nonnative_old_prefix_reference_used=False if not rule['native_control'] else None)
            write(folder / 'q1-entry-control.json', gate, exclusive=True)
            entries.append(gate)
        return original_post(url, *args, **kwargs)

    def summary(trial, scene, seed, records, helper):
        rule = frozen['trials'][trial]
        result = original_summary(trial, scene, int(seed), records, helper)
        result.update(saved_q0_attention_arm=rule['arm'], q0_vision_noise_source_seed=rule['q0_vision_noise_source_seed'],
            q0_action_noise_source_seed=195, later_noise_base=195, noise_seed_base=195,
            actual_query_seeds=rule['actual_query_seeds'], seed_rule=rule['seed_rule'])
        return result

    requests.post, simulator.measure, simulator.summarize_trial = post, measure, summary
    try:
        for label, v, arm in PLAN:
            active = label
            if arm != 'native':
                assert len(native_controls) == 2
            # Each legacy call resets its seed-only seen_noise set. Q0 pair=False;
            # the real server separately compares later draws across all ten cases.
            simulator.execute_trials(out, [label], collector, rollout)
            folder = out / 'closed-loop' / label
            actual_summary = read(folder / 'summary.json')
            gate = strict_windows(read(folder / 'trajectory.json'), actual_summary, np)
            write(folder / 'strict-windows.json', gate, exclusive=True)
            windows.append(dict(case=label, selected_objects=actual_summary['selected_objects'],
                files_sha256={name: sha(folder / name) for name in ('strict-windows.json', 'summary.json', 'trajectory.json')}))
            if arm == 'native':
                control = cross.native_control(out, label, frozen['native_physical_references'][label], np)
                native_controls.append(control)
                write(out / (label + '-control.json'), control, exclusive=True)
            elif arm == 'all_allowed':
                reference = frozen['native_physical_references'][f'V{v}_A195_native']
                old = read(Path(reference['folder']) / 'summary.json')['selected_objects']
                changed = set(old) != set(actual_summary['selected_objects'])
                sham_controls.append(dict(case=label, native_reference_cell=f'V{v}_A195', native_selected_objects=old,
                    sham_selected_objects=actual_summary['selected_objects'], classification_changed=changed,
                    original_native_path_interpretation_gate_passed=not changed,
                    policy='Record failure and continue every preregistered case; no source replacement, retraining or score selection.'))
                write(out / 'sham-interpretation-controls.json', dict(cases=sham_controls,
                    original_native_path_interpretation_gate_passed=all(not item['classification_changed'] for item in sham_controls)))
        done, served = read(out / 'closed-loop/complete.json'), read(out / 'server/complete.json')
        assert done['state'] == served['state'] == 'complete'
        assert done['cases'] == 10 and done['completed_trials'] == [item[0] for item in PLAN]
        assert done['steps_per_case'] == 128 and done['state_records_per_case'] == 129
        assert len(native_controls) == len(sham_controls) == 2 and len(entries) == len(windows) == 10
        assert served['queries'] == 80 and served['saved_q0_predictions'] == 10 and served['native_predictions'] == 70
        assert served['fresh_model_forwards'] == 2100 and served['actual_later_whole_noise_pairs'] == 63
        assert served['q0_whole_noise_pair_checks'] == served['extra_q0_model_forwards'] == 0
        for name, expected in served['files_sha256'].items():
            assert sha(out / 'server' / name) == expected
        both_physics = cross.native_control(out, BOTH[1], dict(folder=str(out / 'closed-loop' / BOTH[0])), np)
        both_later = read(out / 'server/both-later-record-controls.json')
        assert both_later['q1_to_q7_entire_records_and_PNGs_exact'] is True
        write(out / 'both-controls.json', dict(state='exact', cases=list(BOTH), physical=both_physics,
            q1_to_q7_entire_modelrecords_exact=True, server_control_sha256=sha(out / 'server/both-later-record-controls.json'),
            q0_whole_future_vision_record_equality_required=False), exclusive=True)
        case_files = {}
        for label, _, _ in PLAN:
            folder = out / 'closed-loop' / label
            names = ('sim_states.npy', 'actual.mp4', 'initial_controller.npz', 'controller-step16.npz', 'summary.json', 'strict-windows.json',
                'q1-entry-control.json', 'query_contract.json', *PHYSICAL, *[f'input_{q:02d}.png' for q in range(8)],
                *[f'chunk_{q:02d}/{name}' for q in range(8) for name in ('states.pt', 'metadata.json', 'normalized_actions.json')],
                *[f'chunk_00/{name}' for name in FILES if name not in ('states.pt', 'metadata.json', 'normalized_actions.json')])
            case_files[label] = {name: sha(folder / name) for name in names}
        write(out / 'controls.json', dict(native_whole_controls=native_controls, q1_entry_controls=entries,
            strict_windows=windows, sham_classification_controls=sham_controls,
            original_native_path_interpretation_gate_passed=all(not item['classification_changed'] for item in sham_controls),
            both_controls_sha256=sha(out / 'both-controls.json'), actual_case_files_sha256=case_files), exclusive=True)
        files = ('controls.json', 'both-controls.json', 'sham-interpretation-controls.json', 'provenance.json', 'prepared.json',
                 'server/complete.json', 'closed-loop/complete.json', 'summary.json')
        write(out / 'complete.json', dict(state='complete', script_sha256=sha(Path(__file__)), trial_order=[item[0] for item in PLAN],
            physical_trials=10, saved_q0_predictions=10, native_predictions=70, fresh_model_forwards=2100,
            producer_q0_forwards_separate=300, combined_two_stage_model_forwards=2400, actual_later_whole_noise_pairs=63,
            two_native_entire_129_state_5JSON_8PNG_and_8_modelrecords_exact=True,
            both_entire_129_state_5JSON_8PNG_and_7_later_modelrecords_exact=True,
            all_ten_own_cached_q0_action_execution_and_strict_window_gates_passed=True,
            numerical_sham_classification_gate_passed=all(not item['classification_changed'] for item in sham_controls),
            all_ten_cases_executed_regardless_of_sham_classification=True,
            files_sha256={name: sha(out / name) for name in files}, scope=SCOPE), exclusive=True)
    finally:
        requests.post, simulator.measure, simulator.summarize_trial = original_post, original_measure, original_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('prepare', 'server', 'simulate'), required=True)
    args = parser.parse_args()
    baseline, out = args.output.resolve(), args.output.resolve() / 'future-read-execution'
    if args.mode == 'prepare':
        _, frozen, _ = contract(baseline)
        out.mkdir(exist_ok=False)
        (out / 'inputs').symlink_to(frozen['inputs_symlink_target'], target_is_directory=True)
        write(out / 'provenance.json', frozen, exclusive=True)
        write(out / 'prepared.json', dict(state='prepared', provenance_sha256=sha(out / 'provenance.json')), exclusive=True)
        print('[FUTURE-READ-EXECUTION] prepared ' + str(out), flush=True)
        return
    try:
        server(baseline) if args.mode == 'server' else simulate(baseline)
    except Exception as exc:
        if out.is_dir() and not (out / 'wrapper_failed.json').exists():
            write(out / 'wrapper_failed.json', dict(mode=args.mode, error=str(exc), traceback=traceback.format_exc()), exclusive=True)
        raise


if __name__ == '__main__':
    main()
