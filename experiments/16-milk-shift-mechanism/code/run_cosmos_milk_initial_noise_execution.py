"""Execute four saved initial-noise factorial q0 predictions, all later base195.

Use --output BASE --mode prepare|server|simulate. Prepare exclusively creates
BASE/initial-noise-execution; no old source or data is edited. Reuse the real
frozen simulator execute_trials body, one trial per call. Query0 serves saved
initial-noise-factors actions; queries1..7 infer from each new actual image,
with seeds196..202. Four cached replies plus 28 fresh predictions = 840 new
model forwards / 32 requests / four actual 128-action physical trajectories.

Run both native cells first. V195/A195 must equal old x12_seed195 completely;
V198/A198 must equal completed query-noise-cross/q0198_later195 completely.
Mixed cells have no old physical prefix reference: check the common restored
initial state and execution of their own cached q0 actions. Preserve actual
states, controller arrays, images, actions and video. No extra intervention,
training, semantic labeling or claim that query0 irrevocably chose a target.
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
CROSS_SHA = '7e684b1a330877b8dbaad2c27621c3bc8db4893eb25da0c9171ff6d8b56aa2fe'
PROBE_SHA = '9081abb51200d1335bb4f84fde87fc23eec45a97ca46789f2dcf700acaf26c1e'
PORT, LATER_BASE = 8930, 195
PROMPT = 'pick up the milk and place it in the basket'
PLAN = {'V195_A195': (195, 195), 'V198_A198': (198, 198),
        'V198_A195': (198, 195), 'V195_A198': (195, 198)}
FILES = ('states.pt', 'normalized_actions.json', 'metadata.json', 'noise-audit.pt', 'noise-audit.json')
PHYSICAL = ('trajectory.json', 'actions.json', 'executed_actions.json', 'normalized_actions.json', 'denormalized_actions.json')
SCOPE = ('Four X+12cm physical trajectories using saved factorial initial-noise q0 '
         'predictions and the same native later sampling base195. All later images '
         'come from each actual evolving environment. Two complete native controls '
         'precede mixed cells. No network patch, new training, inferred intention, '
         'q0 target commitment or neural-root-cause claim.')


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
    cross_path = ROOT / 'work/run_cosmos_milk_query_noise_cross.py'
    probe_path = ROOT / 'work/probe_cosmos_milk_initial_noise_factors.py'
    assert sha(cross_path) == CROSS_SHA and sha(probe_path) == PROBE_SHA
    cross = load('initial_noise_execution_frozen_cross_contract', cross_path)
    simulator, prior, metrics, cross_root = cross.prepared(baseline)
    prior_complete = read(cross_root / 'complete.json')
    assert prior_complete['state'] == 'complete' and prior_complete['script_sha256'] == CROSS_SHA
    assert prior_complete['native_model_forwards'] == 840 and prior_complete['two_native_complete_physical_controls_exact'] is True
    for filename, expected in prior_complete['files_sha256'].items():
        assert sha(cross_root / filename) == expected
    producer = baseline / 'initial-noise-factors'
    assert not (producer / 'failed.json').exists()
    complete = read(producer / 'complete.json')
    assert complete['state'] == 'complete' and complete['script_sha256'] == PROBE_SHA
    assert complete['fresh_q0_predictions'] == 4 and complete['fresh_model_forwards'] == 120
    assert complete['original_randn_calls_consumed'] == complete['actual_returned_source_draws_observed'] == 8
    for key in ('both_native_entire_q0_records_byte_exact', 'all_current_frame_schedule_layout_padding_and_source_noise_gates_passed',
                'original_randn_symbol_restored', 'all_owned_observer_hooks_removed'):
        assert complete[key] is True
    for key, filename in (('results_sha256', 'results.json'), ('provenance_sha256', 'provenance.json'),
                          ('protocol_sha256', 'protocol.json'), ('sources_sha256', 'sources.json')):
        assert complete[key] == sha(producer / filename)
    results, protocol, producer_sources = (read(producer / (name + '.json')) for name in ('results', 'protocol', 'sources'))
    assert results['state'] == 'complete' and results['fresh_model_forwards'] == 120 and results['fresh_q0_predictions'] == 4
    assert protocol['trial_order'] == list(PLAN) and protocol['image_sha256'] == sha(baseline / 'position-threshold/inputs/x12/input.png')
    np = simulator.np
    stats = read(Path(prior['threshold_contract']['frozen_files']['normalizer_stats']['path']))['global_raw']
    lo, hi = np.asarray(stats['q01']), np.asarray(stats['q99'])
    assert lo.shape == hi.shape == (10,) and np.all(hi - lo > 1e-6)
    scale, offset = np.maximum(hi - lo, 1e-8) / 2, (hi + lo) / 2
    rows = {row['cell']: row for row in results['cases']}
    assert len(rows) == 4 and set(rows) == set(PLAN)
    sources, rules = {}, {}
    for label, (v, a) in PLAN.items():
        row, folder = rows[label], producer / label
        assert row['vision_noise_source_seed'] == v and row['action_noise_source_seed'] == row['runtime_rng_seed'] == a
        assert row['fresh_model_forwards'] == 30 and row['all_actual_noise_source_and_preparation_gates_passed'] is True
        assert row['native_entire_source_record_byte_exact'] is (True if v == a else None)
        assert set(row['files_sha256']) == set(FILES)
        for filename, expected in row['files_sha256'].items():
            assert sha(folder / filename) == expected
        metadata = read(folder / 'metadata.json')
        assert metadata['seed'] == metadata['runtime_rng_seed'] == a and metadata['vision_noise_source_seed'] == v and metadata['action_noise_source_seed'] == a
        assert metadata['model_calls'] == 30 and metadata['prepare_calls'] == 1 and metadata['input_png_sha256'] == protocol['image_sha256']
        normalized = np.asarray(read(folder / 'normalized_actions.json'))
        assert normalized.shape == (16, 10) and np.isfinite(normalized).all() and normalized.tolist() == row['final_normalized_actions']
        sources[label] = dict(folder=str(folder), files_sha256=row['files_sha256'], normalized_actions=normalized.tolist(),
            raw_actions=(normalized * scale + offset).tolist(), vision_noise_source_seed=v, action_noise_source_seed=a)
        rules[label] = dict(q0_vision_noise_source_seed=v, q0_action_noise_source_seed=a, later_noise_base=LATER_BASE,
            actual_query_seeds=[a, *range(196, 203)], native_control=v == a,
            seed_rule='q0: saved source has generator=A with separate V/A draws; q1..7:195+query',
            q0_whole_noise_pairing=False)
    controls = read(cross_root / 'controls.json')
    native = {'V195_A195': Path(prior['sources']['195']['folder']), 'V198_A198': cross_root / 'closed-loop/q0198_later195'}
    native_sources = {}
    for label, folder in native.items():
        controller_trial = 'native195' if label == 'V195_A195' else 'q0198_later195'
        controller_folder = cross_root / 'closed-loop' / controller_trial
        gate = next(row for row in controls['q1_entry_controls'] if row['trial'] == controller_trial)
        assert sha(controller_folder / 'controller-step16.npz') == gate['actual_controller_sha256']
        files = ('sim_states.npy', 'summary.json', *PHYSICAL, *[f'input_{query:02d}.png' for query in range(8)],
                 *[f'chunk_{query:02d}/{name}' for query in range(8) for name in ('states.pt', 'metadata.json', 'normalized_actions.json')])
        native_sources[label] = dict(folder=str(folder), files_current_sha256={name: sha(folder / name) for name in files},
            controller_step16_path=str(controller_folder / 'controller-step16.npz'), controller_step16_sha256=gate['actual_controller_sha256'],
            seal_scope='Whole native reference case hashes first observed by this preparation. Earlier completed cross controls and their existing controller hashes are separately anchored.')
    frozen = dict(script_sha256=sha(Path(__file__)), cross_script_sha256=CROSS_SHA, probe_script_sha256=PROBE_SHA,
        threshold_contract=prior['threshold_contract'], actual_transformer_sha256=prior['actual_transformer_sha256'],
        actual_pipeline_source_sha256=prior['actual_pipeline_source_sha256'], metrics_path=str(metrics),
        prior_cross_documents_sha256={name: sha(cross_root / name) for name in ('complete.json', 'provenance.json', 'prepared.json', 'controls.json')},
        producer_documents_sha256={name: sha(producer / name) for name in ('complete.json', 'results.json', 'provenance.json', 'protocol.json', 'sources.json')},
        producer_source_cases=producer_sources['cases'], producer_sources_sha256=sha(producer / 'sources.json'),
        q0_sources=sources, native_physical_references=native_sources, trials=rules, trial_order=list(PLAN),
        inputs_symlink_target=str(baseline / 'position-threshold/inputs'), url=f'http://127.0.0.1:{PORT}',
        current_execution_saved_q0=4, native_predictions=28, fresh_model_forwards=840,
        earlier_producer_q0_model_forwards_separate=120, later_whole_noise_pair_checks=21, scope=SCOPE)
    simulator.SHIFTS = dict(x00=0, x09=9, x12=12)
    simulator.SEEDS = (195,)
    simulator.TRIALS = {label: ('x12', cross.QuerySeed(a, LATER_BASE)) for label, (_, a) in PLAN.items()}
    simulator.URL, simulator.SCOPE = frozen['url'], SCOPE
    return simulator, frozen, cross


def prepared(baseline):
    simulator, frozen, cross = contract(baseline)
    out = baseline / 'initial-noise-execution'
    assert out.is_dir() and not (out / 'wrapper_failed.json').exists()
    assert (out / 'inputs').is_symlink() and (out / 'inputs').resolve() == Path(frozen['inputs_symlink_target']).resolve()
    assert read(out / 'provenance.json') == frozen
    saved = read(out / 'prepared.json')
    assert saved['state'] == 'prepared' and saved['provenance_sha256'] == sha(out / 'provenance.json')
    return simulator, frozen, cross, out


def verify_cached_q0(actual, source_records, audit, v, a, runtime, scan):
    runtime.require_exact(actual['pure_noise'], [source_records[v]['pure_noise'][0], source_records[a]['pure_noise'][1]], 'cached actual source draw pair')
    if v == a:
        runtime.require_exact(actual, source_records[a], 'cached native ENTIRE original q0 record')
    prepared = actual['prepared_latents_and_masks']
    runtime.require_exact(prepared[0], source_records[v]['prepared_latents_and_masks'][0], 'cached chosen prepared vision')
    runtime.require_exact(prepared[2], source_records[a]['prepared_latents_and_masks'][2], 'cached chosen prepared action')
    for index in range(12):
        if index not in (0, 2):
            runtime.require_exact(prepared[index], source_records[195]['prepared_latents_and_masks'][index], 'cached common prepare/' + str(index))
    kwargs = actual['model_input']
    runtime.require_exact(kwargs['vision_tokens'], source_records[v]['model_input']['vision_tokens'], 'cached V initial kwargs')
    runtime.require_exact(kwargs['action_tokens'], source_records[a]['model_input']['action_tokens'], 'cached A initial kwargs')
    for key in kwargs:
        if key not in ('vision_tokens', 'action_tokens'):
            runtime.require_exact(kwargs[key], source_records[195]['model_input'][key], 'cached other initial kwargs/' + key)
    assert len(audit['calls']) == 2 and len(audit['model_steps']) == 30
    assert audit['original_symbol_restored'] is audit['observer_hooks_removed'] is True
    runtime.require_exact(audit['initial_model_kwargs'], kwargs, 'cached actual initial observer')
    for index, call in enumerate(audit['calls']):
        runtime.require_exact(call['original_generated_discarded'], source_records[a]['pure_noise'][index], 'cached consumed original draw')
        runtime.require_exact(call['actual_returned'], actual['pure_noise'][index], 'cached returned/inner noise bytes')
        for key, value in (('original_generated_metadata', call['original_generated_discarded']), ('returned_metadata', call['actual_returned'])):
            for field, expected in scan.tensor_meta(value, runtime.torch).items():
                assert call[key][field] == expected
    runtime.require_exact(audit['calls'][0]['generator_state_after'], audit['calls'][1]['generator_state_before'], 'cached RNG state chain')
    for key in ('timesteps', 'sigmas'):
        runtime.require_exact(actual[key], source_records[195][key], 'cached full schedule/' + key)


def server(baseline):
    _, frozen, _, out = prepared(baseline)
    destination = out / 'server'
    destination.mkdir(exist_ok=False)
    served, later_noise, paired, saved_count, native_count = [], {}, 0, 0, 0
    failed, http = False, None
    started = time.perf_counter()
    try:
        os.environ['HF_HUB_OFFLINE'] = '1'
        paths = frozen['threshold_contract']['frozen_files']
        base = load('initial_noise_execution_normal_runtime', Path(paths['normal_runtime']['path']))
        base.TASKS['milk'] = (7, 'pick_up_the_milk_and_place_it_in_the_basket', PROMPT, 'milk_1')
        factory = load('initial_noise_execution_factory', Path(paths['factory']['path']))
        factory.OUT = destination
        assert sha(factory.SCHEDULER) == paths['scheduler']['sha256']
        runtime = base.NormalRuntime(factory)
        torch, model = runtime.torch, runtime.pipe.transformer
        assert not model.training and not model.is_cache_enabled and not torch.is_grad_enabled()
        assert sha(Path(inspect.getfile(type(model)))) == frozen['actual_transformer_sha256']
        assert sha(Path(inspect.getfile(type(runtime.pipe)))) == frozen['actual_pipeline_source_sha256']
        scan = load('initial_noise_execution_record_checks', Path(frozen['metrics_path']))
        source_records = {}
        for seed in (195, 198):
            source = frozen['producer_source_cases'][str(seed)]
            folder = Path(source['q0_folder'])
            assert sha(folder / 'states.pt') == source['q0_files_current_sha256']['states.pt']
            source_records[seed] = torch.load(folder / 'states.pt', map_location='cpu', weights_only=True, mmap=True)
            scan.validate_record(source_records[seed], torch)
        q0 = {}
        for label, (v, a) in PLAN.items():
            source = frozen['q0_sources'][label]
            folder = Path(source['folder'])
            record = torch.load(folder / 'states.pt', map_location='cpu', weights_only=True, mmap=True)
            audit = torch.load(folder / 'noise-audit.pt', map_location='cpu', weights_only=True, mmap=True)
            scan.validate_record(record, torch)
            verify_cached_q0(record, source_records, audit, v, a, runtime, scan)
            assert record['actions'].tolist() == source['normalized_actions']
            q0[label] = record
        write(destination / 'saved-q0-controls.json', dict(state='exact', all_four_actual_V_A_draw_pairs_and_preparation_exact=True,
            native_two_entire_q0_records_exact=True, q0_whole_noise_pair_checks=0,
            q0_pairing='Four different whole initial noise pairs; same action seed alone never implies whole noise equality.',
            extra_q0_model_forwards=0), exclusive=True)
        write(destination / 'provenance.json', dict(script_sha256=sha(Path(__file__)), execution_provenance_sha256=sha(out / 'provenance.json'),
            saved_q0_controls_sha256=sha(destination / 'saved-q0-controls.json'), component_hooks_installed=False,
            current_execution_saved_q0=4, native_predictions=28, expected_fresh_model_forwards=840, scope=SCOPE), exclusive=True)

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                assert self.path == '/health'
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'ready')

            def do_POST(self):
                nonlocal failed, paired, saved_count, native_count
                try:
                    assert self.path == '/predict'
                    request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                    assert set(request) == {'trial', 'query', 'seed', 'prompt'}
                    label, query = request['trial'], request['query']
                    assert label in PLAN and type(query) is int and 0 <= query < 8
                    rule = frozen['trials'][label]
                    seed = rule['actual_query_seeds'][query]
                    assert request['seed'] == seed and request['prompt'] == PROMPT
                    assert (label, query) not in served and all((label, old) in served for old in range(query))
                    image = out / 'closed-loop' / label / f'input_{query:02d}.png'
                    chunk = image.parent / f'chunk_{query:02d}'
                    source = frozen['q0_sources'][label]
                    if query == 0:
                        assert image.read_bytes() == (out / 'inputs/x12/input.png').read_bytes()
                        source_folder = Path(source['folder'])
                        chunk.mkdir(exist_ok=False)
                        for filename in FILES:
                            assert sha(source_folder / filename) == source['files_sha256'][filename]
                            if filename != 'metadata.json':
                                shutil.copyfile(source_folder / filename, chunk / filename)
                        record, metadata = q0[label], read(source_folder / 'metadata.json')
                        saved_count += 1
                        paired_noise = False
                    else:
                        record, metadata = runtime.predict('milk', image, seed, chunk)
                        native_count += 1
                        paired_noise = query in later_noise
                        if paired_noise:
                            runtime.require_exact(record['pure_noise'], later_noise[query], 'actual later195 whole noise/query' + str(query))
                            paired += 1
                        else:
                            later_noise[query] = record['pure_noise']
                    scan.validate_record(record, torch)
                    for key in ('timesteps', 'sigmas'):
                        runtime.require_exact(record[key], source_records[195][key], 'execution complete native schedule/' + key)
                    if rule['native_control']:
                        reference = frozen['native_physical_references'][label]
                        reference_folder = Path(reference['folder'])
                        assert image.read_bytes() == (reference_folder / f'input_{query:02d}.png').read_bytes()
                        refname = f'chunk_{query:02d}/states.pt'
                        assert sha(reference_folder / refname) == reference['files_current_sha256'][refname]
                        runtime.require_exact(record, torch.load(reference_folder / refname, map_location='cpu', weights_only=True, mmap=True), 'native entire saved query/' + label + '/' + str(query))
                    assert metadata['seed'] == seed and metadata['model_calls'] == 30 and metadata['prepare_calls'] == 1
                    metadata.update(trial=label, query=query, shift_cm=12, image_path=str(image), input_png_sha256=sha(image),
                        saved_q0_prediction=query == 0, current_execution_model_calls=0 if query == 0 else 30,
                        source_record_model_calls=30 if query == 0 else None,
                        q0_vision_noise_source_seed=rule['q0_vision_noise_source_seed'], q0_action_noise_source_seed=rule['q0_action_noise_source_seed'],
                        later_noise_base=LATER_BASE, actual_query_seeds=rule['actual_query_seeds'], seed_rule=rule['seed_rule'],
                        component_intervention_in_this_execution=False, initial_noise_replacement_in_this_execution=False,
                        paired_noise_checked_exact=paired_noise,
                        saved_q0_source_states_sha256=source['files_sha256']['states.pt'] if query == 0 else None)
                    write(chunk / 'metadata.json', metadata)
                    assert read(chunk / 'normalized_actions.json') == record['actions'].tolist()
                    served.append((label, query))
                    write(destination / 'progress.json', dict(queries=len(served), saved_q0_predictions=saved_count, native_predictions=native_count,
                        fresh_model_forwards=30 * native_count, actual_later_whole_noise_pairs=paired, served_queries=served, elapsed_s=time.perf_counter() - started))
                    print('[INITIAL-NOISE-EXECUTION] ' + json.dumps(dict(queries=len(served), trial=label, query=query, actual_seed=seed)), flush=True)
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(dict(actions=record['actions'].tolist(), metadata=metadata,
                        paired_noise_checked_exact=paired_noise), allow_nan=False).encode())
                except Exception as exc:
                    failed = True
                    write(destination / 'failed.json', dict(error=str(exc), traceback=traceback.format_exc(), served_queries=served))
                    self.send_error(500, str(exc))

        http = HTTPServer(('127.0.0.1', PORT), Handler)
        http.timeout = 30
        write(destination / 'ready.json', dict(state='ready', url=frozen['url']), exclusive=True)
        while not failed and len(served) < 32:
            http.handle_request()
            assert not (out / 'closed-loop/failed.json').exists() and not (out / 'wrapper_failed.json').exists()
        assert not failed and len(served) == 32 and saved_count == 4 and native_count == 28 and paired == 21
        write(destination / 'complete.json', dict(state='complete', queries=32, saved_q0_predictions=4,
            native_predictions=28, fresh_model_forwards=840, actual_later_whole_noise_pairs=21,
            q0_whole_noise_pair_checks=0, extra_q0_model_forwards=0,
            native_two_all_eight_query_records_entire_exact=True, component_hooks_installed=False,
            provenance_sha256=sha(destination / 'provenance.json'), saved_q0_controls_sha256=sha(destination / 'saved-q0-controls.json'), scope=SCOPE), exclusive=True)
    except Exception as exc:
        if not (destination / 'failed.json').exists():
            write(destination / 'failed.json', dict(error=str(exc), traceback=traceback.format_exc(), served_queries=served))
        raise
    finally:
        if http is not None:
            http.server_close()


def simulate(baseline):
    simulator, frozen, cross, out = prepared(baseline)
    assert not (out / 'closed-loop').exists(), 'No partial output retry'
    os.environ.setdefault('MUJOCO_GL', 'egl')
    sys.path.insert(0, '/home/current/work/openpi-demo/cosmos-policy')
    sys.path.insert(0, str(ROOT / 'cosmos-framework'))
    import requests
    from cosmos_framework.simulation.libero.closed_loop_eval import _framewise_action_to_delta, _remap_gripper
    collector = simulator.load_file('initial_noise_execution_collector', ROOT / 'work/collect_cosmos_goal_pair.py')
    rollout = simulator.load_file('initial_noise_execution_controller', ROOT / 'work/rollout_cosmos_goal_pair.py')
    np = simulator.np
    original_post, original_measure, original_summary = requests.post, simulator.measure, simulator.summarize_trial
    active, native_controls, entries = None, [], []

    def measure(env, obs, step, helper):
        result = original_measure(env, obs, step, helper)
        if step == 16:
            arrays = helper.controller_state(env.env)
            folder = out / 'closed-loop' / active
            np.savez_compressed(folder / 'controller-step16.npz', **arrays)
            if frozen['trials'][active]['native_control']:
                path = Path(frozen['native_physical_references'][active]['controller_step16_path'])
                with np.load(path, allow_pickle=False) as source:
                    assert set(arrays) == set(source.files)
                    for key in arrays:
                        assert rollout.array_exact(arrays[key], source[key]), ('native actual controller16 bytes', active, key)
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
            expected = {'normalized_actions.json': norm, 'denormalized_actions.json': raw,
                        'actions.json': native, 'executed_actions.json': np.clip(native, -1, 1)}
            for name, value in expected.items():
                actual = np.asarray(read(folder / name))
                assert actual.dtype == value.dtype and actual.shape == value.shape and actual.tobytes(order='C') == value.tobytes(order='C'), ('own cached q0 actual execution', active, name)
            trajectory = read(folder / 'trajectory.json')
            assert len(trajectory) == 17 and [row['step'] for row in trajectory] == list(range(17))
            assert (folder / 'input_01.png').is_file() and (folder / 'controller-step16.npz').is_file()
            if rule['native_control']:
                reference = Path(frozen['native_physical_references'][active]['folder'])
                old_states = np.load(reference / 'sim_states.npy', allow_pickle=False)[:17]
                assert state.tobytes(order='C') == old_states.tobytes(order='C')
                assert trajectory == read(reference / 'trajectory.json')[:17]
                assert (folder / 'input_01.png').read_bytes() == (reference / 'input_01.png').read_bytes()
            gate = dict(trial=active, all_16_own_cached_normalized_raw_native_clipped_actions_byte_exact=True,
                common_initial_sim_state_byte_exact=True, actual_prefix_state_records=17,
                actual_prefix_states_sha256=sha(folder / 'sim_states.npy'), actual_trajectory_sha256=sha(folder / 'trajectory.json'),
                actual_input01_PNG_sha256=sha(folder / 'input_01.png'), actual_controller16_original_dtype_arrays_sha256=sha(folder / 'controller-step16.npz'),
                old_native_prefix_states_trajectory_PNG_and_controller_bytes_exact=True if rule['native_control'] else None,
                mixed_old_prefix_reference_exists=False if not rule['native_control'] else None,
                mixed_prefix_scope='Mixed cells execute their own saved q0 actions and produce their own actual state/image/controller; no old mixed trajectory reference.')
            write(folder / 'q1-entry-control.json', gate, exclusive=True)
            entries.append(gate)
        return original_post(url, *args, **kwargs)

    def summary(trial, scene, seed, records, helper):
        rule = frozen['trials'][trial]
        result = original_summary(trial, scene, int(seed), records, helper)
        result.update(q0_vision_noise_source_seed=rule['q0_vision_noise_source_seed'], q0_action_noise_source_seed=rule['q0_action_noise_source_seed'],
            later_noise_base=LATER_BASE, noise_seed_base=LATER_BASE, noise_seed_base_applies_to='queries1..7 only; q0 has separately saved V/A sources',
            actual_query_seeds=rule['actual_query_seeds'], seed_rule=rule['seed_rule'])
        return result

    requests.post, simulator.measure, simulator.summarize_trial = post, measure, summary
    try:
        for label in PLAN:
            active = label
            if not frozen['trials'][label]['native_control']:
                assert len(native_controls) == 2
            # The legacy transport's seed-only seen_noise set must reset per trial;
            # same A at q0 does not mean identical whole V/A noise pairs.
            simulator.execute_trials(out, [label], collector, rollout)
            if frozen['trials'][label]['native_control']:
                control = cross.native_control(out, label, frozen['native_physical_references'][label], np)
                native_controls.append(control)
                write(out / (label + '-control.json'), control, exclusive=True)
        done, served = read(out / 'closed-loop/complete.json'), read(out / 'server/complete.json')
        assert done['state'] == 'complete' and done['cases'] == 4 and done['completed_trials'] == list(PLAN)
        assert len(native_controls) == 2 and len(entries) == 4
        assert served['state'] == 'complete' and served['queries'] == 32 and served['native_predictions'] == 28
        assert served['fresh_model_forwards'] == 840 and served['actual_later_whole_noise_pairs'] == 21 and served['q0_whole_noise_pair_checks'] == 0
        write(out / 'controls.json', dict(native_whole_controls=native_controls, q1_entry_controls=entries), exclusive=True)
        files = ('controls.json', 'provenance.json', 'prepared.json', 'server/complete.json', 'closed-loop/complete.json', 'summary.json')
        write(out / 'complete.json', dict(state='complete', script_sha256=sha(Path(__file__)), trial_order=list(PLAN), physical_trials=4,
            saved_q0_predictions=4, native_predictions=28, fresh_model_forwards=840, earlier_separate_q0_probe_model_forwards=120,
            two_native_complete_129_state_and_all_action_trajectory_and_PNG_controls_exact=True,
            all_four_own_cached_q0_action_execution_controls_exact=True, actual_later_whole_noise_pairs=21,
            files_sha256={name: sha(out / name) for name in files}, scope=SCOPE), exclusive=True)
    finally:
        requests.post, simulator.measure, simulator.summarize_trial = original_post, original_measure, original_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('prepare', 'server', 'simulate'), required=True)
    args = parser.parse_args()
    baseline, out = args.output.resolve(), args.output.resolve() / 'initial-noise-execution'
    if args.mode == 'prepare':
        _, frozen, _ = contract(baseline)
        out.mkdir(exist_ok=False)
        (out / 'inputs').symlink_to(frozen['inputs_symlink_target'], target_is_directory=True)
        write(out / 'provenance.json', frozen, exclusive=True)
        write(out / 'prepared.json', dict(state='prepared', provenance_sha256=sha(out / 'provenance.json')), exclusive=True)
        print('[INITIAL-NOISE-EXECUTION] prepared ' + str(out), flush=True)
        return
    try:
        if args.mode == 'server':
            server(baseline)
        else:
            simulate(baseline)
    except Exception as exc:
        if out.is_dir() and not (out / 'wrapper_failed.json').exists():
            write(out / 'wrapper_failed.json', dict(mode=args.mode, error=str(exc), traceback=traceback.format_exc()), exclusive=True)
        raise


if __name__ == '__main__':
    main()
