"""Cross saved q0 actions with later native noise streams at the same X+12cm state.

Use --output BASE --mode prepare|server|simulate. Only prepare creates absent
BASE/query-noise-cross. Reuse the frozen physical rollout without editing it.
Run native195/native198 first, then the two crossed streams. Query0 replays
the original saved sixteen actions; queries1..7 use current images and seeds
later_noise_base + query. Four saved q0 replies plus 28 native predictions
give 32 queries / 840 model forwards and four 128-step physical trials.

Before q1, require source q0 physical states/trajectory/actions/input01 exact.
Capture actual controller arrays; crossed prefixes must also match the already
completed native counterpart's arrays byte-for-byte. The old trajectory has
controller JSON values, not an independently saved step16 dtype/byte manifest.
This separates early physical-state differences from later sampling streams;
it does not identify a network cause or establish q0 target commitment.
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
READER_SHA = 'a71604115ee797cef97662f29c9f76e536d84f7e5515d1a725526c0a0d08d2eb'
PORT = 8929
PROMPT = 'pick up the milk and place it in the basket'
PLAN = {'native195': (195, 195), 'native198': (198, 198),
        'q0195_later198': (195, 198), 'q0198_later195': (198, 195)}
PHYSICAL_JSON = ('trajectory.json', 'actions.json', 'executed_actions.json',
                 'normalized_actions.json', 'denormalized_actions.json')
SCOPE = ('Four X+12cm executions crossing saved q0 source195/198 with later '
         'native sampling base195/198. Same initial physical state and instruction; '
         'no network intervention, new warmup, training, q0 target-commitment or '
         'neural-root-cause claim. Historical seed196 trials remain unchanged.')


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


def actual_seeds(source, later):
    return [source, *[later + query for query in range(1, 8)]]


class QuerySeed(int):
    """Adapt the legacy's sole `seed + query` API without false q0 metadata."""
    def __new__(cls, source, later):
        value = int.__new__(cls, later)
        value.source = source
        return value

    def __add__(self, query):
        assert type(query) is int and 0 <= query < 8
        return self.source if query == 0 else int(self) + query


def contract(baseline):
    reader_path = ROOT / 'work/probe_cosmos_milk_target_reader_inputs.py'
    assert sha(reader_path) == READER_SHA
    reader = load('noise_cross_frozen_threshold_checks', reader_path)
    original, cases, threshold = reader.threshold_sources(baseline)
    threshold_root = baseline / 'position-threshold'
    paths = {key: Path(value['path']) for key, value in original['frozen_files'].items()}
    simulator = load('noise_cross_frozen_physics', paths['simulator'])
    assert Path(simulator.__file__).resolve() == paths['simulator'].resolve()
    feedback = read(baseline / 'feedback-inputs/provenance.json')
    metrics_path = ROOT / 'work/probe_cosmos_milk_component_pulses.py'
    assert sha(metrics_path) == feedback['pulse_contract_sha256']
    assert {key: sha(paths[key]) for key in ('normal_runtime', 'factory', 'components')} == feedback['source_sha256']
    stats = read(paths['normalizer_stats'])['global_raw']
    np = simulator.np
    lo, hi = np.asarray(stats['q01']), np.asarray(stats['q99'])
    assert lo.shape == hi.shape == (10,) and np.all(hi - lo > 1e-6)
    scale, offset = np.maximum(hi - lo, 1e-8) / 2, (hi + lo) / 2
    sources = {}
    source_files = ('sim_states.npy', 'initial_controller.npz', 'summary.json', *PHYSICAL_JSON,
                    *[f'input_{query:02d}.png' for query in range(8)],
                    *[f'chunk_{query:02d}/{name}' for query in range(8)
                      for name in ('states.pt', 'metadata.json', 'normalized_actions.json')])
    for seed in (195, 198):
        folder = threshold_root / 'closed-loop' / f'x12_seed{seed}'
        norm = np.asarray(read(folder / 'normalized_actions.json'))
        raw = np.asarray(read(folder / 'denormalized_actions.json'))
        assert norm.shape == raw.shape == (128, 10) and np.isfinite(norm).all() and np.isfinite(raw).all()
        assert read(folder / 'normalized_actions.json')[:16] == read(folder / 'chunk_00/normalized_actions.json')
        assert np.array_equal(raw[:16], norm[:16] * scale + offset)
        trajectory = read(folder / 'trajectory.json')
        assert len(trajectory) == 129 and [row['step'] for row in trajectory] == list(range(129))
        assert 'controller' in trajectory[16] and trajectory[16]['controller'].keys() == trajectory[0]['controller'].keys()
        sources[str(seed)] = dict(folder=str(folder), q0_source=cases[str(seed)],
            files_current_sha256={name: sha(folder / name) for name in source_files},
            q0_normalized_actions=read(folder / 'normalized_actions.json')[:16],
            q0_raw_actions=read(folder / 'denormalized_actions.json')[:16],
            seal_scope='These whole-trial hashes are first observed and sealed by this preparation, not a historical threshold whole-case seal.')
    rules = {trial: dict(q0_source_seed=source, later_noise_base=later,
        actual_query_seeds=actual_seeds(source, later),
        seed_rule='q0: saved source seed; queries1..7: later_noise_base + query',
        native_control=source == later) for trial, (source, later) in PLAN.items()}
    frozen = dict(script_sha256=sha(Path(__file__)), reader_sha256=READER_SHA,
        threshold_contract=original, threshold_checks=threshold, sources=sources, trials=rules,
        extra_code={str(path): sha(path) for path in (reader_path, metrics_path)},
        feedback_provenance_sha256=sha(baseline / 'feedback-inputs/provenance.json'),
        actual_transformer_sha256=feedback['actual_transformer_sha256'],
        actual_pipeline_source_sha256=feedback['actual_pipeline_source_sha256'],
        url=f'http://127.0.0.1:{PORT}', trial_order=list(PLAN),
        saved_q0_predictions=4, native_predictions=28, native_model_forwards=840,
        scope=SCOPE, inputs_symlink_target=str(threshold_root / 'inputs'))
    simulator.SHIFTS = dict(x00=0, x09=9, x12=12)
    simulator.SEEDS = (195, 198)
    simulator.TRIALS = {trial: ('x12', QuerySeed(source, later)) for trial, (source, later) in PLAN.items()}
    simulator.URL, simulator.SCOPE = frozen['url'], SCOPE
    return simulator, frozen, metrics_path


def prepared(baseline):
    simulator, frozen, metrics_path = contract(baseline)
    out = baseline / 'query-noise-cross'
    assert out.is_dir() and not (out / 'wrapper_failed.json').exists()
    assert (out / 'inputs').is_symlink() and (out / 'inputs').resolve() == Path(frozen['inputs_symlink_target']).resolve()
    assert read(out / 'provenance.json') == frozen
    gate = read(out / 'prepared.json')
    assert gate['state'] == 'prepared' and gate['provenance_sha256'] == sha(out / 'provenance.json')
    return simulator, frozen, metrics_path, out


def server(baseline):
    _, frozen, metrics_path, out = prepared(baseline)
    destination = out / 'server'
    destination.mkdir(exist_ok=False)
    served, noises, paired_count, native_count, saved_count = [], {}, 0, 0, 0
    http, failed = None, False
    try:
        os.environ['HF_HUB_OFFLINE'] = '1'
        paths = frozen['threshold_contract']['frozen_files']
        base = load('noise_cross_native_runtime', Path(paths['normal_runtime']['path']))
        base.TASKS['milk'] = (7, 'pick_up_the_milk_and_place_it_in_the_basket', PROMPT, 'milk_1')
        factory = load('noise_cross_checked_factory', Path(paths['factory']['path']))
        factory.OUT = destination
        assert sha(factory.SCHEDULER) == paths['scheduler']['sha256']
        runtime = base.NormalRuntime(factory)
        torch = runtime.torch
        model = runtime.pipe.transformer
        assert not model.training and not model.is_cache_enabled and not torch.is_grad_enabled()
        assert sha(Path(inspect.getfile(type(model)))) == frozen['actual_transformer_sha256']
        assert sha(Path(inspect.getfile(type(runtime.pipe)))) == frozen['actual_pipeline_source_sha256']
        scan = load('noise_cross_record_contract', metrics_path)
        q0 = {}
        for seed, source in frozen['sources'].items():
            folder = Path(source['folder'])
            record = torch.load(folder / 'chunk_00/states.pt', map_location='cpu', weights_only=True, mmap=True)
            scan.validate_record(record, torch)
            assert record['actions'].tolist() == source['q0_normalized_actions']
            q0[int(seed)] = record
        schedule = q0[195]['timesteps'], q0[195]['sigmas']
        runtime.require_exact(schedule, (q0[198]['timesteps'], q0[198]['sigmas']), 'original schedules')
        write(destination / 'provenance.json', dict(script_sha256=sha(Path(__file__)),
            wrapper_provenance_sha256=sha(out / 'provenance.json'),
            extra_q0_model_forwards=0, component_hooks_installed=False, plan=frozen['trials']), exclusive=True)
        started = time.perf_counter()

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                assert self.path == '/health'
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'ready')

            def do_POST(self):
                nonlocal failed, paired_count, native_count, saved_count
                try:
                    assert self.path == '/predict'
                    request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                    assert set(request) == {'trial', 'query', 'seed', 'prompt'}
                    trial, query = request['trial'], request['query']
                    assert trial in PLAN and type(query) is int and 0 <= query < 8
                    rule = frozen['trials'][trial]
                    seed = rule['actual_query_seeds'][query]
                    assert request['seed'] == seed and request['prompt'] == PROMPT
                    assert (trial, query) not in served and all((trial, previous) in served for previous in range(query))
                    source = frozen['sources'][str(rule['q0_source_seed'])]
                    old = Path(source['folder'])
                    run = out / 'closed-loop' / trial
                    image, chunk = run / f'input_{query:02d}.png', run / f'chunk_{query:02d}'
                    if query == 0:
                        assert image.read_bytes() == (old / 'input_00.png').read_bytes()
                        chunk.mkdir(exist_ok=False)
                        for name in ('states.pt', 'normalized_actions.json'):
                            assert sha(old / 'chunk_00' / name) == source['files_current_sha256']['chunk_00/' + name]
                            shutil.copyfile(old / 'chunk_00' / name, chunk / name)
                        record, metadata = q0[seed], read(old / 'chunk_00/metadata.json')
                        saved_count += 1
                    else:
                        record, metadata = runtime.predict('milk', image, seed, chunk)
                        native_count += 1
                        if rule['native_control']:
                            assert image.read_bytes() == (old / f'input_{query:02d}.png').read_bytes()
                            reference_name = f'chunk_{query:02d}/states.pt'
                            assert sha(old / reference_name) == source['files_current_sha256'][reference_name]
                            reference = torch.load(old / reference_name, map_location='cpu', weights_only=True, mmap=True)
                            runtime.require_exact(record, reference, f'{trial}/q{query}/ENTIRE_native_record')
                    scan.validate_record(record, torch)
                    runtime.require_exact((record['timesteps'], record['sigmas']), schedule, 'common actual schedule')
                    key = seed, query
                    paired = key in noises
                    if paired:
                        runtime.require_exact(record['pure_noise'], noises[key], f'actual repeated noise/{seed}/q{query}')
                        paired_count += 1
                    else:
                        noises[key] = record['pure_noise']
                    assert metadata['seed'] == seed and metadata['task_index'] == 7 and metadata['prompt'] == PROMPT
                    metadata.update(trial=trial, query=query, shift_cm=12, image_path=str(image),
                        input_png_sha256=sha(image), saved_q0_prediction=query == 0,
                        current_execution_model_calls=0 if query == 0 else 30,
                        source_record_model_calls=30 if query == 0 else None,
                        q0_source_seed=rule['q0_source_seed'], later_noise_base=rule['later_noise_base'],
                        actual_query_seeds=rule['actual_query_seeds'], seed_rule=rule['seed_rule'],
                        component_intervention_in_this_execution=False,
                        source_q0_states_sha256=source['files_current_sha256']['chunk_00/states.pt'] if query == 0 else None)
                    write(chunk / 'metadata.json', metadata)
                    assert read(chunk / 'normalized_actions.json') == record['actions'].tolist()
                    served.append((trial, query))
                    write(destination / 'progress.json', dict(queries=len(served), saved_q0_predictions=saved_count,
                        native_predictions=native_count, native_model_forwards=30 * native_count,
                        actual_query_seeds=frozen['trials'], served_queries=served, elapsed_s=time.perf_counter() - started))
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(dict(actions=record['actions'].tolist(), metadata=metadata,
                        paired_noise_checked_exact=paired), allow_nan=False).encode())
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
        assert not failed and len(served) == 32 and saved_count == 4 and native_count == 28 and paired_count == 16
        write(destination / 'complete.json', dict(state='complete', queries=32, saved_q0_predictions=4,
            native_predictions=28, native_model_forwards=840, paired_noise_checks=16,
            extra_q0_model_forwards=0, component_hooks_installed=False,
            all_native_control_query_records_entire_exact=True,
            provenance_sha256=sha(destination / 'provenance.json'), wrapper_provenance_sha256=sha(out / 'provenance.json')), exclusive=True)
    except Exception as exc:
        if not (destination / 'failed.json').exists():
            write(destination / 'failed.json', dict(error=str(exc), traceback=traceback.format_exc(), served_queries=served))
        raise
    finally:
        if http is not None:
            http.server_close()


def native_control(out, trial, source, np):
    current, old = out / 'closed-loop' / trial, Path(source['folder'])
    a, b = np.load(current / 'sim_states.npy', allow_pickle=False), np.load(old / 'sim_states.npy', allow_pickle=False)
    assert a.shape == b.shape and a.shape[0] == 129 and a.dtype == b.dtype
    assert a.tobytes(order='C') == b.tobytes(order='C')
    files = (*PHYSICAL_JSON, *[f'input_{query:02d}.png' for query in range(8)])
    for name in files:
        assert (current / name).read_bytes() == (old / name).read_bytes(), ('native whole physical artifact', trial, name)
    return dict(trial=trial, all_129_sim_state_bytes_exact=True,
        all_128_action_and_trajectory_JSON_and_8_PNG_bytes_exact=True,
        files_sha256={name: sha(current / name) for name in ('sim_states.npy', *files)})


def simulate(baseline):
    simulator, frozen, _, out = prepared(baseline)
    assert not (out / 'closed-loop').exists(), 'No retry or partial output reuse'
    os.environ.setdefault('MUJOCO_GL', 'egl')
    sys.path.insert(0, '/home/current/work/openpi-demo/cosmos-policy')
    sys.path.insert(0, str(ROOT / 'cosmos-framework'))
    import requests
    collector = simulator.load_file('noise_cross_collector', ROOT / 'work/collect_cosmos_goal_pair.py')
    rollout = simulator.load_file('noise_cross_controller', ROOT / 'work/rollout_cosmos_goal_pair.py')
    np = simulator.np
    original_post, original_measure, original_summary = requests.post, simulator.measure, simulator.summarize_trial
    active, controls, prefix_checks = None, [], []
    native_controllers = {}

    def measure(env, obs, step, helper):
        result = original_measure(env, obs, step, helper)
        if step == 16:
            arrays = helper.controller_state(env.env)
            np.savez_compressed(out / 'closed-loop' / active / 'controller-step16.npz', **arrays)
            rule = frozen['trials'][active]
            source_seed = rule['q0_source_seed']
            if rule['native_control']:
                native_controllers[source_seed] = {key: value.copy() for key, value in arrays.items()}
            else:
                reference = native_controllers[source_seed]
                assert arrays.keys() == reference.keys()
                assert all(rollout.array_exact(value, reference[key]) for key, value in arrays.items())
        return result

    def post(url, *args, **kwargs):
        request = kwargs['json']
        assert url == frozen['url'] + '/predict' and request['trial'] == active
        query, rule = request['query'], frozen['trials'][active]
        assert request['seed'] == rule['actual_query_seeds'][query]
        if query == 1:
            folder = out / 'closed-loop' / active
            old = Path(frozen['sources'][str(rule['q0_source_seed'])]['folder'])
            a, b = np.load(folder / 'sim_states.npy', allow_pickle=False), np.load(old / 'sim_states.npy', allow_pickle=False)[:17]
            assert a.shape == b.shape and a.shape[0] == 17 and a.dtype == b.dtype
            assert a.tobytes(order='C') == b.tobytes(order='C')
            for name in PHYSICAL_JSON:
                count = 17 if name == 'trajectory.json' else 16
                assert read(folder / name) == read(old / name)[:count], ('q0 source prefix', active, name)
            assert (folder / 'input_01.png').read_bytes() == (old / 'input_01.png').read_bytes()
            assert (folder / 'controller-step16.npz').is_file()
            gate = dict(trial=active, q0_source_seed=rule['q0_source_seed'],
                old_17_sim_state_bytes_exact=True, old_17_trajectory_and_controller_values_exact=True,
                old_16_normalized_raw_native_executed_actions_exact=True, old_input01_PNG_bytes_exact=True,
                actual_controller_original_dtype_arrays_saved=True,
                controller_bytes_exact_native_counterpart=not rule['native_control'],
                controller_evidence='Old controller field values/shape are exact via trajectory JSON. Full dtype/raw-byte exact is additionally checked against the new completed native counterpart for crossed cases.',
                actual_controller_sha256=sha(folder / 'controller-step16.npz'))
            write(folder / 'q1-entry-control.json', gate, exclusive=True)
            prefix_checks.append(gate)
        return original_post(url, *args, **kwargs)

    def summary(trial, scene, seed, records, helper):
        rule = frozen['trials'][trial]
        result = original_summary(trial, scene, int(seed), records, helper)
        result.update(q0_source_seed=rule['q0_source_seed'], later_noise_base=rule['later_noise_base'],
            actual_query_seeds=rule['actual_query_seeds'], seed_rule=rule['seed_rule'],
            noise_seed_base=int(seed), noise_seed_base_applies_to='queries1..7 only; q0 replays its saved source')
        return result

    requests.post, simulator.measure, simulator.summarize_trial = post, measure, summary
    try:
        for trial in PLAN:
            active = trial
            simulator.execute_trials(out, [trial], collector, rollout)
            rule = frozen['trials'][trial]
            if rule['native_control']:
                source = frozen['sources'][str(rule['q0_source_seed'])]
                gate = native_control(out, trial, source, np)
                controls.append(gate)
                write(out / (trial + '-control.json'), gate, exclusive=True)
        result = read(out / 'closed-loop/complete.json')
        assert result['state'] == 'complete' and result['cases'] == 4 and result['completed_trials'] == list(PLAN)
        assert len(controls) == 2 and len(prefix_checks) == 4
        served = read(out / 'server/complete.json')
        assert served['state'] == 'complete' and served['queries'] == 32 and served['native_predictions'] == 28
        write(out / 'controls.json', dict(native_whole_controls=controls, q1_entry_controls=prefix_checks), exclusive=True)
        files = ('controls.json', 'provenance.json', 'prepared.json', 'server/complete.json', 'closed-loop/complete.json', 'summary.json')
        write(out / 'complete.json', dict(state='complete', trial_order=list(PLAN), physical_trials=4,
            saved_q0_predictions=4, native_predictions=28, native_model_forwards=840,
            two_native_complete_physical_controls_exact=True, four_q1_entry_source_prefixes_exact=True,
            crossed_step16_controller_original_dtype_arrays_exact=True,
            files_sha256={name: sha(out / name) for name in files}, script_sha256=sha(Path(__file__)), scope=SCOPE), exclusive=True)
    finally:
        requests.post, simulator.measure, simulator.summarize_trial = original_post, original_measure, original_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('prepare', 'server', 'simulate'), required=True)
    args = parser.parse_args()
    baseline, out = args.output.resolve(), args.output.resolve() / 'query-noise-cross'
    if args.mode == 'prepare':
        _, frozen, _ = contract(baseline)
        out.mkdir(exist_ok=False)
        (out / 'inputs').symlink_to(frozen['inputs_symlink_target'], target_is_directory=True)
        write(out / 'provenance.json', frozen, exclusive=True)
        write(out / 'prepared.json', dict(state='prepared', provenance_sha256=sha(out / 'provenance.json')), exclusive=True)
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
