"""Serve paired milk-position trials with checked normal inference.

Model-library files are unchanged. Observational component hooks record only
the sixteen actual GEN action rows, and only at query zero in each trial.
The first query is repeated without those hooks to check numerical neutrality.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer

ROOT = Path('/home/current/work/cosmos3')
PROMPT = 'pick up the milk and place it in the basket'
SEEDS = (195, 196, 198)
SHIFTS = {'x00': 0, 'x06': 6, 'x15': 15}
PORT = 8925


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    # Simulator collection owns the root and inputs. No existing server run is reused.
    input_complete = json.loads((out / 'inputs/complete.json').read_text())
    assert input_complete['state'] == 'matched_inputs_verified'
    assert input_complete['metadata_sha256'] == sha(out / 'inputs/metadata.json')
    assert input_complete['source_center_pixels_exact'] is True
    assert not (out / 'inputs/failed.json').exists()
    input_metadata = json.loads((out / 'inputs/metadata.json').read_text())
    assert input_metadata['prompt'] == PROMPT and input_metadata['shifts_cm'] == SHIFTS
    (out / 'server').mkdir(exist_ok=False)
    server_out = out / 'server'
    os.environ['HF_HUB_OFFLINE'] = '1'
    served, noise_by_query, schedule_reference = [], {}, None
    server, components = None, None
    try:
        base_path = ROOT / 'work/serve_cosmos_goal_pair.py'
        factory_path = ROOT / 'work/probe_cosmos_language_routes.py'
        component_path = ROOT / 'work/cosmos_component_interventions.py'
        base = load_file('milk_shift_normal_runtime', base_path)
        base.TASKS['milk'] = (7, 'pick_up_the_milk_and_place_it_in_the_basket', PROMPT, 'milk_1')
        factory = load_file('milk_shift_checked_factory', factory_path)
        factory.OUT = server_out
        write_json(server_out / 'provenance.json', dict(
            script_sha256=sha(Path(__file__)), normal_runtime_sha256=sha(base_path),
            factory_sha256=sha(factory_path), components_sha256=sha(component_path),
            inputs_metadata_sha256=sha(out / 'inputs/metadata.json'),
            pixel_helper_sha256=sha(ROOT / 'work/check_cosmos_official_pixels_cpu.py'),
            scheduler_sha256=sha(factory.SCHEDULER),
            prompt=PROMPT, shifts_cm=SHIFTS, seeds=list(SEEDS), queries_per_trial=8,
            library_edits=False, local_interventions=False,
            scope='Paired position baseline. Query-zero observational attention/MLP component recordings; no claim of localized causal mechanism.'))
        began = time.perf_counter()
        runtime = base.NormalRuntime(factory)
        module = load_file('milk_shift_component_hooks', component_path)
        components = module.CosmosComponentInterventions(runtime.pipe.transformer)
        trials = {f'{position}_seed{seed}': (position, seed)
                  for position in SHIFTS for seed in SEEDS}
        failed = False

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path != '/health':
                    self.send_error(404)
                    return
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'ready')

            def do_POST(self):
                nonlocal failed, schedule_reference
                try:
                    assert self.path == '/predict'
                    request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                    assert set(request) == {'trial', 'query', 'seed', 'prompt'}
                    trial, query = request['trial'], request['query']
                    assert trial in trials and type(query) is int and 0 <= query < 8
                    position, seed = trials[trial]
                    assert request['seed'] == seed + query and request['prompt'] == PROMPT
                    assert (trial, query) not in served, 'No duplicate query or automatic retry.'
                    assert all((trial, previous) in served for previous in range(query))
                    run = out / 'closed-loop' / trial
                    image = run / f'input_{query:02d}.png'
                    assert image.is_file()
                    if query == 0:
                        assert image.read_bytes() == (out / 'inputs' / position / 'input.png').read_bytes()
                    capture = query == 0
                    if capture:
                        components.begin_capture(trial, expected_steps=30)
                    try:
                        record, metadata = runtime.predict('milk', image, seed + query,
                                                           run / f'chunk_{query:02d}')
                    finally:
                        if capture:
                            component_cache = components.end()
                    if capture:
                        runtime.torch.save(component_cache, run / 'chunk_00/components.pt')
                        assert component_cache['complete'] and component_cache['n_steps'] == 30
                        assert component_cache['captured_component_count'] == 30 * 36 * 2
                    if schedule_reference is None:
                        schedule_reference = (record['timesteps'], record['sigmas'])
                    runtime.require_exact((record['timesteps'], record['sigmas']),
                                          schedule_reference, 'common_schedule')
                    key = (seed, query)
                    paired = key in noise_by_query
                    if paired:
                        runtime.require_exact(record['pure_noise'], noise_by_query[key],
                                              f'paired_noise/{seed}/{query}')
                    else:
                        noise_by_query[key] = record['pure_noise']
                    if not served:
                        # All NormalRuntime hooks stay identical; remove only the new component observers.
                        components.reset()
                        repeated, _ = runtime.predict('milk', image, seed + query,
                                                      server_out / 'observer_free_repeat')
                        runtime.require_exact(record, repeated, 'component_observers_numerically_neutral')
                        write_json(server_out / 'observer_control.json', dict(
                            trial=trial, seed=seed, all_normal_runtime_tensors_exact=True))
                    metadata.update(trial=trial, query=query, shift_cm=SHIFTS[position],
                                    component_capture=capture,
                                    paired_noise_checked_exact=paired)
                    write_json(run / f'chunk_{query:02d}/metadata.json', metadata)
                    served.append((trial, query))
                    write_json(server_out / 'progress.json', dict(
                        state='running', served_queries=served, predictions=len(served), total_predictions=72,
                        last_elapsed_s=metadata['elapsed_s'], elapsed_s=time.perf_counter() - began))
                    value = dict(actions=record['actions'].tolist(), metadata=metadata,
                                 paired_noise_checked_exact=paired)
                    print(f'[MILK-SHIFT] {len(served)}/72 {trial} q{query}: {metadata["elapsed_s"]:.2f}s', flush=True)
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(value, allow_nan=False).encode())
                except Exception as exc:
                    failed = True
                    write_json(server_out / 'failed.json', dict(error=str(exc),
                               traceback=traceback.format_exc(), served_queries=served))
                    self.send_error(500, str(exc))

        server = HTTPServer(('127.0.0.1', PORT), Handler)
        server.timeout = 30
        write_json(server_out / 'ready.json', dict(state='ready', url=f'http://127.0.0.1:{PORT}',
                                                  elapsed_s=time.perf_counter() - began))
        print(f'[MILK-SHIFT] Ready localhost:{PORT}', flush=True)
        while not failed and len(served) < 72:
            server.handle_request()
            assert not (out / 'closed-loop/failed.json').exists(), 'Simulator stopped; see its preserved error.'
        assert not failed and len(served) == 72
        write_json(server_out / 'complete.json', dict(state='complete', predictions=72,
                  query_zero_component_captures=9, paired_noise_checks=48,
                  observer_free_repeat_all_tensors_exact=True))
    except Exception as exc:
        if not (server_out / 'failed.json').exists():
            write_json(server_out / 'failed.json', dict(error=str(exc), traceback=traceback.format_exc()))
        raise
    finally:
        if components is not None:
            components.reset()
        if server is not None:
            server.server_close()


if __name__ == '__main__':
    main()
