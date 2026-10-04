"""Execute saved q0 component pulses, then normal closed-loop inference.

Three previously measured pulses are used once, at the same verified x15
input and seed. All subsequent predictions are native. The control q0 must
repeat every recorded baseline tensor exactly. This server never edits weights.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer

ROOT = Path('/home/current/work/cosmos3')
PROMPT = 'pick up the milk and place it in the basket'
PORT = 8926


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Completed baseline root')
    args = parser.parse_args()
    baseline = args.output.resolve()
    out = baseline / 'candidate-execution'
    selection = read(out / 'selection.json')
    pulse_complete = read(baseline / 'component-pulses/complete.json')
    assert pulse_complete['state'] == 'complete' and pulse_complete['scan_predictions'] == 56
    results = read(baseline / 'component-pulses/results.json')
    assert sha(baseline / 'component-pulses/results.json') == pulse_complete['results_sha256']
    top = results['top_three_candidates']
    assert len(top) == 3 and selection['seed'] == results['paired_seed']
    seed = selection['seed']
    plan = {'control': None, **{f'candidate{i:02d}': row for i, row in enumerate(top)}}
    server_out = out / 'server'
    server_out.mkdir(exist_ok=False)
    os.environ['HF_HUB_OFFLINE'] = '1'
    served, noises, failed, server = [], {}, False, None
    try:
        base_path = ROOT / 'work/serve_cosmos_goal_pair.py'
        factory_path = ROOT / 'work/probe_cosmos_language_routes.py'
        baseline_provenance = read(baseline / 'server/provenance.json')
        assert sha(base_path) == baseline_provenance['normal_runtime_sha256']
        assert sha(factory_path) == baseline_provenance['factory_sha256']
        base = load_file('pulse_execution_normal_runtime', base_path)
        base.TASKS['milk'] = (7, 'pick_up_the_milk_and_place_it_in_the_basket', PROMPT, 'milk_1')
        factory = load_file('pulse_execution_checked_factory', factory_path)
        factory.OUT = server_out
        write(server_out / 'provenance.json', dict(
            script_sha256=sha(Path(__file__)), selection_sha256=sha(out / 'selection.json'),
            baseline_provenance=baseline_provenance, pulse_complete=pulse_complete,
            plan=plan, paired_seed=seed, queries_per_trial=8,
            intervention='Only saved q0 pulse; q1 through q7 normal inference on actual new images.',
            scope='Four actual executions: baseline control plus three exploratory pulse candidates. Not a mechanism confirmation.'))
        began = time.perf_counter()
        runtime = base.NormalRuntime(factory)
        torch = runtime.torch
        ref = torch.load(baseline / f'closed-loop/x15_seed{seed}/chunk_00/states.pt',
                         map_location='cpu', weights_only=True)
        assert (out / 'inputs/x15/input.png').read_bytes() == (baseline / f'closed-loop/x15_seed{seed}/input_00.png').read_bytes()

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path != '/health':
                    self.send_error(404)
                    return
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'ready')

            def do_POST(self):
                nonlocal failed
                try:
                    assert self.path == '/predict'
                    request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                    assert set(request) == {'trial', 'query', 'seed', 'prompt'}
                    trial, query = request['trial'], request['query']
                    assert trial in plan and type(query) is int and 0 <= query < 8
                    assert request['seed'] == seed + query and request['prompt'] == PROMPT
                    assert (trial, query) not in served
                    assert all((trial, q) in served for q in range(query))
                    run = out / 'closed-loop' / trial
                    image, destination = run / f'input_{query:02d}.png', run / f'chunk_{query:02d}'
                    assert image.is_file()
                    if query == 0:
                        assert image.read_bytes() == (baseline / 'inputs/x15/input.png').read_bytes()
                    candidate = plan[trial] if query == 0 else None
                    if candidate is None:
                        record, metadata = runtime.predict('milk', image, seed + query, destination)
                        if query == 0:
                            runtime.require_exact(record, ref, 'execution_control_q0_all_baseline_tensors')
                    else:
                        source = baseline / 'component-pulses/runs' / candidate['run']
                        assert sha(source / 'states.pt') == candidate['states_sha256']
                        assert sha(source / 'patch_report.pt') == candidate['patch_report_sha256']
                        assert sha(source / 'metadata.json') == candidate['runtime_metadata_sha256']
                        record = torch.load(source / 'states.pt', map_location='cpu', weights_only=True)
                        for key in ('pure_noise', 'timesteps', 'sigmas', 'model_input', 'prepared_latents_and_masks'):
                            runtime.require_exact(record[key], ref[key], 'saved_pulse_native_input/' + key)
                        destination.mkdir(exist_ok=False)
                        for name in ('states.pt', 'normalized_actions.json', 'patch_report.pt', 'patch_report.json', 'metrics.json'):
                            shutil.copyfile(source / name, destination / name)
                        metadata = read(source / 'metadata.json')
                    key = query
                    paired = key in noises
                    if paired:
                        runtime.require_exact(record['pure_noise'], noises[key], f'paired_execution_noise/{query}')
                    else:
                        noises[key] = record['pure_noise']
                    metadata.update(trial=trial, query=query, shift_cm=15, image_path=str(image),
                                    input_png_sha256=sha(image), paired_noise_checked_exact=paired,
                                    saved_q0_pulse=candidate is not None,
                                    pulse_run=None if candidate is None else candidate['run'])
                    write(destination / 'metadata.json', metadata)
                    served.append((trial, query))
                    write(server_out / 'progress.json', dict(state='running', predictions=len(served),
                          total_predictions=32, served_queries=served, elapsed_s=time.perf_counter() - began))
                    value = dict(actions=record['actions'].tolist(), metadata=metadata, paired_noise_checked_exact=paired)
                    print(f'[PULSE-EXECUTION] {len(served)}/32 {trial} q{query}', flush=True)
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(value, allow_nan=False).encode())
                except Exception as exc:
                    failed = True
                    write(server_out / 'failed.json', dict(error=str(exc), traceback=traceback.format_exc(), served_queries=served))
                    self.send_error(500, str(exc))

        server = HTTPServer(('127.0.0.1', PORT), Handler)
        server.timeout = 30
        write(server_out / 'ready.json', dict(state='ready', url=f'http://127.0.0.1:{PORT}'))
        while not failed and len(served) < 32:
            server.handle_request()
            assert not (out / 'closed-loop/failed.json').exists(), 'Simulator failed; see its preserved diagnostics.'
        assert not failed and len(served) == 32
        write(server_out / 'complete.json', dict(state='complete', queries=32,
              normal_predictions=29, saved_single_pulse_predictions=3,
              control_q0_all_baseline_tensors_exact=True, paired_noise_checks=24))
    except Exception as exc:
        if not (server_out / 'failed.json').exists():
            write(server_out / 'failed.json', dict(error=str(exc), traceback=traceback.format_exc()))
        raise
    finally:
        if server is not None:
            server.server_close()


if __name__ == '__main__':
    main()
