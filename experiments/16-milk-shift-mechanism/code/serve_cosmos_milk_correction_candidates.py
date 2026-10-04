"""Serve four saved-q0 correction candidates, followed by normal predictions.

Run after rollout_cosmos_milk_correction_candidates.py --prepare-only.
Port8928 is separate from position-threshold port8927. Four saved q0 records
and 28 normal predictions serve 32 queries. No intervention hooks are installed
and no extra q0 forward is run. Actual q1..7 input images are supplied by the
unchanged baseline simulator. No checkpoint or inference-library file is edited.
"""

import argparse
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer


PROMPT = 'pick up the milk and place it in the basket'
PORT = 8928


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Completed original baseline root')
    baseline = parser.parse_args().output.resolve()
    wrapper = load_file('correction_execution_physical_contract', Path(__file__).with_name('rollout_cosmos_milk_correction_candidates.py'))
    simulator, selection, provenance = wrapper.verify_prepared(baseline)
    out = baseline / 'correction-execution'
    server_out = out / 'server'
    server_out.mkdir(exist_ok=False)
    sha, read = wrapper.sha, wrapper.read_json
    os.environ['HF_HUB_OFFLINE'] = '1'
    served, noises, paired_checks = [], {}, 0
    saved_count, native_count, failed, server = 0, 0, False, None
    try:
        base_path = Path(provenance['code']['normal_runtime']['path'])
        factory_path = Path(provenance['code']['factory']['path'])
        base = load_file('correction_execution_normal_runtime', base_path)
        base.TASKS['milk'] = (7, 'pick_up_the_milk_and_place_it_in_the_basket', PROMPT, 'milk_1')
        factory = load_file('correction_execution_checked_factory', factory_path)
        factory.OUT = server_out
        persistence = read(baseline / 'correction-persistence/provenance.json')
        assert sha(factory.SCHEDULER) == persistence['scheduler_sha256']
        write(server_out / 'provenance.json', dict(script_sha256=sha(Path(__file__)),
            selection_sha256=sha(out / 'selection.json'), execution_provenance_sha256=sha(out / 'provenance.json'),
            plan=selection['trials'], port=PORT, seed=wrapper.SEED,
            q0_saved_predictions=4, native_new_image_predictions=28, component_interventions_installed=False,
            scope=wrapper.SCOPE))
        began = time.perf_counter()
        runtime = base.NormalRuntime(factory)
        torch = runtime.torch
        assert not runtime.pipe.transformer.training and not torch.is_grad_enabled()
        assert not runtime.pipe.transformer.is_cache_enabled
        assert sha(Path(inspect.getfile(type(runtime.pipe.transformer)))) == persistence['actual_transformer_sha256']
        assert sha(Path(inspect.getfile(type(runtime.pipe)))) == persistence['actual_pipeline_source_sha256']
        scan = load_file('correction_execution_saved_record_checks', Path(provenance['code']['pulse_contract']['path']))
        reference = torch.load(baseline / 'closed-loop/x15_seed198/chunk_00/states.pt', map_location='cpu', weights_only=True, mmap=True)
        scan.validate_record(reference, torch)
        sources = read(baseline / 'correction-persistence/seed198/sources.json')
        p_reference = torch.load(Path(sources['feedback']['feedback_P_states']['path']), map_location='cpu', weights_only=True, mmap=True)
        q0 = {}
        for trial in wrapper.TRIALS:
            chosen = selection['trials'][trial]
            source = Path(chosen['directory'])
            for name in wrapper.FILES:
                assert sha(source / name) == chosen['files_sha256'][name]
            record = torch.load(source / 'states.pt', map_location='cpu', weights_only=True, mmap=True)
            scan.validate_record(record, torch)
            scan.validate_run_contract(record, reference, runtime, 'saved_q0/' + trial)
            assert record['actions'].tolist() == read(source / 'normalized_actions.json')
            if trial == 'native_R':
                runtime.require_exact(record, reference, 'saved_native_R_ENTIRE_baseline_record')
            elif trial == 'P':
                runtime.require_exact(record, p_reference, 'saved_P_ENTIRE_feedback_record')
            else:
                for key in ('readout', 'action_velocity'):
                    runtime.require_exact(record[key][:21], p_reference[key][:21], trial + '/actual_P_before_reset/' + key)
                runtime.require_exact(record['action_states'][:22], p_reference['action_states'][:22], trial + '/actual_P_solver_before_reset')
            q0[trial] = record
        assert (out / 'inputs/x15/input.png').read_bytes() == (baseline / 'closed-loop/x15_seed198/input_00.png').read_bytes()
        write(server_out / 'saved-q0-control.json', dict(state='exact', trials=list(wrapper.TRIALS),
            native_R_entire_baseline_record_bit_exact=True, P_entire_feedback_record_bit_exact=True,
            all_two_draw_noise_schedule_initial_kwargs_prepared_contracts_exact=True,
            persistent_and_random_before_reset_prefix_bit_exact_P=True, extra_q0_model_forwards=0,
            source_states_sha256={trial: selection['trials'][trial]['files_sha256']['states.pt'] for trial in wrapper.TRIALS}))

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path != '/health':
                    self.send_error(404)
                    return
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'ready')

            def do_POST(self):
                nonlocal failed, paired_checks, saved_count, native_count
                try:
                    assert self.path == '/predict'
                    request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                    assert set(request) == {'trial', 'query', 'seed', 'prompt'}
                    trial, query = request['trial'], request['query']
                    assert trial in wrapper.TRIALS and type(query) is int and 0 <= query < 8
                    assert request['seed'] == wrapper.SEED + query and request['prompt'] == PROMPT
                    assert (trial, query) not in served and all((trial, previous) in served for previous in range(query))
                    run = out / 'closed-loop' / trial
                    image, destination = run / f'input_{query:02d}.png', run / f'chunk_{query:02d}'
                    assert image.is_file()
                    if query == 0:
                        assert image.read_bytes() == (baseline / 'inputs/x15/input.png').read_bytes()
                        chosen = selection['trials'][trial]
                        source = Path(chosen['directory'])
                        for name in wrapper.FILES:
                            assert sha(source / name) == chosen['files_sha256'][name]
                        record = q0[trial]
                        destination.mkdir(exist_ok=False)
                        for name in wrapper.FILES:
                            if name != 'metadata.json':
                                shutil.copyfile(source / name, destination / name)
                        metadata = read(source / 'metadata.json')
                        saved_count += 1
                    else:
                        record, metadata = runtime.predict('milk', image, wrapper.SEED + query, destination)
                        native_count += 1
                    scan.validate_record(record, torch)
                    runtime.require_exact((record['timesteps'], record['sigmas']),
                                          (reference['timesteps'], reference['sigmas']), 'actual_query_common_schedule')
                    paired = query in noises
                    if paired:
                        runtime.require_exact(record['pure_noise'], noises[query], f'actual_paired_raw_noise/query{query}')
                        paired_checks += 1
                    else:
                        noises[query] = record['pure_noise']
                    metadata.update(trial=trial, query=query, shift_cm=15, image_path=str(image),
                        input_png_sha256=sha(image), paired_noise_checked_exact=paired,
                        saved_q0_prediction=query == 0, saved_q0_arm=trial if query == 0 else None,
                        component_intervention_in_this_execution=False,
                        saved_source_states_sha256=selection['trials'][trial]['files_sha256']['states.pt'] if query == 0 else None)
                    write(destination / 'metadata.json', metadata)
                    assert read(destination / 'normalized_actions.json') == record['actions'].tolist()
                    served.append((trial, query))
                    write(server_out / 'progress.json', dict(state='running', queries=len(served), total_queries=32,
                        saved_q0_predictions=saved_count, native_new_image_predictions=native_count,
                        served_queries=served, elapsed_s=time.perf_counter() - began))
                    print(f'[CORRECTION-EXECUTION] {len(served)}/32 {trial} q{query}', flush=True)
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(dict(actions=record['actions'].tolist(), metadata=metadata,
                        paired_noise_checked_exact=paired), allow_nan=False).encode())
                except Exception as exc:
                    failed = True
                    write(server_out / 'failed.json', dict(error=str(exc), traceback=traceback.format_exc(), served_queries=served))
                    self.send_error(500, str(exc))

        server = HTTPServer(('127.0.0.1', PORT), Handler)
        server.timeout = 30
        write(server_out / 'ready.json', dict(state='ready', url=f'http://127.0.0.1:{PORT}', saved_q0_gate='exact'))
        while not failed and len(served) < 32:
            server.handle_request()
            assert not (out / 'closed-loop/failed.json').exists()
            assert not (out / 'wrapper_failed.json').exists()
        assert not failed and len(served) == 32 and saved_count == 4 and native_count == 28 and paired_checks == 24
        write(server_out / 'complete.json', dict(state='complete', queries=32, saved_q0_predictions=4,
            native_new_image_predictions=28, paired_noise_checks=24, extra_q0_model_forwards=0,
            saved_native_R_entire_record_equals_baseline=True, all_saved_q0_contracts_exact=True,
            component_interventions_installed=False, saved_q0_control_sha256=sha(server_out / 'saved-q0-control.json'),
            provenance_sha256=sha(server_out / 'provenance.json'), scope=wrapper.SCOPE))
    except Exception as exc:
        if not (server_out / 'failed.json').exists():
            write(server_out / 'failed.json', dict(error=str(exc), traceback=traceback.format_exc(), served_queries=served))
        raise
    finally:
        if server is not None:
            server.server_close()


if __name__ == '__main__':
    main()
