"""Execute the frozen language x nine-layer-cut cases with actual feedback.

The physical scene remains LIBERO object task7, X+12cm. Only policy language
changes. Q0 comes from its own producer, then seven native feedback queries
use seeds196..202. No q0 ranking. Every preregistered case executes128 actions.
This file preserves the original simulator controller, restoration, capture,
measurement and official action conversion. Scene and policy prompts are
distinct; task7's success predicate continues to mean milk-in-basket.
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
PROBE_SHA = '74ac42db450a1a7cd308c837bf75049cd2dcae858a4869dab1f0f60450c69b13'
HEAD_ADAPTER_SHA = '6767ee5bf166ebee8c0db7172472610311cf9611f88f72b04d6c10509fa3d064'
SIMULATOR_SHA = '7060f610814cbf6b13fb79ca2da78af73bba542d479ac2b8a73110097543d845'
HELPER_SHA = 'e0c5073ee320bbc173eebf827126c9431043a3efaa960eb0a8ccff289cae30ed'
CROSS_SHA = '7e684b1a330877b8dbaad2c27621c3bc8db4893eb25da0c9171ff6d8b56aa2fe'
VALUE_ADAPTER_SHA = 'e900f970e4049f6db1755274d68fa38366a9a29916354ccc8357090c44165a51'
PORT = 8937
STAGE = 'future-language-selectivity-execution'
SCENE_PROMPT = 'pick up the milk and place it in the basket'
SCOPE = ('Fixed X12 task7 physical scene; three actual policy instructions, two '
         'future-noise sources and native/all_allowed/full nine-layer cut. '
         'Q0 only is intervened; seven later native feedback predictions. '
         'Strict grasp of every object is scored independently of task7 milk '
         'basket completion. Language interaction is assessed from actual '
         'producer arrays, not inferred from grasp alone. No identity decoder, '
         'functional brain region, X15 repair or benchmark success-rate claim.')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, exclusive=False):
    with Path(path).open('x' if exclusive else 'w') as stream:
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
    assert len(PROBE_SHA) == 64 and all(c in '0123456789abcdef' for c in PROBE_SHA), 'Producer is not frozen; no execution/output allowed'
    probe_path = ROOT / 'work/probe_cosmos_milk_language_selectivity.py'
    assert sha(probe_path) == PROBE_SHA
    assert sha(ROOT / 'work/run_cosmos_milk_future_head_execution.py') == HEAD_ADAPTER_SHA
    assert sha(ROOT / 'work/run_cosmos_milk_future_value_execution.py') == VALUE_ADAPTER_SHA
    assert sha(ROOT / 'work/run_cosmos_milk_future_read_execution.py') == HELPER_SHA
    assert sha(ROOT / 'work/run_cosmos_milk_query_noise_cross.py') == CROSS_SHA
    probe = load('language_execution_frozen_producer', probe_path)
    producer = baseline / probe.STAGE
    assert producer.is_dir() and not (producer / 'failed.json').exists()
    docs = {name: read(producer / (name + '.json')) for name in ('complete', 'results', 'protocol', 'provenance')}
    complete, result, protocol, provenance = (docs[k] for k in ('complete', 'results', 'protocol', 'provenance'))
    assert complete['state'] == result['state'] == 'complete'
    assert complete['script_sha256'] == provenance['script_sha256'] == PROBE_SHA
    assert complete['fresh_q0_predictions'] == 18 and complete['fresh_model_forwards'] == 540
    assert result['official_dispatch_counts'] == complete['official_dispatch_counts'] == probe.expected_counts()
    assert complete['total_official_dispatch_calls'] == 52650
    assert complete['actual_selected_site_captures'] == 2430
    for name in ('results', 'protocol', 'provenance'):
        assert complete[name + '_sha256'] == sha(producer / (name + '.json'))
    old_out = baseline / 'future-head-execution'
    old_complete, old = read(old_out / 'complete.json'), read(old_out / 'provenance.json')
    assert old_complete['state'] == 'complete' and old_complete['script_sha256'] == HEAD_ADAPTER_SHA
    for name, expected in old_complete['files_sha256'].items():
        assert sha(old_out / name) == expected
    assert old_complete['eight_entire129_state_5JSON_8PNG_and_all8_modelrecords_controls_exact'] is True
    old_controls = read(old_out / 'controls.json')
    original = old['threshold_contract']
    for item in original['frozen_files'].values():
        assert sha(Path(item['path'])) == item['sha256']
    simulator_path = Path(original['frozen_files']['simulator']['path'])
    assert sha(simulator_path) == SIMULATOR_SHA
    simulator = load('language_execution_original_simulator', simulator_path)
    helper = load('language_execution_strict_helper', ROOT / 'work/run_cosmos_milk_future_read_execution.py')
    cross = load('language_execution_byte_controls', ROOT / 'work/run_cosmos_milk_query_noise_cross.py')
    prompts = dict(milk=SCENE_PROMPT, cheese='pick up the cheese and place it in the basket',
                   cream_cheese='pick up the cream cheese and place it in the basket')
    targets = dict(milk='milk_1', cheese='cream_cheese_1', cream_cheese='cream_cheese_1')
    rows, q0, trials, references = result['cases'], {}, {}, {}
    assert len(rows) == 18 and len({r['case'] for r in rows}) == 18
    assert {(r['arm'], r['goal'], r['vision_noise_source_seed']) for r in rows} == {
        (arm, goal, v) for arm in ('native', 'all_allowed', 'full_hardmask') for goal in prompts for v in (195, 198)}
    assert [(r['arm'], r['goal'], r['vision_noise_source_seed']) for r in rows] == [
        (arm, goal, v) for arm in ('native', 'all_allowed', 'full_hardmask') for goal in prompts for v in (195, 198)]
    native_labels = {goal: {} for goal in prompts}
    for row in rows:
        label, arm, goal, v = row['case'], row['arm'], row['goal'], row['vision_noise_source_seed']
        folder = producer / label
        assert row['action_noise_source_seed'] == row['runtime_rng_seed'] == 195
        assert row['fresh_model_forwards'] == 30
        for name, expected in row['files_sha256'].items():
            assert sha(folder / name) == expected
        assert len(row['boundary_files']) == 30
        for t, item in enumerate(row['boundary_files']):
            assert item['step'] == t and item['file'] == f'boundaries-t{t:02d}.pt'
            assert sha(folder / item['file']) == item['sha256']
        manifest = read(folder / 'site-captures.json')
        assert manifest['state'] == 'complete' and manifest['sites'] == len(manifest['files']) == 135
        assert [(i['step'], i['layer_zero_based']) for i in manifest['files']] == [(t, l) for t in range(15) for l in range(9)]
        for item in manifest['files']:
            assert sha(folder / item['file']) == item['sha256']
        metadata = read(folder / 'metadata.json')
        assert metadata['prompt'] == prompts[goal] and metadata['seed'] == 195
        assert metadata['task_index'] == 7
        assert metadata['input_png_sha256'] == sha(baseline / 'position-threshold/inputs/x12/input.png')
        norm = simulator.np.asarray(read(folder / 'normalized_actions.json'))
        assert norm.shape == (16, 10) and simulator.np.isfinite(norm).all()
        q0[label] = dict(folder=str(folder), files_sha256=row['files_sha256'], boundary_files=row['boundary_files'], normalized_actions=norm.tolist())
        window = dict(steps=[0, 14], layers=[0, 8]) if arm == 'full_hardmask' else None
        trials[label] = dict(goal=goal, policy_prompt=prompts[goal], arm=arm, selected_window=window,
            q0_vision_noise_source_seed=v, actual_query_seeds=list(range(195, 203)), whole_reference_control=goal == 'milk')
        if arm == 'native':
            native_labels[goal][str(v)] = label
        if goal == 'milk':
            ref_label = f'V{v}_A195_{arm}'
            ref_folder = old_out / 'closed-loop' / ref_label
            files = old_controls['actual_case_files_sha256'][ref_label]
            references[label] = dict(folder=str(ref_folder), reference_case=ref_label, files_current_sha256=files)
    assert len(references) == 6
    frozen = dict(script_sha256=sha(Path(__file__)), producer_script_sha256=PROBE_SHA, producer_stage=probe.STAGE,
        producer_documents_sha256={name: sha(producer / (name + '.json')) for name in docs},
        threshold_contract=original, actual_transformer_sha256=old['actual_transformer_sha256'],
        actual_pipeline_source_sha256=old['actual_pipeline_source_sha256'], metrics_path=old['metrics_path'],
        components_path=old['components_path'], producer_source_cases=old['producer_source_cases'],
        q0_sources=q0, trials=trials, trial_order=[r['case'] for r in rows], native_labels=native_labels,
        scene_prompt=SCENE_PROMPT, scene_task_index=7, shift_cm=12, prompts=prompts, targets=targets,
        whole_physical_references=references, inputs_symlink_target=str(baseline / 'position-threshold/inputs'),
        old_head_complete_sha256=sha(old_out / 'complete.json'), old_head_controls_sha256=sha(old_out / 'controls.json'),
        producer_protocol=protocol, url=f'http://127.0.0.1:{PORT}', saved_q0_predictions=18,
        requests=144, native_predictions=126, fresh_model_forwards=3780, producer_q0_forwards=540,
        combined_two_stage_model_forwards=4320, later_whole_noise_pairs=119,
        all18_execute_without_score_selection=True, scope=SCOPE)
    simulator.SHIFTS = dict(x00=0, x09=9, x12=12)
    simulator.SCOPE = SCOPE
    return frozen, probe, simulator, helper, cross


def prepared(baseline):
    frozen, probe, simulator, helper, cross = contract(baseline)
    out = baseline / STAGE
    assert out.is_dir() and not (out / 'wrapper_failed.json').exists()
    assert (out / 'inputs').is_symlink()
    assert (out / 'inputs').resolve() == Path(frozen['inputs_symlink_target']).resolve()
    assert read(out / 'provenance.json') == frozen
    assert read(out / 'prepared.json') == dict(state='prepared', provenance_sha256=sha(out / 'provenance.json'))
    return frozen, probe, simulator, helper, cross, out


def verify_saved(record, audit, folder, native, sources, v, runtime, components, probe):
    """All30 small raw fields; four full raw calls; omitted calls retain live SHA."""
    torch = runtime.torch
    runtime.require_exact(record['pure_noise'], [sources[v]['pure_noise'][0], sources[195]['pure_noise'][1]], 'selected entire original V/A draws')
    runtime.require_exact(record['prepared_latents_and_masks'], native['prepared_latents_and_masks'], 'same-language/V complete preparation')
    runtime.require_exact(record['model_input'], native['model_input'], 'same-language/V whole first kwargs')
    runtime.require_exact(record['model_input'], audit['initial_model_kwargs'], 'observed initial kwargs')
    runtime.require_exact(record['action_states'][0], sources[195]['action_states'][0], 'original A195 solver sample')
    assert torch.count_nonzero(record['action_states'][..., 10:]) == 0
    clean = record['model_input']['vision_tokens'][0][0, :, 0]
    clean_meta = probe.tensor_meta(clean, runtime)
    for step in range(30):
        data = torch.load(folder / f'boundaries-t{step:02d}.pt', map_location='cpu', weights_only=True, mmap=True)
        runtime.require_exact(data['actual_pack'], audit['model_steps'][step], 'noise observer pack/' + str(step))
        runtime.require_exact(data['actual_model_action_tokens'], record['action_states'][step, 0].to(torch.bfloat16), 'actual solver BF16 cast/' + str(step))
        assert data['actual_current_frame_sha256'] == clean_meta['sha256']
        assert data['actual_current_frame_metadata'] == clean_meta
        runtime.require_exact(data['actual_model_action_timesteps'], data['actual_pack']['metadata']['action_timesteps'], 'actual action timestep/' + str(step))
        runtime.require_exact(data['solver_sigma_from_same_frozen_schedule'], record['sigmas'][step], 'actual same frozen sigma/' + str(step))
        runtime.require_exact(data['actual_model_action_output'], record['action_velocity'][step], 'actual whole model action output/' + str(step))
        assert data['action_hidden'].shape == (37, 16, 4096)
        assert data['action_hidden'].dtype == torch.bfloat16 and bool(torch.isfinite(data['action_hidden']).all())
        if step in probe.CURRENT_STEPS:
            kwargs = data['actual_model_kwargs']
            runtime.require_exact(data['actual_pack'], dict(layout=components._layout(kwargs), metadata={key: kwargs.get(key) for key in components._STRUCTURAL_KEYS}), 'actual complete pack/' + str(step))
            runtime.require_exact(kwargs['action_tokens'][0], data['actual_model_action_tokens'], 'saved action input/' + str(step))
            runtime.require_exact(kwargs['vision_tokens'][0][0, :, 0], clean, 'actual clean current raw clamp/' + str(step))
            runtime.require_exact(data['model_output'][2][0], data['actual_model_action_output'], 'saved model tuple action output/' + str(step))
            assert probe.tree_sha(kwargs, runtime) == data['actual_model_kwargs_sha256']
            assert probe.tree_sha(data['model_output'], runtime) == data['model_output_sha256']
            assert probe.tree_meta(kwargs, runtime) == data['actual_model_kwargs_metadata']
            assert probe.tree_meta(data['model_output'], runtime) == data['model_output_metadata']
            assert data['current_hidden'].shape == (37, 50, 4096)
        else:
            assert data['actual_model_kwargs'] is data['model_output'] is data['current_hidden'] is None
            for key in ('actual_model_kwargs_sha256', 'model_output_sha256'):
                assert len(data[key]) == 64 and all(c in '0123456789abcdef' for c in data[key])
        if step == 0:
            runtime.require_exact(data['actual_model_kwargs'], record['model_input'], 'saved first full kwargs')


def server(baseline):
    frozen, probe, _, _, _, out = prepared(baseline)
    destination = out / 'server'
    destination.mkdir(exist_ok=False)
    served, later_noise, references = [], {}, []
    cached = fresh = pairs = 0
    http, failed, started = None, False, time.perf_counter()
    try:
        os.environ['HF_HUB_OFFLINE'] = '1'
        files = frozen['threshold_contract']['frozen_files']
        base = load('language_execution_runtime', Path(files['normal_runtime']['path']))
        for goal, prompt in frozen['prompts'].items():
            base.TASKS[goal] = (7, 'pick_up_the_milk_and_place_it_in_the_basket', prompt, frozen['targets'][goal])
        factory = load('language_execution_factory', Path(files['factory']['path']))
        factory.OUT = destination
        assert sha(factory.SCHEDULER) == files['scheduler']['sha256']
        runtime = base.NormalRuntime(factory)
        torch, model = runtime.torch, runtime.pipe.transformer
        assert not model.training and not model.is_cache_enabled and not torch.is_grad_enabled()
        assert sha(Path(inspect.getfile(type(model)))) == frozen['actual_transformer_sha256']
        assert sha(Path(inspect.getfile(type(runtime.pipe)))) == frozen['actual_pipeline_source_sha256']
        initial_hooks = hooks(model)
        assert all(not pre and not post for _, pre, post in initial_hooks)
        scan = load('language_execution_metrics', Path(frozen['metrics_path']))
        components = load('language_execution_layout', Path(frozen['components_path']))
        sources, q0 = {}, {}
        for v in (195, 198):
            source = frozen['producer_source_cases'][str(v)]
            folder = Path(source['q0_folder'])
            assert sha(folder / 'states.pt') == source['q0_files_current_sha256']['states.pt']
            sources[v] = torch.load(folder / 'states.pt', map_location='cpu', weights_only=True, mmap=True)
        for label in frozen['trial_order']:
            source = frozen['q0_sources'][label]
            folder = Path(source['folder'])
            q0[label] = torch.load(folder / 'states.pt', map_location='cpu', weights_only=True, mmap=True)
        controls = []
        for label in frozen['trial_order']:
            rule, source, record = frozen['trials'][label], frozen['q0_sources'][label], q0[label]
            folder = Path(source['folder'])
            native = q0[frozen['native_labels'][rule['goal']][str(rule['q0_vision_noise_source_seed'])]]
            audit = torch.load(folder / 'noise-audit.pt', map_location='cpu', weights_only=True, mmap=True)
            scan.validate_record(record, torch)
            verify_saved(record, audit, folder, native, sources, rule['q0_vision_noise_source_seed'], runtime, components, probe)
            if rule['whole_reference_control']:
                ref = frozen['whole_physical_references'][label]
                path = Path(ref['folder']) / 'chunk_00/states.pt'
                assert sha(path) == ref['files_current_sha256']['chunk_00/states.pt']
                runtime.require_exact(record, torch.load(path, map_location='cpu', weights_only=True, mmap=True), 'ENTIRE accepted milk q0/' + label)
            controls.append(dict(case=label, own_language_native_full_preparation_and_initial_kwargs_exact=True,
                                 six_milk_reference_record_exact=True if rule['whole_reference_control'] else None))
        write(destination / 'saved-q0-controls.json', dict(state='exact', cases=controls, extra_q0_model_forwards=0), exclusive=True)

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
                    seed, prompt = rule['actual_query_seeds'][query], rule['policy_prompt']
                    assert request['seed'] == seed and request['prompt'] == prompt
                    image = out / 'closed-loop' / label / f'input_{query:02d}.png'
                    chunk = image.parent / f'chunk_{query:02d}'
                    if query == 0:
                        assert image.read_bytes() == (out / 'inputs/x12/input.png').read_bytes()
                        chunk.mkdir(exist_ok=False)
                        folder = Path(source['folder'])
                        # Immutable large records are linked, not duplicated. Metadata
                        # is separately serialized and never linked to the producer.
                        for name in ('states.pt', 'normalized_actions.json'):
                            assert sha(folder / name) == source['files_sha256'][name]
                            os.link(folder / name, chunk / name)
                        record, metadata = q0[label], read(folder / 'metadata.json')
                        cached, paired = cached + 1, False
                    else:
                        record, metadata = runtime.predict(rule['goal'], image, seed, chunk)
                        fresh += 1
                        paired = query in later_noise
                        if paired:
                            runtime.require_exact(record['pure_noise'], later_noise[query], 'entire later two-draw pair/' + str(query))
                            pairs += 1
                        else:
                            later_noise[query] = record['pure_noise']
                    scan.validate_record(record, torch)
                    assert metadata['prompt'] == prompt and metadata['seed'] == seed
                    assert metadata['task_index'] == 7, 'Do not overwrite an incompatible source scene index'
                    for key in ('timesteps', 'sigmas'):
                        runtime.require_exact(record[key], sources[195][key], 'entire frozen schedule/' + key)
                    assert hooks(model) == initial_hooks
                    if rule['whole_reference_control']:
                        ref = frozen['whole_physical_references'][label]
                        folder, name = Path(ref['folder']), f'chunk_{query:02d}/states.pt'
                        assert sha(folder / name) == ref['files_current_sha256'][name]
                        assert image.read_bytes() == (folder / f'input_{query:02d}.png').read_bytes()
                        runtime.require_exact(record, torch.load(folder / name, map_location='cpu', weights_only=True, mmap=True), 'ENTIRE accepted milk feedback/' + label + '/' + str(query))
                        references.append(dict(case=label, query=query, whole_record_exact=True, actual_feedback_PNG_bytes_exact=True))
                    metadata.update(trial=label, query=query, shift_cm=12, scene_prompt=SCENE_PROMPT,
                        policy_prompt=prompt, policy_goal=rule['goal'], task_index=7,
                        image_path=str(image), input_png_sha256=sha(image), saved_q0_prediction=query == 0,
                        current_execution_model_calls=0 if query == 0 else 30,
                        q0_vision_noise_source_seed=rule['q0_vision_noise_source_seed'],
                        q0_action_noise_source_seed=195, q0_runtime_rng_seed=195,
                        saved_q0_attention_arm=rule['arm'], saved_q0_window=rule['selected_window'],
                        actual_query_seeds=rule['actual_query_seeds'], seed_rule='195 + query_index after cached q0',
                        component_intervention_in_this_execution=False, attention_mask_intervention_in_this_execution=False,
                        paired_noise_checked_exact=paired, source_producer_folder=source['folder'] if query == 0 else None,
                        source_q0_states_sha256=source['files_sha256']['states.pt'] if query == 0 else None)
                    write(chunk / 'metadata.json', metadata)
                    assert read(chunk / 'normalized_actions.json') == record['actions'].tolist()
                    served.append((label, query))
                    write(destination / 'progress.json', dict(queries=len(served), cached_q0=cached,
                        native_predictions=fresh, fresh_model_forwards=30 * fresh, actual_later_whole_noise_pairs=pairs,
                        served_queries=served, elapsed_s=time.perf_counter() - started))
                    print('[LANGUAGE-EXECUTION] ' + json.dumps(dict(requests=len(served), case=label, query=query)), flush=True)
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
        while not failed and len(served) < 144:
            http.handle_request()
            assert not (out / 'closed-loop/failed.json').exists() and not (out / 'wrapper_failed.json').exists()
        assert not failed and len(served) == 144 and cached == 18 and fresh == 126 and pairs == 119
        expected = [(label, q) for label in frozen['trial_order'] if frozen['trials'][label]['whole_reference_control'] for q in range(8)]
        assert [(r['case'], r['query']) for r in references] == expected and len(references) == 48
        assert hooks(model) == initial_hooks
        write(destination / 'whole-reference-query-controls.json', dict(state='exact', queries=references), exclusive=True)
        write(destination / 'complete.json', dict(state='complete', queries=144, saved_q0_predictions=18,
            native_predictions=126, fresh_model_forwards=3780, actual_later_whole_noise_pairs=119,
            six_milk_controls_all8_entire_records_and_PNGs_exact=True, final_model_hooks_none=True,
            files_sha256={name: sha(destination / name) for name in ('saved-q0-controls.json', 'whole-reference-query-controls.json')}, scope=SCOPE), exclusive=True)
    finally:
        if http is not None:
            http.server_close()


def execute_trials(output, selected_trials, collector, rollout, simulator, frozen):
    # Physical loop copied from the frozen simulator. Only policy routing,
    # provenance, q0 pairing and a read-only controller16 capture are adapted.
    np = simulator.np
    URL, STATS = frozen['url'], Path(frozen['threshold_contract']['frozen_files']['normalizer_stats']['path'])
    PROMPT, SHIFTS, SEEDS = SCENE_PROMPT, simulator.SHIFTS, (195,)
    TRIALS = {label: ('x12', 195) for label in frozen['trial_order']}
    digest, write_json = sha, write
    load_capture, make_env = simulator.load_capture, simulator.make_env
    reset_controller, capture = simulator.reset_controller, simulator.capture
    initial_contact_checks, measure = simulator.initial_contact_checks, simulator.measure
    CRITERION = simulator.CRITERION
    import imageio.v2 as imageio
    from PIL import Image
    import requests
    from cosmos_framework.simulation.libero.closed_loop_eval import _framewise_action_to_delta, _remap_gripper

    inputs = output / 'inputs'
    complete = json.loads((inputs / 'complete.json').read_text())
    assert complete['state'] == 'matched_inputs_verified' and complete['additional_warmup_steps'] == 0
    assert not (inputs / 'failed.json').exists()
    assert complete['metadata_sha256'] == digest(inputs / 'metadata.json')
    metadata = json.loads((inputs / 'metadata.json').read_text())
    assert metadata['script_sha256'] == digest(Path(simulator.__file__)), 'Collection script changed'
    assert metadata['collector_sha256'] == digest(Path(collector.__file__))
    assert metadata['controller_helper_sha256'] == digest(Path(rollout.__file__))
    assert metadata['prompt'] == PROMPT and metadata['shifts_cm'] == SHIFTS
    references, controllers = {}, {}
    for scene in SHIFTS:
        folder = inputs / scene
        references[scene] = load_capture(folder, metadata, collector)
        assert metadata['scenes'][scene]['arrays_sha256'] == digest(folder / 'arrays.json')
        assert metadata['scenes'][scene]['controller_sha256'] == digest(folder / 'controller.npz')
        with np.load(folder / 'controller.npz', allow_pickle=False) as archive:
            controllers[scene] = {key: archive[key].copy() for key in archive.files}
    health = requests.get(URL + '/health', timeout=10)
    health.raise_for_status()
    assert health.text == 'ready'
    stats = json.loads(STATS.read_text())['global_raw']
    lo, hi = np.asarray(stats['q01']), np.asarray(stats['q99'])
    assert lo.shape == hi.shape == (10,) and np.all(hi - lo > 1e-6)
    scale, offset = np.maximum(hi - lo, 1e-8) / 2, (hi + lo) / 2
    destination = output / 'closed-loop'
    provenance = dict(script_sha256=digest(Path(__file__)), inputs_metadata_sha256=digest(inputs / 'metadata.json'),
        stats_sha256=digest(STATS), scene_prompt=PROMPT, policy_prompts={label: frozen['trials'][label]['policy_prompt'] for label in frozen['trial_order']}, frozen_simulator_sha256=digest(Path(simulator.__file__)), pixels='official', schedule='official',
        use_system_prompt=False, queries_per_trial=8, actions_per_query=16, steps_per_trial=128,
        seeds=list(SEEDS), shifts_cm=SHIFTS, additional_warmup_steps=0, selection_criterion=CRITERION,
        action_conversion='Quantile affine inverse; _framewise_action_to_delta(6d); _remap_gripper(zero_one); clip [-1,1].',
        scope=SCOPE)
    if destination.exists():
        assert not (destination / 'failed.json').exists(), 'Prior failed batch requires review; no automatic retry'
        assert json.loads((destination / 'provenance.json').read_text()) == provenance
    else:
        destination.mkdir(exist_ok=False)
        write_json(destination / 'provenance.json', provenance)
    for trial in selected_trials:
        assert not (destination / trial).exists(), f'Refusing existing trial: {trial}'
    results = []
    if (destination / 'summary.json').exists():
        results = json.loads((destination / 'summary.json').read_text())['cases']
    env, _ = make_env()
    trial, schedule_reference, seen_noise = None, None, set()
    try:
        for trial in selected_trials:
            scene, seed = TRIALS[trial]
            policy_prompt = frozen['trials'][trial]['policy_prompt']
            run = destination / trial
            run.mkdir(exist_ok=False)
            env.reset()
            obs = collector.restore(env, references[scene])
            controller, controller_checks = reset_controller(env.env, rollout, controllers[scene])
            np.savez_compressed(run / 'initial_controller.npz', **controller)
            initial = capture(env, obs, collector)
            checks = collector.compare(references[scene], initial)
            write_json(run / 'initial_checks.json', checks)
            write_json(run / 'initial_controller_checks.json', controller_checks)
            assert all(row['exact'] for row in checks), ('Initial restoration', trial)
            write_json(run / 'initial_contact_checks.json', initial_contact_checks(env.env))
            records = [measure(env, obs, 0, rollout)]
            states = [np.array(env.get_sim_state(), copy=True)]
            actions, executed, normalized, denormalized, contracts = [], [], [], [], []
            with imageio.get_writer(str(run / 'actual.mp4'), fps=20, codec='libx264', macro_block_size=1) as writer:
                writer.append_data(initial['images']['input'])
                for query in range(8):
                    pixels = np.concatenate([obs[name + '_image'][::-1, ::-1] for name in collector.CAMERAS], axis=1).copy()
                    image_path = run / f'input_{query:02d}.png'
                    Image.fromarray(pixels).save(image_path)
                    if query == 0:
                        assert image_path.read_bytes() == (inputs / scene / 'input.png').read_bytes(), 'Initial PNG bytes changed'
                    request = dict(trial=trial, query=query, seed=seed + query, prompt=policy_prompt)
                    reply = requests.post(URL + '/predict', json=request, timeout=600)
                    reply.raise_for_status()
                    value = reply.json()
                    meta = value['metadata']
                    chunk = run / f'chunk_{query:02d}'
                    saved_meta = json.loads((chunk / 'metadata.json').read_text())
                    assert meta == saved_meta
                    for key, expected in dict(trial=trial, query=query, shift_cm=SHIFTS[scene], seed=seed + query,
                        prompt=policy_prompt, pixels='official', schedule='official', use_system_prompt=False,
                        domain_name='libero', domain_id=5, raw_action_dim=10, chunk_size=16,
                        num_inference_steps=30, model_calls=30, prepare_calls=1).items():
                        assert meta[key] == expected, (trial, query, key, meta.get(key), expected)
                    assert meta['input_png_sha256'] == digest(image_path)
                    schedule = meta['timesteps']
                    assert len(schedule) == 30 and all(type(item) is int for item in schedule)
                    assert schedule[10] == 666 and schedule[20] == 333
                    if schedule_reference is None:
                        schedule_reference = schedule
                    assert schedule == schedule_reference
                    assert type(value['paired_noise_checked_exact']) is bool
                    if query > 0 and (seed, query) in seen_noise:
                        assert value['paired_noise_checked_exact'] is True
                    seen_noise.add((seed, query))
                    norm = np.asarray(value['actions'])
                    saved = np.asarray(json.loads((chunk / 'normalized_actions.json').read_text()))
                    assert norm.shape == (16, 10) and np.isfinite(norm).all() and np.array_equal(norm, saved)
                    assert (chunk / 'states.pt').is_file(), 'Model arrays missing'
                    contracts.append(dict(request=request, server_metadata=meta,
                        paired_noise_checked_exact=value['paired_noise_checked_exact'], response_saved_actions_exact=True))
                    raw = norm * scale + offset
                    native = np.asarray([_remap_gripper(_framewise_action_to_delta(item, '6d').tolist(), 'zero_one') for item in raw])
                    assert native.shape == (16, 7) and np.isfinite(native).all()
                    for index, command in enumerate(native):
                        sent = np.clip(command, -1, 1)
                        obs, _, done, _ = env.step(sent.tolist())
                        actions.append(command.tolist())
                        executed.append(sent.tolist())
                        normalized.append(norm[index].tolist())
                        denormalized.append(raw[index].tolist())
                        record = measure(env, obs, len(actions), rollout)
                        if len(actions) == 16:
                            np.savez_compressed(run / 'controller-step16.npz', **rollout.controller_state(env.env))
                        record['environment_done'] = bool(done)
                        records.append(record)
                        states.append(np.array(env.get_sim_state(), copy=True))
                        writer.append_data(np.concatenate([obs[name + '_image'][::-1, ::-1] for name in collector.CAMERAS], axis=1).copy())
                    for name, content in (('trajectory.json', records), ('actions.json', actions),
                        ('executed_actions.json', executed), ('normalized_actions.json', normalized),
                        ('denormalized_actions.json', denormalized), ('query_contract.json', contracts)):
                        write_json(run / name, content)
                    np.save(run / 'sim_states.npy', np.stack(states))
                    status = dict(trial=trial, completed_queries=query + 1, completed_steps=len(actions),
                                  completed_cases=len(results), requested_trials=selected_trials)
                    write_json(destination / 'status.json', status)
                    write_json(output / 'status.json', dict(stage='closed_loop', **status))
                    print('[MILK-SHIFT]', trial, query + 1, records[-1]['grasped'], flush=True)
            assert len(records) == len(states) == 129
            assert len(actions) == len(executed) == len(normalized) == len(denormalized) == 128
            assert len(contracts) == len(list(run.glob('input_*.png'))) == 8
            summary = simulator.summarize_trial(trial, scene, seed, records, rollout)
            rule = frozen['trials'][trial]
            summary.update(prompt=policy_prompt, policy_prompt=policy_prompt, scene_prompt=SCENE_PROMPT, policy_goal=rule['goal'], requested_target=frozen['targets'][rule['goal']], strict_requested_target_selected=summary['selected_objects'] == [frozen['targets'][rule['goal']]], saved_q0_attention_arm=rule['arm'], q0_vision_noise_source_seed=rule['q0_vision_noise_source_seed'], q0_action_noise_source_seed=195, actual_query_seeds=rule['actual_query_seeds'], milk_in_basket_is_scene_predicate_not_changed_policy_success=True)
            write_json(run / 'summary.json', summary)
            results.append(summary)
            batch_summary = dict(cases=results, scope=SCOPE, selection_criterion=CRITERION,
                                 completed_trials=[item['trial'] for item in results], default_trials=list(TRIALS))
            write_json(destination / 'summary.json', batch_summary)
            write_json(output / 'summary.json', batch_summary)
    except Exception as exc:
        failure = dict(trial=trial, type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc(),
                       completed_trials=[item['trial'] for item in results])
        write_json(destination / 'failed.json', failure)
        write_json(output / 'simulator_failed.json', failure)
        raise
    finally:
        env.close()
    completed = dict(state='complete' if len(results) == len(TRIALS) else 'selected_trials_complete',
        cases=len(results), completed_trials=[item['trial'] for item in results], steps_per_case=128,
        queries_per_case=8, state_records_per_case=129, additional_warmup_steps=0)
    write_json(destination / 'complete.json', completed)
    write_json(output / 'simulator_complete.json', completed)
    print('[MILK-SHIFT] COMPLETE', json.dumps(completed), flush=True)

def simulate(baseline):
    frozen, _, simulator, helper, cross, out = prepared(baseline)
    assert not (out / 'closed-loop').exists(), 'Do not repeat a partial physical batch'
    os.environ.setdefault('MUJOCO_GL', 'egl')
    sys.path.insert(0, '/home/current/work/openpi-demo/cosmos-policy')
    sys.path.insert(0, str(ROOT / 'cosmos-framework'))
    from cosmos_framework.simulation.libero.closed_loop_eval import _framewise_action_to_delta, _remap_gripper
    np = simulator.np
    collector = simulator.load_file('language_execution_collector', ROOT / 'work/collect_cosmos_goal_pair.py')
    rollout = simulator.load_file('language_execution_controller', ROOT / 'work/rollout_cosmos_goal_pair.py')
    stats = read(Path(frozen['threshold_contract']['frozen_files']['normalizer_stats']['path']))['global_raw']
    lo, hi = np.asarray(stats['q01']), np.asarray(stats['q99'])
    scale, offset = np.maximum(hi - lo, 1e-8) / 2, (hi + lo) / 2
    initial = np.load(out / 'inputs/x12/state.npy', allow_pickle=False)
    whole_controls, comparisons, case_files = [], [], {}
    for label in frozen['trial_order']:
        execute_trials(out, [label], collector, rollout, simulator, frozen)
        folder, rule = out / 'closed-loop' / label, frozen['trials'][label]
        summary, trajectory = read(folder / 'summary.json'), read(folder / 'trajectory.json')
        state = np.load(folder / 'sim_states.npy', allow_pickle=False)
        assert state.shape == (129, initial.size) and state.dtype == initial.dtype and np.isfinite(state).all()
        assert rollout.array_exact(state[0], initial)
        assert all(row['exact'] for row in read(folder / 'initial_checks.json'))
        assert all(value is True for value in read(folder / 'initial_controller_checks.json').values())
        assert read(folder / 'initial_contact_checks.json') == read(out / 'inputs/x12/initial_contact_checks.json')
        with np.load(folder / 'initial_controller.npz', allow_pickle=False) as current:
            with np.load(out / 'inputs/x12/controller.npz', allow_pickle=False) as original:
                assert set(current.files) == set(original.files)
                assert all(rollout.array_exact(current[key], original[key]) for key in current.files)
        norm = np.asarray(read(folder / 'normalized_actions.json'))
        assert norm.shape == (128, 10) and norm.dtype == np.float64 and np.isfinite(norm).all()
        assert rollout.array_exact(norm, np.concatenate([np.asarray(read(folder / f'chunk_{q:02d}/normalized_actions.json')) for q in range(8)]))
        assert rollout.array_exact(norm[:16], np.asarray(frozen['q0_sources'][label]['normalized_actions']))
        raw = norm * scale + offset
        native = np.asarray([_remap_gripper(_framewise_action_to_delta(item, '6d').tolist(), 'zero_one') for item in raw])
        assert native.shape == (128, 7) and native.dtype == np.float64 and np.isfinite(native).all()
        for name, value in (('denormalized_actions.json', raw), ('actions.json', native), ('executed_actions.json', np.clip(native, -1, 1))):
            assert rollout.array_exact(np.asarray(read(folder / name)), value), ('all128 official conversions', label, name)
        strict = helper.strict_windows(trajectory, summary, np)
        write(folder / 'strict-windows.json', strict, exclusive=True)
        if rule['whole_reference_control']:
            reference = frozen['whole_physical_references'][label]
            for name, expected in reference['files_current_sha256'].items():
                assert sha(Path(reference['folder']) / name) == expected
            control = cross.native_control(out, label, reference, np)
            for name in ('initial_controller.npz', 'controller-step16.npz'):
                with np.load(folder / name, allow_pickle=False) as current:
                    with np.load(Path(reference['folder']) / name, allow_pickle=False) as old:
                        assert set(current.files) == set(old.files)
                        assert all(rollout.array_exact(current[key], old[key]) for key in current.files)
            whole_controls.append(dict(**control, reference_case=reference['reference_case'], typed_controller0_and16_exact=True))
        milk0 = np.asarray(trajectory[0]['objects']['milk_1'])
        cheese0 = np.asarray(trajectory[0]['objects']['cream_cheese_1'])
        direction = milk0 - cheese0
        length = float(np.linalg.norm(direction))
        assert length > 1e-6
        direction = direction / length
        eef0 = np.asarray(trajectory[0]['eef_xyz'])
        eef16 = np.asarray(trajectory[16]['eef_xyz'])
        first_contacts = {name: next((i for i, row in enumerate(trajectory) if row['finger_contacts'][name]['left'] or row['finger_contacts'][name]['right']), None) for name in trajectory[0]['objects']}
        comparisons.append(dict(case=label, goal=rule['goal'], policy_prompt=rule['policy_prompt'], arm=rule['arm'],
            vision_noise_source_seed=rule['q0_vision_noise_source_seed'], selected_objects=summary['selected_objects'],
            strict_requested_target_selected=summary['strict_requested_target_selected'],
            first_selection_window_start=summary['first_selection_step'], first_selected_object=summary['first_selected_object'],
            per_object=summary['per_object'], first_any_finger_contacts=first_contacts,
            eef_q0_displacement_m=(eef16 - eef0).tolist(), initial_cheese_to_milk_unit_direction=direction.tolist(),
            q0_eef_projection_toward_milk_from_cheese_m=float(np.dot(eef16 - eef0, direction)),
            initial_object_separation_m=length, final_milk_in_basket_scene_predicate=bool(trajectory[-1]['milk_in_basket']),
            strict_windows_sha256=sha(folder / 'strict-windows.json')))
        names = ('sim_states.npy', 'actual.mp4', 'initial_controller.npz', 'controller-step16.npz', 'summary.json',
            'strict-windows.json', 'query_contract.json', 'initial_checks.json', 'initial_controller_checks.json', 'initial_contact_checks.json',
            'trajectory.json', 'actions.json', 'executed_actions.json', 'normalized_actions.json', 'denormalized_actions.json',
            *[f'input_{q:02d}.png' for q in range(8)],
            *[f'chunk_{q:02d}/{name}' for q in range(8) for name in ('states.pt', 'metadata.json', 'normalized_actions.json')])
        case_files[label] = {name: sha(folder / name) for name in names}
    done, served = read(out / 'closed-loop/complete.json'), read(out / 'server/complete.json')
    assert done['state'] == served['state'] == 'complete' and done['cases'] == 18
    assert done['completed_trials'] == frozen['trial_order'] and len(whole_controls) == 6
    assert served['queries'] == 144 and served['saved_q0_predictions'] == 18 and served['native_predictions'] == 126
    assert served['fresh_model_forwards'] == 3780 and served['actual_later_whole_noise_pairs'] == 119
    assert served['six_milk_controls_all8_entire_records_and_PNGs_exact'] is served['final_model_hooks_none'] is True
    for name, expected in served['files_sha256'].items():
        assert sha(out / 'server' / name) == expected
    by_case = {row['case']: row for row in comparisons}
    descriptive = []
    for v in (195, 198):
        labels = {goal: next(label for label, rule in frozen['trials'].items() if rule['goal'] == goal and rule['arm'] == 'full_hardmask' and rule['q0_vision_noise_source_seed'] == v) for goal in frozen['prompts']}
        pair = [by_case[labels[goal]] for goal in ('milk', 'cheese')]
        descriptive.append(dict(vision_noise_source_seed=v, cases=labels,
            both_primary_instructions_strict_only_milk=all(row['selected_objects'] == ['milk_1'] for row in pair),
            both_primary_instructions_match_their_targets=all(row['strict_requested_target_selected'] for row in pair),
            canonical_cream_cheese_matches_target=by_case[labels['cream_cheese']]['strict_requested_target_selected'],
            status='behavioral_description_not_the_internal_primary_test'))
    write(out / 'comparisons.json', dict(state='all18_complete', cases=comparisons, cut_language_pairs=descriptive,
        all18_executed_without_score_selection=True, internal_primary_requires_actual_array_analysis=True, scope=SCOPE), exclusive=True)
    write(out / 'controls.json', dict(six_whole_milk_controls=whole_controls, actual_case_files_sha256=case_files,
        all18_own_cached_q0_actions_and_all128_official_conversions_exact=True, all18_strict_windows_recomputed=True), exclusive=True)
    names = ('controls.json', 'comparisons.json', 'provenance.json', 'prepared.json', 'server/complete.json', 'closed-loop/complete.json', 'summary.json')
    write(out / 'complete.json', dict(state='complete', script_sha256=sha(Path(__file__)), physical_trials=18,
        trial_order=frozen['trial_order'], saved_q0_predictions=18, native_predictions=126,
        fresh_model_forwards=3780, producer_q0_forwards=540, combined_two_stage_model_forwards=4320,
        actual_later_whole_noise_pairs=119, actual_physical_actions=2304,
        six_entire129_state_5JSON_8PNG_and_all8_modelrecords_controls_exact=True,
        all18_cached_q0_action_and_strict_window_gates_passed=True, all18_executed_without_score_selection=True,
        internal_primary_evaluated=False, files_sha256={name: sha(out / name) for name in names}, scope=SCOPE), exclusive=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('prepare', 'server', 'simulate'), required=True)
    args = parser.parse_args()
    baseline, out = args.output.resolve(), args.output.resolve() / STAGE
    if args.mode == 'prepare':
        frozen, _, _, _, _ = contract(baseline)
        assert shutil.disk_usage(baseline).free >= 2 * 1024 ** 3, 'Need 2GiB reserved for physical records'
        out.mkdir(exist_ok=False)
        (out / 'inputs').symlink_to(frozen['inputs_symlink_target'], target_is_directory=True)
        write(out / 'provenance.json', frozen, exclusive=True)
        write(out / 'prepared.json', dict(state='prepared', provenance_sha256=sha(out / 'provenance.json')), exclusive=True)
        print('[LANGUAGE-EXECUTION] prepared ' + str(out), flush=True)
        return
    try:
        server(baseline) if args.mode == 'server' else simulate(baseline)
    except Exception as exc:
        if out.is_dir() and not (out / 'wrapper_failed.json').exists():
            write(out / 'wrapper_failed.json', dict(mode=args.mode, error=str(exc), traceback=traceback.format_exc()), exclusive=True)
        raise


if __name__ == '__main__':
    main()
