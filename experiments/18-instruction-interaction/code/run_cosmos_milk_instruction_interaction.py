"""Execute all eight L35 instruction-suffix/MLP cases with native feedback.

No cached q0 is recomputed. The two live-self cases must reproduce their
accepted X15/V198 natural trajectories, all eight model records and typed
controllers. Every later query keeps the recipient instruction. No q0 score
selects cases, and no completed or failed physical output is retried.
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
PROBE_SHA = '745c4d8864bc94751791cf5aa81cb74afc0a678ca95dfad998b0381431c61429'
MATCHED_RUNNER_SHA = '6b5ea13b8ce888aecac5cdb42b1a23ae9fec81972edf51f241f1e4429008217c'
PORT = 8940
STAGE = 'future-instruction-interaction-execution'
SCENE_PROMPT = 'pick up the milk and place it in the basket'
SCENES = dict(x15=15)
PROMPTS = dict(milk_box='pick up the milk box and place it in the basket',
               cream_cheese='pick up the cream cheese and place it in the basket')
TARGETS = dict(milk_box='milk_1', cream_cheese='cream_cheese_1')
ARMS = ('live_self', 'suffix_joint', 'suffix_joint_native_mlp', 'matched_random')
DIRECTIONS = (('milk_box', 'cream_cheese'), ('cream_cheese', 'milk_box'))
PLAN = tuple((f'x15_{recipient}_from_{donor}_V198_A195_{arm}', recipient, donor, arm)
             for arm in ARMS for recipient, donor in DIRECTIONS)
SCOPE = ('Eight preregistered X15/V198 q0 interventions, each followed by seven '
    'native recipient-instruction queries with seed195+q. Only two natural '
    'matched-name references are eligible. Suffix transfer is instruction-context '
    'transfer, not a decoded identity, named head or unique MLP mediator. The '
    'same-live MLP comparison is controlled locally, while later trajectories '
    'are dynamic. Actual BF16 random strength matches pre-W only. Strict grasp, '
    'approach displacement and original milk-in-basket predicate remain separate. '
    'No original milk-instruction 15cm repair or half-denoising withdrawal test.')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


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
    """Read-only seals and fixed plan before output/model construction."""
    assert len(PROBE_SHA) == 64 and all(c in '0123456789abcdef' for c in PROBE_SHA), 'Producer is not frozen'
    path = ROOT / 'work/probe_cosmos_milk_instruction_interaction.py'
    assert sha(path) == PROBE_SHA
    probe = load('instruction_execution_producer', path)
    assert probe.STAGE == 'instruction-interaction' and tuple(probe.PLAN) == PLAN
    assert probe.PROMPTS == PROMPTS and probe.TARGETS == TARGETS
    context = probe.source_contract(baseline)
    legacy = context['modules']['matched_adapter']
    assert sha(Path(legacy.__file__)) == MATCHED_RUNNER_SHA
    simulator, helper, cross = (context['modules'][key] for key in ('simulator', 'execution_helper', 'cross'))
    accepted, producer = context['matched_frozen'], baseline / probe.STAGE
    assert producer.is_dir() and not (producer / 'failed.json').exists()
    documents = ('results', 'protocol', 'provenance', 'sources', 'primary-prediction')
    docs = {name: read(producer / (name + '.json')) for name in ('complete', *documents)}
    done, result, provenance, protocol = (docs[k] for k in ('complete', 'results', 'provenance', 'protocol'))
    assert done['state'] == result['state'] == 'complete'
    assert done['script_sha256'] == provenance['script_sha256'] == PROBE_SHA
    assert done['fresh_q0_predictions'] == 8 and done['fresh_model_forwards'] == 240
    assert done['official_dispatch_counts'] == result['official_dispatch_counts'] == probe.expected_counts()
    assert done['total_official_dispatch_calls'] == 17520
    for key in ('two_live_self_entire_accepted_recipient_q0_records_exact',
                'actual_equal_length_two_target_ID_and_suffix_position_gates_passed',
                'all240_same_live_native_W_postnorm_MLP_baselines_computed',
                'all240_original_QKV_and_CPU_CUDA_global_RNG_byte_gates_passed',
                'all240_actual_BF16_residual_and_selected_block_reconstruction_gates_passed',
                'all60_random_actual_BF16_strength_gates_passed',
                'all_original_symbols_restored', 'all_owned_hooks_removed'):
        assert done[key] is True
    for name in documents:
        assert done[name.replace('-', '_') + '_sha256'] == sha(producer / (name + '.json'))
    assert provenance['sources_sha256'] == done['sources_sha256']
    assert provenance['protocol_sha256'] == done['protocol_sha256']
    assert protocol['trial_order'] == result['preregistered_physics_cases'] == [row[0] for row in PLAN]
    assert protocol['primary_hypothesis'] == probe.PRIMARY
    assert docs['primary-prediction'] == dict(status='not_evaluated_q0_only', **probe.PRIMARY, scope=probe.SCOPE)
    assert result['primary_physical_prediction_evaluated'] is done['primary_physical_prediction_evaluated'] is False
    assert docs['sources']['matched_references'] == context['matched_references']
    assert docs['sources']['frozen_contract'] == context['frozen_contract']
    scene = accepted['scenes']['x15']
    assert scene['shift_cm'] == 15 and Path(scene['folder']).name == 'x15'
    for name, expected in scene['files_sha256'].items():
        assert sha(Path(scene['folder']) / name) == expected
    rows, q0, trials, references = result['cases'], {}, {}, {}
    assert [(r['case'], r['recipient_goal'], r['donor_goal'], r['arm']) for r in rows] == list(PLAN)
    for row in rows:
        label, recipient, donor, arm = (row[k] for k in ('case', 'recipient_goal', 'donor_goal', 'arm'))
        assert row['scene'] == 'x15' and row['goal'] == recipient
        assert row['vision_noise_source_seed'] == 198 and row['action_noise_source_seed'] == row['runtime_rng_seed'] == 195
        assert row['control_reference_case'] == f'x15_{recipient}_V198_A195_native'
        folder = producer / label
        assert set(row['files_sha256']) == set(probe.FILES)
        for name, expected in row['files_sha256'].items():
            assert sha(folder / name) == expected
        manifest = read(folder / 'site-captures.json')
        assert manifest['state'] == 'complete' and manifest['captures'] == 30
        assert manifest['files'] == row['site_files'] == row['boundary_files']
        assert len(row['site_files']) == 30
        for step, item in enumerate(row['site_files']):
            assert item['step'] == step and item['layer_zero_based'] == 35
            assert item['file'] == f'site-t{step:02d}-L35.pt' and item['action_boundary_ids'] == [36]
            assert sha(folder / item['file']) == item['sha256']
        meta = read(folder / 'metadata.json')
        assert meta['prompt'] == PROMPTS[recipient] and meta['seed'] == 195 and meta['task_index'] == 7
        assert meta['input_png_sha256'] == sha(Path(scene['image']))
        norm = simulator.np.asarray(read(folder / 'normalized_actions.json'))
        assert norm.shape == (16, 10) and simulator.np.isfinite(norm).all()
        q0[label] = dict(folder=str(folder), files_sha256=row['files_sha256'], site_files=row['site_files'],
                        normalized_actions=norm.tolist(), control_reference_case=row['control_reference_case'])
        trials[label] = dict(scene='x15', shift_cm=15, goal=recipient, recipient_goal=recipient, donor_goal=donor,
            policy_prompt=PROMPTS[recipient], arm=arm, selected_window=probe.WINDOW,
            q0_vision_noise_source_seed=198, actual_query_seeds=list(range(195, 203)),
            whole_reference_control=arm == 'live_self')
        if arm == 'live_self':
            source = context['matched_references'][recipient]
            assert source['strict_selected_objects'] == [TARGETS[recipient]]
            ref_folder = baseline / 'future-matched-names-execution/closed-loop' / source['case']
            files = source['physical_case_files_sha256']
            for name, expected in files.items():
                assert sha(ref_folder / name) == expected
            references[label] = dict(folder=str(ref_folder), reference_case=source['case'], files_current_sha256=files)
    assert len(rows) == 8 and len(references) == 2
    frozen = dict(script_sha256=sha(Path(__file__)), producer_script_sha256=PROBE_SHA,
        producer_stage=probe.STAGE, producer_documents_sha256={name: sha(producer / (name + '.json')) for name in docs},
        threshold_contract=accepted['threshold_contract'], actual_transformer_sha256=accepted['actual_transformer_sha256'],
        actual_pipeline_source_sha256=accepted['actual_pipeline_source_sha256'], metrics_path=accepted['metrics_path'],
        components_path=accepted['components_path'], producer_source_cases=accepted['producer_source_cases'],
        language_producer_path=accepted['language_producer_path'], language_producer_sha256=accepted['language_producer_sha256'],
        matched_adapter_sha256=MATCHED_RUNNER_SHA, execution_body_sha256=MATCHED_RUNNER_SHA,
        scenes=dict(x15=scene), scene_prompt=SCENE_PROMPT, prompts=PROMPTS, targets=TARGETS,
        q0_sources=q0, trials=trials, trial_order=[row[0] for row in PLAN], whole_physical_references=references,
        natural_sources=context['matched_references'], producer_protocol=protocol, url=f'http://127.0.0.1:{PORT}',
        saved_q0_predictions=8, requests=64, native_predictions=56, fresh_model_forwards=1680,
        producer_q0_forwards=240, combined_two_stage_model_forwards=1920,
        later_whole_noise_pairs=49, physical_actions=1024, primary_prediction=probe.PRIMARY,
        all8_execute_without_score_selection=True, scope=SCOPE)
    # These are isolated imported modules; no accepted source or artifact is edited.
    legacy.SCENES, legacy.SCOPE, simulator.SHIFTS, simulator.SCOPE = SCENES, SCOPE, SCENES, SCOPE
    return frozen, probe, legacy, simulator, helper, cross


def prepared(baseline):
    frozen, probe, legacy, simulator, helper, cross = contract(baseline)
    out = baseline / STAGE
    assert out.is_dir() and not (out / 'complete.json').exists(), 'Completed execution is immutable'
    for name in ('wrapper_failed.json', 'simulator_failed.json', 'server/failed.json', 'closed-loop/failed.json'):
        assert not (out / name).exists()
    assert read(out / 'provenance.json') == frozen
    assert read(out / 'prepared.json') == dict(state='prepared', provenance_sha256=sha(out / 'provenance.json'))
    assert read(out / 'primary-preregistered.json') == dict(state='preregistered', **probe.PRIMARY)
    assert not (out / 'inputs/metadata.json').exists()
    assert (out / 'inputs/x15').is_symlink() and (out / 'inputs/x15').resolve() == Path(frozen['scenes']['x15']['folder']).resolve()
    return frozen, probe, legacy, simulator, helper, cross, out


def verify_saved(record, audit, folder, reference, runtime, components, language, probe):
    """Read actual saved action rows; full arrays have live hashes, not raw exports."""
    torch = runtime.torch
    runtime.require_exact(record['pure_noise'], reference['pure_noise'], 'entire natural recipient two-draw pair')
    for key in ('prepared_latents_and_masks', 'model_input', 'timesteps', 'sigmas'):
        runtime.require_exact(record[key], reference[key], 'natural recipient initial preparation/' + key)
    runtime.require_exact(record['model_input'], audit['initial_model_kwargs'], 'inner/outer actual initial kwargs')
    assert torch.count_nonzero(record['action_states'][..., 10:]) == 0
    report, lc = read(folder / 'observer-report.json'), torch.load(folder / 'language-contract.pt', map_location='cpu', weights_only=True, mmap=True)
    arm = lc['arm']
    assert report['state'] == 'complete' and report['counts'] == probe.expected_counts(arm)
    for key in ('all37_by30_finite_live_metadata_recorded', 'all_original_QKV_and_global_RNG_bytes_preserved',
                'all_nonaction_GEN_rows_native_through_attention_W_norm_MLP_block_byte_exact',
                'same_live_full266_native_W_residual_postnorm_MLP_baseline_computed',
                'all30_same_input_BF16_residual_and_actual_block_bytes_reconstructed',
                'all_owned_hooks_removed', 'original_dispatch_symbol_restored'):
        assert report[key] is True
    runtime.require_exact(lc['first_model_kwargs'], record['model_input'], 'actual recipient language contract')
    assert lc['language']['und_len'] == 122 and lc['language']['target_positions'] == [48, 49]
    clean = language.tensor_meta(record['model_input']['vision_tokens'][0][0, :, 0], runtime)
    for step, entry in enumerate(report['site_files']):
        data = torch.load(folder / entry['file'], map_location='cpu', weights_only=True, mmap=True)
        assert data['step'] == step and data['layer_zero_based'] == 35 and data['case_arm'] == arm
        runtime.require_exact(data['actual_pack'], audit['model_steps'][step], 'all30 actual pack')
        runtime.require_exact(data['actual_pack']['layout'], components._layout(record['model_input']), 'actual saved layout')
        runtime.require_exact(data['actual_model_action_tokens'], record['action_states'][step, 0].to(torch.bfloat16), 'actual FP32 solver cast')
        runtime.require_exact(data['actual_model_action_output'], record['action_velocity'][step], 'actual model action output')
        runtime.require_exact(data['actual_model_action_timesteps'], data['actual_pack']['metadata']['action_timesteps'], 'actual model timestep')
        runtime.require_exact(data['solver_sigma_from_same_frozen_schedule'], record['sigmas'][step], 'actual sigma')
        assert data['actual_current_frame_sha256'] == clean['sha256']
        assert data['current_hidden'] is data['und_hidden'] is data['actual_model_kwargs'] is data['model_output'] is None
        assert len(data['hidden_boundary_metadata']) == 37 and len(data['site_events']) == 36
        for key in ('x', 'W0', 'W1', 'm0', 'm1'):
            value = data[key + '_explicit_action16']
            assert value.shape == (16, 4096) and value.dtype == torch.bfloat16 and bool(torch.isfinite(value).all())
        for key in ('Z0', 'Zdonor', 'Zselected'):
            value = data[key + '_explicit_action16']
            assert value.shape == (1, 16, 32, 128) and value.dtype == torch.bfloat16 and bool(torch.isfinite(value).all())
        for key, meta in data['actual_endpoint_metadata'].items():
            actual = language.tensor_meta(data[key], runtime)
            assert all(actual[field] == meta[field] for field in ('shape', 'dtype', 'sha256'))
        x, w0, w1, m0, m1 = (data[key + '_explicit_action16'] for key in ('x', 'W0', 'W1', 'm0', 'm1'))
        r0, r1 = x + w0, x + w1
        for key, value in (('r0', r0), ('r1', r1), ('r1_plus_m0', r1 + m0), ('r1_plus_m1', r1 + m1)):
            assert language.tensor_meta(value, runtime)['sha256'] == data[key + '_explicit_action16_metadata']['sha256']
        runtime.require_exact(data['action_hidden'], r1 + (m0 if arm == 'suffix_joint_native_mlp' else m1), 'actual ordered BF16 block endpoint')
        assert data['two_actual_BF16_addition_reconstruction_gates'] is True
        for rng in ('cpu', 'cuda'):
            runtime.require_exact(data['global_RNG_before'][rng], data['global_RNG_after'][rng], 'auxiliary global RNG unchanged')
            runtime.require_exact(data['global_RNG_before'][rng], data['global_RNG_after_actual_block'][rng], 'main block global RNG unchanged')
        if arm == 'matched_random':
            random = data['random']
            assert random['seed'] == 104729 + 1000 * DIRECTIONS.index((data['recipient'], data['donor'])) + step
            assert random['strength_gate_pass'] is True and random['relative_error'] <= probe.RANDOM_TOLERANCE == .02
            z0, zdonor, selected = (data[key + '_explicit_action16'] for key in ('Z0', 'Zdonor', 'Zselected'))
            target, actual = float((zdonor.double() - z0.double()).norm()), float((selected.double() - z0.double()).norm())
            assert target == random['target_l2_fp64'] and actual == random['realized_l2_fp64']
            error = 0. if target == actual == 0 else abs(actual / target - 1.)
            assert error == random['relative_error'] and error <= .02
            generator = torch.Generator(device='cpu').manual_seed(random['seed'])
            runtime.require_exact(generator.get_state(), random['generator_state_before'], 'private RNG initial state')
            permutation = torch.randperm(128, generator=generator)
            signs = (torch.randint(0, 2, (1, 16, 32, 128), generator=generator, dtype=torch.int64) * 2 - 1).float()
            runtime.require_exact(permutation, random['permutation'], 'actual private permutation')
            runtime.require_exact(signs.to(torch.int8), random['signs'], 'actual private signs')
            runtime.require_exact(generator.get_state(), random['generator_state_after'], 'private RNG final state')
            delta = (zdonor.float() - z0.float()).index_select(-1, permutation) * signs * random['alpha']
            runtime.require_exact(delta, data['random_actual_delta_FP32'], 'actual pointwise FP32 random witness')
            realized = (z0.float() + delta).to(torch.bfloat16)
            realized = torch.where(delta == 0, z0, realized)
            runtime.require_exact(realized, selected, 'actual pointwise BF16 random result')
    if arm == 'live_self':
        runtime.require_exact(record, reference, 'ENTIRE natural recipient q0 record')


def server(baseline):
    frozen, probe, _, _, _, _, out = prepared(baseline)
    destination = out / 'server'
    destination.mkdir(exist_ok=False)
    served, later_noise, references = [], {}, []
    cached = fresh = pairs = 0
    http, failed = None, False
    try:
        os.environ['HF_HUB_OFFLINE'] = '1'
        files = frozen['threshold_contract']['frozen_files']
        base = load('instruction_execution_runtime', Path(files['normal_runtime']['path']))
        for goal, prompt in PROMPTS.items():
            base.TASKS[goal] = (7, 'pick_up_the_milk_and_place_it_in_the_basket', prompt, TARGETS[goal])
        factory = load('instruction_execution_factory', Path(files['factory']['path']))
        factory.OUT = destination
        assert sha(factory.SCHEDULER) == files['scheduler']['sha256']
        runtime = base.NormalRuntime(factory)
        torch, model = runtime.torch, runtime.pipe.transformer
        assert not model.training and not model.is_cache_enabled and not torch.is_grad_enabled()
        assert sha(Path(inspect.getfile(type(model)))) == frozen['actual_transformer_sha256']
        assert sha(Path(inspect.getfile(type(runtime.pipe)))) == frozen['actual_pipeline_source_sha256']
        initial_hooks = hooks(model)
        assert all(not pre and not post for _, pre, post in initial_hooks)
        scan = load('instruction_execution_metrics', Path(frozen['metrics_path']))
        components = load('instruction_execution_layout', Path(frozen['components_path']))
        assert sha(Path(frozen['language_producer_path'])) == frozen['language_producer_sha256']
        language = load('instruction_execution_raw_hash_helpers', Path(frozen['language_producer_path']))
        q0, natural, controls = {}, {}, []
        for goal, source in frozen['natural_sources'].items():
            natural[goal] = torch.load(Path(source['folder']) / 'states.pt', map_location='cpu', weights_only=True, mmap=True)
        for label in frozen['trial_order']:
            source, rule = frozen['q0_sources'][label], frozen['trials'][label]
            folder = Path(source['folder'])
            record = torch.load(folder / 'states.pt', map_location='cpu', weights_only=True, mmap=True)
            scan.validate_record(record, torch)
            audit = torch.load(folder / 'noise-audit.pt', map_location='cpu', weights_only=True, mmap=True)
            verify_saved(record, audit, folder, natural[rule['goal']], runtime, components, language, probe)
            q0[label] = record
            controls.append(dict(case=label, entire_initial_preparation_and_two_draws_exact=True,
                actual_saved_rows_sigma_and_ordered_BF16_additions_verified=True,
                live_self_entire_record_exact=rule['whole_reference_control']))
        write(destination / 'saved-q0-controls.json', dict(state='exact', cases=controls, extra_q0_model_forwards=0), exclusive=True)

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                assert self.path == '/health'
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'ready')

            def do_POST(self):
                nonlocal failed, cached, fresh, pairs
                try:
                    assert self.path == '/predict'
                    request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                    assert set(request) == {'trial', 'query', 'seed', 'prompt'}
                    label, query = request['trial'], request['query']
                    assert type(query) is int and 0 <= query < 8 and (label, query) not in served
                    assert label == frozen['trial_order'][len(served) // 8] and query == len(served) % 8
                    rule, source = frozen['trials'][label], frozen['q0_sources'][label]
                    seed, prompt = rule['actual_query_seeds'][query], rule['policy_prompt']
                    assert request['seed'] == seed and request['prompt'] == prompt
                    image = out / 'closed-loop' / label / f'input_{query:02d}.png'
                    chunk = image.parent / f'chunk_{query:02d}'
                    if query == 0:
                        assert image.read_bytes() == (out / 'inputs/x15/input.png').read_bytes()
                        chunk.mkdir(exist_ok=False)
                        folder = Path(source['folder'])
                        for name in ('states.pt', 'normalized_actions.json'):
                            assert sha(folder / name) == source['files_sha256'][name]
                            os.link(folder / name, chunk / name)
                        record, metadata, paired = q0[label], read(folder / 'metadata.json'), False
                        cached += 1
                    else:
                        record, metadata = runtime.predict(rule['goal'], image, seed, chunk)
                        fresh += 1
                        paired = query in later_noise
                        if paired:
                            runtime.require_exact(record['pure_noise'], later_noise[query], 'entire actual later two-draw pair')
                            pairs += 1
                        else:
                            later_noise[query] = record['pure_noise']
                    scan.validate_record(record, torch)
                    assert metadata['prompt'] == prompt and metadata['seed'] == seed and metadata['task_index'] == 7
                    assert metadata['input_png_sha256'] == sha(image)
                    for key in ('timesteps', 'sigmas'):
                        runtime.require_exact(record[key], natural[rule['goal']][key], 'entire actual frozen schedule')
                    assert hooks(model) == initial_hooks
                    if rule['whole_reference_control']:
                        ref = frozen['whole_physical_references'][label]
                        folder, name = Path(ref['folder']), f'chunk_{query:02d}/states.pt'
                        assert sha(folder / name) == ref['files_current_sha256'][name]
                        assert image.read_bytes() == (folder / f'input_{query:02d}.png').read_bytes()
                        runtime.require_exact(record, torch.load(folder / name, map_location='cpu', weights_only=True, mmap=True), 'ENTIRE natural live-self feedback')
                        references.append(dict(case=label, query=query, entire_record_exact=True, actual_feedback_PNG_bytes_exact=True))
                    metadata.update(trial=label, query=query, shift_cm=15, scene='x15', scene_prompt=SCENE_PROMPT,
                        policy_prompt=prompt, policy_goal=rule['goal'], recipient_goal=rule['recipient_goal'], donor_goal=rule['donor_goal'],
                        image_path=str(image), saved_q0_prediction=query == 0, current_execution_model_calls=0 if query == 0 else 30,
                        q0_vision_noise_source_seed=198, q0_action_noise_source_seed=195, q0_runtime_rng_seed=195,
                        saved_q0_attention_arm=rule['arm'], saved_q0_window=rule['selected_window'],
                        actual_query_seeds=rule['actual_query_seeds'], seed_rule='195 + query_index after cached q0',
                        component_intervention_in_this_execution=False, attention_mask_intervention_in_this_execution=False,
                        paired_noise_checked_exact=paired, source_producer_folder=source['folder'] if query == 0 else None,
                        source_q0_states_sha256=source['files_sha256']['states.pt'] if query == 0 else None)
                    write(chunk / 'metadata.json', metadata)
                    assert read(chunk / 'normalized_actions.json') == record['actions'].tolist()
                    served.append((label, query))
                    write(destination / 'progress.json', dict(queries=len(served), cached_q0=cached, native_predictions=fresh,
                        fresh_model_forwards=30 * fresh, actual_later_whole_noise_pairs=pairs, served_queries=served))
                    print('[INSTRUCTION-INTERACTION] ' + json.dumps(dict(requests=len(served), case=label, query=query)), flush=True)
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
        while not failed and len(served) < 64:
            http.handle_request()
            assert not (out / 'closed-loop/failed.json').exists() and not (out / 'wrapper_failed.json').exists()
        assert not failed and len(served) == 64 and cached == 8 and fresh == 56 and pairs == 49
        expected = [(label, q) for label in frozen['trial_order'] if frozen['trials'][label]['whole_reference_control'] for q in range(8)]
        assert [(row['case'], row['query']) for row in references] == expected and len(references) == 16
        assert hooks(model) == initial_hooks
        write(destination / 'whole-reference-query-controls.json', dict(state='exact', queries=references), exclusive=True)
        write(destination / 'complete.json', dict(state='complete', queries=64, saved_q0_predictions=8,
            native_predictions=56, fresh_model_forwards=1680, actual_later_whole_noise_pairs=49,
            two_live_self_all8_entire_records_and_PNGs_exact=True, final_model_hooks_none=True,
            files_sha256={name: sha(destination / name) for name in ('saved-q0-controls.json', 'whole-reference-query-controls.json')}, scope=SCOPE), exclusive=True)
    finally:
        if http is not None:
            http.server_close()


def simulate(baseline):
    frozen, probe, legacy, simulator, helper, cross, out = prepared(baseline)
    assert not (out / 'closed-loop').exists(), 'Do not repeat a partial physical batch'
    os.environ.setdefault('MUJOCO_GL', 'egl')
    sys.path.insert(0, '/home/current/work/openpi-demo/cosmos-policy')
    sys.path.insert(0, str(ROOT / 'cosmos-framework'))
    from cosmos_framework.simulation.libero.closed_loop_eval import _framewise_action_to_delta, _remap_gripper
    np = simulator.np
    collector = simulator.load_file('instruction_execution_collector', ROOT / 'work/collect_cosmos_goal_pair.py')
    rollout = simulator.load_file('instruction_execution_controller', ROOT / 'work/rollout_cosmos_goal_pair.py')
    stats = read(Path(frozen['threshold_contract']['frozen_files']['normalizer_stats']['path']))['global_raw']
    lo, hi = np.asarray(stats['q01']), np.asarray(stats['q99'])
    scale, offset = np.maximum(hi - lo, 1e-8) / 2, (hi + lo) / 2
    whole_controls, comparisons, case_files = [], [], {}
    for label in frozen['trial_order']:
        # The frozen6b physical body creates the original environment/controller.
        # Its own __file__ records that body truthfully; this wrapper is separately sealed.
        legacy.execute_trials(out, [label], collector, rollout, simulator, frozen)
        folder, rule = out / 'closed-loop' / label, frozen['trials'][label]
        inputs = out / 'inputs/x15'
        initial = np.load(inputs / 'state.npy', allow_pickle=False)
        state, trajectory, summary = np.load(folder / 'sim_states.npy', allow_pickle=False), read(folder / 'trajectory.json'), read(folder / 'summary.json')
        assert state.shape == (129, initial.size) and state.dtype == initial.dtype and np.isfinite(state).all()
        assert rollout.array_exact(state[0], initial) and len(trajectory) == 129
        assert all(row['exact'] for row in read(folder / 'initial_checks.json'))
        assert all(value is True for value in read(folder / 'initial_controller_checks.json').values())
        assert read(folder / 'initial_contact_checks.json') == read(inputs / 'initial_contact_checks.json')
        with np.load(folder / 'initial_controller.npz', allow_pickle=False) as current:
            with np.load(inputs / 'controller.npz', allow_pickle=False) as original:
                assert set(current.files) == set(original.files)
                assert all(rollout.array_exact(current[key], original[key]) for key in current.files)
        norm = np.asarray(read(folder / 'normalized_actions.json'))
        assert norm.shape == (128, 10) and norm.dtype == np.float64 and np.isfinite(norm).all()
        assert rollout.array_exact(norm, np.concatenate([np.asarray(read(folder / f'chunk_{q:02d}/normalized_actions.json')) for q in range(8)]))
        assert rollout.array_exact(norm[:16], np.asarray(frozen['q0_sources'][label]['normalized_actions']))
        raw = norm * scale + offset
        native = np.asarray([_remap_gripper(_framewise_action_to_delta(item, '6d').tolist(), 'zero_one') for item in raw])
        for name, value in (('denormalized_actions.json', raw), ('actions.json', native), ('executed_actions.json', np.clip(native, -1, 1))):
            assert rollout.array_exact(np.asarray(read(folder / name)), value), ('all128 official conversions', label, name)
        strict = helper.strict_windows(trajectory, summary, np)
        write(folder / 'strict-windows.json', strict, exclusive=True)
        if rule['whole_reference_control']:
            reference = frozen['whole_physical_references'][label]
            control = cross.native_control(out, label, reference, np)
            for name in ('initial_controller.npz', 'controller-step16.npz'):
                with np.load(folder / name, allow_pickle=False) as current:
                    with np.load(Path(reference['folder']) / name, allow_pickle=False) as old:
                        assert set(current.files) == set(old.files)
                        assert all(rollout.array_exact(current[key], old[key]) for key in current.files)
            whole_controls.append(dict(**control, reference_case=reference['reference_case'], typed_controller0_and16_exact=True))
        milk0, cheese0 = (np.asarray(trajectory[0]['objects'][name]) for name in ('milk_1', 'cream_cheese_1'))
        direction = milk0 - cheese0
        length = float(np.linalg.norm(direction))
        assert length > 1e-6
        direction /= length
        eef0, eef16 = (np.asarray(trajectory[i]['eef_xyz']) for i in (0, 16))
        comparisons.append(dict(case=label, scene='x15', goal=rule['goal'], recipient_goal=rule['recipient_goal'], donor_goal=rule['donor_goal'], arm=rule['arm'],
            selected_objects=summary['selected_objects'], strict_requested_recipient_selected=summary['selected_objects'] == [TARGETS[rule['recipient_goal']]],
            strict_only_donor_selected=summary['selected_objects'] == [TARGETS[rule['donor_goal']]],
            first_selection_window_start=summary['first_selection_step'], first_selected_object=summary['first_selected_object'],
            per_object=summary['per_object'], q0_eef_path_xyz_m=[row['eef_xyz'] for row in trajectory[:17]],
            eef_q0_displacement_m=(eef16 - eef0).tolist(), q0_eef_projection_toward_milk_from_cheese_m=float(np.dot(eef16 - eef0, direction)),
            initial_cheese_to_milk_unit_direction=direction.tolist(), initial_object_separation_m=length,
            final_milk_in_basket_scene_predicate=bool(trajectory[-1]['milk_in_basket']), strict_windows_sha256=sha(folder / 'strict-windows.json')))
        names = ('sim_states.npy', 'actual.mp4', 'initial_controller.npz', 'controller-step16.npz', 'summary.json',
            'strict-windows.json', 'query_contract.json', 'initial_checks.json', 'initial_controller_checks.json', 'initial_contact_checks.json',
            'trajectory.json', 'actions.json', 'executed_actions.json', 'normalized_actions.json', 'denormalized_actions.json',
            *[f'input_{q:02d}.png' for q in range(8)], *[f'chunk_{q:02d}/{name}' for q in range(8) for name in ('states.pt', 'metadata.json', 'normalized_actions.json')])
        case_files[label] = {name: sha(folder / name) for name in names}
    done, served = read(out / 'closed-loop/complete.json'), read(out / 'server/complete.json')
    assert done['state'] == served['state'] == 'complete' and done['cases'] == 8 and done['completed_trials'] == frozen['trial_order']
    assert served['queries'] == 64 and served['saved_q0_predictions'] == 8 and served['native_predictions'] == 56
    assert served['fresh_model_forwards'] == 1680 and served['actual_later_whole_noise_pairs'] == 49
    assert len(whole_controls) == 2 and served['two_live_self_all8_entire_records_and_PNGs_exact'] is True
    for name, expected in served['files_sha256'].items():
        assert sha(out / 'server' / name) == expected
    by_case = {row['case']: row for row in comparisons}
    primary_cases = [by_case[label] for label in probe.PRIMARY['cases']]
    assert len(primary_cases) == 2
    passed = all(row['strict_only_donor_selected'] for row in primary_cases)
    prediction = 'supported' if passed else 'rejected'
    write(out / 'primary-prediction.json', dict(state='evaluated_all8', **probe.PRIMARY, prediction=prediction,
        both_suffix_joint_strict_only_donor=passed, actual_primary_cases=primary_cases,
        rule='All8 with exact live-self controls and frozen random gates; both preregistered J cases must strictly select only donor.',
        original_milk_15cm_repair_claimed=False), exclusive=True)
    write(out / 'comparisons.json', dict(state='all8_complete', cases=comparisons, primary_prediction=prediction,
        MLP_comparison_scope='J versus C is a same-live local controlled intervention policy; later trajectories differ dynamically.',
        random_scope='Actual BF16 pre-W strength matched; W-output-strength inequality remains an alternative.',
        original_milk_15cm_repair_claimed=False, all8_executed_without_score_selection=True, scope=SCOPE), exclusive=True)
    write(out / 'controls.json', dict(two_live_self_whole_controls=whole_controls, actual_case_files_sha256=case_files,
        all8_own_cached_q0_and_all128_official_conversions_exact=True, all8_strict_windows_recomputed=True), exclusive=True)
    names = ('controls.json', 'comparisons.json', 'provenance.json', 'prepared.json', 'primary-preregistered.json',
             'primary-prediction.json', 'server/complete.json', 'closed-loop/complete.json', 'summary.json')
    write(out / 'complete.json', dict(state='complete', script_sha256=sha(Path(__file__)), physical_trials=8,
        trial_order=frozen['trial_order'], saved_q0_predictions=8, native_predictions=56, fresh_model_forwards=1680,
        producer_q0_forwards=240, combined_two_stage_model_forwards=1920, actual_later_whole_noise_pairs=49,
        actual_physical_actions=1024, two_entire129_state_5JSON_8PNG_all8_modelrecords_controller0_16_controls_exact=True,
        all8_cached_q0_action_and_strict_window_gates_passed=True, all8_executed_without_score_selection=True,
        primary_prediction=prediction, both_suffix_joint_strict_only_donor=passed,
        files_sha256={name: sha(out / name) for name in names}, scope=SCOPE), exclusive=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('prepare', 'server', 'simulate'), required=True)
    args = parser.parse_args()
    baseline, out = args.output.resolve(), args.output.resolve() / STAGE
    if args.mode == 'prepare':
        frozen, _, _, _, _, _ = contract(baseline)
        assert shutil.disk_usage(baseline).free >= 700 * 1024 ** 2, 'Need the registered700MiB physical reserve'
        out.mkdir(exist_ok=False)
        (out / 'inputs').mkdir(exist_ok=False)
        (out / 'inputs/x15').symlink_to(frozen['scenes']['x15']['folder'], target_is_directory=True)
        write(out / 'provenance.json', frozen, exclusive=True)
        write(out / 'primary-preregistered.json', dict(state='preregistered', **frozen['primary_prediction']), exclusive=True)
        write(out / 'prepared.json', dict(state='prepared', provenance_sha256=sha(out / 'provenance.json')), exclusive=True)
        print('[INSTRUCTION-INTERACTION] prepared ' + str(out), flush=True)
        return
    try:
        server(baseline) if args.mode == 'server' else simulate(baseline)
    except Exception as exc:
        if out.is_dir() and not (out / 'wrapper_failed.json').exists() and not (out / 'complete.json').exists():
            write(out / 'wrapper_failed.json', dict(mode=args.mode, error=str(exc), traceback=traceback.format_exc()), exclusive=True)
        raise


if __name__ == '__main__':
    main()
