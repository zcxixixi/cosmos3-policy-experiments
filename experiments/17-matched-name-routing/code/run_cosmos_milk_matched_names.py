"""Execute all24 frozen matched-name cases with seven native feedback queries.

Each X12/X15 scene uses its own accepted typed input root. Policy prompts are
real milk-box/cream-cheese instructions, distinct from task7's scene prompt.
No q0 scoring selects cases; no failed/partial case is retried automatically.
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
PROBE_SHA = '0446978830ac36978396fdad5b09df238b0fd6b3b425aaa351a9b40262f94df0'
LANGUAGE_ADAPTER_SHA = '3c42a3a7137ba2dffc78699046198acbb1d924e229fdaf92bf516afe11530437'
PORT = 8938
STAGE = 'future-matched-names-execution'
SCENE_PROMPT = 'pick up the milk and place it in the basket'
PROMPTS = dict(milk_box='pick up the milk box and place it in the basket',
               cream_cheese='pick up the cream cheese and place it in the basket')
TARGETS = dict(milk_box='milk_1', cream_cheese='cream_cheese_1')
SCENES = dict(x12=12, x15=15)
ARMS = ('native', 'all_allowed', 'full_hardmask')
PLAN = tuple((f'{scene}_{goal}_V{v}_A195_{arm}', scene, goal, v, arm)
             for arm in ARMS for scene in SCENES for goal in PROMPTS for v in (195, 198))
SCOPE = ('All24 fixed X12/X15 task7 scenes with equal-packed-length milk-box/cream-cheese '
         'policy prompts and V195/V198, A/runtime195. Q0 only uses its frozen arm; '
         'q1..7 use each recipient instruction and native inference. Strict grasp, '
         'q0 path, first finger contact and milk-in-basket scene predicate are separate. '
         'A successful changed prompt is not repair of the original milk instruction. '
         'No q0 selection, training, new warmup, semantic decoder or root-cause claim.')
PRIMARY = dict(name='H_matched_name_native_targets_x15',
    cases=[label for label, scene, _, _, arm in PLAN if scene == 'x15' and arm == 'native'],
    requested_objects=TARGETS, criterion='Each X15 native V195/V198 case must strictly select only its named object.',
    all24_physics_required=True, numerical_sham_is_separate_cut_attribution_diagnostic=True,
    no_q0_score_or_rank_selection=True, original_milk_15cm_goal_not_declared_repaired=True)


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
    """Validate real producer/legacy seals before output creation or model load."""
    assert len(PROBE_SHA) == 64 and all(c in '0123456789abcdef' for c in PROBE_SHA), 'Producer is not frozen'
    legacy_path = ROOT / 'work/run_cosmos_milk_language_selectivity.py'
    assert sha(legacy_path) == LANGUAGE_ADAPTER_SHA
    legacy = load('matched_names_frozen_language_adapter', legacy_path)
    old_frozen, _, simulator, helper, cross = legacy.contract(baseline)
    probe_path = ROOT / 'work/probe_cosmos_milk_matched_names.py'
    assert sha(probe_path) == PROBE_SHA
    probe = load('matched_names_frozen_producer', probe_path)
    assert probe.STAGE == 'matched-names' and tuple(probe.PLAN) == PLAN and probe.PRIMARY == PRIMARY
    context = probe.source_contract(baseline)
    producer = baseline / probe.STAGE
    assert producer.is_dir() and not (producer / 'failed.json').exists()
    documents = ('results', 'protocol', 'provenance', 'sources', 'primary-prediction', 'matched-language-contract')
    docs = {name: read(producer / (name + '.json')) for name in ('complete', *documents)}
    complete, result, protocol, provenance = (docs[k] for k in ('complete', 'results', 'protocol', 'provenance'))
    assert complete['state'] == result['state'] == 'complete'
    assert complete['script_sha256'] == provenance['script_sha256'] == PROBE_SHA
    assert complete['fresh_q0_predictions'] == 24 and complete['fresh_model_forwards'] == 720
    assert complete['official_dispatch_counts'] == result['official_dispatch_counts'] == probe.expected_counts()
    assert complete['total_official_dispatch_calls'] == 70200
    assert complete['maximum_stage_bytes'] == probe.MAX_BYTES
    for key in ('six_X12_canonical_entire_accepted_language_records_exact',
                'actual_equal_length_two_target_ID_packing_gate_passed',
                'all720_actual_final_action_input_output_timestep_sigma_raw_saved',
                'all_original_symbols_restored', 'all_owned_hooks_removed'):
        assert complete[key] is True
    assert complete['full37_UND_current_action_raw_steps'] == complete['full_model_kwargs_tuple_raw_steps'] == [0]
    assert complete['full_QKV_raw_sites'] == 24 and complete['site_raw_chain_captures'] == 0
    for name in documents:
        assert complete[name.replace('-', '_') + '_sha256'] == sha(producer / (name + '.json'))
    assert provenance['sources_sha256'] == complete['sources_sha256']
    assert provenance['protocol_sha256'] == complete['protocol_sha256']
    assert protocol['trial_order'] == result['preregistered_physics_cases'] == [item[0] for item in PLAN]
    assert protocol['prompts'] == PROMPTS and protocol['target_objects'] == TARGETS
    assert protocol['primary_hypothesis'] == PRIMARY
    assert protocol['q0_predictions'] == 24 and protocol['model_forwards'] == 720
    assert docs['primary-prediction'] == dict(status='not_evaluated_q0_only', **PRIMARY, scope=probe.SCOPE)
    assert result['primary_physical_prediction_evaluated'] is complete['primary_physical_prediction_evaluated'] is False
    sources = docs['sources']
    assert sources['original_sources'] == context['original_noise_sources']
    assert sources['scenes'] == protocol['scenes'] == context['scenes']
    assert sources['frozen_contract'] == context['frozen_contract']
    matched = docs['matched-language-contract']
    assert matched['state'] == 'exact' and matched['goals'] == list(PROMPTS)
    assert matched['target_token_ids'] == probe.TARGET_IDS
    assert matched['actual_prompt_token_lengths'] == dict.fromkeys(PROMPTS, 11)
    assert matched['full_physics_required'] is True
    assert [(item['scene'], item['vision_noise_source_seed']) for item in matched['cases']] == [
        (scene, v) for scene in SCENES for v in (195, 198)]
    matched_pairs = {}
    for item in matched['cases']:
        positions = item['changed_actual_input_id_positions']
        assert len(positions) == 2 and positions[1] == positions[0] + 1
        assert item['actual_und_len'] == 122 and item['actual_joint_length'] == 388
        assert item['complete_initial_kwargs_only_two_target_IDs_changed'] is True
        assert item['all30_other_pack_fields_and_positions_byte_exact'] is True
        matched_pairs[(item['scene'], item['vision_noise_source_seed'])] = item
    old_out = baseline / 'future-language-selectivity-execution'
    old_complete, old = read(old_out / 'complete.json'), read(old_out / 'provenance.json')
    assert old_complete['state'] == 'complete' and old_complete['script_sha256'] == LANGUAGE_ADAPTER_SHA
    assert old_complete['physical_trials'] == 18
    assert old_complete['six_entire129_state_5JSON_8PNG_and_all8_modelrecords_controls_exact'] is True
    for name, expected in old_complete['files_sha256'].items():
        assert sha(old_out / name) == expected
    old_controls = read(old_out / 'controls.json')
    scenes = {}
    for scene, shift in SCENES.items():
        source = context['scenes'][scene]
        folder = Path(source['folder'])
        assert folder.name == scene and source['shift_cm'] == shift
        assert Path(source['image']).resolve() == (folder / 'input.png').resolve()
        for name, expected in source['files_sha256'].items():
            assert sha(folder / name) == expected
        typed_root = folder.parent
        ic, im = read(typed_root / 'complete.json'), read(typed_root / 'metadata.json')
        assert ic['state'] == 'matched_inputs_verified' and ic['additional_warmup_steps'] == 0
        assert not (typed_root / 'failed.json').exists()
        assert ic['metadata_sha256'] == sha(typed_root / 'metadata.json')
        assert im['prompt'] == SCENE_PROMPT and im['shifts_cm'][scene] == shift
        assert im['scenes'][scene]['shift_cm'] == shift
        for key, name in (('input_png_sha256', 'input.png'), ('state_sha256', 'state.npy'),
                          ('arrays_sha256', 'arrays.json'), ('controller_sha256', 'controller.npz')):
            assert im['scenes'][scene][key] == sha(folder / name)
        scenes[scene] = dict(folder=str(folder), image=str(folder / 'input.png'), shift_cm=shift,
            input_root=str(typed_root), metadata_sha256=sha(typed_root / 'metadata.json'),
            complete_sha256=sha(typed_root / 'complete.json'), files_sha256=source['files_sha256'])
    rows, q0, trials, references = result['cases'], {}, {}, {}
    assert len(rows) == 24 and len({r['case'] for r in rows}) == 24
    assert [(r['case'], r['scene'], r['goal'], r['vision_noise_source_seed'], r['arm']) for r in rows] == list(PLAN)
    native_labels = {scene: {goal: {} for goal in PROMPTS} for scene in SCENES}
    for row in rows:
        label, scene, goal, v, arm = (row[k] for k in ('case', 'scene', 'goal', 'vision_noise_source_seed', 'arm'))
        folder = producer / label
        assert row['action_noise_source_seed'] == row['runtime_rng_seed'] == 195
        assert row['fresh_model_forwards'] == 30
        assert set(row['files_sha256']) == set(probe.FILES)
        for name, expected in row['files_sha256'].items():
            assert sha(folder / name) == expected
        assert len(row['boundary_files']) == 30
        for step, item in enumerate(row['boundary_files']):
            assert item['step'] == step and item['file'] == f'boundaries-t{step:02d}.pt'
            assert sha(folder / item['file']) == item['sha256']
        manifest = read(folder / 'dispatch-captures.json')
        assert manifest['state'] == 'complete' and manifest['files'] == row['dispatch_files']
        assert manifest['full_QKV_raw_steps'] == [0] and len(row['dispatch_files']) == 30
        for step, item in enumerate(row['dispatch_files']):
            assert item['step'] == step and item['layer_zero_based'] == 35
            assert item['file'] == f'dispatch-t{step:02d}-L35.pt' and item['full_QKV_raw_saved'] is (step == 0)
            assert sha(folder / item['file']) == item['sha256']
        metadata = read(folder / 'metadata.json')
        assert metadata['prompt'] == PROMPTS[goal] and metadata['seed'] == 195
        assert metadata['task_index'] == 7 and metadata['scene'] == scene
        assert metadata['input_png_sha256'] == sha(Path(scenes[scene]['image']))
        assert row['existing_entire_canonical_record_exact'] is (scene == 'x12' and goal == 'cream_cheese')
        assert row['actual_und_len'] == 122
        assert row['actual_target_positions'] == matched_pairs[(scene, v)]['changed_actual_input_id_positions']
        assert row['actual_target_token_ids'] == matched['target_token_ids'][goal]
        norm = simulator.np.asarray(read(folder / 'normalized_actions.json'))
        assert norm.shape == (16, 10) and simulator.np.isfinite(norm).all()
        q0[label] = dict(folder=str(folder), files_sha256=row['files_sha256'],
            boundary_files=row['boundary_files'], dispatch_files=row['dispatch_files'], normalized_actions=norm.tolist())
        control = scene == 'x12' and goal == 'cream_cheese'
        trials[label] = dict(scene=scene, shift_cm=SCENES[scene], goal=goal, policy_prompt=PROMPTS[goal], arm=arm,
            selected_window=dict(steps=[0, 14], layers=[0, 8]) if arm == 'full_hardmask' else None,
            q0_vision_noise_source_seed=v, actual_query_seeds=list(range(195, 203)), whole_reference_control=control)
        if arm == 'native':
            native_labels[scene][goal][str(v)] = label
        if control:
            ref_label = f'cream_cheese_V{v}_A195_{arm}'
            ref_folder = old_out / 'closed-loop' / ref_label
            files = old_controls['actual_case_files_sha256'][ref_label]
            for name, expected in files.items():
                assert sha(ref_folder / name) == expected
            references[label] = dict(folder=str(ref_folder), reference_case=ref_label, files_current_sha256=files)
    assert len(references) == 6
    frozen = dict(script_sha256=sha(Path(__file__)), producer_script_sha256=PROBE_SHA, producer_stage=probe.STAGE,
        producer_documents_sha256={name: sha(producer / (name + '.json')) for name in docs},
        threshold_contract=old_frozen['threshold_contract'], actual_transformer_sha256=old['actual_transformer_sha256'],
        actual_pipeline_source_sha256=old['actual_pipeline_source_sha256'], metrics_path=old['metrics_path'],
        components_path=old['components_path'], producer_source_cases=old['producer_source_cases'],
        language_producer_path=str(ROOT / 'work/probe_cosmos_milk_language_selectivity.py'),
        language_producer_sha256=old_frozen['producer_script_sha256'],
        q0_sources=q0, trials=trials, trial_order=[r['case'] for r in rows], native_labels=native_labels,
        scene_prompt=SCENE_PROMPT, scene_task_index=7, scenes=scenes, prompts=PROMPTS, targets=TARGETS,
        whole_physical_references=references, accepted_language_complete_sha256=sha(old_out / 'complete.json'),
        accepted_language_controls_sha256=sha(old_out / 'controls.json'), legacy_adapter_sha256=LANGUAGE_ADAPTER_SHA,
        producer_protocol=protocol, url=f'http://127.0.0.1:{PORT}', saved_q0_predictions=24,
        requests=192, native_predictions=168, fresh_model_forwards=5040, producer_q0_forwards=720,
        combined_two_stage_model_forwards=5760, later_whole_noise_pairs=161, physical_actions=3072,
        primary_prediction=PRIMARY, all24_execute_without_score_selection=True, scope=SCOPE)
    simulator.SHIFTS = SCENES
    simulator.SCOPE = SCOPE
    return frozen, probe, simulator, helper, cross


def prepared(baseline):
    frozen, probe, simulator, helper, cross = contract(baseline)
    out = baseline / STAGE
    assert out.is_dir() and not (out / 'wrapper_failed.json').exists()
    assert not (out / 'complete.json').exists(), 'Completed execution is immutable; do not repeat'
    assert not (out / 'server/failed.json').exists() and not (out / 'simulator_failed.json').exists()
    assert not (out / 'closed-loop/failed.json').exists()
    assert read(out / 'provenance.json') == frozen
    assert read(out / 'prepared.json') == dict(state='prepared', provenance_sha256=sha(out / 'provenance.json'))
    assert read(out / 'primary-preregistered.json') == dict(state='preregistered', **PRIMARY)
    assert not (out / 'inputs/metadata.json').exists(), 'Do not invent a merged input metadata document'
    for scene, item in frozen['scenes'].items():
        assert (out / 'inputs' / scene).is_symlink()
        assert (out / 'inputs' / scene).resolve() == Path(item['folder']).resolve()
    return frozen, probe, simulator, helper, cross, out


def scene_inputs(output, frozen, simulator, collector, rollout):
    """Load each scene with its own real accepted metadata, never a merged one."""
    references, controllers, np = {}, {}, simulator.np
    for scene, item in frozen['scenes'].items():
        folder, typed_root = output / 'inputs' / scene, Path(item['input_root'])
        assert folder.resolve() == Path(item['folder']).resolve()
        assert sha(typed_root / 'metadata.json') == item['metadata_sha256']
        assert sha(typed_root / 'complete.json') == item['complete_sha256']
        metadata, complete = read(typed_root / 'metadata.json'), read(typed_root / 'complete.json')
        assert complete['state'] == 'matched_inputs_verified' and complete['additional_warmup_steps'] == 0
        assert complete['metadata_sha256'] == item['metadata_sha256']
        assert not (typed_root / 'failed.json').exists()
        assert metadata['script_sha256'] == sha(Path(simulator.__file__))
        assert metadata['collector_sha256'] == sha(Path(collector.__file__))
        assert metadata['controller_helper_sha256'] == sha(Path(rollout.__file__))
        assert metadata['prompt'] == SCENE_PROMPT and metadata['shifts_cm'][scene] == SCENES[scene]
        for name, expected in item['files_sha256'].items():
            assert sha(folder / name) == expected
        references[scene] = simulator.load_capture(folder, metadata, collector)
        with np.load(folder / 'controller.npz', allow_pickle=False) as archive:
            controllers[scene] = {key: archive[key].copy() for key in archive.files}
    return references, controllers


def verify_saved(record, audit, folder, native, sources, v, runtime, components, language):
    """Audit all30 small arrays and t0 full raw; omitted full arrays retain live SHA."""
    torch = runtime.torch
    runtime.require_exact(record['pure_noise'], [sources[v]['pure_noise'][0], sources[195]['pure_noise'][1]], 'entire selected raw V/A draws')
    runtime.require_exact(record['prepared_latents_and_masks'], native['prepared_latents_and_masks'], 'own scene/name/V entire preparation')
    runtime.require_exact(record['model_input'], native['model_input'], 'own scene/name/V entire first kwargs')
    runtime.require_exact(record['model_input'], audit['initial_model_kwargs'], 'observed initial kwargs')
    runtime.require_exact(record['action_states'][0], sources[195]['action_states'][0], 'original A195 FP32 solver sample')
    assert torch.count_nonzero(record['action_states'][..., 10:]) == 0
    clean = record['model_input']['vision_tokens'][0][0, :, 0]
    clean_meta = language.tensor_meta(clean, runtime)
    report = read(folder / 'observer-report.json')
    assert report['state'] == 'complete' and report['full37_boundary_raw_steps'] == [0]
    assert report['final_action_boundary_raw_steps'] == list(range(30)) and report['full_QKV_raw_sites'] == 1
    for key in ('all_actual_dispatch_return_flatten_equals_pre_W_byte_exact',
                'all1080_actual_full266_projection_calls_preserved', 'future200_and_UND_native_returns_preserved',
                'all37_by30_observed_hidden_finite', 'all_original_QKV_and_CPU_CUDA_global_RNG_bytes_preserved',
                'all_owned_hooks_removed', 'original_dispatch_symbol_restored'):
        assert report[key] is True
    assert report['extra_model_attention_replay_projection_MLP_calls'] == report['site_raw_chain_captures'] == 0
    for step in range(30):
        data = torch.load(folder / f'boundaries-t{step:02d}.pt', map_location='cpu', weights_only=True, mmap=True)
        runtime.require_exact(data['actual_pack'], audit['model_steps'][step], 'actual observer pack/' + str(step))
        runtime.require_exact(data['actual_model_action_tokens'], record['action_states'][step, 0].to(torch.bfloat16), 'actual BF16 solver cast/' + str(step))
        runtime.require_exact(data['actual_model_action_output'], record['action_velocity'][step], 'actual whole action output/' + str(step))
        runtime.require_exact(data['actual_model_action_timesteps'], data['actual_pack']['metadata']['action_timesteps'], 'actual timestep/' + str(step))
        runtime.require_exact(data['solver_sigma_from_same_frozen_schedule'], record['sigmas'][step], 'actual sigma/' + str(step))
        assert data['actual_current_frame_sha256'] == clean_meta['sha256']
        assert data['actual_current_frame_metadata'] == clean_meta
        action = data['action_hidden']
        assert action.shape == ((37, 16, 4096) if step == 0 else (16, 4096))
        assert action.dtype == torch.bfloat16 and bool(torch.isfinite(action).all())
        assert data['action_boundary_ids'] == (list(range(37)) if step == 0 else [36])
        meta = data['hidden_boundary_metadata']
        assert len(meta) == 37 and [item['boundary'] for item in meta] == list(range(37))
        assert len(data['site_events']) == 36
        for boundary, item in enumerate(meta):
            for group, shape in (('und', [122, 4096]), ('current', [50, 4096]), ('action', [16, 4096])):
                assert item[group]['shape'] == shape and item[group]['dtype'] == 'torch.bfloat16'
                assert len(item[group]['sha256']) == 64
            if step == 0:
                raw_groups = dict(und=data['und_hidden'][boundary], current=data['current_hidden'][boundary], action=action[boundary])
            elif boundary == 36:
                raw_groups = dict(action=action)
            else:
                raw_groups = {}
            for group, value in raw_groups.items():
                assert bool(torch.isfinite(value).all())
                witness = language.tensor_meta(value, runtime)
                assert all(item[group][key] == witness[key] for key in ('shape', 'dtype', 'bytes', 'sha256'))
        if step == 0:
            kwargs = data['actual_model_kwargs']
            runtime.require_exact(kwargs, record['model_input'], 't0 saved complete actual kwargs')
            runtime.require_exact(data['actual_pack'], dict(layout=components._layout(kwargs), metadata={key: kwargs.get(key) for key in components._STRUCTURAL_KEYS}), 't0 complete pack')
            runtime.require_exact(kwargs['vision_tokens'][0][0, :, 0], clean, 't0 raw current clamp')
            runtime.require_exact(data['model_output'][2][0], data['actual_model_action_output'], 't0 actual full tuple action output')
            assert language.tree_sha(kwargs, runtime) == data['actual_model_kwargs_sha256']
            assert language.tree_sha(data['model_output'], runtime) == data['model_output_sha256']
            assert data['current_hidden'].shape == (37, 50, 4096) and data['und_hidden'].shape == (37, 122, 4096)
        else:
            assert data['actual_model_kwargs'] is data['model_output'] is data['current_hidden'] is data['und_hidden'] is None
            for key in ('actual_model_kwargs_sha256', 'model_output_sha256'):
                assert len(data[key]) == 64 and all(c in '0123456789abcdef' for c in data[key])
    manifest = read(folder / 'dispatch-captures.json')
    assert report['dispatch_capture_manifest_sha256'] == sha(folder / 'dispatch-captures.json')
    assert manifest['files'] == report['dispatch_files'] and len(manifest['files']) == 30
    lc = torch.load(folder / 'language-contract.pt', map_location='cpu', weights_only=True, mmap=True)
    target_positions = lc['language']['target_positions']
    for step, entry in enumerate(manifest['files']):
        assert entry['step'] == step and sha(folder / entry['file']) == entry['sha256']
        dispatch = torch.load(folder / entry['file'], map_location='cpu', weights_only=True, mmap=True)
        assert dispatch['step'] == step and dispatch['layer_zero_based'] == 35
        assert dispatch['native_kwargs'] == dict(is_causal=False, enable_gqa=True, backend=None, parallel_config=None)
        runtime.require_exact(dispatch['target_key_indexes'], torch.tensor(target_positions), 'actual target key indexes')
        assert dispatch['full_QKV_raw_saved'] is (step == 0)
        for key, shape in (('Qaction16', (1, 16, 32, 128)), ('Ktarget2', (1, 2, 8, 128)),
                           ('Vtarget2', (1, 2, 8, 128)), ('Znative_action16', (1, 16, 32, 128))):
            value = dispatch[key]
            assert value.shape == shape and value.dtype == torch.bfloat16 and bool(torch.isfinite(value).all())
            witness = language.tensor_meta(value, runtime)
            assert all(dispatch['sparse_metadata'][key][field] == witness[field] for field in ('shape', 'dtype', 'bytes', 'sha256'))
        native_arm = dispatch['arm'] == 'native'
        assert dispatch['causal_suffix_raw_saved'] is native_arm
        runtime.require_exact(dispatch['causal_suffix_key_indexes'], torch.arange(target_positions[0], 122), 'actual native causal suffix indexes')
        if native_arm:
            assert dispatch['Zmerged_action16'] is dispatch['sparse_metadata']['Zmerged_action16'] is None
            for prefix in ('K', 'V'):
                key = prefix + 'causal_suffix'
                value = dispatch[key]
                assert value.shape == (1, 122 - target_positions[0], 8, 128)
                assert value.dtype == torch.bfloat16 and bool(torch.isfinite(value).all())
                witness = language.tensor_meta(value, runtime)
                assert all(dispatch['sparse_metadata'][key][field] == witness[field] for field in ('shape', 'dtype', 'bytes', 'sha256'))
                runtime.require_exact(dispatch[prefix + 'target2'], value.index_select(1, torch.tensor(target_positions) - target_positions[0]), 'target K/V bound to actual native causal suffix')
        else:
            assert dispatch['Kcausal_suffix'] is dispatch['Vcausal_suffix'] is None
            value = dispatch['Zmerged_action16']
            assert value.shape == (1, 16, 32, 128) and value.dtype == torch.bfloat16 and bool(torch.isfinite(value).all())
            witness = language.tensor_meta(value, runtime)
            assert all(dispatch['sparse_metadata']['Zmerged_action16'][field] == witness[field] for field in ('shape', 'dtype', 'bytes', 'sha256'))
        assert dispatch['actual_pre_W_metadata']['sha256'] == dispatch['actual_returned_output_metadata']['sha256']
        if step == 0:
            assert dispatch['Q'].shape == (1, 266, 32, 128) and dispatch['K'].shape == dispatch['V'].shape == (1, 388, 8, 128)
            for value, meta in zip((dispatch['Q'], dispatch['K'], dispatch['V']), dispatch['QKV_metadata']):
                assert value.dtype == torch.bfloat16 and bool(torch.isfinite(value).all())
                witness = language.tensor_meta(value, runtime)
                assert all(meta[key] == witness[key] for key in ('shape', 'dtype', 'bytes', 'sha256'))
            for key in ('native_output', 'actual_returned_output'):
                value = dispatch[key]
                assert value.shape == (1, 266, 32, 128) and value.dtype == torch.bfloat16 and bool(torch.isfinite(value).all())
                witness = language.tensor_meta(value, runtime)
                assert all(dispatch[key + '_metadata'][field] == witness[field] for field in ('shape', 'dtype', 'bytes', 'sha256'))
            action_rows = dispatch['indexes']['action_rows']
            runtime.require_exact(dispatch['Qaction16'], dispatch['Q'].index_select(1, action_rows), 't0 selected action Q rows')
            runtime.require_exact(dispatch['Znative_action16'], dispatch['native_output'].index_select(1, action_rows), 't0 selected native action output rows')
            for prefix in ('K', 'V'):
                runtime.require_exact(dispatch[prefix + 'target2'], dispatch[prefix].index_select(1, torch.tensor(target_positions)), 't0 selected target K/V rows')
            actual_action = dispatch['Znative_action16'] if native_arm else dispatch['Zmerged_action16']
            runtime.require_exact(actual_action, dispatch['actual_returned_output'].index_select(1, action_rows), 't0 actual returned action rows')
            runtime.require_exact(dispatch['actual_pre_W'], dispatch['actual_returned_output'].squeeze(0).flatten(-2, -1), 'actual full L35 pre-W flatten')
        else:
            assert all(dispatch[key] is None for key in ('Q', 'K', 'V', 'native_output', 'actual_returned_output', 'actual_pre_W'))
            for meta in (*dispatch['QKV_metadata'], dispatch['native_output_metadata'], dispatch['actual_returned_output_metadata']):
                assert len(meta['sha256']) == 64


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
        base = load('matched_names_execution_runtime', Path(files['normal_runtime']['path']))
        for goal, prompt in frozen['prompts'].items():
            base.TASKS[goal] = (7, 'pick_up_the_milk_and_place_it_in_the_basket', prompt, frozen['targets'][goal])
        factory = load('matched_names_execution_factory', Path(files['factory']['path']))
        factory.OUT = destination
        assert sha(factory.SCHEDULER) == files['scheduler']['sha256']
        runtime = base.NormalRuntime(factory)
        torch, model = runtime.torch, runtime.pipe.transformer
        assert not model.training and not model.is_cache_enabled and not torch.is_grad_enabled()
        assert sha(Path(inspect.getfile(type(model)))) == frozen['actual_transformer_sha256']
        assert sha(Path(inspect.getfile(type(runtime.pipe)))) == frozen['actual_pipeline_source_sha256']
        initial_hooks = hooks(model)
        assert all(not pre and not post for _, pre, post in initial_hooks)
        scan = load('matched_names_execution_metrics', Path(frozen['metrics_path']))
        components = load('matched_names_execution_layout', Path(frozen['components_path']))
        assert sha(Path(frozen['language_producer_path'])) == frozen['language_producer_sha256']
        language = load('matched_names_raw_hash_helpers', Path(frozen['language_producer_path']))
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
            native = q0[frozen['native_labels'][rule['scene']][rule['goal']][str(rule['q0_vision_noise_source_seed'])]]
            audit = torch.load(folder / 'noise-audit.pt', map_location='cpu', weights_only=True, mmap=True)
            scan.validate_record(record, torch)
            verify_saved(record, audit, folder, native, sources, rule['q0_vision_noise_source_seed'], runtime, components, language)
            if rule['whole_reference_control']:
                ref = frozen['whole_physical_references'][label]
                path = Path(ref['folder']) / 'chunk_00/states.pt'
                assert sha(path) == ref['files_current_sha256']['chunk_00/states.pt']
                runtime.require_exact(record, torch.load(path, map_location='cpu', weights_only=True, mmap=True), 'ENTIRE accepted X12 cream_cheese q0/' + label)
            controls.append(dict(case=label, own_language_native_full_preparation_and_initial_kwargs_exact=True,
                                 six_X12_cream_cheese_reference_record_exact=True if rule['whole_reference_control'] else None))
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
                        assert image.read_bytes() == (out / 'inputs' / rule['scene'] / 'input.png').read_bytes()
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
                    assert metadata['input_png_sha256'] == sha(image), 'Do not relabel an incompatible source image'
                    for key in ('timesteps', 'sigmas'):
                        runtime.require_exact(record[key], sources[195][key], 'entire frozen schedule/' + key)
                    assert hooks(model) == initial_hooks
                    if rule['whole_reference_control']:
                        ref = frozen['whole_physical_references'][label]
                        folder, name = Path(ref['folder']), f'chunk_{query:02d}/states.pt'
                        assert sha(folder / name) == ref['files_current_sha256'][name]
                        assert image.read_bytes() == (folder / f'input_{query:02d}.png').read_bytes()
                        runtime.require_exact(record, torch.load(folder / name, map_location='cpu', weights_only=True, mmap=True), 'ENTIRE accepted X12 cream_cheese feedback/' + label + '/' + str(query))
                        references.append(dict(case=label, query=query, whole_record_exact=True, actual_feedback_PNG_bytes_exact=True))
                    metadata.update(trial=label, query=query, shift_cm=SCENES[rule['scene']], scene_prompt=SCENE_PROMPT,
                        policy_prompt=prompt, policy_goal=rule['goal'], scene=rule['scene'], task_index=7,
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
                    print('[MATCHED-NAMES] ' + json.dumps(dict(requests=len(served), case=label, query=query)), flush=True)
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
        while not failed and len(served) < 192:
            http.handle_request()
            assert not (out / 'closed-loop/failed.json').exists() and not (out / 'wrapper_failed.json').exists()
        assert not failed and len(served) == 192 and cached == 24 and fresh == 168 and pairs == 161
        expected = [(label, q) for label in frozen['trial_order'] if frozen['trials'][label]['whole_reference_control'] for q in range(8)]
        assert [(r['case'], r['query']) for r in references] == expected and len(references) == 48
        assert hooks(model) == initial_hooks
        write(destination / 'whole-reference-query-controls.json', dict(state='exact', queries=references), exclusive=True)
        write(destination / 'complete.json', dict(state='complete', queries=192, saved_q0_predictions=24,
            native_predictions=168, fresh_model_forwards=5040, actual_later_whole_noise_pairs=161,
            six_X12_cream_cheese_controls_all8_entire_records_and_PNGs_exact=True, final_model_hooks_none=True,
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
    TRIALS = {label: (frozen['trials'][label]['scene'], 195) for label in frozen['trial_order']}
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
    references, controllers = scene_inputs(output, frozen, simulator, collector, rollout)
    health = requests.get(URL + '/health', timeout=10)
    health.raise_for_status()
    assert health.text == 'ready'
    stats = json.loads(STATS.read_text())['global_raw']
    lo, hi = np.asarray(stats['q01']), np.asarray(stats['q99'])
    assert lo.shape == hi.shape == (10,) and np.all(hi - lo > 1e-6)
    scale, offset = np.maximum(hi - lo, 1e-8) / 2, (hi + lo) / 2
    destination = output / 'closed-loop'
    provenance = dict(script_sha256=digest(Path(__file__)), accepted_typed_input_roots={scene: dict(input_root=item['input_root'], metadata_sha256=item['metadata_sha256'], complete_sha256=item['complete_sha256']) for scene, item in frozen['scenes'].items()},
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
                    print('[MATCHED-NAMES]', trial, query + 1, records[-1]['grasped'], flush=True)
            assert len(records) == len(states) == 129
            assert len(actions) == len(executed) == len(normalized) == len(denormalized) == 128
            assert len(contracts) == len(list(run.glob('input_*.png'))) == 8
            summary = simulator.summarize_trial(trial, scene, seed, records, rollout)
            rule = frozen['trials'][trial]
            summary.update(prompt=policy_prompt, policy_prompt=policy_prompt, scene_prompt=SCENE_PROMPT, policy_goal=rule['goal'], scene=scene, requested_target=frozen['targets'][rule['goal']], strict_requested_target_selected=summary['selected_objects'] == [frozen['targets'][rule['goal']]], saved_q0_attention_arm=rule['arm'], q0_vision_noise_source_seed=rule['q0_vision_noise_source_seed'], q0_action_noise_source_seed=195, actual_query_seeds=rule['actual_query_seeds'], milk_in_basket_is_scene_predicate_not_changed_policy_success=True)
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
    print('[MATCHED-NAMES] COMPLETE', json.dumps(completed), flush=True)

def simulate(baseline):
    frozen, _, simulator, helper, cross, out = prepared(baseline)
    assert not (out / 'closed-loop').exists(), 'Do not repeat a partial physical batch'
    os.environ.setdefault('MUJOCO_GL', 'egl')
    sys.path.insert(0, '/home/current/work/openpi-demo/cosmos-policy')
    sys.path.insert(0, str(ROOT / 'cosmos-framework'))
    from cosmos_framework.simulation.libero.closed_loop_eval import _framewise_action_to_delta, _remap_gripper
    np = simulator.np
    collector = simulator.load_file('matched_names_execution_collector', ROOT / 'work/collect_cosmos_goal_pair.py')
    rollout = simulator.load_file('matched_names_execution_controller', ROOT / 'work/rollout_cosmos_goal_pair.py')
    stats = read(Path(frozen['threshold_contract']['frozen_files']['normalizer_stats']['path']))['global_raw']
    lo, hi = np.asarray(stats['q01']), np.asarray(stats['q99'])
    scale, offset = np.maximum(hi - lo, 1e-8) / 2, (hi + lo) / 2
    whole_controls, comparisons, case_files = [], [], {}
    for label in frozen['trial_order']:
        execute_trials(out, [label], collector, rollout, simulator, frozen)
        folder, rule = out / 'closed-loop' / label, frozen['trials'][label]
        inputs = out / 'inputs' / rule['scene']
        initial = np.load(inputs / 'state.npy', allow_pickle=False)
        summary, trajectory = read(folder / 'summary.json'), read(folder / 'trajectory.json')
        state = np.load(folder / 'sim_states.npy', allow_pickle=False)
        assert state.shape == (129, initial.size) and state.dtype == initial.dtype and np.isfinite(state).all()
        assert rollout.array_exact(state[0], initial)
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
        comparisons.append(dict(case=label, scene=rule['scene'], shift_cm=SCENES[rule['scene']], goal=rule['goal'], policy_prompt=rule['policy_prompt'], arm=rule['arm'],
            vision_noise_source_seed=rule['q0_vision_noise_source_seed'], selected_objects=summary['selected_objects'],
            strict_requested_target_selected=summary['strict_requested_target_selected'],
            first_selection_window_start=summary['first_selection_step'], first_selected_object=summary['first_selected_object'],
            per_object=summary['per_object'], first_any_finger_contacts=first_contacts,
            first_both_finger_contacts={name: next((i for i, row in enumerate(trajectory) if row['finger_contacts'][name]['both']), None) for name in trajectory[0]['objects']},
            q0_eef_path_xyz_m=[row['eef_xyz'] for row in trajectory[:17]],
            q0_distance_to_objects_m={name: [float(np.linalg.norm(np.asarray(row['eef_xyz']) - np.asarray(row['objects'][name]))) for row in trajectory[:17]] for name in ('milk_1', 'cream_cheese_1')},
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
    assert done['state'] == served['state'] == 'complete' and done['cases'] == 24
    assert done['completed_trials'] == frozen['trial_order'] and len(whole_controls) == 6
    assert served['queries'] == 192 and served['saved_q0_predictions'] == 24 and served['native_predictions'] == 168
    assert served['fresh_model_forwards'] == 5040 and served['actual_later_whole_noise_pairs'] == 161
    assert served['six_X12_cream_cheese_controls_all8_entire_records_and_PNGs_exact'] is served['final_model_hooks_none'] is True
    for name, expected in served['files_sha256'].items():
        assert sha(out / 'server' / name) == expected
    by_case = {row['case']: row for row in comparisons}
    sham_controls = []
    for scene in SCENES:
        for goal in PROMPTS:
            for v in (195, 198):
                native_label, sham_label = (f'{scene}_{goal}_V{v}_A195_{arm}' for arm in ('native', 'all_allowed'))
                a, b = by_case[native_label], by_case[sham_label]
                sham_controls.append(dict(scene=scene, goal=goal, vision_noise_source_seed=v,
                    native_case=native_label, all_allowed_case=sham_label,
                    native_selected_objects=a['selected_objects'], all_allowed_selected_objects=b['selected_objects'],
                    selected_objects_equal=a['selected_objects'] == b['selected_objects'],
                    criterion='Entire strict selected-object list; timing and path remain separate raw descriptions.'))
    assert len(sham_controls) == 8
    sham_gate = all(item['selected_objects_equal'] for item in sham_controls)
    primary_cases = [by_case[label] for label in PRIMARY['cases']]
    assert len(primary_cases) == 4
    both_names_both_V = all(row['selected_objects'] == [TARGETS[row['goal']]] for row in primary_cases)
    prediction = 'supported' if both_names_both_V else 'rejected'
    write(out / 'primary-prediction.json', dict(state='evaluated_all24', **PRIMARY,
        prediction=prediction, numerical_sham_classification_gate_passed=sham_gate,
        all_four_X15_native_cases_strict_only_requested_target=both_names_both_V,
        actual_primary_cases=[dict(case=row['case'], requested_target=TARGETS[row['goal']],
            selected_objects=row['selected_objects'], passed=row['strict_requested_target_selected']) for row in primary_cases],
        rule='All24 and actual packing required; all four X15 native cases pass: supported; otherwise rejected. Sham is a separate cut-attribution diagnostic.',
        failure_does_not_establish_language_ignored=True, changed_prompt_success_not_original_milk_repair=True), exclusive=True)
    paired_descriptions = []
    for arm in ARMS:
        for scene in SCENES:
            for v in (195, 198):
                labels = {goal: f'{scene}_{goal}_V{v}_A195_{arm}' for goal in PROMPTS}
                paired_descriptions.append(dict(scene=scene, arm=arm, vision_noise_source_seed=v, cases=labels,
                    both_names_strict_only_requested_target=all(by_case[label]['strict_requested_target_selected'] for label in labels.values()),
                    status='Full descriptive pair; does not replace the preregistered X15-native primary.'))
    write(out / 'comparisons.json', dict(state='all24_complete', cases=comparisons,
        matched_name_pairs=paired_descriptions, numerical_sham_classification_controls=sham_controls,
        numerical_sham_classification_gate_passed=sham_gate, primary_prediction=prediction,
        all24_executed_without_score_selection=True, original_milk_instruction_repair_claimed=False,
        path_first_contact_strict_grasp_and_basket_are_separate=True, scope=SCOPE), exclusive=True)
    write(out / 'controls.json', dict(six_whole_X12_cream_cheese_controls=whole_controls,
        actual_case_files_sha256=case_files, numerical_sham_classification_controls=sham_controls,
        numerical_sham_classification_gate_passed=sham_gate,
        all24_own_cached_q0_actions_and_all128_official_conversions_exact=True,
        all24_strict_windows_recomputed=True), exclusive=True)
    names = ('controls.json', 'comparisons.json', 'provenance.json', 'prepared.json',
             'primary-preregistered.json', 'primary-prediction.json',
             'server/complete.json', 'closed-loop/complete.json', 'summary.json')
    write(out / 'complete.json', dict(state='complete', script_sha256=sha(Path(__file__)), physical_trials=24,
        trial_order=frozen['trial_order'], saved_q0_predictions=24, native_predictions=168,
        fresh_model_forwards=5040, producer_q0_forwards=720, combined_two_stage_model_forwards=5760,
        actual_later_whole_noise_pairs=161, actual_physical_actions=3072,
        six_entire129_state_5JSON_8PNG_all8_modelrecords_and_controller0_16_controls_exact=True,
        all24_cached_q0_action_and_strict_window_gates_passed=True, all24_executed_without_score_selection=True,
        numerical_sham_classification_gate_passed=sham_gate, primary_prediction=prediction,
        all_four_X15_native_cases_strict_only_requested_target=both_names_both_V,
        files_sha256={name: sha(out / name) for name in names}, scope=SCOPE), exclusive=True)

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
        (out / 'inputs').mkdir(exist_ok=False)
        for scene, item in frozen['scenes'].items():
            (out / 'inputs' / scene).symlink_to(item['folder'], target_is_directory=True)
        write(out / 'provenance.json', frozen, exclusive=True)
        write(out / 'primary-preregistered.json', dict(state='preregistered', **PRIMARY), exclusive=True)
        write(out / 'prepared.json', dict(state='prepared', provenance_sha256=sha(out / 'provenance.json')), exclusive=True)
        print('[MATCHED-NAMES] prepared ' + str(out), flush=True)
        return
    try:
        server(baseline) if args.mode == 'server' else simulate(baseline)
    except Exception as exc:
        if out.is_dir() and not (out / 'wrapper_failed.json').exists() and not (out / 'complete.json').exists():
            write(out / 'wrapper_failed.json', dict(mode=args.mode, error=str(exc), traceback=traceback.format_exc()), exclusive=True)
        raise


if __name__ == '__main__':
    main()
