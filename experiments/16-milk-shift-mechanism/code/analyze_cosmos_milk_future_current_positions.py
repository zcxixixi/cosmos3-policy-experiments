"""CPU audit of the twelve completed X+15/X+18cm frozen first-nine-layer trials.

Require the completed q0 producer and all twelve complete physical trials.
Only create absent BASE/future-current-position-analysis. Check 360 actual
boundary files, 96 whole model records, 1536 official action conversions,
129 contact/lift records per trial and 77 later whole two-draw noise pairs.
Four natives are fresh observations, not repetitions of old X12 trajectories.
The sole primary requires four cut cases selecting only milk AND all four
same-scene/V numerical shams matching their native classifications.

No model construction, forward, attention replay, solver, environment, physics,
training or CUDA initialization. Array RMS ratios are not cm, probabilities,
success rates or semantic anatomy. Only along-X transfer is tested.
"""

import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import time
import traceback


ROOT = Path('/home/current/work/cosmos3')
PROBE_SHA = 'c9cc366e002d8c89ab9c779af35b0191c4ec6d56a157da77499914c6b8605146'
EXECUTION_SHA = '24c237af1c137d4979c358b665f1906ae02acae464140b2a371f4e98789b76f2'
METHODS_SHA = 'ca88ff9c7ee62beef9851fdfaea1c9fddd6e33c0f99058ebee16f26a28a4af9c'
SCENES = {'x15': 15, 'x18': 18}
ARMS = ('native', 'all_allowed', 'early_l1_9')
WINDOWS = {'early_l1_9': dict(steps=[0, 14], layers=[0, 8])}
PLAN = tuple((f'{scene}_V{v}_A195_{arm}', scene, v, arm)
             for arm in ARMS for scene in SCENES for v in (195, 198))
CROSS_PLAN = tuple((scene, arm) for arm in ARMS for scene in SCENES)
FLOOR = 1e-12
SCOPE = ('Post hoc CPU audit of twelve preregistered X+15/X+18cm trials, two fixed V '
         'draws with A195. Four native runs are first observations, not independent '
         'repetitions or old X12 whole-record controls. All eight masked runs share '
         'the full36x30 union66 all-allowed numerical background; four cuts block '
         'future200->current50 only at steps0..14/layers0..8. Only all four cut cases '
         'selecting strictly only milk AND all four shams retaining their own native '
         'classifications support the primary. Native milk preservation is not rescue. '
         'RMS is an array distance, not probability, cm, success rate, functional '
         'specialization or a mediation fraction. Net block updates combine attention '
         'and MLP residuals and BF16 rounding. Hard masks also redistribute attention. '
         'Only current/action boundaries are saved, not future boundaries or full QKV. '
         'No new model, solver, environment, physics, training or general root-cause claim.')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def methods():
    path = ROOT / 'work/analyze_cosmos_milk_future_current_layer_groups.py'
    assert sha(path) == METHODS_SHA
    # Only the frozen CPU numerical/byte/schema helpers are used. Its main,
    # old PLAN, physical-control audit and plots are never executed.
    return load('position_analysis_frozen_cpu_methods', path)


def sealed(baseline):
    path = ROOT / 'work/run_cosmos_milk_future_current_position_execution.py'
    assert sha(path) == EXECUTION_SHA
    adapter = load('position_analysis_frozen_execution', path)
    simulator, frozen, _, helper, output = adapter.prepared(baseline)
    assert adapter.PLAN == PLAN and adapter.WINDOWS == WINDOWS
    assert frozen['producer_script_sha256'] == PROBE_SHA
    protocol = frozen['producer_protocol']
    assert protocol['trial_order'] == [row[0] for row in PLAN]
    assert protocol['primary_cases'] == [row[0] for row in PLAN[8:]]
    assert protocol['actual_model_forwards'] == 360 and protocol['total_official_dispatch_calls'] == 35100
    assert protocol['all_allowed_dispatch_calls'] == 8640 and protocol['cut_dispatch_calls'] == 540
    assert protocol['cut_sites_per_window_case'] == 135 and protocol['masked_sites_per_case'] == 1080
    for name in ('wrapper_failed.json', 'simulator_failed.json', 'server/failed.json', 'closed-loop/failed.json'):
        assert not (output / name).exists()
    complete, controls, primary = (read(output / (name + '.json')) for name in ('complete', 'controls', 'primary-prediction'))
    assert complete['state'] == 'complete' and complete['script_sha256'] == EXECUTION_SHA
    assert complete['trial_order'] == [row[0] for row in PLAN] and complete['physical_trials'] == 12
    assert complete['saved_q0_predictions'] == 12 and complete['native_predictions'] == 84
    assert complete['fresh_model_forwards'] == 2520 and complete['actual_later_whole_noise_pairs'] == 77
    assert complete['producer_q0_forwards_separate'] == 360 and complete['combined_two_stage_model_forwards'] == 2880
    assert complete['all12_own_cached_q0_action_execution_and_strict_window_gates_passed'] is True
    assert complete['all12_cases_executed_without_q0_selection'] is True
    assert complete['native_milk_preservation_is_not_rescue'] is True and complete['functional_specificity_established'] is False
    for name, expected in complete['files_sha256'].items():
        assert sha(output / name) == expected
    server, simulator_complete = read(output / 'server/complete.json'), read(output / 'closed-loop/complete.json')
    assert server['state'] == simulator_complete['state'] == 'complete'
    assert server['queries'] == 96 and server['saved_q0_predictions'] == 12 and server['native_predictions'] == 84
    assert server['fresh_model_forwards'] == 2520 and server['actual_later_whole_noise_pairs'] == 77
    assert server['q0_whole_noise_pair_checks'] == server['extra_q0_model_forwards'] == 0
    assert server['final_model_hooks_none'] is True
    assert server['component_hooks_installed'] is server['attention_dispatch_wrapper_installed'] is False
    for name, expected in server['files_sha256'].items():
        assert sha(output / 'server' / name) == expected
    assert simulator_complete['cases'] == 12 and simulator_complete['steps_per_case'] == 128
    assert simulator_complete['state_records_per_case'] == 129 and simulator_complete['completed_trials'] == [row[0] for row in PLAN]
    assert read(output / 'summary.json') == read(output / 'closed-loop/summary.json')
    assert controls['primary_prediction_sha256'] == sha(output / 'primary-prediction.json')
    assert set(controls['actual_case_files_sha256']) == {row[0] for row in PLAN}
    for label, hashes in controls['actual_case_files_sha256'].items():
        for name, expected in hashes.items():
            assert sha(output / 'closed-loop' / label / name) == expected
    saved = read(output / 'server/saved-q0-controls.json')
    assert saved['state'] == 'exact' and saved['all12_same_fresh_scene_V_native_preparation_gates_passed'] is True
    assert saved['q0_whole_noise_pair_checks'] == saved['extra_q0_model_forwards'] == 0
    assert [row['case'] for row in saved['cases']] == [row[0] for row in PLAN]
    for row, (label, scene, _, arm) in zip(saved['cases'], PLAN):
        assert row['scene'] == scene and row['arm'] == arm and row['fresh_native_record_is_observation_not_old_whole_control'] is True
        assert row['actual_consumed_returned_noise_prepared_clamp_initial_full_kwargs_and_schedule_exact'] is True
        assert row['all30_saved_actual_complete_model_kwargs_solver_casts_and_current_clamps_exact'] is True
        assert row['q0_model_forwards_in_this_execution'] == 0
    return adapter, simulator, frozen, helper, output, complete, controls, primary


def prefix_files(folder, states, trajectory, entry, np):
    """Rebuild the frozen writer's actual first17 serializations from final arrays."""
    hashes = entry['actual_prefix_files_sha256']
    assert set(hashes) == {'sim_states.npy', 'trajectory.json', 'input_01.png', 'initial_controller.npz', 'controller-step16.npz'}
    stream = io.BytesIO()
    np.save(stream, states[:17])
    assert hashlib.sha256(stream.getvalue()).hexdigest() == hashes['sim_states.npy']
    encoded = (json.dumps(trajectory[:17], indent=2, allow_nan=False) + '\n').encode()
    assert hashlib.sha256(encoded).hexdigest() == hashes['trajectory.json']
    for name in ('input_01.png', 'initial_controller.npz', 'controller-step16.npz'):
        assert sha(folder / name) == hashes[name]


def audit_records(baseline, adapter, frozen, helper, output, controls, primary, utils, torch, np):
    exact, npexact, checked, tensor_meta = utils.exact, utils.npexact, utils.checked, utils.tensor_meta
    normal = frozen['threshold_contract']['frozen_files']
    base = load('position_analysis_cpu_record_checks', Path(normal['normal_runtime']['path']))
    runtime = object.__new__(base.NormalRuntime)  # Never construct a pipeline/model.
    runtime.torch = torch
    scan = load('position_analysis_record_schema', Path(frozen['metrics_path']))
    rollout = load('position_analysis_quantify', Path(normal['controller_helper']['path']))
    forward, gripper, converter_sources = utils.converters(torch)
    stats = read(Path(normal['normalizer_stats']['path']))['global_raw']
    lo, hi = np.asarray(stats['q01']), np.asarray(stats['q99'])
    assert lo.shape == hi.shape == (10,) and np.all(hi - lo > 1e-6)
    scale, offset = np.maximum(hi - lo, 1e-8) / 2, (hi + lo) / 2
    initial, originals, natives, native_info = {}, {}, {}, {}
    for scene in SCENES:
        folder = output / 'inputs' / scene
        with np.load(folder / 'controller.npz', allow_pickle=False) as archive:
            controller = {key: archive[key].copy() for key in archive.files}
        initial[scene] = (np.load(folder / 'state.npy', allow_pickle=False), controller)
    for v in (195, 198):
        source = frozen['original_noise_sources'][str(v)]
        originals[v] = checked(Path(source['q0_folder']) / 'states.pt', source['q0_files_current_sha256']['states.pt'], torch)
    for label, _, _, _ in PLAN[:4]:
        source = frozen['q0_sources'][label]
        folder = Path(source['folder'])
        natives[label] = checked(folder / 'states.pt', source['files_sha256']['states.pt'], torch)
        native_info[label] = checked(folder / 'new-input-contract.pt', source['files_sha256']['new-input-contract.pt'], torch)
    records, summaries, audits, later_noise, query_checks = {}, {}, {}, {}, []
    arrays = {key: np.zeros((12, 128, size)) for key, size in (('normalized_actions', 10), ('denormalized_actions', 10),
              ('native_commands', 7), ('executed_commands', 7))}
    object_names, pair_count = None, 0
    entries = {row['case']: row for row in controls['q1_entry_controls']}
    windows = {row['case']: row for row in controls['strict_windows']}
    assert set(entries) == set(windows) == {row[0] for row in PLAN}
    for case, (label, scene, v, arm) in enumerate(PLAN):
        folder, source = output / 'closed-loop' / label, frozen['q0_sources'][label]
        files = controls['actual_case_files_sha256'][label]
        initial_state, initial_controller = initial[scene]
        states = np.load(folder / 'sim_states.npy', allow_pickle=False)
        assert states.shape == (129, initial_state.size) and states.dtype == initial_state.dtype and np.isfinite(states).all()
        assert npexact(states[0], initial_state, np)
        trajectory, summary = read(folder / 'trajectory.json'), read(folder / 'summary.json')
        assert len(trajectory) == 129 and [row['step'] for row in trajectory] == list(range(129))
        assert summary['scene'] == scene and summary['shift_cm'] == SCENES[scene]
        assert summary['saved_q0_attention_arm'] == arm and summary['saved_q0_window'] == WINDOWS.get(arm)
        assert summary['actual_query_seeds'] == list(range(195, 203))
        if object_names is None:
            object_names = tuple(trajectory[0]['objects'])
            arrays['object_xyz'] = np.zeros((12, 129, len(object_names), 3))
            arrays['finger_contacts'] = np.zeros((12, 129, len(object_names), 2), dtype=bool)
            arrays['strict_lift_m'] = np.zeros((12, 129, len(object_names)))
            arrays['strict_contact_and_lift'] = np.zeros((12, 129, len(object_names)), dtype=bool)
        assert tuple(trajectory[0]['objects']) == object_names
        strict = helper.strict_windows(trajectory, summary, np)
        assert strict == read(folder / 'strict-windows.json')
        per_object, selected = rollout.quantify(trajectory)
        assert selected == summary['selected_objects'] and per_object == summary['per_object']
        assert windows[label]['selected_objects'] == selected and windows[label]['scene'] == scene and windows[label]['arm'] == arm
        summaries[label] = summary
        for oi, name in enumerate(object_names):
            xyz = np.asarray([row['objects'][name] for row in trajectory])
            contact = np.asarray([[row['finger_contacts'][name]['left'], row['finger_contacts'][name]['right']] for row in trajectory], dtype=bool)
            assert xyz.shape == (129, 3) and np.isfinite(xyz).all()
            assert (contact[:, 0] & contact[:, 1]).tolist() == [row['finger_contacts'][name]['both'] for row in trajectory]
            lift = xyz[:, 2] - xyz[0, 2]
            flags = contact.all(axis=1) & (lift > .02)
            starts = [i for i in range(125) if bool(flags[i:i + 5].all())]
            assert starts == strict['objects'][name]['five_record_window_start_steps']
            arrays['object_xyz'][case, :, oi], arrays['finger_contacts'][case, :, oi] = xyz, contact
            arrays['strict_lift_m'][case, :, oi], arrays['strict_contact_and_lift'][case, :, oi] = lift, flags
        typed = {}
        for step, filename in ((0, 'initial_controller.npz'), (16, 'controller-step16.npz')):
            with np.load(folder / filename, allow_pickle=False) as archive:
                typed[step] = {key: archive[key].copy() for key in archive.files}
            assert set(typed[step]) == set(initial_controller)
            for key, value in typed[step].items():
                assert value.dtype.kind in 'bfiu' and np.isfinite(value).all()
                assert npexact(np.asarray(trajectory[step]['controller'][key], dtype=value.dtype), value, np)
                if step == 0:
                    assert npexact(value, initial_controller[key], np)
        for row in trajectory:
            template = typed[0 if row['step'] == 0 else 16]
            assert set(row['controller']) == set(template)
            for key, value in row['controller'].items():
                observed = np.asarray(value, dtype=template[key].dtype)
                assert observed.shape == template[key].shape and np.isfinite(observed).all()
        assert read(folder / 'initial_contact_checks.json') == read(output / 'inputs' / scene / 'initial_contact_checks.json')
        assert all(row['exact'] for row in read(folder / 'initial_checks.json'))
        assert all(value is True for value in read(folder / 'initial_controller_checks.json').values())
        entry = read(folder / 'q1-entry-control.json')
        assert entry == entries[label] and entry['scene'] == scene and entry['saved_attention_arm'] == arm
        assert entry['selected_window'] == WINDOWS.get(arm) and entry['actual_prefix_state_records'] == 17
        assert entry['all_16_own_cached_q0_normalized_raw_native_clipped_actions_byte_exact'] is True
        assert entry['own_scene_initial_sim_state_controller_capture_and_contacts_exact'] is True
        assert entry['old_X12_controller_or_trajectory_reference_used'] is False
        prefix_files(folder, states, trajectory, entry, np)
        contracts = read(folder / 'query_contract.json')
        assert len(contracts) == 8
        case_queries = []
        for query in range(8):
            chunk, name = folder / f'chunk_{query:02d}', f'chunk_{query:02d}/states.pt'
            record = checked(chunk / 'states.pt', files[name], torch)
            scan.validate_record(record, torch)
            metadata = read(chunk / 'metadata.json')
            assert metadata == contracts[query]['server_metadata']
            assert contracts[query]['request'] == dict(trial=label, query=query, seed=195 + query, prompt=frozen['producer_protocol']['prompt'])
            assert contracts[query]['paired_noise_checked_exact'] is (query > 0 and case > 0)
            assert contracts[query]['response_saved_actions_exact'] is True
            assert metadata['scene'] == scene and metadata['shift_cm'] == SCENES[scene]
            assert metadata['seed'] == 195 + query and metadata['model_calls'] == 30 and metadata['prepare_calls'] == 1
            assert metadata['current_execution_model_calls'] == (0 if query == 0 else 30)
            assert metadata['saved_q0_prediction'] is (query == 0)
            assert metadata['saved_q0_attention_arm'] == arm and metadata['saved_q0_window'] == WINDOWS.get(arm)
            assert metadata['q0_vision_noise_source_seed'] == v and metadata['q0_action_noise_source_seed'] == 195
            assert metadata['actual_query_seeds'] == list(range(195, 203))
            assert metadata['paired_noise_checked_exact'] is (query > 0 and case > 0)
            assert metadata['input_png_sha256'] == sha(folder / f'input_{query:02d}.png')
            assert metadata['attention_mask_intervention_in_this_execution'] is metadata['component_intervention_in_this_execution'] is False
            assert metadata['initial_noise_replacement_in_this_execution'] is False
            assert len(record['pure_noise']) == 2 and record['pure_noise'][0].shape == (1, 48, 5, 10, 20) and record['pure_noise'][1].shape == (16, 64)
            assert all(value.dtype == torch.float32 and value.device.type == 'cpu' for value in record['pure_noise'])
            for key in ('timesteps', 'sigmas'):
                assert exact(record[key], originals[195][key], torch)
            assert record['action_states'].shape == (31, 1, 16, 64) and record['action_states'].dtype == torch.float32
            assert torch.count_nonzero(record['action_states'][..., 10:]) == 0
            if query == 0:
                native_label = f'{scene}_V{v}_A195_native'
                original = checked(Path(source['folder']) / 'states.pt', source['files_sha256']['states.pt'], torch)
                assert exact(record, original, torch)
                witness = checked(chunk / 'noise-audit.pt', files['chunk_00/noise-audit.pt'], torch)
                adapter.verify_cached(record, witness, natives[native_label], originals, v, runtime, scan)
                info = checked(chunk / 'new-input-contract.pt', files['chunk_00/new-input-contract.pt'], torch)
                assert info['scene'] == scene and info['vision_noise_source_seed'] == v and info['native_reference_case'] == native_label
                assert all(value is True for value in info['actual_input_checks'].values())
                assert exact(info['actual_pack'], witness['model_steps'], torch)
                assert exact(info['actual_pack'], native_info[native_label]['actual_pack'], torch)
                assert exact(info['first_model_kwargs'], record['model_input'], torch)
                assert exact(info['first_model_kwargs'], native_info[native_label]['first_model_kwargs'], torch)
                assert exact(info['actual_clean_current'], native_info[native_label]['actual_clean_current'], torch)
                assert exact(info['original_solver_initial_action_FP32'], record['action_states'][0], torch)
                assert exact(record['prepared_latents_and_masks'][0][0, :, 0].to(torch.bfloat16), info['actual_clean_current'], torch)
                assert metadata['saved_q0_source_states_sha256'] == source['files_sha256']['states.pt']
                assert (folder / 'input_00.png').read_bytes() == (output / 'inputs' / scene / 'input.png').read_bytes()
                for item in source['boundary_files']:
                    assert files['chunk_00/' + item['file']] == item['sha256']
                records[label] = record
            elif query in later_noise:
                assert exact(record['pure_noise'], later_noise[query], torch)
                pair_count += 1
            else:
                later_noise[query] = record['pure_noise']
            norm = np.asarray(read(chunk / 'normalized_actions.json'))
            assert norm.dtype == np.float64 and norm.shape == (16, 10) and np.isfinite(norm).all()
            assert npexact(norm, np.asarray(record['actions'].tolist()), np)
            raw_actions = norm * scale + offset
            native = np.asarray([gripper(forward(row, '6d').tolist(), 'zero_one') for row in raw_actions])
            assert native.shape == (16, 7) and native.dtype == np.float64 and np.isfinite(native).all()
            segment = slice(query * 16, (query + 1) * 16)
            for key, value in (('normalized_actions', norm), ('denormalized_actions', raw_actions), ('native_commands', native), ('executed_commands', np.clip(native, -1, 1))):
                arrays[key][case, segment] = value
            case_queries.append(dict(query=query, states_sha256=files[name], saved_q0_equals_own_producer_entire_record=query == 0,
                actual_raw_two_draw_pair_exact=query > 0 and case > 0,
                pure_noise_tensor_sha256=[tensor_meta(value, torch) for value in record['pure_noise']]))
        for key, filename in (('normalized_actions', 'normalized_actions.json'), ('denormalized_actions', 'denormalized_actions.json'),
                              ('native_commands', 'actions.json'), ('executed_commands', 'executed_actions.json')):
            assert npexact(np.asarray(read(folder / filename)), arrays[key][case], np), ('Actual whole128 converter bytes', label, filename)
        audits[label] = dict(scene=scene, source_files_sha256=files, normalized_raw_native_clipped_all128_bytes_exact=True,
            strict_windows_recomputed_from_actual_contacts_and_positions=True, initial_controller_equal_own_scene_bytes=True,
            typed_controller0_and16_equal_serialized_records=True, all129_serialized_controller_schemas_finite=True,
            q1_prefix_first17_serializations_reconstructed_exact=True, old_X12_whole_trajectory_comparison_performed=False,
            native_independent_repeat_performed=False, queries=case_queries, selected_objects=selected, per_object=per_object,
            strict_window_start_steps={name: row['five_record_window_start_steps'] for name, row in strict['objects'].items()},
            q0_raw_tensor_hashes_computed_here={key: tensor_meta(records[label][key], torch) for key in ('readout', 'action_velocity', 'action_states', 'actions')})
        query_checks.extend(dict(case=label, **row) for row in case_queries)
        print('[FUTURE-CURRENT-POSITION-ANALYSIS] physical audit ' + label, flush=True)
    assert len(query_checks) == 96 and pair_count == 77
    for label, scene, v, _ in PLAN:
        native = records[f'{scene}_V{v}_A195_native']
        assert exact(records[label]['pure_noise'], native['pure_noise'], torch)
        assert exact(records[label]['action_states'][0], originals[195]['action_states'][0], torch)
    for v in (195, 198):
        assert exact(records[f'x15_V{v}_A195_native']['pure_noise'], records[f'x18_V{v}_A195_native']['pure_noise'], torch)
    for scene in SCENES:
        assert exact(records[f'{scene}_V195_A195_native']['pure_noise'][1], records[f'{scene}_V198_A195_native']['pure_noise'][1], torch)
        assert not exact(records[f'{scene}_V195_A195_native']['pure_noise'][0], records[f'{scene}_V198_A195_native']['pure_noise'][0], torch)
    sham_rows = []
    for label, scene, v, arm in PLAN[4:8]:
        native_label = f'{scene}_V{v}_A195_native'
        native_objects, sham_objects = summaries[native_label]['selected_objects'], summaries[label]['selected_objects']
        changed = native_objects != sham_objects
        sham_rows.append(dict(case=label, scene=scene, native_reference_case=native_label,
            native_selected_objects=native_objects, sham_selected_objects=sham_objects, classification_changed=changed,
            original_native_path_interpretation_gate_passed=not changed, numerical_sham_is_native_noop=False))
    assert controls['sham_classification_controls'] == sham_rows
    sham_gate = all(not row['classification_changed'] for row in sham_rows)
    assert controls['original_native_path_interpretation_gate_passed'] is sham_gate
    expected = []
    for label, scene, v, _ in PLAN[8:]:
        native_label, sham_label = f'{scene}_V{v}_A195_native', f'{scene}_V{v}_A195_all_allowed'
        selected = summaries[label]['selected_objects']
        native_objects, sham_objects = summaries[native_label]['selected_objects'], summaries[sham_label]['selected_objects']
        only_milk, same, native_milk = selected == ['milk_1'], native_objects == sham_objects, native_objects == ['milk_1']
        effect = ('sham_classification_gate_failed' if not same else 'did_not_strictly_select_only_milk' if not only_milk
                  else 'preserved_native_milk' if native_milk else 'rescued_failed_native')
        expected.append(dict(case=label, scene=scene, vision_noise_source_seed=v, selected_objects=selected,
            strict_only_milk=only_milk, same_scene_V_native=native_label, same_scene_V_sham=sham_label,
            native_selected_objects=native_objects, sham_selected_objects=sham_objects,
            sham_matches_native_classification=same, native_already_only_milk=native_milk, effect=effect,
            strict_windows_sha256=sha(output / 'closed-loop' / label / 'strict-windows.json')))
    all_only_milk = all(row['strict_only_milk'] for row in expected)
    prediction = 'supported' if all_only_milk and sham_gate else 'rejected'
    assert frozen['primary_cases'] == [row[0] for row in PLAN[8:]] and primary['cases'] == expected
    assert primary['state'] == 'evaluated_after_all12_complete' and primary['primary_window'] == 'early_l1_9'
    assert primary['prediction'] == prediction and primary['all_four_only_milk'] is all_only_milk
    assert primary['numerical_sham_classification_gate_passed'] is sham_gate
    assert primary['all12_cases_executed'] is primary['alternate_window_or_descriptive_metric_cannot_replace_primary'] is True
    assert primary['native_milk_preservation_is_not_rescue'] is True and primary['functional_specificity_established'] is False
    arrays['object_names'] = np.asarray(object_names)
    return records, summaries, arrays, dict(physical_cases=audits, actual_records=96, actual_converted_actions=1536,
        actual_later_whole_two_draw_pairs=77, old_whole_controls_rechecked=0, new_native_first_observations=4,
        native_independent_repeats=0, primary_prediction=prediction, all_four_cut_only_milk=all_only_milk,
        sham_classification_gate_passed=sham_gate, primary_cases=expected, converter_sources=converter_sources)


def propagation(baseline, frozen, records, utils, torch, np):
    source = baseline / 'future-current-positions'
    rows = {row['case']: dict(row) for row in read(source / 'results.json')['cases']}
    components = load('position_analysis_actual_layout', Path(frozen['components_path']))
    arrays, manifests = {}, {row[0]: [] for row in PLAN}
    exact, checked, rms, relative = utils.exact, utils.checked, utils.rms, utils.relative
    for group in ('current', 'action'):
        for metric, size in (('hidden_rms', 37), ('vs_sham_hidden_delta_rms', 37), ('vs_sham_hidden_relative', 37),
                             ('net_update_rms', 36), ('vs_sham_net_update_delta_rms', 36)):
            arrays[metric + '_' + group] = np.zeros((12, 30, size))
        arrays['cross_V_hidden_delta_rms_' + group] = np.zeros((6, 30, 37))
        arrays['cross_V_hidden_relative_' + group] = np.zeros((6, 30, 37))
    for label, _, _, _ in PLAN:
        row = rows[label]
        mapping = checked(source / label / 'indexes-and-masks.pt', row['files_sha256']['indexes-and-masks.pt'], torch)
        index = mapping['indexes']
        for key, expected in (('current_rows', torch.arange(50)), ('action_rows', torch.arange(250, 266)),
                              ('current_keys', torch.arange(121, 171)), ('future_keys', torch.arange(171, 371)),
                              ('action_keys', torch.arange(371, 387)), ('future_rows', torch.arange(50, 250)),
                              ('text_keys', torch.arange(121)), ('union_rows', torch.cat((torch.arange(50), torch.arange(250, 266))))):
            assert exact(index[key], expected, torch)
        assert index['und_len'] == 121 and index['gen_len'] == 266 and index['full_joint_length'] == 387
        allowed = torch.ones((1, 1, 266, 387), dtype=torch.bool)
        assert exact(mapping['masks']['all_allowed']['mask'], allowed, torch)
        allowed[0, 0, index['current_rows'][:, None], index['future_keys'][None, :]] = False
        assert exact(mapping['masks']['cut_future_current']['mask'], allowed, torch)
        row['_mapping'] = mapping
    count, cut_count = 0, 0
    for step in range(30):
        batch, fields = {}, {}
        for label, _, _, _ in PLAN:
            data, manifest = utils.boundary_data(source, rows[label], step, records[label], components, torch)
            batch[label] = data
            fields[label] = {group: data[group + '_hidden'].float().numpy() for group in ('current', 'action')}
            manifests[label].append(manifest)
            count, cut_count = count + 1, cut_count + manifest['active_cut_sites']
        for case, (label, scene, v, arm) in enumerate(PLAN):
            sham_label, native_label = f'{scene}_V{v}_A195_all_allowed', f'{scene}_V{v}_A195_native'
            data = batch[label]
            assert exact(data['actual_pack'], batch[native_label]['actual_pack'], torch)
            if step == 0:
                assert exact(data['actual_model_kwargs'], batch[native_label]['actual_model_kwargs'], torch)
                if arm == 'early_l1_9':
                    assert exact(data['actual_model_kwargs'], batch[sham_label]['actual_model_kwargs'], torch)
                    for group in ('current', 'action'):
                        assert exact(data[group + '_hidden'][:1], batch[sham_label][group + '_hidden'][:1], torch)
            for group in ('current', 'action'):
                value, reference = fields[label][group], fields[sham_label][group]
                delta = value.astype(np.float64) - reference
                update, ref_update = np.diff(value.astype(np.float64), axis=0), np.diff(reference.astype(np.float64), axis=0)
                arrays['hidden_rms_' + group][case, step] = rms(value, (1, 2), np)
                arrays['net_update_rms_' + group][case, step] = rms(update, (1, 2), np)
                arrays['vs_sham_hidden_delta_rms_' + group][case, step] = rms(delta, (1, 2), np)
                arrays['vs_sham_hidden_relative_' + group][case, step] = relative(rms(delta, (1, 2), np), rms(reference, (1, 2), np), np)
                arrays['vs_sham_net_update_delta_rms_' + group][case, step] = rms(update - ref_update, (1, 2), np)
                if v == 198:
                    pair_label = f'{scene}_V195_A195_{arm}'
                    pair = fields[pair_label][group]
                    cross = value.astype(np.float64) - pair
                    ci = CROSS_PLAN.index((scene, arm))
                    arrays['cross_V_hidden_delta_rms_' + group][ci, step] = rms(cross, (1, 2), np)
                    arrays['cross_V_hidden_relative_' + group][ci, step] = relative(rms(cross, (1, 2), np), rms(pair, (1, 2), np), np)
                    if step == 0:
                        assert exact(data[group + '_hidden'][0], batch[pair_label][group + '_hidden'][0], torch)
        print('[FUTURE-CURRENT-POSITION-ANALYSIS] boundaries step ' + str(step), flush=True)
    assert count == 360 and cut_count == 540
    arrays['q0_action_states_FP32'] = np.stack([records[label]['action_states'][:, 0].numpy() for label, _, _, _ in PLAN])
    arrays['q0_action_velocity_BF16_export_FP32'] = np.stack([records[label]['action_velocity'].float().numpy() for label, _, _, _ in PLAN])
    arrays['q0_final_normalized_actions'] = np.stack([records[label]['actions'].numpy() for label, _, _, _ in PLAN])
    arrays['q0_vs_sham_solver_XYZ_rms'], arrays['q0_vs_sham_velocity_XYZ_rms'] = np.zeros((12, 31)), np.zeros((12, 30))
    arrays['q0_vs_sham_full_readout_rms'] = np.zeros((12, 30))
    arrays['q0_vs_sham_final_all10_rms'], arrays['q0_vs_sham_final_XYZ_rms'] = np.zeros(12), np.zeros(12)
    labels = [row[0] for row in PLAN]
    for case, (label, scene, v, arm) in enumerate(PLAN):
        si = labels.index(f'{scene}_V{v}_A195_all_allowed')
        arrays['q0_vs_sham_solver_XYZ_rms'][case] = rms(arrays['q0_action_states_FP32'][case, :, :, :3].astype(np.float64) - arrays['q0_action_states_FP32'][si, :, :, :3], (1, 2), np)
        arrays['q0_vs_sham_velocity_XYZ_rms'][case] = rms(arrays['q0_action_velocity_BF16_export_FP32'][case, :, :, :3].astype(np.float64) - arrays['q0_action_velocity_BF16_export_FP32'][si, :, :, :3], (1, 2), np)
        arrays['q0_vs_sham_full_readout_rms'][case] = rms(records[label]['readout'].float().numpy().astype(np.float64) - records[labels[si]]['readout'].float().numpy(), (1, 2), np)
        delta = arrays['q0_final_normalized_actions'][case].astype(np.float64) - arrays['q0_final_normalized_actions'][si]
        arrays['q0_vs_sham_final_all10_rms'][case], arrays['q0_vs_sham_final_XYZ_rms'][case] = rms(delta, None, np), rms(delta[:, :3], None, np)
        if arm == 'early_l1_9':
            assert exact(records[label]['action_states'][:1], records[labels[si]]['action_states'][:1], torch)
    arrays['sigmas'], arrays['timesteps'] = records[labels[0]]['sigmas'].numpy(), records[labels[0]]['timesteps'].numpy()
    arrays['case_labels'], arrays['case_scenes'] = np.asarray(labels), np.asarray([row[1] for row in PLAN])
    arrays['case_vision_source_seeds'], arrays['case_arms'] = np.asarray([row[2] for row in PLAN]), np.asarray([row[3] for row in PLAN])
    arrays['cross_V_case_scenes'], arrays['cross_V_arms'] = np.asarray([row[0] for row in CROSS_PLAN]), np.asarray([row[1] for row in CROSS_PLAN])
    return arrays, manifests


def plots(out, arrays, summaries, audit, np):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    labels = [row[0] for row in PLAN]
    columns = [(scene, v) for scene in SCENES for v in (195, 198)]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1.6, 1]}, constrained_layout=True)
    values, texts = np.zeros((3, 4)), {}
    for ai, arm in enumerate(ARMS):
        for ci, (scene, v) in enumerate(columns):
            summary = summaries[f'{scene}_V{v}_A195_{arm}']
            objects = summary['selected_objects']
            code = 1 if objects == ['milk_1'] else 2 if objects == ['cream_cheese_1'] else 3 if set(objects) == {'milk_1', 'cream_cheese_1'} else 0 if not objects else 4
            values[ai, ci] = code
            text = 'Only milk' if code == 1 else 'Only cheese' if code == 2 else 'Milk + cheese' if code == 3 else 'No strict selection' if code == 0 else ', '.join(objects)
            first = summary['first_selection_step']
            texts[ai, ci] = text + ('\nfirst step ' + str(first) if first is not None else '')
    axes[0].imshow(values, cmap=ListedColormap(('#d9dce2', '#63b18a', '#e4ad64', '#af94cd', '#8ca1b3')), vmin=-.5, vmax=4.5, aspect='auto')
    axes[0].set_xticks(range(4), [scene.upper() + ' / V' + str(v) for scene, v in columns])
    axes[0].set_yticks(range(3), ['Native first observation', 'Global numerical sham', 'Early layers1..9 — PRIMARY'])
    axes[0].axhline(1.5, color='#273749', lw=2)
    for ai in range(3):
        for ci in range(4):
            axes[0].text(ci, ai, texts[ai, ci], ha='center', va='center', fontsize=10, weight='bold' if ai == 2 else 'normal')
    axes[0].set_title('All twelve real 128-step trials: both-finger contact + lift >2cm for five records')
    colors = ('#176c93', '#b34f40')
    for scene, color in zip(SCENES, colors):
        for v, ls in ((195, '-'), (198, '--')):
            idx = labels.index(f'{scene}_V{v}_A195_early_l1_9')
            axes[1].plot(range(31), arrays['q0_vs_sham_solver_XYZ_rms'][idx], color=color, ls=ls, label=scene.upper() + ' / V' + str(v))
    axes[1].axvline(15, color='#999999', lw=.8, ls=':')
    axes[1].set_xlabel('Actual saved q0 solver sample index 0..30')
    axes[1].set_ylabel('XYZ RMS to own scene/V sham\n(normalized action units)')
    axes[1].legend(fontsize=9, ncol=2)
    axes[1].grid(alpha=.2)
    fig.suptitle('Frozen first-nine-layer transfer: ' + audit['primary_prediction'].upper() + '\nFour cut cases only milk AND all four shams preserve native classifications', fontsize=14, weight='bold')
    fig.savefig(out / 'strict-window-outcomes.png', dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for col, group in enumerate(('current', 'action')):
        for scene, color in zip(SCENES, colors):
            for v, ls in ((195, '-'), (198, '--')):
                idx = labels.index(f'{scene}_V{v}_A195_early_l1_9')
                axes[0, col].plot(range(30), arrays['vs_sham_hidden_relative_' + group][idx, :, 36] * 100,
                    color=color, ls=ls, label=scene.upper() + ' / V' + str(v))
        for ci, (scene, arm) in enumerate(CROSS_PLAN):
            axes[1, col].plot(range(30), arrays['cross_V_hidden_relative_' + group][ci, :, 36] * 100,
                label=scene.upper() + ' / ' + arm, ls=':' if arm == 'native' else '--' if arm == 'all_allowed' else '-')
        axes[0, col].set_title(group.capitalize() + ': cut minus OWN scene/V sham, boundary36')
        axes[1, col].set_title(group.capitalize() + ': V198 minus V195, SAME scene/arm')
        for row in range(2):
            axes[row, col].set_xlabel('Denoising index 0..29 (sigma varies)')
            axes[row, col].set_ylabel('100 × array RMS distance / reference RMS')
            axes[row, col].grid(alpha=.2)
    axes[0, 1].legend(fontsize=8, ncol=2)
    axes[1, 1].legend(fontsize=7, ncol=2)
    fig.suptitle('Actual array distances; ratios are not cm, probabilities or task success', fontsize=13)
    fig.savefig(out / 'same-sham-and-cross-source.png', dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(4, 2, figsize=(12, 12), constrained_layout=True)
    for col, group in enumerate(('current', 'action')):
        values = arrays['vs_sham_net_update_delta_rms_' + group][8:12]
        vmax = max(float(np.max(values)), FLOOR)
        for row, (scene, v) in enumerate(columns):
            image = axes[row, col].imshow(values[row].T, origin='lower', aspect='auto', vmin=0, vmax=vmax, cmap='viridis', interpolation='nearest')
            axes[row, col].set_title(scene.upper() + ' / V' + str(v) + ' / ' + group)
            axes[row, col].set_xlabel('Denoising index 0..29')
            axes[row, col].set_ylabel('Zero-based decoder block 0..35')
            axes[row, col].set_yticks([0, 8, 9, 35])
            axes[row, col].axhline(8.5, color='white', lw=.8, ls='--')
            axes[row, col].axvline(14.5, color='white', lw=.8, ls='--')
        fig.colorbar(image, ax=axes[:, col].tolist(), label='RMS difference of actual net block updates', shrink=.8)
    fig.suptitle('Fixed layers1..9: actual block-update differences to OWN scene/V sham\nH[b+1]−H[b] includes attention + MLP; not a functional brain map', fontsize=13)
    fig.savefig(out / 'primary-block-net-updates.png', dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    baseline = parser.parse_args().output.resolve()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import numpy as np
    import torch
    assert not torch.cuda.is_initialized()
    utils = methods()
    adapter, _, frozen, helper, physical, complete, controls, primary = sealed(baseline)
    out = baseline / 'future-current-position-analysis'
    out.mkdir(exist_ok=False)
    started = time.perf_counter()
    try:
        write(out / 'provenance.json', dict(script_sha256=sha(Path(__file__)), inherited_CPU_methods_sha256=METHODS_SHA,
            producer_script_sha256=PROBE_SHA, execution_script_sha256=EXECUTION_SHA,
            source_execution_complete_sha256=sha(physical / 'complete.json'), source_execution_documents_sha256=complete['files_sha256'],
            producer_documents_sha256=frozen['producer_documents_sha256'], new_scene_input_sources=frozen['scenes'],
            historical_source_evidence=frozen['source_evidence'], device='CPU', scope=SCOPE,
            model_forwards=0, attention_dispatch_calls=0, solver_steps=0, environment_creations=0, physics_calls=0))
        records, summaries, arrays, audit = audit_records(baseline, adapter, frozen, helper, physical, controls, primary, utils, torch, np)
        propagated, manifests = propagation(baseline, frozen, records, utils, torch, np)
        arrays.update(propagated)
        assert complete['primary_prediction'] == audit['primary_prediction']
        assert complete['primary_all_four_only_milk'] is audit['all_four_cut_only_milk']
        assert complete['numerical_sham_classification_gate_passed'] is audit['sham_classification_gate_passed']
        for key, value in arrays.items():
            if value.dtype.kind == 'f' and 'relative' not in key:
                assert np.isfinite(value).all(), key
        np.savez_compressed(out / 'actual-arrays.npz', **arrays)
        assert (out / 'actual-arrays.npz').stat().st_size < 100 * 1024 * 1024
        with np.load(out / 'actual-arrays.npz', allow_pickle=False) as saved:
            assert set(saved.files) == set(arrays)
            for key, value in arrays.items():
                assert saved[key].dtype == value.dtype and saved[key].shape == value.shape and saved[key].tobytes(order='C') == value.tobytes(order='C'), key
            for i, (label, _, _, _) in enumerate(PLAN):
                for key, expected in (('q0_action_states_FP32', records[label]['action_states'][:, 0].numpy()),
                                      ('q0_action_velocity_BF16_export_FP32', records[label]['action_velocity'].float().numpy()),
                                      ('q0_final_normalized_actions', records[label]['actions'].numpy())):
                    assert utils.npexact(saved[key][i], expected, np), ('Actual PT-to-NPZ export bytes', label, key)
        write(out / 'audit.json', audit)
        write(out / 'actual-boundary-manifest.json', manifests)
        plots(out, arrays, summaries, audit, np)
        analysis = dict(scope=SCOPE, case_order=[row[0] for row in PLAN], scenes=SCENES,
            primary_prediction=audit['primary_prediction'], primary_cases=audit['primary_cases'],
            primary_requires_all_four_strict_only_milk_AND_all_four_shams_preserve_native=True,
            selected_steps_zero_based_inclusive=[0, 14], selected_layers_zero_based_inclusive=[0, 8],
            q0_total_official_dispatch_calls=35100, q0_extra_official_dispatch_calls=9180,
            active_cut_sites_checked=540, active_cut_sites_per_cut_case=135, all_allowed_dispatch_calls=8640,
            physical_selected_objects={label: row['selected_objects'] for label, row in summaries.items()},
            definitions=dict(boundaries='0=actual decoder input;1..36=actual block outputs after attention+MLP residuals;36 precedes final norm.',
                net_update='H[b+1]−H[b] computed in float64 from lossless BF16→FP32 exports; not a separate attention or MLP increment.',
                own_reference='Each case minus its SAME scene AND V global all-allowed sham on its own dynamic trajectory.',
                cross_V_reference='V198−V195 within the SAME scene/arm; denominator V195 same scene/arm. No cross-position hidden-distance conflation.',
                relative='100×RMS(delta)/RMS(reference); null/NaN if reference<=1e-12; not cm, probability, success or mediation.',
                solver='Actual saved FP32 action_states, not reconstructed from BF16 inputs; XYZ first3 normalized action dimensions.',
                actual_input='Every saved BF16 action input equals the actual FP32 solver sample cast, and current frame0 is clamped.'),
            gap_floor=FLOOR, invalid_relative_denominators={key: int(np.count_nonzero(~np.isfinite(value))) for key, value in arrays.items() if 'relative' in key},
            q0_final_action_distances={label: dict(XYZ_RMS_to_own_scene_V_sham=float(arrays['q0_vs_sham_final_XYZ_rms'][i]),
                all10_RMS_to_own_scene_V_sham=float(arrays['q0_vs_sham_final_all10_rms'][i])) for i, (label, _, _, _) in enumerate(PLAN)},
            actual_arrays={key: dict(shape=list(value.shape), dtype=str(value.dtype)) for key, value in arrays.items()},
            site_gate_limit='Frozen live dispatch/mask/footprint/QKV/RNG reports checked with source SHA. Full QKV was not saved; no independent attention recomputation.',
            controller_limit='Typed actual arrays at0/16; all129 serialized controller fields checked. Intermediate original dtypes unavailable.',
            native_reference_limit='Four fresh observations; zero independent native repeats and zero old X12 whole-trajectory comparisons.',
            functional_specificity_established=False, native_milk_preservation_is_not_rescue=True,
            actual_physical_actions_checked=1536, whole_noise_pairs_checked=77, whole_model_records_checked=96,
            actual_boundary_files_checked=360, new_native_first_observations=4, old_whole_controls_rechecked=0,
            model_forwards=0, attention_dispatch_calls=0, solver_steps=0, environment_creations=0, physics_calls=0,
            cuda_initialized=torch.cuda.is_initialized())
        assert not torch.cuda.is_initialized()
        write(out / 'analysis.json', analysis)
        names = ('provenance.json', 'audit.json', 'actual-boundary-manifest.json', 'actual-arrays.npz', 'analysis.json',
                 'strict-window-outcomes.png', 'same-sham-and-cross-source.png', 'primary-block-net-updates.png')
        write(out / 'complete.json', dict(state='complete', script_sha256=sha(Path(__file__)), files_sha256={name: sha(out / name) for name in names},
            primary_prediction=audit['primary_prediction'], all_four_cut_only_milk=audit['all_four_cut_only_milk'],
            numerical_sham_classification_gate_passed=audit['sham_classification_gate_passed'], actual_physical_actions_checked=1536,
            whole_noise_pairs_checked=77, whole_model_records_checked=96, actual_boundary_files_checked=360,
            active_cut_sites_checked=540, q0_total_official_dispatch_calls=35100,
            new_native_first_observations=4, native_independent_repeats=0, old_whole_controls_rechecked=0,
            model_forwards=0, attention_dispatch_calls=0, solver_steps=0, environment_creations=0, physics_calls=0,
            cuda_initialized=False, elapsed_s=time.perf_counter() - started, scope=SCOPE))
        print('[FUTURE-CURRENT-POSITION-ANALYSIS] complete ' + json.dumps(dict(primary_prediction=audit['primary_prediction'], elapsed_s=time.perf_counter() - started)), flush=True)
    except Exception as exc:
        write(out / 'failed.json', dict(error=str(exc), traceback=traceback.format_exc(), scope=SCOPE))
        raise


if __name__ == '__main__':
    main()
