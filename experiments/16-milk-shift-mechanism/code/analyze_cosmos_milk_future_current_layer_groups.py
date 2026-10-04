"""CPU audit and plots of twelve completed early nine-layer-group trials.

Use --output BASE; exclusively create BASE/future-current-layer-analysis. Require
the completed twelve q0 producer and all twelve complete physical trials.
Independently check saved arrays, four whole native/sham controls, 1536 actual
action conversions, strict contact/lift windows, 77 later whole noise pairs,
and all360 actual decoder-boundary files before plotting. No model creation,
forward, attention replay, solver update, environment or CUDA initialization.

Only layers1..9 BOTH strictly selecting only milk test the new primary; no
other group can replace it. The previous eighteen-layer primary remains
rejected; its sealed results are references, not additive contributions.
Current50/action16 are real packed token populations,
not language/motor regions. Adjacent boundaries measure the combined net
attention+MLP block update in decoder coordinates. Array RMS distances are
not grasp probabilities, information amounts or a causal mediation share.
"""

import argparse
import ast
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback


ROOT = Path('/home/current/work/cosmos3')
PROBE_SHA = 'b5af0c4c63dbeee37c81a783c8c7f1f73146a6c9c679c0130899756b7f835689'
EXECUTION_SHA = 'e6d5abcb19d866f372d1373ba97a383f8093f1917859ff9714451f8e6138d40f'
INHERITED_ANALYZER_SHA = '04758f1ea7768f6bdd725b328787b391b4b12dbc08a32626768d9110fdfcc609'
PREVIOUS_PROBE_SHA = '8fef06cae12291884ee8a8fa60c981605f7813ccaa1c372f76f426be9662eb69'
PREVIOUS_EXECUTION_SHA = '19b907e5760f5873f4b1d0b834018a594f8d7420827369e731bc701ae4bcb08a'
CONVERTER_SHA = '1a2de0ae18b59de6455c4dca650f6db20de9beb9c39f8efa2b77f0913c2ba4be'
POSE_SHA = '73daa21bc5d66d94ee9217b6065b50ab81ee644d07067bb360326e4edf490380'
WINDOWS = {'early_l1_9': dict(steps=[0, 14], layers=[0, 8]),
           'early_l10_18': dict(steps=[0, 14], layers=[9, 17]),
           'early_l19_27': dict(steps=[0, 14], layers=[18, 26]),
           'early_l28_36': dict(steps=[0, 14], layers=[27, 35])}
ARMS = ('native', 'all_allowed', *WINDOWS)
PLAN = tuple((f'V{v}_A195_{arm}', v, arm) for arm in ARMS for v in (195, 198))
FLOOR = 1e-12
SCOPE = ('Post hoc CPU audit of completed fixed-A195, V195/V198, X+12cm early nine-layer-group experiments. '
         'Every physical outcome is reported; only the preregistered layers1..9 pair tests the new primary. '
         'The rejected eighteen-layer primary is retained as a sealed reference; effects are not added. '
         'Higher groups are equal-site-count layer comparisons, not irrelevant semantic controls. '
         'All ten masked arms use the same full36x30 union66 all-allowed arithmetic background. '
         'RMS is a saved-array distance, not function, language/motor anatomy, object information, '
         'probability or success percentage. Decoder block net updates combine attention and MLP '
         'residual additions and BF16 rounding. A hard mask also redistributes remaining attention. '
         'Current/action boundaries are available, future hidden boundaries and full QKV are not. '
         'No new model, solver, physics, training or general target/identity/root-cause experiment.')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    with path.open('x') as stream:
        stream.write(json.dumps(data, indent=2, allow_nan=False) + '\n')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def raw(value, torch):
    return value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def exact(a, b, torch):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype and raw(a, torch) == raw(b, torch)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(exact(a[key], b[key], torch) for key in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(exact(x, y, torch) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def npexact(a, b, np):
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b) and a.tobytes(order='C') == b.tobytes(order='C')


def tensor_meta(value, torch):
    return dict(shape=list(value.shape), dtype=str(value.dtype), sha256=hashlib.sha256(raw(value, torch)).hexdigest())


def checked(path, expected, torch):
    assert sha(path) == expected, ('Source file SHA', str(path))
    return torch.load(path, map_location='cpu', weights_only=True, mmap=True)


def rms(value, axes, np):
    return np.sqrt(np.mean(np.square(value, dtype=np.float64), axis=axes, dtype=np.float64))


def relative(delta, reference, np):
    result = np.full_like(delta, np.nan, dtype=np.float64)
    np.divide(delta, reference, out=result, where=reference > FLOOR)
    return result


def converters(torch):
    # Resolve the real installed module and its directly imported pose_utils
    # from actual source; do not guess a rotation-helper path or reimplement it.
    sys.path.insert(0, '/home/current/work/openpi-demo/cosmos-policy')
    sys.path.insert(0, str(ROOT / 'cosmos-framework'))
    name = 'cosmos_framework.simulation.libero.closed_loop_eval'
    spec = importlib.util.find_spec(name)
    path = Path(spec.origin)
    assert sha(path) == CONVERTER_SHA and not torch.cuda.is_initialized()
    dependencies = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            dependencies.update(alias.name for alias in node.names if alias.name.split('.')[-1] == 'pose_utils')
        elif isinstance(node, ast.ImportFrom):
            module = importlib.util.resolve_name('.' * node.level + (node.module or ''), name.rsplit('.', 1)[0]) if node.level else node.module
            if module and module.split('.')[-1] == 'pose_utils':
                dependencies.add(module)
            elif module:
                dependencies.update(module + '.' + alias.name for alias in node.names if alias.name == 'pose_utils')
    assert len(dependencies) == 1, 'Expected one actually imported pose_utils dependency'
    pose_name = next(iter(dependencies))
    pose_path = Path(importlib.util.find_spec(pose_name).origin)
    assert sha(pose_path) == POSE_SHA and not torch.cuda.is_initialized()
    module = importlib.import_module(name)
    forward, gripper = module._framewise_action_to_delta, module._remap_gripper
    assert Path(inspect.getfile(forward)) == path and Path(inspect.getfile(gripper)) == path
    assert not torch.cuda.is_initialized()
    return forward, gripper, dict(converter_path=str(path), converter_sha256=CONVERTER_SHA,
        pose_utils_path=str(pose_path), pose_utils_sha256=POSE_SHA,
        function_sha256={func.__name__: hashlib.sha256(inspect.getsource(func).encode()).hexdigest() for func in (forward, gripper)},
        convention='Actual np.asarray(JSON) float64; quantile affine inverse; official row6d converter; official zero_one gripper; clip[-1,1]. No TorchFP32 reconversion.')


def sealed(baseline):
    path = ROOT / 'work/run_cosmos_milk_future_current_layer_execution.py'
    assert sha(path) == EXECUTION_SHA
    adapter = load('future_current_layer_analysis_execution', path)
    _, frozen, cross, helper, output = adapter.prepared(baseline)
    assert adapter.PLAN == PLAN and adapter.WINDOWS == WINDOWS and frozen['producer_script_sha256'] == PROBE_SHA
    protocol = frozen['producer_protocol']
    assert protocol['primary_window'] == 'early_l1_9' and protocol['primary_cases'] == [PLAN[4][0], PLAN[5][0]]
    assert protocol['model_forwards'] == 360 and protocol['total_official_dispatch_calls'] == 37800
    assert protocol['extra_all_allowed_dispatch_calls'] == 10800 and protocol['extra_cut_dispatch_calls'] == 1080
    assert protocol['cut_sites_per_window_case'] == 135 and protocol['masked_sites_per_case'] == 1080
    for name in ('wrapper_failed.json', 'simulator_failed.json', 'server/failed.json', 'closed-loop/failed.json'):
        assert not (output / name).exists()
    complete, controls, primary = (read(output / (name + '.json')) for name in ('complete', 'controls', 'primary-prediction'))
    assert complete['state'] == 'complete' and complete['script_sha256'] == EXECUTION_SHA
    assert complete['trial_order'] == [item[0] for item in PLAN] and complete['physical_trials'] == 12
    assert complete['saved_q0_predictions'] == 12 and complete['native_predictions'] == 84 and complete['fresh_model_forwards'] == 2520
    assert complete['producer_q0_forwards_separate'] == 360 and complete['combined_two_stage_model_forwards'] == 2880
    assert complete['actual_later_whole_noise_pairs'] == 77
    for key in ('four_native_and_sham_entire_129_state_5JSON_8PNG_and_8_modelrecords_exact',
                'all12_own_cached_q0_action_execution_and_strict_window_gates_passed', 'all12_cases_executed_without_q0_selection'):
        assert complete[key] is True
    for name, expected in complete['files_sha256'].items():
        assert sha(output / name) == expected
    server, simulator = read(output / 'server/complete.json'), read(output / 'closed-loop/complete.json')
    assert server['state'] == simulator['state'] == 'complete' and server['queries'] == 96
    assert server['saved_q0_predictions'] == 12 and server['native_predictions'] == 84 and server['fresh_model_forwards'] == 2520
    assert server['actual_later_whole_noise_pairs'] == 77 and server['q0_whole_noise_pair_checks'] == server['extra_q0_model_forwards'] == 0
    assert server['four_controls_all_eight_query_records_entire_exact'] is True
    assert server['component_hooks_installed'] is server['attention_dispatch_wrapper_installed'] is False
    for name, expected in server['files_sha256'].items():
        assert sha(output / 'server' / name) == expected
    assert simulator['cases'] == 12 and simulator['steps_per_case'] == 128 and simulator['state_records_per_case'] == 129
    assert simulator['completed_trials'] == [item[0] for item in PLAN]
    assert read(output / 'summary.json') == read(output / 'closed-loop/summary.json')
    assert controls['primary_prediction_sha256'] == sha(output / 'primary-prediction.json')
    assert set(controls['actual_case_files_sha256']) == {item[0] for item in PLAN}
    for label, files in controls['actual_case_files_sha256'].items():
        for name, expected in files.items():
            assert sha(output / 'closed-loop' / label / name) == expected
    return adapter, frozen, cross, helper, output, complete, controls, primary


def audit_records(baseline, frozen, helper, output, controls, primary, torch, np):
    normal = frozen['threshold_contract']['frozen_files']
    base = load('future_current_layer_analysis_normal_checks', Path(normal['normal_runtime']['path']))
    runtime = object.__new__(base.NormalRuntime)  # Only pure CPU byte-check methods; never __init__/make_runtime.
    runtime.torch = torch
    scan = load('future_current_layer_analysis_record_checks', Path(frozen['metrics_path']))
    rollout = load('future_current_layer_analysis_quantify', Path(normal['controller_helper']['path']))
    forward, gripper, converter_sources = converters(torch)
    stats = read(Path(normal['normalizer_stats']['path']))['global_raw']
    lo, hi = np.asarray(stats['q01']), np.asarray(stats['q99'])
    scale, offset = np.maximum(hi - lo, 1e-8) / 2, (hi + lo) / 2
    initial_state = np.load(output / 'inputs/x12/state.npy', allow_pickle=False)
    with np.load(output / 'inputs/x12/controller.npz', allow_pickle=False) as data:
        initial_controller = {key: data[key].copy() for key in data.files}
    originals, native_pair = {}, {}
    for v in (195, 198):
        source = frozen['producer_source_cases'][str(v)]
        originals[v] = checked(Path(source['q0_folder']) / 'states.pt', source['q0_files_current_sha256']['states.pt'], torch)
        source = frozen['native_q0_source_pair'][str(v)]
        native_pair[v] = checked(Path(source['folder']) / 'states.pt', source['files_sha256']['states.pt'], torch)
    records, summaries, audit, later_noise, pair_count, object_names = {}, {}, {}, {}, 0, None
    arrays = dict(normalized_actions=np.zeros((12, 128, 10)), denormalized_actions=np.zeros((12, 128, 10)),
        native_commands=np.zeros((12, 128, 7)), executed_commands=np.zeros((12, 128, 7)))
    whole_checks, query_checks = [], []
    for case, (label, v, arm) in enumerate(PLAN):
        folder, source = output / 'closed-loop' / label, frozen['q0_sources'][label]
        files = controls['actual_case_files_sha256'][label]
        states = np.load(folder / 'sim_states.npy', allow_pickle=False)
        assert states.shape == (129, initial_state.size) and states.dtype == initial_state.dtype and np.isfinite(states).all()
        assert npexact(states[0], initial_state, np)
        trajectory, summary = read(folder / 'trajectory.json'), read(folder / 'summary.json')
        assert len(trajectory) == 129 and [item['step'] for item in trajectory] == list(range(129))
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
        summaries[label] = summary
        for oi, name in enumerate(object_names):
            xyz = np.asarray([row['objects'][name] for row in trajectory])
            contact = np.asarray([[row['finger_contacts'][name]['left'], row['finger_contacts'][name]['right']] for row in trajectory], dtype=bool)
            assert xyz.shape == (129, 3) and np.isfinite(xyz).all()
            assert (contact[:, 0] & contact[:, 1]).tolist() == [row['finger_contacts'][name]['both'] for row in trajectory]
            lift = xyz[:, 2] - xyz[0, 2]
            actual_strict = contact.all(axis=1) & (lift > .02)
            starts = [i for i in range(125) if bool(actual_strict[i:i + 5].all())]
            assert starts == strict['objects'][name]['five_record_window_start_steps']
            arrays['object_xyz'][case, :, oi], arrays['finger_contacts'][case, :, oi] = xyz, contact
            arrays['strict_lift_m'][case, :, oi], arrays['strict_contact_and_lift'][case, :, oi] = lift, actual_strict
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
            assert metadata['seed'] == 195 + query and metadata['model_calls'] == 30 and metadata['prepare_calls'] == 1
            assert metadata['current_execution_model_calls'] == (0 if query == 0 else 30)
            assert metadata['saved_q0_prediction'] is (query == 0) and metadata['whole_prior_control'] is (case < 4)
            assert metadata['input_png_sha256'] == sha(folder / f'input_{query:02d}.png')
            assert metadata['attention_mask_intervention_in_this_execution'] is metadata['component_intervention_in_this_execution'] is False
            assert metadata['initial_noise_replacement_in_this_execution'] is False
            assert len(record['pure_noise']) == 2 and record['pure_noise'][0].shape == (1, 48, 5, 10, 20) and record['pure_noise'][1].shape == (16, 64)
            assert all(value.dtype == torch.float32 and value.device.type == 'cpu' for value in record['pure_noise'])
            for key in ('timesteps', 'sigmas'):
                assert exact(record[key], originals[195][key], torch)
            assert record['action_states'].dtype == torch.float32 and record['action_states'].shape == (31, 1, 16, 64)
            assert torch.count_nonzero(record['action_states'][..., 10:]) == 0
            if query == 0:
                original = checked(Path(source['folder']) / 'states.pt', source['files_sha256']['states.pt'], torch)
                assert exact(record, original, torch)
                witness = checked(chunk / 'noise-audit.pt', files['chunk_00/noise-audit.pt'], torch)
                helper.verify_cached(record, witness, originals, native_pair, arm, v, runtime, scan)
                records[label] = record
            elif query in later_noise:
                assert exact(record['pure_noise'], later_noise[query], torch)
                pair_count += 1
            else:
                later_noise[query] = record['pure_noise']
            if case < 4:
                reference = frozen['whole_physical_references'][label]
                path = Path(reference['folder']) / name
                prior = checked(path, reference['files_current_sha256'][name], torch)
                assert exact(record, prior, torch)
                assert (folder / f'input_{query:02d}.png').read_bytes() == (Path(reference['folder']) / f'input_{query:02d}.png').read_bytes()
            norm = np.asarray(read(chunk / 'normalized_actions.json'))
            assert norm.dtype == np.float64 and norm.shape == (16, 10) and np.isfinite(norm).all()
            assert npexact(norm, np.asarray(record['actions'].tolist()), np)
            raw_actions = norm * scale + offset
            native = np.asarray([gripper(forward(item, '6d').tolist(), 'zero_one') for item in raw_actions])
            assert native.shape == (16, 7) and native.dtype == np.float64 and np.isfinite(native).all()
            segment = slice(query * 16, (query + 1) * 16)
            for key, value in (('normalized_actions', norm), ('denormalized_actions', raw_actions), ('native_commands', native), ('executed_commands', np.clip(native, -1, 1))):
                arrays[key][case, segment] = value
            case_queries.append(dict(query=query, states_sha256=files[name], whole_prior_record_exact=case < 4,
                actual_raw_two_draw_pair_exact=query > 0 and case > 0,
                pure_noise_tensor_sha256=[tensor_meta(value, torch) for value in record['pure_noise']]))
        for key, filename in (('normalized_actions', 'normalized_actions.json'), ('denormalized_actions', 'denormalized_actions.json'),
                              ('native_commands', 'actions.json'), ('executed_commands', 'executed_actions.json')):
            assert npexact(np.asarray(read(folder / filename)), arrays[key][case], np), ('Actual whole128 converter bytes', label, filename)
        if case < 4:
            ref = frozen['whole_physical_references'][label]
            old = Path(ref['folder'])
            assert npexact(states, np.load(old / 'sim_states.npy', allow_pickle=False), np)
            for name in (*adapter_physical(), *[f'input_{q:02d}.png' for q in range(8)]):
                assert (folder / name).read_bytes() == (old / name).read_bytes()
            with np.load(old / 'controller-step16.npz', allow_pickle=False) as old_controller:
                assert set(typed[16]) == set(old_controller.files) and all(npexact(value, old_controller[key], np) for key, value in typed[16].items())
            whole_checks.append(dict(case=label, all129_state_and_5JSON8PNG_and_8whole_records_and_controller16_exact=True))
        audit[label] = dict(source_files_sha256=files, normalized_raw_native_clipped_all128_bytes_exact=True,
            strict_windows_recomputed_from_actual_contacts_and_positions=True, initial_controller_common_bytes_exact=True,
            actual_typed_controller0_and16_equal_serialized_records=True, all129_serialized_controller_schemas_finite=True,
            controller_limit='Original typed arrays exist at0/16. Intermediate JSON schemas/values are checked, without claiming unavailable original dtypes.',
            queries=case_queries, q0_raw_tensor_hashes_computed_here={key: tensor_meta(records[label][key], torch)
                for key in ('readout', 'action_velocity', 'action_states', 'actions')}, selected_objects=summary['selected_objects'], per_object=per_object,
            strict_window_start_steps={name: item['five_record_window_start_steps'] for name, item in strict['objects'].items()})
        query_checks.extend(dict(case=label, **item) for item in case_queries)
        print('[FUTURE-CURRENT-LAYER-ANALYSIS] physical audit ' + label, flush=True)
    assert len(query_checks) == 96 and pair_count == 77 and len(whole_checks) == 4
    for label, v, arm in PLAN:
        assert exact(records[label]['pure_noise'], records[f'V{v}_A195_native']['pure_noise'], torch)
        assert exact(records[label]['action_states'][0], records[PLAN[0][0]]['action_states'][0], torch)
    assert exact(records['V195_A195_native']['pure_noise'][1], records['V198_A195_native']['pure_noise'][1], torch)
    primary_cases = frozen['primary_cases']
    assert primary_cases == [PLAN[4][0], PLAN[5][0]] and primary['primary_window'] == 'early_l1_9'
    expected_primary = [dict(case=label, selected_objects=summaries[label]['selected_objects'],
        strict_only_milk=summaries[label]['selected_objects'] == ['milk_1'], strict_windows_sha256=sha(output / 'closed-loop' / label / 'strict-windows.json')) for label in primary_cases]
    supported = all(row['strict_only_milk'] for row in expected_primary)
    assert primary['state'] == 'evaluated_after_all12_complete' and primary['cases'] == expected_primary
    assert primary['prediction'] == ('supported' if supported else 'rejected') and primary['both_only_milk'] is supported
    assert primary['all12_cases_executed'] is primary['other_windows_cannot_replace_primary'] is primary['descriptive_milk_inclusion_or_max_lift_cannot_replace_primary'] is True
    sham_passed = all(set(summaries[f'V{v}_A195_native']['selected_objects']) == set(summaries[f'V{v}_A195_all_allowed']['selected_objects']) for v in (195, 198))
    assert controls['original_native_path_interpretation_gate_passed'] is sham_passed
    arrays['object_names'] = np.asarray(object_names)
    return records, summaries, arrays, dict(physical_cases=audit, four_whole_controls=whole_checks, actual_records=96,
        actual_converted_actions=1536, actual_later_whole_two_draw_pairs=77, primary_prediction=primary['prediction'],
        sham_classification_gate_passed=sham_passed, converter_sources=converter_sources)


def adapter_physical():
    return ('trajectory.json', 'actions.json', 'executed_actions.json', 'normalized_actions.json', 'denormalized_actions.json')


def boundary_data(source, row, step, record, components, torch):
    item = row['boundary_files'][step]
    data = checked(source / row['case'] / item['file'], item['sha256'], torch)
    assert data['step'] == step and len(data['site_events']) == 36 and data['case_arm'] == row['arm']
    kwargs, index = data['actual_model_kwargs'], row['_mapping']['indexes']
    assert exact(data['actual_model_action_timesteps'], kwargs['action_timesteps'], torch)
    assert exact(data['solver_sigma_from_same_frozen_schedule'], record['sigmas'][step], torch)
    assert exact(kwargs['action_tokens'][0], record['action_states'][step, 0].to(torch.bfloat16), torch)
    assert kwargs['vision_tokens'][0].shape == (1, 48, 5, 10, 20) and kwargs['action_tokens'][0].shape == (16, 64)
    assert exact(kwargs['vision_tokens'][0][0, :, 0], record['model_input']['vision_tokens'][0][0, :, 0], torch)
    layout = components._layout(kwargs)
    assert exact(layout['action_rows'], index['action_rows'], torch)
    assert exact(kwargs['vision_sequence_indexes'].flatten(), torch.arange(121, 371), torch)
    actual_pack = dict(layout=layout, metadata={key: kwargs.get(key) for key in components._STRUCTURAL_KEYS})
    assert exact(data['actual_pack'], actual_pack, torch)
    if step == 0:
        assert exact(kwargs, record['model_input'], torch)
    output = data['model_output']
    assert type(output) is tuple and len(output) == 3 and type(output[2]) is list and len(output[2]) == 1
    assert exact(output[2][0], record['action_velocity'][step], torch)
    for key, count in (('current_hidden', 50), ('action_hidden', 16)):
        value = data[key]
        assert value.shape == (37, count, 4096) and value.dtype == torch.bfloat16 and value.device.type == 'cpu'
        assert bool(torch.isfinite(value).all())
    window = WINDOWS.get(row['arm'])
    assert data['selected_window'] == window
    active_sites = 0
    for layer, event in enumerate(data['site_events']):
        active = window is not None and window['steps'][0] <= step <= window['steps'][1] and window['layers'][0] <= layer <= window['layers'][1]
        arm = 'native' if row['arm'] == 'native' else 'cut_future_current' if active else 'all_allowed'
        assert event['step'] == step and event['layer_zero_based'] == layer and event['arm'] == arm
        assert event['selected_window_cut_active'] is active and event['window_name'] == row['arm']
        assert event['extra_cut_calls'] == int(active) and event['extra_all_allowed_calls'] == int(row['arm'] != 'native')
        assert event['to_add_out_shape'] == [266, 4096] and event['to_add_out_actual_calls'] == 1
        assert event['actual_dispatch_return_flatten_equals_pre_W_byte_exact'] is event['UND_dispatch_return_unmodified'] is True
        if row['arm'] != 'native':
            assert event['substituted_query_rows'] == 66 and event['future200_native_bytes_preserved'] is True
            assert event['mask'] == tensor_meta(row['_mapping']['masks'][arm]['mask'], torch)
            assert event['all_allowed_mask'] == tensor_meta(row['_mapping']['masks']['all_allowed']['mask'], torch)
            assert event['actual_Q_K_V_bytes_and_global_CPU_CUDA_RNG_unchanged'] is True
            if active:
                assert event['all_unblocked_GEN_rows_vs_same_input_all_allowed_byte_exact'] is True
        active_sites += int(active)
    return data, dict(step=step, source_file_sha256=item['sha256'], active_cut_sites=active_sites,
        actual_boundary_tensor_hashes={key: tensor_meta(data[key], torch) for key in ('current_hidden', 'action_hidden')},
        actual_action_input_tensor_hash=tensor_meta(kwargs['action_tokens'][0], torch), sigma=float(record['sigmas'][step]))


def propagation(baseline, frozen, records, torch, np):
    source = baseline / 'future-current-layer-groups'
    result = read(source / 'results.json')
    rows = {row['case']: dict(row) for row in result['cases']}
    component_path = Path(frozen['threshold_contract']['frozen_files']['components']['path'])
    components = load('future_current_layer_analysis_layout', component_path)
    manifests, arrays, counts = {label: [] for label, _, _ in PLAN}, {}, dict(files=0, cut_sites=0)
    for group in ('current', 'action'):
        for metric, size in (('hidden_rms', 37), ('vs_sham_hidden_delta_rms', 37), ('vs_sham_hidden_relative', 37),
                             ('net_update_rms', 36), ('vs_sham_net_update_delta_rms', 36)):
            arrays[metric + '_' + group] = np.zeros((12, 30, size))
        arrays['cross_V_hidden_delta_rms_' + group] = np.zeros((6, 30, 37))
        arrays['cross_V_hidden_relative_' + group] = np.zeros((6, 30, 37))
    for label, _, _ in PLAN:
        row, folder = rows[label], source / label
        mapping = checked(folder / 'indexes-and-masks.pt', row['files_sha256']['indexes-and-masks.pt'], torch)
        index = mapping['indexes']
        for key, expected in (('current_rows', torch.arange(50)), ('action_rows', torch.arange(250, 266)),
                              ('current_keys', torch.arange(121, 171)), ('future_keys', torch.arange(171, 371)),
                              ('action_keys', torch.arange(371, 387)), ('future_rows', torch.arange(50, 250)),
                              ('text_keys', torch.arange(121)), ('union_rows', torch.cat((torch.arange(50), torch.arange(250, 266))))):
            assert exact(index[key], expected, torch)
        assert index['und_len'] == 121 and index['gen_len'] == 266 and index['full_joint_length'] == 387
        expected_mask = torch.ones((1, 1, 266, 387), dtype=torch.bool)
        assert exact(mapping['masks']['all_allowed']['mask'], expected_mask, torch)
        expected_mask[0, 0, index['current_rows'][:, None], index['future_keys'][None, :]] = False
        assert exact(mapping['masks']['cut_future_current']['mask'], expected_mask, torch)
        rows[label]['_mapping'] = mapping
    for step in range(30):
        batch, fields = {}, {}
        for label, _, _ in PLAN:
            data, manifest = boundary_data(source, rows[label], step, records[label], components, torch)
            batch[label] = data
            fields[label] = {group: data[group + '_hidden'].float().numpy() for group in ('current', 'action')}
            manifests[label].append(manifest)
            counts['files'] += 1
            counts['cut_sites'] += manifest['active_cut_sites']
        for case, (label, v, arm) in enumerate(PLAN):
            sham_label = f'V{v}_A195_all_allowed'
            window = WINDOWS.get(arm)
            if case < 4:
                old = frozen['source_evidence']['q0_controls'][label]
                item = old['boundary_files'][step]
                saved = checked(Path(old['folder']) / item['file'], item['sha256'], torch)
                for key in ('current_hidden', 'action_hidden', 'model_output', 'actual_pack'):
                    assert exact(batch[label][key], saved[key], torch)
            elif step <= window['steps'][0]:
                assert exact(batch[label]['actual_model_kwargs'], batch[sham_label]['actual_model_kwargs'], torch)
                if step < window['steps'][0]:
                    assert exact(batch[label]['model_output'], batch[sham_label]['model_output'], torch)
                n = 37 if step < window['steps'][0] else window['layers'][0] + 1
                for key in ('current_hidden', 'action_hidden'):
                    assert exact(batch[label][key][:n], batch[sham_label][key][:n], torch)
            for group in ('current', 'action'):
                value, reference = fields[label][group], fields[sham_label][group]
                amp, delta = rms(value, (1, 2), np), value.astype(np.float64) - reference
                update, ref_update = np.diff(value.astype(np.float64), axis=0), np.diff(reference.astype(np.float64), axis=0)
                arrays['hidden_rms_' + group][case, step] = amp
                arrays['net_update_rms_' + group][case, step] = rms(update, (1, 2), np)
                arrays['vs_sham_hidden_delta_rms_' + group][case, step] = rms(delta, (1, 2), np)
                arrays['vs_sham_hidden_relative_' + group][case, step] = relative(rms(delta, (1, 2), np), rms(reference, (1, 2), np), np)
                arrays['vs_sham_net_update_delta_rms_' + group][case, step] = rms(update - ref_update, (1, 2), np)
                if v == 198:
                    pair = fields[f'V195_A195_{arm}'][group]
                    cross = value.astype(np.float64) - pair
                    arrays['cross_V_hidden_delta_rms_' + group][case // 2, step] = rms(cross, (1, 2), np)
                    arrays['cross_V_hidden_relative_' + group][case // 2, step] = relative(rms(cross, (1, 2), np), rms(pair, (1, 2), np), np)
                    if step == 0:
                        assert exact(batch[label][group + '_hidden'][0], batch[f'V195_A195_{arm}'][group + '_hidden'][0], torch)
        print('[FUTURE-CURRENT-LAYER-ANALYSIS] boundaries step ' + str(step), flush=True)
    assert counts == dict(files=360, cut_sites=1080)
    for case, (label, v, arm) in enumerate(PLAN):
        record, sham = records[label], records[f'V{v}_A195_all_allowed']
        if arm in WINDOWS:
            first = WINDOWS[arm]['steps'][0]
            for key in ('readout', 'action_velocity'):
                assert exact(record[key][:first], sham[key][:first], torch)
            assert exact(record['action_states'][:first + 1], sham['action_states'][:first + 1], torch)
    arrays['q0_action_states_FP32'] = np.stack([records[label]['action_states'][:, 0].numpy() for label, _, _ in PLAN])
    arrays['q0_action_velocity_BF16_export_FP32'] = np.stack([records[label]['action_velocity'].float().numpy() for label, _, _ in PLAN])
    arrays['q0_final_normalized_actions'] = np.stack([records[label]['actions'].numpy() for label, _, _ in PLAN])
    arrays['q0_vs_sham_solver_XYZ_rms'] = np.zeros((12, 31))
    arrays['q0_vs_sham_velocity_XYZ_rms'] = np.zeros((12, 30))
    arrays['q0_vs_sham_full_readout_rms'] = np.zeros((12, 30))
    arrays['q0_vs_sham_final_all10_rms'] = np.zeros(12)
    arrays['q0_vs_sham_final_XYZ_rms'] = np.zeros(12)
    for case, (label, v, _) in enumerate(PLAN):
        si = 2 if v == 195 else 3
        arrays['q0_vs_sham_solver_XYZ_rms'][case] = rms(arrays['q0_action_states_FP32'][case, :, :, :3].astype(np.float64) - arrays['q0_action_states_FP32'][si, :, :, :3], (1, 2), np)
        arrays['q0_vs_sham_velocity_XYZ_rms'][case] = rms(arrays['q0_action_velocity_BF16_export_FP32'][case, :, :, :3].astype(np.float64) - arrays['q0_action_velocity_BF16_export_FP32'][si, :, :, :3], (1, 2), np)
        arrays['q0_vs_sham_full_readout_rms'][case] = rms(records[label]['readout'].float().numpy().astype(np.float64) - records[PLAN[si][0]]['readout'].float().numpy(), (1, 2), np)
        delta = arrays['q0_final_normalized_actions'][case].astype(np.float64) - arrays['q0_final_normalized_actions'][si]
        arrays['q0_vs_sham_final_all10_rms'][case], arrays['q0_vs_sham_final_XYZ_rms'][case] = rms(delta, None, np), rms(delta[:, :3], None, np)
    arrays['sigmas'] = records[PLAN[0][0]]['sigmas'].numpy()
    arrays['timesteps'] = records[PLAN[0][0]]['timesteps'].numpy()
    arrays['case_labels'], arrays['arms'] = np.asarray([item[0] for item in PLAN]), np.asarray(ARMS)
    return arrays, manifests


def plots(out, arrays, summaries, prediction, np):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    colors = ('#d9dce2', '#63b18a', '#e4ad64', '#af94cd', '#8ca1b3')
    values, texts = np.zeros((6, 2)), []
    for i, arm in enumerate(ARMS):
        text = []
        for j, v in enumerate((195, 198)):
            summary = summaries[f'V{v}_A195_{arm}']
            objects = summary['selected_objects']
            values[i, j] = 1 if objects == ['milk_1'] else 2 if objects == ['cream_cheese_1'] else 3 if set(objects) == {'milk_1', 'cream_cheese_1'} else 0 if not objects else 4
            label = 'Only milk' if values[i, j] == 1 else 'Only cheese' if values[i, j] == 2 else 'Milk + cheese' if values[i, j] == 3 else 'No strict selection' if values[i, j] == 0 else ', '.join(objects)
            step = summary['first_selection_step']
            text.append(label + ('\nfirst step ' + str(step) if step is not None else ''))
        texts.append(text)
    fig = plt.figure(figsize=(11, 9), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.7, 1))
    ax = fig.add_subplot(grid[0, :])
    ax.imshow(values, cmap=ListedColormap(colors), vmin=-.5, vmax=4.5, aspect='auto')
    ax.set_xticks([0, 1], ['V195 / A195', 'V198 / A195'])
    ax.set_yticks(range(6), ['Native control', 'Global numerical sham', 'Early / layers1..9 — PRIMARY',
        'Early / layers10..18', 'Early / layers19..27', 'Early / layers28..36'])
    for i in range(6):
        for j in range(2):
            ax.text(j, i, texts[i][j], ha='center', va='center', fontsize=10, weight='bold' if i == 2 else 'normal')
    ax.axhline(1.5, color='#273749', lw=2)
    ax.axhline(2.5, color='#273749', lw=2)
    ax.set_title('All twelve real 128-step trials: strict contact + lift selection', fontsize=12)
    palette = ('#176c93', '#b34f40', '#54964b', '#8c66ae')
    for j, v in enumerate((195, 198)):
        ax = fig.add_subplot(grid[1, j])
        for wi, arm in enumerate(WINDOWS):
            idx = [item[0] for item in PLAN].index(f'V{v}_A195_{arm}')
            ax.plot(range(31), arrays['q0_vs_sham_solver_XYZ_rms'][idx], color=palette[wi], label=arm.replace('_', ' '), lw=2 if wi == 0 else 1.3)
        ax.axvline(15, color='#999999', lw=.8, ls=':')
        ax.set_title(f'V{v}: saved q0 solver XYZ distance to own sham', fontsize=10)
        ax.set_xlabel('Actual solver sample index 0..30')
        ax.set_ylabel('XYZ RMS difference (normalized action units)')
        ax.grid(alpha=.2)
        if j == 1:
            ax.legend(fontsize=8)
    fig.suptitle('Primary layers1..9 prediction: ' + prediction.upper() + '\nBoth must select only milk; other windows cannot replace this prediction', fontsize=14, weight='bold')
    fig.savefig(out / 'strict-window-outcomes.png', dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for col, group in enumerate(('current', 'action')):
        for wi, arm in enumerate(WINDOWS):
            for v, ls in ((195, '-'), (198, '--')):
                idx = [item[0] for item in PLAN].index(f'V{v}_A195_{arm}')
                axes[0, col].plot(range(30), arrays['vs_sham_hidden_relative_' + group][idx, :, 36] * 100,
                    color=palette[wi], ls=ls, label=arm.replace('_', ' ') + ' / V' + str(v), lw=1.4)
        for ai, arm in enumerate(ARMS):
            axes[1, col].plot(range(30), arrays['cross_V_hidden_relative_' + group][ai, :, 36] * 100,
                label=arm.replace('_', ' '), ls=':' if ai < 2 else '-', lw=1.2)
        axes[0, col].set_title(group.capitalize() + ': window minus SAME-V sham, boundary36')
        axes[1, col].set_title(group.capitalize() + ': V198 minus V195, SAME arm, boundary36')
        for row in range(2):
            axes[row, col].set_xlabel('Denoising index 0..29 (sigma varies)')
            axes[row, col].set_ylabel('100 × array RMS distance / reference RMS')
            axes[row, col].grid(alpha=.2)
    axes[0, 1].legend(fontsize=7, ncol=2)
    axes[1, 1].legend(fontsize=8, ncol=2)
    fig.suptitle('Actual output-boundary distances; percentages are array ratios\nNot probabilities, task success or fractions of explained behavior', fontsize=13)
    fig.savefig(out / 'same-sham-and-cross-source.png', dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for col, group in enumerate(('current', 'action')):
        values = arrays['vs_sham_net_update_delta_rms_' + group][4:6]
        vmax = max(float(np.max(values)), 1e-12)
        for row, v in enumerate((195, 198)):
            image = axes[row, col].imshow(values[row].T, origin='lower', aspect='auto', vmin=0, vmax=vmax, cmap='viridis', interpolation='nearest')
            axes[row, col].set_title(f'Primary layers1..9 / V{v} / ' + group)
            axes[row, col].set_xlabel('Denoising index 0..29')
            axes[row, col].set_ylabel('Zero-based decoder block 0..35')
            axes[row, col].set_yticks([0, 8, 9, 35])
            axes[row, col].axhline(8.5, color='white', lw=.8, ls='--')
            axes[row, col].axvline(14.5, color='white', lw=.8, ls='--')
        fig.colorbar(image, ax=axes[:, col].tolist(), label='RMS difference of actual net block updates', shrink=.8)
    fig.suptitle('Preregistered layers1..9: how actual block updates differ from own sham\nNet update = H[b+1] − H[b], combining attention and MLP residuals', fontsize=13)
    fig.savefig(out / 'primary-block-net-updates.png', dpi=170)
    plt.close(fig)


def previous_eighteen_layer_reference(frozen):
    # The frozen producer's source_contract has already checked every prior
    # document, boundary and physical-file SHA. Preserve that manifest and its
    # actual outcomes; this analyzer does not rerun the old numerical audit.
    evidence = frozen['source_evidence']['previous_eighteen_layer_stages']
    assert evidence['producer_script_sha256'] == PREVIOUS_PROBE_SHA
    assert evidence['physics_script_sha256'] == PREVIOUS_EXECUTION_SHA
    assert evidence['old_primary_prediction'] == 'rejected' and evidence['old_primary_both_only_milk'] is False
    rows = evidence['actual_twelve_physical_results']
    expected_arms = ('native', 'all_allowed', 'early_low', 'early_high', 'late_low', 'late_high')
    labels = [f'V{v}_A195_{arm}' for arm in expected_arms for v in (195, 198)]
    assert [row['case'] for row in rows] == labels
    assert set(evidence['q0_cases']) == set(evidence['physical_cases']) == set(labels)
    for row in rows:
        label = row['case']
        assert len(evidence['q0_cases'][label]['boundary_files']) == 30
        hashes = evidence['physical_cases'][label]['files_sha256']
        assert row['summary_sha256'] == hashes['summary.json']
        assert row['strict_windows_sha256'] == hashes['strict-windows.json']
    selected = {row['case']: row['selected_objects'] for row in rows}
    assert selected['V195_A195_early_low'] == ['milk_1'] and selected['V198_A195_early_low'] == []
    return dict(**evidence,
        verification_scope='Prior complete/document/360-boundary/physical-file hashes checked by the pinned producer source_contract. Prior numerical outcomes are sealed references; no old model, physics, or numerical-audit rerun here.',
        interpretation_limit='An eighteen-layer joint mask and its nine-layer subranges can interact nonlinearly and follow different trajectories. Effects are never summed or called additive contributions; old rejection is never replaced by a new descriptive group.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    baseline = parser.parse_args().output.resolve()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import numpy as np
    import torch
    assert not torch.cuda.is_initialized()
    adapter, frozen, _, helper, physical, complete, controls, primary = sealed(baseline)
    out = baseline / 'future-current-layer-analysis'
    out.mkdir(exist_ok=False)
    started = time.perf_counter()
    try:
        previous = previous_eighteen_layer_reference(frozen)
        write(out / 'provenance.json', dict(script_sha256=sha(Path(__file__)), inherited_analyzer_sha256=INHERITED_ANALYZER_SHA,
            previous_eighteen_layer_stages=previous, producer_script_sha256=PROBE_SHA,
            execution_script_sha256=EXECUTION_SHA, source_execution_complete_sha256=sha(physical / 'complete.json'),
            source_execution_documents_sha256=complete['files_sha256'], producer_documents_sha256=frozen['producer_documents_sha256'],
            scope=SCOPE, device='CPU', model_forwards=0, attention_dispatch_calls=0, solver_steps=0, physics_calls=0))
        records, summaries, arrays, audit = audit_records(baseline, frozen, helper, physical, controls, primary, torch, np)
        propagated, manifests = propagation(baseline, frozen, records, torch, np)
        arrays.update(propagated)
        assert complete['primary_prediction'] == audit['primary_prediction'] and complete['numerical_sham_classification_gate_passed'] is audit['sham_classification_gate_passed']
        for key, value in arrays.items():
            if value.dtype.kind == 'f' and 'relative' not in key:
                assert np.isfinite(value).all(), key
        np.savez_compressed(out / 'actual-arrays.npz', **arrays)
        assert (out / 'actual-arrays.npz').stat().st_size < 100 * 1024 * 1024, 'Public NPZ must be strictly smaller than100MiB'
        with np.load(out / 'actual-arrays.npz', allow_pickle=False) as saved:
            assert set(saved.files) == set(arrays)
            for key, value in arrays.items():
                actual = saved[key]
                assert actual.dtype == value.dtype and actual.shape == value.shape and actual.tobytes(order='C') == value.tobytes(order='C'), key
            for i, (label, _, _) in enumerate(PLAN):
                for key, expected in (('q0_action_states_FP32', records[label]['action_states'][:, 0].numpy()),
                                      ('q0_action_velocity_BF16_export_FP32', records[label]['action_velocity'].float().numpy()),
                                      ('q0_final_normalized_actions', records[label]['actions'].numpy())):
                    assert npexact(saved[key][i], expected, np), ('Actual PT-to-NPZ export bytes', label, key)
        write(out / 'audit.json', audit)
        write(out / 'actual-boundary-manifest.json', manifests)
        plots(out, arrays, summaries, audit['primary_prediction'], np)
        analysis = dict(scope=SCOPE, case_order=[item[0] for item in PLAN], primary_prediction=audit['primary_prediction'],
            primary_cases=frozen['primary_cases'], primary_requires_strict_only_milk_for_each_case=True,
            selected_layers_zero_based_inclusive=[0, 8], selected_steps_zero_based_inclusive=[0, 14],
            preregistered_layer_groups=WINDOWS, previous_eighteen_layer_stages=previous,
            equal_135_site_budgets_are_not_equal_perturbation_strengths=True,
            q0_total_official_dispatch_calls=37800, q0_extra_official_dispatch_calls=11880,
            active_cut_sites_checked=1080, active_cut_sites_per_window_case=135,
            no_other_window_promoted=True, physical_selected_objects={label: summary['selected_objects'] for label, summary in summaries.items()},
            definitions=dict(boundaries='0=actual decoder input;1..36=actual whole block outputs after attention+MLP residuals, before final norm.',
                net_update='Actual H[b+1]−H[b], float64 arithmetic on lossless BF16→FP32 exports; not an isolated attention/MLP increment.',
                same_V_reference='Every window compared with its SAME-V global all-allowed sham on its own dynamic trajectory.',
                cross_V_reference='V198−V195 within EACH arm; reference RMS is V195 same arm. Native/sham/windows remain distinct.',
                relative='100×RMS(delta)/RMS(reference); null/NaN if reference<=1e-12. Not task success, restoration or causal mediation.',
                solver='Actual saved FP32 action_states, not reconstructed from BF16 inputs. XYZ is first3 normalized action dimensions.',
                input_cast='At every step actual BF16 action_tokens equals saved FP32 solver sample cast; no new solver/model evaluation.'),
            gap_floor=FLOOR, invalid_relative_denominators={key: int(np.count_nonzero(~np.isfinite(value))) for key, value in arrays.items() if 'relative' in key},
            q0_final_action_distances={label: dict(XYZ_RMS_to_same_V_sham=float(arrays['q0_vs_sham_final_XYZ_rms'][i]),
                all10_RMS_to_same_V_sham=float(arrays['q0_vs_sham_final_all10_rms'][i])) for i, (label, _, _) in enumerate(PLAN)},
            actual_arrays={key: dict(shape=list(value.shape), dtype=str(value.dtype)) for key, value in arrays.items()},
            site_gate_limit='Recorded live c106/8fef-derived layer-group footprint/QKV/RNG/unblocked-query gates and masks are checked. Full QKV was not saved; no independent attention recomputation is claimed.',
            controller_limit='Typed actual controller arrays at0/16; all129 serialized schemas/finite values. No unavailable intermediate original dtype arrays invented.',
            actual_physical_actions_checked=1536, whole_noise_pairs_checked=77, whole_model_records_checked=96,
            actual_boundary_files_checked=360, four_entire_native_and_sham_controls_rechecked=True,
            model_forwards=0, attention_dispatch_calls=0, solver_steps=0, physics_calls=0, cuda_initialized=torch.cuda.is_initialized())
        assert not torch.cuda.is_initialized()
        write(out / 'analysis.json', analysis)
        names = ('provenance.json', 'audit.json', 'actual-boundary-manifest.json', 'actual-arrays.npz', 'analysis.json',
                 'strict-window-outcomes.png', 'same-sham-and-cross-source.png', 'primary-block-net-updates.png')
        write(out / 'complete.json', dict(state='complete', script_sha256=sha(Path(__file__)), files_sha256={name: sha(out / name) for name in names},
            primary_prediction=audit['primary_prediction'], actual_physical_actions_checked=1536, whole_noise_pairs_checked=77,
            whole_model_records_checked=96, actual_boundary_files_checked=360, active_cut_sites_checked=1080,
            q0_total_official_dispatch_calls=37800, model_forwards=0, attention_dispatch_calls=0,
            solver_steps=0, physics_calls=0, cuda_initialized=False, elapsed_s=time.perf_counter() - started, scope=SCOPE))
        print('[FUTURE-CURRENT-LAYER-ANALYSIS] complete ' + json.dumps(dict(primary_prediction=audit['primary_prediction'], elapsed_s=time.perf_counter() - started)), flush=True)
    except Exception as exc:
        write(out / 'failed.json', dict(error=str(exc), traceback=traceback.format_exc(), scope=SCOPE))
        raise


if __name__ == '__main__':
    main()
