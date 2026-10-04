"""Execute the twelve frozen X+15/X+18cm first-nine-layer q0 cases.

Use --output BASE --mode prepare|server|simulate. Create only an absent
BASE/future-current-position-execution. Four fresh native q0 observations,
four global all-allowed shams and four early_l1_9 cuts are executed in the
producer's fixed order. Each uses its own saved q0 for sixteen actions and
seven actual native feedback queries at seeds196..202: twelve cached plus
84 fresh predictions, 96 requests, 2520 fresh forwards and 77 actual later
whole two-draw noise pairs. Producer forwards are separately360, combined2880.

X+0cm is only a collection/restore reference. Fresh native q0 records from
the same new scene and V are the preparation references. There is no old
X+12cm whole-trajectory equality or independent native-repeat claim. The
only primary criterion requires all four cut cases to select exactly milk
by two-finger contact and lift>.02m for five consecutive records, with all
four sham classifications matching their same-scene/V native cases. A
native milk result preserved by the cut is preservation, not rescue. This
bounded generalization prerequisite does not establish functional or
cortex-like specificity. All old sources/artifacts remain immutable.
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
PREVIOUS_SHA = 'e6d5abcb19d866f372d1373ba97a383f8093f1917859ff9714451f8e6138d40f'
HELPER_SHA = 'e0c5073ee320bbc173eebf827126c9431043a3efaa960eb0a8ccff289cae30ed'
SIMULATOR_SHA = '7060f610814cbf6b13fb79ca2da78af73bba542d479ac2b8a73110097543d845'
COLLECTOR_SHA = 'b06454696a61b2ca9c570cbeac3f3d8149a4bf5179f51dafdb75a5d70cbbc945'
CONTROLLER_SHA = 'bc18a41d97dcdf7c321ef093faf4eea076678298f269efa5cf8fc9c623157dc7'
PROBE_SHA = 'c9cc366e002d8c89ab9c779af35b0191c4ec6d56a157da77499914c6b8605146'
CROSS_SHA = '7e684b1a330877b8dbaad2c27621c3bc8db4893eb25da0c9171ff6d8b56aa2fe'
PORT, LATER_BASE = 8934, 195
PROMPT = 'pick up the milk and place it in the basket'
SHIFTS = dict(x00=0, x15=15, x18=18)
WINDOWS = {'early_l1_9': dict(steps=[0, 14], layers=[0, 8])}
PLAN = tuple((f'{scene}_V{v}_A195_{arm}', scene, v, arm)
             for arm in ('native', 'all_allowed', 'early_l1_9')
             for scene in ('x15', 'x18') for v in (195, 198))
PHYSICAL = ('trajectory.json', 'actions.json', 'executed_actions.json',
            'normalized_actions.json', 'denormalized_actions.json')
SCOPE = ('Twelve preregistered X+15/X+18cm physical trajectories using their own '
         'fresh saved q0 predictions, then native inference from actual feedback at '
         'later base195. X+0cm is a restore reference only. Same-scene/V native q0 '
         'records validate preparation, with no old X+12cm physical equality or '
         'native-repeat claim. All four fixed early_l1_9 cases selecting exactly '
         'milk and all four shams matching native classifications test the sole '
         'primary prediction. Native milk retained by a cut is preservation. '
         'Hard masking also redistributes attention. This is a generalization '
         'prerequisite, with no functional/cortex-like specificity, semantic '
         'identity, neural root cause, training, selection or additional window claim.')


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


def hooks(model):
    """Record all actual forward hook registries without installing any hooks."""
    return [(name, tuple(module._forward_pre_hooks), tuple(module._forward_hooks))
            for name, module in model.named_modules()]


def contract(baseline):
    previous_path = ROOT / 'work/run_cosmos_milk_future_current_layer_execution.py'
    helper_path = ROOT / 'work/run_cosmos_milk_future_read_execution.py'
    probe_path = ROOT / 'work/probe_cosmos_milk_future_current_positions.py'
    cross_path = ROOT / 'work/run_cosmos_milk_query_noise_cross.py'
    assert sha(previous_path) == PREVIOUS_SHA and sha(helper_path) == HELPER_SHA
    assert sha(probe_path) == PROBE_SHA and sha(cross_path) == CROSS_SHA
    probe = load('position_execution_frozen_producer', probe_path)
    context = probe.prepared(baseline)
    assert isinstance(context, dict) and probe.PLAN == PLAN and probe.WINDOWS == WINDOWS
    original, evidence, paths, scenes = (context[name] for name in ('frozen_contract', 'source_evidence', 'paths', 'scenes'))
    simulator_path = Path(original['frozen_files']['simulator']['path'])
    assert sha(simulator_path) == SIMULATOR_SHA
    assert sha(Path(original['frozen_files']['collector']['path'])) == COLLECTOR_SHA
    assert sha(Path(original['frozen_files']['controller_helper']['path'])) == CONTROLLER_SHA
    simulator = load('position_execution_original_simulator', simulator_path)
    cross = load('position_execution_frozen_query_seed', cross_path)
    helper = load('position_execution_frozen_strict_windows', helper_path)
    producer = context['out']
    assert producer == baseline / 'future-current-positions'
    assert producer.is_dir() and not (producer / 'failed.json').exists()
    complete, results, protocol, provenance, sources, primary = (read(producer / (name + '.json')) for name in
        ('complete', 'results', 'protocol', 'provenance', 'sources', 'primary-prediction'))
    assert complete['state'] == results['state'] == 'complete'
    assert complete['script_sha256'] == provenance['script_sha256'] == PROBE_SHA
    for name in ('results', 'protocol', 'provenance', 'sources'):
        assert complete[name + '_sha256'] == sha(producer / (name + '.json'))
    assert complete['primary_prediction_sha256'] == results['primary_prediction_sha256'] == sha(producer / 'primary-prediction.json')
    assert complete['fresh_q0_predictions'] == results['fresh_q0_predictions'] == 12
    assert complete['fresh_model_forwards'] == results['fresh_model_forwards'] == 360
    assert complete['total_official_dispatch_calls'] == results['total_official_dispatch_calls'] == 35100
    assert complete['original_dispatch_calls'] == 25920 and complete['extra_dispatch_calls'] == 9180
    assert complete['original_randn_calls_consumed'] == complete['returned_source_draws_observed'] == 24
    assert complete['new_native_reference_predictions'] == 4
    for key in ('no_extra_source_forwards', 'four_new_native_actual_scene_input_contracts_passed',
                'all8_masked_initial_inputs_equal_own_scene_V_native', 'all4_cut_before_first_cut_own_scene_V_sham_prefix_exact',
                'all_active_cut_unblocked_GEN_vs_same_input_allallowed_exact', 'all8_masked_full1080_sites_return_common_union66',
                'all360_actual_complete_model_kwargs_saved', 'all_original_symbols_restored', 'all_owned_hooks_removed'):
        assert complete[key] is True
    assert complete['physics_calls'] == complete['later_query_predictions'] == results['physics_calls'] == results['later_query_predictions'] == 0
    assert complete['primary_physical_prediction_evaluated'] is results['primary_physical_prediction_evaluated'] is False
    assert results['official_dispatch_counts'] == dict(model_before=360, model_after=360, native_UND_dispatch=12960,
        native_GEN_dispatch=12960, all_allowed_dispatch=8640, cut_dispatch=540)
    expected_plan = [dict(case=label, scene=scene, shift_cm=SHIFTS[scene], vision_noise_source_seed=v,
        action_noise_source_seed=195, runtime_rng_seed=195, arm=arm, window=WINDOWS.get(arm)) for label, scene, v, arm in PLAN]
    labels, primary_cases = [item[0] for item in PLAN], [item[0] for item in PLAN[8:]]
    assert protocol['plan'] == expected_plan and protocol['trial_order'] == results['preregistered_physics_cases'] == labels
    assert protocol['case_scene'] == {label: scene for label, scene, _, _ in PLAN}
    assert protocol['scenes'] == scenes and protocol['prompt'] == PROMPT
    assert protocol['windows_zero_based_inclusive'] == WINDOWS and protocol['primary_window'] == 'early_l1_9'
    assert protocol['cut_sites_per_window_case'] == 135 and protocol['masked_sites_per_case'] == 1080
    assert protocol['common_masked_union66'] is protocol['no_extra_source_forwards'] is True
    assert protocol['primary_cases'] == primary['cases'] == primary_cases
    assert primary['status'] == 'not_evaluated_q0_only' and primary['all_four_selected_objects_must_equal'] == ['milk_1']
    assert primary['sham_preserves_own_native_classification_required'] is primary['all12_physical_cases_preregistered'] is True
    assert primary['q0_numeric_rank_used'] is False
    assert protocol['primary_evaluation'] == dict(status='not_evaluated_q0_only', all_four_cut_selected_objects_must_equal=['milk_1'],
        numerical_shams_must_preserve_own_native_selected_objects=True, no_other_location_or_window_promoted=True)
    physical = protocol['later_physics_preregistered']
    assert physical == dict(trial_order=labels, cases=12, cached_q0=12, later_noise_base=195,
        q1_to_q7_seeds=list(range(196, 203)), requests=96, later_predictions=84, later_model_forwards=2520,
        total_two_stage_model_forwards=2880, steps_per_case=128, queries_per_case=8, whole_noise_pairs_expected=77,
        all12_execute_without_q0_score_selection=True, stop_native_interpretation_if_sham_changes_classification=True,
        execute_all12_even_if_sham_changes_classification=True)
    assert sources == dict(source_evidence=evidence, original_noise_sources=context['original_noise_sources'],
        scenes=scenes, frozen_contract=original)
    assert provenance['source_sha256'] == {key: sha(path) for key, path in paths.items()}
    assert provenance['source_evidence'] == evidence and provenance['new_source_first_observations'] is True
    assert provenance['independent_nohook_repeat_performed'] is False and provenance['extra_source_forwards'] == 0
    upstream = evidence['frozen_X12_evidence']['anchors']['upstream']
    assert provenance['actual_transformer_sha256'] == upstream['actual_transformer_sha256']
    assert provenance['actual_pipeline_source_sha256'] == upstream['actual_pipeline_source_sha256']
    assert complete['new_native_source_checks_sha256'] == results['new_native_source_checks_sha256']
    assert set(complete['new_native_source_checks_sha256']) == set(scenes) == {'x15', 'x18'}
    for scene, expected in complete['new_native_source_checks_sha256'].items():
        assert scene in scenes and sha(producer / (scene + '-new-native-source-checks.json')) == expected
        checks = read(producer / (scene + '-new-native-source-checks.json'))
        assert set(checks) == {'195', '198'}
        for item in checks.values():
            for key in ('all_thirty_layout_schedule_metadata_exact', 'current_condition_frame0_all_thirty_model_calls_byte_exact',
                        'prepared_vision_entire_tensor_equals_V_source', 'prepared_action_entire_tensor_equals_A_source',
                        'other_ten_prepared_fields_byte_exact', 'other_initial_model_kwargs_byte_exact',
                        'all_prepared_model_input_and_31_solver_state_padding54_zero', 'action_condition_mask_all_zero'):
                assert item[key] is True
    np = simulator.np
    stats = read(Path(original['frozen_files']['normalizer_stats']['path']))['global_raw']
    lo, hi = np.asarray(stats['q01']), np.asarray(stats['q99'])
    assert lo.shape == hi.shape == (10,) and np.all(hi - lo > 1e-6)
    scale, offset = np.maximum(hi - lo, 1e-8) / 2, (hi + lo) / 2
    assert [row['case'] for row in results['cases']] == labels
    q0, rules = {}, {}
    for row, (label, scene, v, arm) in zip(results['cases'], PLAN):
        folder, window = producer / label, WINDOWS.get(arm)
        assert row['scene'] == scene and row['shift_cm'] == SHIFTS[scene] and row['arm'] == arm and row['window'] == window
        assert row['vision_noise_source_seed'] == v and row['action_noise_source_seed'] == row['runtime_rng_seed'] == 195
        assert row['fresh_model_forwards'] == 30 and row['physical_prediction_evaluated'] is False
        assert row['new_native_reference_first_observation'] is (arm == 'native')
        assert row['own_sham_pre_first_cut_prefix_exact'] is (True if window else None)
        assert set(row['files_sha256']) == set(probe.FILES)
        for name, expected in row['files_sha256'].items():
            assert sha(folder / name) == expected
        report, meta, check = (read(folder / name) for name in ('observer-report.json', 'metadata.json', 'control-checks.json'))
        counts = dict(model_before=30, model_after=30, block_before=1080, block_after=1080, pre_W=1080,
            native_UND_dispatch=1080, native_GEN_dispatch=1080, all_allowed_dispatch=0 if arm == 'native' else 1080,
            cut_dispatch=135 if window else 0, substituted_sites=0 if arm == 'native' else 1080)
        assert report['counts'] == row['actual_dispatch_counts'] == counts
        assert report['state'] == 'complete' and report['case_arm'] == arm and report['scene'] == scene and report['selected_window'] == window
        assert report['native_source_first_observation'] is (arm == 'native')
        assert report['independent_nohook_repeat_performed'] is report['X12_entire_input_comparison_performed'] is False
        for key in ('all_actual_dispatch_flatten_pre_W_byte_exact', 'future200_and_UND_native_dispatch_returns_preserved',
                    'all30_actual_complete_model_kwargs_saved', 'all_owned_hooks_removed', 'original_dispatch_symbol_restored'):
            assert report[key] is True
        assert report['original_full_266_row_projection_calls'] == 1080 and report['processor_recompute_calls'] == 0
        assert report['all_masked_sites_same_union66'] is (None if arm == 'native' else True)
        assert report['all_window_cut_unblocked_GEN_rows_vs_same_input_all_allowed_exact'] is (True if window else None)
        assert len(row['boundary_files']) == 30 and report['boundary_files'] == row['boundary_files']
        for step, item in enumerate(row['boundary_files']):
            assert item['step'] == step and item['file'] == f'boundaries-t{step:02d}.pt'
            assert item['current_shape'] == [37, 50, 4096] and item['action_shape'] == [37, 16, 4096]
            assert sha(folder / item['file']) == item['sha256']
        assert meta['case'] == label and meta['scene'] == scene and meta['shift_cm'] == SHIFTS[scene] and meta['arm'] == arm
        assert meta['window_zero_based_inclusive'] == window and meta['seed'] == meta['runtime_rng_seed'] == meta['action_noise_source_seed'] == 195
        assert meta['vision_noise_source_seed'] == v and meta['model_calls'] == 30 and meta['prepare_calls'] == 1
        assert meta['input_png_sha256'] == scenes[scene]['files_sha256']['input.png'] and meta['physical_prediction_evaluated'] is False
        assert meta['native_noop'] is (arm == 'native') and meta['numerical_masked_sham'] is (arm == 'all_allowed')
        assert meta['common_union66_all_allowed_background'] is (arm != 'native')
        assert meta['actual_masked_site_substitutions'] == counts['substituted_sites'] and meta['window_cut_sites'] == counts['cut_dispatch']
        assert all(value is True for value in check['actual_input_checks'].values())
        if window:
            assert check['state'] == 'exact' and check['comparison_case'] == f'{scene}_V{v}_A195_all_allowed'
            assert check['comparison_same_scene_and_V'] is True and check['first_cut_step'] == 0 and check['first_cut_layer_zero_based'] == 0
            for key in ('earlier_complete_model_outputs_and_head_prefix_exact', 'all_actual_complete_kwargs_through_first_cut_step_exact',
                        'earlier_all37_and_first_cut_before_layer_current_action_boundaries_exact', 'solver_samples_through_first_cut_input_exact'):
                assert check[key] is True
        else:
            assert check['state'] == ('new_native_reference_observed' if arm == 'native' else 'new_numerical_sham_observed')
            assert check['whole_prior_record_exact'] is None and check['independent_nohook_repeat_performed'] is False
        normalized = np.asarray(read(folder / 'normalized_actions.json'))
        assert normalized.shape == (16, 10) and np.isfinite(normalized).all() and normalized.tolist() == row['final_normalized_actions']
        q0[label] = dict(folder=str(folder), files_sha256=row['files_sha256'], boundary_files=row['boundary_files'],
            normalized_actions=normalized.tolist(), raw_actions=(normalized * scale + offset).tolist(),
            scene=scene, arm=arm, selected_window=window, masked_attention=arm != 'native', vision_noise_source_seed=v, action_noise_source_seed=195)
        rules[label] = dict(scene=scene, arm=arm, selected_window=window, q0_vision_noise_source_seed=v, q0_action_noise_source_seed=195,
            native_control=arm == 'native', numerical_sham=arm == 'all_allowed', primary_case=arm == 'early_l1_9',
            later_noise_base=195, actual_query_seeds=list(range(195, 203)), q0_whole_noise_pairing=False,
            seed_rule='Saved q0 has separate V/A sources and invocation195; q1..7=195+query.')
    frozen = dict(script_sha256=sha(Path(__file__)), previous_execution_sha256=PREVIOUS_SHA, helper_execution_sha256=HELPER_SHA,
        producer_script_sha256=PROBE_SHA, simulator_sha256=SIMULATOR_SHA, cross_script_sha256=CROSS_SHA,
        threshold_contract=original, source_evidence=evidence, actual_transformer_sha256=upstream['actual_transformer_sha256'],
        actual_pipeline_source_sha256=upstream['actual_pipeline_source_sha256'], metrics_path=str(paths['metrics']), components_path=str(paths['components']),
        original_noise_sources=context['original_noise_sources'], scenes=scenes,
        producer_documents_sha256={name: sha(producer / (name + '.json')) for name in ('complete', 'results', 'provenance', 'protocol', 'sources', 'primary-prediction')},
        producer_native_source_checks_sha256=complete['new_native_source_checks_sha256'], producer_protocol=protocol,
        q0_files=list(probe.FILES), q0_sources=q0, trials=rules, trial_order=labels, primary_window='early_l1_9', primary_cases=primary_cases,
        primary_selected_objects_exact=['milk_1'], primary_numerical_sham_classification_match_required=True,
        all12_execute_without_q0_score_selection=True, inputs_symlink_target=str(context['input_root']), url=f'http://127.0.0.1:{PORT}',
        saved_q0_predictions=12, requests=96, native_predictions=84, fresh_model_forwards=2520,
        producer_q0_forwards_separate=360, combined_two_stage_model_forwards=2880,
        actual_later_whole_noise_pairs_expected=77, q0_whole_noise_pair_checks=0, scope=SCOPE)
    simulator.SHIFTS, simulator.SEEDS = SHIFTS.copy(), (195,)
    simulator.TRIALS = {label: (scene, cross.QuerySeed(195, 195)) for label, scene, _, _ in PLAN}
    simulator.URL, simulator.SCOPE = frozen['url'], SCOPE
    assert all(scene != 'x00' for scene, _ in simulator.TRIALS.values())
    return simulator, frozen, cross, helper


def verify_cached(actual, audit, own_native, original_noise, v, runtime, scan):
    """Verify new-scene preparation against its actual same-scene/V native."""
    torch = runtime.torch
    scan.validate_record(actual, torch)
    runtime.require_exact(actual['pure_noise'],
        [original_noise[v]['pure_noise'][0], original_noise[195]['pure_noise'][1]],
        'actual returned selected V/A raw draws')
    runtime.require_exact(actual['prepared_latents_and_masks'], own_native['prepared_latents_and_masks'],
        'same fresh scene/V complete prepared tuple')
    p = actual['prepared_latents_and_masks']
    assert isinstance(p, tuple) and len(p) == 12 and p[3] == 20 and p[10] == 10
    assert p[0].shape == (1, 48, 5, 10, 20) and p[0].dtype == torch.float32
    assert p[2].shape == (16, 64) and p[2].dtype == torch.float32
    assert p[5].shape == (5, 1, 1) and p[7].shape == (16, 1)
    assert p[8].tolist() == [5]
    runtime.require_exact(p[5].flatten(), torch.tensor([1., 0., 0., 0., 0.], dtype=p[5].dtype),
        'current frame clamped and four future frames noisy')
    assert torch.count_nonzero(p[7]) == torch.count_nonzero(p[2][:, 10:]) == 0
    assert torch.count_nonzero(actual['action_states'][..., 10:]) == 0
    runtime.require_exact(p[0][:, :, 1:], original_noise[v]['prepared_latents_and_masks'][0][:, :, 1:], 'raw V-source future preparation only')
    runtime.require_exact(p[2], original_noise[195]['prepared_latents_and_masks'][2], 'raw A195 action preparation')
    runtime.require_exact(actual['action_states'][0], original_noise[195]['action_states'][0], 'raw A195 initial FP32 solver sample')
    runtime.require_exact(actual['model_input'], own_native['model_input'],
        'same fresh scene/V full initial model kwargs')
    runtime.require_exact(audit['initial_model_kwargs'], actual['model_input'], 'actual first-model observation')
    runtime.require_exact(actual['action_states'][0], own_native['action_states'][0], 'same native initial FP32 action sample')
    assert len(audit['calls']) == 2 and len(audit['model_steps']) == 30
    assert audit['original_symbol_restored'] is audit['observer_hooks_removed'] is audit['dispatch_symbol_restored'] is True
    for i, call in enumerate(audit['calls']):
        assert call['ordinal'] == i and call['original_generated_seed'] == 195
        assert call['returned_source_seed'] == (v if i == 0 else 195)
        assert not torch.equal(call['generator_state_before'], call['generator_state_after'])
        runtime.require_exact(call['original_generated_discarded'], original_noise[195]['pure_noise'][i],
            'actual original consumed runtime195 draw/' + str(i))
        runtime.require_exact(call['actual_returned'], actual['pure_noise'][i], 'actual returned inner draw/' + str(i))
        for key, value in (('original_generated_metadata', call['original_generated_discarded']),
                           ('returned_metadata', call['actual_returned'])):
            for field, expected in scan.tensor_meta(value, torch).items():
                assert call[key][field] == expected
    runtime.require_exact(audit['calls'][0]['generator_state_after'], audit['calls'][1]['generator_state_before'],
        'actual two-draw RNG chain')
    for key in ('timesteps', 'sigmas'):
        runtime.require_exact(actual[key], own_native[key], 'same native complete schedule/' + key)


def verify_full_inputs(actual, audit, folder, source, native_info, scene, v, runtime, components):
    """Replay the saved actual observations without any model or solver call."""
    torch = runtime.torch
    info = torch.load(folder / 'new-input-contract.pt', map_location='cpu', weights_only=True, mmap=True)
    assert info['scene'] == scene and info['vision_noise_source_seed'] == v
    assert info['native_reference_case'] == f'{scene}_V{v}_A195_native'
    assert all(value is True for value in info['actual_input_checks'].values())
    runtime.require_exact(info['actual_pack'], audit['model_steps'], 'actual outer pack/audit thirty steps')
    runtime.require_exact(info['actual_pack'], native_info['actual_pack'], 'same fresh scene/V structural/schedule pack')
    runtime.require_exact(info['first_model_kwargs'], actual['model_input'], 'saved inner/full outer first kwargs')
    runtime.require_exact(info['first_model_kwargs'], native_info['first_model_kwargs'], 'same fresh scene/V full first kwargs')
    runtime.require_exact(info['actual_clean_current'], native_info['actual_clean_current'], 'same fresh scene/V current condition')
    runtime.require_exact(actual['prepared_latents_and_masks'][0][0, :, 0].to(torch.bfloat16), info['actual_clean_current'],
        'actual new-scene prepared FP32 current cast to actual BF16 input')
    runtime.require_exact(info['original_solver_initial_action_FP32'], actual['action_states'][0], 'saved actual initial FP32 solver sample')
    assert actual['action_states'].shape == (31, 1, 16, 64) and actual['action_states'].dtype == torch.float32
    for step, item in enumerate(source['boundary_files']):
        boundary = torch.load(folder / item['file'], map_location='cpu', weights_only=True, mmap=True)
        kwargs = boundary['actual_model_kwargs']
        assert boundary['step'] == step and set(kwargs) == set(actual['model_input'])
        runtime.require_exact(boundary['actual_pack'], info['actual_pack'][step], 'saved actual pack/' + str(step))
        runtime.require_exact({key: kwargs.get(key) for key in components._STRUCTURAL_KEYS},
            info['actual_pack'][step]['metadata'], 'all actual structural kwargs/' + str(step))
        runtime.require_exact(kwargs['action_tokens'][0], actual['action_states'][step][0].to(torch.bfloat16),
            'actual FP32 solver sample to BF16 full kwargs/' + str(step))
        runtime.require_exact(kwargs['vision_tokens'][0][0, :, 0], info['actual_clean_current'], 'actual current clamp/' + str(step))
        runtime.require_exact(boundary['model_output'][2][0], actual['action_velocity'][step], 'actual full tuple action head/' + str(step))
        runtime.require_exact(boundary['solver_sigma_from_same_frozen_schedule'], actual['sigmas'][step], 'actual complete sigma/' + str(step))
        runtime.require_exact(boundary['actual_model_action_timesteps'], kwargs['action_timesteps'], 'actual action timesteps/' + str(step))
        assert boundary['current_hidden'].shape == (37, 50, 4096) and boundary['action_hidden'].shape == (37, 16, 4096)
        assert torch.count_nonzero(kwargs['action_tokens'][0][:, 10:]) == 0
        if step == 0:
            runtime.require_exact(kwargs, actual['model_input'], 'first saved actual complete kwargs')


def prepared(baseline):
    simulator, frozen, cross, helper = contract(baseline)
    out = baseline / 'future-current-position-execution'
    assert out.is_dir() and not (out / 'wrapper_failed.json').exists()
    assert (out / 'inputs').is_symlink() and (out / 'inputs').resolve() == Path(frozen['inputs_symlink_target']).resolve()
    assert read(out / 'provenance.json') == frozen
    saved = read(out / 'prepared.json')
    assert saved['state'] == 'prepared' and saved['provenance_sha256'] == sha(out / 'provenance.json')
    return simulator, frozen, cross, helper, out


def server(baseline):
    _, frozen, _, _, out = prepared(baseline)
    destination = out / 'server'
    destination.mkdir(exist_ok=False)
    served, later_noise, pairs, cached, fresh = [], {}, 0, 0, 0
    failed, http, started = False, None, time.perf_counter()
    try:
        os.environ['HF_HUB_OFFLINE'] = '1'
        paths = frozen['threshold_contract']['frozen_files']
        base = load('position_execution_normal_runtime', Path(paths['normal_runtime']['path']))
        base.TASKS['milk'] = (7, 'pick_up_the_milk_and_place_it_in_the_basket', PROMPT, 'milk_1')
        factory = load('position_execution_factory', Path(paths['factory']['path']))
        factory.OUT = destination
        assert sha(factory.SCHEDULER) == paths['scheduler']['sha256']
        runtime = base.NormalRuntime(factory)
        torch, model = runtime.torch, runtime.pipe.transformer
        assert not model.training and not model.is_cache_enabled and not torch.is_grad_enabled()
        assert sha(Path(inspect.getfile(type(model)))) == frozen['actual_transformer_sha256']
        assert sha(Path(inspect.getfile(type(runtime.pipe)))) == frozen['actual_pipeline_source_sha256']
        initial_hooks = hooks(model)
        assert all(not before and not after for _, before, after in initial_hooks)
        scan = load('position_execution_record_checks', Path(frozen['metrics_path']))
        components = load('position_execution_structural_keys', Path(frozen['components_path']))
        original_noise, q0, native_info = {}, {}, {}
        for seed in (195, 198):
            source = frozen['original_noise_sources'][str(seed)]
            folder = Path(source['q0_folder'])
            assert sha(folder / 'states.pt') == source['q0_files_current_sha256']['states.pt']
            original_noise[seed] = torch.load(folder / 'states.pt', map_location='cpu', weights_only=True, mmap=True)
        for label, _, _, _ in PLAN:
            source = frozen['q0_sources'][label]
            folder = Path(source['folder'])
            q0[label] = torch.load(folder / 'states.pt', map_location='cpu', weights_only=True, mmap=True)
        for label, _, _, _ in PLAN[:4]:
            folder = Path(frozen['q0_sources'][label]['folder'])
            native_info[label] = torch.load(folder / 'new-input-contract.pt', map_location='cpu', weights_only=True, mmap=True)
        controls = []
        for label, scene, v, arm in PLAN:
            source, folder = frozen['q0_sources'][label], Path(frozen['q0_sources'][label]['folder'])
            audit = torch.load(folder / 'noise-audit.pt', map_location='cpu', weights_only=True, mmap=True)
            native_label = f'{scene}_V{v}_A195_native'
            verify_cached(q0[label], audit, q0[native_label], original_noise, v, runtime, scan)
            verify_full_inputs(q0[label], audit, folder, source, native_info[native_label], scene, v, runtime, components)
            assert q0[label]['actions'].tolist() == source['normalized_actions']
            controls.append(dict(case=label, scene=scene, arm=arm, same_scene_V_native_case=native_label,
                actual_consumed_returned_noise_prepared_clamp_initial_full_kwargs_and_schedule_exact=True,
                all30_saved_actual_complete_model_kwargs_solver_casts_and_current_clamps_exact=True,
                fresh_native_record_is_observation_not_old_whole_control=True,
                q0_model_forwards_in_this_execution=0, producer_files_sha256=source['files_sha256']))
        write(destination / 'saved-q0-controls.json', dict(state='exact', cases=controls,
            all12_same_fresh_scene_V_native_preparation_gates_passed=True,
            q0_whole_noise_pair_checks=0, same_seed_does_not_imply_same_whole_V_A_pair=True,
            extra_q0_model_forwards=0), exclusive=True)
        write(destination / 'provenance.json', dict(script_sha256=sha(Path(__file__)),
            execution_provenance_sha256=sha(out / 'provenance.json'),
            saved_q0_controls_sha256=sha(destination / 'saved-q0-controls.json'),
            component_hooks_installed=False, attention_dispatch_wrapper_installed=False,
            saved_q0=12, native_later_predictions=84, expected_fresh_model_forwards=2520, scope=SCOPE), exclusive=True)

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
                    assert label in frozen['trials'] and type(query) is int and 0 <= query < 8
                    assert (label, query) not in served and all((label, i) in served for i in range(query))
                    assert label == frozen['trial_order'][len(served) // 8] and query == len(served) % 8, 'Fixed case/query order'
                    rule, source = frozen['trials'][label], frozen['q0_sources'][label]
                    scene, seed = rule['scene'], rule['actual_query_seeds'][query]
                    assert request['seed'] == seed and request['prompt'] == PROMPT
                    image = out / 'closed-loop' / label / f'input_{query:02d}.png'
                    chunk = image.parent / f'chunk_{query:02d}'
                    if query == 0:
                        assert image.read_bytes() == (out / 'inputs' / scene / 'input.png').read_bytes()
                        folder = Path(source['folder'])
                        chunk.mkdir(exist_ok=False)
                        for name in frozen['q0_files']:
                            assert sha(folder / name) == source['files_sha256'][name]
                            if name != 'metadata.json':
                                shutil.copyfile(folder / name, chunk / name)
                        for item in source['boundary_files']:
                            assert sha(folder / item['file']) == item['sha256']
                            shutil.copyfile(folder / item['file'], chunk / item['file'])
                        record, metadata = q0[label], read(folder / 'metadata.json')
                        cached, paired = cached + 1, False
                    else:
                        record, metadata = runtime.predict('milk', image, seed, chunk)
                        fresh += 1
                        paired = query in later_noise
                        if paired:
                            runtime.require_exact(record['pure_noise'], later_noise[query], 'actual later195 whole two-draw pair/' + str(query))
                            pairs += 1
                        else:
                            later_noise[query] = record['pure_noise']
                    scan.validate_record(record, torch)
                    native_label = f"{scene}_V{rule['q0_vision_noise_source_seed']}_A195_native"
                    for key in ('timesteps', 'sigmas'):
                        runtime.require_exact(record[key], q0[native_label][key], 'complete native schedule/' + key)
                    assert metadata['seed'] == seed and metadata['model_calls'] == 30 and metadata['prepare_calls'] == 1
                    assert hooks(model) == initial_hooks
                    metadata.update(trial=label, query=query, scene=scene, shift_cm=SHIFTS[scene], image_path=str(image),
                        input_png_sha256=sha(image), saved_q0_prediction=query == 0,
                        current_execution_model_calls=0 if query == 0 else 30, source_record_model_calls=30 if query == 0 else None,
                        saved_q0_attention_arm=rule['arm'], saved_q0_was_masked=source['masked_attention'],
                        saved_q0_window=rule['selected_window'], q0_vision_noise_source_seed=rule['q0_vision_noise_source_seed'],
                        q0_action_noise_source_seed=195, q0_runtime_rng_seed=195, later_noise_base=195,
                        actual_query_seeds=rule['actual_query_seeds'], seed_rule=rule['seed_rule'],
                        component_intervention_in_this_execution=False, attention_mask_intervention_in_this_execution=False,
                        initial_noise_replacement_in_this_execution=False, paired_noise_checked_exact=paired,
                        saved_q0_source_states_sha256=source['files_sha256']['states.pt'] if query == 0 else None)
                    write(chunk / 'metadata.json', metadata)
                    assert read(chunk / 'normalized_actions.json') == record['actions'].tolist()
                    served.append((label, query))
                    write(destination / 'progress.json', dict(queries=len(served), saved_q0_predictions=cached, native_predictions=fresh,
                        fresh_model_forwards=30 * fresh, actual_later_whole_noise_pairs=pairs, served_queries=served,
                        elapsed_s=time.perf_counter() - started))
                    print('[FUTURE-CURRENT-POSITION-EXECUTION] ' + json.dumps(dict(requests=len(served), case=label, query=query, seed=seed)), flush=True)
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
        while not failed and len(served) < 96:
            http.handle_request()
            assert not (out / 'closed-loop/failed.json').exists() and not (out / 'wrapper_failed.json').exists()
        assert not failed and len(served) == 96 and cached == 12 and fresh == 84 and pairs == 77
        assert hooks(model) == initial_hooks
        write(destination / 'complete.json', dict(state='complete', queries=96, saved_q0_predictions=12, native_predictions=84,
            fresh_model_forwards=2520, actual_later_whole_noise_pairs=77, q0_whole_noise_pair_checks=0,
            extra_q0_model_forwards=0, final_model_hooks_none=True, component_hooks_installed=False,
            attention_dispatch_wrapper_installed=False, files_sha256={name: sha(destination / name)
                for name in ('provenance.json', 'saved-q0-controls.json')}, scope=SCOPE), exclusive=True)
    finally:
        if http is not None:
            http.server_close()


def simulate(baseline):
    simulator, frozen, _, helper, out = prepared(baseline)
    assert not (out / 'closed-loop').exists(), 'No partial output retry'
    os.environ.setdefault('MUJOCO_GL', 'egl')
    sys.path.insert(0, '/home/current/work/openpi-demo/cosmos-policy')
    sys.path.insert(0, str(ROOT / 'cosmos-framework'))
    import requests
    from cosmos_framework.simulation.libero.closed_loop_eval import _framewise_action_to_delta, _remap_gripper
    collector = simulator.load_file('position_execution_collector', ROOT / 'work/collect_cosmos_goal_pair.py')
    rollout = simulator.load_file('position_execution_controller', ROOT / 'work/rollout_cosmos_goal_pair.py')
    np = simulator.np
    original_post, original_measure, original_summary = requests.post, simulator.measure, simulator.summarize_trial
    active, entries, windows, sham_controls = None, [], [], []

    def measure(env, obs, step, controller):
        result = original_measure(env, obs, step, controller)
        if step == 16:
            np.savez_compressed(out / 'closed-loop' / active / 'controller-step16.npz', **controller.controller_state(env.env))
        return result

    def post(url, *args, **kwargs):
        request = kwargs['json']
        assert url == frozen['url'] + '/predict' and request['trial'] == active
        query, rule = request['query'], frozen['trials'][active]
        assert request['seed'] == rule['actual_query_seeds'][query]
        if query == 1:
            folder, source = out / 'closed-loop' / active, frozen['q0_sources'][active]
            initial_folder = out / 'inputs' / rule['scene']
            state = np.load(folder / 'sim_states.npy', allow_pickle=False)
            initial = np.load(initial_folder / 'state.npy', allow_pickle=False)
            assert state.shape == (17, initial.size) and state.dtype == initial.dtype
            assert state[0].tobytes(order='C') == initial.tobytes(order='C')
            norm, raw = np.asarray(source['normalized_actions']), np.asarray(source['raw_actions'])
            native = np.asarray([_remap_gripper(_framewise_action_to_delta(item, '6d').tolist(), 'zero_one') for item in raw])
            for name, value in {'normalized_actions.json': norm, 'denormalized_actions.json': raw,
                                'actions.json': native, 'executed_actions.json': np.clip(native, -1, 1)}.items():
                assert rollout.array_exact(np.asarray(read(folder / name)), value), ('own cached q0 execution', active, name)
            trajectory = read(folder / 'trajectory.json')
            assert len(trajectory) == 17 and [item['step'] for item in trajectory] == list(range(17))
            assert read(folder / 'initial_contact_checks.json') == read(initial_folder / 'initial_contact_checks.json')
            assert all(item['exact'] for item in read(folder / 'initial_checks.json'))
            assert all(value is True for value in read(folder / 'initial_controller_checks.json').values())
            with np.load(folder / 'initial_controller.npz', allow_pickle=False) as current:
                with np.load(initial_folder / 'controller.npz', allow_pickle=False) as reference:
                    assert set(current.files) == set(reference.files)
                    assert all(rollout.array_exact(current[key], reference[key]) for key in current.files)
            gate = dict(case=active, scene=rule['scene'], saved_attention_arm=rule['arm'], selected_window=rule['selected_window'],
                all_16_own_cached_q0_normalized_raw_native_clipped_actions_byte_exact=True,
                own_scene_initial_sim_state_controller_capture_and_contacts_exact=True,
                actual_prefix_state_records=17, actual_prefix_files_sha256={name: sha(folder / name)
                    for name in ('sim_states.npy', 'trajectory.json', 'input_01.png', 'initial_controller.npz', 'controller-step16.npz')},
                old_X12_controller_or_trajectory_reference_used=False)
            write(folder / 'q1-entry-control.json', gate, exclusive=True)
            entries.append(gate)
        return original_post(url, *args, **kwargs)

    def summary(trial, scene, seed, records, controller):
        rule = frozen['trials'][trial]
        result = original_summary(trial, scene, int(seed), records, controller)
        result.update(saved_q0_attention_arm=rule['arm'], saved_q0_window=rule['selected_window'],
            q0_vision_noise_source_seed=rule['q0_vision_noise_source_seed'], q0_action_noise_source_seed=195,
            q0_runtime_rng_seed=195, later_noise_base=195, noise_seed_base=195,
            actual_query_seeds=rule['actual_query_seeds'], seed_rule=rule['seed_rule'])
        return result

    requests.post, simulator.measure, simulator.summarize_trial = post, measure, summary
    try:
        for label, scene, v, arm in PLAN:
            active = label
            if arm == 'early_l1_9':
                assert len(sham_controls) == 4
            # One call per case resets the old seed-only seen_noise. Q0 is
            # unpaired; the server checks all77 actual later whole draw pairs.
            simulator.execute_trials(out, [label], collector, rollout)
            folder = out / 'closed-loop' / label
            actual = read(folder / 'summary.json')
            gate = helper.strict_windows(read(folder / 'trajectory.json'), actual, np)
            write(folder / 'strict-windows.json', gate, exclusive=True)
            windows.append(dict(case=label, scene=scene, arm=arm, selected_window=WINDOWS.get(arm),
                selected_objects=actual['selected_objects'], files_sha256={name: sha(folder / name)
                    for name in ('strict-windows.json', 'summary.json', 'trajectory.json')}))
            if arm == 'all_allowed':
                native_label = f'{scene}_V{v}_A195_native'
                native_objects = read(out / 'closed-loop' / native_label / 'summary.json')['selected_objects']
                changed = native_objects != actual['selected_objects']
                sham_controls.append(dict(case=label, scene=scene, native_reference_case=native_label,
                    native_selected_objects=native_objects, sham_selected_objects=actual['selected_objects'],
                    classification_changed=changed, original_native_path_interpretation_gate_passed=not changed,
                    numerical_sham_is_native_noop=False))
        done, served = read(out / 'closed-loop/complete.json'), read(out / 'server/complete.json')
        assert done['state'] == served['state'] == 'complete'
        assert done['cases'] == 12 and done['completed_trials'] == [item[0] for item in PLAN]
        assert done['steps_per_case'] == 128 and done['state_records_per_case'] == 129
        assert len(sham_controls) == 4 and len(entries) == len(windows) == 12
        assert served['queries'] == 96 and served['saved_q0_predictions'] == 12 and served['native_predictions'] == 84
        assert served['fresh_model_forwards'] == 2520 and served['actual_later_whole_noise_pairs'] == 77
        assert served['q0_whole_noise_pair_checks'] == served['extra_q0_model_forwards'] == 0
        assert served['final_model_hooks_none'] is True
        for name, expected in served['files_sha256'].items():
            assert sha(out / 'server' / name) == expected
        primary_results = []
        for label, scene, v, arm in PLAN[8:]:
            actual = read(out / 'closed-loop' / label / 'summary.json')
            native_label, sham_label = f'{scene}_V{v}_A195_native', f'{scene}_V{v}_A195_all_allowed'
            native = read(out / 'closed-loop' / native_label / 'summary.json')
            sham = read(out / 'closed-loop' / sham_label / 'summary.json')
            only_milk = actual['selected_objects'] == ['milk_1']
            same = native['selected_objects'] == sham['selected_objects']
            native_milk = native['selected_objects'] == ['milk_1']
            effect = ('sham_classification_gate_failed' if not same else
                      'did_not_strictly_select_only_milk' if not only_milk else
                      'preserved_native_milk' if native_milk else 'rescued_failed_native')
            primary_results.append(dict(case=label, scene=scene, vision_noise_source_seed=v,
                selected_objects=actual['selected_objects'], strict_only_milk=only_milk,
                same_scene_V_native=native_label, same_scene_V_sham=sham_label,
                native_selected_objects=native['selected_objects'], sham_selected_objects=sham['selected_objects'],
                sham_matches_native_classification=same, native_already_only_milk=native_milk,
                effect=effect, strict_windows_sha256=sha(out / 'closed-loop' / label / 'strict-windows.json')))
        all_only_milk = all(item['strict_only_milk'] for item in primary_results)
        sham_gate = all(not item['classification_changed'] for item in sham_controls)
        supported = all_only_milk and sham_gate
        assert [item['case'] for item in primary_results] == frozen['primary_cases']
        write(out / 'primary-prediction.json', dict(state='evaluated_after_all12_complete', primary_window='early_l1_9',
            criterion='Each of four cut cases selected_objects == [milk_1] by both-finger contact and lift >.02m for five consecutive records AND all four same-scene/V shams match native classifications.',
            cases=primary_results, prediction='supported' if supported else 'rejected', all_four_only_milk=all_only_milk,
            numerical_sham_classification_gate_passed=sham_gate, all12_cases_executed=True,
            alternate_window_or_descriptive_metric_cannot_replace_primary=True,
            native_milk_preservation_is_not_rescue=True, functional_specificity_established=False, scope=SCOPE), exclusive=True)
        case_files = {}
        for label, _, _, _ in PLAN:
            folder = out / 'closed-loop' / label
            names = ('sim_states.npy', 'actual.mp4', 'initial_controller.npz', 'controller-step16.npz', 'summary.json',
                'strict-windows.json', 'q1-entry-control.json', 'query_contract.json', 'initial_checks.json',
                'initial_controller_checks.json', 'initial_contact_checks.json', *PHYSICAL,
                *[f'input_{q:02d}.png' for q in range(8)],
                *[f'chunk_{q:02d}/{name}' for q in range(8) for name in ('states.pt', 'metadata.json', 'normalized_actions.json')],
                *[f'chunk_00/{name}' for name in frozen['q0_files'] if name not in ('states.pt', 'metadata.json', 'normalized_actions.json')],
                *[f"chunk_00/{item['file']}" for item in frozen['q0_sources'][label]['boundary_files']])
            case_files[label] = {name: sha(folder / name) for name in names}
        write(out / 'controls.json', dict(q1_entry_controls=entries, strict_windows=windows,
            sham_classification_controls=sham_controls, original_native_path_interpretation_gate_passed=sham_gate,
            primary_prediction_sha256=sha(out / 'primary-prediction.json'), actual_case_files_sha256=case_files), exclusive=True)
        files = ('controls.json', 'primary-prediction.json', 'provenance.json', 'prepared.json',
                 'server/complete.json', 'closed-loop/complete.json', 'summary.json')
        write(out / 'complete.json', dict(state='complete', script_sha256=sha(Path(__file__)), trial_order=[item[0] for item in PLAN],
            physical_trials=12, saved_q0_predictions=12, native_predictions=84, fresh_model_forwards=2520,
            producer_q0_forwards_separate=360, combined_two_stage_model_forwards=2880, actual_later_whole_noise_pairs=77,
            all12_own_cached_q0_action_execution_and_strict_window_gates_passed=True,
            all12_cases_executed_without_q0_selection=True, primary_prediction='supported' if supported else 'rejected',
            primary_all_four_only_milk=all_only_milk, numerical_sham_classification_gate_passed=sham_gate,
            native_milk_preservation_is_not_rescue=True, functional_specificity_established=False,
            files_sha256={name: sha(out / name) for name in files}, scope=SCOPE), exclusive=True)
    finally:
        requests.post, simulator.measure, simulator.summarize_trial = original_post, original_measure, original_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('prepare', 'server', 'simulate'), required=True)
    args = parser.parse_args()
    baseline, out = args.output.resolve(), args.output.resolve() / 'future-current-position-execution'
    if args.mode == 'prepare':
        _, frozen, _, _ = contract(baseline)
        out.mkdir(exist_ok=False)
        (out / 'inputs').symlink_to(frozen['inputs_symlink_target'], target_is_directory=True)
        write(out / 'provenance.json', frozen, exclusive=True)
        write(out / 'prepared.json', dict(state='prepared', provenance_sha256=sha(out / 'provenance.json')), exclusive=True)
        print('[FUTURE-CURRENT-POSITION-EXECUTION] prepared ' + str(out), flush=True)
        return
    try:
        server(baseline) if args.mode == 'server' else simulate(baseline)
    except Exception as exc:
        if out.is_dir() and not (out / 'wrapper_failed.json').exists():
            write(out / 'wrapper_failed.json', dict(mode=args.mode, error=str(exc), traceback=traceback.format_exc()), exclusive=True)
        raise


if __name__ == '__main__':
    main()
