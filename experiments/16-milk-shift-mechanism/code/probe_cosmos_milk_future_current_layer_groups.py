"""Twelve q0 predictions testing four early nine-layer future-to-current cuts.

Use --output BASE; create only absent BASE/future-current-layer-groups.
Reuse the frozen 8fef window observer, c106 full-GEN dispatcher and 9081
initial-noise seam. Repeat two native and two global all-allowed controls,
then cut denoising0..14 in layers0..8,9..17,18..26,27..35 for both V sources
with fixed A195. All ten masked cases keep the same union66 all-allowed
background at every layer/step; only 135 selected sites additionally cut F→C.
This producer runs 360 model forwards / 37800 official attention dispatches,
no new physics. All twelve physical trials remain preregistered.

The only primary physical prediction is early_l1_9: BOTH cases strictly
select only milk. This is one bounded follow-up to the REJECTED eighteen-
layer prediction; other groups cannot replace it, and no unlimited bisection
is authorized. Higher groups are same-size layer comparisons, not known
semantically irrelevant controls. Hard masking also reallocates attention.
"""

import argparse
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import time
import traceback
from types import SimpleNamespace


ROOT = Path('/home/current/work/cosmos3')
BASE_PROBE_SHA = 'c106bba687c6eaf0853ac9abe7aa2767f9464365162399c201e98b2369af7a4a'
WINDOW_PROBE_SHA = '8fef06cae12291884ee8a8fa60c981605f7813ccaa1c372f76f426be9662eb69'
WINDOW_PHYSICS_SHA = '19b907e5760f5873f4b1d0b834018a594f8d7420827369e731bc701ae4bcb08a'
WINDOW_COMPLETE_SHA = 'b77ba339d18f5598d4ebbb520ccc6cb6cd2f685ed03e6b84d38d0a97087404bd'
PHYSICAL_COMPLETE_SHA = '326f29ce22476756fbdcafc796eed083231258f70eb86f974eaaab39ee9dfc4a'
PROMPT = 'pick up the milk and place it in the basket'
WINDOWS = {'early_l1_9': dict(steps=[0, 14], layers=[0, 8]),
           'early_l10_18': dict(steps=[0, 14], layers=[9, 17]),
           'early_l19_27': dict(steps=[0, 14], layers=[18, 26]),
           'early_l28_36': dict(steps=[0, 14], layers=[27, 35])}
PLAN = tuple((f'V{v}_A195_{arm}', v, arm) for arm in ('native', 'all_allowed', *WINDOWS) for v in (195, 198))
FILES = ('states.pt', 'normalized_actions.json', 'metadata.json', 'noise-audit.pt', 'noise-audit.json',
         'observer-report.json', 'indexes-and-masks.pt')
HYPOTHESIS_ZH = '18层剂量太大，同时损害正确抓取；缩到最前9层应保留A纠错而保护B。这是新假设，不能覆盖旧18层主要预测rejected。'
FOLLOW_UP = dict(one_fixed_four_group_test=True, no_unlimited_bisection=True,
    if_primary_rejected='Report rejected; no other group becomes primary. Content versus softmax reallocation remains a separate possible next branch.',
    if_primary_supported='Freeze this range immediately; next test X+15cm and new positions not used to choose it.',
    no_next_experiment_executed_by_this_producer=True)
SCOPE = ('One preregistered four-way partition of early future-to-current cuts in a common '
         'masked arithmetic background, fixed A195 and two V realizations at X+12cm. '
         'Higher layer groups are same-site-count comparisons, not known irrelevant semantic controls; '
         'equal 135-site budgets do not imply equal output perturbation. The old eighteen-layer '
         'primary remains rejected. Actual q0 inputs, full outputs and current/action boundaries '
         'only. Hard masking also renormalizes remaining attention; content and allocation effects '
         'are not separated. No training, new physics, semantic unit or neural root-cause claim.')


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


def cut_active(window, step, layer):
    return (window is not None and window['steps'][0] <= step <= window['steps'][1]
            and window['layers'][0] <= layer <= window['layers'][1])


def source_contract(baseline):
    window_path = ROOT / 'work/probe_cosmos_milk_future_current_windows.py'
    physics_path = ROOT / 'work/run_cosmos_milk_future_current_execution.py'
    assert sha(window_path) == WINDOW_PROBE_SHA and sha(physics_path) == WINDOW_PHYSICS_SHA
    old = load('future_current_layer_groups_frozen_windows', window_path)
    frozen, factor, contract, original_sources, image, paths, upstream = old.source_contract(baseline)
    assert old.BASE_PROBE_SHA == BASE_PROBE_SHA
    producer = baseline / 'future-current-windows'
    physics = baseline / 'future-current-execution'
    assert not (producer / 'failed.json').exists() and not (physics / 'wrapper_failed.json').exists()
    assert not (physics / 'server/failed.json').exists() and not (physics / 'closed-loop/failed.json').exists()
    assert sha(producer / 'complete.json') == WINDOW_COMPLETE_SHA
    assert sha(physics / 'complete.json') == PHYSICAL_COMPLETE_SHA
    docs = {name: read(producer / (name + '.json')) for name in
        ('complete', 'results', 'protocol', 'provenance', 'sources', 'primary-prediction')}
    complete, results, protocol = (docs[name] for name in ('complete', 'results', 'protocol'))
    assert complete['state'] == results['state'] == 'complete'
    assert complete['script_sha256'] == docs['provenance']['script_sha256'] == WINDOW_PROBE_SHA
    for name in ('results', 'protocol', 'provenance', 'sources'):
        assert complete[name + '_sha256'] == sha(producer / (name + '.json'))
    assert complete['primary_prediction_sha256'] == results['primary_prediction_sha256'] == sha(producer / 'primary-prediction.json')
    assert complete['fresh_q0_predictions'] == results['fresh_q0_predictions'] == 12
    assert complete['fresh_model_forwards'] == results['fresh_model_forwards'] == 360
    assert complete['total_official_dispatch_calls'] == results['total_official_dispatch_calls'] == 38880
    assert complete['original_dispatch_calls'] == 25920 and complete['extra_dispatch_calls'] == 12960
    assert complete['original_randn_calls_consumed'] == complete['returned_source_draws_observed'] == 24
    for key in ('two_native_entire_prior_records_exact', 'two_sham_entire_prior_records_exact',
                'all_four_control_all30_full_tuples_and_boundaries_exact', 'all8_windows_before_first_cut_own_sham_prefix_exact',
                'all_active_cut_sites_unblocked_GEN_vs_same_input_allallowed_exact', 'all10_masked_full1080_sites_return_common_union66',
                'all360_actual_complete_model_kwargs_saved', 'all_original_symbols_restored', 'all_owned_hooks_removed'):
        assert complete[key] is True
    assert complete['physics_calls'] == complete['later_query_predictions'] == results['physics_calls'] == 0
    assert complete['primary_physical_prediction_evaluated'] is results['primary_physical_prediction_evaluated'] is False
    assert docs['sources'] == dict(source_evidence=upstream, original_sources=original_sources, frozen_contract=contract)
    assert protocol['windows_zero_based_inclusive'] == old.WINDOWS
    assert protocol['trial_order'] == results['preregistered_physics_cases'] == [item[0] for item in old.PLAN]
    assert docs['primary-prediction']['status'] == 'not_evaluated_q0_only'
    assert docs['primary-prediction']['cases'] == ['V195_A195_early_low', 'V198_A195_early_low']
    assert docs['primary-prediction']['both_selected_objects_must_equal'] == ['milk_1']
    assert results['official_dispatch_counts'] == dict(model_before=360, model_after=360, native_UND_dispatch=12960,
        native_GEN_dispatch=12960, all_allowed_dispatch=10800, cut_dispatch=2160)
    assert [row['case'] for row in results['cases']] == [item[0] for item in old.PLAN]
    q0_cases = {}
    for row, (label, v, arm) in zip(results['cases'], old.PLAN):
        folder = producer / label
        window, controlled = old.WINDOWS.get(arm), arm in ('native', 'all_allowed')
        assert row['arm'] == arm and row['window'] == window and row['vision_noise_source_seed'] == v
        assert row['action_noise_source_seed'] == row['runtime_rng_seed'] == 195 and row['fresh_model_forwards'] == 30
        assert row['existing_entire_control_record_exact'] is (True if controlled else None)
        assert row['own_sham_pre_first_cut_prefix_exact'] is (True if window else None)
        assert row['physical_prediction_evaluated'] is False and set(row['files_sha256']) == set((*FILES, 'control-checks.json'))
        for name, expected in row['files_sha256'].items():
            assert sha(folder / name) == expected
        report = read(folder / 'observer-report.json')
        assert report['counts'] == row['actual_dispatch_counts']
        assert report['counts']['cut_dispatch'] == (270 if window else 0)
        assert report['counts']['all_allowed_dispatch'] == (0 if arm == 'native' else 1080)
        assert report['state'] == 'complete' and report['case_arm'] == arm and report['selected_window'] == window
        assert len(row['boundary_files']) == 30 and report['boundary_files'] == row['boundary_files']
        for step, item in enumerate(row['boundary_files']):
            assert item['step'] == step and item['file'] == f'boundaries-t{step:02d}.pt'
            assert item['current_shape'] == [37, 50, 4096] and item['action_shape'] == [37, 16, 4096]
            assert sha(folder / item['file']) == item['sha256']
        q0_cases[label] = dict(folder=str(folder), arm=arm, window=window,
            files_sha256=row['files_sha256'], boundary_files=row['boundary_files'])
    pc, controls, summary, primary, pprov = (read(physics / (name + '.json')) for name in
        ('complete', 'controls', 'summary', 'primary-prediction', 'provenance'))
    assert pc['state'] == 'complete' and pc['script_sha256'] == pprov['script_sha256'] == WINDOW_PHYSICS_SHA
    for name, expected in pc['files_sha256'].items():
        assert sha(physics / name) == expected
    producer_hashes = {name: sha(producer / (name + '.json')) for name in docs}
    assert pprov['producer_script_sha256'] == WINDOW_PROBE_SHA and pprov['producer_documents_sha256'] == producer_hashes
    assert pprov['source_evidence'] == upstream
    assert pc['physical_trials'] == pc['saved_q0_predictions'] == 12 and pc['native_predictions'] == 84
    assert pc['fresh_model_forwards'] == 2520 and pc['combined_two_stage_model_forwards'] == 2880
    assert pc['actual_later_whole_noise_pairs'] == 77 and pc['producer_q0_forwards_separate'] == 360
    for key in ('four_native_and_sham_entire_129_state_5JSON_8PNG_and_8_modelrecords_exact',
                'all12_own_cached_q0_action_execution_and_strict_window_gates_passed',
                'all12_cases_executed_without_q0_selection', 'numerical_sham_classification_gate_passed'):
        assert pc[key] is True
    assert pc['primary_prediction'] == primary['prediction'] == 'rejected'
    assert pc['primary_both_only_milk'] is primary['both_only_milk'] is False
    assert primary['state'] == 'evaluated_after_all12_complete' and primary['primary_window'] == 'early_low'
    assert primary['all12_cases_executed'] is primary['other_windows_cannot_replace_primary'] is True
    assert primary['descriptive_milk_inclusion_or_max_lift_cannot_replace_primary'] is True
    assert controls['primary_prediction_sha256'] == pc['files_sha256']['primary-prediction.json']
    labels = [item[0] for item in old.PLAN]
    assert pc['trial_order'] == summary['completed_trials'] == summary['default_trials'] == labels
    assert [row['trial'] for row in summary['cases']] == labels
    assert set(controls['actual_case_files_sha256']) == set(labels)
    physical_cases, previous_results = {}, []
    for row, (label, v, arm) in zip(summary['cases'], old.PLAN):
        folder = physics / 'closed-loop' / label
        hashes = controls['actual_case_files_sha256'][label]
        for name, expected in hashes.items():
            assert sha(folder / name) == expected
        assert row == read(folder / 'summary.json')
        assert row['steps'] == 128 and row['queries'] == 8 and row['state_records'] == 129
        assert hashes['chunk_00/states.pt'] == q0_cases[label]['files_sha256']['states.pt']
        assert hashes['chunk_00/normalized_actions.json'] == q0_cases[label]['files_sha256']['normalized_actions.json']
        physical_cases[label] = dict(folder=str(folder), files_sha256=hashes)
        previous_results.append(dict(case=label, vision_noise_source_seed=v, action_noise_source_seed=195,
            arm=arm, window=old.WINDOWS.get(arm), selected_objects=row['selected_objects'],
            first_selection_step=row['first_selection_step'], first_selected_object=row['first_selected_object'],
            per_object={name: dict(first_2cm_for_5frames=value['first_2cm_for_5frames'], max_lift_cm=value['max_lift_cm'])
                        for name, value in row['per_object'].items()}, physical_folder=str(folder),
            summary_sha256=hashes['summary.json'], strict_windows_sha256=hashes['strict-windows.json']))
    selected = {row['case']: row['selected_objects'] for row in previous_results}
    expected_primary = ['V195_A195_early_low', 'V198_A195_early_low']
    assert [row['case'] for row in primary['cases']] == expected_primary
    for row in primary['cases']:
        assert row['selected_objects'] == selected[row['case']]
        assert row['strict_only_milk'] is (row['selected_objects'] == ['milk_1'])
        assert row['strict_windows_sha256'] == physical_cases[row['case']]['files_sha256']['strict-windows.json']
    assert selected[expected_primary[0]] == ['milk_1'] and selected[expected_primary[1]] == []
    for name in ('server/complete.json', 'closed-loop/complete.json'):
        nested = read(physics / name)
        assert nested['state'] == 'complete'
        for relative, expected in nested.get('files_sha256', {}).items():
            assert sha((physics / name).parent / relative) == expected
    sources = dict(upstream,
        previous_eighteen_layer_stages=dict(producer_script_sha256=WINDOW_PROBE_SHA,
            physics_script_sha256=WINDOW_PHYSICS_SHA, producer_documents_sha256=producer_hashes,
            physical_complete_sha256=sha(physics / 'complete.json'), physical_files_sha256=pc['files_sha256'],
            q0_cases=q0_cases, physical_cases=physical_cases, actual_twelve_physical_results=previous_results,
            old_primary_prediction='rejected', old_primary_both_only_milk=False,
            comparison_scope='Existing eighteen-layer joint cuts, never linearly summed with new nine-layer effects. Subsequent dynamic inputs may differ; no additive contribution/mediation decomposition.'))
    return old, frozen, factor, contract, original_sources, image, paths, sources


def observer_class(old, frozen):
    Parent = old.observer_class(frozen)

    class LayerGroupObserver(Parent):
        def __init__(self, *args, case_arm, **kwargs):
            super().__init__(*args, case_arm=case_arm, **kwargs)
            # Parent's dispatch and real boundary exporter use self.window.
            # Keep its frozen module globals, native dispatcher and hooks intact.
            self.window = WINDOWS.get(case_arm)

        def report(self, record):
            assert self.restored and not self.handles and not self.in_model and self.active is None
            assert self.counts['model_before'] == self.counts['model_after'] == 30
            for key in ('block_before', 'block_after', 'pre_W', 'native_UND_dispatch', 'native_GEN_dispatch'):
                assert self.counts[key] == 1080
            assert self.counts['all_allowed_dispatch'] == (0 if self.case_arm == 'native' else 1080)
            assert self.counts['cut_dispatch'] == (135 if self.window else 0)
            assert self.counts['substituted_sites'] == (0 if self.case_arm == 'native' else 1080)
            assert len(self.calls) == len(self.files) == len(self.outputs) == 30
            self.runtime.require_exact(self.torch.stack(self.outputs), record['action_velocity'], 'actual tuple output equals action head at every step')
            assert self.reader.settings(self.torch, self.model.layers[0].self_attn.processor) == self.settings
            return dict(state='complete', case_arm=self.case_arm, selected_window=self.window, counts=self.counts,
                boundary_files=self.files, indexes_and_masks_sha256=sha(self.folder / 'indexes-and-masks.pt'),
                original_full_266_row_projection_calls=1080, all_actual_dispatch_flatten_pre_W_byte_exact=True,
                all_window_cut_unblocked_GEN_rows_vs_same_input_all_allowed_exact=True if self.window else None,
                all_masked_sites_same_union66=True if self.case_arm != 'native' else None,
                future200_and_UND_native_dispatch_returns_preserved=True, all30_actual_complete_model_kwargs_saved=True,
                processor_recompute_calls=0, all_owned_hooks_removed=True, original_dispatch_symbol_restored=True,
                numerical_sham_is_native_noop=False, scope=SCOPE)
    return LayerGroupObserver


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Completed baseline experiment root')
    baseline = parser.parse_args().output.resolve()
    old, frozen, factor, contract, original_sources, image, paths, sources = source_contract(baseline)
    out = baseline / 'future-current-layer-groups'
    out.mkdir(exist_ok=False)
    protocol = dict(plan=[dict(case=label, vision_noise_source_seed=v, action_noise_source_seed=195, runtime_rng_seed=195,
        arm=arm, window=WINDOWS.get(arm)) for label, v, arm in PLAN],
        trial_order=[label for label, _, _ in PLAN], windows_zero_based_inclusive=WINDOWS,
        hypothesis_zh=HYPOTHESIS_ZH, follow_up_rule=FOLLOW_UP, previous_eighteen_layer_primary_prediction='rejected',
        groups_are_same_size_layer_comparisons_not_semantically_irrelevant_controls=True,
        primary_window='early_l1_9', primary_cases=['V195_A195_early_l1_9', 'V198_A195_early_l1_9'],
        primary_physical_prediction='Each primary case selected_objects == [milk_1] under both fingerpad contacts plus >.02m lift for five consecutive records; V195 changes from cheese to only milk and V198 retains only milk.',
        primary_evaluation=dict(status='unexecuted_physics_prediction', success_requires_both_V_only_milk=True,
            milk_inclusion_max_lift_and_other_windows_are_descriptive_not_primary=True,
            rejection_does_not_promote_another_window=True),
        image_path=str(image), image_sha256=sha(image), prompt=PROMPT, shift_cm=12,
        q0_predictions=12, model_forwards=360, physics_calls=0, original_dispatch_calls=25920,
        extra_all_allowed_dispatch_calls=10800, extra_cut_dispatch_calls=1080, total_official_dispatch_calls=37800,
        cut_sites_per_window_case=135, masked_sites_per_case=1080,
        masked_background='All ten masked cases replace union66 by same-input all-allowed on every layer/step; window sites instead select cut-current output for union66. Future200/UND retain native; action-to-current is not blocked.',
        cut_gate='All unblocked GEN query rows including future200 must byte-equal same-input all-allowed at every active cut, using frozen c106 original full-shape dispatch. No residual repair.',
        control_gates='First two native ENTIRE q0 records exact prior native; next two all-allowed ENTIRE q0 records exact prior sham. All30 full tuple outputs and actual current/action boundaries also match those controls.',
        prefix_gate='Window before first active cut matches same-V fresh sham in entire actual model kwargs, earlier complete model outputs/head, FP32 samples and saved current/action boundary prefix.',
        boundary_saved_steps=list(range(30)), public_selected_steps=[0, 10, 20, 29],
        saved_actual_inputs='Complete actually observed model kwargs at all30 steps, no solver state inferred from BF16 kwargs; exact native FP32 schedule stored separately.',
        noise_pair_rule='Within each V all six raw returned V/A pairs equal; across V only A195 is equal. Each original195 draw is consumed before frozen whole-source return.',
        later_physics_preregistered=dict(cases=12, trial_order=[label for label, _, _ in PLAN], cached_q0=12,
            later_noise_base=195, q1_to_q7_seeds=list(range(196, 203)), later_predictions=84, later_model_forwards=2520,
            total_two_stage_model_forwards=2880, no_q0_score_or_rank_selection=True,
            native_and_allallowed_entire129_state_5JSON_8PNG_and_all8_query_records_must_equal_existing_references=True,
            physical_control_references=sources['physical_control_references'], all_four_windows_all_V_reported=True),
        no_training=True, no_residual_compensation=True, scope=SCOPE)
    write(out / 'protocol.json', protocol, exclusive=True)
    write(out / 'sources.json', dict(source_evidence=sources, original_sources=original_sources, frozen_contract=contract), exclusive=True)
    started, completed, rows, observer, noise, runtime = time.perf_counter(), [], [], None, None, None
    counts = dict(model_before=0, model_after=0, native_UND_dispatch=0, native_GEN_dispatch=0, all_allowed_dispatch=0, cut_dispatch=0)

    def progress(stage, **values):
        value = dict(stage=stage, completed=completed, expected_model_forwards=360, actual_counts=counts,
            elapsed_s=time.perf_counter() - started, **values)
        write(out / 'progress.json', value)
        print('[FUTURE-CURRENT-LAYER-GROUPS] ' + json.dumps(value), flush=True)

    try:
        os.environ['HF_HUB_OFFLINE'] = '1'
        base = load('future_current_layer_groups_runtime', paths['normal_runtime'])
        base.TASKS['milk'] = (7, 'pick_up_the_milk_and_place_it_in_the_basket', PROMPT, 'milk_1')
        factory = load('future_current_layer_groups_factory', paths['factory'])
        factory.OUT = out
        assert sha(factory.SCHEDULER) == contract['frozen_files']['scheduler']['sha256']
        progress('loading_model')
        runtime = base.NormalRuntime(factory)
        torch, model = runtime.torch, runtime.pipe.transformer
        assert not model.training and not model.is_cache_enabled and not torch.is_grad_enabled()
        anchors = sources['anchors']
        assert sha(Path(inspect.getfile(type(model)))) == anchors['upstream']['actual_transformer_sha256']
        assert sha(Path(inspect.getfile(type(runtime.pipe)))) == anchors['upstream']['actual_pipeline_source_sha256']
        components, scan, reader = (load('future_current_layer_groups_' + key, paths[key]) for key in ('components', 'metrics', 'reader'))
        checker = components.CosmosComponentInterventions(model)
        processor_module = inspect.getmodule(type(model.layers[0].self_attn.processor))
        dispatch = processor_module.dispatch_attention_fn
        dispatch_path = Path(inspect.getfile(dispatch))
        assert sha(dispatch_path) == frozen.DISPATCH_SHA
        assert hashlib.sha256(inspect.getsource(dispatch).encode()).hexdigest() == read(baseline / 'target-reader-inputs/provenance.json')['dispatch_function_source_sha256']
        dispatch_module = importlib.import_module('diffusers.models.attention_dispatch')
        assert dispatch_module.dispatch_attention_fn is dispatch
        registry = dispatch_module._AttentionBackendRegistry
        backend_name, backend_fn = registry.get_active_backend()
        supported = registry._supported_arg_names[backend_name]
        feasibility = anchors['feasibility_actual_registry']
        assert 'attn_mask' in supported and backend_name.value != '_native_flash'
        assert backend_name.value == feasibility['registry_backend'] and backend_fn.__qualname__ == feasibility['registry_backend_function']
        assert hashlib.sha256(inspect.getsource(backend_fn).encode()).hexdigest() == feasibility['registry_backend_function_source_sha256']
        assert sorted(supported) == feasibility['registry_supported_arguments']
        settings = reader.settings(torch, model.layers[0].self_attn.processor)
        for key, value in feasibility['settings'].items():
            assert settings[key] == value
        for block in model.layers:
            assert block.self_attn.processor._parallel_config is block.self_attn.processor._attention_backend is None
            assert reader.settings(torch, block.self_attn.processor) == settings
        original_records, original_caches, old_records = {}, {}, {}
        for v in (195, 198):
            folder = Path(original_sources[str(v)]['q0_folder'])
            original_records[v] = torch.load(folder / 'states.pt', map_location='cpu', weights_only=True, mmap=True)
            original_caches[v] = torch.load(folder / 'components.pt', map_location='cpu', weights_only=True, mmap=True)
            scan.validate_record(original_records[v], torch)
            assert torch.count_nonzero(original_records[v]['action_states'][..., 10:]) == 0
            scan.validate_cache(original_caches[v], original_records[v], checker, runtime, 'source' + str(v))
            for arm in ('native', 'all_allowed'):
                label = f'V{v}_A195_{arm}'
                directory = Path(sources['q0_controls'][label]['folder'])
                old_records[label] = torch.load(directory / 'states.pt', map_location='cpu', weights_only=True, mmap=True)
                scan.validate_record(old_records[label], torch)
        runtime.require_exact(original_caches[195]['steps'], original_caches[198]['steps'], 'source complete schedule/pack')
        clean_current = original_records[195]['model_input']['vision_tokens'][0][0, :, 0]
        runtime.require_exact(clean_current, original_records[198]['model_input']['vision_tokens'][0][0, :, 0], 'source current condition frame0')
        native_random = runtime.cm.randn_tensor
        provenance = dict(script_sha256=sha(Path(__file__)), protocol_sha256=sha(out / 'protocol.json'), sources_sha256=sha(out / 'sources.json'),
            inherited_fullcut_source_sha256=BASE_PROBE_SHA, inherited_window_observer_source_sha256=WINDOW_PROBE_SHA,
            completed_previous_window_physics_source_sha256=WINDOW_PHYSICS_SHA, noise_seam_source_sha256=frozen.FACTOR_SHA,
            source_sha256={key: sha(path) for key, path in paths.items()}, scheduler_sha256=sha(factory.SCHEDULER),
            actual_transformer_source=str(Path(inspect.getfile(type(model)))), actual_transformer_sha256=sha(Path(inspect.getfile(type(model)))),
            actual_pipeline_source_sha256=sha(Path(inspect.getfile(type(runtime.pipe)))), dispatch_source=str(dispatch_path),
            dispatch_source_sha256=sha(dispatch_path), dispatch_signature=str(inspect.signature(dispatch)),
            registry_backend=backend_name.value, registry_supported_arguments=sorted(supported),
            registry_backend_function_source_sha256=hashlib.sha256(inspect.getsource(backend_fn).encode()).hexdigest(),
            backend_settings=settings, library_edits=False, training=False, physics_calls=0,
            implementation='Frozen8fef actual kwargs/boundary observer and c106 dispatcher/masks/hooks/noise seam inherited. Only self.window and the135-site report override differ; previous sealed stages are checked here.', scope=SCOPE)
        write(out / 'provenance.json', provenance, exclusive=True)
        WindowObserver = observer_class(old, frozen)
        records = {}
        for label, v, arm in PLAN:
            if arm in WINDOWS:
                assert completed[:4] == [item[0] for item in PLAN[:4]] and all(row['existing_entire_control_record_exact'] is True for row in rows[:4])
            directory = out / label
            progress('fresh_q0', case=label)
            observer = WindowObserver(runtime, components, scan, reader, original_caches[195], clean_current,
                old_records[f'V{v}_A195_native'], 'native' if arm == 'native' else 'all_allowed', directory, case_arm=arm)
            noise = factor.InitialNoiseSources(runtime, scan, original_records, v, 195)
            assert observer.original is dispatch and noise.original is native_random
            observer.begin()
            try:
                noise.begin()
                record, metadata = runtime.predict('milk', image, 195, directory)
            finally:
                try:
                    noise.reset()
                finally:
                    observer.reset()
                    for key in counts:
                        counts[key] += observer.counts[key]
                    if directory.exists():
                        torch.save(dict(calls=noise.calls, model_steps=observer.calls, initial_model_kwargs=observer.first,
                            original_symbol_restored=noise.restored, observer_hooks_removed=not observer.handles,
                            dispatch_symbol_restored=observer.restored, actual_counts=observer.counts), directory / 'noise-audit.pt')
                        if observer.in_model:
                            torch.save(dict(step=observer.step, actual_model_kwargs=observer.actual_kwargs,
                                partial_current_action_boundaries=observer.boundaries, site_events=observer.events,
                                actual_counts=observer.counts), directory / 'partial-forward.pt')
            assert runtime.cm.randn_tensor is native_random and processor_module.dispatch_attention_fn is dispatch
            scan.validate_record(record, torch)
            assert metadata['model_calls'] == 30 and metadata['prepare_calls'] == 1 and metadata['seed'] == 195
            assert metadata['input_png_sha256'] == sha(image) and metadata['prompt'] == PROMPT and metadata['task_index'] == 7
            report = observer.report(record)
            prepared = factor.prepared_contract(record, original_records, v, 195, runtime,
                SimpleNamespace(first=observer.first, calls=observer.calls, handle=None))
            noise_report = noise.report(record)
            if arm in ('native', 'all_allowed'):
                reference = sources['q0_controls'][label]
                control = old.saved_control_checks(directory, Path(reference['folder']), record, old_records[label], runtime)
            else:
                sham_label = f'V{v}_A195_all_allowed'
                control = old.prefix_checks(directory, out / sham_label, record, records[sham_label], WINDOWS[arm], runtime)
            write(directory / 'control-checks.json', control, exclusive=True)
            write(directory / 'observer-report.json', report, exclusive=True)
            write(directory / 'noise-audit.json', dict(noise=noise_report, preparation=prepared), exclusive=True)
            metadata.update(case=label, arm=arm, window_zero_based_inclusive=WINDOWS.get(arm), query=0, shift_cm=12,
                vision_noise_source_seed=v, action_noise_source_seed=195, runtime_rng_seed=195,
                numerical_masked_sham=arm == 'all_allowed', native_noop=arm == 'native',
                common_union66_all_allowed_background=arm != 'native', window_cut_sites=observer.counts['cut_dispatch'],
                actual_masked_site_substitutions=observer.counts['substituted_sites'],
                primary_prediction_case=arm == 'early_l1_9', physical_prediction_evaluated=False,
                existing_entire_control_record_exact=True if arm in ('native', 'all_allowed') else None,
                own_sham_pre_first_cut_prefix_exact=True if arm in WINDOWS else None)
            write(directory / 'metadata.json', metadata)
            assert read(directory / 'normalized_actions.json') == record['actions'].tolist()
            rows.append(dict(case=label, arm=arm, window=WINDOWS.get(arm), vision_noise_source_seed=v, action_noise_source_seed=195,
                runtime_rng_seed=195, fresh_model_forwards=30, actual_dispatch_counts=observer.counts,
                existing_entire_control_record_exact=True if arm in ('native', 'all_allowed') else None,
                own_sham_pre_first_cut_prefix_exact=True if arm in WINDOWS else None,
                final_normalized_actions=record['actions'].tolist(), physical_prediction_evaluated=False,
                files_sha256={name: sha(directory / name) for name in (*FILES, 'control-checks.json')}, boundary_files=observer.files))
            records[label] = record
            completed.append(label)
            progress('case_complete', case=label)
            observer = noise = None
        for v in (195, 198):
            reference = records[f'V{v}_A195_native']['pure_noise']
            for label, actual_v, _ in PLAN:
                if actual_v == v:
                    runtime.require_exact(records[label]['pure_noise'], reference, 'within-V six entire returned noise pairs/' + label)
        runtime.require_exact(records['V195_A195_native']['pure_noise'][1], records['V198_A195_native']['pure_noise'][1], 'cross-V A195 action draw only')
        assert completed == [item[0] for item in PLAN]
        assert counts == dict(model_before=360, model_after=360, native_UND_dispatch=12960, native_GEN_dispatch=12960,
            all_allowed_dispatch=10800, cut_dispatch=1080)
        assert runtime.cm.randn_tensor is native_random and processor_module.dispatch_attention_fn is dispatch
        assert not checker._handles and checker._kind is None
        write(out / 'primary-prediction.json', dict(status='not_evaluated_q0_only', primary_window='early_l1_9',
            hypothesis_zh=HYPOTHESIS_ZH, follow_up_rule=FOLLOW_UP, old_eighteen_layer_prediction_remains='rejected',
            cases=['V195_A195_early_l1_9', 'V198_A195_early_l1_9'], both_selected_objects_must_equal=['milk_1'],
            remaining_windows_reported_without_replacing_primary=list(WINDOWS)[1:],
            all12_physical_cases_preregistered=True, q0_numeric_rank_used=False), exclusive=True)
        write(out / 'results.json', dict(state='complete', cases=rows, fresh_q0_predictions=12, fresh_model_forwards=360,
            official_dispatch_counts=counts, total_official_dispatch_calls=37800,
            primary_prediction_sha256=sha(out / 'primary-prediction.json'), primary_physical_prediction_evaluated=False,
            preregistered_physics_cases=[item[0] for item in PLAN], physics_calls=0, scope=SCOPE), exclusive=True)
        write(out / 'complete.json', dict(state='complete', script_sha256=sha(Path(__file__)),
            **{name + '_sha256': sha(out / (name + '.json')) for name in ('results', 'provenance', 'protocol', 'sources')},
            primary_prediction_sha256=sha(out / 'primary-prediction.json'), fresh_q0_predictions=12, fresh_model_forwards=360,
            total_official_dispatch_calls=37800, original_dispatch_calls=25920, extra_dispatch_calls=11880,
            original_randn_calls_consumed=24, returned_source_draws_observed=24,
            two_native_entire_prior_records_exact=True, two_sham_entire_prior_records_exact=True,
            all_four_control_all30_full_tuples_and_boundaries_exact=True, all8_windows_before_first_cut_own_sham_prefix_exact=True,
            all_active_cut_sites_unblocked_GEN_vs_same_input_allallowed_exact=True,
            all10_masked_full1080_sites_return_common_union66=True, all360_actual_complete_model_kwargs_saved=True,
            all_original_symbols_restored=True, all_owned_hooks_removed=True,
            primary_physical_prediction_evaluated=False, physics_calls=0, later_query_predictions=0, scope=SCOPE), exclusive=True)
        progress('complete')
    except Exception as exc:
        write(out / 'failed.json', dict(state='failed', error=str(exc), traceback=traceback.format_exc(), completed=completed,
            actual_counts=counts, active_case=None if observer is None else observer.folder.name,
            active_actual_counts=None if observer is None else observer.counts, physics_calls=0, scope=SCOPE))
        raise
    finally:
        try:
            if noise is not None:
                noise.reset()
        finally:
            if observer is not None:
                observer.reset()


if __name__ == '__main__':
    main()
