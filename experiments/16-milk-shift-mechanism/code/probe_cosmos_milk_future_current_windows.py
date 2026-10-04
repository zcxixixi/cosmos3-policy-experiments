"""Twelve q0 predictions locating a future-to-current cut in four fixed windows.

Use --output BASE; create only absent BASE/future-current-windows. Reuse the
frozen c106 full-GEN dispatcher, masks, initial-noise seam and actual decoder
observers. Two untouched native controls and two full all-allowed shams first;
then early/late denoising by low/high decoder halves for both V195/V198, A195.
All ten masked cases keep the same union66 all-allowed background at EVERY
site. Only the selected 15-step by 18-layer window additionally cuts current
queries from future keys. Outside the window the returned union66 remains
the all-allowed result on its actual dynamic inputs; it is not restored to
native arithmetic, nor required to return to the independent sham trajectory.

This stage runs 360 forwards / 38880 official attention dispatches, no physics.
All twelve later physical trials are preregistered. The sole primary physical
prediction is early-low: BOTH V cases select only milk by the strict contact
and lift criterion. Failure rejects that prediction; the other three windows
are all reported and cannot replace it. No q0 score selects physical trials.
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
PHYSICS_SHA = 'e0c5073ee320bbc173eebf827126c9431043a3efaa960eb0a8ccff289cae30ed'
PROMPT = 'pick up the milk and place it in the basket'
WINDOWS = {'early_low': dict(steps=[0, 14], layers=[0, 17]),
           'early_high': dict(steps=[0, 14], layers=[18, 35]),
           'late_low': dict(steps=[15, 29], layers=[0, 17]),
           'late_high': dict(steps=[15, 29], layers=[18, 35])}
PLAN = tuple((f'V{v}_A195_{name}', v, name) for name in ('native', 'all_allowed', *WINDOWS) for v in (195, 198))
FILES = ('states.pt', 'normalized_actions.json', 'metadata.json', 'noise-audit.pt', 'noise-audit.json',
         'observer-report.json', 'indexes-and-masks.pt')
SCOPE = ('Four preregistered future-to-current step/layer windows in a common '
         'masked arithmetic background, fixed A195 and two V realizations at X+12cm. '
         'Actual q0 inputs, full model outputs and current/action decoder boundaries '
         'only. No training, new physics, semantic-unit label or root-cause claim. '
         'Hard masking also renormalizes the remaining attention; this experiment '
         'does not separate future content from attention allocation.')


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
    path = ROOT / 'work/probe_cosmos_milk_future_read_cuts.py'
    assert sha(path) == BASE_PROBE_SHA
    frozen = load('future_current_windows_frozen_fullcut', path)
    factor, contract, original_sources, image, paths, anchors = frozen.preflight(baseline)
    producer = baseline / 'future-read-cuts'
    assert not (producer / 'failed.json').exists()
    complete, results, provenance = (read(producer / (name + '.json')) for name in ('complete', 'results', 'provenance'))
    assert complete['state'] == results['state'] == 'complete'
    assert complete['script_sha256'] == provenance['script_sha256'] == BASE_PROBE_SHA
    assert complete['fresh_q0_predictions'] == 10 and complete['fresh_model_forwards'] == 300
    assert complete['total_official_dispatch_calls'] == 36720
    for key in ('two_original_native_entire_q0_records_byte_exact', 'all_masked_actual_site_unblocked_queries_exact_vs_same_input_sham',
                'all_future200_and_UND_native_dispatch_rows_retained', 'both_engineering_action_and_boundary_equality_passed',
                'original_noise_and_dispatch_symbols_restored', 'all_owned_hooks_removed'):
        assert complete[key] is True
    for name in ('results', 'provenance', 'protocol', 'sources', 'closure'):
        assert complete[name + '_sha256'] == sha(producer / (name + '.json'))
    sealed = read(producer / 'sources.json')
    assert sealed == dict(anchors=anchors, threshold_sources=original_sources, frozen_contract=contract)
    old_rows = {row['case']: row for row in results['cases']}
    controls = {}
    for arm in ('native', 'all_allowed'):
        for v in (195, 198):
            label, folder = f'V{v}_A195_{arm}', producer / f'V{v}_A195_{arm}'
            row = old_rows[label]
            assert row['arm'] == arm and row['vision_noise_source_seed'] == v and row['action_noise_source_seed'] == 195
            assert row['entire_native_q0_record_byte_exact'] is (True if arm == 'native' else None)
            assert set(row['files_sha256']) == set(FILES)
            for name, expected in row['files_sha256'].items():
                assert sha(folder / name) == expected
            assert len(row['boundary_files']) == 30
            for step, item in enumerate(row['boundary_files']):
                assert item['step'] == step and item['file'] == f'boundaries-t{step:02d}.pt'
                assert sha(folder / item['file']) == item['sha256']
            controls[label] = dict(folder=str(folder), files_sha256=row['files_sha256'], boundary_files=row['boundary_files'])
    physics = baseline / 'future-read-execution'
    pc, physical_controls = read(physics / 'complete.json'), read(physics / 'controls.json')
    assert not (physics / 'wrapper_failed.json').exists()
    assert pc['state'] == 'complete' and pc['script_sha256'] == PHYSICS_SHA
    assert sha(ROOT / 'work/run_cosmos_milk_future_read_execution.py') == PHYSICS_SHA
    assert pc['physical_trials'] == 10 and pc['fresh_model_forwards'] == 2100 and pc['actual_later_whole_noise_pairs'] == 63
    assert pc['numerical_sham_classification_gate_passed'] is True
    for name, expected in pc['files_sha256'].items():
        assert sha(physics / name) == expected
    physical_references = {}
    for label in controls:
        hashes = physical_controls['actual_case_files_sha256'][label]
        folder = physics / 'closed-loop' / label
        for name, expected in hashes.items():
            assert sha(folder / name) == expected
        physical_references[label] = dict(folder=str(folder), files_sha256=hashes)
    source_evidence = dict(base_probe_script_sha256=BASE_PROBE_SHA, anchors=anchors,
        prior_q0_documents_sha256={name: sha(producer / (name + '.json')) for name in ('complete', 'results', 'provenance', 'protocol', 'sources', 'closure')},
        prior_physical_complete_sha256=sha(physics / 'complete.json'), prior_physical_files_sha256=pc['files_sha256'],
        q0_controls=controls, physical_control_references=physical_references,
        historical_control_scope='Completed c106/e0c producer seals are anchored; this stage repeats q0 controls but runs no physical control yet.')
    return frozen, factor, contract, original_sources, image, paths, source_evidence


def observer_class(frozen):
    """Only the schedule and actual per-step input export differ from c106."""
    class WindowObserver(frozen.FutureReadCuts):
        def __init__(self, *args, case_arm, **kwargs):
            super().__init__(*args, **kwargs)
            self.case_arm = case_arm
            self.window = WINDOWS.get(case_arm)
            self.actual_kwargs = None

        def before_model(self, module, args, kwargs):
            super().before_model(module, args, kwargs)
            self.actual_kwargs = self.runtime.cpu(kwargs)

        def dispatch(self, *args, **kwargs):
            active = cut_active(self.window, self.step, self.active) if self.active is not None else False
            expected_arm = 'native' if self.case_arm == 'native' else 'cut_future_current' if active else 'all_allowed'
            if kwargs.get('is_causal') is False:
                self.arm = expected_arm
            output = super().dispatch(*args, **kwargs)
            if kwargs.get('is_causal') is False:
                event = self.events[-1]
                assert event['arm'] == expected_arm
                event.update(window_name=self.case_arm, selected_window_cut_active=active,
                    window_zero_based=self.window, common_masked_union66_background=self.case_arm != 'native')
                assert event['extra_cut_calls'] == int(active)
                assert event['extra_all_allowed_calls'] == int(self.case_arm != 'native')
                assert event['substituted_query_rows'] == (0 if self.case_arm == 'native' else 66)
            return output

        def after_model(self, module, args, output):
            # The c106 exact hook ordering and module observations are preserved;
            # export once with actual kwargs rather than rewrite a sealed PT file.
            assert self.in_model and self.active is None and self.next_layer == 36
            assert len(self.events) == 36 and len(self.boundaries) == 37 and type(output) is tuple and len(output) == 3
            assert type(output[2]) is list and len(output[2]) == 1 and output[2][0].shape == (16, 64)
            for layer, event in enumerate(self.events):
                assert event['layer_zero_based'] == layer and event['step'] == self.step
                assert event['selected_window_cut_active'] == cut_active(self.window, self.step, layer)
            current = self.torch.stack([item['current'] for item in self.boundaries])
            action = self.torch.stack([item['action'] for item in self.boundaries])
            actual_output = self.runtime.cpu(output)
            data = dict(step=self.step, current_hidden=current, action_hidden=action,
                model_output=actual_output, actual_model_kwargs=self.actual_kwargs,
                actual_pack=self.calls[-1], site_events=self.events,
                actual_model_action_timesteps=self.actual_kwargs['action_timesteps'],
                solver_sigma_from_same_frozen_schedule=self.source['sigmas'][self.step].clone(),
                boundary_convention='0=actual decoder input; 1..36=actual block output after attention and MLP residuals; 36 precedes final norm.',
                indexes_and_masks_file='indexes-and-masks.pt', case_arm=self.case_arm, selected_window=self.window)
            path = self.folder / f'boundaries-t{self.step:02d}.pt'
            self.torch.save(data, path)
            self.files.append(dict(step=self.step, file=path.name, sha256=sha(path), current_shape=list(current.shape), action_shape=list(action.shape)))
            self.outputs.append(actual_output[2][0])
            self.counts['model_after'] += 1
            self.in_model = False
            self.boundaries, self.events, self.actual_kwargs = [], [], None

        def report(self, record):
            assert self.restored and not self.handles and not self.in_model and self.active is None
            assert self.counts['model_before'] == self.counts['model_after'] == 30
            for key in ('block_before', 'block_after', 'pre_W', 'native_UND_dispatch', 'native_GEN_dispatch'):
                assert self.counts[key] == 1080
            assert self.counts['all_allowed_dispatch'] == (0 if self.case_arm == 'native' else 1080)
            assert self.counts['cut_dispatch'] == (270 if self.window else 0)
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
    return WindowObserver


def saved_control_checks(folder, old_folder, record, old_record, runtime):
    runtime.require_exact(record, old_record, 'ENTIRE existing native or all-allowed q0 record')
    for step in range(30):
        a = runtime.torch.load(folder / f'boundaries-t{step:02d}.pt', map_location='cpu', weights_only=True, mmap=True)
        b = runtime.torch.load(old_folder / f'boundaries-t{step:02d}.pt', map_location='cpu', weights_only=True, mmap=True)
        for key in ('current_hidden', 'action_hidden', 'model_output', 'actual_pack'):
            runtime.require_exact(a[key], b[key], 'existing control actual boundaries/full tuple/' + str(step) + '/' + key)
    return dict(entire_existing_q0_record_byte_exact=True, all30_existing_full_model_output_tuples_exact=True,
        all30_existing_current_action_37boundaries_exact=True, comparison_folder=str(old_folder))


def prefix_checks(folder, sham_folder, record, sham_record, window, runtime):
    start_step, start_layer = window['steps'][0], window['layers'][0]
    for key in ('readout', 'action_velocity'):
        runtime.require_exact(record[key][:start_step], sham_record[key][:start_step], 'pre-cut complete head prefix/' + key)
    runtime.require_exact(record['action_states'][:start_step + 1], sham_record['action_states'][:start_step + 1], 'pre-cut FP32 solver samples including first-cut input')
    for step in range(start_step + 1):
        a = runtime.torch.load(folder / f'boundaries-t{step:02d}.pt', map_location='cpu', weights_only=True, mmap=True)
        b = runtime.torch.load(sham_folder / f'boundaries-t{step:02d}.pt', map_location='cpu', weights_only=True, mmap=True)
        runtime.require_exact(a['actual_model_kwargs'], b['actual_model_kwargs'], 'pre-cut actual ENTIRE model kwargs/' + str(step))
        if step < start_step:
            runtime.require_exact(a['model_output'], b['model_output'], 'pre-cut ENTIRE model output/' + str(step))
        for key in ('current_hidden', 'action_hidden'):
            n = 37 if step < start_step else start_layer + 1
            runtime.require_exact(a[key][:n], b[key][:n], 'pre-cut actual current/action boundaries/' + str(step) + '/' + key)
    return dict(state='exact', first_cut_step=start_step, first_cut_layer_zero_based=start_layer,
        earlier_complete_model_outputs_and_head_prefix_exact=True, all_actual_complete_kwargs_through_first_cut_step_exact=True,
        earlier_all37_and_first_cut_before_layer_current_action_boundaries_exact=True,
        solver_samples_through_first_cut_input_exact=True,
        after_first_cut_equal_to_independent_sham_required=False,
        scope='Actual saved full kwargs/output and current/action boundaries; no assertion of unsaved full-QKV prefix arrays.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Completed baseline experiment root')
    baseline = parser.parse_args().output.resolve()
    frozen, factor, contract, original_sources, image, paths, sources = source_contract(baseline)
    out = baseline / 'future-current-windows'
    out.mkdir(exist_ok=False)
    protocol = dict(plan=[dict(case=label, vision_noise_source_seed=v, action_noise_source_seed=195, runtime_rng_seed=195,
        arm=arm, window=WINDOWS.get(arm)) for label, v, arm in PLAN],
        trial_order=[label for label, _, _ in PLAN], windows_zero_based_inclusive=WINDOWS,
        primary_window='early_low', primary_cases=['V195_A195_early_low', 'V198_A195_early_low'],
        primary_physical_prediction='Each primary case selected_objects == [milk_1] under both fingerpad contacts plus >.02m lift for five consecutive records; V195 changes from cheese to only milk and V198 retains only milk.',
        primary_evaluation=dict(status='unexecuted_physics_prediction', success_requires_both_V_only_milk=True,
            milk_inclusion_max_lift_and_other_windows_are_descriptive_not_primary=True,
            rejection_does_not_promote_another_window=True),
        image_path=str(image), image_sha256=sha(image), prompt=PROMPT, shift_cm=12,
        q0_predictions=12, model_forwards=360, physics_calls=0, original_dispatch_calls=25920,
        extra_all_allowed_dispatch_calls=10800, extra_cut_dispatch_calls=2160, total_official_dispatch_calls=38880,
        cut_sites_per_window_case=270, masked_sites_per_case=1080,
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
        print('[FUTURE-CURRENT-WINDOWS] ' + json.dumps(value), flush=True)

    try:
        os.environ['HF_HUB_OFFLINE'] = '1'
        base = load('future_current_windows_runtime', paths['normal_runtime'])
        base.TASKS['milk'] = (7, 'pick_up_the_milk_and_place_it_in_the_basket', PROMPT, 'milk_1')
        factory = load('future_current_windows_factory', paths['factory'])
        factory.OUT = out
        assert sha(factory.SCHEDULER) == contract['frozen_files']['scheduler']['sha256']
        progress('loading_model')
        runtime = base.NormalRuntime(factory)
        torch, model = runtime.torch, runtime.pipe.transformer
        assert not model.training and not model.is_cache_enabled and not torch.is_grad_enabled()
        anchors = sources['anchors']
        assert sha(Path(inspect.getfile(type(model)))) == anchors['upstream']['actual_transformer_sha256']
        assert sha(Path(inspect.getfile(type(runtime.pipe)))) == anchors['upstream']['actual_pipeline_source_sha256']
        components, scan, reader = (load('future_current_windows_' + key, paths[key]) for key in ('components', 'metrics', 'reader'))
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
            inherited_fullcut_source_sha256=BASE_PROBE_SHA, noise_seam_source_sha256=frozen.FACTOR_SHA,
            source_sha256={key: sha(path) for key, path in paths.items()}, scheduler_sha256=sha(factory.SCHEDULER),
            actual_transformer_source=str(Path(inspect.getfile(type(model)))), actual_transformer_sha256=sha(Path(inspect.getfile(type(model)))),
            actual_pipeline_source_sha256=sha(Path(inspect.getfile(type(runtime.pipe)))), dispatch_source=str(dispatch_path),
            dispatch_source_sha256=sha(dispatch_path), dispatch_signature=str(inspect.signature(dispatch)),
            registry_backend=backend_name.value, registry_supported_arguments=sorted(supported),
            registry_backend_function_source_sha256=hashlib.sha256(inspect.getsource(backend_fn).encode()).hexdigest(),
            backend_settings=settings, library_edits=False, training=False, physics_calls=0,
            implementation='Frozen c106 dispatcher/masks/block hooks/noise seam inherited. GEN-site schedule, report count and once-written complete actual kwargs export implemented here.', scope=SCOPE)
        write(out / 'provenance.json', provenance, exclusive=True)
        WindowObserver = observer_class(frozen)
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
                control = saved_control_checks(directory, Path(reference['folder']), record, old_records[label], runtime)
            else:
                sham_label = f'V{v}_A195_all_allowed'
                control = prefix_checks(directory, out / sham_label, record, records[sham_label], WINDOWS[arm], runtime)
            write(directory / 'control-checks.json', control, exclusive=True)
            write(directory / 'observer-report.json', report, exclusive=True)
            write(directory / 'noise-audit.json', dict(noise=noise_report, preparation=prepared), exclusive=True)
            metadata.update(case=label, arm=arm, window_zero_based_inclusive=WINDOWS.get(arm), query=0, shift_cm=12,
                vision_noise_source_seed=v, action_noise_source_seed=195, runtime_rng_seed=195,
                numerical_masked_sham=arm == 'all_allowed', native_noop=arm == 'native',
                common_union66_all_allowed_background=arm != 'native', window_cut_sites=observer.counts['cut_dispatch'],
                actual_masked_site_substitutions=observer.counts['substituted_sites'],
                primary_prediction_case=arm == 'early_low', physical_prediction_evaluated=False,
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
            all_allowed_dispatch=10800, cut_dispatch=2160)
        assert runtime.cm.randn_tensor is native_random and processor_module.dispatch_attention_fn is dispatch
        assert not checker._handles and checker._kind is None
        write(out / 'primary-prediction.json', dict(status='not_evaluated_q0_only', primary_window='early_low',
            cases=['V195_A195_early_low', 'V198_A195_early_low'], both_selected_objects_must_equal=['milk_1'],
            remaining_windows_reported_without_replacing_primary=list(WINDOWS)[1:],
            all12_physical_cases_preregistered=True, q0_numeric_rank_used=False), exclusive=True)
        write(out / 'results.json', dict(state='complete', cases=rows, fresh_q0_predictions=12, fresh_model_forwards=360,
            official_dispatch_counts=counts, total_official_dispatch_calls=38880,
            primary_prediction_sha256=sha(out / 'primary-prediction.json'), primary_physical_prediction_evaluated=False,
            preregistered_physics_cases=[item[0] for item in PLAN], physics_calls=0, scope=SCOPE), exclusive=True)
        write(out / 'complete.json', dict(state='complete', script_sha256=sha(Path(__file__)),
            **{name + '_sha256': sha(out / (name + '.json')) for name in ('results', 'provenance', 'protocol', 'sources')},
            primary_prediction_sha256=sha(out / 'primary-prediction.json'), fresh_q0_predictions=12, fresh_model_forwards=360,
            total_official_dispatch_calls=38880, original_dispatch_calls=25920, extra_dispatch_calls=12960,
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
