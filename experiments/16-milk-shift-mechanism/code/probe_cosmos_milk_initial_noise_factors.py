"""Four fresh X+12cm q0 queries crossing two saved initial noise sources.

Use --output BASE. Only absent BASE/initial-noise-factors may be created.
Native V195/A195 and V198/A198 must reproduce their entire old q0 records
before mixed V198/A195 and V195/A198. Every query runs all thirty forwards:
four predictions / 120 model forwards, no physics or later-query inference.

Wrap the actual NormalRuntime cm.randn_tensor seam. First call the original
generator with unchanged arguments, consuming its normal draw; then return
the chosen whole saved FP32 noise tensor. Runtime's own observer records the
returned tensor. Current-frame clamping and action padding remain native.
These two sampling realizations do not name vision noise as hallucination,
action noise as intention, or identify a target commitment or neural cause.
"""

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import time
import traceback


ROOT = Path('/home/current/work/cosmos3')
READER_SHA = 'a71604115ee797cef97662f29c9f76e536d84f7e5515d1a725526c0a0d08d2eb'
READER_SOURCES_SHA = '49621d580b8333147134b94d025d71ccbf7bcdd7256aa14afbf7c6685d142c8d'
PROMPT = 'pick up the milk and place it in the basket'
PLAN = {'V195_A195': (195, 195), 'V198_A198': (198, 198),
        'V198_A195': (198, 195), 'V195_A198': (195, 198)}
SCOPE = ('Four fresh q0 predictions at the same X+12cm image/current condition, '
         'crossing the vision and action FP32 initial draws of two saved sampling '
         'realizations. Native prepare, clamping, masks, full schedule and model '
         'remain unchanged. No training, component patch, path isolation, physics, '
         'later-query experiment or claim of object intention/commitment/root cause.')


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


def source_contract(baseline):
    """Read source provenance before creating output or importing the factory."""
    reader_path = ROOT / 'work/probe_cosmos_milk_target_reader_inputs.py'
    assert sha(reader_path) == READER_SHA
    reader = load('initial_noise_frozen_threshold_reader', reader_path)
    contract, current_cases, threshold = reader.threshold_sources(baseline)
    original_reader = baseline / 'target-reader-inputs'
    assert not (original_reader / 'failed.json').exists()
    complete = read(original_reader / 'complete.json')
    assert complete['state'] == 'complete' and complete['script_sha256'] == READER_SHA
    assert complete['native_q0_predictions'] == 3 and complete['native_model_forwards'] == 90
    assert complete['all_original_entire_q0_records_bit_exact'] is True
    for key, filename in (('sources_sha256', 'sources.json'), ('provenance_sha256', 'provenance.json'),
                          ('results_sha256', 'results.json'), ('protocol_sha256', 'protocol.json')):
        assert complete[key] == sha(original_reader / filename)
    assert complete['sources_sha256'] == READER_SOURCES_SHA
    sealed = read(original_reader / 'sources.json')
    assert sealed['cases'] == current_cases and sealed['threshold'] == threshold
    sources = {str(seed): sealed['cases'][str(seed)] for seed in (195, 198)}
    image = baseline / 'position-threshold/inputs/x12/input.png'
    for seed, source in sources.items():
        folder = Path(source['q0_folder'])
        assert source['trial'] == 'x12_seed' + seed
        assert sha(Path(source['image'])) == source['image_sha256'] == sha(image)
        assert Path(source['image']).read_bytes() == image.read_bytes()
        for name, expected in source['q0_files_current_sha256'].items():
            assert sha(folder / name) == expected
    feedback_path = baseline / 'feedback-inputs/provenance.json'
    feedback = read(feedback_path)
    paths = {key: Path(contract['frozen_files'][key]['path']) for key in ('normal_runtime', 'factory', 'components')}
    paths['metrics'] = ROOT / 'work/probe_cosmos_milk_component_pulses.py'
    assert sha(paths['metrics']) == feedback['pulse_contract_sha256']
    for key in ('normal_runtime', 'factory', 'components'):
        assert sha(paths[key]) == feedback['source_sha256'][key]
    return contract, sources, image, paths, dict(reader_complete_sha256=sha(original_reader / 'complete.json'),
        reader_sources_sha256=READER_SOURCES_SHA, reader_provenance_sha256=sha(original_reader / 'provenance.json'),
        feedback_provenance_sha256=sha(feedback_path), actual_transformer_sha256=feedback['actual_transformer_sha256'],
        actual_pipeline_source_sha256=feedback['actual_pipeline_source_sha256'], threshold=threshold,
        source_hash_scope='Source q0 hashes were first sealed at the earlier completed target-reader stage; not a historical threshold whole-case manifest.')


def meta(value, runtime, scan):
    return dict(**scan.tensor_meta(value, runtime.torch), stride=list(value.stride()), device=str(value.device))


class InitialNoiseSources:
    """One query's outer seam, called by the unchanged inner noise observer."""

    def __init__(self, runtime, scan, records, vision_seed, action_seed):
        self.runtime, self.torch, self.scan = runtime, runtime.torch, scan
        self.records, self.vision_seed, self.action_seed = records, vision_seed, action_seed
        self.original = runtime.cm.randn_tensor
        self.signature = inspect.signature(self.original)
        self.wrapper = self.draw
        self.calls, self.installed, self.restored = [], False, False
        self.generator = None

    def begin(self):
        assert not self.installed and self.runtime.cm.randn_tensor is self.original
        self.runtime.cm.randn_tensor = self.wrapper
        self.installed = True

    def draw(self, *args, **kwargs):
        torch, runtime = self.torch, self.runtime
        index = len(self.calls)
        assert index < 2, 'Unexpected third randn call'
        bound = self.signature.bind(*args, **kwargs)
        generator = bound.arguments.get('generator')
        assert isinstance(generator, torch.Generator) and generator.initial_seed() == self.action_seed
        if self.generator is None:
            self.generator = generator
        assert generator is self.generator, 'The two original draws must share the actual generator'
        before = generator.get_state().clone()
        # Calling the saved original is essential: never recurse into cm.randn_tensor.
        generated = self.original(*args, **kwargs)
        after = generator.get_state().clone()
        assert not torch.equal(before, after), 'Original RNG draw did not consume state'
        source_seed = self.vision_seed if index == 0 else self.action_seed
        source = self.records[source_seed]['pure_noise'][index].clone(memory_format=torch.preserve_format)
        expected_generated = self.records[self.action_seed]['pure_noise'][index]
        assert generated.dtype == source.dtype == torch.float32
        assert tuple(generated.shape) == tuple(source.shape) == ((1, 48, 5, 10, 20) if index == 0 else (16, 64))
        assert generated.stride() == source.stride(), 'Saved source/native draw strides differ'
        generated_cpu = runtime.cpu(generated)
        runtime.require_exact(generated_cpu, expected_generated, 'original consumed draw equals actual A-seed source/' + str(index))
        returned = source.to(device=generated.device, dtype=generated.dtype).clone(memory_format=torch.preserve_format)
        assert returned.stride() == generated.stride() and returned.device == generated.device
        returned_cpu = runtime.cpu(returned)
        runtime.require_exact(returned_cpu, source, 'actual returned whole source draw/' + str(index))
        if self.calls:
            runtime.require_exact(before, self.calls[-1]['generator_state_after'], 'generator state chain between original draws')
        self.calls.append(dict(ordinal=index, modality='vision' if index == 0 else 'action',
            returned_source_seed=source_seed, original_generated_seed=self.action_seed,
            generator_state_before=before, generator_state_after=after,
            original_generated_discarded=generated_cpu, actual_returned=returned_cpu,
            original_generated_metadata=meta(generated, runtime, self.scan), returned_metadata=meta(returned, runtime, self.scan)))
        return returned

    def reset(self):
        if self.installed:
            # NormalRuntime.finally returns this seam to our outer wrapper.
            unchanged = self.runtime.cm.randn_tensor is self.wrapper
            self.runtime.cm.randn_tensor = self.original
            self.installed = False
            self.restored = self.runtime.cm.randn_tensor is self.original
            assert unchanged, 'Inner runtime did not restore the original outer noise wrapper'

    def report(self, record):
        assert self.restored and len(self.calls) == len(record['pure_noise']) == 2
        report = []
        for call, observed in zip(self.calls, record['pure_noise']):
            self.runtime.require_exact(observed, call['actual_returned'], 'inner runtime observed actual injected noise/' + str(call['ordinal']))
            report.append({key: value for key, value in call.items() if key.endswith('_metadata') or key in
                ('ordinal', 'modality', 'returned_source_seed', 'original_generated_seed')})
            report[-1].update(generator_before=self.scan.tensor_meta(call['generator_state_before'], self.torch),
                generator_after=self.scan.tensor_meta(call['generator_state_after'], self.torch),
                actual_inner_observer_noise=self.scan.tensor_meta(observed, self.torch),
                original_generated_matches_A_seed_source_byte_exact=True,
                actual_returned_matches_selected_source_byte_exact=True,
                inner_observer_matches_actual_returned_byte_exact=True)
        return dict(draw_count=2, original_randn_calls=2, invoke_seed=self.action_seed,
            vision_noise_source_seed=self.vision_seed, action_noise_source_seed=self.action_seed,
            original_symbol_restored=True, actual_generator_identity_same_for_both_draws=True,
            original_two_draw_rng_state_chain_exact=True, calls=report)


class PackObserver:
    """Only actual kwargs/layout observations; no processor or model recompute."""

    def __init__(self, runtime, components, cache, clean_current):
        self.runtime, self.torch, self.components = runtime, runtime.torch, components
        self.cache, self.clean_current = cache, clean_current
        self.calls, self.first, self.handle = [], None, None
        self.model = runtime.pipe.transformer

    def before(self, module, args, kwargs):
        assert not args and len(self.calls) < 30
        step = len(self.calls)
        layout = self.components._layout(kwargs)
        actual = dict(layout=layout, metadata={key: self.runtime.cpu(kwargs.get(key)) for key in self.components._STRUCTURAL_KEYS})
        self.runtime.require_exact(actual, self.cache['steps'][step], 'all thirty actual sigma/pack metadata/' + str(step))
        vision = kwargs['vision_tokens'][0]
        action = kwargs['action_tokens'][0]
        assert vision.shape == (1, 48, 5, 10, 20) and vision.dtype == self.torch.bfloat16
        assert action.shape == (16, 64) and action.dtype == self.torch.bfloat16
        self.runtime.require_exact(self.runtime.cpu(vision[0, :, 0]), self.clean_current, 'actual model current condition frame0/' + str(step))
        assert self.torch.count_nonzero(action[:, 10:]) == 0
        if step == 0:
            self.first = self.runtime.cpu(kwargs)
        self.calls.append(actual)

    def begin(self):
        assert self.handle is None
        self.handle = self.model.register_forward_pre_hook(self.before, with_kwargs=True)

    def reset(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def prepared_contract(record, sources, vision_seed, action_seed, runtime, observer):
    torch = runtime.torch
    prepared = record['prepared_latents_and_masks']
    v, a, common = (sources[seed]['prepared_latents_and_masks'] for seed in (vision_seed, action_seed, 195))
    assert isinstance(prepared, tuple) and len(prepared) == 12
    assert prepared[0].shape == (1, 48, 5, 10, 20) and prepared[0].dtype == torch.float32
    assert prepared[2].shape == (16, 64) and prepared[2].dtype == torch.float32
    runtime.require_exact(prepared[0], v[0], 'native prepared vision equals chosen vision source')
    runtime.require_exact(prepared[2], a[2], 'native prepared action equals chosen action source')
    for index in range(12):
        if index not in (0, 2):
            runtime.require_exact(prepared[index], common[index], 'all other actual preparation fields/' + str(index))
    assert prepared[3] == 20 and prepared[8].tolist() == [5] and prepared[10] == 10
    assert prepared[5].shape == (5, 1, 1)
    runtime.require_exact(prepared[5].flatten(), torch.tensor([1., 0., 0., 0., 0.], dtype=prepared[5].dtype), 'actual current/future vision mask')
    assert prepared[7].shape == (16, 1) and torch.count_nonzero(prepared[7]) == 0
    assert torch.count_nonzero(prepared[2][:, 10:]) == 0 and torch.count_nonzero(record['action_states'][..., 10:]) == 0
    runtime.require_exact(prepared[0][:, :, 0], common[0][:, :, 0], 'actual prepared current frame0 same for all cells')
    kwargs = record['model_input']
    runtime.require_exact(kwargs, observer.first, 'inner/outer actual first-model observation')
    for key in kwargs:
        if key not in ('action_tokens', 'vision_tokens'):
            runtime.require_exact(kwargs[key], sources[195]['model_input'][key], 'all initial text/position/layout/schedule/domain kwargs/' + key)
    runtime.require_exact(kwargs['vision_tokens'], sources[vision_seed]['model_input']['vision_tokens'], 'chosen vision-source initial model tokens')
    runtime.require_exact(kwargs['action_tokens'], sources[action_seed]['model_input']['action_tokens'], 'chosen action-source initial model tokens')
    runtime.require_exact(record['action_states'][0], sources[action_seed]['action_states'][0], 'native initial FP32 action solver sample')
    for key in ('timesteps', 'sigmas'):
        runtime.require_exact(record[key], sources[195][key], 'complete actual native schedule/' + key)
    assert len(observer.calls) == 30 and observer.handle is None
    return dict(all_thirty_layout_schedule_metadata_exact=True, actual_model_calls=30,
        processor_recompute_calls=0, all_owned_observer_hooks_removed=True,
        current_condition_frame0_all_thirty_model_calls_byte_exact=True,
        prepared_vision_entire_tensor_equals_V_source=True, prepared_action_entire_tensor_equals_A_source=True,
        other_ten_prepared_fields_byte_exact=True, other_initial_model_kwargs_byte_exact=True,
        native_raw_action_dim=10, all_prepared_model_input_and_31_solver_state_padding54_zero=True,
        action_condition_mask_all_zero=True, vision_conditioned_frame=0, future_noisy_frames=[1, 2, 3, 4])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Completed baseline with position-threshold and sealed target-reader-inputs')
    baseline = parser.parse_args().output.resolve()
    contract, sources, image, paths, upstream = source_contract(baseline)
    out = baseline / 'initial-noise-factors'
    out.mkdir(exist_ok=False)
    protocol = dict(plan={name: dict(vision_noise_source_seed=v, action_noise_source_seed=a,
        runtime_rng_seed=a, native_whole_record_control=v == a) for name, (v, a) in PLAN.items()},
        trial_order=list(PLAN), scene='x12', image_path=str(image), image_sha256=sha(image), prompt=PROMPT,
        q0_predictions=4, forwards_per_prediction=30, actual_model_forwards=120, physics_calls=0,
        component_intervention_count=0, random_draw_rule='Original full draw consumed before whole saved FP32 source replacement; invocation seed equals action source.',
        native_gate='Both same-source ENTIRE NormalRuntime records byte-exact before either mixed cell.',
        no_latent_path_patch=True, no_training=True, no_later_query_inference=True, scope=SCOPE)
    write(out / 'protocol.json', protocol, exclusive=True)
    write(out / 'sources.json', dict(cases=sources, upstream=upstream, frozen_threshold_contract=contract), exclusive=True)
    started, completed, results, observer, noise, runtime = time.perf_counter(), [], [], None, None, None
    actual_observed_model_forwards = 0

    def progress(stage, **values):
        info = dict(stage=stage, completed=completed, q0_predictions_complete=len(completed),
            completed_model_forwards=30 * len(completed), expected_model_forwards=120,
            actual_observed_model_forwards=actual_observed_model_forwards,
            elapsed_s=time.perf_counter() - started, **values)
        write(out / 'progress.json', info)
        print('[INITIAL-NOISE-FACTORS] ' + json.dumps(info), flush=True)

    try:
        os.environ['HF_HUB_OFFLINE'] = '1'
        base = load('initial_noise_checked_normal_runtime', paths['normal_runtime'])
        base.TASKS['milk'] = (7, 'pick_up_the_milk_and_place_it_in_the_basket', PROMPT, 'milk_1')
        factory = load('initial_noise_checked_factory', paths['factory'])
        factory.OUT = out
        assert sha(factory.SCHEDULER) == contract['frozen_files']['scheduler']['sha256']
        progress('loading_model')
        runtime = base.NormalRuntime(factory)
        torch, model = runtime.torch, runtime.pipe.transformer
        assert not model.training and not model.is_cache_enabled and not torch.is_grad_enabled()
        assert sha(Path(inspect.getfile(type(model)))) == upstream['actual_transformer_sha256']
        assert sha(Path(inspect.getfile(type(runtime.pipe)))) == upstream['actual_pipeline_source_sha256']
        scan = load('initial_noise_record_checks', paths['metrics'])
        components = load('initial_noise_readonly_layout', paths['components'])
        checker = components.CosmosComponentInterventions(model)
        original_records, original_caches = {}, {}
        for seed in (195, 198):
            directory = Path(sources[str(seed)]['q0_folder'])
            for name, expected in sources[str(seed)]['q0_files_current_sha256'].items():
                assert sha(directory / name) == expected
            original_records[seed] = torch.load(directory / 'states.pt', map_location='cpu', weights_only=True, mmap=True)
            original_caches[seed] = torch.load(directory / 'components.pt', map_location='cpu', weights_only=True, mmap=True)
            scan.validate_record(original_records[seed], torch)
            assert torch.count_nonzero(original_records[seed]['action_states'][..., 10:]) == 0, 'Original 31 solver snapshots must have zero padding/' + str(seed)
            scan.validate_cache(original_caches[seed], original_records[seed], checker, runtime, 'source' + str(seed))
            assert read(directory / 'normalized_actions.json') == original_records[seed]['actions'].tolist()
        reference = original_records[195]
        for key in ('timesteps', 'sigmas'):
            runtime.require_exact(reference[key], original_records[198][key], 'two source schedules/' + key)
        runtime.require_exact(original_caches[195]['steps'], original_caches[198]['steps'], 'two source full pack metadata')
        for index in range(12):
            if index not in (0, 2):
                runtime.require_exact(reference['prepared_latents_and_masks'][index], original_records[198]['prepared_latents_and_masks'][index], 'two source shared preparation/' + str(index))
        clean_current = reference['model_input']['vision_tokens'][0][0, :, 0]
        runtime.require_exact(clean_current, original_records[198]['model_input']['vision_tokens'][0][0, :, 0], 'two source current condition frame0')
        raw_sources = {str(seed): [meta(value, runtime, scan) for value in original_records[seed]['pure_noise']] for seed in (195, 198)}
        for index in (0, 1):
            assert not torch.equal(original_records[195]['pure_noise'][index], original_records[198]['pure_noise'][index])
        native_symbol = runtime.cm.randn_tensor
        symbol_path = Path(inspect.getfile(native_symbol))
        provenance = dict(script_sha256=sha(Path(__file__)), protocol_sha256=sha(out / 'protocol.json'),
            sources_sha256=sha(out / 'sources.json'), raw_source_noise=raw_sources,
            source_sha256={key: sha(path) for key, path in paths.items()}, upstream=upstream,
            actual_scheduler_sha256=sha(factory.SCHEDULER), actual_randn_source=str(symbol_path), actual_randn_source_sha256=sha(symbol_path),
            actual_randn_signature=str(inspect.signature(native_symbol)),
            actual_randn_function_source_sha256=hashlib.sha256(inspect.getsource(native_symbol).encode()).hexdigest(),
            library_edits=False, training=False, model_component_patches=False, physics_calls=0, scope=SCOPE)
        write(out / 'provenance.json', provenance, exclusive=True)
        for label, (vision_seed, action_seed) in PLAN.items():
            if vision_seed != action_seed:
                assert completed[:2] == ['V195_A195', 'V198_A198']
                assert all(row['native_entire_source_record_byte_exact'] for row in results[:2])
            directory = out / label
            progress('fresh_q0', cell=label)
            observer = PackObserver(runtime, components, original_caches[195], clean_current)
            noise = InitialNoiseSources(runtime, scan, original_records, vision_seed, action_seed)
            assert noise.original is native_symbol
            observer.begin()
            noise.begin()
            try:
                record, metadata = runtime.predict('milk', image, action_seed, directory)
            finally:
                try:
                    noise.reset()
                finally:
                    observer.reset()
                    actual_observed_model_forwards += len(observer.calls)
                # Keep actual witnesses even if predict or a later exact gate fails.
                if directory.exists():
                    torch.save(dict(calls=noise.calls, model_steps=observer.calls, initial_model_kwargs=observer.first,
                        original_symbol_restored=noise.restored, observer_hooks_removed=observer.handle is None), directory / 'noise-audit.pt')
            assert runtime.cm.randn_tensor is native_symbol
            scan.validate_record(record, torch)
            assert metadata['model_calls'] == 30 and metadata['prepare_calls'] == 1 and metadata['seed'] == action_seed
            assert metadata['input_png_sha256'] == sha(image) and metadata['prompt'] == PROMPT and metadata['task_index'] == 7
            native = vision_seed == action_seed
            if native:
                runtime.require_exact(record, original_records[action_seed], label + '/ENTIRE old native source q0 record')
            draw_report = noise.report(record)
            preparation = prepared_contract(record, original_records, vision_seed, action_seed, runtime, observer)
            metadata.update(cell=label, query=0, shift_cm=12, vision_noise_source_seed=vision_seed,
                action_noise_source_seed=action_seed, runtime_rng_seed=action_seed, fresh_q0_prediction=True,
                current_execution_model_calls=30, component_intervention_count=0, initial_noise_source_replacement=True,
                native_entire_source_record_byte_exact=True if native else None,
                source_q0_states_sha256={str(seed): sources[str(seed)]['q0_files_current_sha256']['states.pt'] for seed in (195, 198)})
            write(directory / 'metadata.json', metadata)
            write(directory / 'noise-audit.json', dict(noise=draw_report, preparation=preparation), exclusive=True)
            assert read(directory / 'normalized_actions.json') == record['actions'].tolist()
            files = ('states.pt', 'normalized_actions.json', 'metadata.json', 'noise-audit.pt', 'noise-audit.json')
            results.append(dict(cell=label, vision_noise_source_seed=vision_seed, action_noise_source_seed=action_seed,
                runtime_rng_seed=action_seed, fresh_model_forwards=30, native_entire_source_record_byte_exact=True if native else None,
                all_actual_noise_source_and_preparation_gates_passed=True,
                all_owned_symbols_and_observer_hooks_restored=True,
                final_normalized_actions=record['actions'].tolist(), files_sha256={name: sha(directory / name) for name in files}))
            completed.append(label)
            progress('cell_complete', cell=label)
            observer = noise = None
        assert completed == list(PLAN) and sum(row['fresh_model_forwards'] for row in results) == 120
        assert actual_observed_model_forwards == 120
        assert runtime.cm.randn_tensor is native_symbol and not checker._handles and checker._kind is None
        write(out / 'results.json', dict(state='complete', cases=results, fresh_q0_predictions=4,
            fresh_model_forwards=120, physics_calls=0, later_query_predictions=0, scope=SCOPE), exclusive=True)
        write(out / 'complete.json', dict(state='complete', script_sha256=sha(Path(__file__)),
            results_sha256=sha(out / 'results.json'), provenance_sha256=sha(out / 'provenance.json'),
            protocol_sha256=sha(out / 'protocol.json'), sources_sha256=sha(out / 'sources.json'),
            fresh_q0_predictions=4, fresh_model_forwards=120, original_randn_calls_consumed=8,
            actual_returned_source_draws_observed=8, both_native_entire_q0_records_byte_exact=True,
            all_current_frame_schedule_layout_padding_and_source_noise_gates_passed=True,
            original_randn_symbol_restored=True, all_owned_observer_hooks_removed=True,
            component_patches=0, physics_calls=0, later_query_predictions=0, scope=SCOPE), exclusive=True)
        progress('complete')
    except Exception as exc:
        write(out / 'failed.json', dict(state='failed', error=str(exc), traceback=traceback.format_exc(),
            completed=completed, surviving_complete_model_forwards=30 * len(completed),
            actual_observed_model_forwards=actual_observed_model_forwards, scope=SCOPE))
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
