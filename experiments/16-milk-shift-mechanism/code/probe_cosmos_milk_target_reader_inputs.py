"""Observe three original X+12cm q0 predictions; never intervene on actions.

Use --output BASE. Create only absent BASE/target-reader-inputs. Reproduce
each seed195/196/198 entire saved NormalRuntime record, including future
latents. At actual forward29, L17, transparently observe the original native
attention dispatch arguments/output and the real pre-output-projection input.
Save actual post-normalization/RoPE Q/K, unrotated V, full model kwargs/return.

Budget: three full q0 predictions = 90 model forwards, plus exactly three
extra original full-GEN attention dispatch replays. Replay the retained actual
GPU arguments with their original shapes, strides and kwargs; never slice to
sixteen queries or replace a processor. Require replay output byte equality.
Check same-prompt UND K/V across seeds; each recipient retains its own actual
two noise draws. Backend flags do not prove that a particular kernel ran.
No reader/content factorial, action patch, training or semantic conclusion.
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
SEEDS = (195, 196, 198)
LAYER, STEP, HEADS, KV_HEADS, HEAD_DIM = 17, 29, 32, 8, 128
PROMPT = 'pick up the milk and place it in the basket'
THRESHOLD_SHA = '62b6284d294d00d0323bdf31f68a90b397307ab0bf8804bc060f208a0d140d36'
SCOPE = ('Three observation-only native X+12cm q0 reproductions. Actual L17/t29 '
         'inputs and native attention arithmetic only. No action intervention, '
         'new physics, identity decoder or semantic/target-reader conclusion.')


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


def settings(torch, processor):
    cuda = torch.backends.cuda
    names = ('flash_sdp_enabled', 'mem_efficient_sdp_enabled', 'math_sdp_enabled', 'cudnn_sdp_enabled')
    enabled = {name: getattr(cuda, name)() if callable(getattr(cuda, name, None)) else None for name in names}
    return dict(torch_version=torch.__version__, cuda_version=torch.version.cuda,
        configured_attention_backend=repr(processor._attention_backend),
        configured_parallel=repr(processor._parallel_config), sdp_enabled_flags=enabled,
        matmul_allow_tf32=cuda.matmul.allow_tf32,
        matmul_allow_bf16_reduced_precision_reduction=getattr(cuda.matmul, 'allow_bf16_reduced_precision_reduction', None),
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        grad_enabled=torch.is_grad_enabled(),
        kernel_evidence_limit='Configured backend and enabled flags are recorded, not a profiler observation of the selected kernel.')


def threshold_sources(baseline):
    """Verify recorded provenance before creating output or loading a model."""
    import numpy as np
    from PIL import Image
    wrapper_path = ROOT / 'work/run_cosmos_milk_position_threshold.py'
    assert sha(wrapper_path) == THRESHOLD_SHA
    wrapper = load('reader_frozen_threshold_contract', wrapper_path)
    original = wrapper.frozen_contract(baseline)
    threshold = baseline / 'position-threshold'
    assert not any((threshold / name).exists() for name in
        ('wrapper_failed.json', 'server/failed.json', 'inputs/failed.json', 'closed-loop/failed.json', 'simulator_failed.json'))
    provenance = read(threshold / 'provenance.json')
    assert provenance['contract'] == original
    assert provenance['inputs_metadata_sha256'] == sha(threshold / 'inputs/metadata.json')
    assert provenance['zero_input_control_sha256'] == sha(threshold / 'zero-input-control.json')
    for role, complete_path in (('server', 'server/complete.json'), ('simulator', 'closed-loop/complete.json')):
        complete = read(threshold / f'wrapper-{role}-complete.json')
        assert complete['state'] == 'complete' and complete['provenance_sha256'] == sha(threshold / 'provenance.json')
        assert complete[role + '_complete_sha256'] == sha(threshold / complete_path)
    assert read(threshold / 'wrapper-server-complete.json')['zero_model_control_sha256'] == sha(threshold / 'zero-model-control.json')
    assert read(threshold / 'zero-model-control.json')['predictions_compared'] == 3
    server, simulator = read(threshold / 'server/complete.json'), read(threshold / 'closed-loop/complete.json')
    assert server['state'] == simulator['state'] == 'complete'
    assert server['predictions'] == 72 and server['paired_noise_checks'] == 48
    assert server['query_zero_component_captures'] == 9 and server['observer_free_repeat_all_tensors_exact'] is True
    assert simulator['cases'] == 9 and simulator['queries_per_case'] == 8
    assert simulator['steps_per_case'] == 128 and simulator['state_records_per_case'] == 129
    expected_trials = {f'{scene}_seed{seed}' for scene in ('x00', 'x09', 'x12') for seed in SEEDS}
    assert set(simulator['completed_trials']) == expected_trials
    input_complete, inputs = read(threshold / 'inputs/complete.json'), read(threshold / 'inputs/metadata.json')
    assert input_complete['state'] == 'matched_inputs_verified' and input_complete['source_center_pixels_exact'] is True
    assert input_complete['metadata_sha256'] == sha(threshold / 'inputs/metadata.json')
    assert inputs['seeds'] == list(SEEDS) and inputs['shifts_cm'] == dict(x00=0, x09=9, x12=12)
    assert inputs['prompt'] == PROMPT and inputs['additional_warmup_steps'] == 0
    assert inputs['suite'] == 'libero_object' and inputs['task_index'] == 7
    assert inputs['source_is_already_warmed'] is True
    scene = threshold / 'inputs/x12'
    for field, filename in (('state_sha256', 'state.npy'), ('arrays_sha256', 'arrays.json'),
                            ('controller_sha256', 'controller.npz'), ('input_png_sha256', 'input.png')):
        assert inputs['scenes']['x12'][field] == sha(scene / filename)
    manifest = read(scene / 'arrays.json')
    arrays = dict(state={'sim_state': np.load(scene / 'state.npy', allow_pickle=False)})
    for category in ('fixtures', 'cameras', 'objects', 'observations'):
        with np.load(scene / (category + '.npz'), allow_pickle=False) as archive:
            arrays[category] = {key: archive[key].copy() for key in archive.files}
    arrays['images'] = {name: np.asarray(Image.open(scene / (name + '.png')).convert('RGB'))
                       for name in ('agentview', 'robot0_eye_in_hand', 'input')}
    for category, values in arrays.items():
        assert values.keys() == manifest[category].keys()
        for key, value in values.items():
            actual = dict(shape=list(value.shape), dtype=str(value.dtype),
                          sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest())
            assert actual == manifest[category][key], ('physical scene array', category, key)
    summary = read(threshold / 'summary.json')
    assert summary == read(threshold / 'closed-loop/summary.json')
    assert len(summary['cases']) == 9 and set(summary['completed_trials']) == expected_trials
    rows = {row['trial']: row for row in summary['cases']}
    sources = {}
    for seed in SEEDS:
        trial = f'x12_seed{seed}'
        folder = threshold / 'closed-loop' / trial
        assert rows[trial] == read(folder / 'summary.json')
        assert rows[trial]['noise_seed_base'] == seed and rows[trial]['shift_cm'] == 12
        assert rows[trial]['selected_objects'] == (['milk_1'] if seed == 198 else ['cream_cheese_1'])
        assert (folder / 'input_00.png').read_bytes() == (scene / 'input.png').read_bytes()
        states = np.load(folder / 'sim_states.npy', mmap_mode='r', allow_pickle=False)
        initial = arrays['state']['sim_state']
        assert states.shape == (129, initial.size) and states.dtype == initial.dtype
        assert states[0].tobytes(order='C') == initial.tobytes(order='C')
        with np.load(scene / 'controller.npz', allow_pickle=False) as a, np.load(folder / 'initial_controller.npz', allow_pickle=False) as b:
            assert set(a.files) == set(b.files)
            for key in a.files:
                assert a[key].dtype == b[key].dtype and a[key].shape == b[key].shape
                assert a[key].tobytes(order='C') == b[key].tobytes(order='C'), ('initial controller', trial, key)
        assert all(item['exact'] for item in read(folder / 'initial_checks.json'))
        assert all(read(folder / 'initial_controller_checks.json').values())
        meta = read(folder / 'chunk_00/metadata.json')
        assert meta['seed'] == seed and meta['query'] == 0 and meta['shift_cm'] == 12 and meta['prompt'] == PROMPT
        assert meta['task'] == 'milk' and meta['task_index'] == 7
        assert meta['input_png_sha256'] == sha(folder / 'input_00.png') and meta['model_calls'] == 30 and meta['prepare_calls'] == 1
        names = ('states.pt', 'components.pt', 'metadata.json', 'normalized_actions.json')
        sources[str(seed)] = dict(trial=trial, image=str(folder / 'input_00.png'),
            image_sha256=sha(folder / 'input_00.png'), original_selected_objects=rows[trial]['selected_objects'],
            q0_folder=str(folder / 'chunk_00'),
            q0_files_current_sha256={name: sha(folder / 'chunk_00' / name) for name in names},
            physical_initial_state_current_sha256=sha(folder / 'sim_states.npy'),
            initial_controller_current_sha256=sha(folder / 'initial_controller.npz'),
            physical_initial_state_and_controller_bytes_exact=True,
            source_hash_scope='Q0 and trial artifact hashes are first observed and sealed here; the threshold completion files recorded counts and provenance, not a whole-case file manifest.')
    return original, sources, dict(threshold_root=str(threshold),
        wrapper_sha256=THRESHOLD_SHA, shared_scene_array_manifest_sha256=sha(scene / 'arrays.json'),
        shared_scene_state_sha256=sha(scene / 'state.npy'), shared_scene_controller_sha256=sha(scene / 'controller.npz'),
        all_three_initial_png_bytes_exact=True,
        documents_current_sha256={name: sha(threshold / name) for name in
            ('provenance.json', 'wrapper-server-complete.json', 'wrapper-simulator-complete.json',
             'server/complete.json', 'closed-loop/complete.json', 'inputs/complete.json', 'inputs/metadata.json', 'summary.json')})


class NativeDispatchObserver:
    """Return the original attention result unchanged; observe one actual site."""

    def __init__(self, runtime, components, scan, reference):
        self.runtime, self.torch, self.components, self.scan = runtime, runtime.torch, components, scan
        self.reference, self.model = reference, runtime.pipe.transformer
        self.attn = self.model.layers[LAYER].self_attn
        assert len(self.model.layers) == 36
        assert (self.attn.num_attention_heads, self.attn.num_key_value_heads, self.attn.head_dim) == (HEADS, KV_HEADS, HEAD_DIM)
        assert not self.model.training and not self.torch.is_grad_enabled() and not self.model.is_cache_enabled
        assert self.model._cp_shard_fn is self.model._cp_gather_fn is self.attn.processor._parallel_config is None
        self.module = inspect.getmodule(type(self.attn.processor))
        assert self.module is inspect.getmodule(type(self.model)), 'Unknown attention source module'
        self.original = self.module.dispatch_attention_fn
        self.handles, self.calls = [], dict(model_before=0, model_after=0, target_attn_before=0,
            target_attn_after=0, pre_W=0, native_dispatch=0, selected_causal=0, selected_gen=0, official_attention_replay=0)
        self.step, self.in_model, self.in_attn = -1, False, False
        self.data, self.installed, self.restored = {}, False, False
        self.replay_args = self.replay_kwargs = None
        self.wrapper = self.dispatch
        self.backend_settings = settings(self.torch, self.attn.processor)

    def before_model(self, module, args, kwargs):
        assert not args and not self.in_model
        self.step += 1
        assert self.step < 30
        self.calls['model_before'] += 1
        self.in_model = True
        layout = self.components._layout(kwargs)
        frame = dict(layout=layout, metadata={key: self.runtime.cpu(kwargs.get(key)) for key in self.components._STRUCTURAL_KEYS})
        self.runtime.require_exact(frame, self.reference['steps'][self.step], f'actual_pack/{self.step}')
        if self.step == 0:
            self.runtime.require_exact(self.runtime.cpu(kwargs), self.reference['first_model_kwargs'], 'native initial entire kwargs')
        if self.step == STEP:
            assert kwargs['return_dict'] is False, 'Require the actual native pipeline tuple return contract'
            self.data['model_input'] = self.runtime.cpu(kwargs)
            self.data['layout'] = layout

    def after_model(self, module, args, output):
        assert self.in_model and not self.in_attn
        self.calls['model_after'] += 1
        if self.step == STEP:
            assert type(output) is tuple and len(output) == 3
            self.data['model_output'] = self.runtime.cpu(output)
        self.in_model = False

    def before_attn(self, module, args):
        assert self.in_model and not self.in_attn and len(args) == 3
        self.in_attn = True
        self.calls['target_attn_before'] += 1
        if self.step == STEP:
            und, gen, rotary = args
            layout = self.data['layout']
            assert und.shape == (layout['und_len'], 4096) and gen.shape == (layout['gen_len'], 4096)
            self.data['actual_attn_inputs'] = dict(und_normalized=self.runtime.cpu(und),
                gen_normalized=self.runtime.cpu(gen), rotary_emb=self.runtime.cpu(rotary))

    def after_attn(self, module, args, output):
        assert self.in_attn
        self.calls['target_attn_after'] += 1
        self.in_attn = False

    def dispatch(self, *args, **kwargs):
        assert self.in_model
        self.calls['native_dispatch'] += 1
        selected = self.in_attn and self.step == STEP
        if selected:
            assert len(args) == 3 and set(kwargs) == {'is_causal', 'enable_gqa', 'backend', 'parallel_config'}
            assert kwargs['enable_gqa'] is True and kwargs['parallel_config'] is None
            name = 'causal' if kwargs['is_causal'] is True else 'gen'
            assert type(kwargs['is_causal']) is bool and name not in self.data
            self.calls['selected_' + name] += 1
            q, k, v = args
            layout = self.data['layout']
            q_len, kv_len = (layout['und_len'], layout['und_len']) if name == 'causal' else (layout['gen_len'], layout['und_len'] + layout['gen_len'])
            assert q.shape == (1, q_len, HEADS, HEAD_DIM)
            assert k.shape == v.shape == (1, kv_len, KV_HEADS, HEAD_DIM)
            assert q.dtype == k.dtype == v.dtype == self.torch.bfloat16
            assert all(self.torch.isfinite(value).all() for value in args)
            self.data[name] = dict(q=self.runtime.cpu(q), k=self.runtime.cpu(k), v=self.runtime.cpu(v),
                argument_metadata=[dict(shape=list(value.shape), stride=list(value.stride()), dtype=str(value.dtype), device=str(value.device)) for value in args],
                kwargs=dict(is_causal=kwargs['is_causal'], enable_gqa=kwargs['enable_gqa'],
                            backend_repr=repr(kwargs['backend']), parallel_config_repr=repr(kwargs['parallel_config'])))
            if name == 'gen':
                # Keep the actual objects: no dtype/shape/stride reconstruction.
                self.replay_args, self.replay_kwargs = args, dict(kwargs)
        output = self.original(*args, **kwargs)
        if selected:
            assert output.shape == (1, q_len, HEADS, HEAD_DIM) and output.dtype == self.torch.bfloat16
            self.data[name]['output'] = self.runtime.cpu(output)
        return output

    def before_W(self, module, args):
        assert self.in_attn and len(args) == 1
        self.calls['pre_W'] += 1
        if self.step == STEP:
            value = args[0]
            assert value.shape == (self.data['layout']['gen_len'], 4096) and value.dtype == self.torch.bfloat16
            self.data['pre_W'] = self.runtime.cpu(value)
            self.runtime.require_exact(self.data['pre_W'], self.data['gen']['output'].squeeze(0).flatten(-2, -1), 'actual native dispatch -> pre_W bytes')

    def begin(self):
        assert self.module.dispatch_attention_fn is self.original and not self.installed
        self.module.dispatch_attention_fn = self.wrapper
        self.installed = True
        try:
            self.handles.append(self.model.register_forward_pre_hook(self.before_model, with_kwargs=True))
            self.handles.append(self.model.register_forward_hook(self.after_model))
            self.handles.append(self.attn.register_forward_pre_hook(self.before_attn))
            self.handles.append(self.attn.register_forward_hook(self.after_attn))
            self.handles.append(self.attn.to_add_out.register_forward_pre_hook(self.before_W))
        except BaseException:
            self.reset()
            raise

    def reset(self):
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        if self.installed:
            unchanged = self.module.dispatch_attention_fn is self.wrapper
            self.module.dispatch_attention_fn = self.original
            self.installed = False
            self.restored = self.module.dispatch_attention_fn is self.original
            assert unchanged, 'Another operation changed the wrapped dispatch symbol'

    def replay(self):
        torch, runtime = self.torch, self.runtime
        assert self.restored and not self.handles and not self.in_model and not self.in_attn
        assert self.calls['model_before'] == self.calls['model_after'] == 30
        assert self.calls['target_attn_before'] == self.calls['target_attn_after'] == self.calls['pre_W'] == 30
        assert self.calls['native_dispatch'] == 30 * 36 * 2
        assert self.calls['selected_causal'] == self.calls['selected_gen'] == 1
        assert self.calls['official_attention_replay'] == 0
        assert settings(torch, self.attn.processor) == self.backend_settings
        for name, value, original_meta in zip(('q', 'k', 'v'), self.replay_args, self.data['gen']['argument_metadata']):
            runtime.require_exact(runtime.cpu(value), self.data['gen'][name], 'retained actual replay argument/' + name)
            assert list(value.stride()) == original_meta['stride'] and str(value.device) == original_meta['device']
        device = self.replay_args[0].device
        before_cpu, before_cuda = torch.get_rng_state(), torch.cuda.get_rng_state(device)
        output = self.original(*self.replay_args, **self.replay_kwargs)
        self.calls['official_attention_replay'] += 1
        runtime.require_exact(runtime.cpu(output), self.data['gen']['output'], 'original FULL_GEN original-shape attention replay bytes')
        runtime.require_exact(torch.get_rng_state(), before_cpu, 'official attention replay CPU RNG unchanged')
        runtime.require_exact(torch.cuda.get_rng_state(device), before_cuda, 'official attention replay CUDA RNG unchanged')
        assert settings(torch, self.attn.processor) == self.backend_settings
        self.data['replay_output'] = runtime.cpu(output)
        self.replay_args = self.replay_kwargs = None
        return dict(complete=True, layer_zero_based=LAYER, denoising_step=STEP,
            counts=dict(self.calls), all_owned_hooks_removed=not self.handles,
            original_dispatch_symbol_restored=self.restored,
            original_full_GEN_attention_replay_bit_exact=True, replay_global_CPU_and_CUDA_RNG_unchanged=True,
            original_dispatch_inputs_and_return_unmodified=True, intervention_count=0,
            backend_settings=self.backend_settings, scope=SCOPE)


def token_indexes(kwargs, model, components, runtime, scan):
    """Current-frame indexes are joint/global keys; action queries are GEN rows."""
    torch = runtime.torch
    layout = components._layout(kwargs)
    vision = kwargs['vision_tokens'][0].squeeze(0)
    assert vision.ndim == 4
    channels, frames, height, width = vision.shape
    assert channels == model.config.latent_channel
    patch = int(model.config.latent_patch_size)
    grid = (frames, (height + patch - 1) // patch, (width + patch - 1) // patch)
    assert tuple(kwargs['vision_token_shapes'][0]) == grid
    noisy = kwargs['vision_noisy_frame_indexes'][0].flatten()
    assert noisy.dtype == torch.long and noisy.unique().numel() == noisy.numel()
    assert sorted(noisy.tolist()) == list(range(1, frames)), 'Require current conditioned frame0 and all future frames noisy'
    source = inspect.getsource(type(model)._patchify_and_pack_latents)
    assert 'cthpwq->thwpqc' in source and 'latent = latent.squeeze(0)' in source
    packed, shapes = model._patchify_and_pack_latents(kwargs['vision_tokens'])
    assert shapes == [(frames, height, width)] and packed.shape[0] == grid[0] * grid[1] * grid[2]
    global_vision = kwargs['vision_sequence_indexes'].flatten()
    assert global_vision.numel() == packed.shape[0]
    spatial = grid[1] * grid[2]
    current_keys = global_vision[:spatial]
    assert bool((current_keys >= layout['und_len']).all())
    return dict(und_len=layout['und_len'], gen_len=layout['gen_len'],
        full_joint_length=layout['und_len'] + layout['gen_len'], text_global_key_indexes=kwargs['text_indexes'].flatten(),
        action_global_indexes=layout['global_action_indexes'], action_GEN_query_rows=layout['action_rows'],
        vision_global_key_indexes=global_vision, current_vision_global_key_indexes=current_keys,
        current_vision_GEN_rows=current_keys - layout['und_len'], future_vision_global_key_indexes=global_vision[spatial:],
        conditioned_frames=[0], noisy_frames=noisy, vision_C_T_H_W=list(vision.shape), patch_grid=list(grid),
        packing_method_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        global_key_convention='Actual dispatch allK/allV are concatenated UND then GEN and share the joint/global sequence indexes.',
        query_convention='Only GEN query rows subtract und_len; text/current-vision key indexes remain global.',
        current_condition_tensor=vision[:, :1].clone(),
        head_to_KV_group=[head // (HEADS // KV_HEADS) for head in range(HEADS)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Completed original baseline containing position-threshold')
    baseline = parser.parse_args().output.resolve()
    contract, sources, threshold = threshold_sources(baseline)
    paths = {key: Path(contract['frozen_files'][key]['path']) for key in ('normal_runtime', 'factory', 'components')}
    paths['metrics'] = ROOT / 'work/probe_cosmos_milk_component_pulses.py'
    feedback_provenance = read(baseline / 'feedback-inputs/provenance.json')
    assert sha(paths['metrics']) == feedback_provenance['pulse_contract_sha256']
    out = baseline / 'target-reader-inputs'
    out.mkdir(exist_ok=False)
    protocol = dict(seeds=list(SEEDS), scene='x12', task='milk', task_index=7, prompt=PROMPT,
        loaded_runtime_task_tuple=[7, 'pick_up_the_milk_and_place_it_in_the_basket', PROMPT, 'milk_1'],
        layer_zero_based=LAYER, layer_human=LAYER + 1,
        denoising_step=STEP, actual_action_query_rows=16, heads=HEADS, KV_heads=KV_HEADS, head_dim=HEAD_DIM,
        native_q0_predictions=3, native_model_forwards=90, extra_official_full_GEN_attention_replays=3,
        intervention_count=0, capture_only_selected_step=True, source_noise_pairing='Each seed is checked against its own source; cross-seed noise is different.',
        native_full_record_contract='All NormalRuntime saved fields, including both noise draws, prepared inputs, complete schedule and future latents.',
        source_case_hash_scope='Source q0 case hashes are first observed at this new stage, not claimed as a historical whole-case seal.',
        semantic_result=False, scope=SCOPE)
    write(out / 'protocol.json', protocol, exclusive=True)
    write(out / 'sources.json', dict(threshold=threshold, cases=sources), exclusive=True)
    started, completed, observer = time.perf_counter(), [], None
    try:
        os.environ['HF_HUB_OFFLINE'] = '1'
        base = load('reader_checked_runtime', paths['normal_runtime'])
        base.TASKS['milk'] = (7, 'pick_up_the_milk_and_place_it_in_the_basket', PROMPT, 'milk_1')
        factory = load('reader_checked_factory', paths['factory'])
        factory.OUT = out
        assert sha(factory.SCHEDULER) == contract['frozen_files']['scheduler']['sha256']
        runtime = base.NormalRuntime(factory)
        torch, model = runtime.torch, runtime.pipe.transformer
        components = load('reader_readonly_layout', paths['components'])
        scan = load('reader_record_and_bytes_contract', paths['metrics'])
        checker = components.CosmosComponentInterventions(model)
        assert sha(Path(inspect.getfile(type(model)))) == feedback_provenance['actual_transformer_sha256']
        assert sha(Path(inspect.getfile(type(runtime.pipe)))) == feedback_provenance['actual_pipeline_source_sha256']
        processor_module = inspect.getmodule(type(model.layers[LAYER].self_attn.processor))
        original_dispatch = processor_module.dispatch_attention_fn
        dispatch_path = Path(inspect.getfile(original_dispatch))
        provenance = dict(script_sha256=sha(Path(__file__)), protocol_sha256=sha(out / 'protocol.json'),
            sources_sha256=sha(out / 'sources.json'), threshold=threshold,
            source_sha256={key: sha(path) for key, path in paths.items()},
            actual_transformer_source=str(Path(inspect.getfile(type(model)))),
            actual_transformer_sha256=sha(Path(inspect.getfile(type(model)))),
            actual_pipeline_source_sha256=sha(Path(inspect.getfile(type(runtime.pipe)))),
            dispatch_source=str(dispatch_path), dispatch_source_sha256=sha(dispatch_path),
            dispatch_function_source_sha256=hashlib.sha256(inspect.getsource(original_dispatch).encode()).hexdigest(),
            dispatch_signature=str(inspect.signature(original_dispatch)),
            backend_settings=settings(torch, model.layers[LAYER].self_attn.processor),
            library_edits=False, training=False, model_inputs_changed=False, model_outputs_changed=False,
            action_interventions=0, closed_loop_execution=False, scope=SCOPE)
        write(out / 'provenance.json', provenance, exclusive=True)
        captured, records, results = {}, {}, []
        for seed in SEEDS:
            source = sources[str(seed)]
            source_folder = Path(source['q0_folder'])
            for name, expected in source['q0_files_current_sha256'].items():
                assert sha(source_folder / name) == expected
            original_record = torch.load(source_folder / 'states.pt', map_location='cpu', weights_only=True, mmap=True)
            original_cache = torch.load(source_folder / 'components.pt', map_location='cpu', weights_only=True, mmap=True)
            scan.validate_record(original_record, torch)
            scan.validate_cache(original_cache, original_record, checker, runtime, source['trial'])
            folder = out / f'seed{seed}'
            folder.mkdir(exist_ok=False)
            observer = NativeDispatchObserver(runtime, components, scan, original_cache)
            observer.begin()
            try:
                record, metadata = runtime.predict('milk', Path(source['image']), seed, folder / 'q0')
            finally:
                observer.reset()
                torch.save(observer.data, folder / 'dispatch-t29.pt')
            scan.validate_record(record, torch)
            runtime.require_exact(record, original_record, f'seed{seed}/ENTIRE original q0 record including future latents')
            assert metadata['model_calls'] == 30 and metadata['prepare_calls'] == 1
            assert metadata['task'] == 'milk' and metadata['task_index'] == 7 and metadata['prompt'] == PROMPT
            assert metadata['seed'] == seed and metadata['input_png_sha256'] == source['image_sha256']
            report = observer.replay()
            data = observer.data
            assert type(data['model_output']) is tuple and len(data['model_output']) == 3
            assert type(data['model_output'][2]) is list and len(data['model_output'][2]) == 1
            runtime.require_exact(data['model_output'][2][0], record['action_velocity'][STEP], 'actual captured t29 full return action head')
            indexes = token_indexes(data['model_input'], model, components, runtime, scan)
            und_len = indexes['und_len']
            assert data['gen']['k'].shape[1] == data['gen']['v'].shape[1] == indexes['full_joint_length']
            runtime.require_exact(data['causal']['v'], data['gen']['v'][:, :und_len], 'UND causal/GEN values same actual source')
            torch.save(data, folder / 'dispatch-t29.pt')
            torch.save(data['model_input'], folder / 'model-input-t29.pt')
            torch.save(data['model_output'], folder / 'model-output-t29.pt')
            torch.save(indexes, folder / 'token-indexes.pt')
            write(folder / 'token-indexes.json', {key: scan.tensor_meta(value, torch) if isinstance(value, torch.Tensor) else value for key, value in indexes.items()}, exclusive=True)
            write(folder / 'observer-report.json', report, exclusive=True)
            write(folder / 'dispatch-t29.json', dict(causal={key: scan.tensor_meta(value, torch) if isinstance(value, torch.Tensor) else value for key, value in data['causal'].items()},
                gen={key: scan.tensor_meta(value, torch) if isinstance(value, torch.Tensor) else value for key, value in data['gen'].items()},
                actual_pre_W=scan.tensor_meta(data['pre_W'], torch), replay_output=scan.tensor_meta(data['replay_output'], torch),
                tensor_conventions='Q/K are actual dispatch inputs after native normalization and RoPE; V is native projected V without RoPE.',
                native_dispatch_flatten_equals_pre_W_bytes=True), exclusive=True)
            files = ('dispatch-t29.pt', 'dispatch-t29.json', 'model-input-t29.pt', 'model-output-t29.pt',
                     'token-indexes.pt', 'token-indexes.json', 'observer-report.json',
                     'q0/states.pt', 'q0/metadata.json', 'q0/normalized_actions.json')
            results.append(dict(seed=seed, trial=source['trial'], folder=str(folder),
                original_entire_q0_record_bit_exact=True, official_full_GEN_replay_bit_exact=True,
                counts=report['counts'], intervention_count=0,
                files_sha256={name: sha(folder / name) for name in files}))
            captured[seed], records[seed] = (data, indexes), record
            completed.append(seed)
            write(out / 'progress.json', dict(completed_seeds=completed, q0_predictions=len(completed),
                native_model_calls=30 * len(completed), official_attention_replays=len(completed), elapsed_s=time.perf_counter() - started))
            observer = None
        cross_seed = []
        for seed in SEEDS[1:]:
            a, ai = captured[SEEDS[0]]
            b, bi = captured[seed]
            assert ai['und_len'] == bi['und_len']
            und_len = ai['und_len']
            for key in ('input_ids', 'text_indexes'):
                runtime.require_exact(a['model_input'][key], b['model_input'][key], 'same milk UND text/' + key)
            runtime.require_exact(a['model_input']['position_ids'][..., :und_len], b['model_input']['position_ids'][..., :und_len], 'same UND rotary positions')
            for path in ('causal', 'gen'):
                for key in ('k', 'v'):
                    left, right = a[path][key], b[path][key]
                    if path == 'gen':
                        left, right = left[:, :und_len], right[:, :und_len]
                    runtime.require_exact(left, right, f'same milk UND {path}/{key}/seed{seed}')
            runtime.require_exact(ai['current_condition_tensor'], bi['current_condition_tensor'], 'same actual current condition frame0')
            for key in ('timesteps', 'sigmas'):
                runtime.require_exact(records[SEEDS[0]][key], records[seed][key], 'same cross-seed schedule/' + key)
            draw_differences = []
            for index, (left, right) in enumerate(zip(records[SEEDS[0]]['pure_noise'], records[seed]['pure_noise'])):
                assert left.shape == right.shape and left.dtype == right.dtype
                differs = not torch.equal(left, right)
                assert differs, 'Different sampling seeds must not be reported as paired identical noise'
                draw_differences.append(dict(draw=index, first_seed=scan.tensor_meta(left, torch),
                    second_seed=scan.tensor_meta(right, torch), arrays_different=True))
            cross_seed.append(dict(seeds=[SEEDS[0], seed], same_milk_UND_actual_K_V_bytes_exact=True,
                causal_UND_and_GEN_UND_K_V_checked_separately=True,
                current_condition_frame0_bytes_exact=True, actual_two_noise_draws_different=draw_differences))
        assert sum(row['counts']['model_before'] for row in results) == 90
        assert sum(row['counts']['official_attention_replay'] for row in results) == 3
        assert processor_module.dispatch_attention_fn is original_dispatch
        write(out / 'results.json', dict(state='complete', native_q0_predictions=3, native_model_forwards=90,
            extra_official_attention_replays=3, cases=results, cross_seed_checks=cross_seed,
            all_entire_original_q0_records_bit_exact=True, same_milk_UND_actual_K_V_across_seeds_bit_exact=True,
            source_case_hash_scope=protocol['source_case_hash_scope'], semantic_result=False, scope=SCOPE), exclusive=True)
        write(out / 'complete.json', dict(state='complete', native_q0_predictions=3, native_model_forwards=90,
            extra_official_full_GEN_attention_replays=3, action_interventions=0,
            all_original_entire_q0_records_bit_exact=True, all_full_GEN_native_dispatch_replays_bit_exact=True,
            same_milk_UND_actual_postnorm_RoPE_K_and_projected_V_bit_exact=True,
            original_dispatch_symbol_restored=True, all_owned_hooks_removed=True,
            results_sha256=sha(out / 'results.json'), provenance_sha256=sha(out / 'provenance.json'),
            protocol_sha256=sha(out / 'protocol.json'), sources_sha256=sha(out / 'sources.json'),
            script_sha256=sha(Path(__file__)), semantic_result=False, scope=SCOPE), exclusive=True)
        print('[TARGET-READER-INPUTS] ' + json.dumps(dict(state='complete', native_model_forwards=90, official_attention_replays=3)), flush=True)
    except Exception as exc:
        write(out / 'failed.json', dict(state='failed', error=str(exc), traceback=traceback.format_exc(),
            completed_seeds=completed, native_predictions_completed=len(completed), scope=SCOPE))
        raise
    finally:
        if observer is not None:
            observer.reset()


if __name__ == '__main__':
    main()
