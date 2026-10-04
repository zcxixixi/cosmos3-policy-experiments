"""Ten fresh q0 predictions separating two future-vision attention read edges.

Use --output BASE; only absent BASE/future-read-cuts is created. Fixed A195,
V195/V198, two untouched native controls, two numerical masked shams, and six
future-to-current/action cuts. All masked arms substitute the same 66 GEN
query rows before the original full-266-row output projection. A bool mask
changes native attention arithmetic: the sham is a numerical control, not a
no-op. At each cut site, every unblocked query must equal its same-input
all-allowed result byte for byte. Failure stops; no residual repair is used.

Only q0 inference runs here: 300 model forwards, no physics or training.
All ten later physical trials, including both shams, are preregistered; they
are a separate stage. Both-cut equality is an engineering prediction, not
an object identity interpretation or proof of the original target mechanism.
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
FACTOR_SHA = '9081abb51200d1335bb4f84fde87fc23eec45a97ca46789f2dcf700acaf26c1e'
FEASIBILITY_SHA = '5cc1f9a1daa609cc27262c171967ff66c4884cc951f72f72f9ee0f0cb447ca2c'
EXECUTION_SHA = '985f10b079684689dcd9735f8f55159c97976eaef1ac33ff163011980f6ca9d8'
DISPATCH_SHA = '93a57f07a77f8710233458ce8adc300e3eeaa57cb09a46a94cee02121d38ed88'
PROMPT = 'pick up the milk and place it in the basket'
ARMS = ('native', 'all_allowed', 'cut_future_current', 'cut_future_action', 'cut_future_both')
PLAN = tuple((f'V{v}_A195_{arm}', v, arm) for arm in ARMS for v in (195, 198))
SOURCE_FILES = ('states.pt', 'normalized_actions.json', 'metadata.json', 'noise-audit.pt', 'noise-audit.json')
SCOPE = ('Ten fixed-A195 X+12cm q0 predictions with two future-noise realizations. '
         'Masked attention arithmetic and propagated actual current/action hidden states, '
         'raw action outputs and solver states only. Numerical shams can change the native '
         'trajectory. No training, new physics, latent identity decoder, target commitment '
         'or root-cause claim. All ten physical cases remain a separately preregistered stage.')


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


def preflight(baseline):
    """Use the frozen source seam and already completed feasibility evidence."""
    factor_path = ROOT / 'work/probe_cosmos_milk_initial_noise_factors.py'
    assert sha(factor_path) == FACTOR_SHA
    factor = load('future_cuts_frozen_noise_sources', factor_path)
    contract, threshold_sources, image, paths, upstream = factor.source_contract(baseline)
    producer = baseline / 'initial-noise-factors'
    assert not (producer / 'failed.json').exists()
    complete, results, provenance = (read(producer / (name + '.json')) for name in ('complete', 'results', 'provenance'))
    assert complete['state'] == results['state'] == 'complete'
    assert complete['script_sha256'] == provenance['script_sha256'] == FACTOR_SHA
    assert complete['fresh_q0_predictions'] == 4 and complete['fresh_model_forwards'] == 120
    for key in ('both_native_entire_q0_records_byte_exact', 'all_current_frame_schedule_layout_padding_and_source_noise_gates_passed',
                'original_randn_symbol_restored', 'all_owned_observer_hooks_removed'):
        assert complete[key] is True
    for name in ('results', 'provenance', 'protocol', 'sources'):
        assert complete[name + '_sha256'] == sha(producer / (name + '.json'))
    rows = {row['cell']: row for row in results['cases']}
    assert set(rows) == set(factor.PLAN) and len(results['cases']) == 4
    source_pair = {}
    for v in (195, 198):
        label, folder = f'V{v}_A195', producer / f'V{v}_A195'
        row = rows[label]
        assert row['vision_noise_source_seed'] == v and row['action_noise_source_seed'] == row['runtime_rng_seed'] == 195
        assert row['fresh_model_forwards'] == 30 and row['all_actual_noise_source_and_preparation_gates_passed'] is True
        assert row['all_owned_symbols_and_observer_hooks_restored'] is True
        assert set(row['files_sha256']) == set(SOURCE_FILES)
        for name, expected in row['files_sha256'].items():
            assert sha(folder / name) == expected
        source_pair[str(v)] = dict(cell=label, folder=str(folder), files_sha256=row['files_sha256'])
    feasibility = baseline / 'current-mask-dispatch-output'
    assert not (feasibility / 'failed.json').exists()
    fc, fr, fp = (read(feasibility / (name + '.json')) for name in ('complete', 'results', 'provenance'))
    feasibility_path = ROOT / 'work/probe_cosmos_current_mask_dispatch.py'
    assert sha(feasibility_path) == fc['script_sha256'] == fp['script_sha256'] == FEASIBILITY_SHA
    assert fc['state'] == fr['state'] == 'complete' and fc['official_dispatch_calls'] == 15
    assert fc['model_forwards'] == fc['physics_calls'] == 0 and fr['all_native_full_GEN_replays_byte_exact'] is True
    for name in ('results', 'provenance', 'protocol'):
        assert fc[name + '_sha256'] == sha(feasibility / (name + '.json'))
    assert fp['dispatch_source_sha256'] == DISPATCH_SHA and fp['attn_mask_supported'] is True
    assert len(fr['cases']) == 3 and {row['seed'] for row in fr['cases']} == {195, 196, 198}
    for row in fr['cases']:
        assert row['dispatch_calls'] == 5
        for name, expected in row['files_sha256'].items():
            assert sha(feasibility / f"seed{row['seed']}" / name) == expected
        assert row['report']['all_allowed_current_byte_exact'] is False
    execution = baseline / 'initial-noise-execution'
    ec = read(execution / 'complete.json')
    assert ec['state'] == 'complete' and ec['script_sha256'] == EXECUTION_SHA
    assert sha(ROOT / 'work/run_cosmos_milk_initial_noise_execution.py') == EXECUTION_SHA
    assert ec['physical_trials'] == 4 and ec['fresh_model_forwards'] == 840
    assert ec['two_native_complete_129_state_and_all_action_trajectory_and_PNG_controls_exact'] is True
    assert ec['all_four_own_cached_q0_action_execution_controls_exact'] is True
    for name, expected in ec['files_sha256'].items():
        assert sha(execution / name) == expected
    physical = {}
    for v in (195, 198):
        label, folder = f'V{v}_A195', execution / 'closed-loop' / f'V{v}_A195'
        names = ('sim_states.npy', 'summary.json', 'trajectory.json', 'normalized_actions.json',
                 'denormalized_actions.json', 'actions.json', 'executed_actions.json',
                 *[f'input_{q:02d}.png' for q in range(8)],
                 *[f'chunk_{q:02d}/{n}' for q in range(8) for n in ('states.pt', 'metadata.json', 'normalized_actions.json')])
        physical[str(v)] = dict(cell=label, folder=str(folder), files_current_sha256={name: sha(folder / name) for name in names},
            hash_scope='Whole physical reference files first rehashed here; prior execution complete/document seals are independently anchored.')
    paths['reader'] = ROOT / 'work/probe_cosmos_milk_target_reader_inputs.py'
    assert sha(paths['reader']) == factor.READER_SHA
    anchors = dict(factor_documents_sha256={name: sha(producer / (name + '.json')) for name in ('complete', 'results', 'provenance', 'protocol', 'sources')},
        feasibility_documents_sha256={name: sha(feasibility / (name + '.json')) for name in ('complete', 'results', 'provenance', 'protocol')},
        physical_execution_complete_sha256=sha(execution / 'complete.json'), physical_execution_files_sha256=ec['files_sha256'],
        feasibility_actual_registry=fp, upstream=upstream, source_pair=source_pair, physical_references=physical)
    return factor, contract, threshold_sources, image, paths, anchors


def indexes(kwargs, model, components, runtime):
    """Derive from actual pack metadata and verified time-major source, without recomputing the processor or patchifier."""
    torch = runtime.torch
    layout = components._layout(kwargs)
    assert layout['und_len'] == 121 and layout['gen_len'] == 266
    vision, action = kwargs['vision_tokens'][0], kwargs['action_tokens'][0]
    assert vision.shape == (1, 48, 5, 10, 20) and action.shape == (16, 64)
    assert vision.dtype == action.dtype == torch.bfloat16
    assert model.config.latent_channel == 48 and model.config.latent_patch_size == 2
    assert tuple(kwargs['vision_token_shapes'][0]) == (5, 5, 10)
    assert kwargs['vision_noisy_frame_indexes'][0].flatten().tolist() == [1, 2, 3, 4]
    source = inspect.getsource(type(model)._patchify_and_pack_latents)
    assert 'cthpwq->thwpqc' in source and 'latent = latent.squeeze(0)' in source
    vision_keys = kwargs['vision_sequence_indexes'].detach().cpu().flatten()
    action_rows, action_keys = layout['action_rows'], layout['global_action_indexes']
    assert vision_keys.dtype == torch.long and vision_keys.numel() == 250
    assert action_rows.numel() == action_keys.numel() == 16
    assert torch.equal(action_keys, kwargs['action_sequence_indexes'].detach().cpu().flatten())
    current, future = vision_keys[:50] - 121, vision_keys[50:]
    union = torch.cat((current, action_rows))
    future_rows = future - 121
    assert union.unique().numel() == 66
    assert torch.equal(torch.cat((union, future_rows)).sort().values, torch.arange(266))
    assert torch.equal(kwargs['text_indexes'].detach().cpu().flatten(), torch.arange(121))
    assert torch.equal(torch.cat((kwargs['text_indexes'].detach().cpu().flatten(), vision_keys, action_keys)).sort().values, torch.arange(387))
    return dict(und_len=121, gen_len=266, full_joint_length=387, current_rows=current, action_rows=action_rows,
        union_rows=union, future_rows=future_rows, future_keys=future, current_keys=vision_keys[:50], action_keys=action_keys,
        text_keys=torch.arange(121), packing_source_sha256=hashlib.sha256(source.encode()).hexdigest())


def masks_for(index, torch):
    masks = {}
    for arm in ARMS[1:]:
        mask = torch.ones((1, 1, 266, 387), dtype=torch.bool)
        blocked = (index['current_rows'] if arm == 'cut_future_current' else index['action_rows'] if arm == 'cut_future_action'
                   else index['union_rows'] if arm == 'cut_future_both' else torch.empty(0, dtype=torch.long))
        if blocked.numel():
            mask[0, 0, blocked[:, None], index['future_keys'][None, :]] = False
        unblocked = torch.tensor([i for i in range(266) if i not in set(blocked.tolist())], dtype=torch.long)
        assert int((~mask).sum()) == blocked.numel() * 200
        assert bool(mask[0, 0, unblocked].all()) and bool(mask[0, 0, :, index['text_keys']].all())
        assert bool(mask[0, 0, :, index['current_keys']].all()) and bool(mask[0, 0, :, index['action_keys']].all())
        assert bool(mask.any(dim=-1).all())
        masks[arm] = dict(mask=mask, blocked_rows=blocked, unblocked_rows=unblocked)
    return masks


def byte_exact(a, b, torch):
    return (a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
            and torch.equal(a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8)))


def discrepancy(a, b, torch):
    delta = a.detach().cpu().double() - b.detach().cpu().double()
    return dict(byte_exact=byte_exact(a, b, torch), delta_l2=float(delta.norm()),
        delta_maxabs=float(delta.abs().max()), changed_values=int(torch.count_nonzero(delta)))


class FutureReadCuts:
    """Owned observational hooks plus one dispatch-return row substitution seam."""

    def __init__(self, runtime, components, scan, reader, cache, clean_current, source, arm, folder):
        self.runtime, self.torch, self.components, self.scan = runtime, runtime.torch, components, scan
        self.reader, self.cache, self.clean_current, self.source, self.arm, self.folder = reader, cache, clean_current, source, arm, folder
        self.model = runtime.pipe.transformer
        self.module = inspect.getmodule(type(self.model.layers[0].self_attn.processor))
        self.original, self.wrapper = self.module.dispatch_attention_fn, self.dispatch
        self.handles, self.calls, self.files, self.events, self.boundaries, self.outputs = [], [], [], [], [], []
        self.first, self.index, self.masks, self.gpu_masks = None, None, None, None
        self.step, self.active, self.next_layer = -1, None, 0
        self.in_model, self.installed, self.restored, self.pending = False, False, False, None
        self.counts = dict(model_before=0, model_after=0, block_before=0, block_after=0, pre_W=0,
            native_UND_dispatch=0, native_GEN_dispatch=0, all_allowed_dispatch=0, cut_dispatch=0, substituted_sites=0)
        self.settings = reader.settings(self.torch, self.model.layers[0].self_attn.processor)

    def before_model(self, module, args, kwargs):
        assert not args and not self.in_model and self.active is None
        self.step += 1
        assert self.step < 30 and kwargs['return_dict'] is False
        layout = self.components._layout(kwargs)
        actual = dict(layout=layout, metadata={key: self.runtime.cpu(kwargs.get(key)) for key in self.components._STRUCTURAL_KEYS})
        self.runtime.require_exact(actual, self.cache['steps'][self.step], 'actual schedule/layout metadata/' + str(self.step))
        derived = indexes(kwargs, self.model, self.components, self.runtime)
        if self.index is None:
            self.index = derived
            self.masks = masks_for(derived, self.torch)
            self.gpu_masks = {name: item['mask'].to(kwargs['vision_tokens'][0].device) for name, item in self.masks.items()}
            self.first = self.runtime.cpu(kwargs)
            self.runtime.require_exact(self.first, self.source['model_input'], 'initial entire kwargs of selected source pair')
            self.torch.save(dict(indexes=self.index, masks=self.masks), self.folder / 'indexes-and-masks.pt')
        else:
            self.runtime.require_exact(derived, self.index, 'all-step actual index mapping')
        self.runtime.require_exact(self.runtime.cpu(kwargs['vision_tokens'][0][0, :, 0]), self.clean_current, 'actual conditioned current frame0')
        assert self.torch.count_nonzero(kwargs['action_tokens'][0][:, 10:]) == 0
        self.calls.append(actual)
        self.counts['model_before'] += 1
        self.in_model, self.next_layer = True, 0
        self.boundaries, self.events = [], []

    def selected_hidden(self, gen):
        assert gen.shape == (266, 4096) and gen.dtype == self.torch.bfloat16
        return dict(current=self.runtime.cpu(gen.index_select(0, self.index['current_rows'].to(gen.device))),
                    action=self.runtime.cpu(gen.index_select(0, self.index['action_rows'].to(gen.device))))

    def before_block(self, layer, module, args):
        assert self.in_model and self.active is None and layer == self.next_layer and len(args) == 3
        assert args[0].shape == (121, 4096) and args[1].shape == (266, 4096)
        if layer == 0:
            self.boundaries.append(self.selected_hidden(args[1]))
        self.active, self.pending, self.site_calls = layer, None, []
        self.counts['block_before'] += 1

    def dispatch(self, *args, **kwargs):
        torch = self.torch
        assert self.in_model and self.active is not None
        assert len(args) == 3 and set(kwargs) == {'is_causal', 'enable_gqa', 'backend', 'parallel_config'}
        assert kwargs['enable_gqa'] is True and kwargs['backend'] is kwargs['parallel_config'] is None
        assert type(kwargs['is_causal']) is bool
        causal = kwargs['is_causal']
        assert self.site_calls == ([] if causal else ['UND']), 'Exactly one native UND then one native GEN dispatch per actual layer'
        self.site_calls.append('UND' if causal else 'GEN')
        qlen, kvlen = (121, 121) if causal else (266, 387)
        assert args[0].shape == (1, qlen, 32, 128) and args[1].shape == args[2].shape == (1, kvlen, 8, 128)
        assert all(value.dtype == torch.bfloat16 and bool(torch.isfinite(value).all()) for value in args)
        self.counts['native_UND_dispatch' if causal else 'native_GEN_dispatch'] += 1
        native = self.original(*args, **kwargs)
        assert native.shape == (1, qlen, 32, 128) and native.dtype == torch.bfloat16 and bool(torch.isfinite(native).all())
        if causal:
            return native
        argument_metadata = [dict(shape=list(value.shape), stride=list(value.stride()), dtype=str(value.dtype), device=str(value.device)) for value in args]
        event = dict(step=self.step, layer_zero_based=self.active, arm=self.arm, original_GEN_dispatch_calls=1,
            original_UND_dispatch_calls=1, extra_all_allowed_calls=0, extra_cut_calls=0,
            argument_metadata=argument_metadata, native_kwargs=dict(is_causal=False, enable_gqa=True, backend=None, parallel_config=None),
            to_add_out_shape=[266, 4096], to_add_out_actual_calls=0, processor_recompute_calls=0)
        if self.arm == 'native':
            merged = native
            event.update(substituted_query_rows=0, all_nonselected_native_bytes_preserved=True, UND_dispatch_return_unmodified=True)
        else:
            # Keep these exact live Q/K/V objects, full shapes, strides and kwargs.
            # Transient witnesses are not exported as large QKV attachments.
            original_inputs = tuple(value.clone(memory_format=torch.preserve_format) for value in args)
            cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state(args[0].device).clone()
            self.counts['all_allowed_dispatch'] += 1
            allowed = self.original(*args, **dict(kwargs, attn_mask=self.gpu_masks['all_allowed']))
            event['extra_all_allowed_calls'] = 1
            assert allowed.shape == native.shape and allowed.dtype == native.dtype and bool(torch.isfinite(allowed).all())
            selected = allowed
            if self.arm != 'all_allowed':
                self.counts['cut_dispatch'] += 1
                selected = self.original(*args, **dict(kwargs, attn_mask=self.gpu_masks[self.arm]))
                event['extra_cut_calls'] = 1
                assert selected.shape == native.shape and selected.dtype == native.dtype and bool(torch.isfinite(selected).all())
                unblocked = self.masks[self.arm]['unblocked_rows'].to(selected.device)
                passed = byte_exact(selected.index_select(1, unblocked), allowed.index_select(1, unblocked), torch)
                event['all_unblocked_GEN_rows_vs_same_input_all_allowed_byte_exact'] = passed
                if not passed:
                    torch.save(dict(native=self.runtime.cpu(native), all_allowed=self.runtime.cpu(allowed), cut=self.runtime.cpu(selected),
                        mask=self.masks[self.arm]['mask'], event=event, unblocked_rows=self.masks[self.arm]['unblocked_rows']),
                        self.folder / f'failed-site-t{self.step:02d}-L{self.active:02d}.pt')
                    raise AssertionError('Cut changed an unblocked GEN query relative to its same-input all-allowed dispatch')
            for value, before in zip(args, original_inputs):
                assert byte_exact(value, before, torch), 'Extra official dispatch mutated actual Q/K/V'
            assert byte_exact(cpu_rng, torch.get_rng_state(), torch) and byte_exact(cuda_rng, torch.cuda.get_rng_state(args[0].device), torch)
            union, future = (self.index[key].to(native.device) for key in ('union_rows', 'future_rows'))
            merged = native.clone()
            merged[:, union] = selected[:, union]
            assert byte_exact(merged.index_select(1, future), native.index_select(1, future), torch)
            assert byte_exact(merged.index_select(1, union), selected.index_select(1, union), torch)
            union_native, union_allowed, union_selected = (value.index_select(1, union) for value in (native, allowed, selected))
            event.update(substituted_query_rows=66, selected_current_rows=50, selected_action_rows=16,
                blocked_query_rows=int(self.masks[self.arm]['blocked_rows'].numel()),
                unblocked_query_rows=int(self.masks[self.arm]['unblocked_rows'].numel()),
                forbidden_query_key_pairs=int((~self.masks[self.arm]['mask']).sum()),
                mask=self.scan.tensor_meta(self.masks[self.arm]['mask'], torch),
                all_allowed_mask=self.scan.tensor_meta(self.masks['all_allowed']['mask'], torch),
                native_selected_sha256=self.scan.tensor_meta(union_native, torch),
                same_input_all_allowed_selected_sha256=self.scan.tensor_meta(union_allowed, torch),
                actual_returned_selected_sha256=self.scan.tensor_meta(union_selected, torch),
                numerical_sham_vs_native=discrepancy(union_allowed, union_native, torch),
                cut_vs_same_input_all_allowed=discrepancy(union_selected, union_allowed, torch),
                returned_vs_native=discrepancy(union_selected, union_native, torch),
                future200_native_bytes_preserved=True, UND_dispatch_return_unmodified=True,
                actual_Q_K_V_bytes_and_global_CPU_CUDA_RNG_unchanged=True)
            self.counts['substituted_sites'] += 1
        self.events.append(event)
        self.pending = merged.squeeze(0).flatten(-2, -1)
        return merged

    def before_W(self, layer, module, args):
        assert self.active == layer and self.site_calls == ['UND', 'GEN'] and len(args) == 1 and self.pending is not None
        assert args[0].shape == (266, 4096) and byte_exact(args[0], self.pending, self.torch)
        assert self.events[-1]['to_add_out_actual_calls'] == 0
        self.events[-1]['to_add_out_actual_calls'] = 1
        self.events[-1]['actual_dispatch_return_flatten_equals_pre_W_byte_exact'] = True
        self.pending = None
        self.counts['pre_W'] += 1

    def after_block(self, layer, module, args, output):
        assert self.active == layer and self.pending is None and self.site_calls == ['UND', 'GEN']
        assert type(output) is tuple and len(output) == 2
        self.boundaries.append(self.selected_hidden(output[1]))
        self.active, self.next_layer = None, layer + 1
        self.counts['block_after'] += 1

    def after_model(self, module, args, output):
        assert self.in_model and self.active is None and self.next_layer == 36
        assert len(self.events) == 36 and len(self.boundaries) == 37 and type(output) is tuple and len(output) == 3
        assert type(output[2]) is list and len(output[2]) == 1 and output[2][0].shape == (16, 64)
        current = self.torch.stack([item['current'] for item in self.boundaries])
        action = self.torch.stack([item['action'] for item in self.boundaries])
        actual_output = self.runtime.cpu(output)
        data = dict(step=self.step, current_hidden=current, action_hidden=action,
            model_output=actual_output, actual_pack=self.calls[-1], site_events=self.events,
            boundary_convention='0=actual decoder input; 1..36=actual layer output after both attention and MLP residual additions; boundary36 precedes final norm.',
            indexes_and_masks_file='indexes-and-masks.pt')
        path = self.folder / f'boundaries-t{self.step:02d}.pt'
        self.torch.save(data, path)
        self.files.append(dict(step=self.step, file=path.name, sha256=sha(path), current_shape=list(current.shape), action_shape=list(action.shape)))
        self.outputs.append(actual_output[2][0])
        self.counts['model_after'] += 1
        self.in_model = False
        self.boundaries, self.events = [], []

    def begin(self):
        assert not self.installed and not self.handles and self.module.dispatch_attention_fn is self.original
        assert len(self.model.layers) == 36
        self.module.dispatch_attention_fn = self.wrapper
        self.installed = True
        try:
            self.handles.append(self.model.register_forward_pre_hook(self.before_model, with_kwargs=True))
            self.handles.append(self.model.register_forward_hook(self.after_model))
            for layer, block in enumerate(self.model.layers):
                self.handles.append(block.register_forward_pre_hook(lambda module, args, layer=layer: self.before_block(layer, module, args)))
                self.handles.append(block.register_forward_hook(lambda module, args, output, layer=layer: self.after_block(layer, module, args, output)))
                self.handles.append(block.self_attn.to_add_out.register_forward_pre_hook(lambda module, args, layer=layer: self.before_W(layer, module, args)))
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
            assert unchanged, 'Another operation changed the owned dispatch wrapper'

    def report(self, record):
        assert self.restored and not self.handles and not self.in_model and self.active is None
        assert self.counts['model_before'] == self.counts['model_after'] == 30
        for key in ('block_before', 'block_after', 'pre_W', 'native_UND_dispatch', 'native_GEN_dispatch'):
            assert self.counts[key] == 1080
        assert self.counts['all_allowed_dispatch'] == (0 if self.arm == 'native' else 1080)
        assert self.counts['cut_dispatch'] == (1080 if self.arm.startswith('cut_') else 0)
        assert self.counts['substituted_sites'] == (0 if self.arm == 'native' else 1080)
        assert len(self.calls) == len(self.files) == len(self.outputs) == 30
        self.runtime.require_exact(self.torch.stack(self.outputs), record['action_velocity'], 'full model tuple action output equals actual action head')
        assert self.reader.settings(self.torch, self.model.layers[0].self_attn.processor) == self.settings
        return dict(state='complete', arm=self.arm, counts=self.counts, boundary_files=self.files,
            indexes_and_masks_sha256=sha(self.folder / 'indexes-and-masks.pt'),
            all_actual_dispatch_return_pre_W_byte_exact=True, original_full_266_row_GEMM_calls=1080,
            actual_unblocked_GEN_vs_same_input_all_allowed_exact=True if self.arm.startswith('cut_') else None,
            full_future200_dispatch_rows_preserved_native=True, UND_dispatch_unmodified=True,
            same_union66_substitution_every_masked_site=self.arm != 'native',
            original_dispatch_symbol_restored=True, all_owned_hooks_removed=True, processor_recompute_calls=0,
            numerical_sham_is_native_noop=False, scope=SCOPE)


def compare_case_boundaries(left, right, runtime, first_only=None):
    """Stream the saved actual boundaries; retain only small error summaries."""
    torch = runtime.torch
    rows = []
    for step in ([0] if first_only else range(30)):
        a = torch.load(left / f'boundaries-t{step:02d}.pt', map_location='cpu', weights_only=True, mmap=True)
        b = torch.load(right / f'boundaries-t{step:02d}.pt', map_location='cpu', weights_only=True, mmap=True)
        names = (first_only,) if first_only else ('current_hidden', 'action_hidden')
        for name in names:
            av, bv = (data[name][1:2] if first_only else data[name] for data in (a, b))
            rows.append(dict(step=step, field=name, **discrepancy(av, bv, torch)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Completed baseline experiment root')
    baseline = parser.parse_args().output.resolve()
    factor, contract, threshold_sources, image, paths, anchors = preflight(baseline)
    out = baseline / 'future-read-cuts'
    out.mkdir(exist_ok=False)
    protocol = dict(source_pair=['V195_A195', 'V198_A195'], action_noise_source_seed=195,
        plan=[dict(case=name, vision_noise_source_seed=v, action_noise_source_seed=195, runtime_rng_seed=195, arm=arm) for name, v, arm in PLAN],
        image_path=str(image), image_sha256=sha(image), prompt=PROMPT, shift_cm=12,
        q0_predictions=10, model_forwards=300, steps=30, layers=36, physics_calls=0,
        original_dispatch_calls=21600, extra_all_allowed_dispatch_calls=8640, extra_cut_dispatch_calls=6480,
        total_official_dispatch_calls=36720, processor_recompute_calls=0,
        mask_shape=[1, 1, 266, 387], bool_true_means_allowed=True,
        query_groups=dict(current=50, action=16, future=200, text_UND=121),
        native_control='Two untouched original dispatch cases first; entire q0 record equals its completed fixed-A195 source.',
        masked_common_footprint='All eight masked cases substitute current50+action16 union66 rows; original native future200 and causal UND dispatch retained. Original full266 to_add_out shape/call unchanged.',
        cut_gate='Every unblocked GEN query, including future200, must byte-equal all-allowed computed on the same actual site Q/K/V. Any failure stops with actual output witness; no residual compensation.',
        noise_pair_rule='Within each V family all five entire V/A noise pairs exact. Across V families action draw only exact; invocation seed195 is not a whole-noise matching rule.',
        boundary_saved_steps=list(range(30)), public_selected_steps=[0, 10, 20, 29],
        boundary_scope='37 actual current/action decoder boundaries per step; boundary36 is before final norm. No full QKV exports.',
        engineering_predictions=dict(both='Cross-V all current/action boundaries, full64 action velocities/readouts, all31 FP32 action states and final10dim actions byte-equal; future payload may differ.',
            single_first_layer='At step0, current-cut boundary1 current rows and action-cut boundary1 action rows cross-V byte-equal; subsequent single-edge routes can differ.'),
        later_physics_preregistered=dict(trial_order=[name for name, _, _ in PLAN], cases=10, cached_q0=10, later_noise_base=195,
            q1_to_q7_seeds=list(range(196, 203)), later_predictions=70, later_model_forwards=2100, total_two_stage_model_forwards=2400,
            native_references={str(v): anchors['physical_references'][str(v)]['cell'] for v in (195, 198)},
            no_selection_after_q0_scores=True, both_prediction='If q0 ten-dimension actions equal, Both trials entire129 sim states, five action/trajectory JSON, eight inputPNG and q1..7 entire modelrecords must byte-equal.',
            numerical_sham_interpretation_gate='If either all-allowed physical sham changes its native grasp classification, stop original-native-necessity interpretation; only a modified mask-kernel regime effect can be described. Never change source pair or loosen controls.'),
        no_training=True, no_residual_compensation=True, scope=SCOPE)
    write(out / 'protocol.json', protocol, exclusive=True)
    write(out / 'sources.json', dict(anchors=anchors, threshold_sources=threshold_sources, frozen_contract=contract), exclusive=True)
    started, completed, rows, observer, noise, runtime = time.perf_counter(), [], [], None, None, None
    observed_counts = dict(model_before=0, model_after=0, native_UND_dispatch=0, native_GEN_dispatch=0, all_allowed_dispatch=0, cut_dispatch=0)

    def progress(stage, **values):
        info = dict(stage=stage, completed=completed, expected_model_forwards=300, actual_counts=observed_counts,
                    elapsed_s=time.perf_counter() - started, **values)
        write(out / 'progress.json', info)
        print('[FUTURE-READ-CUTS] ' + json.dumps(info), flush=True)

    try:
        os.environ['HF_HUB_OFFLINE'] = '1'
        base = load('future_cuts_normal_runtime', paths['normal_runtime'])
        base.TASKS['milk'] = (7, 'pick_up_the_milk_and_place_it_in_the_basket', PROMPT, 'milk_1')
        factory = load('future_cuts_factory', paths['factory'])
        factory.OUT = out
        assert sha(factory.SCHEDULER) == contract['frozen_files']['scheduler']['sha256']
        progress('loading_model')
        runtime = base.NormalRuntime(factory)
        torch, model = runtime.torch, runtime.pipe.transformer
        assert not model.training and not model.is_cache_enabled and not torch.is_grad_enabled()
        assert sha(Path(inspect.getfile(type(model)))) == anchors['upstream']['actual_transformer_sha256']
        assert sha(Path(inspect.getfile(type(runtime.pipe)))) == anchors['upstream']['actual_pipeline_source_sha256']
        components, scan, reader = (load('future_cuts_' + key, paths[key]) for key in ('components', 'metrics', 'reader'))
        checker = components.CosmosComponentInterventions(model)
        processor_module = inspect.getmodule(type(model.layers[0].self_attn.processor))
        dispatch = processor_module.dispatch_attention_fn
        dispatch_path = Path(inspect.getfile(dispatch))
        assert sha(dispatch_path) == DISPATCH_SHA
        feasibility = anchors['feasibility_actual_registry']
        assert hashlib.sha256(inspect.getsource(dispatch).encode()).hexdigest() == read(baseline / 'target-reader-inputs/provenance.json')['dispatch_function_source_sha256']
        dispatch_module = importlib.import_module('diffusers.models.attention_dispatch')
        assert dispatch_module.dispatch_attention_fn is dispatch
        registry = dispatch_module._AttentionBackendRegistry
        backend_name, backend_fn = registry.get_active_backend()
        supported = registry._supported_arg_names[backend_name]
        assert 'attn_mask' in supported and backend_name.value != '_native_flash'
        assert backend_name.value == feasibility['registry_backend']
        assert backend_fn.__qualname__ == feasibility['registry_backend_function']
        assert hashlib.sha256(inspect.getsource(backend_fn).encode()).hexdigest() == feasibility['registry_backend_function_source_sha256']
        assert sorted(supported) == feasibility['registry_supported_arguments']
        backend_settings = reader.settings(torch, model.layers[0].self_attn.processor)
        for key, value in feasibility['settings'].items():
            assert backend_settings[key] == value, ('Same recorded arithmetic setting as feasibility', key)
        for block in model.layers:
            assert block.self_attn.processor._parallel_config is None and block.self_attn.processor._attention_backend is None
            assert reader.settings(torch, block.self_attn.processor) == backend_settings
        original_records, original_caches, source_records = {}, {}, {}
        for seed in (195, 198):
            folder = Path(threshold_sources[str(seed)]['q0_folder'])
            original_records[seed] = torch.load(folder / 'states.pt', map_location='cpu', weights_only=True, mmap=True)
            original_caches[seed] = torch.load(folder / 'components.pt', map_location='cpu', weights_only=True, mmap=True)
            scan.validate_record(original_records[seed], torch)
            assert torch.count_nonzero(original_records[seed]['action_states'][..., 10:]) == 0
            scan.validate_cache(original_caches[seed], original_records[seed], checker, runtime, 'source' + str(seed))
            paired = Path(anchors['source_pair'][str(seed)]['folder'])
            source_records[seed] = torch.load(paired / 'states.pt', map_location='cpu', weights_only=True, mmap=True)
            scan.validate_record(source_records[seed], torch)
            runtime.require_exact(source_records[seed]['pure_noise'], [original_records[seed]['pure_noise'][0], original_records[195]['pure_noise'][1]], 'completed fixed-A195 source noise')
            assert read(paired / 'normalized_actions.json') == source_records[seed]['actions'].tolist()
        runtime.require_exact(original_caches[195]['steps'], original_caches[198]['steps'], 'two source schedule/pack metadata')
        clean_current = original_records[195]['model_input']['vision_tokens'][0][0, :, 0]
        runtime.require_exact(clean_current, original_records[198]['model_input']['vision_tokens'][0][0, :, 0], 'two source conditioned current image')
        native_random = runtime.cm.randn_tensor
        provenance = dict(script_sha256=sha(Path(__file__)), protocol_sha256=sha(out / 'protocol.json'), sources_sha256=sha(out / 'sources.json'),
            source_sha256={key: sha(path) for key, path in paths.items()}, noise_seam_source_sha256=FACTOR_SHA,
            actual_transformer_source=str(Path(inspect.getfile(type(model)))), actual_transformer_sha256=sha(Path(inspect.getfile(type(model)))),
            actual_pipeline_source_sha256=sha(Path(inspect.getfile(type(runtime.pipe)))), scheduler_sha256=sha(factory.SCHEDULER),
            dispatch_source=str(dispatch_path), dispatch_source_sha256=sha(dispatch_path), dispatch_signature=str(inspect.signature(dispatch)),
            registry_backend=backend_name.value, registry_backend_function=backend_fn.__qualname__, registry_supported_arguments=sorted(supported),
            registry_backend_function_source_sha256=hashlib.sha256(inspect.getsource(backend_fn).encode()).hexdigest(),
            backend_settings=backend_settings, kernel_evidence_limit='Actual registry and enabled flags recorded; no kernel profiler observation.',
            boundary_observation='Actual decoder pre/post module values, not reconstructed residuals or model recomputations.',
            library_edits=False, training=False, physics_calls=0, scope=SCOPE)
        write(out / 'provenance.json', provenance, exclusive=True)
        records = {}
        for label, v, arm in PLAN:
            if arm != 'native':
                assert completed[:2] == [item[0] for item in PLAN[:2]] and all(row['entire_native_q0_record_byte_exact'] is True for row in rows[:2])
            directory = out / label
            progress('fresh_q0', case=label)
            observer = FutureReadCuts(runtime, components, scan, reader, original_caches[195], clean_current, source_records[v], arm, directory)
            noise = factor.InitialNoiseSources(runtime, scan, original_records, v, 195)
            assert noise.original is native_random and observer.original is dispatch
            observer.begin()
            try:
                noise.begin()
                record, metadata = runtime.predict('milk', image, 195, directory)
            finally:
                try:
                    noise.reset()
                finally:
                    observer.reset()
                    for key in observed_counts:
                        observed_counts[key] += observer.counts[key]
                    if directory.exists():
                        torch.save(dict(calls=noise.calls, model_steps=observer.calls, initial_model_kwargs=observer.first,
                            original_symbol_restored=noise.restored, observer_hooks_removed=not observer.handles,
                            dispatch_symbol_restored=observer.restored, actual_counts=observer.counts), directory / 'noise-audit.pt')
                        if observer.in_model:
                            torch.save(dict(step=observer.step, current_partial_boundaries=observer.boundaries,
                                current_site_events=observer.events, actual_counts=observer.counts), directory / 'partial-forward.pt')
            assert runtime.cm.randn_tensor is native_random and processor_module.dispatch_attention_fn is dispatch
            scan.validate_record(record, torch)
            assert metadata['model_calls'] == 30 and metadata['prepare_calls'] == 1 and metadata['seed'] == 195
            assert metadata['input_png_sha256'] == sha(image) and metadata['prompt'] == PROMPT and metadata['task_index'] == 7
            if arm == 'native':
                runtime.require_exact(record, source_records[v], label + '/ENTIRE completed fixed-A195 source q0')
            report = observer.report(record)
            prepared = factor.prepared_contract(record, original_records, v, 195, runtime,
                SimpleNamespace(first=observer.first, calls=observer.calls, handle=None))
            noise_report = noise.report(record)
            write(directory / 'observer-report.json', report, exclusive=True)
            write(directory / 'noise-audit.json', dict(noise=noise_report, preparation=prepared), exclusive=True)
            metadata.update(case=label, arm=arm, query=0, shift_cm=12, vision_noise_source_seed=v, action_noise_source_seed=195,
                source_cell=f'V{v}_A195', numerical_masked_sham=arm == 'all_allowed', native_noop=arm == 'native',
                actual_masked_union66_site_substitutions=observer.counts['substituted_sites'],
                full_GEN_output_projection_shape=[266, 4096], original_full_GEN_projection_calls=1080,
                entire_native_q0_record_byte_exact=True if arm == 'native' else None)
            write(directory / 'metadata.json', metadata)
            files = (*SOURCE_FILES, 'observer-report.json', 'indexes-and-masks.pt')
            rows.append(dict(case=label, arm=arm, vision_noise_source_seed=v, action_noise_source_seed=195, runtime_rng_seed=195,
                fresh_model_forwards=30, entire_native_q0_record_byte_exact=True if arm == 'native' else None,
                all_noise_preparation_schedule_padding_and_dispatch_footprint_gates_passed=True,
                actual_dispatch_counts=observer.counts, final_normalized_actions=record['actions'].tolist(),
                files_sha256={name: sha(directory / name) for name in files}, boundary_files=observer.files))
            records[label] = record
            completed.append(label)
            progress('case_complete', case=label)
            observer = noise = None
        for v in (195, 198):
            for arm in ARMS:
                runtime.require_exact(records[f'V{v}_A195_{arm}']['pure_noise'], source_records[v]['pure_noise'], 'same-V entire returned noise pair/' + arm)
        runtime.require_exact(records['V195_A195_native']['pure_noise'][1], records['V198_A195_native']['pure_noise'][1], 'cross-V action draw only')
        closure = dict(state='checked', both_record_fields={}, both_boundary_checks=[], single_first_layer_checks={},
            future_vision_payload_equality_required=False, physics_executed=False)
        left, right = (records[f'V{v}_A195_cut_future_both'] for v in (195, 198))
        for key in ('readout', 'action_velocity', 'action_states', 'actions'):
            closure['both_record_fields'][key] = discrepancy(left[key], right[key], torch)
        closure['both_boundary_checks'] = compare_case_boundaries(out / 'V195_A195_cut_future_both', out / 'V198_A195_cut_future_both', runtime)
        for arm, field in (('cut_future_current', 'current_hidden'), ('cut_future_action', 'action_hidden')):
            closure['single_first_layer_checks'][arm] = compare_case_boundaries(out / f'V195_A195_{arm}', out / f'V198_A195_{arm}', runtime, field)
        write(out / 'closure.json', closure, exclusive=True)
        assert all(item['byte_exact'] for item in closure['both_record_fields'].values()), 'Both-cut complete action record engineering equality failed'
        assert all(item['byte_exact'] for item in closure['both_boundary_checks']), 'Both-cut all actual current/action decoder boundaries equality failed'
        assert all(item['byte_exact'] for items in closure['single_first_layer_checks'].values() for item in items), 'Single-cut first-layer engineering equality failed'
        assert completed == [item[0] for item in PLAN]
        assert observed_counts == dict(model_before=300, model_after=300, native_UND_dispatch=10800, native_GEN_dispatch=10800,
            all_allowed_dispatch=8640, cut_dispatch=6480)
        assert not checker._handles and checker._kind is None and runtime.cm.randn_tensor is native_random and processor_module.dispatch_attention_fn is dispatch
        write(out / 'results.json', dict(state='complete', cases=rows, fresh_q0_predictions=10, fresh_model_forwards=300,
            official_dispatch_counts=observed_counts, total_official_dispatch_calls=36720, closure_sha256=sha(out / 'closure.json'),
            numerical_sham_native_equivalence_claimed=False, engineering_cross_V_both_equality_passed=True,
            preregistered_physics_cases=[item[0] for item in PLAN], physics_calls=0, scope=SCOPE), exclusive=True)
        write(out / 'complete.json', dict(state='complete', script_sha256=sha(Path(__file__)),
            **{name + '_sha256': sha(out / (name + '.json')) for name in ('results', 'provenance', 'protocol', 'sources', 'closure')},
            fresh_q0_predictions=10, fresh_model_forwards=300, total_official_dispatch_calls=36720,
            extra_official_dispatch_calls=15120, original_randn_calls_consumed=20, returned_source_draws_observed=20,
            two_original_native_entire_q0_records_byte_exact=True, all_masked_actual_site_unblocked_queries_exact_vs_same_input_sham=True,
            all_future200_and_UND_native_dispatch_rows_retained=True, both_engineering_action_and_boundary_equality_passed=True,
            both_physical_equality_is_unexecuted_preregistered_prediction=True,
            original_noise_and_dispatch_symbols_restored=True, all_owned_hooks_removed=True,
            physics_calls=0, later_query_predictions=0, scope=SCOPE), exclusive=True)
        progress('complete')
    except Exception as exc:
        write(out / 'failed.json', dict(state='failed', error=str(exc), traceback=traceback.format_exc(), completed=completed,
            actual_counts=observed_counts, active_case=None if observer is None else observer.folder.name,
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
