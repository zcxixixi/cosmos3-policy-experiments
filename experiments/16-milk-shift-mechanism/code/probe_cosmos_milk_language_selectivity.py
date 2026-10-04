"""Eighteen preregistered X12 language/selectivity q0 predictions.

Use --output BASE. Only absent BASE/language-selectivity is created.
Milk versus cheese is the primary one-token comparison; cream cheese is a
canonical-name secondary control. Reuse frozen runtime, noise and hooks.
540 model forwards, 52650 official attention dispatches, no physics.
Sparse raw capture is explicit; omitted arrays are null with live byte SHA.
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
HEAD_SHA = 'b261366b969a8f33980cbf4d450c381a18a64fa4b16be2e1dd95d6c7220ae898'
HEAD_COMPLETE_SHA = 'e8a6b61cbe0a4d5da291dea68f15f45059f195c79fb9ad721b7b01670ab81dc0'
STAGE = 'language-selectivity'
SCENE_PROMPT = 'pick up the milk and place it in the basket'
PROMPTS = dict(milk=SCENE_PROMPT, cheese='pick up the cheese and place it in the basket',
               cream_cheese='pick up the cream cheese and place it in the basket')
TARGET_IDS = dict(milk=[14074], cheese=[17163], cream_cheese=[12644, 17163])
PROMPT_LENGTHS = dict(milk=10, cheese=10, cream_cheese=11)
TARGETS = dict(milk='milk_1', cheese='cream_cheese_1', cream_cheese='cream_cheese_1')
GOALS = tuple(PROMPTS)
ARMS = ('native', 'all_allowed', 'full_hardmask')
PLAN = tuple((f'{goal}_V{v}_A195_{arm}', goal, v, arm)
             for arm in ARMS for goal in GOALS for v in (195, 198))
WINDOW = dict(steps=[0, 14], layers=[0, 8])
CURRENT_STEPS = (0, 14, 15, 29)
FILES = ('states.pt', 'normalized_actions.json', 'metadata.json', 'noise-audit.pt', 'noise-audit.json',
         'observer-report.json', 'indexes-and-masks.pt', 'control-checks.json', 'site-captures.json',
         'language-contract.pt')
DISPATCH_KEYS = ('native_UND_dispatch', 'native_GEN_dispatch', 'all_allowed_dispatch',
                 'clone_sham_dispatch', 'value_zero_dispatch', 'cut_dispatch')
SCOPE = ('Eighteen fixed X12 q0 cases, scene task7 unchanged, actual policy instructions '
         'milk/cheese/cream cheese, V195/V198 and A/runtime195. Primary milk/cheese '
         'requires equal actual packed lengths/positions and exactly one changed target ID. '
         'Canonical cream cheese may change packing and is secondary. Same-call hardmask '
         'changes both content and allocation. No identity decoder or universal root cause. '
         'All18 physical cases remain required before evaluating the preregistered hypothesis; '
         'q0 raw production does not evaluate it. Sparse arrays are never described as full capture.')
PRIMARY = dict(name='H_common_shift', primary_goals=['milk', 'cheese'], secondary_goal='cream_cheese',
    measurement='t0 actual action_hidden boundary36; delta=cut-all_allowed for each language',
    cosine_min=0.95, relative_delta_difference_max=0.10, numerical_background_multiplier=10,
    numerical_background='D=norm(sham_milk-sham_cheese); E=max(norm(native-sham) for the two languages); D>10E',
    zero_delta_norms_invalid=True, cosine_gate_redundant_not_independent_evidence=True,
    all18_physics_required=True, no_q0_selection=True)


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


def expected_counts(arm=None):
    if arm is None:
        rows = [expected_counts(item) for _, _, _, item in PLAN]
        return {key: sum(row[key] for row in rows) for key in rows[0]}
    assert arm in ARMS
    return dict(model_before=30, model_after=30, block_before=1080, block_after=1080, pre_W=1080,
        native_UND_dispatch=1080, native_GEN_dispatch=1080, all_allowed_dispatch=1080 if arm != 'native' else 0,
        substituted_sites=1080 if arm != 'native' else 0, clone_sham_dispatch=0, value_zero_dispatch=0,
        cut_dispatch=135 if arm == 'full_hardmask' else 0)


def control_reference_case(goal, v, arm):
    return f'V{v}_A195_{arm}' if goal == 'milk' else None


def source_contract(baseline):
    """Read the real accepted contracts before output creation or torch import."""
    path = ROOT / 'work/probe_cosmos_milk_future_head_groups.py'
    assert sha(path) == HEAD_SHA
    head = load('language_frozen_head_producer', path)
    context = head.source_contract(baseline)
    producer = baseline / head.STAGE
    assert not (producer / 'failed.json').exists() and sha(producer / 'complete.json') == HEAD_COMPLETE_SHA
    docs = {name: read(producer / (name + '.json')) for name in
            ('complete', 'results', 'protocol', 'provenance', 'sources', 'primary-prediction')}
    complete, results = docs['complete'], docs['results']
    assert complete['state'] == results['state'] == 'complete'
    assert complete['script_sha256'] == docs['provenance']['script_sha256'] == HEAD_SHA
    for name in ('results', 'protocol', 'provenance', 'sources', 'primary-prediction'):
        assert complete[name.replace('-', '_') + '_sha256'] == sha(producer / (name + '.json'))
    assert complete['fresh_q0_predictions'] == 18 and complete['fresh_model_forwards'] == 540
    assert complete['official_dispatch_counts'] == head.expected_counts()
    assert complete['total_official_dispatch_calls'] == 57090 and complete['actual_selected_site_captures'] == 2430
    for key in ('two_native_and_two_allallowed_entire_accepted_value_records_and_boundaries_exact',
                'two_full_hardmask_entire_accepted_value_hardmask_records_and_boundaries_exact',
                'all540_actual_complete_model_kwargs_saved', 'all2430_actual_BF16_projection_MLP_residual_chains_byte_exact',
                'all_original_symbols_restored', 'all_owned_hooks_removed'):
        assert complete[key] is True
    assert docs['protocol']['prompt'] == SCENE_PROMPT
    rows = {row['case']: row for row in results['cases']}
    references = {}
    for arm in ARMS:
        for v in (195, 198):
            label = f'V{v}_A195_{arm}'
            row, folder = rows[label], producer / label
            assert row['arm'] == arm and row['vision_noise_source_seed'] == v
            assert row['action_noise_source_seed'] == row['runtime_rng_seed'] == 195
            assert set(row['files_sha256']) == set(head.FILES)
            for name, expected in row['files_sha256'].items():
                assert sha(folder / name) == expected
            assert len(row['boundary_files']) == 30
            for step, item in enumerate(row['boundary_files']):
                assert item['step'] == step and item['file'] == f'boundaries-t{step:02d}.pt'
                assert item['current_shape'] == [37, 50, 4096] and item['action_shape'] == [37, 16, 4096]
                assert sha(folder / item['file']) == item['sha256']
            references[label] = dict(folder=str(folder), files_sha256=row['files_sha256'], boundary_files=row['boundary_files'])
    context['modules']['head'] = head
    context['milk_q0_references'] = references
    context['source_evidence']['accepted_head_stage'] = dict(producer=str(producer), source_sha256=HEAD_SHA,
        complete_sha256=HEAD_COMPLETE_SHA, documents_sha256={name: sha(producer / (name + '.json')) for name in docs},
        six_milk_q0_references=references)
    context['out'] = baseline / STAGE
    return context


def tensor_meta(value, runtime):
    """Hash actual numerical bytes; retain original stride/device separately."""
    if value is None:
        return None
    raw = value.detach().cpu().contiguous().reshape(-1).view(runtime.torch.uint8).numpy().tobytes()
    return dict(shape=list(value.shape), dtype=str(value.dtype), stride=list(value.stride()), device=str(value.device),
                bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def tree_meta(value, runtime):
    if isinstance(value, runtime.torch.Tensor):
        return dict(type='tensor', **tensor_meta(value, runtime))
    if isinstance(value, dict):
        return dict(type='dict', items={key: tree_meta(item, runtime) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return dict(type=type(value).__name__, items=[tree_meta(item, runtime) for item in value])
    assert value is None or type(value) in (bool, int, float, str), type(value)
    return dict(type=type(value).__name__, value=value)


def tree_sha(value, runtime):
    """Canonical tree of CPU numerical bytes; device/stride are not byte equality."""
    def numeric(item):
        if isinstance(item, runtime.torch.Tensor):
            meta = tensor_meta(item, runtime)
            return dict(type='tensor', shape=meta['shape'], dtype=meta['dtype'], sha256=meta['sha256'])
        if isinstance(item, dict):
            return dict(type='dict', items={key: numeric(sub) for key, sub in item.items()})
        if isinstance(item, (tuple, list)):
            return dict(type=type(item).__name__, items=[numeric(sub) for sub in item])
        assert item is None or type(item) in (bool, int, float, str)
        return dict(type=type(item).__name__, value=item)
    return hashlib.sha256(json.dumps(numeric(value), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def unique_span(sequence, subsequence, label):
    matches = [i for i in range(len(sequence) - len(subsequence) + 1) if sequence[i:i + len(subsequence)] == subsequence]
    assert len(matches) == 1, (label, matches)
    return matches[0]


def language_span(kwargs, goal, prompt_ids, runtime):
    ids = runtime.cpu(kwargs['input_ids']).flatten().tolist()
    start = unique_span(ids, prompt_ids, 'actual packed complete prompt/' + goal)
    target = unique_span(prompt_ids, TARGET_IDS[goal], 'actual prompt target/' + goal)
    positions = list(range(start + target, start + target + len(TARGET_IDS[goal])))
    assert [ids[i] for i in positions] == TARGET_IDS[goal]
    return dict(goal=goal, prompt=PROMPTS[goal], input_ids=runtime.cpu(kwargs['input_ids']),
        und_len=int(kwargs['und_len']), sequence_length=int(kwargs['sequence_length']),
        prompt_token_ids=list(prompt_ids), prompt_span=[start, start + len(prompt_ids)],
        target_token_ids=TARGET_IDS[goal], target_positions=positions,
        text_indexes=runtime.cpu(kwargs['text_indexes']), position_ids=runtime.cpu(kwargs['position_ids']),
        inference='Actual complete packed input_ids matched to the actual pipeline tokenizer; no assumed absolute target index.')


def dynamic_indexes(kwargs, model, components, runtime):
    torch, layout = runtime.torch, components._layout(kwargs)
    u, n = layout['und_len'], layout['gen_len']
    assert n == 266 and int(kwargs['sequence_length']) == u + 266
    vision, action = kwargs['vision_tokens'][0], kwargs['action_tokens'][0]
    assert vision.shape == (1, 48, 5, 10, 20) and action.shape == (16, 64)
    assert vision.dtype == action.dtype == torch.bfloat16
    assert model.config.latent_channel == 48 and model.config.latent_patch_size == 2
    assert tuple(kwargs['vision_token_shapes'][0]) == (5, 5, 10)
    assert kwargs['vision_noisy_frame_indexes'][0].flatten().tolist() == [1, 2, 3, 4]
    source = inspect.getsource(type(model)._patchify_and_pack_latents)
    assert 'cthpwq->thwpqc' in source and 'latent = latent.squeeze(0)' in source
    vision_keys = runtime.cpu(kwargs['vision_sequence_indexes']).flatten()
    action_keys, action_rows = layout['global_action_indexes'], layout['action_rows']
    assert torch.equal(vision_keys, torch.arange(u, u + 250))
    assert torch.equal(action_keys, torch.arange(u + 250, u + 266))
    assert torch.equal(action_keys, runtime.cpu(kwargs['action_sequence_indexes']).flatten())
    assert action_keys.numel() == action_rows.numel() == 16
    current, future = vision_keys[:50] - u, vision_keys[50:]
    union, future_rows = torch.cat((current, action_rows)), future - u
    assert union.unique().numel() == 66
    assert torch.equal(torch.cat((union, future_rows)).sort().values, torch.arange(n))
    return dict(und_len=u, gen_len=n, full_joint_length=u + n, current_rows=current, action_rows=action_rows,
        union_rows=union, future_rows=future_rows, future_keys=future, current_keys=vision_keys[:50],
        action_keys=action_keys, text_keys=torch.arange(u),
        packing_source_sha256=hashlib.sha256(source.encode()).hexdigest())


def dynamic_masks(index, torch):
    masks = {}
    for arm in ('all_allowed', 'cut_future_current'):
        mask = torch.ones((1, 1, 266, index['full_joint_length']), dtype=torch.bool)
        blocked = index['current_rows'] if arm == 'cut_future_current' else torch.empty(0, dtype=torch.long)
        if blocked.numel():
            mask[0, 0, blocked[:, None], index['future_keys'][None, :]] = False
        unblocked = torch.tensor([i for i in range(266) if i not in set(blocked.tolist())], dtype=torch.long)
        assert int((~mask).sum()) == blocked.numel() * 200
        for key in ('text_keys', 'current_keys', 'action_keys'):
            assert bool(mask[0, 0, :, index[key]].all())
        assert bool(mask[0, 0, unblocked].all()) and bool(mask.any(dim=-1).all())
        masks[arm] = dict(mask=mask, blocked_rows=blocked, unblocked_rows=unblocked)
    return masks


def observer_class(head, value, windows, frozen):
    Parent = head.observer_class(value, windows, frozen)

    class LanguageObserver(Parent):
        def __init__(self, *args, goal, prompt_ids, native_info, legacy_reference, **kwargs):
            super().__init__(*args, **kwargs)
            self.goal, self.prompt_ids, self.native_info = goal, prompt_ids, native_info
            self.legacy_reference, self.legacy_boundary = legacy_reference, None
            self.und_boundaries, self.language, self.legacy_steps = [], None, []
            self.live_current_metadata = []

        def tensor_meta(self, value):
            return tensor_meta(value, self.runtime)

        def before_model(self, module, args, kwargs):
            assert not args and not self.in_model and self.active is None
            self.step += 1
            assert self.step < 30 and kwargs['return_dict'] is False
            layout = self.components._layout(kwargs)
            actual = dict(layout=layout, metadata={key: self.runtime.cpu(kwargs.get(key)) for key in self.components._STRUCTURAL_KEYS})
            # Only these actual packing fields may change with the instruction.
            packing = {'input_ids', 'und_len', 'sequence_length', 'text_indexes', 'position_ids',
                       'vision_sequence_indexes', 'vision_mse_loss_indexes', 'action_sequence_indexes', 'action_mse_loss_indexes'}
            for key in self.components._STRUCTURAL_KEYS:
                if key not in packing:
                    self.runtime.require_exact(actual['metadata'][key], self.cache['steps'][self.step]['metadata'][key],
                                               'unchanged actual schedule/domain/geometry/' + key)
            if self.native_info is not None:
                self.runtime.require_exact(actual, self.native_info['actual_pack'][self.step], 'same-goal/V actual all30 pack')
            derived = dynamic_indexes(kwargs, self.model, self.components, self.runtime)
            span = language_span(kwargs, self.goal, self.prompt_ids, self.runtime)
            if self.index is None:
                self.index, self.language = derived, span
                self.masks = dynamic_masks(derived, self.torch)
                self.gpu_masks = {name: item['mask'].to(kwargs['vision_tokens'][0].device) for name, item in self.masks.items()}
                self.first = self.runtime.cpu(kwargs)
                if self.native_info is not None:
                    self.runtime.require_exact(self.first, self.native_info['first_model_kwargs'], 'same-goal/V complete first kwargs')
                self.torch.save(dict(indexes=self.index, masks=self.masks), self.folder / 'indexes-and-masks.pt')
            else:
                self.runtime.require_exact(derived, self.index, 'all-step actual dynamic indices')
                self.runtime.require_exact(span, self.language, 'all-step actual instruction tokens/positions')
            self.runtime.require_exact(self.runtime.cpu(kwargs['vision_tokens'][0][0, :, 0]), self.clean_current, 'actual X12 current clamp')
            assert self.torch.count_nonzero(kwargs['action_tokens'][0][:, 10:]) == 0
            self.actual_kwargs = self.runtime.cpu(kwargs)
            if self.legacy_reference is not None:
                item = self.legacy_reference['boundary_files'][self.step]
                self.legacy_boundary = self.torch.load(Path(self.legacy_reference['folder']) / item['file'],
                    map_location='cpu', weights_only=True, mmap=True)
                self.runtime.require_exact(self.actual_kwargs, self.legacy_boundary['actual_model_kwargs'], 'milk live all30 entire kwargs')
            self.calls.append(actual)
            self.counts['model_before'] += 1
            self.in_model, self.next_layer = True, 0
            self.boundaries, self.events, self.und_boundaries = [], [], []

        def before_block(self, layer, module, args):
            # The inherited hook registration is retained; only its fixed UND shape is dynamic here.
            assert self.in_model and self.active is None and layer == self.next_layer and len(args) == 3
            assert args[0].shape == (self.index['und_len'], 4096) and args[1].shape == (266, 4096)
            assert args[0].dtype == args[1].dtype == self.torch.bfloat16
            assert bool(self.torch.isfinite(args[0]).all()) and self.propagation is None
            if layer == 0:
                self.boundaries.append(self.selected_hidden(args[1]))
                if self.step == 0:
                    self.und_boundaries.append(self.runtime.cpu(args[0]))
            self.active, self.pending, self.site_calls = layer, None, []
            self.counts['block_before'] += 1
            if windows.cut_active(WINDOW, self.step, layer):
                current = self.index['current_rows'].to(args[1].device)
                self.propagation = dict(block_input_current50=self.runtime.cpu(args[1].index_select(0, current)))

        def dispatch(self, *args, **kwargs):
            torch, exact = self.torch, frozen.byte_exact
            assert self.in_model and self.active is not None and self.capture_pending is None
            assert len(args) == 3 and set(kwargs) == {'is_causal', 'enable_gqa', 'backend', 'parallel_config'}
            assert kwargs['enable_gqa'] is True and kwargs['backend'] is kwargs['parallel_config'] is None
            causal = kwargs['is_causal']
            assert type(causal) is bool and self.site_calls == ([] if causal else ['UND'])
            self.site_calls.append('UND' if causal else 'GEN')
            u = self.index['und_len']
            qlen, kvlen = (u, u) if causal else (266, self.index['full_joint_length'])
            assert args[0].shape == (1, qlen, 32, 128) and args[1].shape == args[2].shape == (1, kvlen, 8, 128)
            assert all(item.dtype == torch.bfloat16 and bool(torch.isfinite(item).all()) for item in args)
            witnesses = tuple(item.clone(memory_format=torch.preserve_format) for item in args)
            rng = dict(cpu=torch.get_rng_state().clone(), cuda=torch.cuda.get_rng_state(args[0].device).clone())
            self.counts['native_UND_dispatch' if causal else 'native_GEN_dispatch'] += 1
            native = self.original(*args, **kwargs)
            assert native.shape == (1, qlen, 32, 128) and native.dtype == torch.bfloat16 and bool(torch.isfinite(native).all())
            if causal:
                for item, witness in zip(args, witnesses):
                    assert exact(item, witness, torch)
                assert exact(rng['cpu'], torch.get_rng_state(), torch) and exact(rng['cuda'], torch.cuda.get_rng_state(args[0].device), torch)
                self.und_dispatch = dict(native_return_unmodified=True, actual_QKV_bytes_preserved=True, RNG_bytes_preserved=True,
                    QKV_metadata=[self.tensor_meta(item) for item in args], output_metadata=self.tensor_meta(native))
                return native
            capture = windows.cut_active(WINDOW, self.step, self.active)
            active = self.case_arm == 'full_hardmask' and capture
            current, action, future, union = (self.index[key].to(native.device) for key in
                                            ('current_rows', 'action_rows', 'future_rows', 'union_rows'))
            z0 = zk = None
            event = dict(step=self.step, layer_zero_based=self.active, arm=self.case_arm,
                selected_window_cut_active=active, window_zero_based=self.window, capture_window_active=capture,
                common_masked_union66_background=self.case_arm != 'native', original_GEN_dispatch_calls=1,
                original_UND_dispatch_calls=1, extra_all_allowed_calls=0, extra_cut_calls=0,
                extra_value_zero_calls=0, extra_clone_sham_calls=0, to_add_out_shape=[266, 4096],
                to_add_out_actual_calls=0, processor_recompute_calls=0, UND_dispatch=self.und_dispatch,
                selected_query_heads=list(range(32)) if active else [], retained_query_heads=[] if active else list(range(32)))
            if self.case_arm != 'native':
                self.counts['all_allowed_dispatch'] += 1
                z0 = self.original(*args, **dict(kwargs, attn_mask=self.gpu_masks['all_allowed']))
                assert z0.shape == native.shape and z0.dtype == native.dtype and bool(torch.isfinite(z0).all())
                merged = native.clone()
                merged[:, union] = z0[:, union]
                event['extra_all_allowed_calls'] = 1
                if active:
                    self.counts['cut_dispatch'] += 1
                    zk = self.original(*args, **dict(kwargs, attn_mask=self.gpu_masks['cut_future_current']))
                    assert zk.shape == native.shape and zk.dtype == native.dtype and bool(torch.isfinite(zk).all())
                    unblocked = self.masks['cut_future_current']['unblocked_rows'].to(native.device)
                    assert exact(zk[:, unblocked], z0[:, unblocked], torch)
                    merged[:, current] = zk[:, current]
                    event.update(extra_cut_calls=1, all_unblocked_GEN_rows_vs_same_input_all_allowed_byte_exact=True)
                assert exact(merged[:, action], z0[:, action], torch) and exact(merged[:, future], native[:, future], torch)
                self.counts['substituted_sites'] += 1
            else:
                merged = native
            assert merged.dtype == native.dtype and bool(torch.isfinite(merged).all())
            for item, witness in zip(args, witnesses):
                assert exact(item, witness, torch), 'Official calls mutated actual QKV'
            after = dict(cpu=torch.get_rng_state().clone(), cuda=torch.cuda.get_rng_state(args[0].device).clone())
            assert exact(rng['cpu'], after['cpu'], torch) and exact(rng['cuda'], after['cuda'], torch)
            event.update(substituted_query_rows=0 if self.case_arm == 'native' else 66, selected_current_rows=50,
                selected_action_rows=16, future200_native_bytes_preserved=True, UND_dispatch_return_unmodified=True,
                original_QKV_bytes_preserved=True, RNG_bytes_preserved=True)
            if capture:
                executed = dict(Znative=native, Z0=z0, Zv=None, Zk=zk, Zreal=merged, Zclone=None)
                live_endpoints = {name: None if item is None else self.tensor_meta(item[:, current]) for name, item in executed.items()}
                endpoints = {name: self.runtime.cpu(item[:, current]) if item is not None and (name == 'Zreal' or self.step == 0)
                             else None for name, item in executed.items()}
                event['capture_file'] = f'sites/t{self.step:02d}-L{self.active:02d}.pt'
                self.capture_pending = dict(step=self.step, layer_zero_based=self.active, case_arm=self.case_arm, goal=self.goal,
                    current_rows=self.index['current_rows'], future_keys=self.index['future_keys'],
                    selected_query_heads=event['selected_query_heads'], retained_query_heads=torch.tensor(event['retained_query_heads'], dtype=torch.long),
                    Qcurrent50=self.runtime.cpu(args[0][:, current]) if self.step == 0 else None,
                    Kall=self.runtime.cpu(args[1]) if self.step == 0 else None,
                    Vall=self.runtime.cpu(args[2]) if self.step == 0 else None,
                    endpoints=endpoints, missing_endpoints=[name for name, item in endpoints.items() if item is None],
                    executed_endpoints=[name for name, item in executed.items() if item is not None],
                    omitted_executed_endpoint_raw=[name for name, item in executed.items() if item is not None and endpoints[name] is None],
                    live_endpoint_metadata=live_endpoints,
                    live_QKV_metadata=[self.tensor_meta(item) for item in args],
                    live_Qcurrent50_metadata=self.tensor_meta(args[0][:, current]),
                    original_QKV_metadata=[self.tensor_meta(item) for item in args],
                    endpoint_metadata=live_endpoints, RNG_before=rng, RNG_after=after,
                    actual_QKV_bytes_unchanged=True, raw_scope='t0 full Qcurrent/K/V and executed endpoints; t1..14 Zreal only; five observed propagation tensors at all135 sites',
                    raw_QKV_saved=self.step == 0, FP32allocation=None, FP32arithmetic_sham=None, event=dict(event), scope=SCOPE)
            self.events.append(event)
            self.pending = merged.squeeze(0).flatten(-2, -1)
            return merged

        def after_block(self, layer, module, args, output):
            assert type(output) is tuple and len(output) == 2
            assert output[0].shape == (self.index['und_len'], 4096) and output[0].dtype == self.torch.bfloat16
            assert bool(self.torch.isfinite(output[0]).all())
            if self.step == 0:
                self.und_boundaries.append(self.runtime.cpu(output[0]))
            # Frozen implementation seals real propagation, checks both BF16 additions,
            # removes pending state and observes the next current/action boundary.
            super().after_block(layer, module, args, output)

        def after_model(self, module, args, output):
            assert self.in_model and self.active is None and self.next_layer == 36
            assert len(self.events) == 36 and len(self.boundaries) == 37 and type(output) is tuple and len(output) == 3
            assert type(output[2]) is list and len(output[2]) == 1 and output[2][0].shape == (16, 64)
            current = self.torch.stack([item['current'] for item in self.boundaries])
            action = self.torch.stack([item['action'] for item in self.boundaries])
            und = self.torch.stack(self.und_boundaries) if self.step == 0 else None
            assert und is None or und.shape == (37, self.index['und_len'], 4096)
            actual_output = self.runtime.cpu(output)
            if self.legacy_boundary is not None:
                for key, item in (('current_hidden', current), ('action_hidden', action), ('model_output', actual_output)):
                    self.runtime.require_exact(item, self.legacy_boundary[key], 'milk live entire boundary/output/' + key)
                self.legacy_steps.append(self.step)
            full = self.step in CURRENT_STEPS
            frame_meta = self.tensor_meta(self.actual_kwargs['vision_tokens'][0][0, :, 0])
            data = dict(step=self.step, goal=self.goal, case_arm=self.case_arm, selected_window=self.window,
                current_hidden=current if full else None, action_hidden=action, und_hidden=und,
                model_output=actual_output if full else None, actual_model_kwargs=self.actual_kwargs if full else None,
                actual_model_kwargs_sha256=tree_sha(self.actual_kwargs, self.runtime),
                actual_model_kwargs_metadata=tree_meta(self.actual_kwargs, self.runtime),
                model_output_sha256=tree_sha(actual_output, self.runtime), model_output_metadata=tree_meta(actual_output, self.runtime),
                actual_model_action_tokens=self.actual_kwargs['action_tokens'][0],
                actual_model_action_output=actual_output[2][0],
                actual_model_action_timesteps=self.actual_kwargs['action_timesteps'],
                actual_current_frame_sha256=frame_meta['sha256'], actual_current_frame_metadata=frame_meta,
                solver_sigma_from_same_frozen_schedule=self.source['sigmas'][self.step].clone(),
                actual_pack=self.calls[-1], site_events=self.events, current_hidden_metadata=self.tensor_meta(current),
                action_hidden_metadata=self.tensor_meta(action), und_hidden_metadata=self.tensor_meta(und),
                raw_scope=dict(action_hidden='all30 steps,37 boundaries', current_hidden='steps0/14/15/29 only',
                    und_hidden='t0 only,37 boundaries', full_kwargs_and_model_tuple='steps0/14/15/29 only; other steps live SHA, not offline raw equality'),
                boundary_convention='0=actual decoder input; 1..36=actual block outputs; boundary36 precedes final norm.',
                indexes_and_masks_file='indexes-and-masks.pt', milk_accepted_live_full_byte_control=self.legacy_boundary is not None)
            path = self.folder / f'boundaries-t{self.step:02d}.pt'
            self.torch.save(data, path)
            self.files.append(dict(step=self.step, file=path.name, sha256=sha(path),
                current_shape=list(current.shape) if full else None, action_shape=list(action.shape),
                und_shape=None if und is None else list(und.shape), full_kwargs_and_tuple_saved=full,
                actual_model_kwargs_sha256=data['actual_model_kwargs_sha256'], model_output_sha256=data['model_output_sha256']))
            self.live_current_metadata.append(data['current_hidden_metadata'])
            self.outputs.append(actual_output[2][0])
            self.counts['model_after'] += 1
            self.in_model = False
            self.boundaries, self.events, self.actual_kwargs, self.und_boundaries, self.legacy_boundary = [], [], None, [], None

        def report(self, record):
            assert self.restored and not self.handles and not self.in_model and self.active is None
            assert self.capture_pending is self.propagation is None and self.counts == expected_counts(self.case_arm)
            assert len(self.calls) == len(self.files) == len(self.outputs) == 30 and len(self.capture_files) == 135
            assert self.finite_boundary_checks == 1110
            assert self.propagation_counts == dict(projection=135, residual=135, MLP=135, completed=135)
            assert self.legacy_steps == (list(range(30)) if self.legacy_reference is not None else [])
            self.runtime.require_exact(self.torch.stack(self.outputs), record['action_velocity'], 'all30 actual tuple action outputs')
            assert self.reader.settings(self.torch, self.model.layers[0].self_attn.processor) == self.settings
            manifest = dict(state='complete', case_arm=self.case_arm, goal=self.goal, window=WINDOW, sites=135,
                files=self.capture_files, total_bytes=sum(item['bytes'] for item in self.capture_files),
                no_extra_dispatch_for_missing_endpoints=True, full_QKV_steps=[0], full_QKV_sites=9,
                real_pre_W_and_five_propagation_sites=135, live_QKV_and_endpoint_byte_SHA_sites=135,
                missing_raw_rule='Null distinguishes omitted raw; executed_endpoints/live metadata distinguish unexecuted endpoints.', scope=SCOPE)
            write(self.folder / 'site-captures.json', manifest, exclusive=True)
            return dict(state='complete', goal=self.goal, case_arm=self.case_arm, selected_window=self.window,
                counts=self.counts, boundary_files=self.files, indexes_and_masks_sha256=sha(self.folder / 'indexes-and-masks.pt'),
                site_capture_manifest_sha256=sha(self.folder / 'site-captures.json'), captured_sites=135,
                original_full_266_row_projection_calls=1080, all_actual_dispatch_flatten_pre_W_byte_exact=True,
                all_masked_sites_same_union66=self.case_arm != 'native', action16_same_input_Z0_at_all_masked_sites=True,
                future200_and_UND_native_dispatch_returns_preserved=True, all30_actual37_current_action_boundaries_finite=True,
                all135_actual_BF16_projection_MLP_residual_chains_byte_exact=True, propagation_counts=self.propagation_counts,
                actual_action_input_output_saved_all30=True, full_current_kwargs_tuple_saved_steps=list(CURRENT_STEPS),
                full_UND_saved_steps=[0], other_full_raw_arrays_omitted=True,
                milk_all30_live_full_kwargs_current_action_tuple_byte_exact=self.legacy_reference is not None,
                processor_recompute_calls=0, extra_projection_or_MLP_forward_calls=0,
                all_owned_hooks_removed=True, original_dispatch_symbol_restored=True, scope=SCOPE)
    return LanguageObserver


def preparation_checks(record, observer, originals, v, runtime):
    torch = runtime.torch
    prepared = record['prepared_latents_and_masks']
    assert type(prepared) is tuple and len(prepared) == 12
    assert prepared[0].shape == (1, 48, 5, 10, 20) and prepared[0].dtype == torch.float32
    assert prepared[2].shape == (16, 64) and prepared[2].dtype == torch.float32
    runtime.require_exact(prepared[0], originals[v]['prepared_latents_and_masks'][0], 'actual entire selected V preparation')
    runtime.require_exact(prepared[2], originals[195]['prepared_latents_and_masks'][2], 'actual entire A195 preparation')
    for i in range(12):
        if i not in (0, 2):
            runtime.require_exact(prepared[i], originals[195]['prepared_latents_and_masks'][i], 'other ten actual preparation fields')
    assert prepared[3] == 20 and prepared[8].tolist() == [5] and prepared[10] == 10
    runtime.require_exact(prepared[5].flatten(), torch.tensor([1., 0., 0., 0., 0.], dtype=prepared[5].dtype), 'vision condition mask')
    assert prepared[7].shape == (16, 1) and torch.count_nonzero(prepared[7]) == 0
    assert torch.count_nonzero(prepared[2][:, 10:]) == torch.count_nonzero(record['action_states'][..., 10:]) == 0
    runtime.require_exact(prepared[0][0, :, 0].to(torch.bfloat16), observer.clean_current, 'prepared actual current to BF16')
    runtime.require_exact(record['model_input'], observer.first, 'inner/outer complete first kwargs')
    runtime.require_exact(record['model_input']['vision_tokens'], originals[v]['model_input']['vision_tokens'], 'selected V entire initial vision tokens')
    runtime.require_exact(record['model_input']['action_tokens'], originals[195]['model_input']['action_tokens'], 'selected A entire initial action tokens')
    runtime.require_exact(record['action_states'][0], originals[195]['action_states'][0], 'actual A195 initial FP32 solver state')
    for key in ('timesteps', 'sigmas'):
        runtime.require_exact(record[key], originals[195][key], 'complete actual schedule/' + key)
    assert len(observer.calls) == 30
    for step, item in enumerate(observer.files):
        data = torch.load(observer.folder / item['file'], map_location='cpu', weights_only=True, mmap=True)
        runtime.require_exact(data['actual_model_action_tokens'], record['action_states'][step, 0].to(torch.bfloat16), 'all30 actual solver cast')
        runtime.require_exact(data['actual_model_action_output'], record['action_velocity'][step], 'all30 actual action output')
        runtime.require_exact(data['solver_sigma_from_same_frozen_schedule'], record['sigmas'][step], 'all30 actual sigma')
        assert data['actual_current_frame_sha256'] == tensor_meta(observer.clean_current, runtime)['sha256']
    return dict(actual_model_calls=30, prepared_vision_entire_tensor_equals_V_source=True,
        prepared_action_entire_tensor_equals_A_source=True, other_ten_prepared_fields_byte_exact=True,
        initial_vision_action_tokens_and_A195_FP32_solver_sample_byte_exact=True,
        all30_actual_action_solver_casts_outputs_schedule_and_current_SHA_checked=True,
        all_prepared_model_input_and_31_solver_state_padding54_zero=True, action_condition_mask_all_zero=True,
        complete_initial_kwargs_equal_same_goal_V_native=observer.native_info is not None,
        native_first_observation=observer.case_arm == 'native', no_independent_new_goal_native_repeat=True)


def primary_language_contract(native_infos, runtime):
    torch, rows = runtime.torch, []
    for v in (195, 198):
        milk, cheese = (native_infos[(goal, v)] for goal in ('milk', 'cheese'))
        left, right = milk['first_model_kwargs'], cheese['first_model_kwargs']
        assert left.keys() == right.keys()
        for key in left:
            if key != 'input_ids':
                runtime.require_exact(left[key], right[key], 'primary one-token complete initial kwargs/' + key)
        assert left['input_ids'].shape == right['input_ids'].shape
        changed = torch.nonzero(left['input_ids'].flatten() != right['input_ids'].flatten()).flatten().tolist()
        assert changed == milk['language']['target_positions'] == cheese['language']['target_positions'] and len(changed) == 1
        assert [int(left['input_ids'].flatten()[i]) for i in changed] == TARGET_IDS['milk']
        assert [int(right['input_ids'].flatten()[i]) for i in changed] == TARGET_IDS['cheese']
        for a, b in zip(milk['actual_pack'], cheese['actual_pack']):
            runtime.require_exact(a['layout'], b['layout'], 'primary all30 packed lengths/layout')
            for key in a['metadata']:
                if key != 'input_ids':
                    runtime.require_exact(a['metadata'][key], b['metadata'][key], 'primary all30 packed positions/' + key)
        rows.append(dict(vision_noise_source_seed=v, changed_actual_input_id_positions=changed,
            milk_target_ids=TARGET_IDS['milk'], cheese_target_ids=TARGET_IDS['cheese'],
            actual_und_len=int(left['und_len']), actual_sequence_length=int(left['sequence_length']),
            all30_lengths_text_indices_position_ids_and_other_pack_fields_byte_exact=True,
            complete_initial_kwargs_only_one_target_ID_changed=True))
    for goal in GOALS:
        runtime.require_exact(native_infos[(goal, 195)]['actual_pack'], native_infos[(goal, 198)]['actual_pack'], 'same-goal cross-V full structural pack')
    return dict(state='exact', primary_goals=['milk', 'cheese'], cases=rows,
        canonical_name_is_secondary_and_may_change_packing=True,
        canonical_actual_und_len=native_infos[('cream_cheese', 195)]['language']['und_len'], scope=SCOPE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check-source-contract', action='store_true')
    args = parser.parse_args()
    baseline = args.output.resolve()
    context = source_contract(baseline)
    if args.check_source_contract:
        print(json.dumps(dict(state='source_contract_verified', image=str(context['image']),
            milk_q0_references=context['milk_q0_references'], expected_counts=expected_counts(),
            total_official_dispatch_calls=52650), indent=2))
        return
    out, paths, contract = context['out'], context['paths'], context['frozen_contract']
    assert not out.exists(), 'New stage must be absent; failed output is not silently retried'
    out.mkdir(exist_ok=False)
    protocol = dict(plan=[dict(case=label, goal=goal, prompt=PROMPTS[goal], arm=arm,
        vision_noise_source_seed=v, action_noise_source_seed=195, runtime_rng_seed=195,
        window=WINDOW if arm == 'full_hardmask' else None,
        control_reference_case=control_reference_case(goal, v, arm)) for label, goal, v, arm in PLAN],
        trial_order=[label for label, _, _, _ in PLAN], prompts=PROMPTS, scene_prompt=SCENE_PROMPT,
        scene_task_index=7, shift_cm=12, image_path=str(context['image']), image_sha256=sha(context['image']),
        primary_hypothesis=PRIMARY, selected_window_zero_based_inclusive=WINDOW,
        target_token_ids_preregistered=TARGET_IDS, prompt_token_lengths_preregistered=PROMPT_LENGTHS,
        primary_language_gate='Actual full packed milk/cheese input IDs differ only at one target token; all other first kwargs and all30 structural metadata equal.',
        masked_background='All36 layers/all30 steps union66 Z0; only selected current50 uses same-live-QK Zk; action16 Z0, future200 and UND native.',
        q0_predictions=18, model_forwards=540, expected_counts=expected_counts(), total_official_dispatch_calls=52650,
        original_dispatch_calls=38880, extra_all_allowed_dispatch_calls=12960, selected_extra_dispatch_calls=810,
        captures_per_case=135, total_site_captures=2430, max_capture_file_bytes=100_000_000,
        action_boundary_raw_steps=list(range(30)), current_boundary_raw_steps=list(CURRENT_STEPS), UND_boundary_raw_steps=[0],
        full_kwargs_and_tuple_raw_steps=list(CURRENT_STEPS), all30_action_input_output_timesteps_sigma_raw=True,
        full_QKV_site_raw_steps=[0], pre_W_and_five_propagation_raw_sites_per_case=135,
        missing_raw_rule='Explicit null with live numerical-byte SHA; unexecuted endpoints separately named; no capture-only dispatch.',
        six_milk_controls='ENTIRE prior head-stage NormalRuntime records, plus live all30 full kwargs/current/action/tuple equality before sparse saving.',
        later_physics_preregistered=dict(trial_order=[label for label, _, _, _ in PLAN], cases=18, cached_q0=18,
            later_noise_base=195, q1_to_q7_seeds=list(range(196, 203)), later_predictions=126,
            later_model_forwards=3780, total_two_stage_model_forwards=4320, later_whole_noise_pairs=119,
            no_q0_score_or_rank_selection=True), physics_calls=0, no_training=True, no_residual_compensation=True, scope=SCOPE)
    write(out / 'protocol.json', protocol, exclusive=True)
    write(out / 'primary-prediction.json', dict(status='not_evaluated_q0_only', **PRIMARY, scope=SCOPE), exclusive=True)
    write(out / 'sources.json', dict(source_evidence=context['source_evidence'], original_sources=context['original_noise_sources'],
        frozen_contract=contract, milk_q0_references=context['milk_q0_references']), exclusive=True)
    started, completed, rows, runtime, observer, noise = time.perf_counter(), [], [], None, None, None
    counts = {key: 0 for key in expected_counts()}

    def progress(stage, **values):
        document = dict(stage=stage, completed=completed, completed_model_forwards=30 * len(completed),
            expected_model_forwards=540, actual_counts=counts, elapsed_s=time.perf_counter() - started, **values)
        write(out / 'progress.json', document)
        print('[LANGUAGE-SELECTIVITY] ' + json.dumps(document), flush=True)

    try:
        os.environ['HF_HUB_OFFLINE'] = '1'
        head, value, windows, frozen, factor = (context['modules'][key] for key in ('head', 'value', 'windows', 'frozen', 'factor'))
        base, factory = load('language_runtime', paths['normal_runtime']), load('language_factory', paths['factory'])
        for goal in GOALS:
            base.TASKS[goal] = (7, 'pick_up_the_milk_and_place_it_in_the_basket', PROMPTS[goal], TARGETS[goal])
        factory.OUT = out
        assert sha(factory.SCHEDULER) == contract['frozen_files']['scheduler']['sha256']
        progress('loading_model')
        runtime = base.NormalRuntime(factory)
        torch, model = runtime.torch, runtime.pipe.transformer
        components, scan, reader, processor_module, dispatch, backend = value.runtime_contract(baseline, context, runtime)
        checker = components.CosmosComponentInterventions(model)
        originals, caches, references = {}, {}, {}
        for v in (195, 198):
            folder = Path(context['original_noise_sources'][str(v)]['q0_folder'])
            originals[v] = torch.load(folder / 'states.pt', map_location='cpu', weights_only=True, mmap=True)
            caches[v] = torch.load(folder / 'components.pt', map_location='cpu', weights_only=True, mmap=True)
            scan.validate_record(originals[v], torch)
            scan.validate_cache(caches[v], originals[v], checker, runtime, 'accepted_noise_source/' + str(v))
        runtime.require_exact(caches[195]['steps'], caches[198]['steps'], 'accepted complete source schedule/pack')
        clean = originals[195]['model_input']['vision_tokens'][0][0, :, 0]
        runtime.require_exact(clean, originals[198]['model_input']['vision_tokens'][0][0, :, 0], 'accepted X12 conditioned frame')
        for label, reference in context['milk_q0_references'].items():
            references[label] = torch.load(Path(reference['folder']) / 'states.pt', map_location='cpu', weights_only=True, mmap=True)
            scan.validate_record(references[label], torch)
        tokenizer = runtime.pipe.text_tokenizer
        prompt_ids = {goal: tokenizer.encode(prompt, add_special_tokens=False) for goal, prompt in PROMPTS.items()}
        for goal in GOALS:
            assert len(prompt_ids[goal]) == PROMPT_LENGTHS[goal]
            unique_span(prompt_ids[goal], TARGET_IDS[goal], 'actual tokenizer target/' + goal)
        actual_sha = sha(Path(inspect.getfile(type(model))))
        output_head = dict(domain_id=5, norm_state=runtime.cpu(model.norm_moe_gen.state_dict()),
            weight=runtime.cpu(model.action_proj_out.fc.weight[5].reshape(4096, 64)),
            bias=runtime.cpu(model.action_proj_out.bias.weight[5]), source_transformer_sha256=actual_sha,
            capture='one actual read-only parameter extraction; no forward')
        assert output_head['weight'].shape == (4096, 64) and output_head['bias'].shape == (64,)
        assert all(bool(torch.isfinite(item).all()) for item in (*output_head['norm_state'].values(), output_head['weight'], output_head['bias']))
        torch.save(output_head, out / 'output-head-actual.pt')
        assert (out / 'output-head-actual.pt').stat().st_size < 1_048_576
        native_random = runtime.cm.randn_tensor
        provenance = dict(script_sha256=sha(Path(__file__)), protocol_sha256=sha(out / 'protocol.json'),
            sources_sha256=sha(out / 'sources.json'), source_sha256={key: sha(path) for key, path in paths.items()},
            accepted_head_source_sha256=HEAD_SHA, accepted_head_complete_sha256=HEAD_COMPLETE_SHA,
            scheduler_sha256=sha(factory.SCHEDULER), actual_transformer_sha256=actual_sha,
            actual_pipeline_source_sha256=sha(Path(inspect.getfile(type(runtime.pipe)))),
            dispatch_source_sha256=sha(Path(inspect.getfile(dispatch))), dispatch_signature=str(inspect.signature(dispatch)),
            tokenizer_class=type(tokenizer).__name__, actual_prompt_token_ids=prompt_ids,
            output_head_actual_sha256=sha(out / 'output-head-actual.pt'),
            output_head_metadata=dict(domain_id=5, norm_state=tree_meta(output_head['norm_state'], runtime),
                weight=tensor_meta(output_head['weight'], runtime), bias=tensor_meta(output_head['bias'], runtime),
                source_transformer_sha256=actual_sha, capture=output_head['capture'], training=False, patch=False),
            torch_version=torch.__version__, CPU_threads=torch.get_num_threads(), **backend,
            kernel_evidence_limit='Registry/settings recorded; no selected-kernel profiler observation.',
            library_edits=False, training=False, physics_calls=0, scope=SCOPE)
        write(out / 'provenance.json', provenance, exclusive=True)
        Observer = observer_class(head, value, windows, frozen)
        records, native_infos = {}, {}
        for label, goal, v, arm in PLAN:
            if arm != 'native':
                assert completed[:6] == [item[0] for item in PLAN[:6]]
            directory = out / label
            progress('fresh_q0', case=label)
            reference_label = control_reference_case(goal, v, arm)
            reference = context['milk_q0_references'].get(reference_label)
            observer = Observer(runtime, components, scan, reader, caches[195], clean, originals[195],
                'native' if arm == 'native' else 'all_allowed', directory, case_arm=arm,
                goal=goal, prompt_ids=prompt_ids[goal], native_info=native_infos.get((goal, v)), legacy_reference=reference)
            noise = factor.InitialNoiseSources(runtime, scan, originals, v, 195)
            assert observer.original is dispatch and noise.original is native_random
            observer.begin()
            try:
                noise.begin()
                record, metadata = runtime.predict(goal, context['image'], 195, directory)
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
                                pending_site_capture=observer.capture_pending, pending_propagation=observer.propagation,
                                actual_counts=observer.counts), directory / 'partial-forward.pt')
            assert runtime.cm.randn_tensor is native_random and processor_module.dispatch_attention_fn is dispatch
            scan.validate_record(record, torch)
            assert metadata['model_calls'] == 30 and metadata['prepare_calls'] == 1 and metadata['seed'] == 195
            assert metadata['input_png_sha256'] == sha(context['image']) and metadata['prompt'] == PROMPTS[goal] and metadata['task_index'] == 7
            report = observer.report(record)
            preparation = preparation_checks(record, observer, originals, v, runtime)
            noise_report = noise.report(record)
            if reference is not None:
                runtime.require_exact(record, references[reference_label], 'ENTIRE accepted milk head-stage NormalRuntime record')
            info = dict(goal=goal, case=label, native_reference_case=f'{goal}_V{v}_A195_native',
                actual_pack=observer.calls, first_model_kwargs=observer.first, actual_clean_current=observer.clean_current,
                original_solver_initial_action_FP32=record['action_states'][0], language=observer.language,
                actual_input_checks=preparation, scope=SCOPE)
            torch.save(info, directory / 'language-contract.pt')
            if arm == 'native':
                native_infos[(goal, v)] = info
            control = dict(state='exact', native_reference_case=info['native_reference_case'], control_reference_case=reference_label,
                same_goal_V_full_initial_kwargs_and_all30_structural_pack_exact=arm != 'native',
                actual_source_noise_preparation_schedule_clamp_and_padding_passed=True,
                entire_accepted_control_record_exact=reference is not None,
                all30_live_accepted_full_kwargs_current_action_and_tuple_exact=reference is not None,
                independent_new_goal_native_repeat_performed=False, sparse_raw_scope=report['scope'])
            write(directory / 'control-checks.json', control, exclusive=True)
            write(directory / 'observer-report.json', report, exclusive=True)
            write(directory / 'noise-audit.json', dict(noise=noise_report, preparation=preparation), exclusive=True)
            metadata.update(case=label, goal=goal, arm=arm, query=0, scene_task_index=7, scene_prompt=SCENE_PROMPT,
                policy_prompt=PROMPTS[goal], target_object=TARGETS[goal], shift_cm=12,
                vision_noise_source_seed=v, action_noise_source_seed=195, runtime_rng_seed=195,
                window_zero_based_inclusive=observer.window, common_union66_all_allowed_background=arm != 'native',
                producer_script_sha256=provenance['script_sha256'], control_reference_case=reference_label,
                entire_accepted_control_record_exact=reference is not None, captured_sites=135,
                site_capture_manifest_sha256=report['site_capture_manifest_sha256'],
                actual_target_positions=observer.language['target_positions'], actual_target_token_ids=TARGET_IDS[goal],
                raw_current_steps=list(CURRENT_STEPS), raw_UND_steps=[0], full_QKV_site_steps=[0],
                physical_prediction_evaluated=False, scope=SCOPE)
            write(directory / 'metadata.json', metadata)
            assert read(directory / 'normalized_actions.json') == record['actions'].tolist()
            rows.append(dict(case=label, goal=goal, prompt=PROMPTS[goal], arm=arm, window=observer.window,
                vision_noise_source_seed=v, action_noise_source_seed=195, runtime_rng_seed=195,
                fresh_model_forwards=30, actual_dispatch_counts=observer.counts, control_reference_case=reference_label,
                existing_entire_control_record_exact=reference is not None, actual_und_len=observer.index['und_len'],
                actual_target_positions=observer.language['target_positions'], actual_target_token_ids=TARGET_IDS[goal],
                final_normalized_actions=record['actions'].tolist(), physical_prediction_evaluated=False,
                files_sha256={name: sha(directory / name) for name in FILES}, boundary_files=observer.files,
                site_capture_manifest_sha256=report['site_capture_manifest_sha256']))
            records[label] = record
            completed.append(label)
            if len(completed) == 6:
                write(out / 'primary-language-contract.json', primary_language_contract(native_infos, runtime), exclusive=True)
            progress('case_complete', case=label)
            observer = noise = None
        for label, goal, v, arm in PLAN:
            runtime.require_exact(records[label]['pure_noise'], records[f'milk_V{v}_A195_native']['pure_noise'], 'same-V all goals/arms whole returned noise')
        runtime.require_exact(records['milk_V195_A195_native']['pure_noise'][1], records['milk_V198_A195_native']['pure_noise'][1], 'cross-V A195')
        assert counts == expected_counts() and sum(counts[key] for key in DISPATCH_KEYS) == 52650
        assert completed == [item[0] for item in PLAN] and not checker._handles and checker._kind is None
        assert runtime.cm.randn_tensor is native_random and processor_module.dispatch_attention_fn is dispatch
        write(out / 'results.json', dict(state='complete', cases=rows, fresh_q0_predictions=18, fresh_model_forwards=540,
            official_dispatch_counts=counts, total_official_dispatch_calls=52650, physics_calls=0,
            primary_prediction_sha256=sha(out / 'primary-prediction.json'), primary_physical_prediction_evaluated=False,
            preregistered_physics_cases=[item[0] for item in PLAN], scope=SCOPE), exclusive=True)
        write(out / 'complete.json', dict(state='complete', script_sha256=sha(Path(__file__)),
            **{name.replace('-', '_') + '_sha256': sha(out / (name + '.json')) for name in
               ('results', 'provenance', 'protocol', 'sources', 'primary-prediction', 'primary-language-contract')},
            output_head_actual_sha256=sha(out / 'output-head-actual.pt'),
            fresh_q0_predictions=18, fresh_model_forwards=540, official_dispatch_counts=counts,
            total_official_dispatch_calls=52650, original_dispatch_calls=38880, extra_dispatch_calls=13770,
            original_randn_calls_consumed=36, returned_source_draws_observed=36, actual_selected_site_captures=2430,
            six_milk_entire_accepted_head_records_and_live_all30_full_boundary_kwargs_tuple_exact=True,
            actual_primary_packed_single_target_token_gate_passed=True,
            all12_masked_sites_same_union66_background=True, all540_actual_action_input_output_raw_saved=True,
            full_current_boundary_and_kwargs_tuple_raw_steps=list(CURRENT_STEPS), full_UND_boundary_raw_steps=[0],
            all2430_actual_pre_W_and_BF16_projection_MLP_residual_chains_byte_exact=True,
            full_QKV_raw_sites=162, compact_QKV_and_endpoint_live_SHA_sites=2268,
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
