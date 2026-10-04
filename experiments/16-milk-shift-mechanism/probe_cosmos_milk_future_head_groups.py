"""Eighteen preregistered X12 q0 runs on actual per-layer pre-W head groups.

--output BASE creates only absent BASE/future-head-groups. Reuse frozen 0610
source/noise/backend/packing contracts. Each nonnative site uses union66 Z0.
At t0..14/L0 only, G4=16..19 and G6=24..27 use same-call Zk or random delta.
Full_hardmask retains t0..14/L0..8; leave retains that full cut except the
four named L0 heads stay Z0. Every arm captures all135 full-window sites.
All 18 cases are frozen before execution; no q0 selection or candidate swap.
540 model forwards and 57090 official dispatches; no physics or training.
Actual projection/residual/MLP propagation is observed without recomputation.
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
from types import SimpleNamespace


ROOT = Path('/home/current/work/cosmos3')
VALUE_SHA = '0610b1d0320512ff4f5ce4163b52337ccb3ddf3ed1eb8d6c13f51b614097b911'
VALUE_COMPLETE_SHA = '4b274d367595d8da910c50097b0a744f21b38c78b470a869b6238e4d6764d5f3'
GEOMETRY_SHA = '6ba8f644198e7df9f8741d911fd994f6d647333ece8c8ddfb9bbbf412451a4d9'
GEOMETRY_COMPLETE_SHA = '43af90fab87374da2527266ba0c6868792fb0f9cd404ab71c9fccba66e40179a'
GEOMETRY_ANALYSIS_SHA = 'ad23bf5adbb7cc5c5f57208f8b2c643cf6eed00a64800c064fb55f1b972c3093'
STAGE = 'future-head-groups'
PROMPT = 'pick up the milk and place it in the basket'
WINDOW = dict(steps=[0, 14], layers=[0, 8])
FULL_WINDOW = WINDOW
LOCAL_WINDOW = dict(steps=[0, 14], layers=[0, 0])
GROUPS = dict(G4=list(range(16, 20)), G6=list(range(24, 28)))
ARMS = ('native', 'all_allowed', 'full_hardmask', 'head_G4', 'head_G6',
        'leave_G4', 'leave_G6', 'arithmetic_sham_G6', 'random_G6')
PLAN = tuple((f'V{v}_A195_{arm}', v, arm) for arm in ARMS for v in (195, 198))
FILES = ('states.pt', 'normalized_actions.json', 'metadata.json', 'noise-audit.pt', 'noise-audit.json',
         'observer-report.json', 'indexes-and-masks.pt', 'control-checks.json', 'site-captures.json')
DISPATCH_KEYS = ('native_UND_dispatch', 'native_GEN_dispatch', 'all_allowed_dispatch',
                 'clone_sham_dispatch', 'value_zero_dispatch', 'cut_dispatch')
RANDOM_RELATIVE_NORM_TOLERANCE = 0.10
SCOPE = ('X12 fixed image/instruction, V195/V198 with A195, q0 only. G6 names only '
         'layer0 query heads24..27, four parameter head positions repeated over15 '
         'solver steps, 60 head-sites; G4 similarly means layer0 heads16..19. The '
         'complete270-site pre-intervention geometry narrowed the earlier numbered '
         'group candidates to L0 before new behavioral outcomes. Shared head numbers '
         'in other layers do not imply shared function. All135 sites are captured '
         'and all18 cases frozen without posthoc candidate replacement. '
         'Actual pre-W32x128 head axes are patched, never residual4096 slices. Same '
         'actual Q/K is within one call only; later trajectories can diverge. Head '
         'replacement jointly changes future content and allocation. Random_G6 is a '
         'sitewise Frobenius-matched Gaussian direction before BF16 rounding; actual '
         'BF16 strength is reported and may differ. Conditional necessity applies only '
         'to the union66/full-nine-layer-cut background; leave_G6 V195 is the rescue-loss '
         'prediction, V198 is descriptive. No cross-position transfer, common head '
         'semantic function, identity/familiar-action decoder or neural root-cause claim.')
PRIMARY = dict(name='H_G6', primary_arm='head_G6',
    primary_cases=['V195_A195_head_G6', 'V198_A195_head_G6'],
    prediction='Both layer0-only head_G6 cases strictly select only milk, with same-V controls stable.',
    strict_selection='selected_objects == [milk_1], both fingerpad contacts, lift >.02m for five consecutive records',
    controls='Native/all_allowed/full_hardmask/arithmetic_sham_G6 reproduce entire accepted value-stage references.',
    secondary_G4='head_G4 does not replicate both-V only-milk; cannot replace the primary.',
    secondary_conditional_necessity='leave_G6 V195 loses full_hardmask rescue; V198 is reported without a rescue-loss prediction.',
    random_control='Report both random_G6 outcomes and actual BF16 norm error; replication cannot support G6-specific efficacy.',
    specificity_limit='Primary success alone is not group specificity or universal necessity. Candidate narrowed using complete pre-intervention geometry, not new behavioral outcomes.',
    all18_physical_cases_preregistered=True, q0_rank_selection=False, posthoc_candidate_replacement=False)


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
        rows = [expected_counts(item) for _, _, item in PLAN]
        return {key: sum(row[key] for row in rows) for key in rows[0]}
    assert arm in ARMS
    masked = arm != 'native'
    cut_sites = 135 if arm in ('full_hardmask', 'leave_G4', 'leave_G6') else 15 if arm in ('head_G4', 'head_G6', 'random_G6') else 0
    return dict(model_before=30, model_after=30, block_before=1080, block_after=1080, pre_W=1080,
        native_UND_dispatch=1080, native_GEN_dispatch=1080, all_allowed_dispatch=1080 if masked else 0,
        substituted_sites=1080 if masked else 0, clone_sham_dispatch=15 if arm == 'arithmetic_sham_G6' else 0,
        value_zero_dispatch=0, cut_dispatch=cut_sites)


def control_reference_case(v, arm):
    reference = dict(native='native', all_allowed='all_allowed', full_hardmask='hardmask',
                     arithmetic_sham_G6='all_allowed').get(arm)
    return None if reference is None else f'V{v}_A195_{reference}'


def arm_window(arm):
    if arm in ('native', 'all_allowed'):
        return None
    return FULL_WINDOW if arm in ('full_hardmask', 'leave_G4', 'leave_G6') else LOCAL_WINDOW


def selected_heads(arm, layer=0):
    if arm == 'full_hardmask':
        return list(range(32))
    if arm in ('head_G4', 'head_G6', 'arithmetic_sham_G6', 'random_G6'):
        return GROUPS[arm.rsplit('_', 1)[1]] if layer == 0 else []
    if arm in ('leave_G4', 'leave_G6'):
        retained = GROUPS[arm.rsplit('_', 1)[1]]
        return [head for head in range(32) if head not in retained] if layer == 0 else list(range(32))
    return []


def source_contract(baseline):
    """Read original accepted seals before output creation or torch import."""
    path = ROOT / 'work/probe_cosmos_milk_future_value_mechanism.py'
    assert sha(path) == VALUE_SHA
    value = load('head_groups_frozen_value_mechanism', path)
    context = value.source_contract(baseline)
    producer = baseline / value.STAGE
    assert not (producer / 'failed.json').exists() and sha(producer / 'complete.json') == VALUE_COMPLETE_SHA
    names = ('complete', 'results', 'protocol', 'provenance', 'sources', 'primary-prediction')
    docs = {name: read(producer / (name + '.json')) for name in names}
    complete, results = docs['complete'], docs['results']
    assert complete['state'] == results['state'] == 'complete'
    assert complete['script_sha256'] == docs['provenance']['script_sha256'] == VALUE_SHA
    for name in names[1:]:
        assert complete[name.replace('-', '_') + '_sha256'] == sha(producer / (name + '.json'))
    assert complete['fresh_q0_predictions'] == 12 and complete['fresh_model_forwards'] == 360
    assert complete['official_dispatch_counts'] == value.expected_counts() and complete['total_official_dispatch_calls'] == 38070
    assert complete['actual_selected_site_captures'] == 1620
    for key in ('two_native_and_two_allallowed_entire_accepted_records_and_boundaries_exact',
                'two_arithmetic_shams_entire_accepted_allallowed_records_and_boundaries_exact',
                'two_hardmask_entire_accepted_early_l1_9_records_and_boundaries_exact',
                'all10_masked_sites_same_union66_background', 'all360_actual_complete_model_kwargs_saved',
                'all_original_symbols_restored', 'all_owned_hooks_removed'):
        assert complete[key] is True
    assert docs['sources'] == dict(source_evidence=context['source_evidence'],
        original_sources=context['original_noise_sources'], frozen_contract=context['frozen_contract'])
    assert docs['protocol']['selected_window_zero_based_inclusive'] == WINDOW == value.WINDOW
    assert docs['protocol']['prompt'] == PROMPT and value.FILES == FILES
    assert [row['case'] for row in results['cases']] == [item[0] for item in value.PLAN]
    rows, references = {row['case']: row for row in results['cases']}, {}
    for arm in ('native', 'all_allowed', 'hardmask'):
        for v in (195, 198):
            label = f'V{v}_A195_{arm}'
            row, folder = rows[label], producer / label
            assert row['arm'] == arm and row['vision_noise_source_seed'] == v
            assert row['action_noise_source_seed'] == row['runtime_rng_seed'] == 195 and row['fresh_model_forwards'] == 30
            assert row['window'] == (WINDOW if arm == 'hardmask' else None)
            assert set(row['files_sha256']) == set(FILES)
            for name, expected in row['files_sha256'].items():
                assert sha(folder / name) == expected
            assert len(row['boundary_files']) == 30
            for step, item in enumerate(row['boundary_files']):
                assert item['step'] == step and item['file'] == f'boundaries-t{step:02d}.pt'
                assert item['current_shape'] == [37, 50, 4096] and item['action_shape'] == [37, 16, 4096]
                assert sha(folder / item['file']) == item['sha256']
            manifest = read(folder / 'site-captures.json')
            assert manifest['state'] == 'complete' and manifest['sites'] == 135
            assert [(item['step'], item['layer_zero_based']) for item in manifest['files']] == [(t, l) for t in range(15) for l in range(9)]
            for item in manifest['files']:
                assert sha(folder / item['file']) == item['sha256']
            references[label] = dict(folder=str(folder), arm=arm, files_sha256=row['files_sha256'],
                boundary_files=row['boundary_files'], site_capture_manifest_sha256=sha(folder / 'site-captures.json'))
    context['source_evidence'] = dict(context['source_evidence'], accepted_value_stage=dict(producer=str(producer),
        script_sha256=VALUE_SHA, complete_sha256=VALUE_COMPLETE_SHA,
        documents_sha256={name: sha(producer / (name + '.json')) for name in names}, q0_references=references))
    geometry = baseline / 'future-head-geometry'
    assert sha(ROOT / 'work/analyze_cosmos_milk_future_head_geometry.py') == GEOMETRY_SHA
    assert not (geometry / 'failed.json').exists() and sha(geometry / 'complete.json') == GEOMETRY_COMPLETE_SHA
    gc, ga = read(geometry / 'complete.json'), read(geometry / 'analysis.json')
    assert gc['state'] == ga['state'] == 'complete' and gc['script_sha256'] == GEOMETRY_SHA
    assert gc['sites_checked'] == ga['sites_checked'] == 270
    assert gc['files_sha256']['analysis.json'] == GEOMETRY_ANALYSIS_SHA == sha(geometry / 'analysis.json')
    for name, expected in gc['files_sha256'].items():
        assert sha(geometry / name) == expected
    for document in (gc, ga):
        assert document['model_calls'] == document['official_attention_dispatch_calls'] == document['solver_calls'] == document['physics_calls'] == 0
        assert document['CUDA_initialized'] is False
    context['source_evidence']['candidate_geometry'] = dict(folder=str(geometry), source_sha256=GEOMETRY_SHA,
        complete_sha256=GEOMETRY_COMPLETE_SHA, analysis_sha256=GEOMETRY_ANALYSIS_SHA, files_sha256=gc['files_sha256'],
        sites_checked=270, candidate_scope='Only L0 G4/G6, narrowed before new behavioral outcomes; no cross-layer common-function inference.')
    context['modules']['value'] = value
    context['old_q0_references'], context['out'] = references, baseline / STAGE
    return context


def observer_class(value, windows, frozen):
    Parent = value.observer_class(windows, frozen)

    class HeadObserver(Parent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.window = arm_window(self.case_arm)
            self.propagation, self.random_checks = None, []
            self.propagation_counts = dict(projection=0, residual=0, MLP=0, completed=0)

        def before_block(self, layer, module, args):
            assert self.propagation is None
            super().before_block(layer, module, args)
            if windows.cut_active(WINDOW, self.step, layer):
                current = self.index['current_rows'].to(args[1].device)
                self.propagation = dict(block_input_current50=self.runtime.cpu(args[1].index_select(0, current)))

        def begin(self):
            super().begin()
            try:
                for layer, block in enumerate(self.model.layers):
                    self.handles.append(block.self_attn.to_add_out.register_forward_hook(
                        lambda module, args, output, layer=layer: self.observe_projection(layer, output)))
                    self.handles.append(block.post_attention_layernorm_moe_gen.register_forward_pre_hook(
                        lambda module, args, layer=layer: self.observe_residual(layer, args)))
                    self.handles.append(block.mlp_moe_gen.register_forward_hook(
                        lambda module, args, output, layer=layer: self.observe_MLP(layer, output)))
            except BaseException:
                self.reset()
                raise

        def observe_current(self, layer, tensor, key, count):
            assert self.active == layer and tensor.shape == (266, 4096) and tensor.dtype == self.torch.bfloat16
            if self.propagation is not None:
                assert key not in self.propagation and bool(self.torch.isfinite(tensor).all())
                current = self.index['current_rows'].to(tensor.device)
                self.propagation[key] = self.runtime.cpu(tensor.index_select(0, current))
                self.propagation_counts[count] += 1

        def observe_projection(self, layer, output):
            self.observe_current(layer, output, 'to_add_out_current50', 'projection')

        def observe_residual(self, layer, args):
            assert len(args) == 1
            self.observe_current(layer, args[0], 'attention_residual_current50', 'residual')

        def observe_MLP(self, layer, output):
            self.observe_current(layer, output, 'mlp_moe_gen_current50', 'MLP')

        def random_selected(self, z0, zk, heads):
            torch = self.torch
            base, cut = self.runtime.cpu(z0.index_select(2, heads)), self.runtime.cpu(zk.index_select(2, heads))
            delta_cut = cut.float() - base.float()
            generator = torch.Generator(device='cpu')
            seed = 901000 + self.step * 36 + self.active
            generator.manual_seed(seed)
            generator_before = generator.get_state().clone()
            direction = torch.randn(delta_cut.shape, dtype=torch.float32, device='cpu', generator=generator)
            target, raw_norm = delta_cut.norm(), direction.norm()
            assert bool(torch.isfinite(target)) and bool(torch.isfinite(raw_norm)) and float(raw_norm) > 0
            unit = direction / raw_norm
            delta = unit * target
            candidate = base.float() + delta
            result = candidate.to(torch.bfloat16)
            result = torch.where(delta == 0, base, result)
            assert all(bool(torch.isfinite(item).all()) for item in (direction, unit, delta, candidate, result))
            actual = result.double() - base.double()
            target_actual = cut.double() - base.double()
            target_l2, actual_l2 = float(target_actual.norm()), float(actual.norm())
            ratio = None if target_l2 == 0 else actual_l2 / target_l2
            error = 0.0 if target_l2 == actual_l2 == 0 else abs(actual_l2 / target_l2 - 1.0)
            report = dict(step=self.step, layer_zero_based=self.active, seed=seed,
                target_Dk_FP32_l2=float(target), random_delta_FP32_l2=float(delta.norm()),
                target_Dk_actual_BF16_l2=target_l2, random_actual_BF16_l2=actual_l2,
                actual_BF16_norm_ratio=ratio, actual_BF16_relative_norm_error=error,
                relative_norm_tolerance=RANDOM_RELATIVE_NORM_TOLERANCE,
                actual_BF16_equal_strength_within_tolerance=error <= RANDOM_RELATIVE_NORM_TOLERANCE,
                actual_BF16_changed_values=int(torch.count_nonzero(actual)),
                zero_delta_original_bytes_retained=True, zero_target=target_l2 == 0,
                finite=True, independent_CPU_generator=True, no_post_BF16_rescaling=True,
                BF16_quantization_error_l2=float((result.double() - candidate.double()).norm()))
            self.random_checks.append(report)
            return result.to(z0.device), dict(FP32random_Dk_selected=delta_cut, FP32random_direction=direction,
                FP32random_unit_direction=unit, FP32random_delta_selected=delta,
                FP32random_selected=candidate, random_generator_before=generator_before,
                random_generator_after=generator.get_state().clone(), random_control=report)

        def dispatch(self, *args, **kwargs):
            torch, exact = self.torch, frozen.byte_exact
            assert self.in_model and self.active is not None and self.capture_pending is None
            assert len(args) == 3 and set(kwargs) == {'is_causal', 'enable_gqa', 'backend', 'parallel_config'}
            assert kwargs['enable_gqa'] is True and kwargs['backend'] is kwargs['parallel_config'] is None
            causal = kwargs['is_causal']
            assert type(causal) is bool and self.site_calls == ([] if causal else ['UND'])
            self.site_calls.append('UND' if causal else 'GEN')
            qlen, kvlen = (121, 121) if causal else (266, 387)
            assert args[0].shape == (1, qlen, 32, 128) and args[1].shape == args[2].shape == (1, kvlen, 8, 128)
            assert all(item.dtype == torch.bfloat16 and bool(torch.isfinite(item).all()) for item in args)
            self.counts['native_UND_dispatch' if causal else 'native_GEN_dispatch'] += 1
            witnesses = tuple(item.clone(memory_format=torch.preserve_format) for item in args) if not causal else None
            rng = (torch.get_rng_state().clone(), torch.cuda.get_rng_state(args[0].device).clone()) if not causal else None
            native = self.original(*args, **kwargs)
            assert native.shape == (1, qlen, 32, 128) and native.dtype == torch.bfloat16 and bool(torch.isfinite(native).all())
            if causal:
                return native
            capture = windows.cut_active(WINDOW, self.step, self.active)
            active = windows.cut_active(self.window, self.step, self.active)
            current, action, future, union = (self.index[key].to(native.device) for key in
                ('current_rows', 'action_rows', 'future_rows', 'union_rows'))
            active_heads = selected_heads(self.case_arm, self.active) if active else []
            heads = torch.tensor(active_heads, dtype=torch.long, device=native.device)
            retained = torch.tensor([h for h in range(32) if h not in active_heads], dtype=torch.long, device=native.device)
            z0 = zk = cloned = arithmetic_fp32 = None
            random_data = dict(FP32random_Dk_selected=None, FP32random_direction=None, FP32random_unit_direction=None,
                FP32random_delta_selected=None, FP32random_selected=None, random_generator_before=None,
                random_generator_after=None, random_control=None)
            event = dict(step=self.step, layer_zero_based=self.active, arm=self.case_arm, window_name=self.case_arm,
                selected_window_cut_active=active, window_zero_based=self.window, capture_window_active=capture,
                common_masked_union66_background=self.case_arm != 'native', original_GEN_dispatch_calls=1,
                original_UND_dispatch_calls=1, extra_all_allowed_calls=0, extra_cut_calls=0,
                extra_value_zero_calls=0, extra_clone_sham_calls=0, to_add_out_shape=[266, 4096],
                to_add_out_actual_calls=0, processor_recompute_calls=0,
                selected_query_heads=active_heads, retained_query_heads=[h for h in range(32) if h not in active_heads])
            if self.case_arm != 'native':
                self.counts['all_allowed_dispatch'] += 1
                z0 = self.original(*args, **dict(kwargs, attn_mask=self.gpu_masks['all_allowed']))
                assert z0.shape == native.shape and z0.dtype == native.dtype and bool(torch.isfinite(z0).all())
                event['extra_all_allowed_calls'] = 1
                merged = native.clone()
                merged[:, union] = z0[:, union]
                if active:
                    z0_current = z0.index_select(1, current)
                    candidate = z0_current.clone()
                    if self.case_arm == 'arithmetic_sham_G6':
                        value_clone = args[2].clone(memory_format=torch.preserve_format)
                        assert value_clone.stride() == args[2].stride() and exact(value_clone, args[2], torch)
                        self.counts['clone_sham_dispatch'] += 1
                        cloned = self.original(args[0], args[1], value_clone, **dict(kwargs, attn_mask=self.gpu_masks['all_allowed']))
                        assert exact(cloned, z0, torch) and exact(value_clone, args[2], torch)
                        base = z0_current.index_select(2, heads)
                        delta = cloned.index_select(1, current).index_select(2, heads).float() - base.float()
                        arithmetic_fp32 = base.float() + delta
                        selected = arithmetic_fp32.to(torch.bfloat16)
                        selected = torch.where(delta == 0, base, selected)
                        assert exact(selected, base, torch)
                        event.update(extra_clone_sham_calls=1, full_clone_official_dispatch_byte_exact_Z0=True,
                            arithmetic_zero_delta_values=int((delta == 0).sum()), zero_delta_Z0_bytes_retained=True)
                    else:
                        self.counts['cut_dispatch'] += 1
                        zk = self.original(*args, **dict(kwargs, attn_mask=self.gpu_masks['cut_future_current']))
                        assert zk.shape == native.shape and zk.dtype == native.dtype and bool(torch.isfinite(zk).all())
                        unblocked = self.masks['cut_future_current']['unblocked_rows'].to(native.device)
                        assert exact(zk[:, unblocked], z0[:, unblocked], torch)
                        event.update(extra_cut_calls=1, all_unblocked_GEN_rows_vs_same_input_all_allowed_byte_exact=True)
                        selected = zk.index_select(1, current).index_select(2, heads)
                        if self.case_arm == 'random_G6':
                            selected, random_data = self.random_selected(z0_current, zk.index_select(1, current), heads)
                            event['random_control'] = random_data['random_control']
                    candidate.index_copy_(2, heads, selected)
                    assert exact(candidate.index_select(2, retained), z0_current.index_select(2, retained), torch)
                    if self.case_arm not in ('random_G6', 'arithmetic_sham_G6'):
                        assert exact(candidate.index_select(2, heads), zk.index_select(1, current).index_select(2, heads), torch)
                    merged[:, current] = candidate
                    event.update(unselected_current_head_bytes_equal_Z0=True, real_query_head_axis=[32, 128])
                assert exact(merged[:, action], z0[:, action], torch) and exact(merged[:, future], native[:, future], torch)
                self.counts['substituted_sites'] += 1
            else:
                merged = native
            assert merged.dtype == native.dtype and bool(torch.isfinite(merged).all())
            for item, witness in zip(args, witnesses):
                assert exact(item, witness, torch), 'Official calls mutated actual Q/K/V'
            assert exact(rng[0], torch.get_rng_state(), torch) and exact(rng[1], torch.cuda.get_rng_state(args[0].device), torch)
            event.update(substituted_query_rows=0 if self.case_arm == 'native' else 66, selected_current_rows=50,
                selected_action_rows=16, future200_native_bytes_preserved=True, UND_dispatch_return_unmodified=True,
                original_QKV_bytes_preserved=True, RNG_bytes_preserved=True)
            if capture:
                endpoints = {name: None if endpoint is None else self.runtime.cpu(endpoint[:, current]) for name, endpoint in
                    (('Znative', native), ('Z0', z0), ('Zv', None), ('Zk', zk), ('Zreal', merged), ('Zclone', cloned))}
                baseline = endpoints['Znative'] if z0 is None else endpoints['Z0']
                effective = endpoints['Zreal'].double() - baseline.double()
                assert effective.shape == (1, 50, 32, 128)
                headwise = [dict(query_head=h, kv_head=h // 4, delta_l2=float(effective[:, :, h].norm()),
                    delta_maxabs=float(effective[:, :, h].abs().max()), changed_values=int(torch.count_nonzero(effective[:, :, h])),
                    signed_zero_bit_changes=int(((endpoints['Zreal'][:, :, h] == 0) & (baseline[:, :, h] == 0) &
                        (torch.signbit(endpoints['Zreal'][:, :, h]) != torch.signbit(baseline[:, :, h]))).sum())) for h in range(32)]
                event['capture_file'] = f'sites/t{self.step:02d}-L{self.active:02d}.pt'
                self.capture_pending = dict(step=self.step, layer_zero_based=self.active, case_arm=self.case_arm,
                    current_rows=self.index['current_rows'], future_keys=self.index['future_keys'],
                    selected_query_heads=active_heads, retained_query_heads=retained.cpu(),
                    Qcurrent50=self.runtime.cpu(args[0][:, current]), Kall=self.runtime.cpu(args[1]), Vall=self.runtime.cpu(args[2]),
                    endpoints=endpoints, missing_endpoints=[name for name, endpoint in endpoints.items() if endpoint is None],
                    FP32allocation=None, FP32arithmetic_sham=None,
                    FP32arithmetic_sham_selected=None if arithmetic_fp32 is None else self.runtime.cpu(arithmetic_fp32),
                    original_QKV_metadata=[self.tensor_meta(item) for item in args],
                    endpoint_metadata={name: self.tensor_meta(endpoint) for name, endpoint in endpoints.items()},
                    RNG_before=dict(cpu=rng[0], cuda=rng[1]), RNG_after=dict(cpu=torch.get_rng_state().clone(),
                        cuda=torch.cuda.get_rng_state(args[0].device).clone()), actual_QKV_bytes_unchanged=True,
                    headwise_effective_vs_native_or_Z0=headwise, **random_data, event=dict(event), scope=SCOPE)
            self.events.append(event)
            self.pending = merged.squeeze(0).flatten(-2, -1)
            return merged

        def before_W(self, layer, module, args):
            # Bypass only the parent's premature site sealing; retain the frozen
            # original once-only full266 projection input byte gate unchanged.
            frozen.FutureReadCuts.before_W(self, layer, module, args)
            self.events[-1]['actual_pre_W_metadata'] = self.tensor_meta(args[0])
            if self.capture_pending is not None:
                data = self.capture_pending
                current = self.index['current_rows'].to(args[0].device)
                actual = args[0][current].reshape(1, 50, 32, 128)
                assert frozen.byte_exact(self.runtime.cpu(actual), data['endpoints']['Zreal'], self.torch)
                data.update(actual_pre_W_current_bytes_equal_Zreal=True,
                    actual_pre_W_full266_metadata=self.tensor_meta(args[0]), original_full_projection_calls_at_site=1)

        def after_block(self, layer, module, args, output):
            if self.propagation is not None:
                assert self.capture_pending is not None and set(self.propagation) == {
                    'block_input_current50', 'to_add_out_current50', 'attention_residual_current50', 'mlp_moe_gen_current50'}
                current = self.index['current_rows'].to(output[1].device)
                actual = self.runtime.cpu(output[1].index_select(0, current))
                assert bool(self.torch.isfinite(actual).all())
                self.propagation['block_output_current50'] = actual
                residual = self.propagation['block_input_current50'] + self.propagation['to_add_out_current50']
                reconstructed = self.propagation['attention_residual_current50'] + self.propagation['mlp_moe_gen_current50']
                assert residual.dtype == reconstructed.dtype == self.torch.bfloat16
                assert frozen.byte_exact(residual, self.propagation['attention_residual_current50'], self.torch)
                assert frozen.byte_exact(reconstructed, actual, self.torch), 'Actual BF16 residual chain does not reconstruct block output'
                data = self.capture_pending
                data.update(propagation=self.propagation, actual_BF16_residual_chain_byte_exact=True,
                    propagation_formula='BF16(BF16(block_input + actual_to_add_out) + actual_mlp_moe_gen)',
                    propagation_observed_forward_only=True, event=dict(self.events[-1]))
                path = self.folder / self.events[-1]['capture_file']
                path.parent.mkdir(exist_ok=True)
                assert not path.exists()
                self.torch.save(data, path)
                size = path.stat().st_size
                assert size < 100_000_000
                self.capture_files.append(dict(step=self.step, layer_zero_based=layer, file=str(path.relative_to(self.folder)),
                    sha256=sha(path), bytes=size, captured_endpoints=[name for name, endpoint in data['endpoints'].items() if endpoint is not None],
                    missing_endpoints=data['missing_endpoints'], actual_BF16_residual_chain_byte_exact=True))
                self.propagation_counts['completed'] += 1
                self.capture_pending = self.propagation = None
            super().after_block(layer, module, args, output)

        def report(self, record):
            assert self.restored and not self.handles and not self.in_model and self.active is None
            assert self.capture_pending is self.propagation is None and self.counts == expected_counts(self.case_arm)
            assert len(self.calls) == len(self.files) == len(self.outputs) == 30 and len(self.capture_files) == 135
            assert self.finite_boundary_checks == 37 * 30
            assert self.propagation_counts == dict(projection=135, residual=135, MLP=135, completed=135)
            assert [(item['step'], item['layer_zero_based']) for item in self.capture_files] == [(t, l) for t in range(15) for l in range(9)]
            assert len(self.random_checks) == (15 if self.case_arm == 'random_G6' else 0)
            self.runtime.require_exact(self.torch.stack(self.outputs), record['action_velocity'], 'actual all30 model action tuple')
            assert self.reader.settings(self.torch, self.model.layers[0].self_attn.processor) == self.settings
            random_summary = dict(sites=len(self.random_checks), checks=self.random_checks,
                relative_norm_tolerance=RANDOM_RELATIVE_NORM_TOLERANCE,
                actual_BF16_equal_strength_all_sites_within_tolerance=all(item['actual_BF16_equal_strength_within_tolerance'] for item in self.random_checks)
                    if self.random_checks else None,
                tolerance_is_interpretation_only=True, failure_does_not_recalibrate_or_select_cases=True)
            manifest = dict(state='complete', case_arm=self.case_arm, window=WINDOW, sites=135, files=self.capture_files,
                total_bytes=sum(item['bytes'] for item in self.capture_files), no_extra_dispatch_for_missing_endpoints=True,
                endpoint_scope='Actual current50 pre-W32x128 query-head axis, missing endpoints null; same live Q/K/V within site.',
                selected_query_heads=selected_heads(self.case_arm), propagation_counts=self.propagation_counts,
                random_control=random_summary, scope=SCOPE)
            write(self.folder / 'site-captures.json', manifest, exclusive=True)
            return dict(state='complete', case_arm=self.case_arm, selected_window=self.window, counts=self.counts,
                boundary_files=self.files, indexes_and_masks_sha256=sha(self.folder / 'indexes-and-masks.pt'),
                site_capture_manifest_sha256=sha(self.folder / 'site-captures.json'), captured_sites=135,
                selected_query_heads=selected_heads(self.case_arm), original_full_266_row_projection_calls=1080,
                all_actual_dispatch_flatten_pre_W_byte_exact=True, all_masked_sites_same_union66=self.case_arm != 'native',
                future200_and_UND_native_dispatch_returns_preserved=True, action16_same_input_Z0_at_all_masked_sites=True,
                all30_actual_complete_model_kwargs_saved=True, all30_actual37_current_action_boundaries_finite=True,
                arithmetic_sham_all_selected_sites_byte_exact_Z0=True if self.case_arm == 'arithmetic_sham_G6' else None,
                hardmask_unblocked_queries_byte_exact_Z0=True if self.counts['cut_dispatch'] else None,
                all135_actual_BF16_projection_MLP_residual_chains_byte_exact=True, propagation_counts=self.propagation_counts,
                random_control=random_summary, processor_recompute_calls=0, extra_projection_or_MLP_forward_calls=0,
                all_owned_hooks_removed=True, original_dispatch_symbol_restored=True, numerical_sham_is_native_noop=False, scope=SCOPE)
    return HeadObserver


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Original accepted BASE, never a new mechanism stage')
    parser.add_argument('--check-source-contract', action='store_true', help='Read actual accepted seals; no output or model')
    args = parser.parse_args()
    baseline = args.output.resolve()
    context = source_contract(baseline)
    if args.check_source_contract:
        print(json.dumps(dict(state='source_contract_verified', image=str(context['image']),
            accepted_q0_references=list(context['old_q0_references']), expected_counts=expected_counts()), indent=2))
        return
    out, paths, contract = context['out'], context['paths'], context['frozen_contract']
    value, windows, frozen, factor = (context['modules'][key] for key in ('value', 'windows', 'frozen', 'factor'))
    out.mkdir(exist_ok=False)
    protocol = dict(plan=[dict(case=label, vision_noise_source_seed=v, action_noise_source_seed=195, runtime_rng_seed=195,
        arm=arm, window=arm_window(arm),
        selected_query_heads=selected_heads(arm), control_reference_case=control_reference_case(v, arm)) for label, v, arm in PLAN],
        trial_order=[label for label, _, _ in PLAN], prompt=PROMPT, shift_cm=12, image_path=str(context['image']),
        image_sha256=sha(context['image']), primary_hypothesis=PRIMARY, selected_window_zero_based_inclusive=WINDOW,
        query_head_groups=GROUPS, full_window_zero_based_inclusive=FULL_WINDOW, local_window_zero_based_inclusive=LOCAL_WINDOW,
        distinct_parameter_head_positions_per_group=4, intervention_head_sites_per_group=60,
        candidate_geometry_source_sha256=GEOMETRY_SHA, candidate_geometry_analysis_sha256=GEOMETRY_ANALYSIS_SHA,
        group_numbering_does_not_imply_same_cross_layer_function=True,
        masked_background='All nonnative sites union66 Z0; selected current50/head subset changes, action16=Z0/future200=native/UND=native.',
        head_patch_rule='Actual same-call official BF16 Zk replaces selected current50 query-head axis of Z0 before whole266 projection once.',
        arithmetic_sham_rule='Same-valued full387 V clone official all-allowed dispatch equals full Z0; selected G6 FP32 zero difference/sum, final BF16 cast once; zero delta preserves bytes.',
        random_control_rule='Independent CPU Generator seed901000+step*36+layer; FP32 Gaussian direction normalized to actual selected G6 Dk Frobenius norm, FP32 Z0+delta then one BF16 cast; zero delta retains original bytes.',
        random_actual_BF16_relative_norm_tolerance=RANDOM_RELATIVE_NORM_TOLERANCE,
        random_tolerance_is_interpretation_only=True, random_no_post_BF16_rescaling=True,
        q0_predictions=18, model_forwards=540, expected_counts=expected_counts(), total_official_dispatch_calls=57090,
        original_dispatch_calls=38880, extra_all_allowed_dispatch_calls=17280, selected_extra_dispatch_calls=930,
        selected_sites_per_case=135, captures_per_case=135, total_site_captures=2430, max_capture_file_bytes=100_000_000,
        capture_shapes=dict(Qcurrent50=[1, 50, 32, 128], Kall=[1, 387, 8, 128], Vall=[1, 387, 8, 128],
            current_endpoint=[1, 50, 32, 128], selected_G6_FP32=[1, 50, 4, 128], propagation_current=[50, 4096]),
        missing_endpoint_rule='Explicit null; no extra dispatch to fill capture.',
        propagation='Actual block input, to_add_out output, postattention residual, mlp_moe_gen output, block output via owned hooks; CPU BF16 two-add byte gate.',
        actual_model_kwargs_and_boundaries_saved_steps=list(range(30)), offline_reconstruction_is_analysis_only=True,
        later_physics_preregistered=dict(trial_order=[label for label, _, _ in PLAN], cases=18, cached_q0=18,
            later_noise_base=195, q1_to_q7_seeds=list(range(196, 203)), later_predictions=126,
            later_model_forwards=3780, total_two_stage_model_forwards=4320, no_q0_score_or_rank_selection=True),
        physics_calls=0, no_training=True, no_residual_compensation=True, scope=SCOPE)
    write(out / 'protocol.json', protocol, exclusive=True)
    write(out / 'primary-prediction.json', dict(status='not_evaluated_q0_only', **PRIMARY, scope=SCOPE), exclusive=True)
    write(out / 'sources.json', dict(source_evidence=context['source_evidence'], original_sources=context['original_noise_sources'],
        frozen_contract=contract), exclusive=True)
    started, completed, rows, runtime, observer, noise = time.perf_counter(), [], [], None, None, None
    counts = {key: 0 for key in expected_counts()}

    def progress(stage, **values):
        status = dict(stage=stage, completed=completed, completed_model_forwards=30 * len(completed), actual_counts=counts,
            elapsed_s=time.perf_counter() - started, **values)
        write(out / 'progress.json', status)
        print('[FUTURE-HEAD-GROUPS] ' + json.dumps(status), flush=True)

    try:
        os.environ['HF_HUB_OFFLINE'] = '1'
        base, factory = load('head_groups_runtime', paths['normal_runtime']), load('head_groups_factory', paths['factory'])
        base.TASKS['milk'] = (7, 'pick_up_the_milk_and_place_it_in_the_basket', PROMPT, 'milk_1')
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
            scan.validate_cache(caches[v], originals[v], checker, runtime, 'source' + str(v))
        for label, reference in context['old_q0_references'].items():
            references[label] = torch.load(Path(reference['folder']) / 'states.pt', map_location='cpu', weights_only=True, mmap=True)
            scan.validate_record(references[label], torch)
        runtime.require_exact(caches[195]['steps'], caches[198]['steps'], 'complete source schedule/pack')
        clean = originals[195]['model_input']['vision_tokens'][0][0, :, 0]
        runtime.require_exact(clean, originals[198]['model_input']['vision_tokens'][0][0, :, 0], 'conditioned current frame0')
        native_random = runtime.cm.randn_tensor
        provenance = dict(script_sha256=sha(Path(__file__)), protocol_sha256=sha(out / 'protocol.json'),
            sources_sha256=sha(out / 'sources.json'), source_sha256={key: sha(path) for key, path in paths.items()},
            accepted_value_source_sha256=VALUE_SHA, accepted_value_complete_sha256=VALUE_COMPLETE_SHA,
            candidate_geometry_source_sha256=GEOMETRY_SHA, candidate_geometry_analysis_sha256=GEOMETRY_ANALYSIS_SHA,
            accepted_layer_group_source_sha256=value.GROUP_SHA, noise_seam_source_sha256=frozen.FACTOR_SHA,
            scheduler_sha256=sha(factory.SCHEDULER), actual_transformer_sha256=sha(Path(inspect.getfile(type(model)))),
            actual_pipeline_source_sha256=sha(Path(inspect.getfile(type(runtime.pipe)))),
            dispatch_source_sha256=sha(Path(inspect.getfile(dispatch))), dispatch_signature=str(inspect.signature(dispatch)),
            **backend, kernel_evidence_limit='Registry/settings recorded; no selected-kernel profiler observation.',
            library_edits=False, training=False, physics_calls=0, scope=SCOPE)
        write(out / 'provenance.json', provenance, exclusive=True)
        Observer = observer_class(value, windows, frozen)
        records = {}
        for label, v, arm in PLAN:
            if arm not in ('native', 'all_allowed'):
                assert completed[:4] == [item[0] for item in PLAN[:4]]
            directory = out / label
            progress('fresh_q0', case=label)
            observer = Observer(runtime, components, scan, reader, caches[195], clean, references[f'V{v}_A195_native'],
                'native' if arm == 'native' else 'all_allowed', directory, case_arm=arm)
            noise = factor.InitialNoiseSources(runtime, scan, originals, v, 195)
            assert observer.original is dispatch and noise.original is native_random
            observer.begin()
            try:
                noise.begin()
                record, metadata = runtime.predict('milk', context['image'], 195, directory)
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
            assert metadata['input_png_sha256'] == sha(context['image']) and metadata['prompt'] == PROMPT and metadata['task_index'] == 7
            report = observer.report(record)
            prepared = factor.prepared_contract(record, originals, v, 195, runtime,
                SimpleNamespace(first=observer.first, calls=observer.calls, handle=None))
            noise_report = noise.report(record)
            reference_label = control_reference_case(v, arm)
            if reference_label:
                reference = context['old_q0_references'][reference_label]
                control = windows.saved_control_checks(directory, Path(reference['folder']), record, references[reference_label], runtime)
            else:
                baseline_label = f'V{v}_A195_all_allowed'
                control = windows.prefix_checks(directory, out / baseline_label, record, records[baseline_label], WINDOW, runtime)
            control['reference_case'] = reference_label
            write(directory / 'control-checks.json', control, exclusive=True)
            write(directory / 'observer-report.json', report, exclusive=True)
            write(directory / 'noise-audit.json', dict(noise=noise_report, preparation=prepared), exclusive=True)
            metadata.update(case=label, arm=arm, query=0, shift_cm=12, vision_noise_source_seed=v, action_noise_source_seed=195,
                runtime_rng_seed=195, window_zero_based_inclusive=observer.window, common_union66_all_allowed_background=arm != 'native',
                producer_script_sha256=provenance['script_sha256'], inherited_value_source_sha256=VALUE_SHA,
                control_reference_case=reference_label, entire_accepted_control_record_exact=reference_label is not None,
                captured_sites=135, site_capture_manifest_sha256=report['site_capture_manifest_sha256'],
                selected_query_heads=selected_heads(arm), primary_prediction_case=arm == 'head_G6',
                physical_prediction_evaluated=False, scope=SCOPE)
            write(directory / 'metadata.json', metadata)
            assert read(directory / 'normalized_actions.json') == record['actions'].tolist()
            rows.append(dict(case=label, arm=arm, window=observer.window, vision_noise_source_seed=v, action_noise_source_seed=195,
                runtime_rng_seed=195, fresh_model_forwards=30, actual_dispatch_counts=observer.counts, control_reference_case=reference_label,
                existing_entire_control_record_exact=reference_label is not None, selected_query_heads=selected_heads(arm),
                final_normalized_actions=record['actions'].tolist(), physical_prediction_evaluated=False,
                files_sha256={name: sha(directory / name) for name in FILES}, boundary_files=observer.files,
                site_capture_manifest_sha256=report['site_capture_manifest_sha256'], random_control=report['random_control']))
            records[label] = record
            completed.append(label)
            progress('case_complete', case=label)
            observer = noise = None
        for v in (195, 198):
            for label, actual_v, _ in PLAN:
                if v == actual_v:
                    runtime.require_exact(records[label]['pure_noise'], records[f'V{v}_A195_native']['pure_noise'], 'within-V nine whole returned noise pairs')
        runtime.require_exact(records['V195_A195_native']['pure_noise'][1], records['V198_A195_native']['pure_noise'][1], 'cross-V A195 only')
        for t in range(15):
            for layer in (0,):
                captures = [torch.load(out / f'V{v}_A195_random_G6/sites/t{t:02d}-L{layer:02d}.pt',
                    map_location='cpu', weights_only=True) for v in (195, 198)]
                for field in ('FP32random_direction', 'FP32random_unit_direction', 'random_generator_before', 'random_generator_after'):
                    runtime.require_exact(captures[0][field], captures[1][field], 'cross-V independent random direction/' + field)
        assert counts == expected_counts() and sum(counts[key] for key in DISPATCH_KEYS) == 57090
        assert completed == [item[0] for item in PLAN] and not checker._handles and checker._kind is None
        assert runtime.cm.randn_tensor is native_random and processor_module.dispatch_attention_fn is dispatch
        write(out / 'results.json', dict(state='complete', cases=rows, fresh_q0_predictions=18, fresh_model_forwards=540,
            official_dispatch_counts=counts, total_official_dispatch_calls=57090, physics_calls=0,
            primary_prediction_sha256=sha(out / 'primary-prediction.json'), primary_physical_prediction_evaluated=False,
            preregistered_physics_cases=[item[0] for item in PLAN], scope=SCOPE), exclusive=True)
        write(out / 'complete.json', dict(state='complete', script_sha256=sha(Path(__file__)),
            **{name.replace('-', '_') + '_sha256': sha(out / (name + '.json')) for name in ('results', 'provenance', 'protocol', 'sources', 'primary-prediction')},
            fresh_q0_predictions=18, fresh_model_forwards=540, official_dispatch_counts=counts,
            total_official_dispatch_calls=57090, original_dispatch_calls=38880, extra_dispatch_calls=18210,
            original_randn_calls_consumed=36, returned_source_draws_observed=36, actual_selected_site_captures=2430,
            two_native_and_two_allallowed_entire_accepted_value_records_and_boundaries_exact=True,
            two_full_hardmask_entire_accepted_value_hardmask_records_and_boundaries_exact=True,
            two_arithmetic_shams_entire_accepted_value_allallowed_records_and_boundaries_exact=True,
            all16_masked_sites_same_union66_background=True, all540_actual_complete_model_kwargs_saved=True,
            all2430_actual_BF16_projection_MLP_residual_chains_byte_exact=True,
            random15_cross_V_independent_directions_byte_exact=True,
            random_control_actual_BF16_norm_checks_reported=True,
            random_control_actual_BF16_equal_strength_all30_sites_within_tolerance=all(
                row['random_control']['actual_BF16_equal_strength_all_sites_within_tolerance'] for row in rows if row['arm'] == 'random_G6'),
            random_relative_norm_tolerance=RANDOM_RELATIVE_NORM_TOLERANCE, random_tolerance_is_interpretation_only=True,
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
