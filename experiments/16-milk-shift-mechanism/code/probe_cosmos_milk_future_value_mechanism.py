"""Twelve bounded X12 q0 runs separating future values from mask reallocation.

--output BASE creates only absent BASE/future-value-mechanism. Reuse the
accepted b5af/8fef/c106 input, index, noise, boundary and hook contracts.
V195/V198 x fixed A195 x six arms: native, all_allowed, arithmetic_sham,
value_zero, allocation_only, hardmask. All ten masked runs use union66
all-allowed at every layer/step. Only current50 at t0..14/L0..8 changes.
The allocation arm is an additive pre-W counterfactual, not a normalized
softmax attention: BF16 dispatch endpoints -> FP32 difference/sum -> BF16.
Same Q/K means within one actual call, not across diverging arm trajectories.
360 model forwards, 38070 official attention calls, no physics or training.
Every arm captures its actual 135 selected sites without extra dispatches.
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
GROUP_SHA = 'b5af0c4c63dbeee37c81a783c8c7f1f73146a6c9c679c0130899756b7f835689'
GROUP_COMPLETE_SHA = '75731bc1b1cebf59b18181c3495f33cb194ad9ef0c44d2c45fbf26bb3c516d05'
PROMPT = 'pick up the milk and place it in the basket'
STAGE = 'future-value-mechanism'
WINDOW = dict(steps=[0, 14], layers=[0, 8])
ARMS = ('native', 'all_allowed', 'arithmetic_sham', 'value_zero', 'allocation_only', 'hardmask')
PLAN = tuple((f'V{v}_A195_{arm}', v, arm) for arm in ARMS for v in (195, 198))
FILES = ('states.pt', 'normalized_actions.json', 'metadata.json', 'noise-audit.pt', 'noise-audit.json',
         'observer-report.json', 'indexes-and-masks.pt', 'control-checks.json', 'site-captures.json')
DISPATCH_KEYS = ('native_UND_dispatch', 'native_GEN_dispatch', 'all_allowed_dispatch',
                 'clone_sham_dispatch', 'value_zero_dispatch', 'cut_dispatch')
SCOPE = ('X12 fixed image/instruction, V195/V198 and A195, q0 only. The selected early '
         '135-site future-to-current pathway is split into value-zero and an additive '
         'pre-W allocation counterfactual. The latter retains the old future contribution '
         'and adds the nonfuture renormalization increment; it is not a legal normalized '
         'softmax and can increase amplitude. Same actual Q/K applies within each site '
         'only; later inputs and solver history may diverge across arms. BF16 official '
         'endpoints still have kernel rounding error. Future200 and UND dispatch returns '
         'are native; action16 uses same-input all-allowed. No residual repair, offline '
         'reconstruction intervention, training, new physics, identity/familiar-action '
         'decoder, head localization, cross-position mechanism or neural root-cause claim.')
PRIMARY = dict(name='H_content', primary_arm='value_zero',
    primary_cases=['V195_A195_value_zero', 'V198_A195_value_zero'],
    prediction='Both value_zero cases strictly select only milk: V195 changes cheese to milk and V198 preserves milk.',
    strict_selection='selected_objects == [milk_1], both fingerpad contacts, lift >.02m for five consecutive records',
    controls='Same-V native/all_allowed/arithmetic_sham controls remain stable; hardmask reproduces accepted early_l1_9.',
    allocation_only='Report both allocation_only cases; never promote them to a replacement primary prediction.',
    joint_interpretation='Both split arms succeeding cannot establish exclusive attribution. Both failing while hardmask succeeds is compatible with joint/interactive effects, not content alone sufficing.',
    all12_physical_cases_preregistered=True, q0_rank_selection=False)


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
    masked = arm != 'native'
    return dict(model_before=30, model_after=30, block_before=1080, block_after=1080, pre_W=1080,
        native_UND_dispatch=1080, native_GEN_dispatch=1080, all_allowed_dispatch=1080 if masked else 0,
        substituted_sites=1080 if masked else 0, clone_sham_dispatch=135 if arm == 'arithmetic_sham' else 0,
        value_zero_dispatch=135 if arm in ('value_zero', 'allocation_only') else 0,
        cut_dispatch=135 if arm in ('allocation_only', 'hardmask') else 0)


def source_contract(baseline):
    """Read only original accepted stages, before output creation or torch import."""
    path = ROOT / 'work/probe_cosmos_milk_future_current_layer_groups.py'
    assert sha(path) == GROUP_SHA
    groups = load('value_mechanism_frozen_layer_groups', path)
    windows, frozen, factor, contract, sources, image, paths, upstream = groups.source_contract(baseline)
    producer = baseline / 'future-current-layer-groups'
    assert not (producer / 'failed.json').exists() and sha(producer / 'complete.json') == GROUP_COMPLETE_SHA
    names = ('complete', 'results', 'protocol', 'provenance', 'sources', 'primary-prediction')
    docs = {name: read(producer / (name + '.json')) for name in names}
    complete, results = docs['complete'], docs['results']
    assert complete['state'] == results['state'] == 'complete'
    assert complete['script_sha256'] == docs['provenance']['script_sha256'] == GROUP_SHA
    for name in names[1:]:
        assert complete[name.replace('-', '_') + '_sha256'] == sha(producer / (name + '.json'))
    assert complete['fresh_q0_predictions'] == 12 and complete['fresh_model_forwards'] == 360
    assert complete['total_official_dispatch_calls'] == 37800
    assert complete['two_native_entire_prior_records_exact'] is complete['two_sham_entire_prior_records_exact'] is True
    assert complete['all_four_control_all30_full_tuples_and_boundaries_exact'] is True
    assert docs['sources'] == dict(source_evidence=upstream, original_sources=sources, frozen_contract=contract)
    assert docs['protocol']['windows_zero_based_inclusive'] == groups.WINDOWS
    assert groups.WINDOWS['early_l1_9'] == WINDOW and docs['protocol']['prompt'] == PROMPT
    assert [row['case'] for row in results['cases']] == [item[0] for item in groups.PLAN]
    rows, references = {row['case']: row for row in results['cases']}, {}
    for arm in ('native', 'all_allowed', 'early_l1_9'):
        for v in (195, 198):
            label = f'V{v}_A195_{arm}'
            row, folder = rows[label], producer / label
            assert row['vision_noise_source_seed'] == v and row['action_noise_source_seed'] == row['runtime_rng_seed'] == 195
            assert row['fresh_model_forwards'] == 30 and row['window'] == (WINDOW if arm == 'early_l1_9' else None)
            assert set(row['files_sha256']) == set((*groups.FILES, 'control-checks.json'))
            for name, expected in row['files_sha256'].items():
                assert sha(folder / name) == expected
            assert len(row['boundary_files']) == 30
            for step, item in enumerate(row['boundary_files']):
                assert item['step'] == step and item['file'] == f'boundaries-t{step:02d}.pt'
                assert item['current_shape'] == [37, 50, 4096] and item['action_shape'] == [37, 16, 4096]
                assert sha(folder / item['file']) == item['sha256']
            references[label] = dict(folder=str(folder), arm=arm, files_sha256=row['files_sha256'], boundary_files=row['boundary_files'])
    evidence = dict(upstream, accepted_layer_stage=dict(producer=str(producer), script_sha256=GROUP_SHA,
        documents_sha256={name: sha(producer / (name + '.json')) for name in names}, q0_references=references))
    return dict(modules=dict(groups=groups, windows=windows, frozen=frozen, factor=factor),
        frozen_contract=contract, original_noise_sources=sources, image=image, paths=paths,
        source_evidence=evidence, old_q0_references=references, out=baseline / STAGE)


def observer_class(windows, frozen):
    Parent = windows.observer_class(frozen)

    class ValueObserver(Parent):
        def __init__(self, *args, case_arm, **kwargs):
            super().__init__(*args, case_arm=case_arm, **kwargs)
            self.window = None if case_arm in ('native', 'all_allowed') else WINDOW
            self.counts.update(clone_sham_dispatch=0, value_zero_dispatch=0)
            self.capture_pending, self.capture_files = None, []
            self.finite_boundary_checks = 0

        def selected_hidden(self, gen):
            observed = super().selected_hidden(gen)
            finite = all(bool(self.torch.isfinite(value).all()) for value in observed.values())
            if not finite:
                self.torch.save(dict(step=self.step, boundary=len(self.boundaries), actual_current_action=observed),
                    self.folder / f'nonfinite-boundary-t{self.step:02d}-b{len(self.boundaries):02d}.pt')
                raise AssertionError('An actual current/action decoder boundary contains nonfinite values')
            self.finite_boundary_checks += 1
            return observed

        def tensor_meta(self, value):
            if value is None:
                return None
            return dict(**self.scan.tensor_meta(value, self.torch), stride=list(value.stride()), device=str(value.device),
                negative_zero_values=int(((value == 0) & self.torch.signbit(value)).sum().item()))

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
            assert all(value.dtype == torch.bfloat16 and bool(torch.isfinite(value).all()) for value in args)
            self.counts['native_UND_dispatch' if causal else 'native_GEN_dispatch'] += 1
            witnesses = tuple(value.clone(memory_format=torch.preserve_format) for value in args) if not causal else None
            rng = (torch.get_rng_state().clone(), torch.cuda.get_rng_state(args[0].device).clone()) if not causal else None
            native = self.original(*args, **kwargs)
            assert native.shape == (1, qlen, 32, 128) and native.dtype == torch.bfloat16 and bool(torch.isfinite(native).all())
            if causal:
                return native
            capture = windows.cut_active(WINDOW, self.step, self.active)
            active = windows.cut_active(self.window, self.step, self.active)
            current, action, future, union = (self.index[key].to(native.device) for key in
                ('current_rows', 'action_rows', 'future_rows', 'union_rows'))
            z0 = zv = zk = cloned = arithmetic_fp32 = None
            event = dict(step=self.step, layer_zero_based=self.active, arm=self.case_arm, window_name=self.case_arm,
                selected_window_cut_active=active, window_zero_based=self.window, capture_window_active=capture,
                common_masked_union66_background=self.case_arm != 'native', original_GEN_dispatch_calls=1,
                original_UND_dispatch_calls=1, extra_all_allowed_calls=0, extra_cut_calls=0,
                extra_value_zero_calls=0, extra_clone_sham_calls=0, to_add_out_shape=[266, 4096],
                to_add_out_actual_calls=0, processor_recompute_calls=0)
            if self.case_arm != 'native':
                self.counts['all_allowed_dispatch'] += 1
                z0 = self.original(*args, **dict(kwargs, attn_mask=self.gpu_masks['all_allowed']))
                event['extra_all_allowed_calls'] = 1
                assert z0.shape == native.shape and z0.dtype == native.dtype and bool(torch.isfinite(z0).all())
                merged = native.clone()
                merged[:, union] = z0[:, union]
                if active and self.case_arm == 'arithmetic_sham':
                    value_clone = args[2].clone(memory_format=torch.preserve_format)
                    assert value_clone.stride() == args[2].stride() and exact(value_clone, args[2], torch)
                    self.counts['clone_sham_dispatch'] += 1
                    cloned = self.original(args[0], args[1], value_clone, **dict(kwargs, attn_mask=self.gpu_masks['all_allowed']))
                    assert exact(cloned, z0, torch), 'Same-value clone official dispatch differs from Z0'
                    assert exact(value_clone, args[2], torch), 'Clone sham dispatch mutated V'
                    delta = z0[:, current].float() - z0[:, current].float()
                    arithmetic_fp32 = z0[:, current].float() + delta
                    selected = arithmetic_fp32.to(dtype=torch.bfloat16)
                    # Explicitly retain signed-zero and all other original bytes at zero delta.
                    selected = torch.where(delta == 0, z0[:, current], selected)
                    assert exact(selected, z0[:, current], torch), 'FP32 arithmetic sham changed current50 bytes'
                    merged[:, current] = selected
                    event.update(extra_clone_sham_calls=1, full_clone_official_dispatch_byte_exact_Z0=True,
                        arithmetic_zero_delta_values=int((delta == 0).sum().item()), zero_delta_Z0_bytes_retained=True)
                elif active:
                    if self.case_arm in ('value_zero', 'allocation_only'):
                        value_zero = args[2].clone(memory_format=torch.preserve_format)
                        keys = self.index['future_keys'].to(native.device)
                        retained = torch.cat([self.index[key] for key in ('text_keys', 'current_keys', 'action_keys')]).to(native.device)
                        value_zero[:, keys] = 0
                        assert value_zero.stride() == args[2].stride() and int(torch.count_nonzero(value_zero[:, keys])) == 0
                        assert not bool(torch.signbit(value_zero[:, keys]).any())
                        assert exact(value_zero[:, retained], args[2][:, retained], torch)
                        zero_witness = value_zero.clone(memory_format=torch.preserve_format)
                        self.counts['value_zero_dispatch'] += 1
                        zv = self.original(args[0], args[1], value_zero, **dict(kwargs, attn_mask=self.gpu_masks['all_allowed']))
                        assert exact(value_zero, zero_witness, torch), 'Value-zero dispatch mutated its input'
                        event.update(extra_value_zero_calls=1, retained187_V_byte_exact=True,
                            future200_V_zero_with_positive_zero_bits=True, value_zero_input_meta=self.tensor_meta(value_zero))
                    if self.case_arm in ('allocation_only', 'hardmask'):
                        self.counts['cut_dispatch'] += 1
                        zk = self.original(*args, **dict(kwargs, attn_mask=self.gpu_masks['cut_future_current']))
                        unblocked = self.masks['cut_future_current']['unblocked_rows'].to(native.device)
                        assert exact(zk[:, unblocked], z0[:, unblocked], torch), 'Hardmask changed an unblocked query'
                        event.update(extra_cut_calls=1, all_unblocked_GEN_rows_vs_same_input_all_allowed_byte_exact=True)
                    for endpoint in (zv, zk):
                        assert endpoint is None or (endpoint.shape == native.shape and endpoint.dtype == native.dtype and bool(torch.isfinite(endpoint).all()))
                    if self.case_arm == 'allocation_only':
                        delta = zk[:, current].float() - zv[:, current].float()
                        arithmetic_fp32 = z0[:, current].float() + delta
                        selected = arithmetic_fp32.to(dtype=torch.bfloat16)
                        selected = torch.where(delta == 0, z0[:, current], selected)
                        merged[:, current] = selected
                        event.update(allocation_formula='BF16(FP32(Z0)+(FP32(Zk)-FP32(Zv)))',
                            zero_delta_Z0_bytes_retained=True, additive_pre_W_not_normalized_softmax=True)
                    else:
                        merged[:, current] = (zv if self.case_arm == 'value_zero' else zk)[:, current]
                assert exact(merged[:, action], z0[:, action], torch), 'Action16 background differs from same-input Z0'
                assert exact(merged[:, future], native[:, future], torch), 'Future200 native bytes changed'
                self.counts['substituted_sites'] += 1
            else:
                merged = native
            assert bool(torch.isfinite(merged).all()) and merged.dtype == native.dtype
            for value, before in zip(args, witnesses):
                assert exact(value, before, torch), 'Official calls mutated actual Q/K/V'
            assert exact(rng[0], torch.get_rng_state(), torch) and exact(rng[1], torch.cuda.get_rng_state(args[0].device), torch)
            event.update(substituted_query_rows=0 if self.case_arm == 'native' else 66,
                selected_current_rows=50, selected_action_rows=16, future200_native_bytes_preserved=True,
                UND_dispatch_return_unmodified=True, original_QKV_bytes_preserved=True, RNG_bytes_preserved=True)
            if capture:
                endpoints = {name: None if value is None else self.runtime.cpu(value[:, current]) for name, value in
                    (('Znative', native), ('Z0', z0), ('Zv', zv), ('Zk', zk), ('Zreal', merged), ('Zclone', cloned))}
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
                    Qcurrent50=self.runtime.cpu(args[0][:, current]), Kall=self.runtime.cpu(args[1]), Vall=self.runtime.cpu(args[2]),
                    endpoints=endpoints, missing_endpoints=[name for name, value in endpoints.items() if value is None],
                    FP32allocation=self.runtime.cpu(arithmetic_fp32) if self.case_arm == 'allocation_only' else None,
                    FP32arithmetic_sham=self.runtime.cpu(arithmetic_fp32) if self.case_arm == 'arithmetic_sham' else None,
                    original_QKV_metadata=[self.tensor_meta(value) for value in args],
                    endpoint_metadata={name: self.tensor_meta(value) for name, value in endpoints.items()},
                    RNG_before=dict(cpu=rng[0], cuda=rng[1]), RNG_after=dict(cpu=torch.get_rng_state().clone(),
                        cuda=torch.cuda.get_rng_state(args[0].device).clone()), actual_QKV_bytes_unchanged=True,
                    headwise_effective_vs_native_or_Z0=headwise, event=dict(event), scope=SCOPE)
                if arithmetic_fp32 is not None:
                    intended = arithmetic_fp32.detach().cpu().double()
                    self.capture_pending['BF16_quantization_error'] = dict(
                        delta_l2=float((endpoints['Zreal'].double() - intended).norm()),
                        delta_maxabs=float((endpoints['Zreal'].double() - intended).abs().max()))
                    self.capture_pending['FP32_vs_FP64_formula_error'] = frozen.discrepancy(
                        arithmetic_fp32.detach().cpu().double(),
                        endpoints['Z0'].double() + (endpoints['Z0'].double() - endpoints['Z0'].double() if self.case_arm == 'arithmetic_sham'
                            else endpoints['Zk'].double() - endpoints['Zv'].double()), torch)
            self.events.append(event)
            self.pending = merged.squeeze(0).flatten(-2, -1)
            return merged

        def before_W(self, layer, module, args):
            super().before_W(layer, module, args)
            self.events[-1]['actual_pre_W_metadata'] = self.tensor_meta(args[0])
            if self.capture_pending is not None:
                data = self.capture_pending
                current = self.index['current_rows'].to(args[0].device)
                actual = args[0][current].reshape(1, 50, 32, 128)
                assert frozen.byte_exact(self.runtime.cpu(actual), data['endpoints']['Zreal'], self.torch)
                data.update(actual_pre_W_current_bytes_equal_Zreal=True,
                    actual_pre_W_full266_metadata=self.tensor_meta(args[0]), original_full_projection_calls_at_site=1)
                path = self.folder / self.events[-1]['capture_file']
                path.parent.mkdir(exist_ok=True)
                assert not path.exists()
                self.torch.save(data, path)
                size = path.stat().st_size
                assert size < 100_000_000, 'A selected-site capture exceeded the 100MB bound'
                self.capture_files.append(dict(step=self.step, layer_zero_based=layer, file=str(path.relative_to(self.folder)),
                    sha256=sha(path), bytes=size, captured_endpoints=[name for name, value in data['endpoints'].items() if value is not None],
                    missing_endpoints=data['missing_endpoints']))
                self.capture_pending = None

        def report(self, record):
            assert self.restored and not self.handles and not self.in_model and self.active is None and self.capture_pending is None
            assert self.counts == expected_counts(self.case_arm)
            assert len(self.calls) == len(self.files) == len(self.outputs) == 30 and len(self.capture_files) == 135
            assert self.finite_boundary_checks == 37 * 30
            assert [(item['step'], item['layer_zero_based']) for item in self.capture_files] == [(t, l) for t in range(15) for l in range(9)]
            self.runtime.require_exact(self.torch.stack(self.outputs), record['action_velocity'], 'actual model action tuple at all30 steps')
            assert self.reader.settings(self.torch, self.model.layers[0].self_attn.processor) == self.settings
            manifest = dict(state='complete', case_arm=self.case_arm, window=WINDOW, sites=135, files=self.capture_files,
                total_bytes=sum(item['bytes'] for item in self.capture_files), no_extra_dispatch_for_missing_endpoints=True,
                endpoint_scope='Actual current50 pre-W query-head axis, missing endpoints are null, Q/K/V are same live site only.', scope=SCOPE)
            write(self.folder / 'site-captures.json', manifest, exclusive=True)
            return dict(state='complete', case_arm=self.case_arm, selected_window=self.window, counts=self.counts,
                boundary_files=self.files, indexes_and_masks_sha256=sha(self.folder / 'indexes-and-masks.pt'),
                site_capture_manifest_sha256=sha(self.folder / 'site-captures.json'), captured_sites=135,
                original_full_266_row_projection_calls=1080, all_actual_dispatch_flatten_pre_W_byte_exact=True,
                all_masked_sites_same_union66=self.case_arm != 'native', future200_and_UND_native_dispatch_returns_preserved=True,
                action16_same_input_Z0_at_all_masked_sites=True, all30_actual_complete_model_kwargs_saved=True,
                all30_actual37_current_action_boundaries_finite=True,
                arithmetic_sham_all_selected_sites_byte_exact_Z0=True if self.case_arm == 'arithmetic_sham' else None,
                hardmask_unblocked_queries_byte_exact_Z0=True if self.case_arm in ('allocation_only', 'hardmask') else None,
                no_unselected_Zv_vs_Z0_equality_claim=True, processor_recompute_calls=0,
                all_owned_hooks_removed=True, original_dispatch_symbol_restored=True, numerical_sham_is_native_noop=False, scope=SCOPE)
    return ValueObserver


def runtime_contract(baseline, context, runtime):
    """Repeat the accepted runtime/backend gates; never select a new kernel."""
    torch, model = runtime.torch, runtime.pipe.transformer
    frozen, paths, anchors = context['modules']['frozen'], context['paths'], context['source_evidence']['anchors']
    assert not model.training and not model.is_cache_enabled and not torch.is_grad_enabled()
    assert sha(Path(inspect.getfile(type(model)))) == anchors['upstream']['actual_transformer_sha256']
    assert sha(Path(inspect.getfile(type(runtime.pipe)))) == anchors['upstream']['actual_pipeline_source_sha256']
    components, scan, reader = (load('value_mechanism_' + key, paths[key]) for key in ('components', 'metrics', 'reader'))
    processor_module = inspect.getmodule(type(model.layers[0].self_attn.processor))
    dispatch = processor_module.dispatch_attention_fn
    assert sha(Path(inspect.getfile(dispatch))) == frozen.DISPATCH_SHA
    assert hashlib.sha256(inspect.getsource(dispatch).encode()).hexdigest() == read(baseline / 'target-reader-inputs/provenance.json')['dispatch_function_source_sha256']
    dispatch_module = importlib.import_module('diffusers.models.attention_dispatch')
    assert dispatch_module.dispatch_attention_fn is dispatch
    registry = dispatch_module._AttentionBackendRegistry
    backend, backend_fn = registry.get_active_backend()
    supported, reference = registry._supported_arg_names[backend], anchors['feasibility_actual_registry']
    assert 'attn_mask' in supported and backend.value == reference['registry_backend'] and backend.value != '_native_flash'
    assert backend_fn.__qualname__ == reference['registry_backend_function']
    assert hashlib.sha256(inspect.getsource(backend_fn).encode()).hexdigest() == reference['registry_backend_function_source_sha256']
    assert sorted(supported) == reference['registry_supported_arguments']
    settings = reader.settings(torch, model.layers[0].self_attn.processor)
    for key, value in reference['settings'].items():
        assert settings[key] == value
    for block in model.layers:
        assert block.self_attn.processor._parallel_config is block.self_attn.processor._attention_backend is None
        assert reader.settings(torch, block.self_attn.processor) == settings
    return components, scan, reader, processor_module, dispatch, dict(registry_backend=backend.value,
        registry_backend_function=backend_fn.__qualname__, registry_supported_arguments=sorted(supported),
        registry_backend_function_source_sha256=hashlib.sha256(inspect.getsource(backend_fn).encode()).hexdigest(), backend_settings=settings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Original accepted BASE, never a new mechanism stage')
    parser.add_argument('--check-source-contract', action='store_true', help='Read accepted source contracts only; no output or model')
    args = parser.parse_args()
    baseline = args.output.resolve()
    context = source_contract(baseline)
    if args.check_source_contract:
        print(json.dumps(dict(state='source_contract_verified', image=str(context['image']),
            accepted_q0_references=list(context['old_q0_references']), expected_counts=expected_counts()), indent=2))
        return
    out, paths, contract = context['out'], context['paths'], context['frozen_contract']
    out.mkdir(exist_ok=False)
    protocol = dict(plan=[dict(case=label, vision_noise_source_seed=v, action_noise_source_seed=195, runtime_rng_seed=195,
        arm=arm, window=None if arm in ('native', 'all_allowed') else WINDOW) for label, v, arm in PLAN],
        trial_order=[label for label, _, _ in PLAN], prompt=PROMPT, shift_cm=12, image_path=str(context['image']),
        image_sha256=sha(context['image']), primary_hypothesis=PRIMARY, selected_window_zero_based_inclusive=WINDOW,
        masked_background='Every nonnative site returns union66 Z0 except selected current50; action16=Z0, future200=native, UND=native.',
        allocation_formula='BF16(FP32(Z0)+(FP32(Zk)-FP32(Zv))); exactly zero delta retains original Z0 bytes.',
        value_zero_rule='Same Q/K, full387 key set and same all-allowed mask; only future200 V entries become positive zero.',
        arithmetic_sham_rule='Same-valued full-shape V clone official dispatch must equal Z0; FP32 Z0+(Z0-Z0), single BF16 cast, preserve zero-delta Z0 bytes.',
        q0_predictions=12, model_forwards=360, expected_counts=expected_counts(), total_official_dispatch_calls=38070,
        original_dispatch_calls=25920, extra_all_allowed_dispatch_calls=10800, selected_extra_dispatch_calls=1350,
        selected_sites_per_case=135, captures_per_case=135, total_site_captures=1620, max_capture_file_bytes=100_000_000,
        capture_shapes=dict(Qcurrent50=[1, 50, 32, 128], Kall=[1, 387, 8, 128], Vall=[1, 387, 8, 128],
            current_endpoint=[1, 50, 32, 128]), missing_endpoint_rule='Explicit null; no extra call solely for captures.',
        actual_model_kwargs_and_boundaries_saved_steps=list(range(30)), offline_reconstruction_is_analysis_only=True,
        later_physics_preregistered=dict(trial_order=[label for label, _, _ in PLAN], cases=12, cached_q0=12,
            later_noise_base=195, q1_to_q7_seeds=list(range(196, 203)), later_predictions=84,
            later_model_forwards=2520, total_two_stage_model_forwards=2880, no_q0_score_or_rank_selection=True),
        physics_calls=0, no_training=True, no_residual_compensation=True, scope=SCOPE)
    write(out / 'protocol.json', protocol, exclusive=True)
    write(out / 'primary-prediction.json', dict(status='not_evaluated_q0_only', **PRIMARY, scope=SCOPE), exclusive=True)
    write(out / 'sources.json', dict(source_evidence=context['source_evidence'], original_sources=context['original_noise_sources'],
        frozen_contract=contract), exclusive=True)
    started, completed, rows, runtime, observer, noise = time.perf_counter(), [], [], None, None, None
    counts = {key: 0 for key in expected_counts()}

    def progress(stage, **values):
        value = dict(stage=stage, completed=completed, completed_model_forwards=30 * len(completed), actual_counts=counts,
            elapsed_s=time.perf_counter() - started, **values)
        write(out / 'progress.json', value)
        print('[FUTURE-VALUE-MECHANISM] ' + json.dumps(value), flush=True)

    try:
        os.environ['HF_HUB_OFFLINE'] = '1'
        base, factory = load('value_mechanism_runtime', paths['normal_runtime']), load('value_mechanism_factory', paths['factory'])
        base.TASKS['milk'] = (7, 'pick_up_the_milk_and_place_it_in_the_basket', PROMPT, 'milk_1')
        factory.OUT = out
        assert sha(factory.SCHEDULER) == contract['frozen_files']['scheduler']['sha256']
        progress('loading_model')
        runtime = base.NormalRuntime(factory)
        torch, model = runtime.torch, runtime.pipe.transformer
        components, scan, reader, processor_module, dispatch, backend = runtime_contract(baseline, context, runtime)
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
            accepted_layer_group_source_sha256=GROUP_SHA, accepted_fullcut_source_sha256=context['modules']['groups'].BASE_PROBE_SHA,
            accepted_window_source_sha256=context['modules']['groups'].WINDOW_PROBE_SHA,
            noise_seam_source_sha256=context['modules']['frozen'].FACTOR_SHA, scheduler_sha256=sha(factory.SCHEDULER),
            actual_transformer_sha256=sha(Path(inspect.getfile(type(model)))),
            actual_pipeline_source_sha256=sha(Path(inspect.getfile(type(runtime.pipe)))),
            dispatch_source_sha256=sha(Path(inspect.getfile(dispatch))), dispatch_signature=str(inspect.signature(dispatch)),
            **backend, kernel_evidence_limit='Registry/settings recorded; no selected-kernel profiler observation.',
            library_edits=False, training=False, physics_calls=0, scope=SCOPE)
        write(out / 'provenance.json', provenance, exclusive=True)
        Observer = observer_class(context['modules']['windows'], context['modules']['frozen'])
        records = {}
        for label, v, arm in PLAN:
            if arm not in ('native', 'all_allowed'):
                assert completed[:4] == [item[0] for item in PLAN[:4]]
            directory = out / label
            progress('fresh_q0', case=label)
            observer = Observer(runtime, components, scan, reader, caches[195], clean, references[f'V{v}_A195_native'],
                'native' if arm == 'native' else 'all_allowed', directory, case_arm=arm)
            noise = context['modules']['factor'].InitialNoiseSources(runtime, scan, originals, v, 195)
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
                                actual_counts=observer.counts), directory / 'partial-forward.pt')
            assert runtime.cm.randn_tensor is native_random and processor_module.dispatch_attention_fn is dispatch
            scan.validate_record(record, torch)
            assert metadata['model_calls'] == 30 and metadata['prepare_calls'] == 1 and metadata['seed'] == 195
            assert metadata['input_png_sha256'] == sha(context['image']) and metadata['prompt'] == PROMPT and metadata['task_index'] == 7
            report = observer.report(record)
            prepared = context['modules']['factor'].prepared_contract(record, originals, v, 195, runtime,
                SimpleNamespace(first=observer.first, calls=observer.calls, handle=None))
            noise_report = noise.report(record)
            ref_arm = 'all_allowed' if arm == 'arithmetic_sham' else 'early_l1_9' if arm == 'hardmask' else arm
            reference_label = f'V{v}_A195_{ref_arm}' if ref_arm in ('native', 'all_allowed', 'early_l1_9') else None
            if reference_label:
                reference = context['old_q0_references'][reference_label]
                control = context['modules']['windows'].saved_control_checks(directory, Path(reference['folder']), record, references[reference_label], runtime)
            else:
                sham_label = f'V{v}_A195_all_allowed'
                control = context['modules']['windows'].prefix_checks(directory, out / sham_label, record, records[sham_label], WINDOW, runtime)
            control['reference_case'] = reference_label
            write(directory / 'control-checks.json', control, exclusive=True)
            write(directory / 'observer-report.json', report, exclusive=True)
            write(directory / 'noise-audit.json', dict(noise=noise_report, preparation=prepared), exclusive=True)
            metadata.update(case=label, arm=arm, query=0, shift_cm=12, vision_noise_source_seed=v, action_noise_source_seed=195,
                runtime_rng_seed=195, window_zero_based_inclusive=observer.window, common_union66_all_allowed_background=arm != 'native',
                producer_script_sha256=provenance['script_sha256'], inherited_group_source_sha256=GROUP_SHA,
                inherited_fullcut_source_sha256=provenance['accepted_fullcut_source_sha256'], control_reference_case=reference_label,
                entire_accepted_control_record_exact=reference_label is not None, captured_sites=135,
                site_capture_manifest_sha256=report['site_capture_manifest_sha256'], primary_prediction_case=arm == 'value_zero',
                allocation_is_additive_pre_W_counterfactual=arm == 'allocation_only', physical_prediction_evaluated=False, scope=SCOPE)
            write(directory / 'metadata.json', metadata)
            assert read(directory / 'normalized_actions.json') == record['actions'].tolist()
            rows.append(dict(case=label, arm=arm, window=observer.window, vision_noise_source_seed=v, action_noise_source_seed=195,
                runtime_rng_seed=195, fresh_model_forwards=30, actual_dispatch_counts=observer.counts, control_reference_case=reference_label,
                existing_entire_control_record_exact=reference_label is not None,
                final_normalized_actions=record['actions'].tolist(), physical_prediction_evaluated=False,
                files_sha256={name: sha(directory / name) for name in FILES}, boundary_files=observer.files,
                site_capture_manifest_sha256=report['site_capture_manifest_sha256']))
            records[label] = record
            completed.append(label)
            progress('case_complete', case=label)
            observer = noise = None
        for v in (195, 198):
            for label, actual_v, _ in PLAN:
                if v == actual_v:
                    runtime.require_exact(records[label]['pure_noise'], records[f'V{v}_A195_native']['pure_noise'], 'within-V six whole returned noise pairs')
        runtime.require_exact(records['V195_A195_native']['pure_noise'][1], records['V198_A195_native']['pure_noise'][1], 'cross-V A195 only')
        assert counts == expected_counts() and sum(counts[key] for key in DISPATCH_KEYS) == 38070
        assert completed == [item[0] for item in PLAN] and not checker._handles and checker._kind is None
        assert runtime.cm.randn_tensor is native_random and processor_module.dispatch_attention_fn is dispatch
        write(out / 'results.json', dict(state='complete', cases=rows, fresh_q0_predictions=12, fresh_model_forwards=360,
            official_dispatch_counts=counts, total_official_dispatch_calls=38070, physics_calls=0,
            primary_prediction_sha256=sha(out / 'primary-prediction.json'), primary_physical_prediction_evaluated=False,
            preregistered_physics_cases=[item[0] for item in PLAN], scope=SCOPE), exclusive=True)
        write(out / 'complete.json', dict(state='complete', script_sha256=sha(Path(__file__)),
            **{name.replace('-', '_') + '_sha256': sha(out / (name + '.json')) for name in ('results', 'provenance', 'protocol', 'sources', 'primary-prediction')},
            fresh_q0_predictions=12, fresh_model_forwards=360, official_dispatch_counts=counts,
            total_official_dispatch_calls=38070, original_dispatch_calls=25920, extra_dispatch_calls=12150,
            original_randn_calls_consumed=24, returned_source_draws_observed=24, actual_selected_site_captures=1620,
            two_native_and_two_allallowed_entire_accepted_records_and_boundaries_exact=True,
            two_arithmetic_shams_entire_accepted_allallowed_records_and_boundaries_exact=True,
            two_hardmask_entire_accepted_early_l1_9_records_and_boundaries_exact=True,
            all10_masked_sites_same_union66_background=True, all360_actual_complete_model_kwargs_saved=True,
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
