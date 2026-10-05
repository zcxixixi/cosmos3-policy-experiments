"""Eight fixed X15/V198 L35 suffix-to-action q0 interventions; no physics.

Only absent BASE/instruction-interaction is created. All four arms perform
the same full-shape auxiliary GEN attention, W, postnorm and MLP calls.
The native MLP counterfactual uses this call's live block input and native
attention endpoint, never an old trajectory or donor MLP activation.
"""

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import time
import traceback


ROOT = Path('/home/current/work/cosmos3')
MATCHED_PROBE_SHA = '0446978830ac36978396fdad5b09df238b0fd6b3b425aaa351a9b40262f94df0'
MATCHED_RUNNER_SHA = '6b5ea13b8ce888aecac5cdb42b1a23ae9fec81972edf51f241f1e4429008217c'
MATCHED_ANALYZER_SHA = 'a0ff5732f7a9282eebf25d1c463cd4b9765eee786f5fc6eb5c5aa8049e1afcd7'
MATCHED_COMPLETE_SHA = '10c76eceaee3d8baeade3396fc65f55b71a3c286fdaef7cfe464aa0b1a3ac5b4'
MATCHED_PHYSICS_COMPLETE_SHA = '0468105a3110ac9d9bf83071d58963029f2bf1b9b78e35f0cab3541edb293e20'
MATCHED_ANALYSIS_COMPLETE_SHA = '219c43cba7e78a3da774d481cc7843e2b6a519c166cedd7490e21181f5e2b4a0'
STAGE = 'instruction-interaction'
PROMPTS = dict(milk_box='pick up the milk box and place it in the basket',
               cream_cheese='pick up the cream cheese and place it in the basket')
TARGETS = dict(milk_box='milk_1', cream_cheese='cream_cheese_1')
ARMS = ('live_self', 'suffix_joint', 'suffix_joint_native_mlp', 'matched_random')
DIRECTIONS = (('milk_box', 'cream_cheese'), ('cream_cheese', 'milk_box'))
PLAN = tuple((f'x15_{recipient}_from_{donor}_V198_A195_{arm}', recipient, donor, arm)
             for arm in ARMS for recipient, donor in DIRECTIONS)
WINDOW = dict(steps=[0, 29], layers=[35, 35], action_query_rows=[250, 265],
              UND_suffix_keys=[48, 121])
MAX_BYTES = 350 * 1024 ** 2
PHYSICS_RESERVE_BYTES = 700 * 1024 ** 2
MIN_FREE_BYTES = MAX_BYTES + PHYSICS_RESERVE_BYTES + 128 * 1024 ** 2
RANDOM_TOLERANCE = .02
RANDOM_ITERATIONS = 32
FILES = ('states.pt', 'normalized_actions.json', 'metadata.json', 'noise-audit.pt',
         'noise-audit.json', 'observer-report.json', 'indexes-and-masks.pt',
         'control-checks.json', 'language-contract.pt', 'site-captures.json')
DISPATCH_KEYS = ('native_UND_dispatch', 'native_GEN_dispatch', 'aux_GEN_dispatch')
SCOPE = ('X15/V198 two natural recipient instructions with fixed A/runtime195. '
    'Only q0 L35 action16 receives a target-start-to-UND-end natural K/V suffix '
    'counterfactual while Q, other K/V and key count remain actual recipient values. '
    'The suffix is instruction context, not decoded object identity. The same-input '
    'native-MLP arm tests a controlled MLP contribution under swapped attention; '
    'there is no full attention-by-MLP factorial or unique-mediator claim. Random '
    'strength matches actual BF16 pre-W action L2 within 2%, not W-output strength. '
    'All eight physical trials and seven recipient-native feedback queries are '
    'required separately. No original milk-instruction 15cm repair, training, '
    'head scan or natural-trajectory MLP subtraction is claimed.')
PRIMARY = dict(name='H_L35_suffix_bidirectional_target_transfer',
    cases=[row[0] for row in PLAN if row[-1] == 'suffix_joint'],
    criterion='Both suffix_joint executions strictly select only the donor-named object.',
    all8_physics_required=True, live_self_entire_q0_and_physics_controls_required=True,
    all30_random_BF16_strength_gates_required=True, no_q0_ranking=True,
    MLP_secondary='Report J and same-live C in both directions; no full factorial decomposition.',
    random_secondary='If matched_random also transfers targets, semantic specificity is unsupported; '
        'failure to transfer does not remove unequal W-output-strength alternatives.')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, exclusive=False):
    with Path(path).open('x' if exclusive else 'w') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def disk_bytes(folder):
    return sum(path.stat().st_size for path in folder.rglob('*') if path.is_file())


def expected_counts(arm=None):
    if arm is None:
        rows = [expected_counts(row[-1]) for row in PLAN]
        return {key: sum(row[key] for row in rows) for key in rows[0]}
    assert arm in ARMS
    return dict(model_before=30, model_after=30, block_before=1080, block_after=1080,
        pre_W=1080, native_UND_dispatch=1080, native_GEN_dispatch=1080,
        aux_GEN_dispatch=30, aux_W=60, aux_W_native=30, aux_W_donor=30, aux_postnorm=30, aux_MLP=30,
        main_W=1080, main_postnorm=1080, main_MLP=1080,
        action_attention_substitutions=30, action_MLP_substitutions=30 if arm == 'suffix_joint_native_mlp' else 0)


def source_contract(baseline):
    """Read actual completed matched-name seals before any output/model creation."""
    pp, ep, ap = (ROOT / 'work' / name for name in
        ('probe_cosmos_milk_matched_names.py', 'run_cosmos_milk_matched_names.py',
         'analyze_cosmos_milk_matched_names.py'))
    assert (sha(pp), sha(ep), sha(ap)) == (MATCHED_PROBE_SHA, MATCHED_RUNNER_SHA, MATCHED_ANALYZER_SHA)
    matched, adapter, analyzer = (load('instruction_' + name, path) for name, path in
        (('matched_probe', pp), ('matched_runner', ep), ('matched_analyzer', ap)))
    accepted = analyzer.seal(baseline)  # The completed real authority; no runtime/model construction.
    context = matched.source_contract(baseline)
    producer, execution, analysis = baseline / matched.STAGE, baseline / adapter.STAGE, baseline / analyzer.STAGE
    assert sha(producer / 'complete.json') == MATCHED_COMPLETE_SHA
    assert sha(execution / 'complete.json') == MATCHED_PHYSICS_COMPLETE_SHA
    assert sha(analysis / 'complete.json') == MATCHED_ANALYSIS_COMPLETE_SHA
    assert not (analysis / 'failed.json').exists()
    ac, ar, av = (read(analysis / (name + '.json')) for name in ('complete', 'analysis', 'provenance'))
    assert ac['state'] == ar['state'] == 'complete' and ac['script_sha256'] == MATCHED_ANALYZER_SHA
    assert ac['actual_native_suffix_files'] == ar['actual_native_suffix_files'] == 240
    for name, expected in ac['files_sha256'].items():
        assert sha(analysis / name) == expected
    for name, expected in av['producer_documents_sha256'].items():
        assert sha(producer / (name + '.json')) == expected
    for name, expected in av['execution_documents_sha256'].items():
        assert sha(execution / (name + '.json')) == expected
    assert all(ar[key] == 0 for key in ('model_calls', 'official_dispatch_calls', 'solver_calls', 'physics_calls'))
    rows = {row['case']: row for row in accepted['rows']}
    physical = {row['case']: row for row in accepted['comparisons']['cases']}
    references = {}
    for goal in PROMPTS:
        label = f'x15_{goal}_V198_A195_native'
        row, folder, behavior = rows[label], producer / label, physical[label]
        assert row['scene'] == 'x15' and row['goal'] == goal and row['arm'] == 'native'
        assert row['actual_und_len'] == 122 and row['actual_target_positions'] == [48, 49]
        assert behavior['selected_objects'] == [TARGETS[goal]] and behavior['strict_requested_target_selected'] is True
        assert behavior['first_selection_window_start'] == dict(milk_box=64, cream_cheese=67)[goal]
        assert len(row['dispatch_files']) == len(row['boundary_files']) == 30
        references[goal] = dict(case=label, folder=str(folder), files_sha256=row['files_sha256'],
            boundary_files=row['boundary_files'], dispatch_files=row['dispatch_files'],
            strict_selected_objects=behavior['selected_objects'], first_selection_window_start=behavior['first_selection_window_start'],
            physical_case_files_sha256=accepted['controls']['actual_case_files_sha256'][label])
    context['modules']['matched'] = matched
    context['modules'].update(matched_adapter=accepted['adapter'], simulator=accepted['simulator'],
        execution_helper=accepted['helper'], cross=accepted['cross'])
    context['matched_references'] = references
    context['matched_frozen'] = accepted['frozen']
    context['source_evidence']['instruction_interaction_sources'] = dict(
        producer_script_sha256=MATCHED_PROBE_SHA, execution_script_sha256=MATCHED_RUNNER_SHA,
        analyzer_script_sha256=MATCHED_ANALYZER_SHA, producer_complete_sha256=MATCHED_COMPLETE_SHA,
        execution_complete_sha256=MATCHED_PHYSICS_COMPLETE_SHA, analysis_complete_sha256=MATCHED_ANALYSIS_COMPLETE_SHA,
        actual_natural_references=references,
        source_scope='Only X15/V198 two successful names are used. V195 is not an eligible natural donor.')
    context['out'] = baseline / STAGE
    return context


def observer_class(matched, frozen, language):
    Parent = matched.observer_class(frozen, language)

    class InstructionObserver(Parent):
        def __init__(self, *args, recipient, donor, direction_index, donor_reference, recipient_reference, **kwargs):
            super().__init__(*args, **kwargs)
            self.recipient, self.donor, self.direction_index = recipient, donor, direction_index
            self.donor_reference, self.recipient_reference = donor_reference, recipient_reference
            self.counts = dict.fromkeys(expected_counts(self.arm), 0)
            self.aux_phase, self.live_x, self.aux, self.site = None, None, None, None
            self.site_files, self.random_reports = [], []
            self.hidden_metadata, self.full_live_checks = [], 0
            self.prefix_boundary = None

        def begin(self):
            super().begin()
            try:
                for layer, block in enumerate(self.model.layers):
                    self.handles.append(block.self_attn.to_add_out.register_forward_hook(
                        lambda module, args, output, layer=layer: self.after_W(layer, args, output)))
                    self.handles.append(block.post_attention_layernorm_moe_gen.register_forward_pre_hook(
                        lambda module, args, layer=layer: self.before_postnorm(layer, args)))
                    self.handles.append(block.post_attention_layernorm_moe_gen.register_forward_hook(
                        lambda module, args, output, layer=layer: self.after_postnorm(layer, output)))
                    self.handles.append(block.mlp_moe_gen.register_forward_hook(
                        lambda module, args, output, layer=layer: self.after_MLP(layer, args, output)))
            except BaseException:
                self.reset()
                raise

        def observe_boundary(self, und, gen):
            assert und.shape == (122, 4096) and gen.shape == (266, 4096)
            assert und.dtype == gen.dtype == self.torch.bfloat16
            current, action = gen[:50], gen[250:266]
            assert all(bool(self.torch.isfinite(item).all()) for item in (und, gen, current, action))
            self.hidden_metadata.append(dict(boundary=len(self.hidden_metadata),
                und=self.meta(und), current=self.meta(current), action=self.meta(action)))
            self.finite_boundary_checks += 1

        def before_block(self, layer, module, args):
            super().before_block(layer, module, args)
            if layer == 35:
                assert self.live_x is self.aux is self.site is None
                self.live_x = args[1].clone(memory_format=self.torch.preserve_format)

        def before_model(self, module, args, kwargs):
            assert self.aux_phase is self.live_x is self.aux is self.site is None
            super().before_model(module, args, kwargs)
            assert self.index['und_len'] == 122 and self.index['full_joint_length'] == 388
            self.runtime.require_exact(self.index['action_rows'], self.torch.arange(250, 266), 'actual action16 rows')
            assert self.language['target_positions'] == [48, 49]
            if self.step == 0:
                item = self.recipient_reference['boundary_files'][0]
                path = Path(self.recipient_reference['folder']) / item['file']
                assert sha(path) == item['sha256']
                self.prefix_boundary = self.torch.load(path, map_location='cpu', weights_only=True, mmap=True)

        def before_W(self, layer, module, args):
            if self.aux_phase is not None:
                assert self.aux_phase in ('W_native', 'W_donor') and layer == 35 and len(args) == 1
                assert args[0].shape == (266, 4096) and args[0].dtype == self.torch.bfloat16
                self.counts['aux_W'] += 1
                self.counts['aux_W_native' if self.aux_phase == 'W_native' else 'aux_W_donor'] += 1
                return
            frozen.FutureReadCuts.before_W(self, layer, module, args)
            if layer == 35:
                self.site['actual_pre_W_full_metadata'] = self.meta(args[0])
                self.runtime.require_exact(self.runtime.cpu(args[0][250:266]),
                    self.site['Zselected_explicit_action16'].squeeze(0).flatten(-2, -1), 'actual action16 pre-W flatten')

        def after_W(self, layer, args, output):
            assert output.shape == (266, 4096) and output.dtype == self.torch.bfloat16
            assert bool(self.torch.isfinite(output).all())
            if self.aux_phase is not None:
                assert self.aux_phase in ('W_native', 'W_donor') and layer == 35
                return
            assert self.active == layer
            self.counts['main_W'] += 1
            if layer == 35:
                self.site['W1_explicit_action16'] = self.runtime.cpu(output[250:266])
                self.site['actual_endpoint_metadata']['W1_explicit_action16'] = self.meta(output[250:266])
                self.site['W1_full_metadata'] = self.meta(output)
                self.main_W_live = output
                assert frozen.byte_exact(output[:250], self.aux['W0'][:250], self.torch)
                if self.arm == 'live_self':
                    assert frozen.byte_exact(output, self.aux['W0'], self.torch)
                    self.runtime.require_exact(self.site['W1_explicit_action16'], self.site['W0_explicit_action16'], 'Actual self W1 storage reuse')
                    self.site['W1_explicit_action16'] = self.site['W0_explicit_action16']

        def before_postnorm(self, layer, args):
            assert len(args) == 1 and args[0].shape == (266, 4096) and args[0].dtype == self.torch.bfloat16
            if self.aux_phase is not None:
                assert self.aux_phase == 'postnorm' and layer == 35
                self.counts['aux_postnorm'] += 1
                assert frozen.byte_exact(args[0], self.aux['r0'], self.torch)
                return
            self.counts['main_postnorm'] += 1
            assert self.active == layer
            if layer == 35:
                actual = args[0]
                assert frozen.byte_exact(actual, self.live_x + self.main_W_live, self.torch)
                assert frozen.byte_exact(actual[:250], self.aux['r0'][:250], self.torch)
                if self.arm == 'live_self':
                    assert frozen.byte_exact(actual, self.aux['r0'], self.torch)
                self.site['r1_full_metadata'], self.site['r1_explicit_action16_metadata'] = self.meta(actual), self.meta(actual[250:266])
                self.main_r_live = actual

        def after_postnorm(self, layer, output):
            assert output.shape == (266, 4096) and output.dtype == self.torch.bfloat16
            assert bool(self.torch.isfinite(output).all())
            if self.aux_phase is not None:
                assert self.aux_phase == 'postnorm' and layer == 35
                return
            assert self.active == layer
            if layer == 35:
                self.site['main_postnorm_full_metadata'] = self.meta(output)
                assert frozen.byte_exact(output[:250], self.aux['norm0'][:250], self.torch)
                if self.arm == 'live_self':
                    assert frozen.byte_exact(output, self.aux['norm0'], self.torch)

        def after_MLP(self, layer, args, output):
            assert len(args) == 1 and output.shape == (266, 4096) and output.dtype == self.torch.bfloat16
            assert bool(self.torch.isfinite(output).all())
            if self.aux_phase is not None:
                assert self.aux_phase == 'MLP' and layer == 35
                self.counts['aux_MLP'] += 1
                return
            self.counts['main_MLP'] += 1
            assert self.active == layer
            if layer != 35:
                return
            assert frozen.byte_exact(output[:250], self.aux['m0'][:250], self.torch)
            if self.arm == 'live_self':
                assert frozen.byte_exact(output, self.aux['m0'], self.torch)
            self.site['m1_explicit_action16'] = self.runtime.cpu(output[250:266])
            self.site['actual_endpoint_metadata']['m1_explicit_action16'] = self.meta(output[250:266])
            if self.arm == 'live_self':
                self.runtime.require_exact(self.site['m1_explicit_action16'], self.site['m0_explicit_action16'], 'Actual self m1 storage reuse')
                self.site['m1_explicit_action16'] = self.site['m0_explicit_action16']
            self.site['m1_full_metadata'] = self.meta(output)
            self.site['main_MLP_input_full_metadata'] = self.meta(args[0])
            selected = output
            if self.arm == 'suffix_joint_native_mlp':
                selected = output.clone(memory_format=self.torch.preserve_format)
                selected[250:266] = self.aux['m0'][250:266]
                self.counts['action_MLP_substitutions'] += 1
            assert frozen.byte_exact(selected[:250], output[:250], self.torch)
            self.site['selected_MLP_full_metadata'] = self.meta(selected)
            self.site['selected_MLP_explicit_action16_metadata'] = self.meta(selected[250:266])
            self.site['r1_plus_m0_explicit_action16_metadata'] = self.meta(self.main_r_live[250:266] + self.aux['m0'][250:266])
            self.site['r1_plus_m1_explicit_action16_metadata'] = self.meta(self.main_r_live[250:266] + output[250:266])
            self.selected_m_live = selected
            return selected

        def random_action(self, base, donor):
            torch = self.torch
            base_cpu, donor_cpu = self.runtime.cpu(base), self.runtime.cpu(donor)
            natural = donor_cpu.float() - base_cpu.float()
            target = float((donor_cpu.double() - base_cpu.double()).norm())
            generator = torch.Generator(device='cpu')
            seed = 104729 + 1000 * self.direction_index + self.step
            generator.manual_seed(seed)
            state_before = generator.get_state().clone()
            permutation = torch.randperm(128, generator=generator)
            signs = (torch.randint(0, 2, (1, 16, 32, 128), generator=generator, dtype=torch.int64) * 2 - 1).float()
            scrambled = natural.index_select(-1, permutation) * signs
            assert bool(torch.isfinite(scrambled).all())

            def candidate(alpha):
                delta = scrambled * alpha
                result = (base_cpu.float() + delta).to(torch.bfloat16)
                result = torch.where(delta == 0, base_cpu, result)
                realized = float((result.double() - base_cpu.double()).norm())
                error = 0. if target == realized == 0 else abs(realized / target - 1.)
                return error, result, delta, realized

            attempts = []
            if target == 0:
                best, alpha = candidate(0.), 0.
            else:
                low, high, alpha = 0., 4., 1.
                best = candidate(alpha)
                attempts.append(dict(alpha=alpha, realized_l2=best[3], relative_error=best[0]))
                best_alpha = alpha
                for _ in range(RANDOM_ITERATIONS - 1):
                    if best[0] <= RANDOM_TOLERANCE:
                        break
                    alpha = (low + high) / 2
                    item = candidate(alpha)
                    attempts.append(dict(alpha=alpha, realized_l2=item[3], relative_error=item[0]))
                    if item[0] < best[0]:
                        best, best_alpha = item, alpha
                    if item[3] < target:
                        low = alpha
                    else:
                        high = alpha
                alpha = best_alpha
            error, result, delta, realized = best
            report = dict(seed=seed, alpha=alpha, fit_iterations=len(attempts), fit_attempts=attempts,
                tolerance_relative=RANDOM_TOLERANCE, target_l2_fp64=target, realized_l2_fp64=realized,
                relative_error=error, strength_gate_pass=error <= RANDOM_TOLERANCE,
                per_row_head_target_rms=(donor_cpu.double() - base_cpu.double()).square().mean(-1).sqrt(),
                per_row_head_realized_rms=(result.double() - base_cpu.double()).square().mean(-1).sqrt(),
                generator_state_before=state_before, generator_state_after=generator.get_state().clone(),
                permutation=permutation, signs=signs.to(torch.int8), natural_FP32_metadata=self.meta(natural),
                scrambled_FP32_metadata=self.meta(scrambled), actual_delta_FP32_metadata=self.meta(delta),
                actual_BF16_result_metadata=self.meta(result), zero_target_exact_self=target == 0,
                matching_interface='Actual BF16 pre-W all-action16 L2 only; W output strength is not matched.')
            self.site['random'] = report
            self.site['random_actual_delta_FP32'] = delta
            self.random_reports.append({key: report[key] for key in ('seed', 'alpha', 'fit_iterations', 'target_l2_fp64',
                'realized_l2_fp64', 'relative_error', 'strength_gate_pass', 'zero_target_exact_self')})
            assert report['strength_gate_pass'], 'Frozen 2% actual BF16 pre-W random strength gate failed'
            assert len(attempts) <= RANDOM_ITERATIONS and bool(torch.isfinite(result).all())
            return result.to(base.device)

        def dispatch(self, *args, **kwargs):
            torch = self.torch
            assert self.in_model and self.active is not None and self.aux_phase is None
            assert len(args) == 3 and set(kwargs) == {'is_causal', 'enable_gqa', 'backend', 'parallel_config'}
            assert kwargs['enable_gqa'] is True and kwargs['backend'] is kwargs['parallel_config'] is None
            causal = kwargs['is_causal']
            assert type(causal) is bool and self.site_calls == ([] if causal else ['UND'])
            self.site_calls.append('UND' if causal else 'GEN')
            qlen, kvlen = (122, 122) if causal else (266, 388)
            assert args[0].shape == (1, qlen, 32, 128) and args[1].shape == args[2].shape == (1, kvlen, 8, 128)
            assert all(item.dtype == torch.bfloat16 and bool(torch.isfinite(item).all()) for item in args)
            witnesses = tuple(item.clone(memory_format=torch.preserve_format) for item in args)
            rng = dict(cpu=torch.get_rng_state().clone(), cuda=torch.cuda.get_rng_state(args[0].device).clone())
            self.counts['native_UND_dispatch' if causal else 'native_GEN_dispatch'] += 1
            native = self.original(*args, **kwargs)
            assert native.shape == (1, qlen, 32, 128) and native.dtype == torch.bfloat16 and bool(torch.isfinite(native).all())
            returned = native
            if not causal:
                event = dict(step=self.step, layer_zero_based=self.active, arm=self.arm,
                    original_GEN_dispatch_calls=1, original_UND_dispatch_calls=1, extra_all_allowed_calls=0,
                    extra_cut_calls=0, aux_GEN_dispatch_calls=0, to_add_out_shape=[266, 4096],
                    to_add_out_actual_calls=0, processor_recompute_calls=0)
                if self.active == 35:
                    assert self.site is self.aux is None and self.live_x is not None
                    donor_item = self.donor_reference['dispatch_files'][self.step]
                    donor_path = Path(self.donor_reference['folder']) / donor_item['file']
                    assert sha(donor_path) == donor_item['sha256']
                    saved = torch.load(donor_path, map_location='cpu', weights_only=True, mmap=True)
                    assert saved['step'] == self.step and saved['layer_zero_based'] == 35
                    assert saved['causal_suffix_raw_saved'] is True and saved['native_kwargs'] == kwargs
                    self.runtime.require_exact(saved['causal_suffix_key_indexes'], torch.arange(48, 122), 'actual donor suffix positions')
                    for key in ('Kcausal_suffix', 'Vcausal_suffix'):
                        assert saved[key].shape == (1, 74, 8, 128) and saved[key].dtype == torch.bfloat16
                        assert bool(torch.isfinite(saved[key]).all())
                        assert self.meta(saved[key])['sha256'] == saved['sparse_metadata'][key]['sha256']
                    k, v = (item.clone(memory_format=torch.preserve_format) for item in args[1:])
                    if self.arm != 'live_self':
                        k[:, 48:122], v[:, 48:122] = saved['Kcausal_suffix'].to(k.device), saved['Vcausal_suffix'].to(v.device)
                    assert frozen.byte_exact(k[:, :48], args[1][:, :48], torch) and frozen.byte_exact(k[:, 122:], args[1][:, 122:], torch)
                    assert frozen.byte_exact(v[:, :48], args[2][:, :48], torch) and frozen.byte_exact(v[:, 122:], args[2][:, 122:], torch)
                    aux_kv_witness = k.clone(memory_format=torch.preserve_format), v.clone(memory_format=torch.preserve_format)
                    self.counts['aux_GEN_dispatch'] += 1
                    zdonor = self.original(args[0], k, v, **kwargs)
                    assert zdonor.shape == native.shape and zdonor.dtype == native.dtype and bool(torch.isfinite(zdonor).all())
                    assert all(frozen.byte_exact(a, b, torch) for a, b in zip((k, v), aux_kv_witness))
                    if self.arm == 'live_self':
                        assert frozen.byte_exact(zdonor, native, torch), 'Live-clone official endpoint is not exact'
                    self.site = dict(step=self.step, layer_zero_based=35, recipient=self.recipient, donor=self.donor, arm=self.arm,
                        x_explicit_action16=self.runtime.cpu(self.live_x[250:266]),
                        Z0_explicit_action16=self.runtime.cpu(native[:, 250:266]),
                        Zdonor_explicit_action16=self.runtime.cpu(zdonor[:, 250:266]),
                        original_QKV_full_metadata=[self.meta(item) for item in args],
                        auxiliary_KV_full_metadata=[self.meta(k), self.meta(v)],
                        native_GEN_full_metadata=self.meta(native), auxiliary_GEN_full_metadata=self.meta(zdonor),
                        x_full_metadata=self.meta(self.live_x), actual_native_kwargs=dict(kwargs),
                        actual_endpoint_metadata=dict(x_explicit_action16=self.meta(self.live_x[250:266]),
                            Z0_explicit_action16=self.meta(native[:, 250:266]),
                            Zdonor_explicit_action16=self.meta(zdonor[:, 250:266])),
                        actual_indexes=self.index, donor_saved_file=dict(file=str(donor_path), sha256=donor_item['sha256']),
                        actual_live_suffix_metadata=[self.meta(item[:, 48:122]) for item in args[1:]],
                        actual_donor_suffix_metadata=[self.meta(saved[key]) for key in ('Kcausal_suffix', 'Vcausal_suffix')],
                        global_RNG_before=rng, random=None, random_actual_delta_FP32=None)
                    selected = self.random_action(native[:, 250:266], zdonor[:, 250:266]) if self.arm == 'matched_random' else zdonor[:, 250:266]
                    returned = native.clone(memory_format=torch.preserve_format)
                    returned[:, 250:266] = selected
                    self.site['Zselected_explicit_action16'] = self.runtime.cpu(selected)
                    self.site['actual_endpoint_metadata']['Zselected_explicit_action16'] = self.meta(selected)
                    if self.arm == 'live_self':
                        self.runtime.require_exact(self.site['Zdonor_explicit_action16'], self.site['Z0_explicit_action16'], 'Actual self Zdonor storage reuse')
                        self.runtime.require_exact(self.site['Zselected_explicit_action16'], self.site['Z0_explicit_action16'], 'Actual self Zselected storage reuse')
                        self.site['Zdonor_explicit_action16'] = self.site['Zselected_explicit_action16'] = self.site['Z0_explicit_action16']
                    elif self.arm in ('suffix_joint', 'suffix_joint_native_mlp'):
                        self.runtime.require_exact(self.site['Zselected_explicit_action16'], self.site['Zdonor_explicit_action16'], 'Actual donor Zselected storage reuse')
                        self.site['Zselected_explicit_action16'] = self.site['Zdonor_explicit_action16']
                    self.site['returned_GEN_full_metadata'] = self.meta(returned)
                    block = self.model.layers[35]
                    self.aux = {}
                    try:
                        self.aux_phase = 'W_native'
                        self.aux['W0'] = block.self_attn.to_add_out(native.squeeze(0).flatten(-2, -1))
                        self.aux_phase = 'W_donor'
                        self.aux['Wdonor'] = block.self_attn.to_add_out(zdonor.squeeze(0).flatten(-2, -1))
                        self.aux['r0'] = self.live_x + self.aux['W0']
                        self.aux_phase = 'postnorm'
                        self.aux['norm0'] = block.post_attention_layernorm_moe_gen(self.aux['r0'])
                        self.aux_phase = 'MLP'
                        self.aux['m0'] = block.mlp_moe_gen(self.aux['norm0'])
                    finally:
                        self.aux_phase = None
                    assert all(item.shape == (266, 4096) and item.dtype == torch.bfloat16 and bool(torch.isfinite(item).all()) for item in self.aux.values())
                    self.site.update(W0_explicit_action16=self.runtime.cpu(self.aux['W0'][250:266]),
                        m0_explicit_action16=self.runtime.cpu(self.aux['m0'][250:266]),
                        W0_full_metadata=self.meta(self.aux['W0']), r0_full_metadata=self.meta(self.aux['r0']),
                        Wdonor_full_metadata=self.meta(self.aux['Wdonor']),
                        Wdonor_explicit_action16_metadata=self.meta(self.aux['Wdonor'][250:266]),
                        r0_explicit_action16_metadata=self.meta(self.aux['r0'][250:266]),
                        native_postnorm_full_metadata=self.meta(self.aux['norm0']), m0_full_metadata=self.meta(self.aux['m0']))
                    self.site['actual_endpoint_metadata'].update(W0_explicit_action16=self.meta(self.aux['W0'][250:266]),
                        m0_explicit_action16=self.meta(self.aux['m0'][250:266]))
                    self.counts['action_attention_substitutions'] += 1
                    event['aux_GEN_dispatch_calls'] = 1
                assert frozen.byte_exact(returned[:, :250], native[:, :250], torch)
                event.update(substituted_query_rows=16 if self.active == 35 else 0,
                    current50_future200_native_bytes_preserved=True, UND_dispatch_return_unmodified=True)
                self.events.append(event)
                self.pending = returned.squeeze(0).flatten(-2, -1)
            for actual, witness in zip(args, witnesses):
                assert frozen.byte_exact(actual, witness, torch), 'Actual recipient QKV was changed'
            assert frozen.byte_exact(rng['cpu'], torch.get_rng_state(), torch)
            assert frozen.byte_exact(rng['cuda'], torch.cuda.get_rng_state(args[0].device), torch)
            if not causal and self.active == 35:
                self.site['global_RNG_after'] = dict(cpu=torch.get_rng_state().clone(), cuda=torch.cuda.get_rng_state(args[0].device).clone())
            assert bool(torch.isfinite(returned).all())
            return returned

        def after_block(self, layer, module, args, output):
            if layer == 35:
                assert self.active == 35 and type(output) is tuple and len(output) == 2
                assert frozen.byte_exact(args[1], self.live_x, self.torch), 'Actual block input changed during auxiliary calls'
                assert frozen.byte_exact(output[1], self.main_r_live + self.selected_m_live, self.torch)
                assert frozen.byte_exact(output[1][:250], (self.aux['r0'] + self.aux['m0'])[:250], self.torch)
                if self.arm == 'live_self':
                    assert frozen.byte_exact(output[1], self.aux['r0'] + self.aux['m0'], self.torch)
                x, w0, w1, m0, m1 = (self.site[key] for key in
                    ('x_explicit_action16', 'W0_explicit_action16', 'W1_explicit_action16', 'm0_explicit_action16', 'm1_explicit_action16'))
                r0_cpu, r1_cpu = x + w0, x + w1
                assert self.meta(r0_cpu)['sha256'] == self.site['r0_explicit_action16_metadata']['sha256']
                assert self.meta(r1_cpu)['sha256'] == self.site['r1_explicit_action16_metadata']['sha256']
                c0, c1 = r1_cpu + m0, r1_cpu + m1
                assert self.meta(c0)['sha256'] == self.site['r1_plus_m0_explicit_action16_metadata']['sha256']
                assert self.meta(c1)['sha256'] == self.site['r1_plus_m1_explicit_action16_metadata']['sha256']
                actual = self.runtime.cpu(output[1][250:266])
                self.runtime.require_exact(actual, c0 if self.arm == 'suffix_joint_native_mlp' else c1,
                    'Actual BF16 block action equals exactly ordered residual plus selected MLP')
                self.site['action_hidden'] = actual
                self.site['action_boundary_ids'] = [36]
                self.site['actual_postblock_full_metadata'] = self.meta(output[1])
                self.site['actual_postblock_explicit_action16_metadata'] = self.meta(output[1][250:266])
                self.site['two_actual_BF16_addition_reconstruction_gates'] = True
                delta_z = (self.site['Zselected_explicit_action16'].double() - self.site['Z0_explicit_action16'].double()).norm()
                donor_z = (self.site['Zdonor_explicit_action16'].double() - self.site['Z0_explicit_action16'].double()).norm()
                delta_w = (w1.double() - w0.double()).norm()
                donor_w = (self.aux['Wdonor'][250:266].double() - self.aux['W0'][250:266].double()).norm()
                if self.arm != 'matched_random':
                    assert frozen.byte_exact(self.main_W_live[250:266], self.aux['Wdonor'][250:266], self.torch)
                self.site['realized_effect_norms'] = dict(pre_W_selected_l2=float(delta_z), pre_W_donor_l2=float(donor_z),
                    pre_W_selected_to_donor_ratio=None if float(donor_z) == 0 else float(delta_z / donor_z),
                    W_selected_l2=float(delta_w), W_donor_l2=float(donor_w),
                    actualW_delta_ratio=None if float(donor_w) == 0 else float(delta_w / donor_w),
                    W_strength_scope='Same-call actual full266 W outputs; ratio is observed, not matched.')
                self.site['global_RNG_after_actual_block'] = dict(cpu=self.torch.get_rng_state().clone(),
                    cuda=self.torch.cuda.get_rng_state(output[1].device).clone())
                for key, actual_rng in self.site['global_RNG_after_actual_block'].items():
                    assert frozen.byte_exact(actual_rng, self.site['global_RNG_before'][key], self.torch)
                self.full_live_checks += 1
            super().after_block(layer, module, args, output)

        def after_model(self, module, args, output):
            assert self.in_model and self.active is None and self.next_layer == 36
            assert len(self.events) == 36 and len(self.hidden_metadata) == 37 and self.site is not None
            assert type(output) is tuple and len(output) == 3 and type(output[2]) is list and len(output[2]) == 1
            actual_output = self.runtime.cpu(output)
            assert output[2][0].shape == (16, 64) and bool(self.torch.isfinite(output[2][0]).all())
            if self.step == 0:
                assert self.prefix_boundary is not None
                for actual, old in zip(self.hidden_metadata[:36], self.prefix_boundary['hidden_boundary_metadata'][:36]):
                    assert actual['boundary'] == old['boundary']
                    for group in ('und', 'current', 'action'):
                        for key in ('shape', 'dtype', 'sha256'):
                            assert actual[group][key] == old[group][key], ('Actual unchanged t0 decoder prefix', group, key)
                # L35 only the action branch changes; its UND/current remain native.
                for group in ('und', 'current'):
                    for key in ('shape', 'dtype', 'sha256'):
                        assert self.hidden_metadata[36][group][key] == self.prefix_boundary['hidden_boundary_metadata'][36][group][key]
            if self.arm == 'live_self':
                assert self.legacy_boundary is not None
                assert language.tree_sha(actual_output, self.runtime) == self.legacy_boundary['model_output_sha256']
                for actual, old in zip(self.hidden_metadata, self.legacy_boundary['hidden_boundary_metadata']):
                    assert actual['boundary'] == old['boundary']
                    for group in ('und', 'current', 'action'):
                        for key in ('dtype', 'shape', 'sha256'):
                            assert actual[group][key] == old[group][key], ('Live self complete hidden byte evidence', self.step, group, key)
                old_action = self.legacy_boundary['action_hidden'][36] if self.step == 0 else self.legacy_boundary['action_hidden']
                self.runtime.require_exact(self.site['action_hidden'], old_action, 'Live self actual b36 action raw')
            frame = self.meta(self.actual_kwargs['vision_tokens'][0][0, :, 0])
            self.site.update(scene='x15', goal=self.recipient, case_arm=self.arm,
                current_hidden=None, und_hidden=None, actual_model_kwargs=None, model_output=None,
                actual_model_kwargs_sha256=language.tree_sha(self.actual_kwargs, self.runtime),
                model_output_sha256=language.tree_sha(actual_output, self.runtime),
                actual_model_action_tokens=self.actual_kwargs['action_tokens'][0],
                actual_model_action_output=actual_output[2][0],
                actual_model_action_timesteps=self.actual_kwargs['action_timesteps'],
                solver_sigma_from_same_frozen_schedule=self.source['sigmas'][self.step].clone(),
                actual_current_frame_metadata=frame, actual_current_frame_sha256=frame['sha256'],
                actual_pack=self.calls[-1], hidden_boundary_metadata=self.hidden_metadata, site_events=self.events,
                raw_scope=dict(explicit_action16='x/W0/W1/m0/m1/Z0/Zdonor/Zselected and actual postblock action only',
                    r0_r1_and_candidates='Live original SHA; reconstruct from saved actual x/W/m with two ordered BF16 adds',
                    full266_and_QKV='Original live full numerical SHA and exact in-call gates; no full raw arrays saved',
                    full_kwargs_and_tuple='Live full-tree SHA; first kwargs and complete30 NormalRuntime record saved separately',
                    random='Private generator state/permutation/sign and actual FP32 delta; no global RNG draw'),
                endpoint_scope='Zdonor is this recipient call with actual Q and donor suffix, not a saved donor-trajectory attention output.',
                storage_reuse_scope='Only after actual CPU byte equality: J/C Zselected aliases Zdonor; self Zdonor/Zselected aliases Z0 and W1/m1 alias W0/m0. Actual per-endpoint metadata is retained.',
                boundary_convention='Action boundary36 after L35 attention/MLP residual adds, before final norm.')
            path = self.folder / f'site-t{self.step:02d}-L35.pt'
            self.torch.save(self.site, path)
            entry = dict(step=self.step, layer_zero_based=35, file=path.name, sha256=sha(path), bytes=path.stat().st_size,
                action_shape=[16, 4096], action_boundary_ids=[36], full_QKV_raw_saved=False,
                actual_model_kwargs_sha256=self.site['actual_model_kwargs_sha256'], model_output_sha256=self.site['model_output_sha256'])
            self.files.append(entry)
            self.site_files.append(entry)
            self.outputs.append(actual_output[2][0])
            self.counts['model_after'] += 1
            self.in_model = False
            self.boundaries, self.events, self.und_boundaries, self.hidden_metadata = [], [], [], []
            self.actual_kwargs, self.legacy_boundary = None, None
            self.site, self.aux, self.live_x = None, None, None
            self.main_W_live = self.main_r_live = self.selected_m_live = None
            self.prefix_boundary = None

        def report(self, record):
            assert self.restored and not self.handles and not self.in_model and self.active is None
            assert self.aux_phase is self.site is self.aux is self.live_x is None
            assert self.counts == expected_counts(self.arm) and self.finite_boundary_checks == 1110
            assert self.full_live_checks == 30 and len(self.calls) == len(self.files) == len(self.outputs) == 30
            self.runtime.require_exact(self.torch.stack(self.outputs), record['action_velocity'], 'All30 actual action tuple outputs')
            assert self.reader.settings(self.torch, self.model.layers[0].self_attn.processor) == self.settings
            assert len(self.random_reports) == (30 if self.arm == 'matched_random' else 0)
            assert all(row['strength_gate_pass'] for row in self.random_reports)
            write(self.folder / 'site-captures.json', dict(state='complete', files=self.site_files,
                captures=30, explicit_action16_raw_only=True, all_full_QKV_raw_saved=False,
                all_full266_raw_saved=False, random_strength_reports=self.random_reports,
                whole_original_arrays_have_live_numerical_SHA=True, two_ordered_BF16_addition_gates_passed=True, scope=SCOPE), exclusive=True)
            return dict(state='complete', counts=self.counts, boundary_files=self.files, site_files=self.site_files,
                explicit_action16_raw_only=True, whole30_NormalRuntime_record_saved=True,
                full37_raw_boundary_steps=[], all37_by30_finite_live_metadata_recorded=True,
                actual_L35_selected_sites=30, all_original_QKV_and_global_RNG_bytes_preserved=True,
                all_nonaction_GEN_rows_native_through_attention_W_norm_MLP_block_byte_exact=True,
                actual_t0_b0_through_b35_decoder_prefix_and_final_UND_current_byte_exact=True,
                all_main_full266_W_and_MLP_calls_preserved=True,
                same_live_full266_native_W_residual_postnorm_MLP_baseline_computed=True,
                auxiliary_call_counts={key: self.counts[key] for key in
                    ('aux_GEN_dispatch', 'aux_W', 'aux_W_native', 'aux_W_donor', 'aux_postnorm', 'aux_MLP')},
                all30_same_input_BF16_residual_and_actual_block_bytes_reconstructed=True,
                live_self_complete_Z_W_residual_norm_MLP_byte_gates_passed=self.arm == 'live_self',
                random_strength_reports=self.random_reports, all_random_strength_gates_passed=True,
                all_owned_hooks_removed=True, original_dispatch_symbol_restored=True, scope=SCOPE)

    return InstructionObserver


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check-source-contract', action='store_true', help='Read accepted seals only; no output/model')
    args = parser.parse_args()
    baseline = args.output.resolve()
    context = source_contract(baseline)
    if args.check_source_contract:
        print(json.dumps(dict(state='source_contract_verified', stage=STAGE, plan=PLAN,
            natural_donor_cases={key: value['case'] for key, value in context['matched_references'].items()},
            expected_counts=expected_counts(), total_official_dispatch_calls=17520), indent=2))
        return
    out, paths, contract = context['out'], context['paths'], context['frozen_contract']
    assert not out.exists(), 'A completed, failed or partial stage must never be retried or overwritten'
    available = shutil.disk_usage(baseline).free
    assert available >= MIN_FREE_BYTES, 'Insufficient free space for capped producer and physical reserve; no files removed'
    out.mkdir(exist_ok=False)
    labels = [row[0] for row in PLAN]
    protocol = dict(stage=STAGE, trial_order=labels,
        plan=[dict(case=label, scene='x15', goal=recipient, recipient_goal=recipient, donor_goal=donor,
            arm=arm, vision_noise_source_seed=198, action_noise_source_seed=195, runtime_rng_seed=195,
            recipient_prompt=PROMPTS[recipient], donor_prompt=PROMPTS[donor], window=WINDOW)
            for label, recipient, donor, arm in PLAN],
        primary_hypothesis=PRIMARY, prompts=PROMPTS, target_objects=TARGETS,
        scene=context['scenes']['x15'], shift_cm=15, q0_predictions=8, model_forwards=240,
        expected_counts=expected_counts(), total_official_dispatch_calls=17520,
        original_dispatch_calls=17280, auxiliary_GEN_dispatch_calls=240,
        auxiliary_full266_W_native_calls=240, auxiliary_full266_W_donor_calls=240,
        auxiliary_full266_postnorm_calls=240, auxiliary_full266_MLP_calls=240,
        selected_window_zero_based_inclusive=WINDOW,
        suffix_rule='Actual recipient Q and all other K/V unchanged; full388 keys, no mask. Donor saved postRoPE UND-for-GEN K/V at48..121 only. Merge action16 rows250..265.',
        self_rule='Live full K/V clone extra official GEN endpoint; complete Z/W/r/postnorm/MLP equality and accepted entire q0 record required.',
        same_input_MLP_rule='Actual current-call L35 full266 x and native Z0 through official W0, actual BF16 x+W0, official postnorm and MLP give m0. C replaces only action16 main MLP output; original block adds.',
        same_auxiliary_budget_in_all_four_arms=True,
        random_rule=dict(seed='104729 + 1000*direction_index + step', private_CPU_generator=True,
            direction='Fixed 128-channel permutation plus independent signs inside each action row/query head of same-call FP32 donor-minus-native delta.',
            fitting='One scalar alpha in[0,4], at most32 evaluations; firstalpha1, then fixed bisection; no new direction or module call.',
            actual_BF16_action_L2_relative_error_max=RANDOM_TOLERANCE, all_steps_fail_closed=True,
            zero_target='Exactly preserve native bytes', actual_W_output_strength_is_reported_not_matched=True),
        capture_scope='Only explicit action16 eight endpoints plus actual postblock action; r0/r1 and candidates live SHA/ordered BF16 reconstruction. Full QKV/266 arrays omitted with actual SHA.',
        maximum_stage_bytes=MAX_BYTES, physical_reserve_bytes=PHYSICS_RESERVE_BYTES,
        pre_output_minimum_free_bytes=MIN_FREE_BYTES, actual_free_bytes_before_output=available,
        later_physics_preregistered=dict(cases=8, cached_q0=8, later_predictions=56,
            queries=64, q1_to_q7_seeds=list(range(196, 203)), recipient_native_only=True,
            later_model_forwards=1680, combined_model_forwards=1920, later_whole_noise_pairs=49,
            physical_actions=1024, all8_execute_without_q0_ranking=True),
        physics_calls=0, no_training=True, scope=SCOPE)
    write(out / 'protocol.json', protocol, exclusive=True)
    write(out / 'primary-prediction.json', dict(status='not_evaluated_q0_only', **PRIMARY, scope=SCOPE), exclusive=True)
    write(out / 'sources.json', dict(source_evidence=context['source_evidence'],
        original_sources=context['original_noise_sources'], scene=context['scenes']['x15'],
        frozen_contract=contract, matched_references=context['matched_references'],
        accepted_matched_physical_frozen=context['matched_frozen']), exclusive=True)
    started, completed, rows, runtime, observer, noise = time.perf_counter(), [], [], None, None, None
    counts = dict.fromkeys(expected_counts(), 0)

    def progress(stage, **values):
        document = dict(stage=stage, completed=completed, expected_model_forwards=240, actual_counts=counts,
            elapsed_s=time.perf_counter() - started, **values)
        write(out / 'progress.json', document)
        print('[INSTRUCTION-INTERACTION] ' + json.dumps(document), flush=True)

    try:
        os.environ['HF_HUB_OFFLINE'] = '1'
        matched, language, value, frozen, factor = (context['modules'][key] for key in
            ('matched', 'language', 'value', 'frozen', 'factor'))
        base, factory = load('instruction_runtime', paths['normal_runtime']), load('instruction_factory', paths['factory'])
        for goal in PROMPTS:
            base.TASKS[goal] = (7, 'pick_up_the_milk_and_place_it_in_the_basket', PROMPTS[goal], TARGETS[goal])
        factory.OUT = out
        assert sha(factory.SCHEDULER) == contract['frozen_files']['scheduler']['sha256']
        progress('loading_model')
        runtime = base.NormalRuntime(factory)
        torch, model = runtime.torch, runtime.pipe.transformer
        components, scan, reader, processor_module, dispatch, backend = value.runtime_contract(baseline, context, runtime)
        checker = components.CosmosComponentInterventions(model)
        originals, caches, references, infos = {}, {}, {}, {}
        for v in (195, 198):
            folder = Path(context['original_noise_sources'][str(v)]['q0_folder'])
            originals[v] = torch.load(folder / 'states.pt', map_location='cpu', weights_only=True, mmap=True)
            caches[v] = torch.load(folder / 'components.pt', map_location='cpu', weights_only=True, mmap=True)
            scan.validate_record(originals[v], torch)
            scan.validate_cache(caches[v], originals[v], checker, runtime, 'Original selected noise/' + str(v))
        runtime.require_exact(caches[195]['steps'], caches[198]['steps'], 'Actual frozen schedule/domain/layout')
        for goal, reference in context['matched_references'].items():
            folder = Path(reference['folder'])
            references[goal] = torch.load(folder / 'states.pt', map_location='cpu', weights_only=True, mmap=True)
            infos[goal] = torch.load(folder / 'language-contract.pt', map_location='cpu', weights_only=True, mmap=True)
            scan.validate_record(references[goal], torch)
            assert infos[goal]['scene'] == 'x15' and infos[goal]['goal'] == goal
            assert infos[goal]['language']['target_positions'] == [48, 49]
            assert len(infos[goal]['actual_pack']) == 30
            # Validate all30 natural supplies before the first new forward.
            for step, item in enumerate(reference['dispatch_files']):
                path = folder / item['file']
                assert item['step'] == step and sha(path) == item['sha256']
                data = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
                assert data['step'] == step and data['layer_zero_based'] == 35 and data['causal_suffix_raw_saved'] is True
                runtime.require_exact(data['causal_suffix_key_indexes'], torch.arange(48, 122), 'Natural suffix key position equality')
                for key in ('Kcausal_suffix', 'Vcausal_suffix'):
                    item_raw = data[key]
                    assert item_raw.shape == (1, 74, 8, 128) and item_raw.dtype == torch.bfloat16 and bool(torch.isfinite(item_raw).all())
                    meta = language.tensor_meta(item_raw, runtime)
                    for field in ('shape', 'dtype', 'sha256'):
                        assert meta[field] == data['sparse_metadata'][key][field]
        runtime.require_exact(infos['milk_box']['actual_clean_current'], infos['cream_cheese']['actual_clean_current'], 'Both natural names same actual X15 frame')
        left, right = (infos[goal]['first_model_kwargs'] for goal in PROMPTS)
        assert left.keys() == right.keys()
        for key in left:
            if key != 'input_ids':
                runtime.require_exact(left[key], right[key], 'Both natural donors complete initial kwargs except target IDs/' + key)
        changed = torch.nonzero(left['input_ids'].flatten() != right['input_ids'].flatten()).flatten().tolist()
        assert changed == [48, 49]
        for goal, value in zip(PROMPTS, (left, right)):
            assert [int(value['input_ids'].flatten()[i]) for i in changed] == matched.TARGET_IDS[goal]
        prompt_ids = {goal: runtime.pipe.text_tokenizer.encode(prompt, add_special_tokens=False) for goal, prompt in PROMPTS.items()}
        for goal in PROMPTS:
            assert prompt_ids[goal] == infos[goal]['language']['prompt_token_ids'] and len(prompt_ids[goal]) == 11
        native_random = runtime.cm.randn_tensor
        provenance = dict(script_sha256=sha(Path(__file__)), protocol_sha256=sha(out / 'protocol.json'),
            sources_sha256=sha(out / 'sources.json'), source_sha256={key: sha(path) for key, path in paths.items()},
            matched_producer_sha256=MATCHED_PROBE_SHA, matched_runner_sha256=MATCHED_RUNNER_SHA,
            matched_analyzer_sha256=MATCHED_ANALYZER_SHA,
            actual_transformer_sha256=sha(Path(inspect.getfile(type(model)))),
            actual_pipeline_source_sha256=sha(Path(inspect.getfile(type(runtime.pipe)))),
            dispatch_source_sha256=sha(Path(inspect.getfile(dispatch))), dispatch_signature=str(inspect.signature(dispatch)),
            torch_version=str(torch.__version__), CPU_threads=torch.get_num_threads(), actual_prompt_token_ids=prompt_ids,
            tokenizer_class=type(runtime.pipe.text_tokenizer).__name__, **backend,
            owned_auxiliary_scope='Official full266 native W/postnorm/MLP and donor W; same count in each arm.',
            library_edits=False, model_parameter_edits=False, physics_calls=0, scope=SCOPE)
        write(out / 'provenance.json', provenance, exclusive=True)
        Observer = observer_class(matched, frozen, language)
        image = Path(context['scenes']['x15']['image'])
        assert sha(image) == context['scenes']['x15']['files_sha256']['input.png']
        records = {}
        for label, recipient, donor, arm in PLAN:
            reference = context['matched_references'][recipient]
            folder = out / label
            observer = Observer(runtime, components, scan, reader, caches[195], None, originals[195], arm, folder,
                scene='x15', goal=recipient, prompt_ids=prompt_ids[recipient], native_info=infos[recipient],
                legacy_reference=reference if arm == 'live_self' else None,
                recipient=recipient, donor=donor, direction_index=[item[0] for item in DIRECTIONS].index(recipient),
                donor_reference=context['matched_references'][donor], recipient_reference=reference)
            noise = factor.InitialNoiseSources(runtime, scan, originals, 198, 195)
            assert observer.original is dispatch and noise.original is native_random
            progress('fresh_q0', case=label)
            observer.begin()
            try:
                noise.begin()
                record, metadata = runtime.predict(recipient, image, 195, folder)
            finally:
                try:
                    noise.reset()
                finally:
                    observer.reset()
                    for key in counts:
                        counts[key] += observer.counts[key]
                    if folder.exists():
                        torch.save(dict(calls=noise.calls, model_steps=observer.calls, initial_model_kwargs=observer.first,
                            original_symbol_restored=noise.restored, observer_hooks_removed=not observer.handles,
                            dispatch_symbol_restored=observer.restored, actual_counts=observer.counts), folder / 'noise-audit.pt')
                        if observer.in_model:
                            torch.save(dict(step=observer.step, actual_counts=observer.counts, active_layer=observer.active,
                                partial_site=observer.site, live_hidden_metadata=observer.hidden_metadata,
                                actual_model_kwargs_sha256=language.tree_sha(observer.actual_kwargs, runtime)), folder / 'partial-forward.pt')
            assert runtime.cm.randn_tensor is native_random and processor_module.dispatch_attention_fn is dispatch
            scan.validate_record(record, torch)
            assert metadata['model_calls'] == 30 and metadata['prepare_calls'] == 1 and metadata['seed'] == 195
            assert metadata['input_png_sha256'] == sha(image) and metadata['prompt'] == PROMPTS[recipient] and metadata['task_index'] == 7
            report, noise_report = observer.report(record), noise.report(record)
            preparation = matched.input_checks(record, observer, originals, 198, runtime)
            runtime.require_exact(record['prepared_latents_and_masks'], references[recipient]['prepared_latents_and_masks'], 'Entire natural recipient preparation')
            runtime.require_exact(record['pure_noise'], references[recipient]['pure_noise'], 'Entire natural recipient returned noise')
            entire_self = arm == 'live_self'
            if entire_self:
                runtime.require_exact(record, references[recipient], 'ENTIRE accepted natural-recipient q0 live-self record')
            info = dict(scene='x15', goal=recipient, recipient_goal=recipient, donor_goal=donor,
                case=label, arm=arm, native_reference_case=reference['case'], actual_pack=observer.calls,
                first_model_kwargs=observer.first, actual_clean_current=observer.clean_current,
                original_solver_initial_action_FP32=record['action_states'][0], language=observer.language,
                actual_input_checks=preparation, accepted_reference_files_sha256=reference['files_sha256'], scope=SCOPE)
            torch.save(info, folder / 'language-contract.pt')
            control = dict(state='exact', control_reference_case=reference['case'],
                own_scene_recipient_V_entire_initial_kwargs_and_all30_pack_exact=True,
                entire_accepted_recipient_q0_record_byte_exact=entire_self,
                source_noise_preparation_sigma_clamp_padding_exact=True,
                same_auxiliary_budget_in_all_four_arms=True, finite_and_random_strength_gates_passed=True,
                later_feedback_prompt=PROMPTS[recipient], later_feedback_native=True)
            write(folder / 'control-checks.json', control, exclusive=True)
            write(folder / 'observer-report.json', report, exclusive=True)
            write(folder / 'noise-audit.json', dict(noise=noise_report, preparation=preparation), exclusive=True)
            metadata.update(case=label, scene='x15', goal=recipient, recipient_goal=recipient, donor_goal=donor,
                arm=arm, query=0, policy_prompt=PROMPTS[recipient], donor_prompt=PROMPTS[donor],
                target_object=TARGETS[recipient], donor_target_object=TARGETS[donor], shift_cm=15,
                vision_noise_source_seed=198, action_noise_source_seed=195, runtime_rng_seed=195,
                producer_script_sha256=provenance['script_sha256'], control_reference_case=reference['case'],
                actual_selected_window=WINDOW, same_input_native_MLP_baseline=True,
                explicit_action16_raw_only=True, physical_prediction_evaluated=False, scope=SCOPE)
            write(folder / 'metadata.json', metadata)
            assert read(folder / 'normalized_actions.json') == record['actions'].tolist()
            rows.append(dict(case=label, scene='x15', goal=recipient, recipient_goal=recipient, donor_goal=donor,
                prompt=PROMPTS[recipient], donor_prompt=PROMPTS[donor], arm=arm,
                vision_noise_source_seed=198, action_noise_source_seed=195, runtime_rng_seed=195,
                shift_cm=15, fresh_model_forwards=30, actual_dispatch_counts=observer.counts,
                control_reference_case=reference['case'], entire_accepted_recipient_q0_record_byte_exact=entire_self,
                actual_und_len=122, actual_target_positions=[48, 49], actual_suffix_key_positions=[48, 121],
                actual_target_token_ids=matched.TARGET_IDS[recipient], files_sha256={name: sha(folder / name) for name in FILES},
                boundary_files=observer.files, site_files=observer.site_files,
                final_normalized_actions=record['actions'].tolist(), physical_prediction_evaluated=False))
            records[label] = record
            completed.append(label)
            assert disk_bytes(out) <= MAX_BYTES, 'Actual stage exceeds frozen350MiB cap; no deletion or automatic retry'
            progress('case_complete', case=label, actual_stage_bytes=disk_bytes(out))
            observer = noise = None
        assert completed == labels and counts == expected_counts()
        assert sum(counts[key] for key in DISPATCH_KEYS) == 17520
        assert not checker._handles and checker._kind is None
        assert runtime.cm.randn_tensor is native_random and processor_module.dispatch_attention_fn is dispatch
        write(out / 'results.json', dict(state='complete', cases=rows, fresh_q0_predictions=8, fresh_model_forwards=240,
            official_dispatch_counts=counts, total_official_dispatch_calls=17520, preregistered_physics_cases=labels,
            primary_physical_prediction_evaluated=False, physics_calls=0, scope=SCOPE), exclusive=True)
        total = disk_bytes(out)
        assert total < MAX_BYTES - 1_048_576
        write(out / 'complete.json', dict(state='complete', script_sha256=sha(Path(__file__)),
            **{name.replace('-', '_') + '_sha256': sha(out / (name + '.json')) for name in
                ('results', 'provenance', 'protocol', 'sources', 'primary-prediction')},
            fresh_q0_predictions=8, fresh_model_forwards=240, official_dispatch_counts=counts,
            total_official_dispatch_calls=17520, original_dispatch_calls=17280, auxiliary_GEN_dispatch_calls=240,
            auxiliary_full266_W_native_calls=240, auxiliary_full266_W_donor_calls=240,
            auxiliary_full266_postnorm_calls=240, auxiliary_full266_MLP_calls=240,
            original_randn_calls_consumed=16, returned_source_draws_observed=16,
            two_live_self_entire_accepted_recipient_q0_records_exact=True,
            actual_equal_length_two_target_ID_and_suffix_position_gates_passed=True,
            all240_same_live_native_W_postnorm_MLP_baselines_computed=True,
            all240_main_and_auxiliary_call_counts_equal_across_arms=True,
            all240_original_QKV_and_CPU_CUDA_global_RNG_byte_gates_passed=True,
            all240_actual_BF16_residual_and_selected_block_reconstruction_gates_passed=True,
            all60_random_actual_BF16_strength_gates_passed=True,
            all8880_hidden_boundaries_finite_live_SHA_saved=True, site_raw_captures=240,
            explicit_action16_raw_only=True, full_QKV_raw_sites=0, full266_raw_sites=0,
            all8_complete30_NormalRuntime_records_saved=True,
            actual_stage_bytes_before_completion=total, maximum_stage_bytes=MAX_BYTES,
            physical_reserve_bytes=PHYSICS_RESERVE_BYTES,
            all_original_symbols_restored=True, all_owned_hooks_removed=True,
            primary_physical_prediction_evaluated=False, physics_calls=0, later_query_predictions=0, scope=SCOPE), exclusive=True)
        progress('complete')
        assert disk_bytes(out) <= MAX_BYTES
    except Exception as exc:
        write(out / 'failed.json', dict(state='failed', error=str(exc), traceback=traceback.format_exc(),
            completed=completed, actual_counts=counts, active_case=None if observer is None else observer.folder.name,
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
