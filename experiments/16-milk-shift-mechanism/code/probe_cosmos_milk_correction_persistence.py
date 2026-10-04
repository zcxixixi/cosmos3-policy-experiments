"""Test short versus sustained interruption of one candidate correction site.

For each paired seed 195/196/198, use the actual x15 q0 input and x06 donor.
P replaces L17 attention action-row increments at forward 20 once. Candidate
L3 GEN MLP increments are then blended toward the matching native R cache at
forward 21 alone or forwards 21..29. The remaining computation responds freely.

Eight predeclared arms per seed make 24 q0 predictions. Native R, P, and the
R persistent self-replacement controls must match their entire saved records.
P gain1 precedes independently seeded, quantization-aware random controls whose
actual bf16 delta norm matches its corresponding step. No scheduler, inference
library, checkpoint, instruction or simulator code is edited. No MuJoCo runs.
Own only an absent BASE/correction-persistence directory. Continued differences
may reflect solver inheritance; these tests do not identify a milk neuron,
attractor, neural root cause or repaired grasp.
"""

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import time
import traceback


ROOT = Path('/home/current/work/cosmos3')
SEEDS = (195, 196, 198)
PROMPT = 'pick up the milk and place it in the basket'
PULSE_LAYER, PULSE_STEP, RESET_LAYER = 17, 20, 3
RESET_STEPS = tuple(range(21, 30))
NORM_REL_TOL = 1e-3
RANDOM_SEEDS = {seed: {step: 730000000 + seed * 100 + step for step in RESET_STEPS} for seed in SEEDS}
ARMS = (
    dict(name='native_R', pulse=False, reset_steps=(), gain=None, mode='native'),
    dict(name='P', pulse=True, reset_steps=(), gain=None, mode='native'),
    dict(name='R_persistent_self', pulse=False, reset_steps=RESET_STEPS, gain=1., mode='blend_R'),
    dict(name='P_persistent_gain100', pulse=True, reset_steps=RESET_STEPS, gain=1., mode='blend_R'),
    dict(name='P_single_reset', pulse=True, reset_steps=(21,), gain=1., mode='blend_R'),
    dict(name='P_persistent_gain050', pulse=True, reset_steps=RESET_STEPS, gain=.5, mode='blend_R'),
    dict(name='P_persistent_gain200', pulse=True, reset_steps=RESET_STEPS, gain=2., mode='blend_R'),
    dict(name='P_random_normmatched', pulse=True, reset_steps=RESET_STEPS, gain=None, mode='random'),
)
FUTURE_PHYSICAL_CANDIDATES = tuple(dict(seed=198, arm=name) for name in
    ('native_R', 'P', 'P_persistent_gain100', 'P_random_normmatched'))
SCOPE = ('Paired fixed-input q0 computation with one L17 attention pulse and '
         'short/sustained candidate L3 MLP replacement. Later computation and '
         'the multistep solver respond freely. Continuous normalized action '
         'outputs only; no MuJoCo, milk-neuron, attractor or grasp-rescue claim.')


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def random_replacement(current, target, seed, torch):
    """Match the target at the actual bf16 landing point, without global RNG."""
    assert current.device.type == 'cpu' and current.dtype == torch.bfloat16
    assert tuple(current.shape) == (16, 4096) and torch.isfinite(current).all()
    assert math.isfinite(target) and target >= 0
    if target == 0:
        return current.clone(), dict(cpu_generator_seed=seed, target_actual_delta_l2=0.,
            actual_quantized_delta_l2=0., relative_norm_error=0., relative_norm_tolerance=NORM_REL_TOL,
            zero_target_noop=True, calibration_iterations=0, actual_quantized_delta=torch.zeros_like(current, dtype=torch.float64))
    generator = torch.Generator(device='cpu').manual_seed(seed)
    raw = torch.randn(current.shape, generator=generator, device='cpu', dtype=torch.float64)
    unit = raw / raw.norm()

    def candidate(scale):
        value = (current.double() + unit * scale).to(current.dtype)
        delta = value.double() - current.double()
        return value, delta, float(delta.norm())

    lo, hi = 0., target * 2
    for _ in range(16):
        if candidate(hi)[2] >= target:
            break
        hi *= 2
    else:
        raise RuntimeError('Could not bracket the actual rounded random delta norm')
    best, iterations = None, 0
    for iteration in range(64):
        scale = (lo + hi) / 2
        value, delta, norm = candidate(scale)
        error = abs(norm - target) / target
        if best is None or error < best[0]:
            best = error, value, delta, norm, scale
        iterations = iteration + 1
        if error <= NORM_REL_TOL:
            break
        if norm < target:
            lo = scale
        else:
            hi = scale
    error, value, delta, norm, scale = best
    assert error <= NORM_REL_TOL, f'Actual bf16 random norm mismatch: {error}'
    assert torch.isfinite(value).all()
    return value, dict(cpu_generator_seed=seed, target_actual_delta_l2=target,
        actual_quantized_delta_l2=norm, relative_norm_error=error, relative_norm_tolerance=NORM_REL_TOL,
        zero_target_noop=False, calibration_iterations=iterations, scale_before_quantization=scale,
        raw_gaussian=raw, unit_direction=unit, actual_quantized_delta=delta)


class TwoSiteSession:
    """Observe two real outputs each forward; replace only selected increments."""

    def __init__(self, runtime, module, scan):
        self.runtime, self.torch, self.module, self.scan = runtime, runtime.torch, module, scan
        self.model = runtime.pipe.transformer
        assert len(self.model.layers) == 36 and not self.model.training and not self.torch.is_grad_enabled()
        assert not self.model.is_cache_enabled
        assert getattr(self.model, '_cp_shard_fn', None) is None and getattr(self.model, '_cp_gather_fn', None) is None
        assert all(getattr(layer.self_attn.processor, '_parallel_config', None) is None for layer in self.model.layers)
        assert self.model.layers[PULSE_LAYER].self_attn.to_add_out.out_features == 4096
        assert self.model.layers[RESET_LAYER].mlp_moe_gen.down_proj.out_features == 4096
        self.handles, self.active = [], False

    def begin(self, arm, seed, r_cache, d_cache, r_record, actual_r21, actual_p21, pp_frame, random_targets=None):
        assert not self.active and not self.handles
        assert r_cache['complete'] and d_cache['complete']
        assert r_cache['n_steps'] == len(r_cache['steps']) == d_cache['n_steps'] == len(d_cache['steps']) == 30
        self.arm, self.seed, self.r_cache, self.d_cache = arm, seed, r_cache, d_cache
        self.r_record, self.r21, self.p21, self.pp_frame = r_record, actual_r21, actual_p21, pp_frame
        self.random_targets = random_targets
        if arm['mode'] == 'random':
            assert set(random_targets) == set(RESET_STEPS)
            assert all(math.isfinite(value) and value >= 0 for value in random_targets.values())
        self.step, self.inside = -1, False
        self.visited, self.finished, self.events, self.steps = set(), set(), {}, []
        self.first_kwargs, self.input21 = None, None
        self.step21_kwargs_exact = self.step21_mlp_exact = False
        self.active = True
        try:
            self.handles.append(self.model.register_forward_pre_hook(self.before_model, with_kwargs=True))
            self.handles.append(self.model.register_forward_hook(self.after_model))
            self.handles.append(self.model.layers[PULSE_LAYER].self_attn.register_forward_hook(self.component('attention')))
            self.handles.append(self.model.layers[RESET_LAYER].mlp_moe_gen.register_forward_hook(self.component('mlp')))
        except BaseException:
            self.reset()
            raise

    def before_model(self, model, args, kwargs):
        assert not args and not self.inside, 'Exactly one unnested keyword-only model forward required'
        self.inside = True
        self.step += 1
        assert self.step < 30, 'Extra model forward/CFG/recomputation'
        self.layout = self.module._layout(kwargs)
        assert self.layout['action_rows'].numel() == 16
        frame = dict(layout=self.layout, metadata={key: self.runtime.cpu(kwargs.get(key)) for key in self.module._STRUCTURAL_KEYS})
        self.runtime.require_exact(frame, self.r_cache['steps'][self.step], f'{self.arm["name"]}/actual_pack/{self.step}')
        self.steps.append(frame)
        if self.step == 0:
            self.first_kwargs = self.runtime.cpu(kwargs)
            self.runtime.require_exact(self.first_kwargs, self.r_record['model_input'], 'initial_entire_native_kwargs_exact')
        if self.step == 21:
            self.input21 = self.runtime.cpu(kwargs)
            expected = self.p21 if self.arm['pulse'] else self.r21
            self.runtime.require_exact(self.input21, expected, 'actual_first_reset_input_equals_saved_P_or_R_t21')
            self.step21_kwargs_exact = True

    def after_model(self, model, args, output):
        assert self.inside
        assert {(self.step, 'attention'), (self.step, 'mlp')}.issubset(self.visited)
        self.finished.add(self.step)
        self.inside = False

    def component(self, component):
        def after(module, args, output):
            torch, runtime = self.torch, self.runtime
            assert self.inside
            call = (self.step, component)
            assert call not in self.visited, 'Duplicate target site call/recomputation'
            self.visited.add(call)
            layer = PULSE_LAYER if component == 'attention' else RESET_LAYER
            if component == 'attention':
                assert isinstance(output, tuple) and len(output) == 2
                gen = output[1]
            else:
                gen = output
            assert isinstance(gen, torch.Tensor) and gen.shape == (self.layout['gen_len'], 4096)
            assert gen.dtype == torch.bfloat16
            selected = ((component == 'attention' and self.arm['pulse'] and self.step == PULSE_STEP)
                        or (component == 'mlp' and self.step in self.arm['reset_steps']))
            if not selected and not (component == 'mlp' and self.step == 21):
                return
            rows = self.layout['action_rows'].to(gen.device)
            before = runtime.cpu(gen.index_select(0, rows))
            if component == 'mlp' and self.step == 21:
                expected = self.pp_frame if self.arm['pulse'] else self.r_cache['frames'][(21, RESET_LAYER, 'mlp')]
                runtime.require_exact(before, expected, 'first_reset_actual_MLP_before_equals_PP_or_native_R')
                self.step21_mlp_exact = True
            if not selected:
                return
            key = (self.step, layer, component)
            calibration = None
            if component == 'attention':
                runtime.require_exact(before, self.r_cache['frames'][key], 'pulse_actual_attention_before_equals_R')
                source = self.d_cache['frames'][key]
                changed = source.to(device=gen.device, dtype=gen.dtype)
                mode, gain = 'natural_D_pulse', 1.
            elif self.arm['mode'] == 'random':
                changed_cpu, calibration = random_replacement(before, self.random_targets[self.step],
                                                              RANDOM_SEEDS[self.seed][self.step], torch)
                changed = changed_cpu.to(gen.device)
                source = None
                mode, gain = 'actual_norm_matched_random', None
            else:
                source = self.r_cache['frames'][key]
                if self.arm['name'] == 'R_persistent_self':
                    runtime.require_exact(before, source, 'no_pulse_self_reset_actual_before_equals_native_R')
                reference = source.to(device=gen.device, dtype=gen.dtype)
                gain = self.arm['gain']
                changed = reference if gain == 1 else gen.index_select(0, rows) + gain * (reference - gen.index_select(0, rows))
                mode = 'current_plus_gain_times_R_minus_current_in_native_bf16'
            assert changed.shape == (16, 4096) and changed.dtype == gen.dtype and torch.isfinite(changed).all()
            altered = gen.clone()
            altered.index_copy_(0, rows, changed)
            after_cpu = runtime.cpu(altered.index_select(0, rows))
            runtime.require_exact(after_cpu, runtime.cpu(changed), 'actual_selected_after_equals_requested_value')
            if source is not None and gain == 1:
                runtime.require_exact(after_cpu, source, 'gain1_actual_after_bit_exact_source_cache')
            before_all, after_all = runtime.cpu(gen), runtime.cpu(altered)
            mask = torch.ones(self.layout['gen_len'], dtype=torch.bool)
            mask[self.layout['action_rows']] = False
            nonaction_before, nonaction_after = before_all[mask], after_all[mask]
            runtime.require_exact(nonaction_before, nonaction_after, 'actual_entire_nonaction_increment_rows_bytes_exact')
            delta = after_cpu.double() - before.double()
            norm = float(delta.norm())
            if calibration is not None:
                runtime.require_exact(delta, calibration['actual_quantized_delta'], 'actual_random_delta_equals_quantized_calibration')
                target = self.random_targets[self.step]
                error = 0. if target == 0 else abs(norm - target) / target
                if target == 0:
                    assert norm == 0
                else:
                    assert error <= NORM_REL_TOL
                calibration['actual_hook_delta_l2'] = norm
                calibration['actual_hook_relative_error'] = error
            before_meta, after_meta = self.scan.tensor_meta(before, torch), self.scan.tensor_meta(after_cpu, torch)
            self.events[key] = dict(step=self.step, layer=layer, component=component, mode=mode, gain=gain,
                action_rows=self.layout['action_rows'].clone(), before=before, after=after_cpu, actual_delta=delta,
                actual_delta_l2=norm, exact_no_change=before_meta == after_meta,
                before_meta=before_meta, after_meta=after_meta,
                source_cache_meta=None if source is None else self.scan.tensor_meta(source, torch),
                nonaction_rows_count=int(mask.sum()), nonaction_rows_bit_exact=True,
                nonaction_before=self.scan.tensor_meta(nonaction_before, torch),
                nonaction_after=self.scan.tensor_meta(nonaction_after, torch),
                full_GEN_before=self.scan.tensor_meta(before_all, torch), full_GEN_after=self.scan.tensor_meta(after_all, torch),
                random_calibration=calibration)
            if before_meta == after_meta:
                return output
            return (output[0], altered) if component == 'attention' else altered
        return after

    def reset(self):
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        self.active, self.inside = False, False

    def end(self):
        assert self.active
        expected_calls = {(step, component) for step in range(30) for component in ('attention', 'mlp')}
        expected_events = {(step, RESET_LAYER, 'mlp') for step in self.arm['reset_steps']}
        if self.arm['pulse']:
            expected_events.add((PULSE_STEP, PULSE_LAYER, 'attention'))
        complete = (self.step == 29 and not self.inside and self.finished == set(range(30))
                    and self.visited == expected_calls and set(self.events) == expected_events
                    and len(self.steps) == 30 and self.step21_kwargs_exact and self.step21_mlp_exact)
        result = dict(version=1, complete=complete, seed=self.seed, arm=self.arm,
            n_model_forwards=self.step + 1, finished_forwards=len(self.finished),
            attention_site_calls=sum(component == 'attention' for _, component in self.visited),
            mlp_site_calls=sum(component == 'mlp' for _, component in self.visited),
            explicit_processor_recomputations=0, intervention_count=len(self.events),
            expected_intervention_count=len(expected_events), step21_actual_kwargs_exact=self.step21_kwargs_exact,
            step21_actual_MLP_before_exact=self.step21_mlp_exact, first_model_kwargs=self.first_kwargs,
            actual_step21_kwargs=self.input21, steps=self.steps, events=self.events,
            nonaction_rows_bit_exact=all(event['nonaction_rows_bit_exact'] for event in self.events.values()),
            untouched_increment_rows='Every non-action GEN row of each selected output; decoder residuals retained')
        self.reset()
        return result


def session_json(report, scan, torch):
    """JSON hashes accompany the actual tensors in actualsession.pt."""
    def convert(value):
        if isinstance(value, torch.Tensor):
            return scan.tensor_meta(value, torch)
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        return value
    return convert(report)


def load_seed_sources(baseline, feedback, seed, runtime, checker, scan):
    torch = runtime.torch
    records, caches, sources = {}, {}, {}
    contract = read_json(feedback / f'seed{seed}/input-contract.json')
    assert contract['seed'] == seed and contract['both_pure_noise_draws_exact'] is True
    assert contract['complete_schedule_and_30step_metadata_exact'] is True
    for role, scene in (('R', 'x15'), ('D', 'x06')):
        folder = baseline / 'closed-loop' / f'{scene}_seed{seed}'
        records[role] = torch.load(folder / 'chunk_00/states.pt', map_location='cpu', weights_only=True, mmap=True)
        caches[role] = torch.load(folder / 'chunk_00/components.pt', map_location='cpu', weights_only=True, mmap=True)
        scan.validate_record(records[role], torch)
        scan.validate_cache(caches[role], records[role], checker, runtime, role)
        metadata = read_json(folder / 'chunk_00/metadata.json')
        assert metadata['seed'] == seed and metadata['query'] == 0 and metadata['prompt'] == PROMPT
        assert metadata['input_png_sha256'] == sha(folder / 'input_00.png')
        for name in ('states', 'components'):
            assert contract['sources'][role][name + '_sha256'] == sha(folder / 'chunk_00' / (name + '.pt'))
        sources[role] = dict(trial=f'{scene}_seed{seed}', image=str(folder / 'input_00.png'),
            image_sha256=sha(folder / 'input_00.png'), states_sha256=sha(folder / 'chunk_00/states.pt'),
            components_sha256=sha(folder / 'chunk_00/components.pt'))
    r, d = records['R'], records['D']
    for key in ('pure_noise', 'timesteps', 'sigmas'):
        runtime.require_exact(r[key], d[key], f'seed{seed}/paired_complete/{key}')
    runtime.require_exact(caches['R']['steps'], caches['D']['steps'], f'seed{seed}/paired_all_30_metadata')
    runtime.require_exact(r['model_input']['action_tokens'], d['model_input']['action_tokens'], f'seed{seed}/paired_initial_action_noise')
    folder = feedback / f'seed{seed}'
    native_r = torch.load(folder / 'native_R_q0/states.pt', map_location='cpu', weights_only=True, mmap=True)
    p = torch.load(folder / 'pulse_P_q0/states.pt', map_location='cpu', weights_only=True, mmap=True)
    r_inputs = torch.load(folder / 'native_R_q0/model_inputs_20_21.pt', map_location='cpu', weights_only=True, mmap=True)
    p_inputs = torch.load(folder / 'pulse_P_q0/model_inputs_20_21.pt', map_location='cpu', weights_only=True, mmap=True)
    assert set(r_inputs) == set(p_inputs) == {20, 21}
    runtime.require_exact(native_r, r, f'seed{seed}/feedback_native_R_entire_record')
    scan.validate_run_contract(p, r, runtime, f'seed{seed}/feedback_P_native_initial_contract')
    runtime.require_exact(r_inputs[20], p_inputs[20], f'seed{seed}/original_pulse20_input_exact')
    for key in ('readout', 'action_velocity'):
        runtime.require_exact(p[key][:20], r[key][:20], f'seed{seed}/saved_P_before_pulse/{key}')
    runtime.require_exact(p['action_states'][:21], r['action_states'][:21], f'seed{seed}/saved_P_before_pulse_solver')
    pp_folder = folder / 'replays/PP'
    pp_meta = read_json(pp_folder / 'metadata.json')
    assert pp_meta['label'] == f'seed{seed}_PP' and pp_meta['original_denoising_forward'] == 21
    assert pp_meta['component_cache_complete'] is True and pp_meta['component_forward_count'] == 1
    assert pp_meta['component_calls'] == 72 and pp_meta['original_q0_entire_model_output_exact'] is True
    for key, name in (('components_sha256', 'components.pt'), ('arrays_sha256', 'arrays.npz'), ('model_output_sha256', 'model_output.pt')):
        assert pp_meta[key] == sha(pp_folder / name)
    pp = torch.load(pp_folder / 'components.pt', map_location='cpu', weights_only=True, mmap=True)
    assert pp['complete'] is True and pp['n_steps'] == len(pp['steps']) == pp['expected_steps'] == 1
    assert pp['n_layers'] == 36 and pp['captured_component_count'] == pp['component_call_count'] == 72
    runtime.require_exact(pp['site_specs'], checker.site_specs, f'seed{seed}/PP_actual_sites')
    runtime.require_exact(pp['first_model_kwargs'], p_inputs[21], f'seed{seed}/PP_actual_input_equals_P_t21')
    runtime.require_exact(pp['steps'][0], caches['R']['steps'][21], f'seed{seed}/PP_pack_metadata_at_t21')
    pp_frame = pp['frames'][(0, RESET_LAYER, 'mlp')]
    assert pp_frame.shape == (16, 4096) and pp_frame.dtype == torch.bfloat16 and torch.isfinite(pp_frame).all()
    source_paths = dict(feedback_input_contract=folder / 'input-contract.json',
        feedback_native_R_states=folder / 'native_R_q0/states.pt', feedback_P_states=folder / 'pulse_P_q0/states.pt',
        feedback_R_inputs=folder / 'native_R_q0/model_inputs_20_21.pt', feedback_P_inputs=folder / 'pulse_P_q0/model_inputs_20_21.pt',
        feedback_PP_metadata=pp_folder / 'metadata.json', feedback_PP_components=pp_folder / 'components.pt')
    sources['feedback'] = {key: dict(path=str(path), sha256=sha(path)) for key, path in source_paths.items()}
    return records, caches, p, r_inputs[21], p_inputs[21], pp_frame, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Completed original baseline root with feedback-inputs')
    baseline = parser.parse_args().output.resolve()
    scan_path = ROOT / 'work/probe_cosmos_milk_component_pulses.py'
    feedback_script = ROOT / 'work/probe_cosmos_milk_feedback_inputs.py'
    scan = load_file('persistence_frozen_pulse_contract', scan_path)
    scan.validate_baseline(baseline)
    feedback = baseline / 'feedback-inputs'
    assert not (feedback / 'failed.json').exists()
    feedback_complete, feedback_provenance = read_json(feedback / 'complete.json'), read_json(feedback / 'provenance.json')
    assert feedback_complete['state'] == 'complete' and feedback_complete['seeds'] == list(SEEDS)
    assert feedback_complete['q0_predictions'] == 6 and feedback_complete['native_replays_exact'] == 6
    assert feedback_complete['direct_model_forwards'] == 12 and feedback_complete['components_per_forward'] == 72
    assert feedback_complete['summary_sha256'] == sha(feedback / 'summary.json')
    assert feedback_complete['script_sha256'] == feedback_provenance['script_sha256'] == sha(feedback_script)
    assert feedback_provenance['pulse_contract_sha256'] == sha(scan_path)
    assert feedback_provenance['baseline_summary_sha256'] == sha(baseline / 'summary.json')
    feedback_summary = read_json(feedback / 'summary.json')
    assert feedback_summary['all_native_R_and_P_t21_replays_bit_exact'] is True
    assert feedback_summary['entire_t21_model_output_including_vision_exact'] is True
    assert feedback_summary['current_condition_all_seeds_exact'] is True
    paths = dict(normal_runtime=ROOT / 'work/serve_cosmos_goal_pair.py',
                 factory=ROOT / 'work/probe_cosmos_language_routes.py',
                 components=ROOT / 'work/cosmos_component_interventions.py')
    baseline_provenance = read_json(baseline / 'server/provenance.json')
    for key, path in paths.items():
        assert sha(path) == baseline_provenance[key + '_sha256'] == feedback_provenance['source_sha256'][key]
    out = baseline / 'correction-persistence'
    out.mkdir(exist_ok=False)
    protocol = dict(seeds=list(SEEDS), arms=list(ARMS), execution_order=[arm['name'] for arm in ARMS],
        predictions=24, pulse=dict(layer_zero_based=17, component='attention', step=20, gain=1., donor='paired x06'),
        candidate=dict(layer_zero_based=3, layer_human=4, component='mlp', replacement='matching native x15 R increment'),
        random_cpu_seeds=RANDOM_SEEDS, actual_bf16_norm_relative_tolerance=NORM_REL_TOL,
        random_targets='Corresponding actual per-step delta L2 from this seed P_persistent_gain100, evaluated at the random arm actual current landing point',
        future_physical_candidates=list(FUTURE_PHYSICAL_CANDIDATES), physical_candidates_selected_before_scores=True,
        prefix_convention='Before attn20: readout/velocity[:20], action_states[:21] equals R. Before MLP21: readout/velocity[:21], action_states[:22] equals P.',
        first_reset_gate='Actual model kwargs at21 equals saved feedback P input21; actual L3 MLP before equals PP cache local-forward0 L3 MLP',
        later_reset_gate='Later actual incoming states are dynamic, not assumed equal to P or R.', scope=SCOPE)
    write_json(out / 'protocol.json', protocol)
    started, completed, results, controls, session = time.perf_counter(), [], [], [], None

    def progress(stage, run=None):
        value = dict(stage=stage, run=run, completed_predictions=len(completed), total_predictions=24,
                     completed_runs=completed, elapsed_s=time.perf_counter() - started)
        write_json(out / 'progress.json', value)
        print('[CORRECTION-PERSISTENCE] ' + json.dumps(value, allow_nan=False), flush=True)

    try:
        os.environ['HF_HUB_OFFLINE'] = '1'
        base = load_file('persistence_normal_runtime', paths['normal_runtime'])
        base.TASKS['milk'] = (7, 'pick_up_the_milk_and_place_it_in_the_basket', PROMPT, 'milk_1')
        factory = load_file('persistence_checked_factory', paths['factory'])
        factory.OUT = out
        assert sha(factory.SCHEDULER) == baseline_provenance['scheduler_sha256'] == feedback_provenance['scheduler_sha256']
        progress('loading_model')
        runtime = base.NormalRuntime(factory)
        torch, model = runtime.torch, runtime.pipe.transformer
        module = load_file('persistence_component_layout_contract', paths['components'])
        checker = module.CosmosComponentInterventions(model)
        session = TwoSiteSession(runtime, module, scan)
        actual_source = Path(inspect.getfile(type(model)))
        pipeline_source = Path(inspect.getfile(type(runtime.pipe)))
        assert sha(actual_source) == feedback_provenance['actual_transformer_sha256']
        assert sha(pipeline_source) == feedback_provenance['actual_pipeline_source_sha256']
        write_json(out / 'provenance.json', dict(script_sha256=sha(Path(__file__)), protocol_sha256=sha(out / 'protocol.json'),
            source_sha256={key: sha(path) for key, path in paths.items()}, pulse_metric_source_sha256=sha(scan_path),
            feedback_script_sha256=sha(feedback_script), feedback_complete_sha256=sha(feedback / 'complete.json'),
            feedback_provenance_sha256=sha(feedback / 'provenance.json'), feedback_summary_sha256=sha(feedback / 'summary.json'),
            actual_transformer_source=str(actual_source), actual_transformer_sha256=sha(actual_source),
            actual_pipeline_source_sha256=sha(pipeline_source), scheduler_sha256=sha(factory.SCHEDULER),
            library_edits=False, checkpoint_edits=False, closed_loop_execution=False, scope=SCOPE))
        for seed in SEEDS:
            folder = out / f'seed{seed}'
            (folder / 'runs').mkdir(parents=True, exist_ok=False)
            records, caches, p, r21, p21, pp_frame, sources = load_seed_sources(baseline, feedback, seed, runtime, checker, scan)
            r, d = records['R'], records['D']
            write_json(folder / 'sources.json', sources)
            torch.save(dict(recipient_pure_noise=r['pure_noise'], donor_pure_noise=d['pure_noise'],
                timesteps=r['timesteps'], sigmas=r['sigmas'], recipient_steps=caches['R']['steps'], donor_steps=caches['D']['steps'],
                R_actual_input21=r21, P_actual_input21=p21, PP_actual_L3_MLP_increment=pp_frame), folder / 'pair-contract.pt')
            targets = None
            for arm in ARMS:
                label = f'seed{seed}_{arm["name"]}'
                destination = folder / 'runs' / arm['name']
                progress('prediction_running', label)
                session.begin(arm, seed, caches['R'], caches['D'], r, r21, p21, pp_frame,
                              random_targets=targets if arm['mode'] == 'random' else None)
                try:
                    record, metadata = runtime.predict('milk', Path(sources['R']['image']), seed, destination)
                finally:
                    report = session.end()
                    if destination.exists():
                        torch.save(report, destination / 'actualsession.pt')
                        write_json(destination / 'actualsession.json', session_json(report, scan, torch))
                assert report['complete'] is True and report['n_model_forwards'] == report['finished_forwards'] == 30
                assert report['attention_site_calls'] == report['mlp_site_calls'] == 30
                assert report['explicit_processor_recomputations'] == 0 and report['nonaction_rows_bit_exact'] is True
                assert report['intervention_count'] == report['expected_intervention_count']
                scan.validate_run_contract(record, r, runtime, label)
                for key in ('readout', 'action_velocity'):
                    runtime.require_exact(record[key][:20], r[key][:20], label + '/native_before_attn20/' + key)
                runtime.require_exact(record['action_states'][:21], r['action_states'][:21], label + '/native_solver_before_attn20')
                if arm['pulse']:
                    for key in ('readout', 'action_velocity'):
                        runtime.require_exact(record[key][:21], p[key][:21], label + '/actual_P_before_MLP21/' + key)
                    runtime.require_exact(record['action_states'][:22], p['action_states'][:22], label + '/actual_P_solver_before_MLP21')
                expected = r if arm['name'] in ('native_R', 'R_persistent_self') else p if arm['name'] == 'P' else None
                if expected is not None:
                    runtime.require_exact(record, expected, label + '/ENTIRE_saved_control_record_bit_exact')
                    controls.append(dict(run=label, seed=seed, arm=arm['name'], all_saved_fields_bit_exact=True,
                        states_sha256=sha(destination / 'states.pt'), actualsession_sha256=sha(destination / 'actualsession.pt')))
                if arm['name'] == 'P_persistent_gain100':
                    targets = {step: report['events'][(step, RESET_LAYER, 'mlp')]['actual_delta_l2'] for step in RESET_STEPS}
                    write_json(folder / 'random-target-norms.json', dict(reference_arm=arm['name'],
                        actual_bf16_delta_l2_by_step=targets, reference_actualsession_sha256=sha(destination / 'actualsession.pt')))
                metrics = scan.prediction_metrics(r, d, record)
                metrics['final_R_to_P_axis'] = scan.direction_metrics(r['actions'][:, :3], p['actions'][:, :3], record['actions'][:, :3])
                metrics['axis_note'] = ('Each denoising metric uses that step own paired R/D reference, not a fixed axis across steps. '
                    'Sigma is fixed across arms at the same step. Final R-to-P axis is separate from final paired R-to-D axis; projections are continuous and may be negative or >1.')
                write_json(destination / 'metrics.json', metrics)
                norm_rows = [dict(step=event['step'], layer=event['layer'], component=event['component'],
                    mode=event['mode'], gain=event['gain'], actual_delta_l2=event['actual_delta_l2'],
                    exact_no_change=event['exact_no_change'], nonaction_rows_bit_exact=event['nonaction_rows_bit_exact'],
                    random_target_actual_delta_l2=None if event['random_calibration'] is None else event['random_calibration']['target_actual_delta_l2'],
                    random_actual_delta_l2=None if event['random_calibration'] is None else event['random_calibration']['actual_hook_delta_l2'],
                    random_relative_error=None if event['random_calibration'] is None else event['random_calibration']['actual_hook_relative_error'])
                    for _, event in sorted(report['events'].items())]
                write_json(destination / 'norms.json', dict(events=norm_rows, norm_relative_tolerance=NORM_REL_TOL))
                row = dict(run=label, seed=seed, arm=arm['name'], pulse=arm['pulse'], reset_steps=list(arm['reset_steps']),
                    gain=arm['gain'], intervention_count=report['intervention_count'], all_native_contract_fields_exact=True,
                    final_normalized_action_xyz=metrics['final_normalized_action_xyz'], final_R_to_P_axis=metrics['final_R_to_P_axis'],
                    final_normalized_xyz=record['actions'][:, :3].tolist(), normalized_action_shape=list(record['actions'].shape),
                    elapsed_s=metadata['elapsed_s'], sources_sha256=sha(folder / 'sources.json'),
                    artifacts=str(destination), files_sha256={name: sha(destination / name) for name in
                        ('states.pt', 'actualsession.pt', 'actualsession.json', 'metadata.json', 'normalized_actions.json', 'metrics.json', 'norms.json')})
                results.append(row)
                completed.append(label)
                write_json(out / 'results.json', dict(state='running', cases=results, controls=controls, completed=completed,
                    expected_predictions=24, future_physical_candidates=list(FUTURE_PHYSICAL_CANDIDATES), scope=SCOPE))
                progress('prediction_complete', label)
        assert len(results) == len(completed) == 24 and len(controls) == 9
        assert {row['run'] for row in results} == {f'seed{seed}_{arm["name"]}' for seed in SEEDS for arm in ARMS}
        write_json(out / 'results.json', dict(state='complete', predictions=24, cases=results, controls=controls,
            completed=completed, protocol_sha256=sha(out / 'protocol.json'), future_physical_candidates=list(FUTURE_PHYSICAL_CANDIDATES),
            physical_selection_predeclared=True,
            interpretation='Short or sustained increment replacement tests whether this candidate computation changes the later trajectory of internal actions. Continued changes can be inherited by solver state; no actual grasp rescue is established.',
            next_validation='Execute predeclared seed198 native_R/P/gain100/random candidates in the same MuJoCo scene, then vary milk and distractor positions if behavior changes.', scope=SCOPE))
        write_json(out / 'complete.json', dict(state='complete', q0_predictions=24, exact_full_record_controls=9,
            all_two_draw_noise_schedule_prepared_contracts_exact=True, all_site_calls_30_and_nonaction_rows_exact=True,
            first_reset_actual_input_and_PP_increment_exact=True, actual_random_norms_matched_per_step=True,
            results_sha256=sha(out / 'results.json'), protocol_sha256=sha(out / 'protocol.json'),
            provenance_sha256=sha(out / 'provenance.json'), script_sha256=sha(Path(__file__)), scope=SCOPE))
        progress('complete')
    except Exception as exc:
        write_json(out / 'failed.json', dict(state='failed', error=str(exc), traceback=traceback.format_exc(),
            completed_runs=completed, elapsed_s=time.perf_counter() - started, scope=SCOPE))
        raise
    finally:
        if session is not None:
            session.reset()


if __name__ == '__main__':
    main()
