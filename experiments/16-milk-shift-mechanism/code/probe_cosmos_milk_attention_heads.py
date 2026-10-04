"""Localize the measured L17 attention/denoising-step29 effect to real heads.

Own only the absent BASELINE/attention-heads child. The completed coarse scan
must rank layer17_attention_step29 first, with x06_seed198 donor and
x15_seed198 recipient. Hook the actual to_add_out INPUT, whose verified
flattening is [GEN, 32, 128], not 4096 post-projection coordinates. No attention
processor recomputation, library edit, new layer, ROI or MuJoCo rollout.

Four controls precede 32 single-head donor swaps: normal R/D site captures must
match their entire baseline records; all-head self-patch must match R; all-head
donor patch must match the earlier coarse post-projection intervention's ENTIRE
record. Every pulse acts only once, at step29 on the sixteen real action rows.
Verify actual before/after arrays for unchanged non-action rows and unselected
heads. Then test the strongest head in reverse and against three independent
CPU-generated random changes with calibrated, actual bf16 delta norm.

Scores describe continuous fixed-input q0 normalized XYZ outputs. They are not
grasp probabilities, pure object identity or proof of a repaired grasp/root
cause. Raw XYZ values and quantization-aware random-control norms are saved.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time
import traceback


ROOT = Path('/home/current/work/cosmos3')
LAYER, PULSE_STEP = 17, 29
NUM_HEADS, HEAD_DIM = 32, 128
SEED = 198
RECIPIENT_TRIAL, DONOR_TRIAL = 'x15_seed198', 'x06_seed198'
COARSE_RUN = 'layer17_attention_step29'
PROMPT = 'pick up the milk and place it in the basket'
RANDOM_SEEDS = (17002901, 17002902, 17002903)
RANDOM_NORM_REL_TOL = 1e-3
SCOPE = 'Fixed-input q0 real pre-output-projection attention-head causal effects only; no MuJoCo, identity, attractor or root-cause claim.'


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


def tensor_meta(value, torch):
    raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return dict(shape=list(value.shape), dtype=str(value.dtype), sha256=hashlib.sha256(raw).hexdigest())


class HeadSite:
    """One verified pre-W site, counted once in each of thirty model forwards."""

    def __init__(self, runtime, component_module):
        self.runtime, self.module = runtime, component_module
        self.model = runtime.pipe.transformer
        self.attn = self.model.layers[LAYER].self_attn
        self.projection = self.attn.to_add_out
        assert len(self.model.layers) == 36
        assert self.attn.num_attention_heads == NUM_HEADS and self.attn.head_dim == HEAD_DIM
        assert self.projection.in_features == self.projection.out_features == NUM_HEADS * HEAD_DIM == 4096
        assert not self.model.training and not runtime.torch.is_grad_enabled()
        assert getattr(self.model, '_cp_shard_fn', None) is None and getattr(self.model, '_cp_gather_fn', None) is None
        assert getattr(self.attn.processor, '_parallel_config', None) is None
        self.handles, self.active = [], False

    def begin(self, reference_cache, label, *, original_z=None, replacement_z=None, heads=None):
        assert not self.active, 'End the preceding head session first'
        assert reference_cache['complete'] and reference_cache['n_steps'] == len(reference_cache['steps']) == 30
        selected = () if heads is None else tuple(heads)
        if replacement_z is not None:
            assert selected and len(set(selected)) == len(selected)
            assert all(type(head) is int and 0 <= head < NUM_HEADS for head in selected)
            assert original_z.shape == replacement_z.shape == (16, NUM_HEADS, HEAD_DIM)
            assert original_z.dtype == replacement_z.dtype == self.runtime.torch.bfloat16
            assert self.runtime.torch.isfinite(replacement_z).all()
        else:
            assert not selected and original_z is None
        self.reference, self.original_z, self.replacement_z = reference_cache, original_z, replacement_z
        self.heads, self.label = selected, label
        self.step, self.inside, self.active = -1, False, True
        self.pre_calls, self.post_calls, self.finished = set(), set(), set()
        self.frames, self.observed_steps, self.first_kwargs = {}, [], None
        try:
            self.handles.append(self.model.register_forward_pre_hook(self.before_model, with_kwargs=True))
            self.handles.append(self.model.register_forward_hook(self.after_model))
            self.handles.append(self.projection.register_forward_pre_hook(self.before_site))
            self.handles.append(self.projection.register_forward_hook(self.after_site))
        except BaseException:
            self.reset()
            raise

    def before_model(self, model, args, kwargs):
        assert not args and not self.inside, 'Keyword-only, unnested model forward required'
        self.inside = True
        self.step += 1
        assert self.step < 30, 'Extra forward/CFG/recomputation unsupported'
        self.layout = self.module._layout(kwargs)
        assert self.layout['action_rows'].numel() == 16
        frame = dict(layout=self.layout, metadata={key: self.runtime.cpu(kwargs.get(key))
                                                    for key in self.module._STRUCTURAL_KEYS})
        self.runtime.require_exact(frame, self.reference['steps'][self.step], f'{self.label}/all_step_pack/{self.step}')
        self.observed_steps.append(frame)
        if self.step == 0:
            self.first_kwargs = self.runtime.cpu(kwargs)
            self.runtime.require_exact(self.first_kwargs, self.reference['first_model_kwargs'], f'{self.label}/initial_kwargs')

    def before_site(self, projection, args):
        torch = self.runtime.torch
        assert self.inside and self.step not in self.pre_calls, 'Duplicate/outside-forward pre-W site call'
        assert len(args) == 1 and isinstance(args[0], torch.Tensor)
        gen = args[0]
        assert tuple(gen.shape) == (self.layout['gen_len'], NUM_HEADS * HEAD_DIM)
        assert gen.dtype == torch.bfloat16
        self.pre_calls.add(self.step)
        if self.step != PULSE_STEP:
            return
        rows = self.layout['action_rows'].to(gen.device)
        before_z = self.runtime.cpu(gen.index_select(0, rows).reshape(16, NUM_HEADS, HEAD_DIM))
        self.frames['z_before'] = before_z
        if self.replacement_z is None:
            self.frames.update(z_after=before_z.clone(), intervention_count=0)
            return
        self.runtime.require_exact(before_z, self.original_z, f'{self.label}/native_pre_W_action_z')
        changed_actions = before_z.clone()
        changed_actions[:, self.heads, :] = self.replacement_z[:, self.heads, :]
        altered = gen.clone()
        altered.index_copy_(0, rows, changed_actions.reshape(16, 4096).to(gen.device))
        # Check actual whole GEN input before/after, without inferring untouched
        # rows from the fact that index_copy_ was requested.
        before_all, after_all = self.runtime.cpu(gen), self.runtime.cpu(altered)
        cpu_rows = self.layout['action_rows']
        after_z = after_all.index_select(0, cpu_rows).reshape(16, NUM_HEADS, HEAD_DIM)
        untouched_heads = [head for head in range(NUM_HEADS) if head not in self.heads]
        non_action = torch.arange(self.layout['gen_len'])
        non_action = non_action[~torch.isin(non_action, cpu_rows)]
        untouched_before, untouched_after = before_all.index_select(0, non_action), after_all.index_select(0, non_action)
        self.runtime.require_exact(untouched_before, untouched_after, f'{self.label}/every_non_action_row')
        self.runtime.require_exact(before_z[:, untouched_heads, :], after_z[:, untouched_heads, :],
                                   f'{self.label}/every_unselected_action_head')
        self.runtime.require_exact(after_z[:, self.heads, :], self.replacement_z[:, self.heads, :],
                                   f'{self.label}/exact_selected_replacement')
        actual_delta = after_z[:, self.heads, :].double() - before_z[:, self.heads, :].double()
        self.frames.update(z_after=after_z, selected_delta=actual_delta, intervention_count=1,
            actual_selected_delta_l2=float(torch.linalg.vector_norm(actual_delta)),
            unselected_action_heads_exact=True, all_non_action_rows_exact=True,
            non_action_row_count=int(non_action.numel()),
            non_action_before=tensor_meta(untouched_before, torch), non_action_after=tensor_meta(untouched_after, torch),
            full_gen_before=tensor_meta(before_all, torch), full_gen_after=tensor_meta(after_all, torch))
        if torch.equal(before_all, after_all):
            return
        return (altered,)

    def after_site(self, projection, args, output):
        assert self.inside and self.step in self.pre_calls and self.step not in self.post_calls
        assert tuple(output.shape) == (self.layout['gen_len'], 4096)
        self.post_calls.add(self.step)
        if self.step == PULSE_STEP:
            rows = self.layout['action_rows'].to(output.device)
            self.frames['post_W_action_increment'] = self.runtime.cpu(output.index_select(0, rows))

    def after_model(self, model, args, output):
        assert self.step in self.pre_calls and self.step in self.post_calls
        self.finished.add(self.step)
        self.inside = False

    def reset(self):
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        self.active, self.inside = False, False

    def end(self):
        assert self.active, 'No active head session'
        expected = set(range(30))
        complete = (self.step == 29 and self.pre_calls == self.post_calls == self.finished == expected
                    and len(self.observed_steps) == 30 and 'post_W_action_increment' in self.frames)
        result = dict(version=1, label=self.label, complete=complete,
            site='self_attn.to_add_out INPUT, after full attention, before output projection',
            layer=LAYER, denoising_step=PULSE_STEP, num_heads=NUM_HEADS, head_dim=HEAD_DIM,
            selected_heads=self.heads, n_model_forwards=self.step + 1,
            pre_W_site_calls=len(self.pre_calls), post_W_site_calls=len(self.post_calls),
            finished_forwards=len(self.finished), processor_recomputations=0,
            steps=self.observed_steps, first_model_kwargs=self.first_kwargs,
            action_rows=self.reference['steps'][0]['layout']['action_rows'], **self.frames)
        self.reset()
        return result


def site_metadata(report, torch):
    result = {key: value for key, value in report.items()
              if key not in ('steps', 'first_model_kwargs', 'action_rows', 'z_before', 'z_after',
                             'selected_delta', 'post_W_action_increment')}
    result['action_rows'] = report['action_rows'].tolist()
    for key in ('z_before', 'z_after', 'selected_delta', 'post_W_action_increment'):
        if key in report:
            result[key] = tensor_meta(report[key], torch)
    return result


def make_random_replacement(r_z, d_z, head, seed, torch):
    """Calibrate norm AFTER bf16 quantization, using an independent CPU RNG."""
    target_delta = d_z[:, head, :].double() - r_z[:, head, :].double()
    target = float(torch.linalg.vector_norm(target_delta))
    assert math.isfinite(target) and target > 0, 'Selected head has no donor delta to norm-match'
    generator = torch.Generator(device='cpu').manual_seed(seed)
    raw = torch.randn((16, HEAD_DIM), generator=generator, device='cpu', dtype=torch.float64)
    unit = raw / torch.linalg.vector_norm(raw)

    def candidate(scale):
        value = (r_z[:, head, :].double() + unit * scale).to(r_z.dtype)
        delta = value.double() - r_z[:, head, :].double()
        return value, delta, float(torch.linalg.vector_norm(delta))

    lo, hi = 0., target * 2
    for _ in range(16):
        if candidate(hi)[2] >= target:
            break
        hi *= 2
    else:
        raise RuntimeError('Could not bracket quantized random delta norm')
    best, iterations = None, 0
    for iteration in range(64):
        scale = (lo + hi) / 2
        value, delta, norm = candidate(scale)
        error = abs(norm - target) / target
        if best is None or error < best[0]:
            best = (error, value, delta, norm, scale)
        iterations = iteration + 1
        if error <= RANDOM_NORM_REL_TOL:
            break
        if norm < target:
            lo = scale
        else:
            hi = scale
    error, value, delta, norm, scale = best
    assert error <= RANDOM_NORM_REL_TOL, f'Quantized random delta norm mismatch: {error}'
    replacement = r_z.clone()
    replacement[:, head, :] = value
    return replacement, dict(cpu_generator_seed=seed, head=head, target_actual_donor_delta_l2=target,
        actual_quantized_random_delta_l2=norm, relative_norm_error=error, relative_norm_tolerance=RANDOM_NORM_REL_TOL,
        scale_before_quantization=scale, calibration_iterations=iterations,
        raw_gaussian=raw, unit_direction=unit, actual_quantized_delta=delta,
        actual_cosine_with_donor_delta=float((delta * target_delta).sum() / (norm * target)))


def output_metrics(recipient, donor, altered, pulses):
    return dict(final_normalized_action_xyz=pulses.direction_metrics(
                    recipient['actions'][:, :3], donor['actions'][:, :3], altered['actions'][:, :3]),
                final_normalized_action_all10=pulses.direction_metrics(recipient['actions'], donor['actions'], altered['actions']),
                readout_at_pulse=pulses.direction_metrics(recipient['readout'][PULSE_STEP], donor['readout'][PULSE_STEP], altered['readout'][PULSE_STEP]),
                velocity_xyz_at_pulse=pulses.direction_metrics(recipient['action_velocity'][PULSE_STEP, :, :3],
                    donor['action_velocity'][PULSE_STEP, :, :3], altered['action_velocity'][PULSE_STEP, :, :3]),
                recipient_normalized_xyz=recipient['actions'][:, :3].tolist(),
                donor_normalized_xyz=donor['actions'][:, :3].tolist(),
                altered_normalized_xyz=altered['actions'][:, :3].tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Completed baseline root with completed component-pulses')
    args = parser.parse_args()
    baseline = args.output.resolve()
    pulse_path = ROOT / 'work/probe_cosmos_milk_component_pulses.py'
    pulses = load_file('head_checked_pulse_functions', pulse_path)
    _, by_trial = pulses.validate_baseline(baseline)
    assert set(by_trial[RECIPIENT_TRIAL]['selected_objects']) == {'cream_cheese_1'}
    assert set(by_trial[DONOR_TRIAL]['selected_objects']) == {'milk_1'}
    scan = baseline / 'component-pulses'
    assert not (scan / 'failed.json').exists()
    coarse_complete = read_json(scan / 'complete.json')
    assert coarse_complete['state'] == 'complete' and coarse_complete['scan_predictions'] == 56
    assert coarse_complete['control_predictions'] == 4 and coarse_complete['paired_seed'] == SEED
    assert coarse_complete['results_sha256'] == sha(scan / 'results.json')
    assert coarse_complete['validation_sha256'] == sha(scan / 'validation.json')
    assert coarse_complete['script_sha256'] == sha(pulse_path), 'Original direction-metric/scan source changed'
    coarse_results, coarse_validation = read_json(scan / 'results.json'), read_json(scan / 'validation.json')
    assert coarse_results['state'] == 'complete' and len(coarse_results['cases']) == 56
    assert coarse_validation['state'] == 'all_exact_controls_passed' and len(coarse_validation['controls']) == 4
    assert all(row['all_normal_runtime_fields_bit_exact'] is True for row in coarse_validation['controls'])
    assert coarse_results['ranked_runs'][0] == COARSE_RUN, 'Measured top component differs: refuse another site'
    coarse_row = coarse_results['top_three_candidates'][0]
    assert coarse_row['run'] == COARSE_RUN and coarse_row['layer'] == LAYER
    assert coarse_row['component'] == 'attention' and coarse_row['pulse_step'] == PULSE_STEP
    assert coarse_row['recipient_trial'] == RECIPIENT_TRIAL and coarse_row['donor_trial'] == DONOR_TRIAL
    coarse_destination = scan / 'runs' / COARSE_RUN
    for key, filename in (('states_sha256', 'states.pt'), ('patch_report_sha256', 'patch_report.pt'),
                           ('runtime_metadata_sha256', 'metadata.json')):
        assert coarse_row[key] == sha(coarse_destination / filename)
    out = baseline / 'attention-heads'
    out.mkdir(exist_ok=False)
    began, completed, site = time.perf_counter(), [], None

    def progress(state, run=None):
        value = dict(state=state, run=run, completed_predictions=len(completed), total_predictions=40,
                     completed_runs=completed, elapsed_s=time.perf_counter() - began)
        write_json(out / 'progress.json', value)
        print('[ATTENTION-HEADS] ' + json.dumps(value), flush=True)

    try:
        os.environ['HF_HUB_OFFLINE'] = '1'
        paths = dict(normal_runtime=ROOT / 'work/serve_cosmos_goal_pair.py', factory=ROOT / 'work/probe_cosmos_language_routes.py',
                     components=ROOT / 'work/cosmos_component_interventions.py', direction_metrics=pulse_path)
        baseline_provenance = read_json(baseline / 'server/provenance.json')
        for key, field in (('normal_runtime', 'normal_runtime_sha256'), ('factory', 'factory_sha256'), ('components', 'components_sha256')):
            assert sha(paths[key]) == baseline_provenance[field], f'Baseline {key} changed'
        base = load_file('head_normal_runtime', paths['normal_runtime'])
        base.TASKS['milk'] = (7, 'pick_up_the_milk_and_place_it_in_the_basket', PROMPT, 'milk_1')
        factory = load_file('head_checked_factory', paths['factory'])
        factory.OUT = out
        assert sha(factory.SCHEDULER) == baseline_provenance['scheduler_sha256']
        progress('loading_model')
        runtime = base.NormalRuntime(factory)
        torch = runtime.torch
        module = load_file('head_component_layout_contract', paths['components'])
        site = HeadSite(runtime, module)
        checker = module.CosmosComponentInterventions(runtime.pipe.transformer)
        import inspect
        actual_source = Path(inspect.getfile(type(runtime.pipe.transformer)))
        write_json(out / 'provenance.json', dict(script_sha256=sha(Path(__file__)),
            sources={key: dict(path=str(path), sha256=sha(path)) for key, path in paths.items()},
            actual_transformer_source=str(actual_source), actual_transformer_sha256=sha(actual_source),
            baseline_server_provenance=baseline_provenance, coarse_complete_sha256=sha(scan / 'complete.json'),
            coarse_results_sha256=sha(scan / 'results.json'), coarse_states_sha256=sha(coarse_destination / 'states.pt'),
            fixed_site=dict(layer_zero_based=LAYER, layer_human=LAYER + 1, denoising_step=PULSE_STEP,
                            num_heads=NUM_HEADS, head_dim=HEAD_DIM, input_width=4096),
            donor_trial=DONOR_TRIAL, recipient_trial=RECIPIENT_TRIAL, seed=SEED,
            random_seeds=list(RANDOM_SEEDS), actual_random_norm_relative_tolerance=RANDOM_NORM_REL_TOL,
            control_predictions=4, single_head_predictions=32, followup_predictions=4,
            library_edits=False, processor_recomputation=False, scope=SCOPE))
        records, caches, sources = {}, {}, {}
        for role, trial in (('recipient', RECIPIENT_TRIAL), ('donor', DONOR_TRIAL)):
            run = baseline / 'closed-loop' / trial
            records[role] = torch.load(run / 'chunk_00/states.pt', map_location='cpu', weights_only=True, mmap=True)
            caches[role] = torch.load(run / 'chunk_00/components.pt', map_location='cpu', weights_only=True, mmap=True)
            pulses.validate_record(records[role], torch)
            pulses.validate_cache(caches[role], records[role], checker, runtime, role)
            sources[role] = dict(trial=trial, image=str(run / 'input_00.png'), image_sha256=sha(run / 'input_00.png'),
                states_sha256=sha(run / 'chunk_00/states.pt'), components_sha256=sha(run / 'chunk_00/components.pt'))
        r, d = records['recipient'], records['donor']
        r_cache, d_cache = caches['recipient'], caches['donor']
        for key in ('pure_noise', 'timesteps', 'sigmas'):
            runtime.require_exact(r[key], d[key], f'paired_full_{key}')
        runtime.require_exact(r_cache['steps'], d_cache['steps'], 'paired_entire_30step_metadata')
        runtime.require_exact(r['model_input']['action_tokens'], d['model_input']['action_tokens'], 'paired_prepared_action_noise')
        coarse_record = torch.load(coarse_destination / 'states.pt', map_location='cpu', weights_only=True, mmap=True)
        pulses.validate_run_contract(coarse_record, r, runtime, 'coarse_reference')
        contract = dict(state='raw_pair_verified', sources=sources, two_fp32_noise_draws_exact=True,
            full_30step_metadata_exact=True, timesteps_sigmas_exact=True, raw_prepared_action_noise_exact=True,
            controls=[], scope=SCOPE)
        write_json(out / 'validation.json', contract)
        torch.save(dict(recipient_pure_noise=r['pure_noise'], donor_pure_noise=d['pure_noise'],
                        timesteps=r['timesteps'], sigmas=r['sigmas'], recipient_step_metadata=r_cache['steps'],
                        donor_step_metadata=d_cache['steps']), out / 'pair_contract.pt')
        for name in ('controls', 'runs', 'followups'):
            (out / name).mkdir(exist_ok=False)

        def prediction(label, destination, role, *, original_z=None, replacement_z=None, heads=None):
            progress('prediction_running', label)
            site.begin(caches[role], label, original_z=original_z, replacement_z=replacement_z, heads=heads)
            try:
                record, metadata = runtime.predict('milk', Path(sources[role]['image']), SEED, destination)
            finally:
                report = site.end()
                if destination.exists():
                    torch.save(report, destination / 'head_site.pt')
                    write_json(destination / 'head_site.json', site_metadata(report, torch))
            assert report['complete'] and report['n_model_forwards'] == 30
            assert report['pre_W_site_calls'] == report['post_W_site_calls'] == report['finished_forwards'] == 30
            assert report['processor_recomputations'] == 0
            pulses.validate_run_contract(record, records[role], runtime, label)
            for key in ('readout', 'action_velocity'):
                runtime.require_exact(record[key][:PULSE_STEP], records[role][key][:PULSE_STEP], f'{label}/native_prefix/{key}')
            runtime.require_exact(record['action_states'][:PULSE_STEP + 1], records[role]['action_states'][:PULSE_STEP + 1],
                                  f'{label}/native_solver_prefix')
            if replacement_z is not None:
                assert report['intervention_count'] == 1
                assert report['all_non_action_rows_exact'] and report['unselected_action_heads_exact']
            else:
                assert report['intervention_count'] == 0
            return record, metadata, report

        captures = {}
        for role in ('recipient', 'donor'):
            label, destination = f'capture_{role}', out / 'controls' / f'capture_{role}'
            record, _, report = prediction(label, destination, role)
            runtime.require_exact(record, records[role], f'{role}/full_normal_baseline_repeat')
            runtime.require_exact(report['post_W_action_increment'], caches[role]['frames'][(PULSE_STEP, LAYER, 'attention')],
                                  f'{role}/actual_W_output_equals_native_component')
            captures[role] = report
            completed.append(label)
            contract['controls'].append(dict(run=label, all_saved_baseline_fields_bit_exact=True,
                post_W_equals_native_attention_increment=True, states_sha256=sha(destination / 'states.pt'),
                head_site_sha256=sha(destination / 'head_site.pt')))
            write_json(out / 'validation.json', contract)
            progress('control_passed', label)
        r_z, d_z = captures['recipient']['z_before'], captures['donor']['z_before']
        runtime.require_exact(captures['recipient']['steps'], captures['donor']['steps'], 'captured_R_D_entire_metadata')
        torch.save(dict(recipient_z=r_z, donor_z=d_z, recipient_capture=captures['recipient'], donor_capture=captures['donor']),
                   out / 'actual_head_captures.pt')
        for label, replacement, expected in (('self_all32', r_z, r), ('donor_all32_coarse_equivalence', d_z, coarse_record)):
            destination = out / 'controls' / label
            record, _, report = prediction(label, destination, 'recipient', original_z=r_z,
                                           replacement_z=replacement, heads=list(range(NUM_HEADS)))
            runtime.require_exact(record, expected, f'{label}/ENTIRE_record_bit_equivalence')
            if label == 'self_all32':
                assert report['actual_selected_delta_l2'] == 0
            else:
                runtime.require_exact(report['post_W_action_increment'], d_cache['frames'][(PULSE_STEP, LAYER, 'attention')],
                                      'all32_pre_W_equals_coarse_post_W_donor_increment')
            completed.append(label)
            contract['controls'].append(dict(run=label, all_reference_record_fields_bit_exact=True,
                states_sha256=sha(destination / 'states.pt'), head_site_sha256=sha(destination / 'head_site.pt')))
            write_json(out / 'validation.json', contract)
            progress('control_passed', label)
        contract['state'] = 'all_four_exact_controls_passed'
        contract['all32_pre_W_entire_record_equals_coarse_post_W'] = True
        write_json(out / 'validation.json', contract)
        results = []

        def result_row(label, destination, head, role, record, metadata, report, mode, reference, donor_reference):
            metrics = output_metrics(reference, donor_reference, record, pulses)
            write_json(destination / 'metrics.json', metrics)
            return dict(run=label, mode=mode, head_zero_based=head, head_human=head + 1, layer=LAYER,
                denoising_step=PULSE_STEP, recipient_role=role, intervention_count=1,
                actual_selected_delta_l2=report['actual_selected_delta_l2'],
                final_normalized_action_xyz=metrics['final_normalized_action_xyz'],
                final_normalized_action_all10=metrics['final_normalized_action_all10'],
                normalized_xyz=metrics['altered_normalized_xyz'], elapsed_s=metadata['elapsed_s'],
                states_path=str(destination / 'states.pt'), states_sha256=sha(destination / 'states.pt'),
                head_site_sha256=sha(destination / 'head_site.pt'), metrics_sha256=sha(destination / 'metrics.json'))

        for head in range(NUM_HEADS):
            label, destination = f'head{head:02d}', out / 'runs' / f'head{head:02d}'
            record, metadata, report = prediction(label, destination, 'recipient', original_z=r_z, replacement_z=d_z, heads=[head])
            results.append(result_row(label, destination, head, 'recipient', record, metadata, report, 'natural_donor', r, d))
            completed.append(label)
            write_json(out / 'results.json', dict(state='running', single_heads=results, completed_single_heads=len(results), scope=SCOPE))
            progress('single_head_complete', label)
        ranked = sorted((row for row in results if row['final_normalized_action_xyz']['signed_projection_fraction'] is not None),
                        key=lambda row: row['final_normalized_action_xyz']['signed_projection_fraction'], reverse=True)
        assert ranked, 'No meaningful fixed-input baseline XYZ axis; refuse forced head ranking'
        top_head = ranked[0]['head_zero_based']
        followups = []
        # Reverse uses the D capture and D input/metadata contract, not an R
        # cache replay on another observation. All thirty metadata calls refresh.
        label, destination = f'reverse_head{top_head:02d}', out / 'followups' / f'reverse_head{top_head:02d}'
        record, metadata, report = prediction(label, destination, 'donor', original_z=d_z, replacement_z=r_z, heads=[top_head])
        followups.append(result_row(label, destination, top_head, 'donor', record, metadata, report, 'reverse_natural_donor', d, r))
        completed.append(label)
        progress('followup_complete', label)
        for random_seed in RANDOM_SEEDS:
            replacement, random_info = make_random_replacement(r_z, d_z, top_head, random_seed, torch)
            label, destination = f'random_head{top_head:02d}_seed{random_seed}', out / 'followups' / f'random_head{top_head:02d}_seed{random_seed}'
            record, metadata, report = prediction(label, destination, 'recipient', original_z=r_z, replacement_z=replacement, heads=[top_head])
            runtime.require_exact(report['selected_delta'][:, 0, :], random_info['actual_quantized_delta'], f'{label}/actual_random_delta')
            assert abs(report['actual_selected_delta_l2'] - random_info['target_actual_donor_delta_l2']) / random_info['target_actual_donor_delta_l2'] <= RANDOM_NORM_REL_TOL
            torch.save(dict(replacement_z=replacement, **random_info), destination / 'random_control.pt')
            random_meta = {key: value for key, value in random_info.items() if not isinstance(value, torch.Tensor)}
            write_json(destination / 'random_control.json', random_meta)
            row = result_row(label, destination, top_head, 'recipient', record, metadata, report, 'actual_norm_matched_random', r, d)
            row['random_control'] = random_meta
            row['random_control_sha256'] = sha(destination / 'random_control.pt')
            followups.append(row)
            completed.append(label)
            progress('followup_complete', label)
        assert len(results) == 32 and len(contract['controls']) == 4 and len(followups) == 4 and len(completed) == 40
        final = dict(state='complete', single_heads=results, ranked_heads=[row['head_zero_based'] for row in ranked],
            top_three_heads=ranked[:3], followups=followups, controls=contract['controls'],
            baseline_normalized_xyz=dict(recipient=r['actions'][:, :3].tolist(), donor=d['actions'][:, :3].tolist(),
                                         coarse_all32=coarse_record['actions'][:, :3].tolist()),
            predictions=40, single_head_predictions=32, followup_predictions=4, control_predictions=4,
            score='Signed continuous normalized XYZ projection on paired donor-minus-recipient axis; not grasp probability.',
            scope=SCOPE)
        write_json(out / 'results.json', final)
        write_json(out / 'complete.json', dict(state='complete', predictions=40, controls=4, single_heads=32, followups=4,
            all32_pre_W_entire_record_equals_coarse_post_W=True, all_actual_non_action_and_unselected_head_checks_exact=True,
            all_noise_schedule_pack_prefix_checks_exact=True, results_sha256=sha(out / 'results.json'),
            validation_sha256=sha(out / 'validation.json'), captures_sha256=sha(out / 'actual_head_captures.pt'),
            script_sha256=sha(Path(__file__)), direction_metric_source_sha256=sha(pulse_path), scope=SCOPE))
        progress('complete')
    except Exception as exc:
        write_json(out / 'failed.json', dict(state='failed', type=type(exc).__name__, error=str(exc),
            traceback=traceback.format_exc(), completed_runs=completed, elapsed_s=time.perf_counter() - began, scope=SCOPE))
        raise
    finally:
        if site is not None:
            site.reset()


if __name__ == '__main__':
    main()
