"""Locate local computation of the fixed-t21 action-input reverse response.

Use completed feedback-inputs/seed198/replays/{RR,PR}. PR has the pulse-P
action sample and R vision; RR has native R action and the same R vision.
Keep actual PR kwargs unchanged. Replace one of 36 attention/GEN-MLP action
increments with its saved RR value, and let the remaining layers respond.

Four exact controls precede 72 interventions: native RR/PR and PR self-patch
attention/MLP. Follow the strongest positive XYZ restoration with RR<-PR
and three independent CPU random changes norm-matched AFTER bf16 rounding.
The script owns only an absent action-correction-sites child of --output.
No scheduler step, training, inference-library edit or MuJoCo execution.
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
SEED, REPLAY_STEP = 198, 21
COMPONENTS = ('attention', 'mlp')
SELF_LAYER = 17
RANDOM_SEEDS = (21019801, 21019802, 21019803)
RANDOM_NORM_REL_TOL = 1e-3
SCOPE = ('Single seed198, actual fixed t21 PR action-only inputs and unchanged R vision/sigma. '
         'One local action-row component increment before residual addition. '
         'Downstream layers respond freely. Raw Flow velocity effects only; '
         'no target-identity, success-probability, attractor or grasp-root-cause claim.')


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


def to_device(value, torch, device):
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: to_device(item, torch, device) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(to_device(item, torch, device) for item in value)
    return value


def require_finite(value, torch):
    if isinstance(value, torch.Tensor):
        assert torch.isfinite(value).all(), 'Nonfinite actual transformer output'
    elif isinstance(value, dict):
        for item in value.values():
            require_finite(item, torch)
    elif isinstance(value, (tuple, list)):
        for item in value:
            require_finite(item, torch)


def validate_single_cache(cache, hooks, runtime, label):
    torch = runtime.torch
    expected = {(0, layer, component) for layer in range(36) for component in COMPONENTS}
    assert cache['version'] == 1 and cache['complete'] is True
    assert cache['n_steps'] == len(cache['steps']) == cache['expected_steps'] == 1
    assert cache['n_layers'] == len(hooks.layers) == 36
    assert set(cache['frames']) == expected
    assert cache['captured_component_count'] == cache['component_call_count'] == 72
    assert cache['processor_recompute_calls'] == 0
    assert not cache['missing_component_calls'] and not cache['unfinished_forwards']
    runtime.require_exact(cache['site_specs'], hooks.site_specs, label + '/actual_model_sites')
    kwargs = cache['first_model_kwargs']
    rows = kwargs['action_mse_loss_indexes'].flatten() - int(kwargs['und_len'])
    assert rows.dtype == torch.long and rows.numel() == rows.unique().numel() == 16
    assert bool(((rows >= 0) & (rows < int(kwargs['sequence_length']) - int(kwargs['und_len']))).all())
    runtime.require_exact(cache['steps'][0]['layout']['action_rows'], rows, label + '/action_rows')
    assert cache['steps'][0]['layout']['gen_len'] == int(kwargs['sequence_length']) - int(kwargs['und_len'])
    for key, frame in cache['frames'].items():
        assert frame.shape == (16, 4096) and frame.dtype == torch.bfloat16, (label, key)
        assert torch.isfinite(frame).all(), (label, key)
    for key, value in cache['steps'][0]['metadata'].items():
        runtime.require_exact(value, kwargs.get(key), label + '/actual_metadata/' + key)
    return rows


def load_reference(folder, hooks, runtime, scan, label):
    """NPZ floats must round-trip to the actual recorded bf16 head bytes."""
    import numpy as np
    torch = runtime.torch
    metadata = read_json(folder / 'metadata.json')
    assert metadata['label'] == f'seed{SEED}_{label}'
    assert metadata['original_denoising_forward'] == REPLAY_STEP
    assert metadata['actual_boundaries'] == 37 and metadata['component_calls'] == 72
    assert metadata['component_cache_complete'] is True and metadata['component_forward_count'] == 1
    assert metadata['all_observation_hooks_removed'] is True
    for key, filename in (('arrays_sha256', 'arrays.npz'), ('components_sha256', 'components.pt'),
                          ('model_output_sha256', 'model_output.pt')):
        assert metadata[key] == sha(folder / filename), f'{label}/{filename} artifact changed'
    cache = torch.load(folder / 'components.pt', map_location='cpu', weights_only=True, mmap=True)
    rows = validate_single_cache(cache, hooks, runtime, label)
    runtime.require_exact(rows.tolist(), metadata['action_rows'], label + '/saved_rows')
    tensors = {}
    with np.load(folder / 'arrays.npz', allow_pickle=False) as arrays:
        for key, shape in (('action_boundary_hidden', (37, 16, 4096)),
                           ('readout', (16, 4096)), ('action_velocity', (16, 64))):
            raw = torch.from_numpy(arrays[key].copy())
            assert raw.dtype == torch.float32 and raw.shape == shape and torch.isfinite(raw).all()
            tensor = raw.to(torch.bfloat16)
            runtime.require_exact(tensor.float(), raw, label + '/lossless_saved_bf16/' + key)
            tensors[key] = tensor
    for key, name in (('readout', 'readout'), ('action_velocity', 'velocity')):
        runtime.require_exact(scan.tensor_meta(tensors[key], torch), metadata[name], label + '/actual_head_hash/' + key)
    output = torch.load(folder / 'model_output.pt', map_location='cpu', weights_only=True, mmap=True)
    require_finite(output, torch)
    return dict(cache=cache, kwargs=cache['first_model_kwargs'], hidden=tensors['action_boundary_hidden'],
                readout=tensors['readout'], velocity=tensors['action_velocity'], model_output=output,
                metadata=metadata, folder=folder)


class SingleIncrementPatch:
    """Intentional RR/PR action-input mismatch; no initial-noise gate bypass."""

    def __init__(self, runtime, reference, layer, component, replacement):
        self.runtime, self.torch = runtime, runtime.torch
        self.reference, self.layer, self.component = reference, layer, component
        self.replacement = replacement
        self.rows = reference['cache']['steps'][0]['layout']['action_rows']
        self.expected_before = reference['cache']['frames'][(0, layer, component)]
        assert component in COMPONENTS and 0 <= layer < 36
        assert replacement.shape == self.expected_before.shape and replacement.dtype == self.expected_before.dtype
        assert self.torch.isfinite(replacement).all()
        self.events = []

    def __call__(self, module, args, output):
        torch, runtime = self.torch, self.runtime
        assert not self.events, 'Target increment called more than once'
        if self.component == 'attention':
            assert isinstance(output, tuple) and len(output) == 2
            gen = output[1]
        else:
            assert isinstance(output, torch.Tensor)
            gen = output
        gen_len = self.reference['cache']['steps'][0]['layout']['gen_len']
        assert gen.shape == (gen_len, 4096) and gen.dtype == self.expected_before.dtype
        rows = self.rows.to(gen.device)
        before = runtime.cpu(gen.index_select(0, rows))
        runtime.require_exact(before, self.expected_before, 'target_actual_before_equals_recipient_cache')
        changed = self.replacement.to(device=gen.device)
        assert changed.dtype == gen.dtype
        altered = gen.clone()
        altered.index_copy_(0, rows, changed)
        after = runtime.cpu(altered.index_select(0, rows))
        runtime.require_exact(after, self.replacement, 'target_actual_after_equals_requested_source')
        mask = torch.ones(gen_len, device=gen.device, dtype=torch.bool)
        mask[rows] = False
        runtime.require_exact(gen[mask], altered[mask], 'target_nonaction_increment_rows_bit_exact')
        delta = after.double() - before.double()
        event = dict(layer_zero_based=self.layer, component=self.component, local_forward=0,
            original_denoising_forward=REPLAY_STEP, action_rows=self.rows.clone(),
            before=before, after=after, actual_delta=delta, actual_delta_l2=float(delta.norm()),
            nonaction_rows_count=int(mask.sum()), nonaction_rows_bit_exact=True,
            before_equals_recipient_cache=True, after_equals_requested_source=True,
            und_increment_unchanged=self.component == 'attention',
            exact_no_change=bool(torch.equal(before, after)),
            patch_site='Component output increment before decoder residual addition')
        self.events.append(event)
        if event['exact_no_change']:
            return output
        if self.component == 'attention':
            result = (output[0], altered)
            runtime.require_exact(result[0], output[0], 'target_UND_attention_increment_bit_exact')
            return result
        return altered


def exact_reference_result(actual, reference, runtime, label):
    for key in ('hidden', 'readout', 'velocity', 'model_output'):
        runtime.require_exact(actual[key], reference[key], label + '/entire_native_' + key + '_exact')
    for key in ('steps', 'site_specs', 'first_model_kwargs', 'frames'):
        runtime.require_exact(actual['cache'][key], reference['cache'][key], label + '/native_cache_' + key)


def run_forward(runtime, hooks, scan, reference, label, destination, patch=None, exact=False):
    """Only one actual model call; patch runs before read-only helper capture."""
    import numpy as np
    torch, model = runtime.torch, runtime.pipe.transformer
    assert not model.training and not torch.is_grad_enabled() and not model.is_cache_enabled
    assert hooks._kind is None
    kwargs = to_device(reference['kwargs'], torch, next(model.parameters()).device)
    rows = reference['cache']['steps'][0]['layout']['action_rows'].to(next(model.parameters()).device)
    boundaries, readouts, velocities, calls, handles = [], [], [], [], []
    cache = None

    def before_model(module, args, actual_kwargs):
        assert not args and not calls, 'Exactly one un-nested model call required'
        runtime.require_exact(runtime.cpu(actual_kwargs), reference['kwargs'], label + '/actual_inputs_unchanged')
        calls.append(1)

    def before_first(module, args):
        boundaries.append(runtime.cpu(args[1].index_select(0, rows)))

    def after_layer(module, args, output):
        boundaries.append(runtime.cpu(output[1].index_select(0, rows)))

    def before_head(module, args):
        readouts.append(runtime.cpu(args[0]))

    def after_head(module, args, output):
        velocities.append(runtime.cpu(output))

    destination.mkdir(exist_ok=False)
    try:
        handles.append(model.register_forward_pre_hook(before_model, with_kwargs=True))
        if patch is not None:
            block = model.layers[patch.layer]
            target = block.self_attn if patch.component == 'attention' else block.mlp_moe_gen
            handles.append(target.register_forward_hook(patch))
        hooks.begin_capture(label, expected_steps=1)
        handles.append(model.layers[0].register_forward_pre_hook(before_first))
        handles.extend(block.register_forward_hook(after_layer) for block in model.layers)
        handles.append(model.action_proj_out.register_forward_pre_hook(before_head))
        handles.append(model.action_proj_out.register_forward_hook(after_head))
        output = runtime.cpu(model(**kwargs))
    finally:
        for handle in reversed(handles):
            handle.remove()
        if hooks._kind is not None:
            cache = hooks.end()
    assert len(calls) == 1 and len(boundaries) == 37 and len(readouts) == len(velocities) == 1
    validate_single_cache(cache, hooks, runtime, label)
    runtime.require_exact(cache['first_model_kwargs'], reference['kwargs'], label + '/captured_actual_inputs')
    runtime.require_exact(cache['steps'], reference['cache']['steps'], label + '/full_pack_and_sigma_unchanged')
    hidden, readout, velocity = torch.stack(boundaries), readouts[0], velocities[0]
    assert hidden.shape == (37, 16, 4096) and readout.shape == (16, 4096) and velocity.shape == (16, 64)
    assert hidden.dtype == readout.dtype == velocity.dtype == torch.bfloat16
    require_finite((hidden, readout, velocity, output), torch)
    actual = dict(hidden=hidden, readout=readout, velocity=velocity, model_output=output, cache=cache)
    event = None
    if patch is not None:
        assert len(patch.events) == 1
        event = patch.events[0]
        site = (0, patch.layer, patch.component)
        runtime.require_exact(cache['frames'][site], event['after'], label + '/observer_sees_actual_patched_increment')
        target_order = patch.layer * 2 + COMPONENTS.index(patch.component)
        for layer in range(36):
            for index, component in enumerate(COMPONENTS):
                if layer * 2 + index < target_order:
                    key = (0, layer, component)
                    runtime.require_exact(cache['frames'][key], reference['cache']['frames'][key], label + '/upstream_exact/' + str(key))
        runtime.require_exact(hidden[:patch.layer + 1], reference['hidden'][:patch.layer + 1], label + '/upstream_boundaries_exact')
        torch.save(event, destination / 'component_event.pt')
    if exact:
        exact_reference_result(actual, reference, runtime, label)
    torch.save(cache, destination / 'components.pt')
    torch.save(output, destination / 'model_output.pt')
    # Preserve the actual bf16 tensors as well as portable lossless FP32 NPZ.
    torch.save(dict(action_boundary_hidden=hidden, readout=readout, action_velocity=velocity), destination / 'actual_head_boundaries.pt')
    np.savez_compressed(destination / 'arrays.npz', action_boundary_hidden=hidden.float().numpy(),
                        readout=readout.float().numpy(), action_velocity=velocity.float().numpy())
    metadata = dict(label=label, seed=SEED, original_denoising_forward=REPLAY_STEP,
        recipient_context=reference['metadata']['label'], actual_model_calls=1, actual_boundaries=37,
        component_cache_complete=True, component_calls=72, target_intervention_count=0 if patch is None else 1,
        full_native_return_head_boundaries_components_bit_exact=exact,
        full_actual_model_kwargs_and_sigma_unchanged=True, action_rows=rows.cpu().tolist(),
        all_owned_hooks_removed=True, readout=scan.tensor_meta(readout, torch),
        velocity=scan.tensor_meta(velocity, torch), boundary=scan.tensor_meta(hidden, torch),
        layer_convention='Boundary 0 before L0; boundaries 1..36 after zero-based layers 0..35',
        files_sha256={name: sha(destination / name) for name in
            ('components.pt', 'model_output.pt', 'actual_head_boundaries.pt', 'arrays.npz')}, scope=SCOPE)
    if event is not None:
        metadata['component_event_sha256'] = sha(destination / 'component_event.pt')
        metadata['component_event'] = {key: value for key, value in event.items() if not isinstance(value, torch.Tensor)}
        metadata['component_event'].update(before=scan.tensor_meta(event['before'], torch),
            after=scan.tensor_meta(event['after'], torch), actual_delta=scan.tensor_meta(event['actual_delta'], torch))
    write_json(destination / 'metadata.json', metadata)
    actual.update(event=event, metadata=metadata)
    return actual


def make_random_replacement(current, natural_source, seed, torch):
    """Equal actual increment-delta norm, not nominal pre-rounding noise norm."""
    target_delta = natural_source.double() - current.double()
    target = float(target_delta.norm())
    assert math.isfinite(target) and target > 0, 'No natural increment difference to norm-match'
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
        raise RuntimeError('Cannot bracket the actual rounded random delta norm')
    best, iterations = None, 0
    for iteration in range(64):
        scale = (lo + hi) / 2
        value, delta, norm = candidate(scale)
        error = abs(norm - target) / target
        if best is None or error < best[0]:
            best = error, value, delta, norm, scale
        iterations = iteration + 1
        if error <= RANDOM_NORM_REL_TOL:
            break
        if norm < target:
            lo = scale
        else:
            hi = scale
    error, value, delta, norm, scale = best
    assert error <= RANDOM_NORM_REL_TOL, f'Actual bf16 random norm mismatch: {error}'
    return value, dict(cpu_generator_seed=seed, actual_natural_delta_l2=target,
        actual_random_delta_l2=norm, relative_norm_error=error,
        relative_norm_tolerance=RANDOM_NORM_REL_TOL, scale_before_quantization=scale,
        calibration_iterations=iterations, raw_gaussian=raw, unit_direction=unit,
        actual_quantized_delta=delta, actual_cosine_with_natural_delta=float((delta * target_delta).sum() / (norm * target)))


def result_row(scan, baseline_r, baseline_d, pr, rr, actual, label, destination,
               layer=None, component=None, reverse=False):
    source, endpoint = (rr, pr) if reverse else (pr, rr)
    restoration = scan.direction_metrics(source['velocity'][:, :3], endpoint['velocity'][:, :3], actual['velocity'][:, :3])
    assert restoration['gap_meaningful'] and restoration['signed_projection_fraction'] is not None
    r, d = baseline_r['action_velocity'][REPLAY_STEP, :, :3], baseline_d['action_velocity'][REPLAY_STEP, :, :3]
    delta = actual['velocity'].double() - source['velocity'].double()
    return dict(run=label, layer_zero_based=layer, component=component,
        input_context='RR' if reverse else 'PR', intervention='PR_increment_into_RR' if reverse else 'RR_increment_into_PR',
        restoration_xyz=restoration,
        restoration_formula='dot(patchedPR-PR,RR-PR)/||RR-PR||^2 XYZ' if not reverse else
                            'dot(patchedRR-RR,PR-RR)/||PR-RR||^2 XYZ',
        original_t21_D_minus_R_effect_xyz=scan.direction_metrics(r, d, r.double() + delta[:, :3]),
        original_t21_D_minus_R_total_xyz=scan.direction_metrics(r, d, actual['velocity'][:, :3]),
        raw_velocity_all64_delta_l2=float(delta.norm()),
        actual_increment_delta_l2=None if actual['event'] is None else actual['event']['actual_delta_l2'],
        artifacts=str(destination), metadata_sha256=sha(destination / 'metadata.json'), scope=SCOPE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Completed nine-trial baseline root')
    baseline = parser.parse_args().output.resolve()
    paths = dict(normal_runtime=ROOT / 'work/serve_cosmos_goal_pair.py',
                 factory=ROOT / 'work/probe_cosmos_language_routes.py',
                 components=ROOT / 'work/cosmos_component_interventions.py')
    scan_path = ROOT / 'work/probe_cosmos_milk_component_pulses.py'
    feedback_script = ROOT / 'work/probe_cosmos_milk_feedback_inputs.py'
    scan = load_file('correction_existing_pulse_contract', scan_path)
    scan.validate_baseline(baseline)
    feedback = baseline / 'feedback-inputs'
    complete, provenance = read_json(feedback / 'complete.json'), read_json(feedback / 'provenance.json')
    assert not (feedback / 'failed.json').exists()
    assert complete['state'] == 'complete' and complete['seeds'] == [195, 196, 198]
    assert complete['native_replays_exact'] == 6 and complete['direct_model_forwards'] == 12
    assert complete['boundaries_per_forward'] == 37 and complete['components_per_forward'] == 72
    assert complete['summary_sha256'] == sha(feedback / 'summary.json')
    assert complete['script_sha256'] == provenance['script_sha256'] == sha(feedback_script)
    assert provenance['pulse_contract_sha256'] == sha(scan_path)
    assert provenance['baseline_summary_sha256'] == sha(baseline / 'summary.json')
    assert provenance['fixed_replay_step'] == REPLAY_STEP
    assert provenance['recipient'] == 'x15' and provenance['donor'] == 'x06'
    summary = read_json(feedback / 'summary.json')
    assert summary['all_native_R_and_P_t21_replays_bit_exact'] is True
    assert summary['entire_t21_model_output_including_vision_exact'] is True
    assert summary['current_condition_all_seeds_exact'] is True
    baseline_provenance = read_json(baseline / 'server/provenance.json')
    for key, path in paths.items():
        assert sha(path) == provenance['source_sha256'][key] == baseline_provenance[key + '_sha256']
    out = baseline / 'action-correction-sites'
    out.mkdir(exist_ok=False)
    for child in ('controls', 'runs', 'followups'):
        (out / child).mkdir(exist_ok=False)
    started, completed, hooks = time.perf_counter(), [], None

    def progress(stage, **fields):
        value = dict(stage=stage, completed_forwards=len(completed), elapsed_s=time.perf_counter() - started, **fields)
        write_json(out / 'progress.json', value)
        print('[ACTION-CORRECTION-SITES] ' + json.dumps(value, allow_nan=False), flush=True)

    try:
        os.environ['HF_HUB_OFFLINE'] = '1'
        base = load_file('correction_normal_runtime', paths['normal_runtime'])
        factory = load_file('correction_checked_factory', paths['factory'])
        factory.OUT = out
        assert sha(factory.SCHEDULER) == provenance['scheduler_sha256'] == baseline_provenance['scheduler_sha256']
        progress('loading_model')
        runtime = base.NormalRuntime(factory)
        torch, model = runtime.torch, runtime.pipe.transformer
        assert not model.training and not torch.is_grad_enabled() and not model.is_cache_enabled
        module = load_file('correction_component_observers', paths['components'])
        hooks = module.CosmosComponentInterventions(model)
        transformer_source = Path(inspect.getfile(type(model)))
        pipeline_source = Path(inspect.getfile(type(runtime.pipe)))
        assert sha(transformer_source) == provenance['actual_transformer_sha256']
        assert sha(pipeline_source) == provenance['actual_pipeline_source_sha256']
        folder = feedback / f'seed{SEED}'
        refs = {label: load_reference(folder / 'replays' / label, hooks, runtime, scan, label) for label in ('RR', 'PR')}
        rr, pr = refs['RR'], refs['PR']
        runtime.require_exact(rr['kwargs'].keys(), pr['kwargs'].keys(), 'paired_RR_PR_kwargs_keys')
        for key in rr['kwargs']:
            if key != 'action_tokens':
                runtime.require_exact(rr['kwargs'][key], pr['kwargs'][key], 'only_action_input_differs/' + key)
        assert len(rr['kwargs']['action_tokens']) == len(pr['kwargs']['action_tokens']) == 1
        ar, ap = rr['kwargs']['action_tokens'][0], pr['kwargs']['action_tokens'][0]
        assert ar.shape == ap.shape and ar.dtype == ap.dtype and not torch.equal(ar, ap)
        require_finite((ar, ap), torch)
        runtime.require_exact(rr['cache']['steps'], pr['cache']['steps'], 'paired_full_pack_schedule_metadata')
        records, caches, sources = {}, {}, {}
        contract = read_json(folder / 'input-contract.json')
        assert contract['seed'] == SEED and contract['both_pure_noise_draws_exact'] is True
        assert contract['complete_schedule_and_30step_metadata_exact'] is True
        for role, scene in (('R', 'x15'), ('D', 'x06')):
            run = baseline / 'closed-loop' / f'{scene}_seed{SEED}' / 'chunk_00'
            records[role] = torch.load(run / 'states.pt', map_location='cpu', weights_only=True, mmap=True)
            caches[role] = torch.load(run / 'components.pt', map_location='cpu', weights_only=True, mmap=True)
            scan.validate_record(records[role], torch)
            scan.validate_cache(caches[role], records[role], hooks, runtime, role)
            for name in ('states', 'components'):
                assert contract['sources'][role][name + '_sha256'] == sha(run / (name + '.pt'))
            sources[role] = dict(states_sha256=sha(run / 'states.pt'), components_sha256=sha(run / 'components.pt'))
        r, d = records['R'], records['D']
        for key in ('pure_noise', 'timesteps', 'sigmas'):
            runtime.require_exact(r[key], d[key], 'both_noise_draws_and_full_schedule/' + key)
        runtime.require_exact(caches['R']['steps'], caches['D']['steps'], 'paired_30step_pack_text_metadata')
        native_r = torch.load(folder / 'native_R_q0/states.pt', map_location='cpu', weights_only=True, mmap=True)
        pulse_p = torch.load(folder / 'pulse_P_q0/states.pt', map_location='cpu', weights_only=True, mmap=True)
        runtime.require_exact(native_r, r, 'saved_native_R_all_record_fields_exact')
        scan.validate_run_contract(pulse_p, r, runtime, 'saved_P_full_noise_schedule_contract')
        r_inputs = torch.load(folder / 'native_R_q0/model_inputs_20_21.pt', map_location='cpu', weights_only=True)
        p_inputs = torch.load(folder / 'pulse_P_q0/model_inputs_20_21.pt', map_location='cpu', weights_only=True)
        r_outputs = torch.load(folder / 'native_R_q0/model_outputs_20_21.pt', map_location='cpu', weights_only=True)
        runtime.require_exact(rr['kwargs'], r_inputs[REPLAY_STEP], 'actual_saved_RR_t21_kwargs')
        actual_pr = dict(r_inputs[REPLAY_STEP])
        actual_pr['action_tokens'] = p_inputs[REPLAY_STEP]['action_tokens']
        runtime.require_exact(pr['kwargs'], actual_pr, 'actual_saved_PR_t21_kwargs')
        runtime.require_exact(rr['model_output'], r_outputs[REPLAY_STEP], 'RR_full_return_equals_actual_q0_t21_return')
        runtime.require_exact(rr['readout'], r['readout'][REPLAY_STEP], 'RR_actual_q0_head_readout')
        runtime.require_exact(rr['velocity'], r['action_velocity'][REPLAY_STEP], 'RR_actual_q0_head_velocity')
        for layer in range(36):
            for component in COMPONENTS:
                runtime.require_exact(rr['cache']['frames'][(0, layer, component)], caches['R']['frames'][(REPLAY_STEP, layer, component)], 'RR_original_component')
        axis = scan.direction_metrics(r['action_velocity'][REPLAY_STEP, :, :3], d['action_velocity'][REPLAY_STEP, :, :3], pr['velocity'][:, :3])
        restoration_gap = scan.direction_metrics(pr['velocity'][:, :3], rr['velocity'][:, :3], pr['velocity'][:, :3])
        assert axis['gap_meaningful'] and restoration_gap['gap_meaningful'], 'Invalid raw XYZ gap: refuse projection scores'
        input_contract = dict(seed=SEED, original_denoising_forward=REPLAY_STEP,
            sigma=float(r['sigmas'][REPLAY_STEP]), timestep=int(r['timesteps'][REPLAY_STEP]),
            actual_RR_vs_PR_only_action_tokens_differ=True, vision_text_structure_full_sigma_exact=True,
            both_actual_pure_noise_draws_exact=True, all_30_schedule_pack_metadata_exact=True,
            native_R_full_saved_record_exact=True, RR_full_return_matches_native_t21=True,
            action_input_delta_l2=float((ap.double() - ar.double()).norm()),
            action_rows=rr['cache']['steps'][0]['layout']['action_rows'].tolist(),
            PR_original_axis_xyz=axis, PR_to_RR_gap_xyz=restoration_gap, sources=sources, scope=SCOPE)
        write_json(out / 'input-contract.json', input_contract)
        write_json(out / 'provenance.json', dict(script_sha256=sha(Path(__file__)),
            source_sha256={key: sha(path) for key, path in paths.items()}, pulse_contract_sha256=sha(scan_path),
            feedback_script_sha256=sha(feedback_script), feedback_complete_sha256=sha(feedback / 'complete.json'),
            feedback_provenance_sha256=sha(feedback / 'provenance.json'), feedback_summary_sha256=sha(feedback / 'summary.json'),
            actual_transformer_source=str(transformer_source), actual_transformer_sha256=sha(transformer_source),
            actual_pipeline_source_sha256=sha(pipeline_source), scheduler_sha256=sha(factory.SCHEDULER),
            reference_metadata_sha256={label: sha(ref['folder'] / 'metadata.json') for label, ref in refs.items()},
            baseline_records=sources, seed=SEED, original_denoising_forward=REPLAY_STEP,
            random_seeds=list(RANDOM_SEEDS), random_actual_norm_relative_tolerance=RANDOM_NORM_REL_TOL,
            expected_single_model_forwards_if_positive_site=80, library_edits=False, q0_predictions=0, scope=SCOPE))
        controls, cases, followups = [], [], []
        for label in ('RR', 'PR'):
            run_label = 'native_' + label
            progress('native_control', run=run_label)
            run_forward(runtime, hooks, scan, refs[label], run_label, out / 'controls' / run_label, exact=True)
            completed.append(run_label)
            controls.append(dict(run=run_label, full_return_head_boundaries_all_components_bit_exact=True))
        for component in COMPONENTS:
            label = 'PR_self_' + component
            progress('self_control', run=label)
            patch = SingleIncrementPatch(runtime, pr, SELF_LAYER, component, pr['cache']['frames'][(0, SELF_LAYER, component)])
            actual = run_forward(runtime, hooks, scan, pr, label, out / 'controls' / label, patch=patch, exact=True)
            assert actual['event']['exact_no_change'] is True and actual['event']['actual_delta_l2'] == 0
            completed.append(label)
            controls.append(dict(run=label, layer_zero_based=SELF_LAYER, component=component,
                                 full_return_head_boundaries_all_components_bit_exact=True))
        for layer in range(36):
            for component in COMPONENTS:
                label = f'layer{layer:02d}_{component}'
                progress('site', run=label)
                patch = SingleIncrementPatch(runtime, pr, layer, component, rr['cache']['frames'][(0, layer, component)])
                destination = out / 'runs' / label
                actual = run_forward(runtime, hooks, scan, pr, label, destination, patch=patch)
                row = result_row(scan, r, d, pr, rr, actual, label, destination, layer, component)
                row['selection_role'] = 'Single-context intervention ranking; no held-out confirmation'
                cases.append(row)
                completed.append(label)
                write_json(out / 'scan-partial.json', dict(completed_sites=len(cases), cases=cases, scope=SCOPE))
                progress('site_complete', run=label, restoration_xyz=row['restoration_xyz']['signed_projection_fraction'])
        assert len(cases) == 72 and len(completed) == 76
        ranking = sorted(cases, key=lambda row: row['restoration_xyz']['signed_projection_fraction'], reverse=True)
        positive = [row for row in ranking if row['restoration_xyz']['signed_projection_fraction'] > 0 and row['actual_increment_delta_l2'] > 0]
        selected = positive[0] if positive else None
        if selected is not None:
            layer, component = selected['layer_zero_based'], selected['component']
            site = (0, layer, component)
            label = f'reverse_RR_layer{layer:02d}_{component}'
            progress('reverse_control', run=label)
            patch = SingleIncrementPatch(runtime, rr, layer, component, pr['cache']['frames'][site])
            destination = out / 'followups' / label
            actual = run_forward(runtime, hooks, scan, rr, label, destination, patch=patch)
            followups.append(result_row(scan, r, d, pr, rr, actual, label, destination, layer, component, reverse=True))
            completed.append(label)
            for random_seed in RANDOM_SEEDS:
                replacement, info = make_random_replacement(pr['cache']['frames'][site], rr['cache']['frames'][site], random_seed, torch)
                label = f'random_layer{layer:02d}_{component}_seed{random_seed}'
                progress('random_control', run=label)
                patch = SingleIncrementPatch(runtime, pr, layer, component, replacement)
                destination = out / 'followups' / label
                actual = run_forward(runtime, hooks, scan, pr, label, destination, patch=patch)
                runtime.require_exact(actual['event']['actual_delta'], info['actual_quantized_delta'], label + '/actual_random_delta_exact')
                error = abs(actual['event']['actual_delta_l2'] - info['actual_natural_delta_l2']) / info['actual_natural_delta_l2']
                assert error <= RANDOM_NORM_REL_TOL
                torch.save(dict(replacement=replacement, **info), destination / 'random_control.pt')
                random_metadata = {key: value for key, value in info.items() if not isinstance(value, torch.Tensor)}
                write_json(destination / 'random_control.json', random_metadata)
                row = result_row(scan, r, d, pr, rr, actual, label, destination, layer, component)
                row.update(intervention='actual_bf16_norm_matched_random', random_control=random_metadata,
                           random_control_sha256=sha(destination / 'random_control.pt'))
                followups.append(row)
                completed.append(label)
            assert len(followups) == 4 and len(completed) == 80
        else:
            progress('no_positive_site', reason='No single RR increment gives positive raw XYZ restoration; no site selected for follow-ups')
        write_json(out / 'results.json', dict(state='complete', seed=SEED, original_denoising_forward=REPLAY_STEP,
            controls=controls, cases=cases, restoration_ranking=ranking, selected_site=selected, followups=followups,
            scan_forwards=72, control_forwards=4, followup_forwards=len(followups), completed=completed,
            input_contract=input_contract, scope=SCOPE,
            predictions={'localized_correction': 'Replacing a local PR increment by RR reduces the immediate action-only reverse response.',
                         'distributed_correction': 'No individual increment restores it substantially; this scan cannot identify a unique site.'},
            limitations=['Selected by XYZ restoration in the same seed and fixed t21 context, with no held-out confirmation.',
                         'Natural RR/PR activity differs because of the action input; the swap does not isolate an entire path.',
                         'Only the target increment nonaction rows are preserved; downstream vision/action computation responds freely.',
                         'A restoration fraction is a signed raw velocity projection, not target identity or behavioral success.']))
        write_json(out / 'complete.json', dict(state='complete', seed=SEED, original_denoising_forward=REPLAY_STEP,
            single_model_forwards=len(completed), q0_predictions=0, scan_forwards=72, exact_control_forwards=4,
            reverse_control_forwards=0 if selected is None else 1, random_control_forwards=0 if selected is None else 3,
            full_return_head_boundaries_component_controls_bit_exact=True,
            all_actual_inputs_full_sigma_unchanged=True, all_target_before_after_and_nonaction_rows_exact=True,
            selected_positive_site=selected is not None, results_sha256=sha(out / 'results.json'),
            script_sha256=sha(Path(__file__)), scope=SCOPE))
        progress('complete')
    except Exception as exc:
        write_json(out / 'failed.json', dict(state='failed', error=str(exc), traceback=traceback.format_exc(),
            completed=completed, elapsed_s=time.perf_counter() - started, scope=SCOPE))
        raise
    finally:
        if hooks is not None:
            hooks.reset()


if __name__ == '__main__':
    main()
