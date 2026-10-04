"""Twenty-four fixed-t21 forwards testing real L3 gated MLP units.

R=RR replay; P=PR replay (P action sample, R vision), never the PP replay.
Capture native gate-pre, actual SiLU gate, up, product z, input x and down
output. Select nested32/128/512 units by a declared local linear proxy, then
freeze indices before intervention output scores. Own hooks change only actual
action-row units. No scheduler, simulator, training or model-library edits.
Random controls match actual bf16 down-output delta norms, not z norms; their
extra full-GEN down.forward evaluations are explicitly counted. Generic inputs
change only RR XYZ, perpendicular to the natural XYZ input difference.
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
SEED, STEP, LAYER, WIDTH = 198, 21, 3, 12288
SIZES = (32, 128, 512)
SUPPORT_SEEDS = (3219801, 3219802, 3219803)
COEFFICIENT_SEEDS = (4219801, 4219802, 4219803)
INPUT_SEEDS = (5219801, 5219802, 5219803)
NORM_TOL, ORTHOGONAL_TOL, MAX_GAIN = 1e-3, 1e-3, 128.
MAX_CALIBRATION_CALLS = 80
SCOPE = ('One seed198, fixed actual t21 model inputs. Real L3 SiLU-gated MLP units '
         'and raw Flow XYZ velocity only. No solver step, physics, target identity, '
         'semantic gate, inhibitory-neuron, probability or root-cause claim. '
         'Proxy selection is not an additive partition of final model effects.')


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


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def protocol():
    return dict(seed=SEED, original_denoising_forward=STEP, layer_zero_based=LAYER,
        num_real_units=WIDTH, unit_convention='Columns at actual down_proj input; zero based, not heads.',
        contexts={'R': 'RR=(R action,R vision)', 'P': 'PR=(P action,R vision), not PP'},
        expected_model_forwards=24, calibration_down_forward_calls_counted_separately=True,
        forward_groups={'native_controls': 2, 'all_unit_self': 1, 'all_unit_swap_equivalence': 1,
            'nested_natural': 3, 'reverse128': 1, 'gate_up_factorial128': 4,
            'random_support128': 3, 'same_support_random_coefficients128': 3, 'generic_input_native_and_patch': 6},
        strong_predictions={'sparse128': 'r128 >= 0.5 * existing actual whole-L3-MLP restoration',
            'gate_vs_up': 'Natural, not norm-equalized, gate-only signed restoration > up-only signed restoration'},
        strong_prediction_note='Report each branch actual down delta L2 and branch input relative amplitude; a larger g-only effect alone does not establish selection semantics.',
        proxy='s_j=sum_i[(zR-zP)_ij * dot(W_down[:,j],(M_R-M_P)_i)] in float64; descending signed score, ties by index.',
        nested_sizes=list(SIZES), selection_frozen_before_intervention_output_scores=True,
        restore_formula='dot(Vtrial-V_PR,V_RR-V_PR)/||V_RR-V_PR||^2 on sixteen raw XYZ velocity rows',
        generic_restore_formula='For each Gi, dot(VGi_patch-VGi_native,V_RR-VGi_native)/||V_RR-VGi_native||^2; same gap guard, separate denominator.',
        gate_up_cells={'PP': 'gP*uP self', 'RP': 'gR*uP gate-only', 'PR': 'gP*uR up-only', 'RR': 'gR*uR joint'},
        random_support_seeds=list(SUPPORT_SEEDS), same_support_coefficient_seeds=list(COEFFICIENT_SEEDS),
        generic_xyz_input_seeds=list(INPUT_SEEDS), random_norm_target='Actual down-output action-row delta L2 of the natural128 model forward.',
        random_support_sampling='Uniform randperm across all12288; reject sets contained in top512, never subsample top units.',
        same_support_noise='Independent CPU float64 Gaussian coefficient direction at the same128 units; calibrated positive gain, not a natural donor.',
        down_calibration='Actual down_proj.forward on full GEN shape, bypassing module hooks. Native full GEN reconstruction must first be bit-exact.',
        maximum_positive_gain=MAX_GAIN, maximum_direct_down_calls_per_random_arm=MAX_CALIBRATION_CALLS,
        actual_norm_relative_tolerance=NORM_TOL, generic_actual_xyz_cosine_tolerance=ORTHOGONAL_TOL,
        generic_xyz_target='Actual norm of (PRaction-RRaction)[:,:3]; other61 action coordinates and all other kwargs remain RR.',
        invalid_gap_policy='Use existing ABS1e-6/REL1e-4 guard. Null scores are not forced positive and do not establish support.',
        entire_transformer_parameter_bytes_hashed_before_and_after=True, scope=SCOPE)


def parameter_hashes(model, torch):
    """Bound CPU memory while hashing every actual named transformer parameter."""
    result, aggregate = {}, hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        value = parameter.detach().contiguous().reshape(-1).view(torch.uint8)
        digest = hashlib.sha256()
        for begin in range(0, value.numel(), 8 * 1024 * 1024):
            digest.update(value[begin:begin + 8 * 1024 * 1024].cpu().numpy().tobytes())
        meta = dict(shape=list(parameter.shape), dtype=str(parameter.dtype), bytes=value.numel(), sha256=digest.hexdigest())
        result[name] = meta
        aggregate.update(json.dumps([name, meta], sort_keys=True).encode())
    return dict(parameters=result, aggregate_sha256=aggregate.hexdigest(), parameter_count=len(result))


def json_tensors(value, scan, torch):
    if isinstance(value, torch.Tensor):
        return scan.tensor_meta(value, torch)
    if isinstance(value, dict):
        return {str(key): json_tensors(item, scan, torch) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_tensors(item, scan, torch) for item in value]
    return value


class UnitSession:
    """Own pre-down/branch hooks; never patch the old whole-MLP output interface."""

    def __init__(self, runtime, reference, native_site=None, source_site=None,
                 indices=(), mode='native', target=None, random_seed=None):
        self.runtime, self.torch, self.reference = runtime, runtime.torch, reference
        self.block = runtime.pipe.transformer.layers[LAYER].mlp_moe_gen
        self.native, self.source, self.mode = native_site, source_site, mode
        self.indices = self.torch.tensor(list(indices), dtype=self.torch.long)
        assert self.indices.unique().numel() == self.indices.numel()
        assert bool(((self.indices >= 0) & (self.indices < WIDTH)).all())
        self.rows = reference['cache']['steps'][0]['layout']['action_rows']
        self.gen_len = reference['cache']['steps'][0]['layout']['gen_len']
        self.target, self.random_seed = target, random_seed
        self.handles, self.values, self.calls = [], {}, {}
        self.calibration = dict(direct_down_forward_calls=0, tested_gains=[], full_GEN_shape=[self.gen_len, WIDTH],
            bypasses_module_hooks=True, native_reconstruction_full_GEN_bit_exact=False)
        self.branch_events, self.unit_events = [], []

    def once(self, key):
        self.calls[key] = self.calls.get(key, 0) + 1
        assert self.calls[key] == 1, 'Duplicate target hook/recomputation: ' + key

    def capture(self, key, value, width):
        assert value.shape == (self.gen_len, width) and value.dtype == self.torch.bfloat16
        assert self.torch.isfinite(value).all()
        self.values[key] = self.runtime.cpu(value)
        return self.values[key]

    def assert_footprint(self, before, after):
        torch = self.torch
        nonaction = torch.ones(self.gen_len, dtype=torch.bool)
        nonaction[self.rows] = False
        unused = torch.ones(WIDTH, dtype=torch.bool)
        unused[self.indices] = False
        self.runtime.require_exact(before[nonaction], after[nonaction], 'all_nonaction_unit_rows_bit_exact')
        self.runtime.require_exact(before[self.rows][:, unused], after[self.rows][:, unused], 'all_unselected_action_units_bit_exact')

    def replace(self, value, selected):
        changed = value.clone()
        rows, cols = self.rows.to(value.device), self.indices.to(value.device)
        changed[rows[:, None], cols[None, :]] = selected.to(device=value.device, dtype=value.dtype)
        before, after = self.runtime.cpu(value), self.runtime.cpu(changed)
        self.assert_footprint(before, after)
        self.runtime.require_exact(after[self.rows][:, self.indices], self.runtime.cpu(selected), 'actual_selected_unit_values_equal_requested_values')
        return changed

    def before_mlp(self, module, args):
        self.once('mlp_input')
        assert len(args) == 1
        value = self.capture('x', args[0], 4096)
        if self.native is not None:
            self.runtime.require_exact(value, self.native['x'], 'actual_MLP_input_equals_native_context')

    def gate_pre(self, module, args, output):
        self.once('gate_pre')
        value = self.capture('gate_pre', output, WIDTH)
        if self.native is not None:
            self.runtime.require_exact(value, self.native['gate_pre'], 'actual_gate_pre_equals_native_context')

    def branch(self, key):
        def after(module, args, output):
            self.once(key)
            before = self.capture(key + '_before', output, WIDTH)
            if self.native is not None:
                self.runtime.require_exact(before, self.native[key + '_before'], 'actual_native_branch/' + key)
            selected = self.mode == 'factor_' + key or self.mode in ('factor_joint', 'factor_self')
            effective = output
            if selected:
                source = self.native if self.mode == 'factor_self' else self.source
                replacement = source[key + '_before'][self.rows][:, self.indices]
                effective = self.replace(output, replacement)
                self.branch_events.append(dict(branch=key, before=before, after=self.runtime.cpu(effective)))
            self.capture(key + '_effective', effective, WIDTH)
            return None if self.torch.equal(before, self.values[key + '_effective']) else effective
        return after

    def direct_down(self, value, gain):
        self.calibration['direct_down_forward_calls'] += 1
        assert self.calibration['direct_down_forward_calls'] <= MAX_CALIBRATION_CALLS
        self.calibration['tested_gains'].append(gain)
        assert value.shape == (self.gen_len, WIDTH)
        # Direct native Linear.forward intentionally bypasses all module hooks.
        return self.block.down_proj.forward(value)

    def calibrate(self, z):
        torch, runtime = self.torch, self.runtime
        assert self.native is not None and self.target is not None and math.isfinite(self.target) and self.target > 0
        baseline = self.direct_down(z, 0.)
        runtime.require_exact(runtime.cpu(baseline), self.native['down'], 'full_GEN_native_down_reconstruction_bit_exact')
        self.calibration['native_reconstruction_full_GEN_bit_exact'] = True
        current = self.values['z_down_input'][self.rows][:, self.indices]
        if self.mode == 'random_support':
            direction = self.source['z_down_input'][self.rows][:, self.indices].double() - current.double()
        else:
            generator = torch.Generator(device='cpu').manual_seed(self.random_seed)
            raw = torch.randn(current.shape, generator=generator, dtype=torch.float64, device='cpu')
            direction = raw / raw.norm()
            self.calibration['raw_gaussian'] = raw
        assert float(direction.norm()) > 0
        self.calibration['direction'] = direction

        def candidate(gain):
            selected = (current.double() + gain * direction).to(current.dtype)
            changed = self.replace(z, selected)
            output = self.direct_down(changed, gain)
            norm = float((output.index_select(0, self.rows.to(z.device)).double() - baseline.index_select(0, self.rows.to(z.device)).double()).norm())
            return changed, norm

        lo, hi, best = 0., 1., None
        while True:
            changed, norm = candidate(hi)
            error = abs(norm - self.target) / self.target
            best = (error, changed, norm, hi) if best is None or error < best[0] else best
            if norm >= self.target or error <= NORM_TOL:
                break
            lo, hi = hi, hi * 2
            assert hi <= MAX_GAIN, 'Cannot bracket actual down-output norm within declared maximum gain'
        for _ in range(64):
            if best[0] <= NORM_TOL:
                break
            gain = (lo + hi) / 2
            changed, norm = candidate(gain)
            error = abs(norm - self.target) / self.target
            if error < best[0]:
                best = error, changed, norm, gain
            if norm < self.target:
                lo = gain
            else:
                hi = gain
        error, changed, norm, gain = best
        assert error <= NORM_TOL, 'Actual bf16 down-output calibration failed; tolerance is fixed'
        self.calibration.update(selected_gain=gain, maximum_tested_gain=max(self.calibration['tested_gains']),
            target_actual_down_delta_l2=self.target, calibrated_actual_down_delta_l2=norm,
            calibrated_relative_error=error, cpu_generator_seed=self.random_seed,
            calculation='Full-GEN native Linear.forward; norm uses actual rounded action-row outputs')
        return changed

    def before_down(self, module, args):
        self.once('down_input')
        assert len(args) == 1
        z = args[0]
        self.capture('z_down_input', z, WIDTH)
        product = self.values['g_effective'] * self.values['u_effective']
        self.runtime.require_exact(self.values['z_down_input'], product, 'actual_down_input_equals_actual_postSiLU_gate_times_up')
        native_product = self.values['g_before'] * self.values['u_before']
        self.values['z_native_branch_product'] = native_product
        if self.native is not None:
            self.runtime.require_exact(native_product, self.native['z_down_input'], 'native_before_branch_product_equals_actual_native_z')
        landed = z
        if self.mode in ('z_swap', 'z_self', 'random_support', 'random_coefficients'):
            self.runtime.require_exact(self.values['z_down_input'], self.native['z_down_input'], 'actual_z_before_equals_native_context_bit_exact')
            if self.mode.startswith('random_'):
                landed = self.calibrate(z)
            else:
                source = self.native if self.mode == 'z_self' else self.source
                landed = self.replace(z, source['z_down_input'][self.rows][:, self.indices])
            self.unit_events.append(dict(kind='direct_actual_z_entry_replacement'))
        self.capture('z_landed', landed, WIDTH)
        self.assert_footprint(native_product, self.values['z_landed'])
        return None if self.torch.equal(self.values['z_down_input'], self.values['z_landed']) else (landed,)

    def after_down(self, module, args, output):
        self.once('down_output')
        actual = self.capture('down', output, 4096)
        if self.native is not None:
            mask = self.torch.ones(self.gen_len, dtype=self.torch.bool)
            mask[self.rows] = False
            self.runtime.require_exact(actual[mask], self.native['down'][mask], 'actual_down_nonaction_output_rows_bit_exact')
            norm = float((actual[self.rows].double() - self.native['down'][self.rows].double()).norm())
            self.values['actual_down_delta_l2'] = norm
            if self.mode.startswith('random_'):
                error = abs(norm - self.target) / self.target
                assert error <= NORM_TOL, 'Real model down output failed the fixed actual norm gate'
                self.calibration.update(actual_model_down_delta_l2=norm, actual_model_relative_error=error)

    def after_mlp(self, module, args, output):
        self.once('mlp_output')
        self.runtime.require_exact(self.runtime.cpu(output), self.values['down'], 'actual_MLP_increment_equals_actual_down_output')

    def begin(self):
        try:
            self.handles.append(self.block.register_forward_pre_hook(self.before_mlp))
            self.handles.append(self.block.gate_proj.register_forward_hook(self.gate_pre))
            self.handles.append(self.block.act_fn.register_forward_hook(self.branch('g')))
            self.handles.append(self.block.up_proj.register_forward_hook(self.branch('u')))
            self.handles.append(self.block.down_proj.register_forward_pre_hook(self.before_down))
            self.handles.append(self.block.down_proj.register_forward_hook(self.after_down))
            self.handles.append(self.block.register_forward_hook(self.after_mlp))
        except BaseException:
            self.reset()
            raise

    def reset(self):
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()

    def end(self):
        self.reset()
        expected = {'mlp_input', 'gate_pre', 'g', 'u', 'down_input', 'down_output', 'mlp_output'}
        return dict(complete=set(self.calls) == expected and all(count == 1 for count in self.calls.values()),
            native_module_calls=self.calls, mode=self.mode, selected_indices=self.indices,
            action_rows=self.rows, site=self.values, branch_events=self.branch_events,
            direct_z_events=self.unit_events, calibration=self.calibration,
            tensor_conventions={'z_down_input': 'Actual down prehook input, already mixed for branch interventions',
                'z_native_branch_product': 'Product of captured unmodified branch values; native reference, not a claimed factorial down-entry before',
                'z_landed': 'Actual effective down input after any direct z-entry replacement'},
            residual='Original decoder residual addition is outside MLP and was not hooked or overwritten', scope=SCOPE)


def run(runtime, hooks, correction, scan, reference, label, folder, native=None,
        source=None, indices=(), mode='native', target=None, random_seed=None, exact=False):
    import numpy as np
    session = UnitSession(runtime, reference, native, source, indices, mode, target, random_seed)
    report, actual = None, None
    session.begin()
    try:
        actual = correction.run_forward(runtime, hooks, scan, reference, label, folder, patch=None, exact=exact)
    finally:
        report = session.end()
        if folder.exists():
            runtime.torch.save(report, folder / 'unit_site.pt')
            write(folder / 'unit_site.json', json_tensors(report, scan, runtime.torch))
    assert report['complete'] is True
    torch = runtime.torch
    site, rows = report['site'], report['action_rows']
    runtime.require_exact(site['down'][rows], actual['cache']['frames'][(0, LAYER, 'mlp')], 'observer_captures_actual_down_increment')
    metadata = read(folder / 'metadata.json')
    metadata['legacy_whole_output_patch_count'] = metadata.pop('target_intervention_count')
    metadata.update(unit_target_intervention_count=len(report['branch_events']) + len(report['direct_z_events']),
        unit_hook_native_calls=report['native_module_calls'],
        extra_direct_down_forward_calibration_calls=report['calibration']['direct_down_forward_calls'],
        unit_site_sha256=sha(folder / 'unit_site.pt'), mode=mode,
        intervention_count_scope='Own branch/pre-down unit hooks; legacy output-patch interface was unused')
    write(folder / 'metadata.json', metadata)
    arrays = {key: value[rows].float().numpy() for key, value in site.items() if isinstance(value, torch.Tensor) and value.ndim == 2}
    np.savez_compressed(folder / 'unit_arrays.npz', **arrays)
    actual.update(unit=report, folder=folder, metadata=metadata)
    return actual


def input_control(rr, pr, seed, torch, runtime):
    ar, ap = rr['kwargs']['action_tokens'][0], pr['kwargs']['action_tokens'][0]
    assert ar.shape == ap.shape == (16, 64) and ar.dtype == ap.dtype
    delta = ap.double() - ar.double()
    xyz, other = delta[:, :3], delta[:, 3:]
    target = float(xyz.norm())
    assert math.isfinite(target) and target > 0
    generator = torch.Generator(device='cpu').manual_seed(seed)
    raw = torch.randn(xyz.shape, generator=generator, dtype=torch.float64, device='cpu')
    perpendicular = raw - xyz * ((raw * xyz).sum() / xyz.square().sum())
    unit = perpendicular / perpendicular.norm()
    assert abs(float((unit * xyz).sum() / target)) <= 1e-10
    scale = target
    best = None
    for iteration in range(32):
        changed = ar.clone()
        changed[:, :3] = (ar[:, :3].double() + scale * unit).to(ar.dtype)
        actual = changed[:, :3].double() - ar[:, :3].double()
        norm = float(actual.norm())
        assert norm > 0
        error = abs(norm - target) / target
        cosine = float((actual * xyz).sum() / (norm * target))
        if best is None or error < best[0]:
            best = error, changed, actual, norm, cosine, scale, iteration + 1
        if error <= NORM_TOL and abs(cosine) <= ORTHOGONAL_TOL:
            best = error, changed, actual, norm, cosine, scale, iteration + 1
            break
        scale *= target / norm
    error, changed, actual, norm, cosine, scale, iterations = best
    assert error <= NORM_TOL and abs(cosine) <= ORTHOGONAL_TOL, 'Actual generic input norm/orthogonality gate failed'
    runtime.require_exact(changed[:, 3:], ar[:, 3:], 'generic_other61_input_dimensions_bit_exact_RR')
    kwargs = dict(rr['kwargs'])
    kwargs['action_tokens'] = [changed]
    for key in kwargs:
        if key != 'action_tokens':
            runtime.require_exact(kwargs[key], rr['kwargs'][key], 'generic_all_other_kwargs_exact_RR/' + key)
    reference = dict(rr)
    reference['kwargs'] = kwargs
    # Capture-only helper structural step metadata is unchanged by XYZ values.
    cache = dict(rr['cache'])
    cache['first_model_kwargs'] = kwargs
    reference['cache'] = cache
    audit = dict(cpu_generator_seed=seed, raw_direction=raw, projected_direction=unit,
        actual_xyz_delta=actual, actual_action_tokens=changed,
        natural_full_delta=delta, natural_xyz_delta_l2=target, natural_other61_delta_l2=float(other.norm()),
        natural_all64_delta_l2=float(delta.norm()), natural_xyz_fraction_of_all64_delta_l2=target / float(delta.norm()),
        actual_xyz_delta_l2=norm, actual_norm_relative_error=error, actual_cosine_with_natural_xyz=cosine,
        scale_before_quantization=scale, calibration_iterations=iterations,
        actual_dtype=str(ar.dtype), other61_and_all_other_kwargs_bit_exact_RR=True,
        note='XYZ-only controls are not equal to the complete PR action-input difference.')
    return reference, audit


def load_whole(folder, row, correction, runtime):
    torch = runtime.torch
    metadata = read(folder / 'metadata.json')
    assert sha(folder / 'metadata.json') == row['metadata_sha256']
    assert metadata['label'] == 'layer03_mlp' and metadata['target_intervention_count'] == 1
    assert metadata['component_event']['before_equals_recipient_cache'] is True
    assert metadata['component_event']['after_equals_requested_source'] is True
    assert metadata['component_event']['nonaction_rows_bit_exact'] is True
    for name, expected in metadata['files_sha256'].items():
        assert sha(folder / name) == expected
    assert sha(folder / 'component_event.pt') == metadata['component_event_sha256']
    tensors = torch.load(folder / 'actual_head_boundaries.pt', map_location='cpu', weights_only=True, mmap=True)
    result = dict(cache=torch.load(folder / 'components.pt', map_location='cpu', weights_only=True, mmap=True),
        model_output=torch.load(folder / 'model_output.pt', map_location='cpu', weights_only=True, mmap=True),
        hidden=tensors['action_boundary_hidden'], readout=tensors['readout'], velocity=tensors['action_velocity'])
    correction.require_finite(result, torch)
    return result


def select_units(rr_site, pr_site, down_weight, torch):
    rows = rr_site['rows']
    dz = rr_site['site']['z_down_input'][rows].double() - pr_site['site']['z_down_input'][rows].double()
    dm = rr_site['site']['down'][rows].double() - pr_site['site']['down'][rows].double()
    proxy = (dz * (dm @ down_weight.double())).sum(0)
    assert proxy.shape == (WIDTH,) and torch.isfinite(proxy).all()
    ranking = sorted(range(WIDTH), key=lambda index: (-float(proxy[index]), index))
    nested = {str(size): ranking[:size] for size in SIZES}
    support = {}
    for seed in SUPPORT_SEEDS:
        generator = torch.Generator(device='cpu').manual_seed(seed)
        draw = torch.randperm(WIDTH, generator=generator)[:128].tolist()
        assert len(set(draw)) == 128 and not set(draw).issubset(set(nested['512']))
        support[str(seed)] = dict(draw_order=draw, indices=sorted(draw))
    return dict(ranking=ranking, nested_indices=nested, random_support_sets=support,
        proxy_signed_scores=proxy.tolist(), number_positive_proxy_scores=int((proxy > 0).sum()),
        scope='Local Δz and true down-W proxy only; no intervention output scores used.'), proxy


def metric_row(actual, reference, endpoint, scan, label, context):
    report = actual['unit']
    source, target, changed = reference['velocity'][:, :3], endpoint['velocity'][:, :3], actual['velocity'][:, :3]
    restore = scan.direction_metrics(source, target, changed)
    site, rows, indices = report['site'], report['action_rows'], report['selected_indices']
    before = site['z_native_branch_product'][rows]
    landed = site['z_landed'][rows]
    delta = landed.double() - before.double()
    branch_amplitudes = {}
    for name in ('g', 'u'):
        original, altered = site[name + '_before'][rows][:, indices], site[name + '_effective'][rows][:, indices]
        base_norm = float(original.double().norm())
        norm = float((altered.double() - original.double()).norm())
        branch_amplitudes[name] = dict(actual_selected_delta_l2=norm, original_selected_l2=base_norm,
                                       relative_l2=None if base_norm == 0 else norm / base_norm)
    return dict(run=label, input_context=context, mode=report['mode'], selected_units=indices.tolist(),
        restoration_xyz=restore, raw_xyz_velocity=changed.tolist(), reference_raw_xyz_velocity=source.tolist(),
        endpoint_raw_xyz_velocity=target.tolist(), raw_velocity_all64_delta_l2=float((actual['velocity'].double() - reference['velocity'].double()).norm()),
        actual_down_delta_l2=site.get('actual_down_delta_l2'), actual_z_delta_l2=float(delta.norm()),
        actual_z_delta_max_abs=float(delta.abs().max()), actual_z_delta_min=float(delta.min()), actual_z_delta_max=float(delta.max()),
        branch_input_amplitudes=branch_amplitudes,
        calibration={key: value for key, value in report['calibration'].items() if key not in ('direction', 'raw_gaussian')},
        scope=SCOPE)


def seal_case(actual, row, runtime, scan):
    folder = actual['folder']
    assert 'files_sha256' not in row and 'artifacts' not in row, 'Seal each case exactly once'
    write(folder / 'metrics.json', row, exclusive=True)
    files = ('metadata.json', 'components.pt', 'model_output.pt', 'actual_head_boundaries.pt', 'arrays.npz',
             'unit_site.pt', 'unit_site.json', 'unit_arrays.npz', 'metrics.json')
    row['files_sha256'] = {name: sha(folder / name) for name in files}
    row['artifacts'] = str(folder)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Completed baseline with feedback-inputs and action-correction-sites')
    baseline = parser.parse_args().output.resolve()
    paths = dict(normal_runtime=ROOT / 'work/serve_cosmos_goal_pair.py', factory=ROOT / 'work/probe_cosmos_language_routes.py',
        components=ROOT / 'work/cosmos_component_interventions.py', pulse=ROOT / 'work/probe_cosmos_milk_component_pulses.py',
        feedback=ROOT / 'work/probe_cosmos_milk_feedback_inputs.py', correction=ROOT / 'work/probe_cosmos_milk_action_correction_sites.py')
    scan = load_file('units_frozen_pulse_metrics', paths['pulse'])
    scan.validate_baseline(baseline)
    correction = load_file('units_frozen_correction_capture', paths['correction'])
    feedback, sites = baseline / 'feedback-inputs', baseline / 'action-correction-sites'
    assert not (feedback / 'failed.json').exists() and not (sites / 'failed.json').exists()
    feedback_complete, feedback_prov = read(feedback / 'complete.json'), read(feedback / 'provenance.json')
    complete, results, site_prov = read(sites / 'complete.json'), read(sites / 'results.json'), read(sites / 'provenance.json')
    assert feedback_complete['state'] == complete['state'] == results['state'] == 'complete'
    assert feedback_complete['seeds'] == [195, 196, 198]
    assert feedback_complete['native_replays_exact'] == 6 and feedback_complete['direct_model_forwards'] == 12
    assert feedback_complete['boundaries_per_forward'] == 37 and feedback_complete['components_per_forward'] == 72
    assert feedback_complete['summary_sha256'] == sha(feedback / 'summary.json')
    assert feedback_complete['script_sha256'] == feedback_prov['script_sha256'] == sha(paths['feedback'])
    assert complete['results_sha256'] == sha(sites / 'results.json')
    assert complete['script_sha256'] == site_prov['script_sha256'] == sha(paths['correction'])
    assert complete['scan_forwards'] == 72 and complete['exact_control_forwards'] == 4
    assert complete['full_return_head_boundaries_component_controls_bit_exact'] is True
    assert complete['all_actual_inputs_full_sigma_unchanged'] is complete['all_target_before_after_and_nonaction_rows_exact'] is True
    assert complete['seed'] == results['seed'] == SEED and complete['original_denoising_forward'] == STEP
    whole_row = next(row for row in results['cases'] if row['run'] == 'layer03_mlp')
    assert whole_row['input_context'] == 'PR' and whole_row['layer_zero_based'] == LAYER and whole_row['component'] == 'mlp'
    assert whole_row['restoration_xyz']['gap_meaningful'] is True
    whole_score = whole_row['restoration_xyz']['signed_projection_fraction']
    assert math.isfinite(whole_score) and whole_score > 0
    baseline_prov = read(baseline / 'server/provenance.json')
    for name in ('normal_runtime', 'factory', 'components'):
        assert sha(paths[name]) == baseline_prov[name + '_sha256'] == site_prov['source_sha256'][name] == feedback_prov['source_sha256'][name]
    assert site_prov['pulse_contract_sha256'] == feedback_prov['pulse_contract_sha256'] == sha(paths['pulse'])
    assert site_prov['feedback_complete_sha256'] == sha(feedback / 'complete.json')
    assert site_prov['feedback_provenance_sha256'] == sha(feedback / 'provenance.json')
    assert site_prov['feedback_summary_sha256'] == sha(feedback / 'summary.json')
    feedback_summary = read(feedback / 'summary.json')
    for key in ('all_native_R_and_P_t21_replays_bit_exact', 'entire_t21_model_output_including_vision_exact', 'current_condition_all_seeds_exact'):
        assert feedback_summary[key] is True
    out = baseline / 'mlp-units'
    out.mkdir(exist_ok=False)
    declaration = protocol()
    declaration['existing_whole_actual_restoration_fraction'] = whole_score
    declaration['preregistered_half_whole_threshold'] = .5 * whole_score
    write(out / 'protocol.json', declaration, exclusive=True)
    started, finished, cases, controls, hooks = time.perf_counter(), [], [], [], None

    def progress(stage, **fields):
        value = dict(stage=stage, completed_model_forwards=len(finished), expected_model_forwards=24,
                     completed=finished, elapsed_s=time.perf_counter() - started, **fields)
        write(out / 'progress.json', value)
        print('[MLP-UNITS] ' + json.dumps(value, allow_nan=False), flush=True)

    try:
        os.environ['HF_HUB_OFFLINE'] = '1'
        base = load_file('units_checked_normal_runtime', paths['normal_runtime'])
        factory = load_file('units_checked_factory', paths['factory'])
        factory.OUT = out
        assert sha(factory.SCHEDULER) == site_prov['scheduler_sha256'] == baseline_prov['scheduler_sha256']
        progress('loading_model')
        runtime = base.NormalRuntime(factory)
        torch, model = runtime.torch, runtime.pipe.transformer
        assert not model.training and not torch.is_grad_enabled() and not model.is_cache_enabled
        assert len(model.layers) == 36
        mlp = model.layers[LAYER].mlp_moe_gen
        assert mlp.hidden_act == 'silu' and isinstance(mlp.act_fn, torch.nn.SiLU)
        assert mlp.gate_proj.weight.shape == mlp.up_proj.weight.shape == (WIDTH, 4096)
        assert mlp.down_proj.weight.shape == (4096, WIDTH)
        assert mlp.gate_proj.bias is mlp.up_proj.bias is mlp.down_proj.bias is None
        assert all(parameter.dtype == torch.bfloat16 for parameter in mlp.parameters())
        transformer = Path(inspect.getfile(type(model)))
        assert sha(transformer) == site_prov['actual_transformer_sha256'] == feedback_prov['actual_transformer_sha256']
        pipeline = Path(inspect.getfile(type(runtime.pipe)))
        assert sha(pipeline) == site_prov['actual_pipeline_source_sha256'] == feedback_prov['actual_pipeline_source_sha256']
        module = load_file('units_readonly_component_capture', paths['components'])
        hooks = module.CosmosComponentInterventions(model)
        refs = {label: correction.load_reference(feedback / f'seed{SEED}/replays' / label, hooks, runtime, scan, label) for label in ('RR', 'PR')}
        rr, pr = refs['RR'], refs['PR']
        for label, reference in refs.items():
            assert sha(reference['folder'] / 'metadata.json') == site_prov['reference_metadata_sha256'][label]
        runtime.require_exact(rr['kwargs'].keys(), pr['kwargs'].keys(), 'fixed_RR_PR_actual_kwarg_keys')
        for key in rr['kwargs']:
            if key != 'action_tokens':
                runtime.require_exact(rr['kwargs'][key], pr['kwargs'][key], 'fixed_RR_PR_only_action_sample_differs/' + key)
        runtime.require_exact(rr['cache']['steps'], pr['cache']['steps'], 'fixed_RR_PR_pack_sigma_metadata')
        assert rr['kwargs']['action_tokens'][0].shape == pr['kwargs']['action_tokens'][0].shape == (16, 64)
        assert len(rr['kwargs']['action_tokens']) == len(pr['kwargs']['action_tokens']) == 1
        assert not torch.equal(rr['kwargs']['action_tokens'][0], pr['kwargs']['action_tokens'][0])
        whole = load_whole(sites / 'runs/layer03_mlp', whole_row, correction, runtime)
        runtime.require_exact(whole['cache']['first_model_kwargs'], pr['kwargs'], 'existing_whole_swap_actual_PR_context')
        contract = read(feedback / f'seed{SEED}/input-contract.json')
        assert contract['both_pure_noise_draws_exact'] is contract['complete_schedule_and_30step_metadata_exact'] is True
        assert contract['seed'] == SEED
        native_records, native_caches, input_sources = {}, {}, {}
        for role, scene in (('R', 'x15'), ('D', 'x06')):
            folder = baseline / 'closed-loop' / f'{scene}_seed{SEED}' / 'chunk_00'
            for name in ('states', 'components'):
                assert contract['sources'][role][name + '_sha256'] == sha(folder / (name + '.pt'))
            native_records[role] = torch.load(folder / 'states.pt', map_location='cpu', weights_only=True, mmap=True)
            native_caches[role] = torch.load(folder / 'components.pt', map_location='cpu', weights_only=True, mmap=True)
            scan.validate_record(native_records[role], torch)
            scan.validate_cache(native_caches[role], native_records[role], hooks, runtime, role)
            input_sources[role] = {name + '_sha256': sha(folder / (name + '.pt')) for name in ('states', 'components')}
        for key in ('pure_noise', 'timesteps', 'sigmas'):
            runtime.require_exact(native_records['R'][key], native_records['D'][key], 'original_complete_paired_contract/' + key)
        runtime.require_exact(native_caches['R']['steps'], native_caches['D']['steps'], 'original_complete_30step_metadata')
        runtime.require_exact(native_records['R']['model_input']['action_tokens'], native_records['D']['model_input']['action_tokens'], 'original_actual_initial_action_noise_inputs')
        saved_inputs = {}
        for label, subdir in (('R', 'native_R_q0'), ('P', 'pulse_P_q0')):
            path = feedback / f'seed{SEED}' / subdir / 'model_inputs_20_21.pt'
            saved_inputs[label] = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
            input_sources[label + '_actual_model_inputs_sha256'] = sha(path)
        runtime.require_exact(rr['kwargs'], saved_inputs['R'][STEP], 'RR_actual_saved_t21_input')
        expected_pr = dict(saved_inputs['R'][STEP])
        expected_pr['action_tokens'] = saved_inputs['P'][STEP]['action_tokens']
        runtime.require_exact(pr['kwargs'], expected_pr, 'PR_actual_saved_action_only_t21_input')
        saved_r_path = feedback / f'seed{SEED}' / 'native_R_q0/states.pt'
        runtime.require_exact(torch.load(saved_r_path, map_location='cpu', weights_only=True, mmap=True), native_records['R'], 'saved_native_R_complete_record')
        r_outputs_path = feedback / f'seed{SEED}' / 'native_R_q0/model_outputs_20_21.pt'
        saved_r_outputs = torch.load(r_outputs_path, map_location='cpu', weights_only=True, mmap=True)
        runtime.require_exact(rr['model_output'], saved_r_outputs[STEP], 'RR_actual_saved_t21_full_model_return')
        runtime.require_exact(rr['readout'], native_records['R']['readout'][STEP], 'RR_actual_native_t21_readout')
        runtime.require_exact(rr['velocity'], native_records['R']['action_velocity'][STEP], 'RR_actual_native_t21_velocity')
        for layer in range(36):
            for component in ('attention', 'mlp'):
                runtime.require_exact(rr['cache']['frames'][(0, layer, component)], native_caches['R']['frames'][(STEP, layer, component)], 'RR_original_t21_component')
        input_sources.update(saved_native_R_states_sha256=sha(saved_r_path), saved_native_R_outputs_sha256=sha(r_outputs_path))
        natural_input_delta = pr['kwargs']['action_tokens'][0].double() - rr['kwargs']['action_tokens'][0].double()
        write(out / 'input-contract.json', dict(seed=SEED, original_denoising_forward=STEP,
            both_actual_pure_noise_draws_exact=True, all_30_schedule_pack_metadata_exact=True,
            actual_RR_and_PR_from_saved_t21_inputs=True, PR_only_action_input_differs=True,
            RR_native_full_return_head_and_all_components_exact=True, raw_sources=input_sources,
            natural_action_xyz_delta_l2=float(natural_input_delta[:, :3].norm()),
            natural_action_other61_delta_l2=float(natural_input_delta[:, 3:].norm()),
            natural_action_all64_delta_l2=float(natural_input_delta.norm()), scope=SCOPE), exclusive=True)
        progress('hash_all_parameter_bytes_before')
        parameter_hash_started = time.perf_counter()
        parameter_before = parameter_hashes(model, torch)
        parameter_before_elapsed = time.perf_counter() - parameter_hash_started
        write(out / 'parameters-before.json', parameter_before, exclusive=True)
        write(out / 'provenance.json', dict(script_sha256=sha(Path(__file__)), protocol_sha256=sha(out / 'protocol.json'),
            sources={name: dict(path=str(path), sha256=sha(path)) for name, path in paths.items()},
            input_json_sha256={str(path): sha(path) for path in (feedback / 'complete.json', feedback / 'provenance.json',
                feedback / f'seed{SEED}/input-contract.json', sites / 'complete.json', sites / 'results.json', sites / 'provenance.json')},
            whole_metadata_sha256=whole_row['metadata_sha256'], actual_transformer_sha256=sha(transformer),
            actual_pipeline_source_sha256=sha(pipeline), input_contract_sha256=sha(out / 'input-contract.json'),
            actual_MLP_forward_source_sha256=hashlib.sha256(inspect.getsource(type(mlp).forward).encode()).hexdigest(),
            actual_Linear_forward_source_sha256=hashlib.sha256(inspect.getsource(type(mlp.down_proj).forward).encode()).hexdigest(),
            actual_torch_version=torch.__version__, full_parameter_before_sha256=sha(out / 'parameters-before.json'),
            full_parameter_before_hash_elapsed_s=parameter_before_elapsed,
            library_edits=False, training=False, model_calls_expected=24, scope=SCOPE), exclusive=True)
        captured = {}
        for context in ('RR', 'PR'):
            label = 'native_' + context
            progress('native_exact_capture', run=label)
            actual = run(runtime, hooks, correction, scan, refs[context], label, out / label, exact=True)
            captured[context] = actual
            finished.append(label)
            controls.append(dict(run=label, full_return_head_boundaries_all_components_bit_exact=True,
                unit_site_sha256=sha(actual['folder'] / 'unit_site.pt'), metadata_sha256=sha(actual['folder'] / 'metadata.json'),
                files_sha256={name: sha(actual['folder'] / name) for name in ('unit_site.json', 'unit_arrays.npz')}))
        rsite, psite = captured['RR']['unit'], captured['PR']['unit']
        runtime.require_exact(rsite['action_rows'], psite['action_rows'], 'native_unit_rows_align')
        rows = rsite['action_rows']
        rr_units, pr_units = rsite['site'], psite['site']
        runtime.require_exact(rr_units['down'][rows], rr['cache']['frames'][(0, LAYER, 'mlp')], 'actual_RR_MLP_increment')
        runtime.require_exact(pr_units['down'][rows], pr['cache']['frames'][(0, LAYER, 'mlp')], 'actual_PR_MLP_increment')
        progress('freeze_proxy_selection')
        down_weight = runtime.cpu(mlp.down_proj.weight)
        selection, proxy = select_units(dict(rows=rows, site=rr_units), dict(rows=rows, site=pr_units), down_weight, torch)
        groups = selection['nested_indices']
        all_selected = sorted(set(groups['512']).union(*(set(value['indices']) for value in selection['random_support_sets'].values())))
        torch.save(dict(indices=all_selected, actual_down_W_columns=down_weight[:, all_selected],
            native_RR_action_z=rr_units['z_down_input'][rows], native_PR_action_z=pr_units['z_down_input'][rows],
            proxy_signed_scores=proxy), out / 'selection-arrays.pt')
        selection.update(selection_arrays_sha256=sha(out / 'selection-arrays.pt'),
            native_unit_capture_sha256={context: sha(captured[context]['folder'] / 'unit_site.pt') for context in ('RR', 'PR')},
            actual_full_down_weight=scan.tensor_meta(down_weight, torch), entire_transformer_parameter_bytes_sha256=parameter_before['aggregate_sha256'])
        write(out / 'selection.json', selection, exclusive=True)
        selection_sha = sha(out / 'selection.json')
        write(out / 'selection-frozen.json', dict(selection_sha256=selection_sha, protocol_sha256=sha(out / 'protocol.json'),
            existing_whole_restoration=whole_score, before_first_intervention_output_score=True), exclusive=True)

        def trial(label, reference=pr, native=pr_units, source=rr_units, indices=(), mode='z_swap', target=None, seed=None,
                  endpoint=rr, context='PR', metric_reference=None, extra_metrics=None):
            assert sha(out / 'selection.json') == selection_sha
            progress('single_forward', run=label)
            actual = run(runtime, hooks, correction, scan, reference, label, out / label,
                native, source, indices, mode, target, seed)
            finished.append(label)
            row = metric_row(actual, reference if metric_reference is None else metric_reference, endpoint, scan, label, context)
            if extra_metrics is not None:
                row.update(extra_metrics)
            cases.append(seal_case(actual, row, runtime, scan))
            return actual, row

        self_actual, _ = trial('self_all12288', indices=range(WIDTH), mode='z_self', source=pr_units)
        correction.exact_reference_result(self_actual, pr, runtime, 'all_unit_self_ENTIRE_record')
        controls.append(dict(run='self_all12288', full_return_head_boundaries_all_components_bit_exact=True))
        full, _ = trial('swap_all12288', indices=range(WIDTH))
        runtime.require_exact(full['unit']['site']['down'][rows], rr_units['down'][rows], 'all_unit_actual_down_action_equals_RR')
        correction.exact_reference_result(full, whole, runtime, 'preW_all_units_ENTIRE_existing_postW_whole_swap_record')
        controls.append(dict(run='swap_all12288', full_return_head_boundaries_all_components_bit_exact=True))
        sparse = {}
        for size in SIZES:
            sparse[size] = trial(f'natural{size:03d}', indices=groups[str(size)])
        trial('reverse128', reference=rr, native=rr_units, source=pr_units, indices=groups['128'], endpoint=pr, context='RR_reverse')
        factorial = {}
        for cell, mode in (('PP', 'factor_self'), ('RP', 'factor_g'), ('PR', 'factor_u'), ('RR', 'factor_joint')):
            factorial[cell] = trial('factor128_' + cell, indices=groups['128'], mode=mode)
        correction.exact_reference_result(factorial['PP'][0], pr, runtime, 'factor_PP_ENTIRE_native_PR_self_record')
        runtime.require_exact(factorial['RR'][0]['unit']['site']['z_landed'], sparse[128][0]['unit']['site']['z_landed'], 'factor_joint128_actual_z_equals_natural128')
        correction.exact_reference_result(factorial['RR'][0], sparse[128][0], runtime, 'factor_joint_ENTIRE_natural128_record')
        target = sparse[128][0]['unit']['site']['actual_down_delta_l2']
        assert target > 0, 'Natural128 has no actual down-output change to norm-match'
        for seed in SUPPORT_SEEDS:
            trial(f'random_support128_seed{seed}', indices=selection['random_support_sets'][str(seed)]['indices'],
                  mode='random_support', target=target, seed=seed)
        for seed in COEFFICIENT_SEEDS:
            trial(f'random_coefficients128_seed{seed}', indices=groups['128'], mode='random_coefficients', target=target, seed=seed)
        generic = []
        for seed in INPUT_SEEDS:
            reference, audit = input_control(rr, pr, seed, torch, runtime)
            label = f'generic_native_seed{seed}'
            progress('generic_native_forward', run=label)
            original = run(runtime, hooks, correction, scan, reference, label, out / label)
            finished.append(label)
            torch.save(audit, original['folder'] / 'generic_input.pt')
            write(original['folder'] / 'generic_input.json', json_tensors(audit, scan, torch))
            row = metric_row(original, original, rr, scan, label, 'RR_plus_orthogonal_XYZ')
            row.update(generic_input_sha256=sha(original['folder'] / 'generic_input.pt'), generic_input_metadata_sha256=sha(original['folder'] / 'generic_input.json'))
            cases.append(seal_case(original, row, runtime, scan))
            gap = rr['velocity'][:, :3].double() - original['velocity'][:, :3].double()
            natural_gap = rr['velocity'][:, :3].double() - pr['velocity'][:, :3].double()
            gap_cosine = None if float(gap.norm()) == 0 else float((gap * natural_gap).sum() / (gap.norm() * natural_gap.norm()))
            # Each Gi's observed native velocity sets its own restoration denominator.
            changed, row = trial(f'generic_restore128_seed{seed}', reference=reference, native=original['unit']['site'],
                source=rr_units, indices=groups['128'], endpoint=rr, context='RR_plus_orthogonal_XYZ',
                metric_reference=original, extra_metrics=dict(generic_gap_cosine_with_natural_PR_gap=gap_cosine,
                    generic_native_run=label, generic_input_sha256=sha(original['folder'] / 'generic_input.pt')))
            generic.append(dict(seed=seed, native_run=label, patched_run=row['run'], own_axis_restoration=row['restoration_xyz']))
        assert len(finished) == 24 and len(cases) == 22 and len(controls) == 4
        assert sha(out / 'selection.json') == selection_sha
        progress('verify_all_parameter_bytes_unchanged')
        parameter_hash_started = time.perf_counter()
        parameter_after = parameter_hashes(model, torch)
        parameter_after_elapsed = time.perf_counter() - parameter_hash_started
        assert parameter_after == parameter_before, 'Transformer parameter bytes changed'
        write(out / 'parameters-after.json', parameter_after, exclusive=True)
        r128 = sparse[128][1]['restoration_xyz']['signed_projection_fraction']
        gate = factorial['RP'][1]['restoration_xyz']['signed_projection_fraction']
        up = factorial['PR'][1]['restoration_xyz']['signed_projection_fraction']
        assessable = all(value is not None for value in (r128, gate, up))
        predictions = dict(half_whole_threshold=.5 * whole_score, whole_actual_restoration=whole_score,
            natural128_restoration=r128, gate_only_restoration=gate, up_only_restoration=up,
            natural128_at_least_half_whole=None if r128 is None else r128 >= .5 * whole_score,
            natural_gate_stronger_than_up=None if gate is None or up is None else gate > up,
            verdict='inconclusive_gap_floor' if not assessable else 'supported' if r128 >= .5 * whole_score and gate > up else 'rejected',
            meaning='Two preregistered raw-velocity predictions only; not target specificity, selection semantics or physical correction.')
        interaction = (factorial['RR'][0]['velocity'].double() - factorial['RP'][0]['velocity'].double()
                       - factorial['PR'][0]['velocity'].double() + factorial['PP'][0]['velocity'].double())
        torch.save(dict(raw_velocity_all64=interaction, raw_velocity_xyz=interaction[:, :3]), out / 'factorial-interaction.pt')
        calibration_calls = sum(row['calibration']['direct_down_forward_calls'] for row in cases)
        write(out / 'results.json', dict(state='complete', seed=SEED, original_denoising_forward=STEP,
            model_forwards=24, real_model_down_module_calls=24, extra_direct_down_forward_calibration_calls=calibration_calls,
            completed=finished, controls=controls, cases=cases, predictions=predictions, generic_controls=generic,
            factorial_interaction_raw_xyz=interaction[:, :3].tolist(), factorial_interaction_sha256=sha(out / 'factorial-interaction.pt'),
            selection_sha256=selection_sha, protocol_sha256=sha(out / 'protocol.json'),
            parameter_hash_files_sha256={name: sha(out / name) for name in ('parameters-before.json', 'parameters-after.json')},
            parameter_byte_hash_elapsed_s=dict(before=parameter_before_elapsed, after=parameter_after_elapsed),
            all_transformer_parameter_bytes_unchanged=True, scope=SCOPE), exclusive=True)
        write(out / 'complete.json', dict(state='complete', model_forwards=24, real_model_down_module_calls=24,
            extra_direct_down_forward_calibration_calls=calibration_calls,
            native_and_self_entire_records_bit_exact=True, all12288_preW_entire_existing_whole_record_bit_exact=True,
            joint128_entire_natural128_record_bit_exact=True, all_selected_unit_and_nonaction_footprints_exact=True,
            all_real_down_output_random_norm_errors_within_fixed_tolerance=True,
            all_actual_generic_inputs_norm_matched_and_orthogonal=True, all_parameter_bytes_unchanged=True,
            results_sha256=sha(out / 'results.json'), protocol_sha256=sha(out / 'protocol.json'), selection_sha256=selection_sha,
            provenance_sha256=sha(out / 'provenance.json'), script_sha256=sha(Path(__file__)), scope=SCOPE), exclusive=True)
        progress('complete', hypothesis_verdict=predictions['verdict'])
    except Exception as exc:
        write(out / 'failed.json', dict(state='failed', error=str(exc), traceback=traceback.format_exc(),
            completed=finished, elapsed_s=time.perf_counter() - started, scope=SCOPE))
        raise
    finally:
        if hooks is not None:
            hooks.reset()


if __name__ == '__main__':
    main()
