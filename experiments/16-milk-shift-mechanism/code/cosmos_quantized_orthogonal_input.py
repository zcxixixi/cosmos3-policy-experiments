"""CPU-only construction of an actually rounded orthogonal XYZ control.

Repair the input constructor, not the model or the registered tolerances.
The first passing input is selected without any model-output measurements.
"""

import math

NORM_TOL = 1e-3
ORTHOGONAL_TOL = 1e-3
MAX_ITERATIONS = 10000


def build_input(ar, ap, seed, torch):
    assert ar.device.type == ap.device.type == 'cpu'
    assert ar.shape == ap.shape == (16, 64) and ar.dtype == ap.dtype
    assert torch.isfinite(ar).all() and torch.isfinite(ap).all()
    base = ar[:, :3].double()
    natural = ap.double() - ar.double()
    xyz = natural[:, :3]
    target = float(xyz.norm())
    assert math.isfinite(target) and target > 0
    axis = xyz / target
    generator = torch.Generator(device='cpu').manual_seed(seed)
    raw = torch.randn(xyz.shape, generator=generator, dtype=torch.float64, device='cpu')
    unit = raw - xyz * ((raw * xyz).sum() / xyz.square().sum())
    unit = unit / unit.norm()
    assert abs(float((unit * xyz).sum() / target)) <= 1e-10
    desired = target * unit
    desired_history, landed_history, trace = [], [], []
    passed = False
    for iteration in range(MAX_ITERATIONS):
        desired_history.append(desired.clone())
        landed = (base + desired).to(ar.dtype)
        landed_history.append(landed.clone())
        actual = landed.double() - base
        norm = float(actual.norm())
        assert math.isfinite(norm) and norm > 0
        cosine = float((actual * axis).sum() / norm)
        error = abs(norm - target) / target
        trace.append(dict(iteration=iteration + 1, actual_delta_l2=norm,
                          actual_norm_relative_error=error, actual_cosine=cosine))
        if error <= NORM_TOL and abs(cosine) <= ORTHOGONAL_TOL:
            passed = True
            break
        # Correct both errors observed AFTER rounding, in the unrounded proposal.
        desired = desired + actual * (target / norm - 1) - axis * (actual * axis).sum()
    assert passed, 'Actual rounded norm/orthogonality gate failed; do not widen tolerance'
    changed = ar.clone()
    changed[:, :3] = landed
    assert torch.equal(changed[:, 3:].view(torch.uint8), ar[:, 3:].contiguous().view(torch.uint8))
    actual = changed[:, :3].double() - base
    actual_norm = float(actual.norm())
    actual_cosine = float((actual * xyz).sum() / (actual_norm * target))
    actual_error = abs(actual_norm - target) / target
    assert actual_error <= NORM_TOL and abs(actual_cosine) <= ORTHOGONAL_TOL
    return changed, dict(
        cpu_generator_seed=seed, raw_direction=raw, projected_direction=unit,
        natural_full_delta=natural, actual_xyz_delta=actual, actual_action_tokens=changed,
        natural_xyz_delta_l2=target, natural_other61_delta_l2=float(natural[:, 3:].norm()),
        natural_all64_delta_l2=float(natural.norm()),
        actual_xyz_delta_l2=actual_norm, actual_norm_relative_error=actual_error,
        actual_cosine_with_natural_xyz=actual_cosine,
        actual_cosine_with_original_random_orthogonal_direction=float((actual * unit).sum() / actual_norm),
        desired_delta_history=torch.stack(desired_history),
        actual_landed_xyz_history=torch.stack(landed_history), calibration_trace=trace,
        calibration_iterations=len(trace), maximum_iterations=MAX_ITERATIONS,
        actual_dtype=str(ar.dtype), norm_tolerance=NORM_TOL,
        orthogonal_cosine_tolerance=ORTHOGONAL_TOL,
        constructor='Deterministic feedback of actual rounded length and natural-axis error; first passing iteration.',
        selection_uses_model_outputs=False,
        note='XYZ-only control, not the complete PR action change; original 18-forward failure is preserved.')
