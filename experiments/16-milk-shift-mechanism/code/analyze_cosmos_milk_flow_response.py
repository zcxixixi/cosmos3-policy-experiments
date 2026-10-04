"""CPU interpretation of saved t21 Flow responses; no new inference or physics.

Check the frozen 18+6 experiment, actual scheduler/pipeline sources, saved
FP32 solver samples, and their actual BF16 model inputs. Apply the verified
x_clean = x_sigma - sigma*v conversion without taking any scheduler step.
PR is a counterfactual P-action/R-vision replay; PP is the native P replay.
Generic cases have artificial BF16 inputs, not saved FP32 solver trajectories.
Only an absent BASE/flow-response-analysis directory may be written.
"""

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path


ROOT = Path('/home/current/work/cosmos3')
STEP = 21
SEEDS = (5219801, 5219802, 5219803)
PLOT_SHA = 'd9a3f91c1ed3de83236c40f73892bd41a630d1ec1b8f343767b03d758aeaddc7'
SCHEDULER = ROOT / 'cosmos-framework/cosmos_framework/model/generator/diffusion/samplers/fm_solvers_unipc.py'
PIPELINE = ROOT / 'diffusers/src/diffusers/pipelines/cosmos/pipeline_cosmos3_omni.py'
SCHEDULER_SHA = '03aef1959f273b704ca4954f69b2a34df0fdd412f6acc8ab91625eeed78cf4fe'
PIPELINE_SHA = 'ec2f051ec5703926c82145cca24bfcb93222019234fade9aafa905f7ae589eca'
SCOPE = ('Post hoc CPU interpretation of existing seed198 fixed-t21 raw Flow XYZ outputs. '
         'The same-clean reference dv=dx/sigma is an algebraic comparison, not an identified '
         'mechanism, memory, object identity, grasp result or fraction of causal explanation. '
         'The converted clean estimate is not the next multistep solver state or final action.')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text())


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_mask(torch):
    """Execute only the exact checked static masking function, not the pipeline."""
    text = PIPELINE.read_text()
    tree = ast.parse(text)
    methods = [node for parent in tree.body if isinstance(parent, ast.ClassDef)
               for node in parent.body if isinstance(node, ast.FunctionDef)
               and node.name == '_mask_velocity_predictions']
    assert len(methods) == 1
    node = copy.deepcopy(methods[0])
    assert any(isinstance(item, ast.Name) and item.id == 'staticmethod' for item in node.decorator_list)
    node.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
    namespace = dict(torch=torch)
    exec(compile(ast.fix_missing_locations(module), str(PIPELINE), 'exec'), namespace)
    segment = ast.get_source_segment(text, methods[0])
    return namespace[node.name], dict(method=node.name, first_line=methods[0].lineno,
        last_line=methods[0].end_lineno, method_source_sha256=hashlib.sha256(segment.encode()).hexdigest())


def array_fields(folder, torch, runtime):
    cache = torch.load(folder / 'components.pt', map_location='cpu', weights_only=True, mmap=True)
    output = torch.load(folder / 'model_output.pt', map_location='cpu', weights_only=True, mmap=True)
    assert isinstance(output, tuple) and len(output) == 3
    assert isinstance(output[2], list) and len(output[2]) == 1
    velocity = output[2][0]
    assert velocity.device.type == 'cpu' and velocity.shape == (16, 64) and velocity.dtype == torch.bfloat16
    assert torch.isfinite(velocity).all()
    head_path = folder / 'actual_head_boundaries.pt'
    if head_path.exists():
        head = torch.load(head_path, map_location='cpu', weights_only=True, mmap=True)
        runtime.require_exact(velocity, head['action_velocity'], 'actual action head/tuple[2][0]')
    else:
        import numpy as np
        with np.load(folder / 'arrays.npz', allow_pickle=False) as data:
            expected = velocity.float().numpy()
            assert data['action_velocity'].dtype == expected.dtype and data['action_velocity'].shape == expected.shape
            assert data['action_velocity'].tobytes(order='C') == expected.tobytes(order='C')
    sources = {name: sha(folder / name) for name in ('components.pt', 'model_output.pt', 'metadata.json', 'arrays.npz')}
    if head_path.exists():
        sources['actual_head_boundaries.pt'] = sha(head_path)
    return dict(kwargs=cache['first_model_kwargs'], output=output, raw_velocity=velocity, sources=sources)


def metrics(dx, dv, sigma, torch):
    """All arrays are XYZ; the denominator is each case's own input difference."""
    x, v = dx.double(), dv.double()
    norm = float(x.norm())
    assert x.shape == v.shape == (16, 3) and math.isfinite(norm)
    meaningful = norm / math.sqrt(x.numel()) > 1e-6
    clean = x - sigma * v
    gain = float((x * v).sum()) / (norm * norm) if meaningful else None
    parallel = None if gain is None else 1 - sigma * gain
    perpendicular = None if parallel is None else clean - parallel * x
    vnorm = float(v.norm())
    return dict(input_xyz_l2=norm, velocity_xyz_delta_l2=vnorm,
        input_xyz_rms=norm / math.sqrt(x.numel()), velocity_xyz_delta_rms=vnorm / math.sqrt(x.numel()),
        gap_meaningful=meaningful, input_gap_absolute_rms_floor=1e-6,
        velocity_gain_along_input=gain, same_clean_reference_gain=1 / sigma,
        cosine_velocity_with_input=None if not meaningful or vnorm == 0 else float((x * v).sum()) / (norm * vnorm),
        input_axis_subtraction_fraction=None if gain is None else sigma * gain,
        clean_difference_parallel_fraction=parallel,
        clean_difference_norm_fraction=None if not meaningful else float(clean.norm()) / norm,
        clean_difference_perpendicular_norm_fraction=None if perpendicular is None else float(perpendicular.norm()) / norm,
        clean_difference_xyz_l2=float(clean.norm()),
        definitions='gain=dot(dx,dv)/||dx||²; dc=dx-sigma*dv; parallel=dot(dc,dx)/||dx||²; norm=||dc||/||dx||. XYZ has 48 values; no physical unit conversion.')


def audit(baseline, torch):
    source = Path(__file__).resolve().parent
    plot_path = source / 'plot_cosmos_milk_mlp_units.py'
    assert sha(plot_path) == PLOT_SHA
    assert sha(SCHEDULER) == SCHEDULER_SHA and sha(PIPELINE) == PIPELINE_SHA
    plot = load('flow_frozen_units_audit', plot_path)
    _, previous_values, previous_sources, _ = plot.audit(baseline, torch)
    runtime = plot.CPUValues(torch)
    feedback = baseline / 'feedback-inputs'
    complete, summary, provenance = (read(feedback / (name + '.json')) for name in ('complete', 'summary', 'provenance'))
    assert not (feedback / 'failed.json').exists() and complete['state'] == 'complete'
    assert complete['summary_sha256'] == sha(feedback / 'summary.json')
    assert complete['script_sha256'] == provenance['script_sha256'] == sha(source / 'probe_cosmos_milk_feedback_inputs.py')
    assert provenance['scheduler_sha256'] == SCHEDULER_SHA and provenance['actual_pipeline_source_sha256'] == PIPELINE_SHA
    assert summary['all_native_R_and_P_t21_replays_bit_exact'] is summary['entire_t21_model_output_including_vision_exact'] is True
    original = baseline / 'mlp-units'
    phase2 = baseline / 'mlp-generic-inputs'
    phase1_prov = read(original / 'provenance.json')
    assert phase1_prov['actual_pipeline_source_sha256'] == PIPELINE_SHA
    site_prov = read(baseline / 'action-correction-sites/provenance.json')
    assert site_prov['scheduler_sha256'] == SCHEDULER_SHA
    folder = feedback / 'seed198'
    contract = read(folder / 'input-contract.json')
    assert contract['seed198_P_all_coarse_record_fields_exact'] is True
    records, observed_inputs, observed_outputs, record_sources = {}, {}, {}, {}
    for role, name in (('R', 'native_R_q0'), ('P', 'pulse_P_q0')):
        directory = folder / name
        records[role] = torch.load(directory / 'states.pt', map_location='cpu', weights_only=True, mmap=True)
        observed_inputs[role] = torch.load(directory / 'model_inputs_20_21.pt', map_location='cpu', weights_only=True, mmap=True)
        observed_outputs[role] = torch.load(directory / 'model_outputs_20_21.pt', map_location='cpu', weights_only=True, mmap=True)
        metadata = read(directory / 'metadata.json')
        assert metadata['seed'] == 198 and metadata['raw_action_dim'] == 10 and metadata['model_calls'] == 30
        assert metadata['observational_input_capture_steps'] == [20, 21] and metadata['role'] == role
        record_sources[role] = {name: sha(directory / name) for name in
            ('states.pt', 'metadata.json', 'model_inputs_20_21.pt', 'model_outputs_20_21.pt')}
        plot.load('flow_record_validator', source / 'probe_cosmos_milk_component_pulses.py').validate_record(records[role], torch)
    r, p = records['R'], records['P']
    original_r = baseline / 'closed-loop/x15_seed198/chunk_00/states.pt'
    assert sha(original_r) == contract['sources']['R']['states_sha256']
    runtime.require_exact(r, torch.load(original_r, map_location='cpu', weights_only=True, mmap=True), 'R entire native saved record')
    coarse = read(baseline / 'component-pulses/results.json')
    matches = [row for row in coarse['cases'] if (row['layer'], row['component'], row['pulse_step']) == (17, 'attention', 20)]
    assert len(matches) == 1
    pulse_source = baseline / 'component-pulses/runs' / matches[0]['run'] / 'states.pt'
    assert sha(pulse_source) == matches[0]['states_sha256']
    runtime.require_exact(p, torch.load(pulse_source, map_location='cpu', weights_only=True, mmap=True), 'P entire saved coarse pulse record')
    for key in ('pure_noise', 'timesteps', 'sigmas', 'prepared_latents_and_masks'):
        runtime.require_exact(r[key], p[key], 'R/P original preparation/' + key)
    sampler = load('flow_actual_CPU_scheduler', SCHEDULER)
    scheduler = sampler.FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1.)
    scheduler.set_timesteps(30, device='cpu')
    runtime.require_exact(scheduler.timesteps, r['timesteps'], 'actual scheduler 30 timesteps')
    runtime.require_exact(scheduler.sigmas, r['sigmas'], 'actual scheduler 31 sigmas')
    assert scheduler.config.prediction_type == 'flow_prediction' and scheduler.predict_x0 and not scheduler.config.thresholding
    scheduler._step_index = STEP
    sigma = float(scheduler.sigmas[STEP])
    assert sigma > 0 and int(r['timesteps'][STEP]) == 299
    mask, mask_source = source_mask(torch)
    prepared = r['prepared_latents_and_masks']
    assert isinstance(prepared, tuple) and len(prepared) == 12 and prepared[10] == 10
    action_mask = prepared[7]
    assert action_mask.shape == (16, 1) and torch.count_nonzero(action_mask) == 0
    refs = {label: array_fields(folder / 'replays' / label, torch, runtime) for label in ('RR', 'PR', 'PP')}
    rr, pr, pp = (refs[label] for label in ('RR', 'PR', 'PP'))
    for role, label in (('R', 'RR'), ('P', 'PP')):
        runtime.require_exact(refs[label]['kwargs'], observed_inputs[role][STEP], 'native observed kwargs/' + label)
        runtime.require_exact(refs[label]['output'], observed_outputs[role][STEP], 'native entire observed output/' + label)
        runtime.require_exact(refs[label]['raw_velocity'], records[role]['action_velocity'][STEP], 'native raw actual action head/' + label)
        runtime.require_exact(records[role]['action_states'][STEP, 0].to(torch.bfloat16), refs[label]['kwargs']['action_tokens'][0], 'actual FP32 state cast/' + label)
    runtime.require_exact(pr['kwargs']['action_tokens'], pp['kwargs']['action_tokens'], 'PR P action input')
    runtime.require_exact({k: v for k, v in pr['kwargs'].items() if k != 'action_tokens'},
                          {k: v for k, v in rr['kwargs'].items() if k != 'action_tokens'}, 'PR all other kwargs are RR')
    inputs = {label: ref['kwargs']['action_tokens'][0].float() for label, ref in refs.items()}
    variants = {label: ref for label, ref in refs.items()}
    for name in ('natural128', 'swap_all12288'):
        variants[name] = array_fields(original / name, torch, runtime)
        runtime.require_exact(variants[name]['kwargs'], pr['kwargs'], 'PR unit intervention input/' + name)
    generic_audits = {}
    for seed in SEEDS:
        name, changed_name = f'generic_native_seed{seed}', f'generic_restore128_seed{seed}'
        for label in (name, changed_name):
            variants[label] = array_fields(phase2 / label, torch, runtime)
        changed_kwargs = variants[name]['kwargs']
        runtime.require_exact(variants[changed_name]['kwargs'], changed_kwargs, 'generic same input native and patched')
        runtime.require_exact({k: v for k, v in changed_kwargs.items() if k != 'action_tokens'},
                              {k: v for k, v in rr['kwargs'].items() if k != 'action_tokens'}, 'generic all other RR kwargs')
        actual_input = changed_kwargs['action_tokens'][0]
        assert actual_input.dtype == torch.bfloat16 and actual_input.shape == (16, 64)
        runtime.require_exact(actual_input[:, 3:], rr['kwargs']['action_tokens'][0][:, 3:], 'generic untouched other 61 dimensions')
        inputs[name] = actual_input.float()
        lift = r['action_states'][STEP, 0] + (inputs[name] - inputs['RR'])
        lift_exact = torch.equal(lift.to(torch.bfloat16).reshape(-1).view(torch.uint8), actual_input.reshape(-1).view(torch.uint8))
        generic_audits[str(seed)] = dict(proposed_FP32_lift_cast_matches_actual_BF16_input=lift_exact,
            proposed_lift_is_saved_solver_sample=False, convention='Actual BF16 model input converted losslessly to FP32; no saved generic solver sample.',
            audit_pt_sha256=sha(phase2 / f'input_seed{seed}/audit.pt'), audit_json_sha256=sha(phase2 / f'input_seed{seed}/audit.json'))
    arrays, rows, masking = {}, [], {}
    for label, actual in variants.items():
        output = actual['output']
        raw_before_mask = actual['raw_velocity'].clone()
        _, _, velocity = mask(preds_vision=output[0], preds_sound=output[1], vision_condition_mask=[prepared[5]],
            sound_condition_mask=None if prepared[6] is None else [prepared[6]], preds_action=output[2],
            action_condition_mask=[action_mask], raw_action_dim=prepared[10])
        runtime.require_exact(output[2][0], raw_before_mask, 'source mask preserves actual raw model output')
        runtime.require_exact(actual['raw_velocity'], raw_before_mask, 'source mask preserves raw alias')
        assert velocity.dtype == torch.bfloat16 and velocity.shape == (16, 64)
        runtime.require_exact(velocity[:, :10], actual['raw_velocity'][:, :10], 'actual first10 raw velocity equals masked')
        assert torch.count_nonzero(velocity[:, 10:]) == 0
        actual['masked_velocity'] = velocity.float()
        arrays[label + '_raw_velocity'] = actual['raw_velocity'].float().numpy()
        arrays[label + '_masked_velocity'] = actual['masked_velocity'].numpy()
        masking[label] = dict(raw_padded54_l2=float(actual['raw_velocity'][:, 10:].double().norm()),
            masked_padded54_l2=0., first10_and_XYZ_bit_exact=True, raw_output_unmodified_by_source_mask_byte_exact=True,
            source_files_sha256=actual['sources'])
    cases = [dict(key='native_PP_FP32', label='Native P: FP32 solver states', convention='Two saved native FP32 action samples; P action and P future vision.',
                  native='PP', altered=None, sample_r=r['action_states'][STEP, 0], sample_x=p['action_states'][STEP, 0]),
             dict(key='counterfactual_PR_FP32', label='PR: saved FP32 action; R vision', convention='Actual saved P/R FP32 action samples with counterfactual PR velocity at unchanged R vision; not a native P trajectory.',
                  native='PR', altered='natural128', sample_r=r['action_states'][STEP, 0], sample_x=p['action_states'][STEP, 0]),
             dict(key='PR_BF16_convention', label='PR: BF16 input convention', convention='Actual PR/RR BF16 model inputs losslessly cast to FP32; separate denominator from saved FP32 samples.',
                  native='PR', altered='natural128', sample_r=inputs['RR'], sample_x=inputs['PR'])]
    for number, seed in enumerate(SEEDS, 1):
        name = f'generic_native_seed{seed}'
        cases.append(dict(key=f'generic{number}_BF16_convention', label=f'Generic {number}: BF16 input only', convention=generic_audits[str(seed)]['convention'],
            native=name, altered=f'generic_restore128_seed{seed}', sample_r=inputs['RR'], sample_x=inputs[name], seed=seed))
    for case in cases:
        key, sample_r, sample_x = case['key'], case['sample_r'], case['sample_x']
        assert sample_r.dtype == sample_x.dtype == torch.float32 and sample_r.shape == sample_x.shape == (16, 64)
        assert torch.count_nonzero(sample_r[:, 10:]) == torch.count_nonzero(sample_x[:, 10:]) == 0
        v_r, v_x = variants['RR']['masked_velocity'], variants[case['native']]['masked_velocity']
        clean_r = scheduler.convert_model_output(v_r, sample=sample_r)
        clean_x = scheduler.convert_model_output(v_x, sample=sample_x)
        runtime.require_exact(clean_r, sample_r - scheduler.sigmas[STEP] * v_r, 'actual source FP32 clean conversion/R')
        runtime.require_exact(clean_x, sample_x - scheduler.sigmas[STEP] * v_x, 'actual source FP32 clean conversion/X')
        dx, dv = sample_x.double()[:, :3] - sample_r.double()[:, :3], v_x.double()[:, :3] - v_r.double()[:, :3]
        algebra = dx - sigma * dv
        fp32_difference = clean_x.double()[:, :3] - clean_r.double()[:, :3]
        row = {k: value for k, value in case.items() if k not in ('sample_r', 'sample_x')}
        row.update(metrics=metrics(dx, dv, sigma, torch),
            input_other61_delta_l2=float((sample_x.double()[:, 3:] - sample_r.double()[:, 3:]).norm()),
            input_all64_delta_l2=float((sample_x.double() - sample_r.double()).norm()),
            actual_source_FP32_conversion_difference_l2=float(fp32_difference.norm()),
            actual_source_FP32_vs_Float64_algebra_difference_l2=float((fp32_difference - algebra).norm()),
            sample_dtype='torch.float32', model_input_dtype='torch.bfloat16')
        for name, value in (('sample_R', sample_r), ('sample_X', sample_x), ('input_XYZ_delta', dx), ('velocity_XYZ_delta', dv),
                            ('clean_R_FP32_source_conversion', clean_r), ('clean_X_FP32_source_conversion', clean_x),
                            ('clean_XYZ_delta_Float64_algebra', algebra), ('clean_XYZ_delta_FP32_source_conversion', fp32_difference)):
            arrays[key + '_' + name] = value.numpy()
        if case['altered'] is not None:
            patched = variants[case['altered']]['masked_velocity']
            patched_dv = patched.double()[:, :3] - v_r.double()[:, :3]
            row['selected128_same_input_metrics'] = metrics(dx, patched_dv, sigma, torch)
            arrays[key + '_S128_velocity_XYZ_delta'] = patched_dv.numpy()
            arrays[key + '_S128_clean_XYZ_delta_Float64_algebra'] = (dx - sigma * patched_dv).numpy()
        if case['native'] == 'PR':
            whole_dv = variants['swap_all12288']['masked_velocity'].double()[:, :3] - v_r.double()[:, :3]
            row['whole12288_same_input_metrics'] = metrics(dx, whole_dv, sigma, torch)
            arrays[key + '_whole12288_velocity_XYZ_delta'] = whole_dv.numpy()
            arrays[key + '_whole12288_clean_XYZ_delta_Float64_algebra'] = (dx - sigma * whole_dv).numpy()
        rows.append(row)
    for name, value in inputs.items():
        arrays[name + '_actual_BF16_input_as_FP32'] = value.numpy()
    sources = dict(script_sha256=sha(Path(__file__)), frozen_full_array_auditor_sha256=PLOT_SHA,
        actual_scheduler=dict(path=str(SCHEDULER), sha256=SCHEDULER_SHA), actual_pipeline=dict(path=str(PIPELINE), sha256=PIPELINE_SHA),
        extracted_actual_mask_method=mask_source, previous_experiment_audit_sources=previous_sources,
        feedback_json_sha256={name: sha(feedback / (name + '.json')) for name in ('complete', 'summary', 'provenance')},
        input_contract_sha256=sha(folder / 'input-contract.json'), saved_records_current_sha256=record_sources,
        original_R_states_sha256=sha(original_r), previously_sealed_P_states_sha256=sha(pulse_source), replay_files=masking)
    return rows, arrays, dict(sigma=sigma, sigma_inverse=1 / sigma, timestep_label=int(r['timesteps'][STEP]), denoising_forward_zero_based=STEP,
        schedule_bit_exact=True, scheduler_config=dict(scheduler.config), guidance_scale=1., raw_action_dim=10,
        zero_action_condition_mask=True, actual_FP32_samples_cast_to_BF16_RR_PP_exact=True,
        actual_native_R_P_entire_saved_outputs_exact=True, PR_other_kwargs_all_RR_exact=True,
        generic=generic_audits, previous_units_preregistered_verdict=previous_values['predictions']['verdict']), sources


def render(path, rows, sigma):
    import numpy as np
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 6.7), sharey=True)
    labels = [row['label'] for row in rows]
    positions = np.arange(len(rows))
    parallel = [100 * row['metrics']['clean_difference_parallel_fraction'] for row in rows]
    magnitude = [100 * row['metrics']['clean_difference_norm_fraction'] for row in rows]
    axes[0].barh(positions, parallel, color=['#777777', '#D55E00', '#D55E00', '#0072B2', '#0072B2', '#0072B2'], height=.6)
    axes[0].axvline(0, color='#555555', linewidth=1)
    axes[0].set_title('A  Remaining difference along the input change')
    axes[0].set_xlabel('100 × dot(Δclean, Δinput) / ||Δinput||² (%)')
    axes[0].set_yticks(positions, labels, fontsize=10)
    for position, value in zip(positions, parallel):
        axes[0].annotate(f'{value:+.2f}%', (value, position), xytext=(-5 if value < 0 else 5, 0), textcoords='offset points',
                         ha='right' if value < 0 else 'left', va='center', fontsize=9)
    axes[1].scatter(magnitude, positions, color='#333333', label='Existing unpatched output', s=48, zorder=3)
    for position, row, value in zip(positions, rows, magnitude):
        axes[1].annotate(f'{value:.2f}%', (value, position), xytext=(5, -10), textcoords='offset points', fontsize=9)
        if 'selected128_same_input_metrics' in row:
            changed = 100 * row['selected128_same_input_metrics']['clean_difference_norm_fraction']
            axes[1].plot([value, changed], [position, position], color='#bbbbbb', linewidth=1.2)
            axes[1].scatter(changed, position, color='#009E73', marker='s', s=38, label='Same input + saved S128 swap' if position == 1 else None)
        if 'whole12288_same_input_metrics' in row:
            changed = 100 * row['whole12288_same_input_metrics']['clean_difference_norm_fraction']
            axes[1].scatter(changed, position, color='#CC79A7', marker='D', s=34, label='Same input + saved whole swap' if position == 1 else None)
    axes[1].set_title('B  Total difference still remains')
    axes[1].set_xlabel('100 × ||Δclean|| / ||Δinput|| (%)')
    axes[1].set_xlim(left=0)
    axes[1].legend(loc='lower right', fontsize=8)
    for axis in axes:
        axis.grid(axis='x', alpha=.2)
        axis.set_axisbelow(True)
    axes[0].invert_yaxis()
    left, right = axes[0].get_xlim()
    axes[0].set_xlim(left - 3, right + 2)
    figure.suptitle(f'Existing t21 Flow response: clean estimate = input − σ × raw velocity\nσ = {sigma:.9f}; same-clean reference is velocity change = input change / σ', fontsize=13)
    figure.subplots_adjust(left=.25, right=.98, top=.83, bottom=.24, wspace=.42)
    figure.text(.04, .135, 'A near-zero axis remainder does not mean all differences disappear: panel B includes the off-axis remainder.', fontsize=10)
    figure.text(.04, .085, 'FP32 solver samples and BF16 input conventions have different denominators. Generic inputs have no saved solver trajectory.', fontsize=9)
    figure.text(.04, .035, 'Post hoc CPU algebra, no new model or physics experiment. These are normalized action-latent differences, not probabilities or physical velocity.', fontsize=9)
    figure.savefig(path, dpi=175)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Completed baseline with failed phase1 units and completed generic continuation')
    baseline = parser.parse_args().output.resolve()
    destination = baseline / 'flow-response-analysis'
    assert not destination.exists(), 'Refuse to overwrite an existing flow analysis directory'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    assert not torch.cuda.is_initialized()
    rows, arrays, contract, sources = audit(baseline, torch)
    assert all(row['metrics']['gap_meaningful'] for row in rows), 'Cannot draw ratios below the stated input gap floor'
    assert not torch.cuda.is_initialized()
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    destination.mkdir(exist_ok=False)
    archive = destination / 'actual-flow-arrays.npz'
    np.savez_compressed(archive, **arrays)
    with np.load(archive, allow_pickle=False) as saved:
        assert set(saved.files) == set(arrays)
        for name, value in arrays.items():
            assert saved[name].dtype == value.dtype and saved[name].shape == value.shape
            assert saved[name].tobytes(order='C') == value.tobytes(order='C')
    figure = destination / 'milk-flow-response.png'
    render(figure, rows, contract['sigma'])
    value = dict(state='complete', schema_version=1, interpretation=SCOPE, source=sources, contract=contract, cases=rows,
        model_calls=0, physics_calls=0, scheduler_step_calls=0, CUDA_initialized=False,
        array_archive=dict(path=str(archive), sha256=sha(archive), keys={name: dict(shape=list(array.shape), dtype=str(array.dtype)) for name, array in arrays.items()}),
        figure=dict(path=str(figure), sha256=sha(figure)),
        original_phase1_state='failed', combined_existing_model_forwards=24, combined_forward_scope='18 original surviving forwards + 6 continuation; no original complete24 or old parameter-after evidence.',
        clean_conversion_scope='CPU application of actual source convert_model_output; no claim that the clean estimate was saved by the original run or equals action_states22.',
        figure_score_scope='Each row uses its own XYZ input difference; percentage is a geometric ratio, not a fraction of mechanism explained.')
    result = destination / 'analysis.json'
    with result.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
    with (destination / 'complete.json').open('x') as stream:
        stream.write(json.dumps(dict(state='complete', analysis_sha256=sha(result), script_sha256=sha(Path(__file__)),
            actual_array_sha256=sha(archive), figure_sha256=sha(figure), model_calls=0, CUDA_initialized=False), indent=2) + '\n')
    assert not torch.cuda.is_initialized()
    print('[FLOW-RESPONSE] ' + json.dumps(dict(state='complete', output=str(destination), sigma=contract['sigma'], model_calls=0)), flush=True)


if __name__ == '__main__':
    main()
