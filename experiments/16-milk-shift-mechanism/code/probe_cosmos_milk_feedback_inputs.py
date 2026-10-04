"""Fixed-sigma t21 input interchange after one L17 attention pulse at t20.

R is x15, D is paired x06. P is R with the natural D component increment at
zero-based layer 17, attention, denoising forward 20 only. At forward 21 test
RR, PR, RP, PP action/vision inputs, preserving every other actual R input.

Falsifiable predictions: if changed future-video inputs carry the immediate
negative XYZ velocity response, RP or the conditional visual effect PP-PR
should produce it; if the action sample carries it, PR or PP-RP should.
A joint-only response indicates interaction. These are conditional latent
input effects, not proof of an incorrect video, attractor or grasp root cause.
The script owns only an absent feedback-inputs child of --output.
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
SEEDS = (195, 196, 198)
PULSE_LAYER, PULSE_STEP, REPLAY_STEP = 17, 20, 21
CAPTURE_STEPS = (20, 21)
PULSE_RUN = 'layer17_attention_step20'
PROMPT = 'pick up the milk and place it in the basket'
SCOPE = ('Fixed actual t21 inputs, same sigma and unchanged current visual condition. '
         'Conditional action/future-vision latent effects on raw Flow velocity only. '
         'No scheduler update or MuJoCo execution in the four-cell replay; '
         'no attractor, incorrect-video, object-identity or full root-cause claim.')


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def prediction_with_inputs(runtime, hooks, scan, image, seed, destination, label,
                           recipient_cache=None, donor_cache=None, rows=None):
    model, torch = runtime.pipe.transformer, runtime.torch
    captured, captured_outputs, calls = {}, {}, []

    def observe(module, args, kwargs):
        assert not args
        step = len(calls)
        calls.append(step)
        if step in CAPTURE_STEPS:
            captured[step] = runtime.cpu(kwargs)

    def observe_output(module, args, output):
        step = len(calls) - 1
        if step in CAPTURE_STEPS:
            captured_outputs[step] = runtime.cpu(output)

    handles = [model.register_forward_pre_hook(observe, with_kwargs=True),
               model.register_forward_hook(observe_output)]
    patch_report = None
    try:
        if donor_cache is not None:
            hooks.begin_patch(recipient_cache, layer=PULSE_LAYER, component='attention',
                              steps=[PULSE_STEP], gain=1, donor_cache=donor_cache)
        try:
            record, metadata = runtime.predict('milk', image, seed, destination)
        finally:
            if donor_cache is not None:
                patch_report = hooks.end()
    finally:
        for handle in handles:
            handle.remove()
        if hooks._kind is not None:
            hooks.reset()
    assert len(calls) == 30 and set(captured) == set(captured_outputs) == set(CAPTURE_STEPS), label
    torch.save(captured, destination / 'model_inputs_20_21.pt')
    torch.save(captured_outputs, destination / 'model_outputs_20_21.pt')
    if patch_report is not None:
        torch.save(patch_report, destination / 'patch_report.pt')
        write_json(destination / 'patch_report.json', scan.report_metadata(patch_report, torch))
        scan.validate_patch_report(patch_report, recipient_cache, donor_cache, rows,
                                   PULSE_LAYER, 'attention', PULSE_STEP, runtime)
    metadata.update(observational_input_capture_steps=list(CAPTURE_STEPS), role=label)
    write_json(destination / 'metadata.json', metadata)
    return record, captured, captured_outputs, patch_report


def verify_future_vision(r, p, model, runtime, scan):
    """Reject unfamiliar layouts instead of guessing a temporal tensor axis."""
    torch = runtime.torch
    pack_source = inspect.getsource(type(model)._patchify_and_pack_latents)
    time_source = inspect.getsource(type(model)._apply_timestep_embeds_to_noisy_tokens)
    compact = ''.join(pack_source.split())
    assert 'latent=latent.squeeze(0)' in compact, 'Unverified vision batch/channel layout'
    assert '_,t_actual,h_actual,w_actual=latent.shape' in compact, 'Unverified C,T,H,W layout'
    assert 'cthpwq->thwpqc' in pack_source, 'Unverified time-first packed token order'
    assert 'noisy_indexes_i * spatial_numel_i' in time_source, 'Unverified noisy-frame token indexing'
    runtime.require_exact(set(r), set(p), 'actual_t21_kwargs_keys')
    for key in r:
        if key not in ('action_tokens', 'vision_tokens'):
            runtime.require_exact(r[key], p[key], f'actual_t21_unchanged/{key}')
    assert len(r['vision_tokens']) == len(p['vision_tokens']) == 1
    rv, pv = r['vision_tokens'][0], p['vision_tokens'][0]
    assert rv.shape == pv.shape and rv.dtype == pv.dtype
    assert torch.isfinite(rv).all() and torch.isfinite(pv).all()
    rr, pp = rv.squeeze(0), pv.squeeze(0)
    assert rr.ndim == pp.ndim == 4, 'Canonical source must receive a C,T,H,W latent after squeeze(0)'
    channels, frames, height, width = rr.shape
    assert channels == model.config.latent_channel
    patch_size = int(model.config.latent_patch_size)
    grid = (frames, (height + patch_size - 1) // patch_size, (width + patch_size - 1) // patch_size)
    assert tuple(r['vision_token_shapes'][0]) == grid, 'Runtime latent and metadata grid disagree'
    noisy = r['vision_noisy_frame_indexes'][0].cpu().flatten()
    assert noisy.dtype == torch.long and 0 < noisy.numel() < frames
    assert noisy.unique().numel() == noisy.numel() and bool(((noisy > 0) & (noisy < frames)).all())
    conditioned = sorted(set(range(frames)) - set(noisy.tolist()))
    assert conditioned == [0], 'Require exactly the current frame 0 conditioned and all future frames noisy'
    runtime.require_exact(rr[:, conditioned], pp[:, conditioned], 't21_current_condition_latent_exact')
    # Exercise the actual pure packing method on CPU, independently checking
    # the metadata-to-token mapping used to identify current/future frames.
    packed_r, shapes_r = model._patchify_and_pack_latents(r['vision_tokens'])
    packed_p, shapes_p = model._patchify_and_pack_latents(p['vision_tokens'])
    assert shapes_r == shapes_p == [(frames, height, width)]
    spatial = grid[1] * grid[2]
    assert packed_r.shape[0] == packed_p.shape[0] == frames * spatial
    assert r['vision_sequence_indexes'].numel() == frames * spatial
    clean_rows = torch.arange(spatial)
    runtime.require_exact(packed_r[clean_rows], packed_p[clean_rows], 't21_packed_current_condition_exact')
    return dict(runtime_vision_shape=list(rv.shape), verified_squeezed_order='C,T,H,W',
        runtime_patch_grid=list(grid), current_condition_frames=conditioned, future_noisy_frames=noisy.tolist(),
        current_condition_bit_exact=True, packed_current_condition_bit_exact=True,
        future_latents_bit_exact=bool(torch.equal(rr[:, noisy], pp[:, noisy])),
        future_latent_delta_l2=float((pp[:, noisy].float() - rr[:, noisy].float()).norm()),
        action_sample_delta_l2=float((p['action_tokens'][0].float() - r['action_tokens'][0].float()).norm()),
        packing_method_source_sha256=hashlib.sha256(pack_source.encode()).hexdigest(),
        timestep_method_source_sha256=hashlib.sha256(time_source.encode()).hexdigest())


def direct_replay(runtime, hooks, scan, cpu_kwargs, label, destination, expected=None, expected_output=None):
    """Capture real boundaries, component increments and head in one forward."""
    torch, model = runtime.torch, runtime.pipe.transformer
    assert not model.training and not torch.is_grad_enabled() and not model.is_cache_enabled
    device = next(model.parameters()).device
    kwargs = to_device(cpu_kwargs, torch, device)
    rows = kwargs['action_mse_loss_indexes'].flatten() - int(kwargs['und_len'])
    assert rows.numel() == rows.unique().numel() == 16
    boundaries, readouts, velocities, handles = [], [], [], []

    def before_first(module, args):
        boundaries.append(runtime.cpu(args[1].index_select(0, rows)))

    def after_layer(module, args, output):
        boundaries.append(runtime.cpu(output[1].index_select(0, rows)))

    def before_head(module, args):
        readouts.append(runtime.cpu(args[0]))

    def after_head(module, args, output):
        velocities.append(runtime.cpu(output))

    destination.mkdir(exist_ok=False)
    hooks.begin_capture(label, expected_steps=1)
    try:
        handles.append(model.layers[0].register_forward_pre_hook(before_first))
        handles.extend(block.register_forward_hook(after_layer) for block in model.layers)
        handles.append(model.action_proj_out.register_forward_pre_hook(before_head))
        handles.append(model.action_proj_out.register_forward_hook(after_head))
        model_output = runtime.cpu(model(**kwargs))
    finally:
        for handle in handles:
            handle.remove()
        component_cache = hooks.end()
    assert component_cache['complete'] is True and component_cache['n_steps'] == 1
    assert component_cache['captured_component_count'] == component_cache['component_call_count'] == 72
    assert len(boundaries) == 37 and len(readouts) == len(velocities) == 1
    hidden, readout, velocity = torch.stack(boundaries), readouts[0], velocities[0]
    assert hidden.shape == (37, 16, 4096) and readout.shape == (16, 4096) and velocity.shape == (16, 64)
    assert all(torch.isfinite(value).all() for value in (hidden, readout, velocity))
    runtime.require_exact(component_cache['first_model_kwargs'], cpu_kwargs, label + '/actual_replay_inputs')
    if expected is not None:
        runtime.require_exact(readout, expected['readout'][REPLAY_STEP], label + '/actual_q0_readout_bit_exact')
        runtime.require_exact(velocity, expected['action_velocity'][REPLAY_STEP], label + '/actual_q0_velocity_bit_exact')
        assert expected_output is not None
        runtime.require_exact(model_output, expected_output, label + '/actual_q0_entire_model_output_bit_exact')
    torch.save(component_cache, destination / 'components.pt')
    torch.save(model_output, destination / 'model_output.pt')
    import numpy as np
    np.savez_compressed(destination / 'arrays.npz', action_boundary_hidden=hidden.float().numpy(),
                        readout=readout.float().numpy(), action_velocity=velocity.float().numpy())
    write_json(destination / 'metadata.json', dict(label=label, original_denoising_forward=REPLAY_STEP,
        timestep_values=cpu_kwargs['action_timesteps'].cpu().tolist(), actual_boundaries=37,
        layer_convention='Boundary 0 before L0; boundaries 1..36 after zero-based layers 0..35',
        original_q0_readout_and_velocity_exact=expected is not None,
        original_q0_entire_model_output_exact=expected is not None,
        component_cache_complete=True, component_forward_count=1, component_calls=72,
        action_rows=rows.cpu().tolist(), all_observation_hooks_removed=True,
        arrays_sha256=sha(destination / 'arrays.npz'), components_sha256=sha(destination / 'components.pt'),
        model_output_sha256=sha(destination / 'model_output.pt'),
        readout=scan.tensor_meta(readout, torch), velocity=scan.tensor_meta(velocity, torch), scope=SCOPE))
    return dict(hidden=hidden, readout=readout, velocity=velocity, components=component_cache)


def summarize_cells(runtime, scan, recipient, donor, cells, folder):
    torch = runtime.torch
    v = {key: value['velocity'].double() for key, value in cells.items()}
    effects = dict(action_only=v['PR'] - v['RR'], future_vision_only=v['RP'] - v['RR'],
                   interaction=v['PP'] - v['PR'] - v['RP'] + v['RR'], total_pulse=v['PP'] - v['RR'],
                   future_vision_given_P_action=v['PP'] - v['PR'],
                   action_given_P_future_vision=v['PP'] - v['RP'])
    error = effects['total_pulse'] - effects['action_only'] - effects['future_vision_only'] - effects['interaction']
    assert float(error.abs().max()) < 1e-12, '2x2 arithmetic decomposition failed'
    r, d = recipient['action_velocity'][REPLAY_STEP, :, :3], donor['action_velocity'][REPLAY_STEP, :, :3]
    result = dict(cells={key: dict(velocity_xyz=scan.direction_metrics(r, d, value['velocity'][:, :3]),
                                  velocity_all64_delta_l2=float((value['velocity'].double() - v['RR']).norm()))
                        for key, value in cells.items()},
        effects={key: dict(velocity_xyz=scan.direction_metrics(r, d, r.double() + effect[:, :3]),
                          raw_all64_delta_l2=float(effect.norm())) for key, effect in effects.items()},
        effect_formulas={'action_only': 'PR-RR', 'future_vision_only': 'RP-RR',
                        'interaction': 'PP-PR-RP+RR', 'total_pulse': 'PP-RR',
                        'future_vision_given_P_action': 'PP-PR', 'action_given_P_future_vision': 'PP-RP'},
        decomposition_max_abs_error=float(error.abs().max()),
        axis='Same t21 paired D-R actual raw Flow velocity XYZ across all 16 predicted action rows',
        limitations=['Only the action/future-video samples in a fixed t21 context are interchanged.',
                     'Joint interaction cannot be uniquely assigned to action or vision.',
                     'No scheduler or visual feedback runs between replay cells; scores are not grasp probabilities.'],
        scope=SCOPE)
    import numpy as np
    arrays = {f'{key}_velocity': value['velocity'].float().numpy() for key, value in cells.items()}
    arrays.update({f'{key}_effect': value.numpy() for key, value in effects.items()})
    for key, value in cells.items():
        arrays[f'{key}_boundary_delta_l2_to_RR'] = (value['hidden'].float() - cells['RR']['hidden'].float()).flatten(1).norm(dim=1).numpy()
    np.savez_compressed(folder / 'factorial-arrays.npz', **arrays,
                        recipient_velocity=r.float().numpy(), donor_velocity=d.float().numpy())
    write_json(folder / 'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Completed nine-trial baseline root')
    baseline = parser.parse_args().output.resolve()
    scan_path = ROOT / 'work/probe_cosmos_milk_component_pulses.py'
    scan = load_file('feedback_existing_pulse_contract', scan_path)
    scan.validate_baseline(baseline)
    coarse = baseline / 'component-pulses'
    coarse_complete, coarse_results = read_json(coarse / 'complete.json'), read_json(coarse / 'results.json')
    assert not (coarse / 'failed.json').exists()
    assert coarse_complete['state'] == 'complete' and coarse_complete['scan_predictions'] == 56
    assert coarse_complete['control_predictions'] == 4
    assert coarse_complete['all_noise_schedule_pack_checks_exact'] is True
    assert coarse_complete['all_repeat_and_selfpatch_fields_bit_exact'] is True
    assert coarse_results['state'] == 'complete' and len(coarse_results['cases']) == 56
    assert coarse_complete['paired_seed'] == coarse_results['paired_seed'] == 198
    assert coarse_complete['results_sha256'] == sha(coarse / 'results.json')
    assert coarse_complete['script_sha256'] == sha(scan_path)
    coarse_case = next(row for row in coarse_results['cases'] if row['run'] == PULSE_RUN)
    coarse_state = coarse / 'runs' / PULSE_RUN / 'states.pt'
    assert coarse_case['states_sha256'] == sha(coarse_state)
    out = baseline / 'feedback-inputs'
    out.mkdir(exist_ok=False)
    started, completed, summaries = time.perf_counter(), [], []
    hooks = None

    def progress(stage, **fields):
        value = dict(stage=stage, completed=completed, elapsed_s=time.perf_counter() - started, **fields)
        write_json(out / 'progress.json', value)
        print('[FEEDBACK-INPUTS] ' + json.dumps(value, allow_nan=False), flush=True)

    try:
        os.environ['HF_HUB_OFFLINE'] = '1'
        paths = dict(normal_runtime=ROOT / 'work/serve_cosmos_goal_pair.py',
                     factory=ROOT / 'work/probe_cosmos_language_routes.py',
                     components=ROOT / 'work/cosmos_component_interventions.py')
        provenance = read_json(baseline / 'server/provenance.json')
        for key in paths:
            assert sha(paths[key]) == provenance[key + '_sha256'], f'Baseline {key} source changed'
        base = load_file('feedback_normal_runtime', paths['normal_runtime'])
        base.TASKS['milk'] = (7, 'pick_up_the_milk_and_place_it_in_the_basket', PROMPT, 'milk_1')
        factory = load_file('feedback_checked_factory', paths['factory'])
        factory.OUT = out
        assert sha(factory.SCHEDULER) == provenance['scheduler_sha256']
        progress('loading_model')
        runtime = base.NormalRuntime(factory)
        torch, model = runtime.torch, runtime.pipe.transformer
        assert not model.training and not torch.is_grad_enabled() and not model.is_cache_enabled
        module = load_file('feedback_component_observers', paths['components'])
        hooks = module.CosmosComponentInterventions(model)
        transformer_source = Path(inspect.getfile(type(model)))
        assert sha(transformer_source) == read_json(coarse / 'provenance.json')['actual_transformer_sha256']
        write_json(out / 'provenance.json', dict(script_sha256=sha(Path(__file__)),
            source_sha256={key: sha(path) for key, path in paths.items()}, pulse_contract_sha256=sha(scan_path),
            actual_transformer_source=str(transformer_source), actual_transformer_sha256=sha(transformer_source),
            actual_pipeline_source_sha256=sha(Path(inspect.getfile(type(runtime.pipe)))),
            scheduler_sha256=sha(factory.SCHEDULER), baseline_summary_sha256=sha(baseline / 'summary.json'),
            coarse_results_sha256=sha(coarse / 'results.json'), seeds=list(SEEDS), recipient='x15', donor='x06',
            pulse_layer_zero_based=PULSE_LAYER, pulse_step=PULSE_STEP, fixed_replay_step=REPLAY_STEP,
            pulse_component='attention', pulse_gain=1, library_edits=False, scope=SCOPE))
        for seed in SEEDS:
            folder = out / f'seed{seed}'
            folder.mkdir(exist_ok=False)
            records, caches, sources = {}, {}, {}
            for role, scene in (('R', 'x15'), ('D', 'x06')):
                run = baseline / 'closed-loop' / f'{scene}_seed{seed}'
                records[role] = torch.load(run / 'chunk_00/states.pt', map_location='cpu', weights_only=True, mmap=True)
                caches[role] = torch.load(run / 'chunk_00/components.pt', map_location='cpu', weights_only=True, mmap=True)
                scan.validate_record(records[role], torch)
                rows = scan.validate_cache(caches[role], records[role], hooks, runtime, role)
                metadata = read_json(run / 'chunk_00/metadata.json')
                assert metadata['trial'] == f'{scene}_seed{seed}'
                assert metadata['seed'] == seed and metadata['query'] == 0 and metadata['prompt'] == PROMPT
                assert metadata['input_png_sha256'] == sha(run / 'input_00.png')
                sources[role] = dict(trial=f'{scene}_seed{seed}', image=str(run / 'input_00.png'),
                    image_sha256=sha(run / 'input_00.png'), states_sha256=sha(run / 'chunk_00/states.pt'),
                    components_sha256=sha(run / 'chunk_00/components.pt'))
            r, d = records['R'], records['D']
            for key in ('pure_noise', 'timesteps', 'sigmas'):
                runtime.require_exact(r[key], d[key], f'seed{seed}/paired_{key}')
            runtime.require_exact(caches['R']['steps'], caches['D']['steps'], f'seed{seed}/paired_30step_metadata')
            runtime.require_exact(r['model_input']['action_tokens'], d['model_input']['action_tokens'], f'seed{seed}/paired_initial_action')
            progress('native_R_query', seed=seed)
            repeated, r_inputs, r_outputs, _ = prediction_with_inputs(runtime, hooks, scan, Path(sources['R']['image']),
                                                         seed, folder / 'native_R_q0', 'R')
            runtime.require_exact(repeated, r, f'seed{seed}/observer_native_R_full_record')
            progress('pulse_P_query', seed=seed)
            p, p_inputs, p_outputs, report = prediction_with_inputs(runtime, hooks, scan, Path(sources['R']['image']), seed,
                folder / 'pulse_P_q0', 'P', caches['R'], caches['D'], rows)
            scan.validate_run_contract(p, r, runtime, f'seed{seed}/pulse')
            for key in ('readout', 'action_velocity'):
                runtime.require_exact(p[key][:PULSE_STEP], r[key][:PULSE_STEP], f'seed{seed}/before_pulse/{key}')
            runtime.require_exact(p['action_states'][:PULSE_STEP + 1], r['action_states'][:PULSE_STEP + 1], f'seed{seed}/before_pulse_solver')
            runtime.require_exact(r_inputs[PULSE_STEP], p_inputs[PULSE_STEP], f'seed{seed}/actual_pulse_input_exact')
            coarse_exact = seed == 198
            if coarse_exact:
                reference_p = torch.load(coarse_state, map_location='cpu', weights_only=True, mmap=True)
                runtime.require_exact(p, reference_p, 'seed198/observer_P_full_coarse_record_exact')
            vision_contract = verify_future_vision(r_inputs[REPLAY_STEP], p_inputs[REPLAY_STEP], model, runtime, scan)
            write_json(folder / 'input-contract.json', dict(seed=seed, sources=sources,
                both_pure_noise_draws_exact=True, complete_schedule_and_30step_metadata_exact=True,
                native_R_all_record_fields_exact=True, seed198_P_all_coarse_record_fields_exact=coarse_exact,
                other_t21_kwargs_all_exact=True, vision=vision_contract, scope=SCOPE))
            replay_folder = folder / 'replays'
            replay_folder.mkdir(exist_ok=False)
            cells = {}
            # Establish both natural fixed-input replay controls before hybrids.
            for label in ('RR', 'PP', 'PR', 'RP'):
                mixed = dict(r_inputs[REPLAY_STEP])
                mixed['action_tokens'] = (p_inputs if label[0] == 'P' else r_inputs)[REPLAY_STEP]['action_tokens']
                mixed['vision_tokens'] = (p_inputs if label[1] == 'P' else r_inputs)[REPLAY_STEP]['vision_tokens']
                if label in ('RR', 'PP'):
                    runtime.require_exact(mixed, (r_inputs if label == 'RR' else p_inputs)[REPLAY_STEP], f'{label}/actual_native_input')
                progress('direct_replay', seed=seed, cell=label)
                cells[label] = direct_replay(runtime, hooks, scan, mixed, f'seed{seed}_{label}',
                    replay_folder / label, expected=r if label == 'RR' else p if label == 'PP' else None,
                    expected_output=r_outputs[REPLAY_STEP] if label == 'RR' else p_outputs[REPLAY_STEP] if label == 'PP' else None)
                if label == 'RR':
                    for layer in range(36):
                        for component in ('attention', 'mlp'):
                            runtime.require_exact(cells[label]['components']['frames'][(0, layer, component)],
                                caches['R']['frames'][(REPLAY_STEP, layer, component)],
                                f'seed{seed}/native_replay_component_exact/{layer}/{component}')
                completed.append(f'seed{seed}_{label}')
            result = summarize_cells(runtime, scan, r, d, cells, folder)
            result.update(seed=seed, timestep=int(r['timesteps'][REPLAY_STEP]), sigma=float(r['sigmas'][REPLAY_STEP]),
                pulse_t20_velocity_xyz=scan.direction_metrics(r['action_velocity'][PULSE_STEP, :, :3],
                    d['action_velocity'][PULSE_STEP, :, :3], p['action_velocity'][PULSE_STEP, :, :3]),
                final_P_normalized_action_xyz=scan.direction_metrics(r['actions'][:, :3], d['actions'][:, :3], p['actions'][:, :3]))
            write_json(folder / 'summary.json', result)
            summaries.append(result)
            progress('seed_complete', seed=seed,
                     t21_total_projection=result['effects']['total_pulse']['velocity_xyz']['signed_projection_fraction'])
        assert len(summaries) == 3 and len(completed) == 12
        write_json(out / 'summary.json', dict(seeds=summaries,
            predictions={'future_video_carrier': 'RP-RR or PP-PR carries negative t21 D-R XYZ velocity projection.',
                         'action_sample_carrier': 'PR-RR or PP-RP carries negative t21 D-R XYZ velocity projection.',
                         'interaction': 'A joint-only negative response must not be assigned uniquely to one latent.'},
            current_condition_all_seeds_exact=True, all_native_R_and_P_t21_replays_bit_exact=True,
            entire_t21_model_output_including_vision_exact=True,
            interpretation='Interchange tests identify conditional latent contributions at t21; no downstream closed-loop root-cause claim.', scope=SCOPE))
        write_json(out / 'complete.json', dict(state='complete', seeds=list(SEEDS), q0_predictions=6,
            native_replays_exact=6, direct_model_forwards=12, boundaries_per_forward=37,
            components_per_forward=72, summary_sha256=sha(out / 'summary.json'),
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
