"""Bounded CPU audit of eight completed instruction-suffix/MLP experiments.

Read only the new stage seals, 240 actual sites, 64 prediction records and
eight physical trajectories. Previously completed natural references are
bound through their completion seals and the new execution controls; their
entire upstream archive is not recursively audited again. No model, dispatch,
solver, environment, physics or CUDA initialization. Create an absent stage.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
import traceback

ROOT = Path('/home/current/work/cosmos3')
PROBE_SHA = '745c4d8864bc94751791cf5aa81cb74afc0a678ca95dfad998b0381431c61429'
EXECUTION_SHA = '1e9e5ab03df34d333ee0905529af2de53c3a872da6139fa65cefa0aac2b55876'
LAUNCHER_SHA = '617a11698c9e009ec354f8a3580ed9d17808b518739a602d858e4d3f4f5eb40e'
PRESERVED_FAILURE_SHA = '4af7ac7aa6c9d80a9e1c9061073b3c5f4e94ad6cdd9cf9db67f6d57b062e3c2e'
PRODUCER_COMPLETE_SHA = '05f2a956cc33a391897143255adddac65019e186a421dcd880d7562f4352d646'
PHYSICS_COMPLETE_SHA = '3e88b11dd6dac3199aa0d84cfb8d7d964ec2e6a13b84cb01f2dd76f409cf0388'
CPU_SHA = 'ca88ff9c7ee62beef9851fdfaea1c9fddd6e33c0f99058ebee16f26a28a4af9c'
OLD_SEALS = {
    'matched-names': '10c76eceaee3d8baeade3396fc65f55b71a3c286fdaef7cfe464aa0b1a3ac5b4',
    'future-matched-names-execution': '0468105a3110ac9d9bf83071d58963029f2bf1b9b78e35f0cab3541edb293e20',
    'matched-names-analysis': '219c43cba7e78a3da774d481cc7843e2b6a519c166cedd7490e21181f5e2b4a0'}
STAGE = 'instruction-interaction-analysis'
FLOOR = 1e-12
SCOPE = ('Only X15/V198 two eligible matched-name recipients, L35 action16, q0 all30 steps. '
    'Each MLP response uses the same live input. C block candidates are local counterfactuals, '
    'not J trajectory values. Cross-arm later differences contain dynamic input changes. '
    'Cosine/cancellation describe arithmetic, not identity suppression or unique mediation. '
    'Random strength matches actual BF16 pre-W overall L2 only; post-W inequality is retained. '
    'No original milk instruction at15cm, failed V195, cross-position or withdrawal repair claim. '
    'Post-W4096 coordinates are not attention heads; only pre-W32x128 has the real head axis.')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clean(value, np):
    if isinstance(value, dict):
        return {key: clean(item, np) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item, np) for item in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist(), np)
    if isinstance(value, np.generic):
        return clean(value.item(), np)
    return None if isinstance(value, float) and not np.isfinite(value) else value


def seal(base):
    paths = {'producer': ROOT / 'work/probe_cosmos_milk_instruction_interaction.py',
             'execution': ROOT / 'work/run_cosmos_milk_instruction_interaction.py',
             'cpu': ROOT / 'work/analyze_cosmos_milk_future_current_layer_groups.py'}
    assert [sha(paths[k]) for k in paths] == [PROBE_SHA, EXECUTION_SHA, CPU_SHA]
    assert sha(ROOT / 'work/run_cosmos_milk_instruction_interaction_egl.py') == LAUNCHER_SHA
    assert len(PHYSICS_COMPLETE_SHA) == 64, 'Physics completion is not frozen'
    probe, cpu = load('interaction_analysis_probe', paths['producer']), load('interaction_analysis_cpu', paths['cpu'])
    producer, execution = base / probe.STAGE, base / 'future-instruction-interaction-execution-egl'
    assert sha(producer / 'complete.json') == PRODUCER_COMPLETE_SHA
    assert sha(execution / 'complete.json') == PHYSICS_COMPLETE_SHA
    for folder, names in ((producer, ('failed.json',)), (execution, ('wrapper_failed.json', 'simulator_failed.json', 'server/failed.json', 'closed-loop/failed.json'))):
        assert all(not (folder / name).exists() for name in names)
    pc, ec = read(producer / 'complete.json'), read(execution / 'complete.json')
    assert pc['state'] == ec['state'] == 'complete' and pc['script_sha256'] == PROBE_SHA and ec['script_sha256'] == EXECUTION_SHA
    for name in ('results', 'provenance', 'protocol', 'sources', 'primary-prediction'):
        assert sha(producer / (name + '.json')) == pc[name.replace('-', '_') + '_sha256']
    for name, expected in ec['files_sha256'].items():
        assert sha(execution / name) == expected
    frozen, controls, comparisons = (read(execution / (name + '.json')) for name in ('provenance', 'controls', 'comparisons'))
    assert frozen['script_sha256'] == EXECUTION_SHA and frozen['producer_script_sha256'] == PROBE_SHA
    recovery = frozen['execution_recovery']
    failed = base / 'future-instruction-interaction-execution'
    assert sha(failed / 'wrapper_failed.json') == PRESERVED_FAILURE_SHA
    assert recovery == dict(launcher_sha256=LAUNCHER_SHA, execution_body_sha256=EXECUTION_SHA,
        preserved_failed_stage=str(failed), preserved_failure_sha256=PRESERVED_FAILURE_SHA,
        original_failure_before_environment_and_query=True,
        required_environment=dict(MUJOCO_GL='egl', MUJOCO_EGL_DEVICE_ID='0'), scientific_plan_changed=False, q0_recomputed=False)
    for name, expected in frozen['producer_documents_sha256'].items():
        assert sha(producer / (name + '.json')) == expected
    assert read(execution / 'prepared.json')['provenance_sha256'] == sha(execution / 'provenance.json')
    assert read(execution / 'primary-preregistered.json') == dict(state='preregistered', **probe.PRIMARY)
    assert frozen['primary_prediction'] == probe.PRIMARY
    for stage, expected in OLD_SEALS.items():
        assert sha(base / stage / 'complete.json') == expected
    rows = read(producer / 'results.json')['cases']
    assert [(r['case'], r['recipient_goal'], r['donor_goal'], r['arm']) for r in rows] == list(probe.PLAN)
    labels = [row[0] for row in probe.PLAN]
    assert ec['trial_order'] == frozen['trial_order'] == labels
    assert [r['case'] for r in comparisons['cases']] == labels and comparisons['state'] == 'all8_complete'
    assert pc['official_dispatch_counts'] == probe.expected_counts() and pc['site_raw_captures'] == 240
    assert pc['fresh_model_forwards'] == 240 and pc['total_official_dispatch_calls'] == 17520
    assert (ec['physical_trials'], ec['saved_q0_predictions'], ec['native_predictions'], ec['fresh_model_forwards'], ec['actual_physical_actions'], ec['actual_later_whole_noise_pairs']) == (8, 8, 56, 1680, 1024, 49)
    assert ec['combined_two_stage_model_forwards'] == 1920
    assert ec['two_entire129_state_5JSON_8PNG_all8_modelrecords_controller0_16_controls_exact'] is True
    server = read(execution / 'server/complete.json')
    assert (server['queries'], server['saved_q0_predictions'], server['native_predictions'], server['actual_later_whole_noise_pairs']) == (64, 8, 56, 49)
    assert server['final_model_hooks_none'] is server['two_live_self_all8_entire_records_and_PNGs_exact'] is True
    for name, expected in server['files_sha256'].items():
        assert sha(execution / 'server' / name) == expected
    assert len(controls['two_live_self_whole_controls']) == 2
    for row in rows:
        folder = producer / row['case']
        assert set(row['files_sha256']) == set(probe.FILES)
        for name, expected in row['files_sha256'].items():
            assert sha(folder / name) == expected
        manifest = read(folder / 'site-captures.json')
        assert manifest['files'] == row['site_files'] == row['boundary_files'] and manifest['captures'] == 30
    assert set(controls['actual_case_files_sha256']) == set(labels)
    for case, files in controls['actual_case_files_sha256'].items():
        for name, expected in files.items():
            assert sha(execution / 'closed-loop' / case / name) == expected
    return dict(base=base, producer=producer, execution=execution, probe=probe, cpu=cpu, rows=rows,
        pc=pc, ec=ec, frozen=frozen, controls=controls, comparisons=comparisons, paths=paths)


def physical(c, torch, np):
    cpu, frozen = c['cpu'], c['frozen']
    forward, gripper, converter_sources = cpu.converters(torch)
    stats_path = Path(frozen['threshold_contract']['frozen_files']['normalizer_stats']['path'])
    assert sha(stats_path) == frozen['threshold_contract']['frozen_files']['normalizer_stats']['sha256']
    stats = read(stats_path)['global_raw']
    lo, hi = np.asarray(stats['q01']), np.asarray(stats['q99'])
    assert lo.shape == hi.shape == (10,) and np.all(hi > lo)
    scale, offset = np.maximum(hi - lo, 1e-8) / 2, (hi + lo) / 2
    q0, later, pairs, outcomes, schedule = {}, {}, 0, [], None
    for ci, row in enumerate(c['rows']):
        label, folder = row['case'], c['execution'] / 'closed-loop' / row['case']
        trajectory, summary, strict = (read(folder / (name + '.json')) for name in ('trajectory', 'summary', 'strict-windows'))
        assert len(trajectory) == 129 and [r['step'] for r in trajectory] == list(range(129))
        initial = np.load(c['execution'] / 'inputs/x15/state.npy', allow_pickle=False)
        states = np.load(folder / 'sim_states.npy', allow_pickle=False)
        assert states.shape == (129, initial.size) and cpu.npexact(states[0], initial, np) and np.isfinite(states).all()
        found, firsts = [], []
        for name in trajectory[0]['objects']:
            xyz = np.asarray([r['objects'][name] for r in trajectory], np.float64)
            contacts = np.asarray([[r['finger_contacts'][name]['left'], r['finger_contacts'][name]['right']] for r in trajectory], bool)
            assert np.isfinite(xyz).all() and contacts.all(1).tolist() == [r['finger_contacts'][name]['both'] for r in trajectory]
            valid = contacts.all(1) & ((xyz[:, 2] - xyz[0, 2]) > .02)
            starts = [i for i in range(125) if valid[i:i + 5].all()]
            assert starts == strict['objects'][name]['five_record_window_start_steps']
            assert summary['per_object'][name]['first_2cm_for_5frames'] == (starts[0] if starts else None)
            if starts:
                found.append(name)
                firsts.append(starts[0])
        assert found == summary['selected_objects'] and summary['first_selection_step'] == (min(firsts) if firsts else None)
        for step, name in ((0, 'initial_controller.npz'), (16, 'controller-step16.npz')):
            with np.load(folder / name, allow_pickle=False) as archive:
                for key in archive.files:
                    assert cpu.npexact(archive[key], np.asarray(trajectory[step]['controller'][key], dtype=archive[key].dtype), np)
                    if step == 0:
                        with np.load(c['execution'] / 'inputs/x15/controller.npz', allow_pickle=False) as initial_controller:
                            assert cpu.npexact(archive[key], initial_controller[key], np)
        norm_parts = []
        contracts = read(folder / 'query_contract.json')
        assert len(contracts) == 8
        for query in range(8):
            chunk, rule = folder / f'chunk_{query:02d}', frozen['trials'][label]
            rec = torch.load(chunk / 'states.pt', map_location='cpu', weights_only=True, mmap=True)
            assert rec['action_states'].shape == (31, 1, 16, 64) and rec['action_velocity'].shape == (30, 16, 64)
            assert rec['readout'].shape == (30, 16, 4096) and rec['actions'].shape == (16, 10)
            assert len(rec['pure_noise']) == 2 and torch.count_nonzero(rec['action_states'][..., 10:]) == 0
            assert [tuple(v.shape) for v in rec['pure_noise']] == [(1, 48, 5, 10, 20), (16, 64)]
            assert all(v.dtype == torch.float32 and bool(torch.isfinite(v).all()) for v in rec['pure_noise'])
            assert all(bool(torch.isfinite(rec[key]).all()) for key in ('action_states', 'action_velocity', 'readout', 'actions', 'timesteps', 'sigmas'))
            if schedule is None:
                schedule = {key: rec[key] for key in ('timesteps', 'sigmas')}
            assert all(cpu.exact(rec[key], value, torch) for key, value in schedule.items())
            meta = read(chunk / 'metadata.json')
            assert meta == contracts[query]['server_metadata']
            assert contracts[query]['request'] == dict(trial=label, query=query, seed=195 + query, prompt=rule['policy_prompt'])
            assert meta['input_png_sha256'] == sha(folder / f'input_{query:02d}.png') and meta['prompt'] == c['probe'].PROMPTS[row['recipient_goal']]
            assert meta['seed'] == 195 + query and meta['saved_q0_prediction'] is (query == 0)
            if query == 0:
                own = torch.load(c['producer'] / label / 'states.pt', map_location='cpu', weights_only=True, mmap=True)
                assert cpu.exact(own, rec, torch), 'Entire cached q0 differs'
                q0[label] = rec
            elif query in later:
                assert cpu.exact(rec['pure_noise'], later[query], torch)
                pairs += 1
            else:
                later[query] = rec['pure_noise']
            norm = np.asarray(read(chunk / 'normalized_actions.json'))
            assert cpu.npexact(norm, np.asarray(rec['actions'].tolist()), np)
            norm_parts.append(norm)
        norm = np.concatenate(norm_parts)
        raw = norm * scale + offset
        command = np.asarray([gripper(forward(a, '6d').tolist(), 'zero_one') for a in raw])
        for name, value in (('normalized_actions', norm), ('denormalized_actions', raw), ('actions', command), ('executed_actions', np.clip(command, -1, 1))):
            assert cpu.npexact(np.asarray(read(folder / (name + '.json'))), value, np)
        comparison = c['comparisons']['cases'][ci]
        assert comparison['selected_objects'] == found and comparison['per_object'] == summary['per_object']
        assert comparison['strict_only_donor_selected'] is (found == [c['probe'].TARGETS[row['donor_goal']]])
        assert comparison['strict_requested_recipient_selected'] is (found == [c['probe'].TARGETS[row['recipient_goal']]])
        outcomes.append(dict(case=label, recipient=row['recipient_goal'], donor=row['donor_goal'], arm=row['arm'],
            selected_objects=found, first_selection_step=summary['first_selection_step'],
            only_donor=comparison['strict_only_donor_selected'], milk_in_basket=bool(trajectory[-1]['milk_in_basket']),
            q0_eef_projection_toward_milk_from_cheese_m=comparison['q0_eef_projection_toward_milk_from_cheese_m']))
    assert pairs == 49
    primary = [r for r in outcomes if r['case'] in c['probe'].PRIMARY['cases']]
    verdict = 'supported' if all(r['only_donor'] for r in primary) else 'rejected'
    assert len(primary) == 2 and verdict == c['ec']['primary_prediction'] == c['comparisons']['primary_prediction']
    pred = read(c['execution'] / 'primary-prediction.json')
    assert pred['prediction'] == verdict and pred['both_suffix_joint_strict_only_donor'] is (verdict == 'supported')
    return q0, outcomes, dict(records=64, conversions=1024, later_whole_noise_pairs=pairs,
        strict_trajectories=8, canonical_controls='Bound to execution seals; no recursive historical replay.', converter_sources=converter_sources)


def metric(a, m, np):
    an, mn = np.linalg.norm(a, axis=-1), np.linalg.norm(m, axis=-1)
    dot = np.sum(a * m, axis=-1)
    cos, projection, ratio = [np.full_like(an, np.nan) for _ in range(3)]
    np.divide(dot, an * mn, out=cos, where=(an > FLOOR) & (mn > FLOOR))
    np.divide(dot, an ** 2, out=projection, where=an > FLOOR)
    np.divide(np.linalg.norm(a + m, axis=-1), an, out=ratio, where=an > FLOOR)
    return dict(A_l2=an, M_l2=mn, cosine=cos, M_projection_on_A=projection, sum_to_A_ratio=ratio,
                cosine_valid=(an > FLOOR) & (mn > FLOOR), A_axis_valid=an > FLOOR)


def sites(c, records, torch, np):
    cpu, arrays, summaries, sources = c['cpu'], {}, [], []
    per_case, random_count, first_inputs = [], 0, {}
    for row in c['rows']:
        case, arm, folder = row['case'], row['arm'], c['producer'] / row['case']
        record = records[case]
        report, lc = read(folder / 'observer-report.json'), torch.load(folder / 'language-contract.pt', map_location='cpu', weights_only=True, mmap=True)
        assert report['counts'] == row['actual_dispatch_counts'] == c['probe'].expected_counts(arm)
        assert lc['language']['und_len'] == 122 and lc['language']['target_positions'] == [48, 49]
        assert cpu.exact(lc['first_model_kwargs'], record['model_input'], torch)
        clean_hash = cpu.tensor_meta(lc['actual_clean_current'], torch)['sha256']
        case_values = []
        for step, entry in enumerate(row['site_files']):
            assert entry['step'] == step and entry['layer_zero_based'] == 35 and entry['action_boundary_ids'] == [36]
            d = cpu.checked(folder / entry['file'], entry['sha256'], torch)
            assert d['step'] == step and d['arm'] == d['case_arm'] == arm and d['recipient'] == row['recipient_goal'] and d['donor'] == row['donor_goal']
            assert cpu.exact(d['actual_model_action_tokens'], record['action_states'][step, 0].to(torch.bfloat16), torch)
            assert cpu.exact(d['actual_model_action_output'], record['action_velocity'][step], torch)
            assert cpu.exact(d['actual_model_action_timesteps'], d['actual_pack']['metadata']['action_timesteps'], torch)
            assert cpu.exact(d['solver_sigma_from_same_frozen_schedule'], record['sigmas'][step], torch)
            assert d['actual_current_frame_sha256'] == clean_hash and len(d['hidden_boundary_metadata']) == 37
            endpoint = {}
            for key in ('x', 'W0', 'W1', 'm0', 'm1', 'Z0', 'Zdonor', 'Zselected'):
                value = d[key + '_explicit_action16']
                assert value.dtype == torch.bfloat16 and value.shape == ((1, 16, 32, 128) if key.startswith('Z') else (16, 4096))
                assert bool(torch.isfinite(value).all())
                meta = cpu.tensor_meta(value, torch)
                assert all(meta[k] == d['actual_endpoint_metadata'][key + '_explicit_action16'][k] for k in meta)
                endpoint[key] = value
            if step == 0 and arm in ('suffix_joint', 'suffix_joint_native_mlp'):
                first_inputs[(row['recipient_goal'], arm)] = endpoint
            x, w0, w1, m0, m1 = (endpoint[k] for k in ('x', 'W0', 'W1', 'm0', 'm1'))
            r0, r1 = x + w0, x + w1
            b0, bj, bc = r0 + m0, r1 + m1, r1 + m0
            for key, value in (('r0', r0), ('r1', r1), ('r1_plus_m0', bc), ('r1_plus_m1', bj)):
                assert cpu.tensor_meta(value, torch)['sha256'] == d[key + '_explicit_action16_metadata']['sha256']
            assert cpu.exact(d['action_hidden'], bc if arm == 'suffix_joint_native_mlp' else bj, torch)
            assert cpu.tensor_meta(d['action_hidden'], torch)['sha256'] == d['actual_postblock_explicit_action16_metadata']['sha256']
            assert cpu.tensor_meta(x, torch)['sha256'] == d['hidden_boundary_metadata'][35]['action']['sha256']
            assert cpu.tensor_meta(d['action_hidden'], torch)['sha256'] == d['hidden_boundary_metadata'][36]['action']['sha256']
            assert cpu.tensor_meta(m0 if arm == 'suffix_joint_native_mlp' else m1, torch)['sha256'] == d['selected_MLP_explicit_action16_metadata']['sha256']
            assert d['two_actual_BF16_addition_reconstruction_gates'] is True
            for state in ('global_RNG_after', 'global_RNG_after_actual_block'):
                assert cpu.exact(d['global_RNG_before'], d[state], torch)
            z0, zd, zs = (endpoint[k] for k in ('Z0', 'Zdonor', 'Zselected'))
            if arm == 'live_self':
                assert all(cpu.exact(a, b, torch) for a, b in ((z0, zd), (z0, zs), (w0, w1), (m0, m1)))
            elif arm != 'matched_random':
                assert cpu.exact(zs, zd, torch)
            else:
                random = d['random']
                seed = 104729 + 1000 * c['probe'].DIRECTIONS.index((row['recipient_goal'], row['donor_goal'])) + step
                g = torch.Generator(device='cpu').manual_seed(seed)
                assert random['seed'] == seed and cpu.exact(g.get_state(), random['generator_state_before'], torch)
                perm = torch.randperm(128, generator=g)
                signs = (torch.randint(0, 2, (1, 16, 32, 128), generator=g, dtype=torch.int64) * 2 - 1).float()
                assert cpu.exact(perm, random['permutation'], torch) and cpu.exact(signs.to(torch.int8), random['signs'], torch)
                assert cpu.exact(g.get_state(), random['generator_state_after'], torch)
                delta = (zd.float() - z0.float()).index_select(-1, perm) * signs * random['alpha']
                assert cpu.exact(delta, d['random_actual_delta_FP32'], torch)
                result = torch.where(delta == 0, z0, (z0.float() + delta).to(torch.bfloat16))
                assert cpu.exact(result, zs, torch)
                target, actual = float((zd.double() - z0.double()).norm()), float((zs.double() - z0.double()).norm())
                # Different Torch reduction kernels may differ in the last scalar bits.
                assert np.isclose(target, random['target_l2_fp64'], rtol=1e-12, atol=1e-12)
                assert np.isclose(actual, random['realized_l2_fp64'], rtol=1e-12, atol=1e-12)
                assert random['strength_gate_pass'] is True and random['relative_error'] <= .02
                assert (0 if target == actual == 0 else abs(actual / target - 1)) <= .02 + 1e-12
                random_count += 1
            a, r, m = [(u.double() - v.double()).numpy() for u, v in ((w1, w0), (r1, r0), (m1, m0))]
            dj, dc = [(u.double() - b0.double()).numpy() for u in (bj, bc)]
            values = metric(a, m, np)
            values.update(actual_J_delta_l2=np.linalg.norm(dj, axis=-1), actual_C_delta_l2=np.linalg.norm(dc, axis=-1),
                R_l2=np.linalg.norm(r, axis=-1), final_rounding_residual_l2=np.linalg.norm(dj - (r + m), axis=-1),
                preW_selected_head_rms=np.sqrt(np.mean((zs.double() - z0.double()).numpy()[0] ** 2, axis=-1)),
                preW_donor_head_rms=np.sqrt(np.mean((zd.double() - z0.double()).numpy()[0] ** 2, axis=-1)))
            pooled = metric(a.reshape(1, -1), m.reshape(1, -1), np)
            effect = d['realized_effect_norms']
            assert np.isclose(float(np.linalg.norm(a)), effect['W_selected_l2'], rtol=1e-12, atol=1e-12)
            values.update(pooled_cosine=pooled['cosine'][0], pooled_sum_to_A_ratio=pooled['sum_to_A_ratio'][0],
                W_selected_l2=effect['W_selected_l2'], W_donor_l2_reported=effect['W_donor_l2'],
                W_ratio_reported=np.nan if effect['actualW_delta_ratio'] is None else effect['actualW_delta_ratio'])
            case_values.append(values)
            sources.append(dict(case=case, step=step, file=entry['file'], sha256=entry['sha256']))
        per_case.append(case_values)
        summaries.append(dict(case=case, arm=arm, mean_actual_J_delta_l2=float(np.mean([v['actual_J_delta_l2'] for v in case_values])),
            mean_actual_C_delta_l2=float(np.mean([v['actual_C_delta_l2'] for v in case_values])),
            W_ratio_reported=[v['W_ratio_reported'] for v in case_values]))
        print('[INTERACTION-ANALYSIS] sites ' + case, flush=True)
    assert random_count == 60
    arrays['t0_J_C_all8_endpoints_exact'] = np.asarray([all(cpu.exact(
        first_inputs[(goal, 'suffix_joint')][key], first_inputs[(goal, 'suffix_joint_native_mlp')][key], torch)
        for key in ('x', 'W0', 'W1', 'm0', 'm1', 'Z0', 'Zdonor', 'Zselected')) for goal, _ in c['probe'].DIRECTIONS])
    assert arrays['t0_J_C_all8_endpoints_exact'].all(), 'Initial J/C inputs or computed candidates differ'
    for key in per_case[0][0]:
        arrays[key] = np.asarray([[v[key] for v in case_values] for case_values in per_case])
    arrays['q0_actions'] = np.stack([records[row['case']]['actions'].double().numpy() for row in c['rows']])
    arrays['q0_actual_velocity'] = np.stack([records[row['case']]['action_velocity'].double().numpy() for row in c['rows']])
    return arrays, summaries, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    base, out = args.output.resolve(), args.output.resolve() / STAGE
    assert not out.exists(), 'Completed or failed analysis is immutable; do not retry'
    c = seal(base)  # Full physical completion is mandatory before output creation.
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    try:
        import numpy as np
        import torch
        assert not torch.cuda.is_initialized()
        torch.set_num_threads(1)
        records, outcomes, physics = physical(c, torch, np)
        arrays, summaries, sources = sites(c, records, torch, np)
        np.savez_compressed(out / 'arrays.npz', **arrays)
        with np.load(out / 'arrays.npz', allow_pickle=False) as archive:
            assert set(archive.files) == set(arrays)
            assert all(archive[k].shape == v.shape and archive[k].dtype == v.dtype and archive[k].tobytes(order='C') == v.tobytes(order='C') for k, v in arrays.items())
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        labels = [f"{r['arm']}: {r['recipient_goal']} -> {r['donor_goal']}" for r in c['rows']]
        for ax, key, title, limits in ((axes[0], 'pooled_cosine', 'Same-call W/MLP delta cosine', (-1, 1)),
                (axes[1], 'actual_J_delta_l2', 'Same-call rounded J block delta L2 (mean16 rows)', (None, None))):
            values = arrays[key] if arrays[key].ndim == 2 else arrays[key].mean(-1)
            im = ax.imshow(values, aspect='auto', origin='upper', vmin=limits[0], vmax=limits[1], cmap='coolwarm' if key == 'pooled_cosine' else 'viridis')
            ax.set_yticks(range(8), labels, fontsize=8)
            ax.set_xlabel('Denoising step (0..29)')
            ax.set_title(title)
            fig.colorbar(im, ax=ax)
        fig.savefig(out / 'same-call-response.png', dpi=160)
        plt.close(fig)
        analysis = dict(state='complete', primary_prediction=c['ec']['primary_prediction'], physical_outcomes=outcomes,
            per_case_same_call_summary=summaries, actual_sites=240, physical_audit=physics, scope=SCOPE,
            both_direction_t0_J_C_same_x_W0_W1_m0_m1_Z_endpoints_byte_exact=True,
            later_J_C_scope='Only t0 pair is matched; later J/C trajectories have dynamic, different inputs.',
            formulas=dict(delta_A='W1.double-W0.double', delta_M='m1.double-m0.double', delta_R='BF16(x+W1).double-BF16(x+W0).double',
                J='BF16(BF16(x+W1)+m1)', C='BF16(BF16(x+W1)+m0)', baseline='BF16(BF16(x+W0)+m0)',
                projection='dot(deltaA,deltaM)/norm(deltaA)^2', sum_ratio='norm(deltaA+deltaM)/norm(deltaA)'),
            W_donor_denominator_scope='Producer actual scalar and live SHA; Wdonor raw not exported. Only W1-W0 numerator independently reconstructed.',
            zero_norm_scope='Undefined axes retained as NPZ NaN+valid masks; JSON null.',
            scalar_reduction_scope='Description-only Float64 reductions verified within1e-12 across Torch kernels; pointwise/RNG/BF16 bytes remain exact.',
            no_new_model_forward=0, no_dispatch=0, no_solver=0, no_physics=0, CUDA_initialized=False)
        write(out / 'analysis.json', clean(analysis, np))
        write(out / 'provenance.json', dict(script_sha256=sha(Path(__file__)), producer_script_sha256=PROBE_SHA,
            execution_script_sha256=EXECUTION_SHA, producer_complete_sha256=PRODUCER_COMPLETE_SHA,
            physics_complete_sha256=PHYSICS_COMPLETE_SHA, old_completion_seals=OLD_SEALS,
            launcher_sha256=LAUNCHER_SHA, preserved_failure_sha256=PRESERVED_FAILURE_SHA,
            new_site_files=sources, new_execution_documents_sha256=c['ec']['files_sha256'],
            source_sha256={k: sha(v) for k, v in c['paths'].items()}, torch_version=str(torch.__version__), scope=SCOPE))
        assert not torch.cuda.is_initialized()
        write(out / 'complete.json', dict(state='complete', script_sha256=sha(Path(__file__)), actual_sites=240,
            actual_physical_trials=8, actual_model_records=64, actual_converted_actions=1024, actual_later_noise_pairs=49,
            primary_prediction=c['ec']['primary_prediction'], CUDA_initialized=False, new_model_forwards=0,
            files_sha256={name: sha(out / name) for name in ('analysis.json', 'provenance.json', 'arrays.npz', 'same-call-response.png')},
            elapsed_seconds=time.monotonic() - started))
        print('[INTERACTION-ANALYSIS] complete ' + str(out), flush=True)
    except Exception as exc:
        write(out / 'failed.json', dict(state='failed', error=str(exc), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
