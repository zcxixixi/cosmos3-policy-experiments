"""CPU audit of saved live head arrays and completed twelve physical trials.

No model, official attention dispatch, solver, environment or CUDA calls.
The QK softmax below is an offline FP32 diagnostic, never an intervention.
All 1620 saved sites are checked. Head numbers are the real pre-W axis;
different arms share Q/K only within a site, not over the whole trajectory.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time


ROOT = Path('/home/current/work/cosmos3')
PROBE_SHA = '0610b1d0320512ff4f5ce4163b52337ccb3ddf3ed1eb8d6c13f51b614097b911'
EXECUTION_SHA = 'e900f970e4049f6db1755274d68fa38366a9a29916354ccc8357090c44165a51'
ARMS = ('native', 'all_allowed', 'arithmetic_sham', 'value_zero', 'allocation_only', 'hardmask')
PLAN = tuple((f'V{v}_A195_{arm}', v, arm) for arm in ARMS for v in (195, 198))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open('x') as f:
        f.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def require_tensor_bytes(a, b, torch, label):
    assert a.shape == b.shape and a.dtype == b.dtype, label
    assert torch.equal(a.contiguous().reshape(-1).view(torch.uint8),
                       b.contiguous().reshape(-1).view(torch.uint8)), label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base = args.output.resolve()
    producer, execution = base / 'future-value-mechanism', base / 'future-value-execution'
    probe_path = ROOT / 'work/probe_cosmos_milk_future_value_mechanism.py'
    execution_path = ROOT / 'work/run_cosmos_milk_future_value_execution.py'
    assert PROBE_SHA and EXECUTION_SHA and sha(probe_path) == PROBE_SHA and sha(execution_path) == EXECUTION_SHA
    probe = load_module('value_analysis_frozen_probe', probe_path)
    assert probe.PLAN == PLAN
    pc, pr, pp = (read(producer / (n + '.json')) for n in ('complete', 'results', 'protocol'))
    ec = read(execution / 'complete.json')
    assert not (producer / 'failed.json').exists() and not (execution / 'wrapper_failed.json').exists()
    assert pc['state'] == pr['state'] == ec['state'] == 'complete'
    assert pc['script_sha256'] == PROBE_SHA and ec['script_sha256'] == EXECUTION_SHA
    for n in ('results', 'provenance', 'protocol', 'sources', 'primary-prediction'):
        assert pc[n.replace('-', '_') + '_sha256'] == sha(producer / (n + '.json'))
    for n, expected in ec['files_sha256'].items():
        assert sha(execution / n) == expected
    controls = read(execution / 'controls.json')
    summary = read(execution / 'summary.json')
    labels = [label for label, _, _ in PLAN]
    assert ec['trial_order'] == summary['completed_trials'] == labels
    assert [row['trial'] for row in summary['cases']] == labels
    assert set(controls['actual_case_files_sha256']) == set(labels)
    physical_sources = []
    assert pr['official_dispatch_counts'] == probe.expected_counts()
    assert pc['actual_selected_site_captures'] == 1620 and pc['fresh_model_forwards'] == 360
    assert pp['trial_order'] == [label for label, _, _ in PLAN]
    assert pc['total_official_dispatch_calls'] == 38070
    out = base / 'future-value-analysis'
    out.mkdir(exist_ok=False)
    started = time.perf_counter()
    import torch
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    rows, site_sources, inputs, states = [], [], {}, {}
    # case, step, layer, actual query head. Mean over current50 x head128.
    delta_rms = np.zeros((12, 15, 9, 32), dtype=np.float64)
    base_rms = np.zeros_like(delta_rms)
    mass = np.full((12, 15, 9, 32, 4), np.nan, dtype=np.float64)
    recon_error = np.full((12, 15, 9, 32), np.nan, dtype=np.float64)
    quantization = np.full((12, 15, 9, 32), np.nan, dtype=np.float64)
    endpoint_sets = []
    same_first_QKV = {}
    for ci, ((label, v, arm), result) in enumerate(zip(PLAN, pr['cases'])):
        assert result['case'] == label and result['arm'] == arm
        folder = producer / label
        for n, expected in result['files_sha256'].items():
            assert sha(folder / n) == expected
        manifest = read(folder / 'site-captures.json')
        assert manifest['state'] == 'complete' and manifest['sites'] == len(manifest['files']) == 135
        assert [(s['step'], s['layer_zero_based']) for s in manifest['files']] == [(t, l) for t in range(15) for l in range(9)]
        record = torch.load(folder / 'states.pt', map_location='cpu', weights_only=True, mmap=True)
        states[label] = record
        inputs[label] = record['model_input']
        assert read(folder / 'normalized_actions.json') == record['actions'].tolist()
        for item in manifest['files']:
            path = folder / item['file']
            assert sha(path) == item['sha256'] and path.stat().st_size == item['bytes']
            d = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
            t, l = item['step'], item['layer_zero_based']
            assert (d['step'], d['layer_zero_based'], d['case_arm']) == (t, l, arm)
            q, k, val = d['Qcurrent50'], d['Kall'], d['Vall']
            assert q.shape == (1, 50, 32, 128) and k.shape == val.shape == (1, 387, 8, 128)
            assert all(a.dtype == torch.bfloat16 and torch.isfinite(a).all() for a in (q, k, val))
            z = d['endpoints']
            expected_endpoints = {'Znative', 'Zreal'}
            if arm != 'native':
                expected_endpoints.add('Z0')
            if arm in ('value_zero', 'allocation_only'):
                expected_endpoints.add('Zv')
            if arm in ('allocation_only', 'hardmask'):
                expected_endpoints.add('Zk')
            if arm == 'arithmetic_sham':
                expected_endpoints.add('Zclone')
            assert set(z) == {'Znative', 'Z0', 'Zv', 'Zk', 'Zreal', 'Zclone'}
            assert {name for name, value in z.items() if value is not None} == expected_endpoints
            assert [n for n, a in z.items() if a is None] == d['missing_endpoints'] == item['missing_endpoints']
            assert all(a is None or (a.shape == q.shape and a.dtype == torch.bfloat16 and torch.isfinite(a).all()) for a in z.values())
            assert d['actual_pre_W_current_bytes_equal_Zreal'] is d['actual_QKV_bytes_unchanged'] is True
            assert d['original_full_projection_calls_at_site'] == 1
            for device in ('cpu', 'cuda'):
                assert torch.equal(d['RNG_before'][device], d['RNG_after'][device])
            ref = z['Znative'] if arm == 'native' else z['Z0']
            delta = z['Zreal'].double() - ref.double()
            delta_rms[ci, t, l] = delta.square().mean(dim=(0, 1, 3)).sqrt().numpy()
            base_rms[ci, t, l] = ref.double().square().mean(dim=(0, 1, 3)).sqrt().numpy()
            assert [x['query_head'] for x in d['headwise_effective_vs_native_or_Z0']] == list(range(32))
            for h, witness in enumerate(d['headwise_effective_vs_native_or_Z0']):
                assert witness['kv_head'] == h // 4
                assert witness['delta_l2'] == float(delta[:, :, h].norm())
            if arm in ('native', 'all_allowed', 'arithmetic_sham'):
                require_tensor_bytes(z['Zreal'], ref, torch, 'Control endpoint bytes, including signed zero')
            elif arm == 'value_zero':
                require_tensor_bytes(z['Zreal'], z['Zv'], torch, 'Value-zero actual endpoint bytes')
            elif arm == 'hardmask':
                require_tensor_bytes(z['Zreal'], z['Zk'], torch, 'Hardmask actual endpoint bytes')
            if arm == 'arithmetic_sham':
                require_tensor_bytes(z['Zclone'], z['Z0'], torch, 'Same-value clone official endpoint bytes')
            if t == l == 0:
                if v not in same_first_QKV:
                    same_first_QKV[v] = (q.clone(), k.clone(), val.clone())
                for a, b in zip((q, k, val), same_first_QKV[v]):
                    assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), 'Own-V first-site actual QKV differs across arms'
            fp = d.get('FP32allocation') if arm == 'allocation_only' else d.get('FP32arithmetic_sham')
            assert (d['FP32allocation'] is not None) is (arm == 'allocation_only')
            assert (d['FP32arithmetic_sham'] is not None) is (arm == 'arithmetic_sham')
            assert (fp is not None) is (arm in ('allocation_only', 'arithmetic_sham'))
            if fp is not None:
                assert fp.shape == q.shape and fp.dtype == torch.float32
                intended = z['Z0'].float() + (z['Zk'].float() - z['Zv'].float() if arm == 'allocation_only' else z['Z0'].float() - z['Z0'].float())
                require_tensor_bytes(fp, intended, torch, 'Actual FP32 formula bytes')
                expected = intended.to(torch.bfloat16)
                increment = z['Zk'].float() - z['Zv'].float() if arm == 'allocation_only' else torch.zeros_like(intended)
                expected = torch.where(increment == 0, z['Z0'], expected)
                require_tensor_bytes(expected, z['Zreal'], torch, 'Actual BF16 formula bytes, including signed zero')
                quantization[ci, t, l] = (z['Zreal'].double() - fp.double()).square().mean(dim=(0, 1, 3)).sqrt().numpy()
            # Offline reconstruction only on the two common all-allowed baselines.
            # No official dispatch or model call, and these values never enter inference.
            if arm == 'all_allowed':
                qh = q[0].float().permute(1, 0, 2)
                kh = k[0].float().repeat_interleave(4, dim=1).permute(1, 2, 0)
                vh = val[0].float().repeat_interleave(4, dim=1).permute(1, 0, 2)
                probabilities = (torch.matmul(qh, kh) / (128 ** .5)).softmax(dim=-1)
                current_keys = d['current_rows'] + 121
                future_keys = d['future_keys']
                index = torch.load(folder / 'indexes-and-masks.pt', map_location='cpu', weights_only=True, mmap=True)['indexes']
                groups = (index['text_keys'], current_keys, future_keys, index['action_keys'])
                assert torch.equal(torch.cat(groups).sort().values, torch.arange(387))
                for gi, keys in enumerate(groups):
                    mass[ci, t, l, :, gi] = probabilities[:, :, keys].double().sum(-1).mean(-1).numpy()
                reconstruction = torch.matmul(probabilities, vh).permute(1, 0, 2)[None]
                recon_error[ci, t, l] = (reconstruction.double() - z['Z0'].double()).square().mean(dim=(0, 1, 3)).sqrt().numpy()
            site_sources.append(dict(case=label, step=t, layer=l, file=item['file'], sha256=item['sha256'], bytes=item['bytes']))
        endpoint_sets.append(dict(case=label, available=manifest['files'][0]['captured_endpoints'], missing=manifest['files'][0]['missing_endpoints']))
        print(json.dumps(dict(stage='case_audited', case=label, sites_checked=len(site_sources))), flush=True)
    # Actual final normalized actions, not decoded meaning or success probabilities.
    actions = np.stack([states[label]['actions'].numpy() for label, _, _ in PLAN])
    action_rms = []
    for (label, v, arm), summary_row in zip(PLAN, summary['cases']):
        a = states[label]['actions'].double()
        b = states[f'V{v}_A195_all_allowed']['actions'].double()
        physical_folder = execution / 'closed-loop' / label
        hashes = controls['actual_case_files_sha256'][label]
        for name in ('summary.json', 'strict-windows.json'):
            assert sha(physical_folder / name) == hashes[name]
        physical = read(physical_folder / 'summary.json')
        strict = read(physical_folder / 'strict-windows.json')
        assert physical == summary_row and physical['trial'] == label
        assert strict['records'] == 129 and set(strict['objects']) == set(physical['per_object'])
        firsts, selected = [], set()
        for name, obj in strict['objects'].items():
            window = obj['first_window']
            first = None if window is None else window[0]
            assert window is None or window == list(range(first, first + 5))
            assert first == (obj['five_record_window_start_steps'][0] if obj['five_record_window_start_steps'] else None)
            assert physical['per_object'][name]['first_2cm_for_5frames'] == first
            if first is not None:
                firsts.append(first)
                selected.add(name)
        assert selected == set(physical['selected_objects'])
        assert physical['first_selection_step'] == (min(firsts) if firsts else None)
        physical_sources.append(dict(case=label, files_sha256={name: hashes[name]
            for name in ('summary.json', 'strict-windows.json')}))
        rows.append(dict(case=label, arm=arm, vision_noise=v,
            xyz_RMS_vs_own_sham=float((a[:, :3] - b[:, :3]).square().mean().sqrt()),
            all10_RMS_vs_own_sham=float((a - b).square().mean().sqrt()),
            selected_objects=physical['selected_objects'], first_selection_step=physical['first_selection_step']))
        action_rms.append(rows[-1]['xyz_RMS_vs_own_sham'])
    arrays = dict(head_delta_rms=delta_rms, head_reference_rms=base_rms,
        offline_source_attention_mass=mass, offline_reconstruction_rms=recon_error,
        BF16_cast_rms=quantization, final_normalized_actions=actions,
        case_names=np.array([a for a, _, _ in PLAN]), source_group_names=np.array(['UND121', 'current50', 'future200', 'action16']))
    np.savez_compressed(out / 'actual-head-metrics.npz', **arrays)
    with np.load(out / 'actual-head-metrics.npz', allow_pickle=False) as saved:
        for name, a in arrays.items():
            assert np.array_equal(saved[name], a, equal_nan=True) if a.dtype.kind in 'fc' else np.array_equal(saved[name], a)
    fig, axes = plt.subplots(4, 2, figsize=(13, 11), constrained_layout=True)
    for row, arm in enumerate(('arithmetic_sham', 'value_zero', 'allocation_only', 'hardmask')):
        for col, v in enumerate((195, 198)):
            ci = [x[0] for x in PLAN].index(f'V{v}_A195_{arm}')
            im = axes[row, col].imshow(delta_rms[ci].mean(axis=0), aspect='auto', origin='lower')
            axes[row, col].set(title=f'{arm} / V{v}', xlabel='Actual query head (0..31)', ylabel='Layer (0..8)')
            fig.colorbar(im, ax=axes[row, col], label='Mean within-site effective RMS')
    fig.savefig(out / 'actual-current-head-change.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for ax, v in zip(axes, (195, 198)):
        ci = [x[0] for x in PLAN].index(f'V{v}_A195_all_allowed')
        im = ax.imshow(mass[ci, :, :, :, 2].mean(axis=0) * 100, origin='lower', aspect='auto', vmin=0, vmax=100)
        ax.set(title=f'All-allowed V{v}', xlabel='Actual query head (0..31)', ylabel='Layer (0..8)')
        fig.colorbar(im, ax=ax, label='Offline future attention mass (%)')
    fig.savefig(out / 'offline-future-attention-mass.png', dpi=160)
    plt.close(fig)
    diagnostics = []
    for ci, (label, _, arm) in enumerate(PLAN):
        if arm == 'all_allowed':
            err, ref = recon_error[ci], base_rms[ci]
            diagnostics.append(dict(case=label, reconstruction_RMS_max=float(err.max()),
                relative_RMS_percent_median=float(np.median(100 * err / np.maximum(ref, 1e-30))),
                relative_RMS_percent_max=float(np.max(100 * err / np.maximum(ref, 1e-30))),
                uniform_future_mass_percent=100 * 200 / 387))
    report = dict(state='complete', sites_checked=1620, cases=rows, endpoint_coverage=endpoint_sets,
        offline_diagnostics=diagnostics, primary_physical_prediction=read(execution / 'primary-prediction.json'),
        actual_source_files_sha256=site_sources, all_first_site_own_V_QKV_bytes_equal=True,
        actual_physical_summary_and_strict_sources_sha256=physical_sources,
        all12_physical_rows_and_first_strict_windows_consistent=True,
        all_actual_endpoint_BF16_bytes_checked_including_signed_zero=True,
        actual_quantized_additive_formula_byte_exact=True, array_roundtrip_exact=True,
        model_calls=0, official_attention_dispatch_calls=0, solver_calls=0, physics_calls=0, CUDA_calls=0,
        interpretation='Heatmaps show actual local intervention amplitude per pre-W head. Offline QK mass includes many future keys and is not semantic importance. No head selected, milk identity decoded, brain region identified or cross-position repair demonstrated.')
    write(out / 'analysis.json', report)
    names = ('actual-head-metrics.npz', 'actual-current-head-change.png', 'offline-future-attention-mass.png', 'analysis.json')
    write(out / 'complete.json', dict(state='complete', script_sha256=sha(Path(__file__)),
        producer_complete_sha256=sha(producer / 'complete.json'), execution_complete_sha256=sha(execution / 'complete.json'),
        files_sha256={n: sha(out / n) for n in names}, elapsed_s=time.perf_counter() - started,
        sites_checked=1620, model_calls=0, official_attention_dispatch_calls=0, solver_calls=0, physics_calls=0, CUDA_calls=0))
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(stage='complete', cases=rows, elapsed_s=time.perf_counter() - started)), flush=True)


if __name__ == '__main__':
    main()
