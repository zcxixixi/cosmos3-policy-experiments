"""CPU geometry of all saved allocation_only sites at X12, V195/V198.

Read the completed future-value-mechanism producer; create only absent
BASE/future-head-geometry. Use its actual BF16 Z0/Zv/Zk endpoints, never
reconstructed attention. Dv=Zv-Z0, Da=Zk-Zv and Dk=Zk-Z0 are evaluated in
Float64. Report all135 sites per V on the real32 pre-W query-head axis and
eight shared-KV groups of four heads. No model, dispatch, solver or physics.
Cosine, cancellation and RMS describe geometry, not head semantics or a
causal decomposition of final behavior. Different sites/arms have dynamic
inputs; aggregate statistics do not identify a common semantic direction.
"""

import argparse
import hashlib
import os
from pathlib import Path
import time


ROOT = Path('/home/current/work/cosmos3')
PROBE_SHA = '0610b1d0320512ff4f5ce4163b52337ccb3ddf3ed1eb8d6c13f51b614097b911'
METHODS_SHA = '7d791c896896a132033574940c1479fb7ca6e0a33a4973936b8ff3023b601e34'
PLAN = tuple((f'V{v}_A195_allocation_only', v) for v in (195, 198))
SITES = tuple((t, l) for t in range(15) for l in range(9))
FLOOR = 1e-12
METRICS = ('cosine_va', 'cosine_vk', 'cosine_ak', 'net_ratio', 'cancellation',
           'rms_v', 'rms_a', 'rms_k', 'norm_v', 'norm_a', 'norm_k', 'dot_va')
SCOPE = ('Actual same-site BF16 official endpoints from allocation_only only, '
         'Float64 differences and descriptive geometry before W. Da is Zk-Zv, '
         'not the quantized allocation arm minus Z0. The eight groups identify '
         'shared KV heads, not brain areas, objects or motion functions. No '
         'new attention reconstruction, inference, physics, head selection or '
         'additive causal percentage. Two V realizations at X12 only.')


def tensor_meta(value, torch):
    raw = value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                sha256=hashlib.sha256(raw).hexdigest())


def geometry(v, a, k, np):
    """Rows are one actual head/group; columns concatenate its observed axes."""
    assert v.shape == a.shape == k.shape and v.ndim == 2
    assert all(x.dtype == np.float64 and np.isfinite(x).all() for x in (v, a, k))
    nv, na, nk = (np.linalg.norm(x, axis=1) for x in (v, a, k))
    dot = np.einsum('ij,ij->i', v, a)

    def cosine(x, y, nx, ny):
        valid = (nx > FLOOR) & (ny > FLOOR)
        result = np.full_like(nx, np.nan)
        np.divide(np.einsum('ij,ij->i', x, y), nx * ny, out=result, where=valid)
        assert np.all(np.abs(result[valid]) <= 1 + 64 * np.finfo(np.float64).eps)
        return result, valid

    cva, va_valid = cosine(v, a, nv, na)
    cvk, vk_valid = cosine(v, k, nv, nk)
    cak, ak_valid = cosine(a, k, na, nk)
    ratio_valid = (nv + na) > FLOOR
    ratio = np.full_like(nv, np.nan)
    np.divide(nk, nv + na, out=ratio, where=ratio_valid)
    assert np.all(ratio[ratio_valid] >= 0)
    assert np.all(ratio[ratio_valid] <= 1 + 64 * np.finfo(np.float64).eps)
    return dict(cosine_va=cva, cosine_vk=cvk, cosine_ak=cak,
        net_ratio=ratio, cancellation=1 - ratio,
        rms_v=nv / np.sqrt(v.shape[1]), rms_a=na / np.sqrt(v.shape[1]),
        rms_k=nk / np.sqrt(v.shape[1]), norm_v=nv, norm_a=na, norm_k=nk,
        dot_va=dot, valid_va=va_valid, valid_vk=vk_valid, valid_ak=ak_valid,
        valid_ratio=ratio_valid)


def distribution(values, np):
    finite = values[np.isfinite(values)]
    if not finite.size:
        return dict(valid_count=0, min=None, q25=None, median=None, q75=None, max=None)
    q = np.quantile(finite, [0, .25, .5, .75, 1])
    return dict(valid_count=int(finite.size), **dict(zip(('min', 'q25', 'median', 'q75', 'max'), map(float, q))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Completed baseline BASE')
    base = parser.parse_args().output.resolve()
    methods_path = ROOT / 'work/analyze_cosmos_milk_future_value_mechanism.py'
    probe_path = ROOT / 'work/probe_cosmos_milk_future_value_mechanism.py'
    # Import only the frozen standard-library helpers; never invoke old main.
    import importlib.util
    spec = importlib.util.spec_from_file_location('head_geometry_frozen_helpers', methods_path)
    assert hashlib.sha256(methods_path.read_bytes()).hexdigest() == METHODS_SHA
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
    sha, read, write = utils.sha, utils.read, utils.write
    assert sha(probe_path) == PROBE_SHA
    probe = utils.load_module('head_geometry_frozen_producer', probe_path)
    producer = base / 'future-value-mechanism'
    assert not (producer / 'failed.json').exists()
    names = ('complete', 'results', 'provenance', 'protocol', 'sources', 'primary-prediction')
    docs = {name: read(producer / (name + '.json')) for name in names}
    pc, results, protocol, provenance = (docs[name] for name in ('complete', 'results', 'protocol', 'provenance'))
    assert pc['state'] == results['state'] == 'complete'
    assert pc['script_sha256'] == provenance['script_sha256'] == PROBE_SHA
    for name in names[1:]:
        assert pc[name.replace('-', '_') + '_sha256'] == sha(producer / (name + '.json'))
    assert protocol['trial_order'] == [row[0] for row in probe.PLAN]
    assert [row['case'] for row in results['cases']] == protocol['trial_order']
    assert pc['official_dispatch_counts'] == results['official_dispatch_counts'] == probe.expected_counts()
    assert pc['actual_selected_site_captures'] == 1620 and pc['fresh_model_forwards'] == 360
    assert pc['total_official_dispatch_calls'] == 38070
    assert protocol['selected_window_zero_based_inclusive'] == dict(steps=[0, 14], layers=[0, 8])
    references = {row['case']: row for row in results['cases']}
    out = base / 'future-head-geometry'
    out.mkdir(exist_ok=False)
    started = time.perf_counter()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    byte_equal = lambda a, b, label: utils.require_tensor_bytes(a, b, torch, label)
    head = {name: np.full((2, 15, 9, 32), np.nan, dtype=np.float64) for name in METRICS}
    group = {name: np.full((2, 15, 9, 8), np.nan, dtype=np.float64) for name in METRICS}
    for data, width in ((head, 32), (group, 8)):
        data.update({key: np.zeros((2, 15, 9, width), dtype=bool)
            for key in ('valid_va', 'valid_vk', 'valid_ak', 'valid_ratio')})
    identity_maxabs = np.zeros((2, 15, 9), dtype=np.float64)
    identity_tolerance = np.zeros_like(identity_maxabs)
    sources, case_sources, summaries = [], [], []
    for ci, (label, v) in enumerate(PLAN):
        row, folder = references[label], producer / label
        assert row['arm'] == 'allocation_only' and row['vision_noise_source_seed'] == v
        assert row['action_noise_source_seed'] == row['runtime_rng_seed'] == 195
        assert row['window'] == dict(steps=[0, 14], layers=[0, 8])
        for name, expected in row['files_sha256'].items():
            assert sha(folder / name) == expected
        manifest = read(folder / 'site-captures.json')
        report = read(folder / 'observer-report.json')
        assert row['site_capture_manifest_sha256'] == report['site_capture_manifest_sha256'] == row['files_sha256']['site-captures.json']
        assert manifest['state'] == 'complete' and manifest['case_arm'] == 'allocation_only'
        assert manifest['sites'] == len(manifest['files']) == 135
        assert [(item['step'], item['layer_zero_based']) for item in manifest['files']] == list(SITES)
        assert report['counts'] == probe.expected_counts('allocation_only')
        assert report['all30_actual37_current_action_boundaries_finite'] is True
        assert report['indexes_and_masks_sha256'] == row['files_sha256']['indexes-and-masks.pt']
        index = torch.load(folder / 'indexes-and-masks.pt', map_location='cpu', weights_only=True, mmap=True)['indexes']
        assert index['und_len'] == 121 and index['gen_len'] == 266 and index['full_joint_length'] == 387
        byte_equal(index['current_keys'], index['current_rows'] + 121, 'Actual current query/key mapping')
        assert [index[key].numel() for key in ('text_keys', 'current_keys', 'future_keys', 'action_keys')] == [121, 50, 200, 16]
        assert torch.equal(torch.cat([index[key] for key in ('text_keys', 'current_keys', 'future_keys', 'action_keys')]).sort().values, torch.arange(387))
        case_sources.append(dict(case=label, files_sha256=row['files_sha256'], site_manifest_sha256=row['site_capture_manifest_sha256']))
        for item in manifest['files']:
            t, l = item['step'], item['layer_zero_based']
            assert item['file'] == f'sites/t{t:02d}-L{l:02d}.pt'
            path = folder / item['file']
            assert sha(path) == item['sha256'] and path.stat().st_size == item['bytes']
            d = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
            assert (d['step'], d['layer_zero_based'], d['case_arm']) == (t, l, 'allocation_only')
            byte_equal(d['current_rows'], index['current_rows'], 'Captured current50 mapping')
            byte_equal(d['future_keys'], index['future_keys'], 'Captured future200 mapping')
            q, k, val = d['Qcurrent50'], d['Kall'], d['Vall']
            assert q.shape == (1, 50, 32, 128) and k.shape == val.shape == (1, 387, 8, 128)
            assert all(x.dtype == torch.bfloat16 and bool(torch.isfinite(x).all()) for x in (q, k, val))
            z = d['endpoints']
            assert set(z) == {'Znative', 'Z0', 'Zv', 'Zk', 'Zreal', 'Zclone'}
            assert {key for key, value in z.items() if value is not None} == {'Znative', 'Z0', 'Zv', 'Zk', 'Zreal'}
            assert d['missing_endpoints'] == item['missing_endpoints'] == ['Zclone']
            assert set(item['captured_endpoints']) == {'Znative', 'Z0', 'Zv', 'Zk', 'Zreal'}
            for key, value in z.items():
                if value is None:
                    assert d['endpoint_metadata'][key] is None
                    continue
                assert value.shape == q.shape and value.dtype == torch.bfloat16 and bool(torch.isfinite(value).all())
                assert all(d['endpoint_metadata'][key][field] == expected for field, expected in tensor_meta(value, torch).items())
            assert d['actual_pre_W_current_bytes_equal_Zreal'] is d['actual_QKV_bytes_unchanged'] is True
            assert d['original_full_projection_calls_at_site'] == 1
            for device in ('cpu', 'cuda'):
                byte_equal(d['RNG_before'][device], d['RNG_after'][device], 'Saved unchanged RNG')
            assert d['FP32allocation'] is not None and d['FP32arithmetic_sham'] is None
            intended = z['Z0'].float() + (z['Zk'].float() - z['Zv'].float())
            byte_equal(d['FP32allocation'], intended, 'Actual FP32 allocation formula')
            quantized = torch.where(z['Zk'].float() - z['Zv'].float() == 0, z['Z0'], intended.to(torch.bfloat16))
            byte_equal(z['Zreal'], quantized, 'Actual BF16 allocation endpoint including signed zero')
            z0, zv, zk = (z[key].double().numpy()[0] for key in ('Z0', 'Zv', 'Zk'))
            dv, da, dk = zv - z0, zk - zv, zk - z0
            residual = dv + da - dk
            tolerance = 8 * np.finfo(np.float64).eps * max(1., float(np.abs(z0).max()), float(np.abs(zv).max()), float(np.abs(zk).max()))
            identity_maxabs[ci, t, l] = float(np.abs(residual).max())
            identity_tolerance[ci, t, l] = tolerance
            assert identity_maxabs[ci, t, l] <= tolerance, 'Float64 endpoint identity failed'
            vectors = tuple(x.transpose(1, 0, 2).reshape(32, 50 * 128) for x in (dv, da, dk))
            hm = geometry(*vectors, np)
            gm = geometry(*(x.reshape(8, 4 * 50 * 128) for x in vectors), np)
            for target, actual in ((head, hm), (group, gm)):
                for key, value in actual.items():
                    target[key][ci, t, l] = value
            sources.append(dict(case=label, step=t, layer_zero_based=l, file=item['file'], sha256=item['sha256'], bytes=item['bytes'],
                endpoint_raw_sha256={key: tensor_meta(z[key], torch)['sha256'] for key in ('Z0', 'Zv', 'Zk')}))
        print(f'[HEAD-GEOMETRY] audited {label}: 135 actual sites', flush=True)
        for g in range(8):
            values = {key: group[key][ci, :, :, g] for key in group}
            nv2 = float(np.square(values['norm_v']).sum())
            na2 = float(np.square(values['norm_a']).sum())
            nk2 = float(np.square(values['norm_k']).sum())
            pooled_denominator = np.sqrt(nv2 * na2)
            pooled_ratio_denominator = np.sqrt(nv2) + np.sqrt(na2)
            summaries.append(dict(case=label, group=g, query_heads=list(range(4 * g, 4 * g + 4)), sites=135,
                cosine_va=distribution(values['cosine_va'], np), net_ratio=distribution(values['net_ratio'], np),
                cancellation=distribution(values['cancellation'], np),
                rms={key: distribution(values['rms_' + key], np) for key in ('v', 'a', 'k')},
                fraction_negative_cosine=None if not values['valid_va'].any() else float((values['cosine_va'][values['valid_va']] < 0).mean()),
                pooled_cosine_va=None if pooled_denominator <= FLOOR * FLOOR else float(values['dot_va'].sum() / pooled_denominator),
                pooled_net_ratio=None if pooled_ratio_denominator <= FLOOR else float(np.sqrt(nk2) / pooled_ratio_denominator)))
    assert len(sources) == 270
    arrays = {f'head_{key}': value for key, value in head.items()}
    arrays.update({f'group_{key}': value for key, value in group.items()})
    arrays.update(identity_maxabs=identity_maxabs, identity_tolerance=identity_tolerance,
        cases=np.array([label for label, _ in PLAN]), steps=np.arange(15), layers_zero_based=np.arange(9),
        query_head_to_kv_group=np.arange(32) // 4)
    np.savez_compressed(out / 'head-geometry.npz', **arrays)
    with np.load(out / 'head-geometry.npz', allow_pickle=False) as saved:
        assert set(saved.files) == set(arrays)
        for key, expected in arrays.items():
            actual = saved[key]
            assert actual.dtype == expected.dtype and actual.shape == expected.shape
            assert actual.tobytes(order='C') == expected.tobytes(order='C'), ('NPZ byte roundtrip', key)
    # A site is a step/layer pair; these show every site, not only firstsite.
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    for ci, (_, v) in enumerate(PLAN):
        for col, (key, lo, hi, cmap) in enumerate((('cosine_va', -1, 1, 'coolwarm'), ('net_ratio', 0, 1, 'viridis'))):
            image = axes[ci, col].imshow(head[key][ci].reshape(135, 32).T, origin='lower', aspect='auto', vmin=lo, vmax=hi, cmap=cmap)
            axes[ci, col].set(title=f'V{v}: {key}', xlabel='Site index = step * 9 + layer (0..134)', ylabel='Actual query head (0..31)')
            for edge in (3.5, 7.5, 11.5, 15.5, 19.5, 23.5, 27.5):
                axes[ci, col].axhline(edge, color='white', linewidth=.5)
            fig.colorbar(image, ax=axes[ci, col], label=key + ' (undefined shown blank)')
    fig.savefig(out / 'all-site-head-geometry.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(4, 8, figsize=(19, 10), constrained_layout=True)
    for ci, (_, v) in enumerate(PLAN):
        for metric_row, (key, lo, hi, cmap) in enumerate((('cosine_va', -1, 1, 'coolwarm'), ('net_ratio', 0, 1, 'viridis'))):
            for g in range(8):
                ax = axes[2 * ci + metric_row, g]
                image = ax.imshow(group[key][ci, :, :, g].T, origin='lower', aspect='auto', vmin=lo, vmax=hi, cmap=cmap)
                ax.set(title=f'V{v} G{g}: h{4*g}..{4*g+3}', xlabel='Denoising step (0..14)', ylabel='Layer (0..8)')
            fig.colorbar(image, ax=axes[2 * ci + metric_row, :].tolist(), label=key)
    fig.savefig(out / 'all-site-group-geometry.png', dpi=160)
    plt.close(fig)
    provenance_out = dict(script_sha256=sha(Path(__file__)), producer_script_sha256=PROBE_SHA, helper_script_sha256=METHODS_SHA,
        producer_documents_sha256={name: sha(producer / (name + '.json')) for name in names},
        selected_case_files=case_sources, actual_site_files=sources,
        original_producer_provenance=provenance, statistical_scope=SCOPE)
    write(out / 'provenance.json', provenance_out)
    write(out / 'analysis.json', dict(state='complete', sites_checked=270, cases=[label for label, _ in PLAN], group_summaries=summaries,
        formulas=dict(Dv='Float64(Zv)-Float64(Z0)', Da='Float64(Zk)-Float64(Zv)', Dk='Float64(Zk)-Float64(Z0)',
            cosine_va='dot(Dv,Da)/(norm(Dv)*norm(Da))', net_ratio='norm(Dk)/(norm(Dv)+norm(Da))', cancellation='1-net_ratio',
            rms='L2/sqrt(number of concatenated current-query/head-channel values)'),
        zero_denominator='Norms <=1e-12 are undefined: NaN plus explicit validity arrays; never zero cosine.',
        group_mapping='Gg concatenates actual query heads 4g..4g+3, each50x128 values; shared KV group, not functional region.',
        pooling='Distributions are per-site. Pooled metrics concatenate all135site vectors via sums of dot products and squared norms; not a mean cosine.',
        identity_max_abs=float(identity_maxabs.max()), identity_checked=True, all_selected_artifact_hashes_checked=True,
        actual_quantized_allocation_formula_bytes_checked=True, NPZ_roundtrip_bytes_checked=True,
        model_calls=0, official_attention_dispatch_calls=0, solver_calls=0, physics_calls=0, CUDA_initialized=False, scope=SCOPE))
    outputs = ('head-geometry.npz', 'all-site-head-geometry.png', 'all-site-group-geometry.png', 'provenance.json', 'analysis.json')
    assert not torch.cuda.is_initialized()
    write(out / 'complete.json', dict(state='complete', script_sha256=sha(Path(__file__)), sites_checked=270,
        files_sha256={name: sha(out / name) for name in outputs}, elapsed_s=time.perf_counter() - started,
        model_calls=0, official_attention_dispatch_calls=0, solver_calls=0, physics_calls=0, CUDA_initialized=False))
    print(f'[HEAD-GEOMETRY] complete: 270 sites, {time.perf_counter()-started:.2f}s', flush=True)


if __name__ == '__main__':
    main()
