"""CPU activity/propagation plots from ten completed saved future-read cases.

Use --output BASE. Create only absent BASE/future-read-activity. Read actual
37-boundary current50/action16 hidden arrays, raw heads and solver records.
No model import/forward, attention replay, solver step, new physics or latent
semantic decoding. Future200 boundary hidden states were not recorded.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback


ROOT = Path('/home/current/work/cosmos3')
PRODUCER_SHA = 'c106bba687c6eaf0853ac9abe7aa2767f9464365162399c201e98b2369af7a4a'
ARMS = ('native', 'all_allowed', 'cut_future_current', 'cut_future_action', 'cut_future_both')
PLAN = tuple((f'V{v}_A195_{arm}', v, arm) for arm in ARMS for v in (195, 198))
SELECTED_STEPS = (0, 10, 20, 29)
DENOMINATOR_FLOOR = 1e-12
SCOPE = ('CPU summaries of actual saved decoder coordinates, current50 and action16. '
         'RMS amplitude is not a firing rate, attention weight, causal importance, '
         'object probability, target decision or task success. Adjacent-boundary '
         'differences combine attention and MLP residual updates. The 4096 decoder '
         'coordinates are not the attention pre-projection heads or 12288 MLP units. '
         'Future200 hidden boundaries were not saved. Model/attention/solver/physics calls: zero.')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    with path.open('x') as stream:
        stream.write(json.dumps(data, indent=2, allow_nan=False) + '\n')


def raw(value, torch):
    return value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def exact(a, b, torch):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype and raw(a, torch) == raw(b, torch)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(exact(a[key], b[key], torch) for key in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(exact(x, y, torch) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def tensor_meta(value, torch):
    return dict(shape=list(value.shape), dtype=str(value.dtype), sha256=hashlib.sha256(raw(value, torch)).hexdigest())


def checked_load(path, expected_sha, torch):
    assert sha(path) == expected_sha, ('Saved tensor file SHA', str(path))
    return torch.load(path, map_location='cpu', weights_only=True, mmap=True)


def rms(array, axes, np):
    return np.sqrt(np.mean(np.square(array, dtype=np.float64), axis=axes, dtype=np.float64))


def relative(delta_rms, reference_rms, np):
    result = np.full_like(delta_rms, np.nan, dtype=np.float64)
    np.divide(delta_rms, reference_rms, out=result, where=reference_rms > DENOMINATOR_FLOOR)
    return result


def audit(baseline, torch):
    source = baseline / 'future-read-cuts'
    assert not (source / 'failed.json').exists()
    complete, results, provenance, protocol, sources, closure = (read(source / (name + '.json'))
        for name in ('complete', 'results', 'provenance', 'protocol', 'sources', 'closure'))
    assert complete['state'] == results['state'] == 'complete'
    assert complete['script_sha256'] == provenance['script_sha256'] == PRODUCER_SHA
    assert sha(ROOT / 'work/probe_cosmos_milk_future_read_cuts.py') == PRODUCER_SHA
    for name in ('results', 'provenance', 'protocol', 'sources', 'closure'):
        assert complete[name + '_sha256'] == sha(source / (name + '.json'))
    assert complete['fresh_q0_predictions'] == 10 and complete['fresh_model_forwards'] == 300
    assert complete['total_official_dispatch_calls'] == 36720 and complete['extra_official_dispatch_calls'] == 15120
    for name in ('two_original_native_entire_q0_records_byte_exact', 'all_masked_actual_site_unblocked_queries_exact_vs_same_input_sham',
                 'all_future200_and_UND_native_dispatch_rows_retained', 'both_engineering_action_and_boundary_equality_passed',
                 'original_noise_and_dispatch_symbols_restored', 'all_owned_hooks_removed'):
        assert complete[name] is True
    assert complete['physics_calls'] == complete['later_query_predictions'] == 0
    assert closure['state'] == 'checked' and closure['physics_executed'] is False
    assert all(row['byte_exact'] for row in closure['both_record_fields'].values())
    assert all(row['byte_exact'] for row in closure['both_boundary_checks'])
    assert all(row['byte_exact'] for items in closure['single_first_layer_checks'].values() for row in items)
    assert protocol['source_pair'] == ['V195_A195', 'V198_A195'] and protocol['action_noise_source_seed'] == 195
    assert [row['case'] for row in results['cases']] == [item[0] for item in PLAN]
    for key, filename in provenance['source_sha256'].items():
        path = Path(sources['frozen_contract']['frozen_files'][key]['path']) if key in sources['frozen_contract']['frozen_files'] else ROOT / 'work' / ('probe_cosmos_milk_target_reader_inputs.py' if key == 'reader' else 'probe_cosmos_milk_component_pulses.py')
        assert sha(path) == filename, ('Frozen source', key)
    assert sha(Path(provenance['actual_transformer_source'])) == provenance['actual_transformer_sha256']
    assert sha(Path(provenance['dispatch_source'])) == provenance['dispatch_source_sha256']
    rows, records, references, manifests = {}, {}, {}, {}
    for row, (label, v, arm) in zip(results['cases'], PLAN):
        assert row['case'] == label and row['arm'] == arm and row['vision_noise_source_seed'] == v
        assert row['action_noise_source_seed'] == row['runtime_rng_seed'] == 195 and row['fresh_model_forwards'] == 30
        assert row['all_noise_preparation_schedule_padding_and_dispatch_footprint_gates_passed'] is True
        assert row['entire_native_q0_record_byte_exact'] is (True if arm == 'native' else None)
        folder = source / label
        for name, expected in row['files_sha256'].items():
            assert sha(folder / name) == expected
        record = checked_load(folder / 'states.pt', row['files_sha256']['states.pt'], torch)
        assert record['readout'].shape == (30, 16, 4096) and record['action_velocity'].shape == (30, 16, 64)
        assert record['readout'].dtype == record['action_velocity'].dtype == torch.bfloat16
        assert record['action_states'].shape == (31, 1, 16, 64) and record['action_states'].dtype == torch.float32
        assert record['actions'].shape == (16, 10) and torch.count_nonzero(record['action_states'][..., 10:]) == 0
        assert len(record['pure_noise']) == 2
        for field in ('readout', 'action_velocity', 'action_states', 'actions'):
            assert record[field].device.type == 'cpu' and bool(torch.isfinite(record[field]).all())
        assert read(folder / 'normalized_actions.json') == record['actions'].tolist()
        metadata, report = read(folder / 'metadata.json'), read(folder / 'observer-report.json')
        assert metadata['arm'] == arm and metadata['case'] == label and metadata['seed'] == 195
        assert metadata['model_calls'] == 30 and metadata['prepare_calls'] == 1
        assert report['state'] == 'complete' and report['all_owned_hooks_removed'] is True
        assert report['counts']['model_before'] == report['counts']['model_after'] == 30
        assert report['counts']['pre_W'] == report['original_full_266_row_GEMM_calls'] == 1080
        assert len(row['boundary_files']) == 30 and [item['step'] for item in row['boundary_files']] == list(range(30))
        assert row['boundary_files'] == report['boundary_files']
        assert all(item['file'] == f"boundaries-t{item['step']:02d}.pt" for item in row['boundary_files'])
        mapping = checked_load(folder / 'indexes-and-masks.pt', row['files_sha256']['indexes-and-masks.pt'], torch)
        index = mapping['indexes']
        assert exact(index['current_rows'], torch.arange(50), torch)
        assert exact(index['action_rows'], torch.arange(250, 266), torch)
        assert index['und_len'] == 121 and index['gen_len'] == 266
        for key, expected in (('current_keys', torch.arange(121, 171)), ('future_keys', torch.arange(171, 371)),
                              ('action_keys', torch.arange(371, 387)), ('future_rows', torch.arange(50, 250)),
                              ('text_keys', torch.arange(121)), ('union_rows', torch.cat((torch.arange(50), torch.arange(250, 266))))):
            assert exact(index[key], expected, torch), ('Actual token mapping', label, key)
        assert set(mapping['masks']) == set(ARMS[1:])
        mask_metadata = {}
        for mask_arm in ARMS[1:]:
            blocked = (index['current_rows'] if mask_arm == 'cut_future_current' else index['action_rows']
                       if mask_arm == 'cut_future_action' else index['union_rows'] if mask_arm == 'cut_future_both'
                       else torch.empty(0, dtype=torch.long))
            expected_mask = torch.ones((1, 1, 266, 387), dtype=torch.bool)
            if blocked.numel():
                expected_mask[0, 0, blocked[:, None], index['future_keys'][None, :]] = False
            unblocked = torch.tensor([i for i in range(266) if i not in set(blocked.tolist())], dtype=torch.long)
            stored = mapping['masks'][mask_arm]
            assert exact(stored['mask'], expected_mask, torch)
            assert exact(stored['blocked_rows'], blocked, torch) and exact(stored['unblocked_rows'], unblocked, torch)
            mask_metadata[mask_arm] = tensor_meta(stored['mask'], torch)
        assert report['indexes_and_masks_sha256'] == row['files_sha256']['indexes-and-masks.pt']
        assert metadata['actual_masked_union66_site_substitutions'] == (0 if arm == 'native' else 1080)
        if arm == 'native':
            reference = sources['anchors']['source_pair'][str(v)]
            prior = checked_load(Path(reference['folder']) / 'states.pt', reference['files_sha256']['states.pt'], torch)
            assert exact(record, prior, torch), ('Entire native source record', label)
            references[v] = record
        rows[label], records[label] = dict(row, analysis_mask_metadata=mask_metadata), record
        manifests[label] = dict(record_file_sha256=row['files_sha256']['states.pt'],
            record_tensor_hashes_computed_here={key: tensor_meta(record[key], torch) for key in ('readout', 'action_velocity', 'action_states', 'actions')})
    for label, v, arm in PLAN:
        record = records[label]
        assert exact(record['pure_noise'], references[v]['pure_noise'], torch)
        for key in ('timesteps', 'sigmas'):
            assert exact(record[key], references[195][key], torch)
    assert exact(references[195]['pure_noise'][1], references[198]['pure_noise'][1], torch)
    for field in ('readout', 'action_velocity', 'action_states', 'actions'):
        assert exact(records['V195_A195_cut_future_both'][field], records['V198_A195_cut_future_both'][field], torch)
    return source, rows, records, manifests, dict(producer_documents_sha256={name: sha(source / (name + '.json'))
        for name in ('complete', 'results', 'provenance', 'protocol', 'sources', 'closure')},
        producer_script_sha256=PRODUCER_SHA, original_native_entire_records_independently_rechecked=True,
        producer_site_footprint_limit='Site metadata/flags and their sealed files are checked. No stored full QKV exists to independently recompute the producer\'s per-site attention gate.',
        tensor_hash_scope='Per-tensor raw-byte hashes are computed by this CPU analysis; source container hashes were already sealed by the completed producer.')


def boundary(source, row, step, record, torch, np):
    item = row['boundary_files'][step]
    data = checked_load(source / row['case'] / item['file'], item['sha256'], torch)
    assert data['step'] == step and len(data['site_events']) == 36
    output = data['model_output']
    assert type(output) is tuple and len(output) == 3 and type(output[2]) is list and len(output[2]) == 1
    assert exact(output[2][0], record['action_velocity'][step], torch)
    fields, hashes = {}, {}
    for name, tokens in (('current_hidden', 50), ('action_hidden', 16)):
        value = data[name]
        assert value.device.type == 'cpu' and value.dtype == torch.bfloat16 and value.shape == (37, tokens, 4096)
        assert list(value.shape) == item['current_shape' if name == 'current_hidden' else 'action_shape']
        assert bool(torch.isfinite(value).all())
        hashes[name] = tensor_meta(value, torch)
        fields[name] = value.float().numpy()
    for layer, event in enumerate(data['site_events']):
        assert event['step'] == step and event['layer_zero_based'] == layer and event['arm'] == row['arm']
        assert event['to_add_out_shape'] == [266, 4096] and event['to_add_out_actual_calls'] == 1
        assert event['actual_dispatch_return_flatten_equals_pre_W_byte_exact'] is True
        assert event['UND_dispatch_return_unmodified'] is True
        if row['arm'] != 'native':
            assert event['substituted_query_rows'] == 66 and event['future200_native_bytes_preserved'] is True
            assert event['mask'] == row['analysis_mask_metadata'][row['arm']]
            assert event['all_allowed_mask'] == row['analysis_mask_metadata']['all_allowed']
            if row['arm'].startswith('cut_'):
                assert event['all_unblocked_GEN_rows_vs_same_input_all_allowed_byte_exact'] is True
    return fields, dict(step=step, source_file=item['file'], source_file_sha256=item['sha256'], raw_tensor_hashes_computed_here=hashes)


def heat(ax, values, np, vmax, title, boundaries=True):
    plot = ax.imshow(np.asarray(values).T, origin='lower', aspect='auto', interpolation='nearest',
                     vmin=0, vmax=max(vmax, 1e-12), cmap='viridis')
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('Denoising index (sigma changes)', fontsize=8)
    ax.set_ylabel('Boundary 0..36' if boundaries else 'Layer update 1..36', fontsize=8)
    ax.set_xticks([0, 10, 20, 29])
    positions = [0, 12, 24, 36] if boundaries else [0, 11, 23, 35]
    ax.set_yticks(positions, positions if boundaries else [1, 12, 24, 36])
    ax.tick_params(labelsize=7)
    return plot


def plots(out, arrays, np):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = [name for name, _, _ in PLAN]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    vmax = max(float(np.max(arrays['update_rms_' + group][0])) for group in ('current', 'action'))
    for col, group in enumerate(('current', 'action')):
        image = heat(axes[col], arrays['update_rms_' + group][0], np, vmax,
                     'Current50' if group == 'current' else 'Action16', boundaries=False)
    fig.colorbar(image, ax=axes.tolist(), label='Net update RMS (decoder units)', shrink=.8)
    fig.suptitle('Native V195 / A195: net decoder-block update\nAttention + MLP combined; shared scale; not a measure of function or importance', fontsize=12)
    fig.savefig(out / 'native-net-update-overview.png', dpi=170)
    plt.close(fig)
    metrics = ((arrays['hidden_rms_current'], 'Current50 hidden RMS'), (arrays['hidden_rms_action'], 'Action16 hidden RMS'),
               (arrays['update_rms_current'], 'Current net update RMS'), (arrays['update_rms_action'], 'Action net update RMS'))
    fig, axes = plt.subplots(4, 4, figsize=(15, 12), constrained_layout=True)
    for col, (values, title) in enumerate(metrics):
        vmax = float(np.max(values))
        for row in range(4):
            image = heat(axes[row, col], values[row], np, vmax, labels[row] + '\n' + title, col < 2)
        fig.colorbar(image, ax=axes[:, col].tolist(), label='Arbitrary decoder units', shrink=.7)
    fig.suptitle('Actual native and numerical-sham amplitudes / combined attention+MLP updates\nRMS magnitude is not causal importance or object information', fontsize=13)
    fig.savefig(out / 'hidden-amplitude-and-update.png', dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(6, 2, figsize=(11, 16), constrained_layout=True)
    vmax = max(float(np.nanmax(arrays[key][4:])) for key in ('reference_relative_current', 'reference_relative_action')) * 100
    for row, case in enumerate(range(4, 10)):
        for col, group in enumerate(('current', 'action')):
            image = heat(axes[row, col], arrays['reference_relative_' + group][case] * 100, np, vmax,
                labels[case] + '\nvs SAME-V all_allowed / ' + group)
    fig.colorbar(image, ax=axes.ravel().tolist(), label='100 × RMS(delta) / RMS(same-V sham)', shrink=.65)
    fig.suptitle('Cut propagation relative to each same-V numerical sham\nLocal array distance; not task success or a causal mediation percentage', fontsize=13)
    fig.savefig(out / 'cut-vs-same-source-sham.png', dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(5, 2, figsize=(11, 14), constrained_layout=True)
    vmax = max(float(np.nanmax(arrays['cross_V_relative_' + group])) for group in ('current', 'action')) * 100
    for row, arm in enumerate(ARMS):
        for col, group in enumerate(('current', 'action')):
            image = heat(axes[row, col], arrays['cross_V_relative_' + group][row] * 100, np, vmax,
                arm + ' / ' + group + '\nV198 minus V195, SAME ARM')
    fig.colorbar(image, ax=axes.ravel().tolist(), label='100 × RMS(V198−V195) / RMS(V195 same arm)', shrink=.65)
    fig.suptitle('Cross-V propagation: native, numerical sham and cuts kept in separate panels\nA195 fixed; sigma varies across columns; future boundaries unavailable', fontsize=13)
    fig.savefig(out / 'cross-source-same-arm-propagation.png', dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(5, 2, figsize=(10, 12), constrained_layout=True)
    values = arrays['cross_V_token_relative_current'][:, 0, 1:3] * 100
    vmax = float(np.nanmax(values))
    for row, arm in enumerate(ARMS):
        for col in range(2):
            image = axes[row, col].imshow(values[row, col].reshape(5, 10), vmin=0, vmax=max(vmax, 1e-12), cmap='viridis', interpolation='nearest')
            axes[row, col].set_title(arm + f' / step0, boundary{col + 1}', fontsize=9)
            axes[row, col].set_xlabel('Packed spatial token column', fontsize=8)
            axes[row, col].set_ylabel('Packed token row', fontsize=8)
    fig.colorbar(image, ax=axes.ravel().tolist(), label='100 × token RMS(delta) / token RMS(V195 same arm)', shrink=.65)
    fig.suptitle('Where current50 coordinates differ at the first two actual layer outputs\nTrue 5×10 packing grid; not object ROI, pixel decoder, brain region or probability', fontsize=12)
    fig.savefig(out / 'current-token-grid-first-two-layers.png', dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Completed baseline experiment root')
    baseline = parser.parse_args().output.resolve()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import numpy as np
    import torch
    assert not torch.cuda.is_initialized()
    source, rows, records, manifests, source_audit = audit(baseline, torch)
    out = baseline / 'future-read-activity'
    out.mkdir(exist_ok=False)
    started, finished = time.perf_counter(), []
    try:
        write(out / 'provenance.json', dict(script_sha256=sha(Path(__file__)), source_audit=source_audit,
            scope=SCOPE, device='CPU', model_forwards=0, attention_dispatch_calls=0, solver_steps=0, physics_calls=0))
        arrays = dict(case_labels=np.asarray([name for name, _, _ in PLAN]), arms=np.asarray(ARMS),
            selected_steps=np.asarray(SELECTED_STEPS), sigmas=records[PLAN[0][0]]['sigmas'].numpy(),
            timesteps=records[PLAN[0][0]]['timesteps'].numpy())
        for group, tokens in (('current', 50), ('action', 16)):
            for key in ('hidden_rms', 'reference_delta_rms', 'reference_relative'):
                arrays[key + '_' + group] = np.zeros((10, 30, 37), dtype=np.float64)
            arrays['update_rms_' + group] = np.zeros((10, 30, 36), dtype=np.float64)
            arrays['hidden_token_rms_' + group] = np.zeros((10, 30, 37, tokens), dtype=np.float64)
            arrays['update_token_rms_' + group] = np.zeros((10, 30, 36, tokens), dtype=np.float64)
            arrays['cross_V_delta_rms_' + group] = np.zeros((5, 30, 37), dtype=np.float64)
            arrays['cross_V_relative_' + group] = np.zeros((5, 30, 37), dtype=np.float64)
            arrays['cross_V_token_relative_' + group] = np.zeros((5, 30, 37, tokens), dtype=np.float64)
            arrays['selected_token0_hidden_' + group] = np.zeros((10, 4, 37, 4096), dtype=np.float32)
            arrays['selected_channel_delta_mean_' + group] = np.zeros((10, 4, 37, 4096), dtype=np.float32)
            arrays['selected_channel_delta_rms_' + group] = np.zeros((10, 4, 37, 4096), dtype=np.float32)
        reference_labels = []
        independent_closure = []
        for case, (label, v, arm) in enumerate(PLAN):
            if arm == 'native':
                reference_label = 'V195_A195_native'
            elif arm == 'all_allowed':
                reference_label = f'V{v}_A195_native'
            else:
                reference_label = f'V{v}_A195_all_allowed'
            reference_labels.append(reference_label)
            pair_label = f'V195_A195_{arm}'
            manifest = []
            for step in range(30):
                actual, hashes = boundary(source, rows[label], step, records[label], torch, np)
                reference = actual if reference_label == label else boundary(source, rows[reference_label], step, records[reference_label], torch, np)[0]
                paired = actual if pair_label == label else boundary(source, rows[pair_label], step, records[pair_label], torch, np)[0]
                manifest.append(hashes)
                for group in ('current', 'action'):
                    value, ref, pair = (data[group + '_hidden'] for data in (actual, reference, paired))
                    amp, delta = rms(value, (1, 2), np), value.astype(np.float64) - ref
                    arrays['hidden_rms_' + group][case, step] = amp
                    arrays['hidden_token_rms_' + group][case, step] = rms(value, 2, np)
                    update = np.diff(value.astype(np.float64), axis=0)
                    arrays['update_rms_' + group][case, step] = rms(update, (1, 2), np)
                    arrays['update_token_rms_' + group][case, step] = rms(update, 2, np)
                    arrays['reference_delta_rms_' + group][case, step] = rms(delta, (1, 2), np)
                    arrays['reference_relative_' + group][case, step] = relative(rms(delta, (1, 2), np), rms(ref, (1, 2), np), np)
                    if v == 198:
                        cross = value.astype(np.float64) - pair
                        if step == 0:
                            assert value[0].tobytes(order='C') == pair[0].tobytes(order='C'), ('t0 same-arm boundary0 must be equal', arm, group)
                        arrays['cross_V_delta_rms_' + group][case // 2, step] = rms(cross, (1, 2), np)
                        arrays['cross_V_relative_' + group][case // 2, step] = relative(rms(cross, (1, 2), np), rms(pair, (1, 2), np), np)
                        arrays['cross_V_token_relative_' + group][case // 2, step] = relative(rms(cross, 2, np), rms(pair, 2, np), np)
                        if arm == 'cut_future_both':
                            passed = value.dtype == pair.dtype and value.shape == pair.shape and value.tobytes(order='C') == pair.tobytes(order='C')
                            assert passed, ('Independent Both hidden byte check', step, group)
                            independent_closure.append(dict(step=step, field=group, BF16_to_FP32_exact_value_bytes_equal=True))
                        if step == 0 and ((arm == 'cut_future_current' and group == 'current') or (arm == 'cut_future_action' and group == 'action')):
                            assert value[1].tobytes(order='C') == pair[1].tobytes(order='C')
                    if step in SELECTED_STEPS:
                        selected = SELECTED_STEPS.index(step)
                        arrays['selected_token0_hidden_' + group][case, selected] = value[:, 0]
                        arrays['selected_channel_delta_mean_' + group][case, selected] = np.mean(delta, axis=1).astype(np.float32)
                        arrays['selected_channel_delta_rms_' + group][case, selected] = rms(delta, 1, np).astype(np.float32)
            manifests[label]['boundary_tensor_hashes_computed_here'] = manifest
            finished.append(label)
            print('[FUTURE-READ-ACTIVITY] ' + json.dumps(dict(case=label, completed=len(finished), elapsed_s=time.perf_counter() - started)), flush=True)
        arrays['reference_case_labels'] = np.asarray(reference_labels)
        for key, value in arrays.items():
            if value.dtype.kind == 'f':
                assert np.isfinite(value).all(), ('Zero/invalid reference norm; refuse percentage', key)
        np.savez_compressed(out / 'activity-arrays.npz', **arrays)
        write(out / 'actual-array-manifest.json', manifests)
        plots(out, arrays, np)
        first_two = {arm: {group: {str(boundary_index): float(arrays['cross_V_relative_' + group][i, 0, boundary_index])
            for boundary_index in (0, 1, 2, 36)} for group in ('current', 'action')} for i, arm in enumerate(ARMS)}
        analysis = dict(scope=SCOPE, cases=[name for name, _, _ in PLAN], selected_steps=list(SELECTED_STEPS),
            boundary_convention='0 is actual decoder input; 1..36 are actual complete decoder-block outputs before the final norm. A difference H[b]−H[b−1] combines attention and MLP residual additions.',
            RMS_definition='sqrt(mean(H²)) across all recorded tokens and all4096 channels; per-token maps average channels only. Float64 arithmetic on lossless BF16→FP32 exports.',
            relative_definition='RMS(Hcase−Href)/RMS(Href); plot percentages multiply by100. This equals an array L2 ratio for equal shapes; not restoration, causal mediation or success.',
            reference_case_labels=reference_labels, cross_V_reference='Each arm separately compares V198−V195 with denominator V195 same arm. Native and sham contrasts are not mixed.',
            denominator_floor=DENOMINATOR_FLOOR, invalid_denominators=0,
            first_step_cross_V_ratios=first_two, independent_Both_boundary_checks=independent_closure,
            all_arms_t0_both_population_input_boundary0_bytes_equal=True,
            first_two_layer_routing_limit='At step0 the blocked population is byte equal after layer0. A difference after layer1 is consistent with the still-open other population bridge under the verified rowwise model and mask gates. These arrays do not decode object identity or show q0 target commitment.',
            token_grid='Actual time-major current-frame 5×10 packed spatial tokens, ordered by recorded current_rows0..49; not image pixels, a semantic object ROI or a named brain region.',
            channel_exports='Token0 values are actual decoder coordinates at predefined selected steps. Channel delta mean/RMS aggregate across current50 or action16 tokens. Consecutive channels are never relabeled as heads or anatomical regions.',
            arrays={key: dict(shape=list(value.shape), dtype=str(value.dtype)) for key, value in arrays.items()},
            source_audit=source_audit, model_forwards=0, attention_dispatch_calls=0, solver_steps=0, physics_calls=0,
            cuda_initialized=torch.cuda.is_initialized())
        assert not torch.cuda.is_initialized()
        write(out / 'analysis.json', analysis)
        names = ('activity-arrays.npz', 'actual-array-manifest.json', 'analysis.json', 'provenance.json',
                 'native-net-update-overview.png',
                 'hidden-amplitude-and-update.png', 'cut-vs-same-source-sham.png',
                 'cross-source-same-arm-propagation.png', 'current-token-grid-first-two-layers.png')
        write(out / 'complete.json', dict(state='complete', script_sha256=sha(Path(__file__)),
            files_sha256={name: sha(out / name) for name in names}, actual_boundary_files=300,
            source_container_hashes_verified=True, future_boundary_states_available=False,
            model_forwards=0, attention_dispatch_calls=0, solver_steps=0, physics_calls=0,
            cuda_initialized=False, scope=SCOPE))
        print('[FUTURE-READ-ACTIVITY] ' + json.dumps(dict(state='complete', elapsed_s=time.perf_counter() - started)), flush=True)
    except Exception as exc:
        write(out / 'failed.json', dict(error=str(exc), traceback=traceback.format_exc(), completed_cases=finished, scope=SCOPE))
        raise


if __name__ == '__main__':
    main()
