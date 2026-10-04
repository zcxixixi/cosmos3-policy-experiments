"""Independent CPU arithmetic from the accepted X15/X18 raw records; no model."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    base = parser.parse_args().output.resolve()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import numpy as np
    import torch
    assert not torch.cuda.is_initialized()
    root = base / 'future-current-position-analysis'
    complete = json.loads((root / 'complete.json').read_text())
    assert complete['state'] == 'complete'
    for name, digest in complete['files_sha256'].items():
        assert sha(root / name) == digest
    physical = base / 'future-current-position-execution'
    producer = base / 'future-current-positions'
    summaries = {row['trial']: row for row in json.loads((physical / 'summary.json').read_text())['cases']}
    arrays = np.load(root / 'actual-arrays.npz', allow_pickle=False)
    labels = arrays['case_labels'].tolist()
    objects = arrays['object_names'].tolist()
    selections, checks, records, hidden = {}, [], {}, {}
    errors = []
    for i, label in enumerate(labels):
        rows = json.loads((physical / 'closed-loop' / label / 'trajectory.json').read_text())
        assert len(rows) == 129
        selection = []
        for j, obj in enumerate(objects):
            height = np.array([row['objects'][obj][2] for row in rows], dtype=np.float64)
            lift = height - height[0]
            contact = np.array([[row['finger_contacts'][obj][side] for side in ('left', 'right')] for row in rows], dtype=bool)
            signal = contact.all(axis=1) & (lift > .02)
            assert np.array_equal(contact, arrays['finger_contacts'][i, :, j])
            assert np.array_equal(lift, arrays['strict_lift_m'][i, :, j])
            assert np.array_equal(signal, arrays['strict_contact_and_lift'][i, :, j])
            starts = [k for k in range(125) if signal[k:k + 5].all()]
            if starts:
                selection.append((starts[0], j, obj))
            checks.append({'case': label, 'object': obj, 'first_strict_window': starts[0] if starts else None})
        selection.sort()
        selections[label] = [obj for _, _, obj in selection]
        assert selections[label] == summaries[label]['selected_objects']
        records[label] = torch.load(producer / label / 'states.pt', map_location='cpu', weights_only=False)
        hidden[label] = torch.load(producer / label / 'boundaries-t00.pt', map_location='cpu', weights_only=False)
        for key, value in (
            ('q0_action_states_FP32', records[label]['action_states'][:, 0].numpy()),
            ('q0_action_velocity_BF16_export_FP32', records[label]['action_velocity'].float().numpy()),
            ('q0_final_normalized_actions', records[label]['actions'].numpy()),
        ):
            assert arrays[key][i].dtype == value.dtype and arrays[key][i].shape == value.shape
            assert arrays[key][i].tobytes(order='C') == value.tobytes(order='C')
    for i, label in enumerate(labels):
        scene, vision, _, arm = label.split('_', 3)
        reference = scene + '_' + vision + '_A195_all_allowed'
        delta = records[label]['actions'].numpy().astype(np.float64) - records[reference]['actions'].numpy()
        for key, value in (('q0_vs_sham_final_XYZ_rms', delta[:, :3]), ('q0_vs_sham_final_all10_rms', delta)):
            errors.append(abs(float(np.sqrt(np.mean(value ** 2))) - arrays[key][i]))
        if arm != 'early_l1_9':
            continue
        for population in ('current', 'action'):
            h = hidden[label][population + '_hidden'].float().numpy().astype(np.float64)
            ref = hidden[reference][population + '_hidden'].float().numpy().astype(np.float64)
            for boundary in (0, 1, 9, 36):
                value = np.sqrt(np.mean((h[boundary] - ref[boundary]) ** 2))
                errors.append(abs(value - arrays['vs_sham_hidden_delta_rms_' + population][i, 0, boundary]))
            for layer in (0, 8, 35):
                delta_update = (h[layer + 1] - h[layer]) - (ref[layer + 1] - ref[layer])
                value = np.sqrt(np.mean(delta_update ** 2))
                errors.append(abs(value - arrays['vs_sham_net_update_delta_rms_' + population][i, 0, layer]))
    gate = all(selections[label] == selections[label.replace('_all_allowed', '_native')] for label in labels if label.endswith('_all_allowed'))
    only_milk = all(selections[label] == ['milk_1'] for label in labels if label.endswith('_early_l1_9'))
    prediction = 'supported' if gate and only_milk else 'rejected'
    assert prediction == complete['primary_prediction']
    assert max(errors) <= 1e-12 and not torch.cuda.is_initialized()
    result = dict(state='complete', script_sha256=sha(Path(__file__)), source_complete_sha256=sha(root / 'complete.json'),
        source_arrays_sha256=sha(root / 'actual-arrays.npz'), physical_conditions_recomputed=12 * 129 * 7,
        physical_cases_recomputed=12, original_PT_exports_byte_checked=36, final_action_distances_recomputed=24,
        t0_hidden_distances_recomputed=32, t0_net_update_distances_recomputed=24, max_numeric_error=float(max(errors)),
        selected_objects=selections, strict_windows=checks, sham_classification_gate=gate, all_four_only_milk=only_milk,
        primary_prediction=prediction, model_forwards=0, solver_steps=0, physics_calls=0, cuda_initialized=False,
        scope='Independent raw trajectory and selected t0 arithmetic only. Does not rerun the main analyzer, independently audit all360 QKV, establish semantics, success rate, functional specificity or root cause.')
    path = root / 'independent-audit.json'
    with path.open('x') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(json.dumps({key: result[key] for key in ('state', 'primary_prediction', 'max_numeric_error', 'physical_conditions_recomputed')}))


if __name__ == '__main__':
    main()
