"""Audit and plot nine completed milk-shift simulator trajectories on CPU.

XY distances describe approach geometry; grasp identity is determined only by
the simultaneous double-finger contact and sustained-lift records. All source
data remain unchanged; trajectory-analysis must be a new directory.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


POSITIONS = ('x00', 'x06', 'x15')
SEEDS = (195, 196, 198)
TARGETS = ('milk_1', 'cream_cheese_1')
TRIALS = [f'{position}_seed{seed}' for position in POSITIONS for seed in SEEDS]
SCOPE = ('这批分析只描述机械臂在接触前实际朝哪里移动，以及接触、抬升的先后顺序。'
         'XY距离是二维投影，EEF高度是末端参考点的高度；两者都不能单独识别抓取目标，'
         '也不能排除夹爪朝向、指尖位置、路径阻挡或接触条件造成的失败。'
         '抓起谁仅依据同一物体连续五帧同时双指接触且抬升超过2厘米；'
         '此分析不定位网络内部原因。')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def read_json(path):
    return json.loads(path.read_text())


def audit_numbers(value, label):
    """Check every actual numeric leaf, including all observations and qvel."""
    if isinstance(value, dict):
        return sum(audit_numbers(item, f'{label}/{key}') for key, item in value.items())
    if isinstance(value, list):
        return sum(audit_numbers(item, f'{label}/{index}') for index, item in enumerate(value))
    if isinstance(value, (int, float)):
        assert math.isfinite(value), (label, value)
        return 1
    assert value is None or isinstance(value, str), (label, type(value))
    return 0


def first_true(mask):
    indexes = np.flatnonzero(mask)
    return int(indexes[0]) if indexes.size else None


def first_window(mask):
    return next((index for index in range(len(mask) - 4) if np.all(mask[index:index + 5])), None)


def approach_window(eef, objects, distances, stop):
    result = dict(first_step=0, last_step=int(stop), samples=int(stop + 1),
        actual_eef_displacement_xyz_cm=((eef[stop] - eef[0]) * 100).tolist(),
        actual_eef_xy_path_length_cm=float(np.linalg.norm(np.diff(eef[:stop + 1, :2], axis=0), axis=1).sum() * 100),
        eef_reference_height_cm=dict(initial=float(eef[0, 2] * 100), final=float(eef[stop, 2] * 100),
                                     minimum=float(eef[:stop + 1, 2].min() * 100), maximum=float(eef[:stop + 1, 2].max() * 100)),
        object_xy_distance_cm={}, xy_distance_shortened_to=[])
    for name in TARGETS:
        values = distances[name][:stop + 1] * 100
        closest = int(values.argmin())
        reduction = float(values[0] - values[-1])
        delta = eef[stop, :2] - eef[0, :2]
        direction = objects[name][0, :2] - eef[0, :2]
        denominator = float(np.linalg.norm(delta) * np.linalg.norm(direction))
        result['object_xy_distance_cm'][name] = dict(initial=float(values[0]), final=float(values[-1]),
            reduction=reduction, minimum=float(values[closest]), minimum_step=closest,
            eef_height_cm_at_minimum=float(eef[closest, 2] * 100),
            displacement_alignment_to_initial_object=None if denominator == 0 else float(np.clip(np.dot(delta, direction) / denominator, -1, 1)))
        if reduction > 1e-7:
            result['xy_distance_shortened_to'].append(name)
    result['interpretation'] = '距离缩短对象可能同时有多个；这个列表不作为目标选择或抓取分类。'
    return result


def load_trial(root, trial, expected_summary):
    run = root / 'closed-loop' / trial
    records = read_json(run / 'trajectory.json')
    summary = read_json(run / 'summary.json')
    assert summary == expected_summary and summary['trial'] == trial
    assert summary['steps'] == 128 and len(records) == 129
    assert [record['step'] for record in records] == list(range(129))
    count = audit_numbers(records, trial + '/trajectory')
    states = np.load(run / 'sim_states.npy', allow_pickle=False)
    assert states.ndim == 2 and states.shape[0] == 129 and np.isfinite(states).all(), trial
    arrays, hashes = {}, {}
    for filename, width in (('normalized_actions.json', 10), ('denormalized_actions.json', 10),
                            ('actions.json', 7), ('executed_actions.json', 7)):
        arrays[filename] = np.asarray(read_json(run / filename), dtype=np.float64)
        assert arrays[filename].shape == (128, width) and np.isfinite(arrays[filename]).all(), (trial, filename)
        hashes[filename] = digest(run / filename)
    assert np.array_equal(np.clip(arrays['actions.json'], -1, 1), arrays['executed_actions.json'])
    assert len(read_json(run / 'query_contract.json')) == 8
    eef = np.asarray([record['eef_xyz'] for record in records], dtype=np.float64)
    assert eef.shape == (129, 3) and np.isfinite(eef).all()
    names = list(records[0]['objects'])
    assert set(TARGETS) <= set(names)
    objects, contact_masks, events, qvel_audit = {}, {}, {}, {}
    for name in names:
        objects[name] = np.asarray([record['objects'][name] for record in records], dtype=np.float64)
        qpos = np.asarray([record['object_qpos'][name] for record in records], dtype=np.float64)
        qvel = np.asarray([record['object_qvel'][name] for record in records], dtype=np.float64)
        assert qpos.shape == (129, 7) and qvel.shape == (129, 6), (trial, name)
        assert np.isfinite(qpos).all() and np.isfinite(qvel).all()
        assert np.array_equal(qpos[:, :3], objects[name])
        qvel_audit[name] = dict(shape=list(qvel.shape), all_finite=True, max_abs=float(np.abs(qvel).max()))
        left = np.asarray([record['finger_contacts'][name]['left'] for record in records], dtype=bool)
        right = np.asarray([record['finger_contacts'][name]['right'] for record in records], dtype=bool)
        both = np.asarray([record['finger_contacts'][name]['both'] for record in records], dtype=bool)
        assert np.array_equal(left & right, both), (trial, name)
        lift = objects[name][:, 2] - objects[name][0, 2]
        strict = both & (lift > .02)
        window = first_window(strict)
        events[name] = dict(first_left_step=first_true(left), first_right_step=first_true(right),
            first_any_finger_step=first_true(left | right), first_both_fingers_step=first_true(both),
            strict_window_start_step=window, strict_confirmed_step=None if window is None else window + 4,
            first_both_eef_xyz=None if not both.any() else eef[first_true(both)].tolist(),
            strict_start_eef_xyz=None if window is None else eef[window].tolist(), max_lift_cm=float(lift.max() * 100))
        assert summary['per_object'][name]['first_contact_frame'] == events[name]['first_both_fingers_step']
        assert summary['per_object'][name]['first_2cm_for_5frames'] == window
        contact_masks[name] = dict(left=left, right=right, both=both, lift_m=lift, strict=strict, qvel=qvel)
    selections = sorted((item['strict_window_start_step'], name) for name, item in events.items()
                        if item['strict_window_start_step'] is not None)
    first = selections[0] if selections else None
    assert sorted(summary['selected_objects']) == sorted(name for _, name in selections)
    assert summary['first_selected_object'] == (None if first is None else first[1])
    assert summary['first_selection_step'] == (None if first is None else first[0])
    finger_first = min((item['first_any_finger_step'] for item in events.values()
                        if item['first_any_finger_step'] is not None), default=None)
    robot_first = next((record['step'] for record in records if any(
        {contact['owner1'][0], contact['owner2'][0]} == {'object', 'robot'}
        for contact in record['all_contacts'])), None)
    physical_first = min((step for step in (finger_first, robot_first) if step is not None), default=None)
    precontact_stop = 128 if physical_first is None else max(0, physical_first - 1)
    distances = {name: np.linalg.norm(eef[:, :2] - objects[name][:, :2], axis=1) for name in TARGETS}
    first_strict_step = None if first is None else first[0]
    pattern_stop = 128 if first is None else first[0]
    both_sequence = [dict(step=step, objects=[name for name in names if contact_masks[name]['both'][step]])
                     for step in range(pattern_stop + 1)
                     if step == 0 or any(contact_masks[name]['both'][step] != contact_masks[name]['both'][step - 1] for name in names)]
    event_order = sorted((item['first_both_fingers_step'], name) for name, item in events.items()
                         if item['first_both_fingers_step'] is not None and item['first_both_fingers_step'] <= pattern_stop)
    q0_commands = arrays['executed_actions.json'][:16]
    metrics = dict(trial=trial, shift_cm=summary['shift_cm'], seed=summary['noise_seed_base'],
        first_strict_object=None if first is None else first[1], first_strict_step=first_strict_step,
        first_strict_confirmed_step=None if first is None else first[0] + 4,
        selected_objects=summary['selected_objects'], first_contact_by_object=events,
        first_any_finger_contact_step=finger_first, first_robot_object_contact_step=robot_first,
        first_physical_contact_step=physical_first,
        q0_contains_physical_contact=physical_first is not None and physical_first <= 16,
        q0_actual_approach=approach_window(eef, objects, distances, 16),
        q0_executed_command_summary=dict(units='Dimensionless controller inputs; these numbers are not displacement in meters.',
            mean_xyz_command=q0_commands[:, :3].mean(axis=0).tolist(),
            min_xyz_command=q0_commands[:, :3].min(axis=0).tolist(), max_xyz_command=q0_commands[:, :3].max(axis=0).tolist(),
            gripper_first=float(q0_commands[0, 6]), gripper_last=float(q0_commands[-1, 6]),
            clipped_command_components=int(np.count_nonzero(arrays['actions.json'][:16] != q0_commands))),
        before_first_physical_contact=approach_window(eef, objects, distances, precontact_stop),
        contact_before_strict_pattern=dict(last_step_in_pattern=pattern_stop,
            first_both_object_order=[dict(step=step, object=name) for step, name in event_order],
            double_finger_state_changes=both_sequence,
            other_objects_with_any_finger_before_first_strict=[name for name, event in events.items()
                if first is not None and name != first[1] and event['first_any_finger_step'] is not None
                and event['first_any_finger_step'] < first[0]],
            other_objects_with_both_fingers_before_first_strict=[name for name, event in events.items()
                if first is not None and name != first[1] and event['first_both_fingers_step'] is not None
                and event['first_both_fingers_step'] < first[0]]),
        data_audit=dict(trajectory_numeric_leaves_checked=count, all_observations_numeric_finite=True,
            sim_states_shape=list(states.shape), sim_states_all_finite=True,
            full_recorded_object_qvel=qvel_audit, all_action_arrays_finite=True,
            executed_equals_clipped_native=True, event_summary_matches_exact=True,
            source_sha256=dict(trajectory=digest(run / 'trajectory.json'), sim_states=digest(run / 'sim_states.npy'),
                               summary=digest(run / 'summary.json'), **hashes)), scope=SCOPE)
    return metrics, dict(eef=eef, objects=objects, distances=distances, events=events,
                         contact_masks=contact_masks, precontact_stop=precontact_stop,
                         physical_first=physical_first, actions=arrays)


def plot_paths(destination, data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = {195: '#1976b8', 196: '#d97510', 198: '#13856d'}
    all_xy = np.concatenate([item['eef'][:, :2] for item in data.values()]
                            + [item['objects'][name][:1, :2] for item in data.values() for name in TARGETS]) * 100
    lo, hi = all_xy.min(axis=0) - 2, all_xy.max(axis=0) + 2
    figure, axes = plt.subplots(1, 3, figsize=(15, 6), sharex=True, sharey=True)
    abbreviations = {'milk_1': 'M', 'cream_cheese_1': 'C', 'butter_1': 'B'}
    for position, axis in zip(POSITIONS, axes):
        reference = data[f'{position}_seed195']
        for name, marker, color in (('milk_1', '*', '#151b24'), ('cream_cheese_1', 's', '#8249a9')):
            for seed in SEEDS:
                assert np.array_equal(data[f'{position}_seed{seed}']['objects'][name][0], reference['objects'][name][0])
            x, y = reference['objects'][name][0, :2] * 100
            axis.scatter(x, y, marker=marker, s=130 if marker == '*' else 65, color=color, zorder=8)
            axis.annotate('milk' if name == 'milk_1' else 'cream cheese', (x, y), xytext=(5, -12), textcoords='offset points', fontsize=8)
        for seed in SEEDS:
            item = data[f'{position}_seed{seed}']
            xy = item['eef'][:, :2] * 100
            axis.plot(xy[:, 0], xy[:, 1], color=colors[seed], linewidth=1.7, alpha=.9)
            axis.scatter(*xy[0], color=colors[seed], s=18, marker='o', zorder=7)
            for start in (8, 24, 48, 80, 112):
                axis.annotate('', xy=xy[start + 4], xytext=xy[start],
                              arrowprops=dict(arrowstyle='->', color=colors[seed], lw=1.6))
                if seed == 198 and start in (8, 48, 112):
                    axis.annotate(f's{start}', xy[start], xytext=(3, 5), textcoords='offset points', fontsize=7, color=colors[seed])
            for name, event in item['events'].items():
                tag = abbreviations.get(name, name.replace('_1', '')[:2])
                both, strict = event['first_both_fingers_step'], event['strict_window_start_step']
                if both is not None:
                    axis.scatter(*xy[both], marker='x', color=colors[seed], s=52, linewidths=1.6, zorder=9)
                    axis.annotate(f'{tag}{both}', xy[both], xytext=(3, -9 - (seed - 195) * 3), textcoords='offset points', fontsize=7, color=colors[seed])
                if strict is not None:
                    axis.scatter(*xy[strict], marker='D', color=colors[seed], edgecolor='white', s=47, zorder=10)
                    axis.annotate(f'{tag}{strict}', xy[strict], xytext=(3, 6 + (seed - 195) * 3), textcoords='offset points', fontsize=7, color=colors[seed])
        axis.set(title=f'Milk X +{int(position[1:])} cm', xlabel='World X (cm)', xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]))
        axis.set_aspect('equal', adjustable='box')
        axis.grid(alpha=.2)
    axes[0].set_ylabel('World Y (cm)')
    legend = [Line2D([], [], color=colors[seed], label=f'seed {seed}') for seed in SEEDS]
    legend += [Line2D([], [], marker='*', color='#151b24', linestyle='', markersize=10, label='initial milk'),
               Line2D([], [], marker='s', color='#8249a9', linestyle='', label='initial cream cheese'),
               Line2D([], [], marker='x', color='#444444', linestyle='', label='first simultaneous double-finger contact'),
               Line2D([], [], marker='D', color='#444444', linestyle='', label='sustained-lift window starts')]
    figure.legend(handles=legend, loc='lower center', ncol=4, fontsize=8)
    figure.suptitle('Actual EEF XY paths; arrows point toward later control steps\nM=milk, C=cream cheese; marker numbers are control steps; XY projection does not identify a grasp', fontsize=11)
    figure.tight_layout(rect=(0, .10, 1, .90))
    figure.savefig(destination / 'actual_eef_xy.png', dpi=170)
    plt.close(figure)


def plot_distances(destination, data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 3, figsize=(13, 10), sharey=True)
    for row, position in enumerate(POSITIONS):
        for col, seed in enumerate(SEEDS):
            item = data[f'{position}_seed{seed}']
            stop = item['precontact_stop']
            steps = np.arange(stop + 1)
            axis = axes[row, col]
            axis.axvspan(0, min(16, stop), color='#9daab5', alpha=.15, label='query 0 (steps 0-16)')
            for name, style, color in (('milk_1', '-', '#1976b8'), ('cream_cheese_1', '--', '#8249a9')):
                values = item['distances'][name] * 100
                axis.plot(steps, values[:stop + 1], style, color=color,
                          label='milk' if name == 'milk_1' else 'cream cheese', linewidth=1.8)
                if stop >= 16:
                    axis.scatter(16, values[16], marker='o', color=color, s=25)
            if item['physical_first'] is not None:
                axis.axvline(item['physical_first'], color='#555555', linestyle=':', linewidth=1,
                             label='first physical contact (excluded)')
            axis.set(title=f'X +{int(position[1:])} cm | seed {seed}', xlabel='Control step',
                     xlim=(0, max(17, stop + 2)))
            if col == 0:
                axis.set_ylabel('EEF to current object centre XY distance (cm)')
            axis.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='lower center', ncol=4, fontsize=9)
    figure.suptitle('Measured XY distances before any recorded robot/object or finger/object contact\nShaded = first action chunk; contact itself is excluded. Height, orientation and finger geometry remain separate.', fontsize=11)
    figure.tight_layout(rect=(0, .05, 1, .93))
    figure.savefig(destination / 'precontact_xy_distances.png', dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Existing completed baseline experiment root')
    args = parser.parse_args()
    root = args.output
    destination = root / 'trajectory-analysis'
    assert not destination.exists(), f'Preserve existing analysis: {destination}'
    complete = read_json(root / 'closed-loop/complete.json')
    batch = read_json(root / 'closed-loop/summary.json')
    assert complete['state'] == 'complete' and complete['cases'] == 9
    assert set(complete['completed_trials']) == set(TRIALS)
    assert len(batch['cases']) == 9 and {item['trial'] for item in batch['cases']} == set(TRIALS)
    assert not (root / 'closed-loop/failed.json').exists()
    summaries = {item['trial']: item for item in batch['cases']}
    metrics, data, derived_arrays = [], {}, {}
    for trial in TRIALS:
        item, arrays = load_trial(root, trial, summaries[trial])
        metrics.append(item)
        data[trial] = arrays
        derived_arrays[trial + '__actual_eef_xyz_m'] = arrays['eef']
        for name in TARGETS:
            derived_arrays[trial + '__' + name + '__xyz_m'] = arrays['objects'][name]
            derived_arrays[trial + '__' + name + '__xy_distance_m'] = arrays['distances'][name]
        for name, masks in arrays['contact_masks'].items():
            for field, values in masks.items():
                derived_arrays[trial + '__' + name + '__' + field] = values
    destination.mkdir(exist_ok=False)
    write_json(destination / 'metrics.json', dict(cases=metrics, scope=SCOPE,
        coordinate_units='Inputs meters; reported distances/heights/displacements centimeters.',
        event_time_rule='Strict start is the first frame of a 5-frame valid window; confirmation occurs four control steps later.',
        precontact_rule='Exclude first contact frame; earliest observed robot/object or any-finger/object contact.',
        script_sha256=digest(Path(__file__)), source_summary_sha256=digest(root / 'closed-loop/summary.json')))
    np.savez_compressed(destination / 'trajectory_arrays.npz', **derived_arrays)
    plot_paths(destination, data)
    plot_distances(destination, data)
    write_json(destination / 'complete.json', dict(state='complete', cases=9, figures=2,
        all_recorded_numeric_values_finite=True, strict_events_match_simulator_summaries=True,
        metrics_sha256=digest(destination / 'metrics.json'), arrays_sha256=digest(destination / 'trajectory_arrays.npz'),
        figures_sha256={name: digest(destination / name) for name in ('actual_eef_xy.png', 'precontact_xy_distances.png')}, scope=SCOPE))
    print(json.dumps(dict(state='complete', output=str(destination), cases=9, scope=SCOPE), ensure_ascii=False))


if __name__ == '__main__':
    main()
