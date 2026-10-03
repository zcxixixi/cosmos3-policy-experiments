"""Plot four saved real pi0.5 EEF trajectories and measure horizontal proximity.

CPU only: reads actual MuJoCo records/PNGs; never imports or runs a model.
Distances describe proximity, not intention probabilities or task success.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


CASES = [('center', 'milk'), ('center', 'butter'), ('far', 'milk'), ('far', 'butter')]
OBJECTS = ['milk_1', 'butter_1', 'cream_cheese_1']
COLORS = {'milk': '#2563eb', 'butter': '#d95f02'}
NOUNS = {'milk': '牛奶', 'butter': '黄油'}
LABELS = {'milk_1': '牛奶初始位置', 'butter_1': '黄油初始位置', 'cream_cheese_1': '奶酪盒初始位置'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def distance_summary(distance_cm):
    close = np.flatnonzero(distance_cm < 4.)
    index = int(np.argmin(distance_cm))
    return dict(min_xy_distance_cm=float(distance_cm[index]), min_distance_step=index,
                first_under_4cm_step=int(close[0]) if close.size else None,
                distance_cm_by_simulator_step=distance_cm.tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output_dir.exists(), 'Refuse overwriting an analysis directory'
    assert json.loads((args.input_dir/'complete.json').read_text())['state'] == 'complete'
    font = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')
    assert font.is_file(), 'Use an existing CJK font; do not download a font'
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=str(font)).get_name(),
                         'font.size': 10, 'axes.unicode_minus': False})
    args.output_dir.mkdir(parents=True)
    records, arrays, reports = {}, {}, {}
    for scene, noun in CASES:
        name = f'{scene}_{noun}_seed198'
        run = args.input_dir/name
        rows = json.loads((run/'trajectory.json').read_text())
        assert len(rows) == 129 and [row['step'] for row in rows] == list(range(129))
        eef = np.asarray([row['eef_xyz'] for row in rows], dtype=np.float64)
        assert eef.shape == (129, 3) and np.isfinite(eef).all()
        objects = {obj: np.asarray([row['objects'][obj] for row in rows]) for obj in OBJECTS}
        records[name], arrays[name] = rows, (eef, objects)
        metrics = {}
        for obj, xyz in objects.items():
            current = np.linalg.norm(eef[:, :2]-xyz[:, :2], axis=1)*100.
            initial = np.linalg.norm(eef[:, :2]-xyz[0, :2], axis=1)*100.
            metrics[obj] = dict(to_object_at_same_step=distance_summary(current),
                                to_object_initial_position=distance_summary(initial))
        reports[name] = dict(scene=scene, instruction_object=noun, simulator_steps=128,
            actual_records=129, eef_xy_cm=(eef[:, :2]*100.).tolist(),
            initial_object_positions_m={obj: xyz[0].tolist() for obj, xyz in objects.items()},
            objects=metrics, source_trajectory=str(run/'trajectory.json'),
            source_trajectory_sha256=digest(run/'trajectory.json'),
            source_complete_sha256=digest(run/'complete.json'))
    old_milk_xy = arrays['center_milk_seed198'][1]['milk_1'][0, :2]
    for noun in ['milk', 'butter']:
        name = f'far_{noun}_seed198'
        eef = arrays[name][0]
        reports[name]['to_empty_old_milk_position'] = distance_summary(
            np.linalg.norm(eef[:, :2]-old_milk_xy, axis=1)*100.)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 7.5), sharex=True, sharey=True)
    for ax, scene, title in zip(axes, ['center', 'far'], ['原布局', '只把牛奶沿 X 移远 15 cm']):
        for noun in ['milk', 'butter']:
            name = f'{scene}_{noun}_seed198'
            xy = arrays[name][0][:, :2]*100.
            ax.plot(xy[:, 0], xy[:, 1], lw=2.2, alpha=.9, color=COLORS[noun], label=f'指令：拿{NOUNS[noun]}')
            ax.scatter(xy[::20, 0], xy[::20, 1], s=15, color=COLORS[noun], zorder=3)
            ax.scatter(*xy[-1], marker='X', s=55, color=COLORS[noun], zorder=5)
            target_step = reports[name]['objects'][noun+'_1']['to_object_at_same_step']['min_distance_step']
            ax.scatter(*xy[target_step], s=70, facecolors='none', edgecolors=COLORS[noun], linewidth=1.5, zorder=6)
        start = arrays[f'{scene}_milk_seed198'][0][0, :2]*100.
        ax.scatter(*start, c='#111827', s=55, marker='s', label='机械臂起点', zorder=7)
        initial = arrays[f'{scene}_milk_seed198'][1]
        for obj, marker, color in [('milk_1', 'o', '#7b3294'), ('butter_1', 'D', '#228833'),
                                   ('cream_cheese_1', 'P', '#666666')]:
            xy = initial[obj][0, :2]*100.
            ax.scatter(*xy, s=95, marker=marker, color=color, edgecolor='white', zorder=8)
            ax.annotate(LABELS[obj], xy, xytext=(5, 7), textcoords='offset points', fontsize=9, color=color)
        if scene == 'far':
            xy = old_milk_xy*100.
            ax.scatter(*xy, s=95, marker='o', facecolors='none', edgecolors='#7b3294', linewidth=1.5, zorder=8)
            ax.annotate('空的旧牛奶位置', xy, xytext=(-8, -17), textcoords='offset points', fontsize=9,
                        color='#7b3294', ha='right')
        ax.set_title(title, fontsize=13)
        ax.set_xlabel('实际 EEF 的 X 坐标（cm）')
        ax.set_ylabel('实际 EEF 的 Y 坐标（cm）')
        ax.grid(alpha=.22)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-19, 17)
        ax.set_ylim(-31, 32)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .94), ncol=3, frameon=False)
    fig.suptitle('实际执行轨迹：语言改变后，手臂有没有朝不同物体靠近？', fontsize=15, y=.99)
    fig.text(.5, .035, '线＝真实 MuJoCo EEF 水平轨迹；圆圈＝离指令物体最近的那一步；X＝最后一步。\n'
             '只看 XY 接近程度，不代表已经抓起，也不是“意图概率”。物体标记为初始位置。',
             ha='center', fontsize=10)
    fig.subplots_adjust(top=.86, bottom=.13, wspace=.22)
    fig.savefig(args.output_dir/'actual_eef_xy.png', dpi=180, bbox_inches='tight')
    fig.savefig(args.output_dir/'actual_eef_xy.svg', bbox_inches='tight')
    plt.close(fig)

    # Show existing simulator frames at start, closest approach to instructed object, and end.
    fig, axes = plt.subplots(4, 3, figsize=(15.5, 10.5))
    frames = []
    for row, (scene, noun) in enumerate(CASES):
        name = f'{scene}_{noun}_seed198'
        nearest = reports[name]['objects'][noun+'_1']['to_object_at_same_step']['min_distance_step']
        steps = [0, nearest, 128]
        for col, (step, label) in enumerate(zip(steps, ['起点', '离指令物体最近', '结束'])):
            path = args.input_dir/name/'frames'/f'frame_{step:04d}.png'
            pixels = np.asarray(Image.open(path).convert('RGB'))
            assert pixels.shape == (256, 512, 3)
            axes[row, col].imshow(pixels)
            axes[row, col].set_title(f'{"原布局" if scene == "center" else "牛奶移远"} / 拿{NOUNS[noun]} / {label} / step {step}', fontsize=10)
            axes[row, col].axis('off')
            frames.append(dict(case=name, step=step, reason=label, path=str(path), sha256=digest(path),
                               eef_xyz_m=records[name][step]['eef_xyz']))
    fig.suptitle('真实执行关键帧（左：固定相机，右：腕部相机；没有生成视频）', fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(args.output_dir/'actual_keyframes.png', dpi=140, bbox_inches='tight')
    plt.close(fig)
    report = dict(scope='Four fixed-state real MuJoCo executions, not a success rate or intention probability.',
        script_path=str(Path(__file__).resolve()), script_sha256=digest(__file__),
        source_complete_sha256=digest(args.input_dir/'complete.json'),
        formulas=dict(to_object_at_same_step='100 * sqrt((eef_x[t]-object_x[t])**2 + (eef_y[t]-object_y[t])**2)',
                      to_object_initial_position='100 * sqrt((eef_x[t]-object_x[0])**2 + (eef_y[t]-object_y[0])**2)',
                      threshold='first simulator record t with XY distance strictly < 4 cm; start record t=0 included',
                      ignored='Z separation, object extent, finger opening and contact; proximity alone is not a grasp.'),
        old_milk_position_m=old_milk_xy.tolist(), cases=reports, keyframes=frames)
    (args.output_dir/'metrics.json').write_text(json.dumps(report, indent=2)+'\n')
    complete = dict(state='complete', script_sha256=digest(__file__),
        files={path.name: digest(path) for path in sorted(args.output_dir.iterdir()) if path.is_file()})
    (args.output_dir/'complete.json').write_text(json.dumps(complete, indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output_dir), script_sha256=complete['script_sha256'],
                         files=complete['files']), indent=2))


if __name__ == '__main__':
    main()
