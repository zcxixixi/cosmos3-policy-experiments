"""Recompute route effects and grasp selection from saved, completed experiments."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rmse(a, b):
    return float(np.sqrt(np.mean((a.astype(np.float64)-b.astype(np.float64))**2)))


def selection(records):
    assert len(records) == 129
    selected, objects = [], {}
    for name in records[0]['objects']:
        xyz = np.asarray([r['objects'][name] for r in records])
        contact = np.asarray([r['finger_contacts'][name]['left'] and
                              r['finger_contacts'][name]['right'] for r in records])
        assert np.array_equal(contact, [name in r['grasped'] for r in records])
        height = xyz[:, 2]-xyz[0, 2]
        valid = contact & (height > .02)
        first = next((i for i in range(len(valid)-4) if valid[i:i+5].all()), None)
        objects[name] = dict(first_contact_and_lift_5frames=first,
                            max_lift_cm=float(height.max()*100), contact_frames=int(contact.sum()))
        if first is not None:
            selected.append(name)
    return selected, objects


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    assert json.loads((out/'offline_complete.json').read_text())['patched_predictions'] == 36
    assert json.loads((out/'closed-loop/complete.json').read_text())['cases'] == 6
    assert not (out/'failed.json').exists() and not (out/'closed-loop/failed.json').exists()
    results = json.loads((out/'result.json').read_text())
    assert len(results['controls']) == 32 and all(c['exact'] for c in results['controls'])
    actions = np.load(out/'actions.npz', allow_pickle=False)
    assert len(actions.files) == 40
    effects = []
    for row in results['cases']:
        label = row['label']
        scene, source, _, target, suffix = label.split('_', 4)
        recipient = actions[f'{scene}_{source}_capture']
        donor = actions[f'{scene}_{target}_capture']
        value = actions[label]
        assert value.shape == (16, 10)
        change = rmse(value[:, :3], recipient[:, :3])
        assert abs(change-row['relative_to_recipient']['action_xyz']['rmse']) < 1e-8
        if suffix in ['empty', 'zero_gain', 'self_full', 'clamp_self', 'indirect_blocked']:
            assert np.array_equal(value, recipient)
        if suffix == 'full':
            assert np.array_equal(value, donor)
        effects.append(dict(label=label, scene=scene, source=source, target=target, mode=suffix,
            normalized_xyz_rmse=change, final_readout_relative_rms_percent=
            row['relative_to_recipient']['readout'][-1]['relative_rms_percent'],
            video_latent_exact_recipient=row['relative_to_recipient']['video']['exact']))
    choices = []
    for item in json.loads((out/'closed-loop/result.json').read_text())['cases']:
        folder = out/'closed-loop'/item['case']
        records = json.loads((folder/'trajectory.json').read_text())
        selected, objects = selection(records)
        assert selected == item['selected_objects']
        contracts = json.loads((folder/'query_contract.json').read_text())
        assert len(contracts) == 8 and contracts[0]['first_chunk_offline_exact'] is True
        for q, contract in enumerate(contracts):
            response = contract['response']
            assert response['seed'] == 198+q
            assert response['current_observation_sha256'] == sha(folder/f'input_{q:02d}.png')
            assert response['rebuilt_caches_for_current_observation']
            assert response['pixels'] == response['schedule'] == 'official'
            assert response['use_system_prompt'] is False
            assert contract['patch_report']['complete'] and contract['patch_report']['gain'] == 1
        actual = np.asarray(json.loads((folder/'executed_actions.json').read_text()))
        raw_native = np.asarray(json.loads((folder/'actions.json').read_text()))
        assert actual.shape == raw_native.shape == (128, 7)
        assert np.array_equal(actual, np.clip(raw_native, -1, 1))
        assert len(list((folder/'frames').glob('frame_*.png'))) == 129
        choices.append(dict(case=item['case'], scene=item['scene'], mode=item['mode'],
            selected_objects=selected, per_object=objects, final_milk_in_basket=records[-1]['milk_in_basket'],
            trajectory_sha256=sha(folder/'trajectory.json'),
            contracts_sha256=sha(folder/'query_contract.json')))
    summary = dict(scope='Two fixed snapshots, one paired seed198. 40 offline predictions and six 128-step route-intervention closed loops. No target semantics or benchmark rate inferred.',
        units='XYZ RMSE is a difference between normalized control arrays, not distance or probability.',
        controls=results['controls'], effects=effects, cases=choices,
        mediator_scope='Future/clean GEN visual K/V reads, with direct language-to-action reads preserved as recipient. Indirect includes feedback; no additive mediation fractions.',
        sources={name: sha(out/name) for name in ['result.json', 'actions.npz', 'readouts_first_last.npz',
            'provenance.json', 'closed-loop/result.json', 'closed-loop/provenance.json']},
        helper_sha256=sha(Path(__file__)))
    (out/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    modes = ['direct', 'indirect', 'indirect_blocked', 'joint', 'full']
    labels = ['只改直接路', '只改预测画面路', '再阻断画面回传', '两条都改', '完整换指令']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, scene, title in zip(axes, ['center', 'far'], ['牛奶原位', '牛奶移远15 cm']):
        values = [next(e['normalized_xyz_rmse'] for e in effects if e['scene'] == scene and
                       e['source'] == 'milk' and e['mode'] == mode) for mode in modes]
        bars = ax.bar(range(5), values, color=['#447cad', '#6eab86', '#aaa', '#8d72b5', '#dc9865'])
        ax.bar_label(bars, labels=[f'{v:.6f}' for v in values], fontsize=8)
        ax.set_xticks(range(5), labels, rotation=28, ha='right', fontproperties=font, fontsize=10)
        ax.set_title(title, fontproperties=font)
        ax.set_ylim(0, .028)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('最终归一化 XYZ 动作差异（RMSE）', fontproperties=font)
    fig.suptitle('改语言的传递路径，会改变动作数字；阻断画面回传后，动作差异为零', fontproperties=font)
    fig.text(.5, .01, '拿牛奶 → 拿黄油；每个画面只用一个对应噪声种子。数值不是米、毫米或抓取概率。',
             ha='center', fontproperties=font, fontsize=10)
    fig.tight_layout(rect=[0, .04, 1, .94])
    fig.savefig(out/'route_action_changes.png', dpi=170)
    plt.close(fig)
    # Each cell uses all actual 16x4096 action readout coordinates from this step.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    vmax = max(row['relative_to_recipient']['readout'][i]['relative_rms_percent']
               for row in results['cases'] for i in range(30))
    for ax, scene, title in zip(axes, ['center', 'far'], ['牛奶原位', '牛奶移远15 cm']):
        data = [[next(row for row in results['cases'] if row['label'] == f'{scene}_milk_to_butter_{mode}')
                 ['relative_to_recipient']['readout'][i]['relative_rms_percent'] for i in range(30)]
                for mode in modes]
        im = ax.imshow(data, aspect='auto', origin='upper', cmap='magma', vmin=0, vmax=vmax)
        ax.set_title(title, fontproperties=font)
        ax.set_xticks([0, 9, 19, 29], ['1', '10', '20', '30'])
        ax.set_yticks(range(5), labels, fontproperties=font)
        ax.set_xlabel('第几次反复计算（每次经过全部36层）', fontproperties=font)
    fig.suptitle('动作头前的真实数组：画面传回动作的路径被阻断后，每次差异都为零', fontproperties=font)
    fig.subplots_adjust(left=.16, right=.86, bottom=.2, top=.8, wspace=.15)
    colorbar = fig.colorbar(im, cax=fig.add_axes([.89, .23, .02, .53]))
    colorbar.set_label('数组相对变化（%）', fontproperties=font)
    fig.savefig(out/'route_readout_changes.png', dpi=170)
    plt.close(fig)
    print(json.dumps(dict(controls=32, predictions=40, choices=[dict(case=c['case'],
        selected=c['selected_objects']) for c in choices]), ensure_ascii=False))


if __name__ == '__main__':
    main()
