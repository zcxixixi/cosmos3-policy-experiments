"""Recompute paired input controls and actual target choices from saved arrays."""
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT = Path('/home/current/work/cosmos3/outputs/official-demos/milk-posttrain/gripper-context/deep-dive')
OUT = ROOT/'runtime-pixel-schedule'
assert json.loads((OUT/'probe_complete.json').read_text())['cases'] == 16
assert json.loads((OUT/'closed-loop/complete.json').read_text())['cases'] == 4
actions = np.load(OUT/'actions.npz')
readouts = np.load(OUT/'readouts_first_last.npz')
vae = np.load(OUT/'vae_arrays.npz')
recorded = json.loads((OUT/'result.json').read_text())
new = json.loads((OUT/'closed-loop/result.json').read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rmse(a, b):
    return float(np.sqrt(np.mean((a.astype(np.float64)-b.astype(np.float64))**2)))


def selection(trajectory):
    names, per_object = [], {}
    assert len(trajectory) == 129
    for name in trajectory[0]['objects']:
        xyz = np.array([row['objects'][name] for row in trajectory])
        both = np.array([name in row['grasped'] for row in trajectory], dtype=bool)
        if 'finger_contacts' in trajectory[0]:
            assert np.array_equal(both, np.array([row['finger_contacts'][name]['left']
                and row['finger_contacts'][name]['right'] for row in trajectory]))
        lift = xyz[:, 2]-xyz[0, 2]
        valid = both & (lift > .02)
        first = next((i for i in range(len(valid)-4) if np.all(valid[i:i+5])), None)
        per_object[name] = dict(first_2cm_for_5frames=first, max_lift_cm=float(lift.max()*100),
                                contact_frames=int(both.sum()))
        if first is not None:
            names.append(name)
    return names, per_object


effects = []
for row in recorded['effects']:
    case = row['case'];base = '_'.join(case.split('_')[:3])+'_legacy_legacy'
    value = rmse(actions[case][:, :3], actions[base][:, :3])
    assert abs(value-row['action_xyz']['rmse']) < 1e-8
    effects.append(dict(case=case, normalized_xyz_rmse=value,
        clean_latent_relative_rms_percent=row['clean_latent']['relative_rms_percent'],
        final_readout_relative_rms_percent=row['readout_profile'][-1]['relative_rms_percent']))
language = []
for scene in ['center', 'far']:
    for pixels, schedule in [('legacy','legacy'),('official','legacy'),('legacy','official'),('official','official')]:
        m=f'{scene}_milk_198_{pixels}_{schedule}';b=f'{scene}_butter_198_{pixels}_{schedule}'
        assert np.array_equal(vae[m+'_pixels'], vae[b+'_pixels'])
        assert np.array_equal(vae[m+'_clean'], vae[b+'_clean'])
        language.append(dict(scene=scene, pixels=pixels, schedule=schedule,
            normalized_xyz_rmse=rmse(actions[m][:,:3], actions[b][:,:3])))
choices = []
for item in new['cases']:
    scene,noun=item['scene'],item['noun'];directory=OUT/'closed-loop'/item['case']
    current=json.loads((directory/'trajectory.json').read_text())
    targets,objects=selection(current)
    assert targets==item['selected_objects']
    for name,x in objects.items():
        assert abs(x['max_lift_cm']-item['per_object'][name]['max_lift_cm'])<1e-9
    old_dir=ROOT/f'runtime-system-prompt/closed-loop/{scene}_{noun}_off'
    old=json.loads((old_dir/'trajectory.json').read_text());old_targets,old_objects=selection(old)
    contract=json.loads((directory/'query_contract.json').read_text())
    assert len(contract)==8 and all(x['pixels']==x['schedule']=='official' for x in contract)
    assert all(x['use_system_prompt'] is False and x['timesteps_match_offline_exact'] for x in contract)
    assert contract[0]['first_chunk_matches_offline_probe_exact'] is True
    choices.append(dict(scene=scene,noun=noun,previous_selected_objects=old_targets,
        corrected_selected_objects=targets,previous_per_object=old_objects,corrected_per_object=objects,
        previous_final_milk_checker=bool(old[-1]['milk_in_basket']),
        corrected_final_milk_checker=bool(current[-1]['milk_in_basket']),
        previous_trajectory_sha256=sha(old_dir/'trajectory.json'),
        corrected_trajectory_sha256=sha(directory/'trajectory.json'),
        previous_source=f'../09-system-prompt/closed-loop/{scene}_{noun}_off',
        corrected_source='closed-loop/'+item['case']))
summary=dict(scope='16 paired predictions at two fixed pictures and seed198; four new corrected closed loops, paired with existing experiment09. Not full official-server equivalence or a success-rate benchmark.',
    selection_criterion='Both-finger contact AND >2cm lift for >=5 consecutive simulation records.',
    units='Normalized action XYZ RMSE is dimensionless, not meters or probability.',
    cases=choices, effects=effects,language=language,
    prefixes=json.loads((OUT/'prefix_controls.json').read_text()),
    sources={name:sha(OUT/name) for name in ['actions.npz','readouts_first_last.npz','vae_arrays.npz','result.json']},
    helper_sha256=sha(Path(__file__)))
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
fig,ax=plt.subplots(figsize=(10,4.8))
labels=['原位／拿牛奶','原位／拿黄油','移远／拿牛奶','移远／拿黄油']
colors=['#4184af','#e59a46','#589570']
for j,(pixels,schedule,label) in enumerate([
    ('official','legacy','只改像素处理'),('legacy','official','只改时间表'),('official','official','两项都改')]):
    values=[]
    for scene,noun in [('center','milk'),('center','butter'),('far','milk'),('far','butter')]:
        values.append(rmse(actions[f'{scene}_{noun}_198_{pixels}_{schedule}'][:,:3],
                           actions[f'{scene}_{noun}_198_legacy_legacy'][:,:3]))
    ax.bar(np.arange(4)+(j-1)*.23,values,width=.22,label=label,color=colors[j])
ax.set_xticks(np.arange(4),labels,fontproperties=font)
ax.set_ylabel('最终 XYZ 动作数组与旧设置的差异（RMSE）',fontproperties=font)
ax.set_title('输入处理会改变数值；是否抓对，要另外看实际执行',fontproperties=font)
ax.legend(prop=font);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.text(.5,.015,'数值是归一化控制命令的差异，不是米、毫米或抓取概率。每组只有一个配对噪声种子。',
         ha='center',fontproperties=font,fontsize=10)
fig.tight_layout(rect=[0,.05,1,1]);fig.savefig(OUT/'action_changes.png',dpi=170);plt.close(fig)
print(json.dumps({'paired_targets':[{k:r[k] for k in ['scene','noun','previous_selected_objects','corrected_selected_objects']}
                                   for r in choices]},indent=2))
