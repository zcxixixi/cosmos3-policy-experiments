"""Plot recorded numerical controls; no inference or model changes."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT = Path('/home/current/work/cosmos3/outputs/official-demos/milk-posttrain/gripper-context/deep-dive')
O = ROOT/'runtime-system-prompt'
font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.size': 12, 'axes.unicode_minus': False})


def labels(ax, title, ylabel=None):
    ax.set_title(title, fontproperties=font, fontsize=15, pad=16)
    if ylabel:
        ax.set_ylabel(ylabel, fontproperties=font)
    for tick in ax.get_xticklabels()+ax.get_yticklabels():
        tick.set_fontproperties(font)


A = np.load(O/'arrays.npz')
fig, ax = plt.subplots(figsize=(9, 4.8))
x = np.arange(2)
for i, (mode, text, color) in enumerate([('on', '保留额外系统提示', '#7f8da7'), ('off', '去掉额外系统提示', '#3478c9')]):
    values = np.array([[np.sqrt(np.mean((A[f'{scene}_milk_{s}_{mode}'][:, :3].astype(float)-A[f'{scene}_butter_{s}_{mode}'][:, :3].astype(float))**2)) for s in [195, 196, 198]] for scene in ['center', 'far']])
    mean = values.mean(axis=1)
    centers = x+(i-.5)*.3
    bars = ax.bar(centers, mean, width=.28, color=color, label=text)
    for j in range(2):
        ax.scatter(centers[j]+np.array([-.06, 0, .06]), values[j], color='#22314a', s=22, zorder=3)
        ax.text(centers[j], mean[j]+.0005, f'{mean[j]:.4f}', ha='center', fontsize=11)
ax.set_xticks(x, ['原始布局', '牛奶移动 15 cm'])
labels(ax, '改“牛奶／黄油”，动作数值改变多少？', '归一化 XYZ 差异（RMSE）')
ax.legend(prop=font, frameon=False)
ax.set_ylim(0, .03)
ax.grid(axis='y', alpha=.2)
fig.text(.5, .015, '柱子是三个随机种子的平均；点是每个种子。不是厘米，也不是抓取成功率。', ha='center', fontproperties=font, fontsize=11)
fig.tight_layout(rect=(0, .07, 1, 1))
fig.savefig(O/'language_effect.png', dpi=180)
plt.close(fig)

vectors = np.load(O/'readout_vectors.npz')
rows = []
names = []
for st in [0, 29]:
    for condition, left, right in [
        ('换目标名称', 'center_milk_198_on', 'center_butter_198_on'),
        ('去掉系统提示', 'center_milk_198_on', 'center_milk_198_off'),
        ('移动牛奶 15 cm', 'center_milk_198_off', 'far_milk_198_off')]:
        a = vectors[f'{left}_step{st}_first_action_readout'].astype(float)
        b = vectors[f'{right}_step{st}_first_action_readout'].astype(float)
        # 4096 actual coordinates, grouped into128 consecutive bins for visibility.
        rows.append(np.sqrt(np.mean((b-a).reshape(128, 32)**2, axis=1))/np.sqrt(np.mean(a*a))*100)
        names.append(('第一次计算' if st == 0 else '最后一次计算')+'：'+condition)
fig, ax = plt.subplots(figsize=(12, 4.6))
im = ax.imshow(rows, aspect='auto', cmap='magma', vmin=0)
ax.set_yticks(range(6), names)
ax.set_xlabel('4096 个真实坐标，每 32 个合成一格；坐标次序未重排', fontproperties=font)
labels(ax, '同一个动作 token 的中间数组：哪些数值变了？')
bar = fig.colorbar(im, ax=ax, pad=.025)
bar.set_label('相对基准向量 RMS 的变化（%）', fontproperties=font)
fig.text(.5, .02, '格子只表示数值变化。没有给某一格贴“牛奶”“奶酪”标签，也不是行为概率。', ha='center', fontproperties=font, fontsize=11)
fig.tight_layout(rect=(0, .06, 1, 1))
fig.savefig(O/'readout_heatmap.png', dpi=180)
plt.close(fig)

I = ROOT/'identity-position'
fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
definitions = [
    ('原始牛奶位置；干扰物不变', O/'closed-loop/center_milk_off/input_00.png', '实际抓牛奶'),
    ('原始牛奶位置；交换奶酪和黄油', I/'center_cream_butter_swap_milk_off/input_00.png', '实际仍抓牛奶'),
    ('牛奶移远；干扰物不变', O/'closed-loop/far_milk_off/input_00.png', '实际抓奶酪盒'),
    ('牛奶移远；交换奶酪和黄油', I/'far_cream_butter_swap_milk_off/input_00.png', '实际抓旧位置上的黄油'),
]
for ax, (title, path, outcome) in zip(axes.flatten(), definitions):
    pixels = plt.imread(path)
    ax.imshow(pixels[:, :256])
    ax.set_title(title, fontproperties=font, fontsize=12)
    ax.set_xlabel(outcome, fontproperties=font, fontsize=13, color='#245ab6')
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle('四个场景都说“拿牛奶”，看实际抓谁', fontproperties=font, fontsize=17)
fig.text(.5, .015, '这里只显示固定相机初始画面；模型实际收到两路相机。此图对应 seed198 的执行结果。', ha='center', fontproperties=font, fontsize=10)
fig.tight_layout(rect=(0, .04, 1, .95))
fig.savefig(I/'scene_comparison.png', dpi=180)
plt.close(fig)
print('PLOTS COMPLETE')
