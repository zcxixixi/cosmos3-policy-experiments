import json,numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/dev/shm/pt-interface-phase');W=R/'web';S=Path('/dev/shm/pt-interface-read-map');r=json.load(open(S/'web/report.json'));a=np.load(S/'actions.npz');meta=json.load(open(S/'web/contexts.json'));ids=np.array([x['source_episode'] for x in meta]);ph=np.array([x['phase'] for x in meta]);groups=np.unique(ids);boot=np.random.default_rng(84161).integers(0,len(groups),(3000,len(groups)));out={};fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
def estimate(num,den,mask):
 v=np.array([[num[(ids==g)&mask].sum(),den[(ids==g)&mask].sum()] for g in groups]);bs=v[boot].sum(1);q=bs[:,0]/bs[:,1];return {'value':float(num[mask].sum()/den[mask].sum()),'ci95':np.quantile(q,[.025,.975]).tolist()}
for row,k in enumerate(['unet','transformer']):
 out[k]={};n=len(r['inventory'][k]);selected=4 if k=='unet' else 0
 for col,f in enumerate(['goal','all']):
  get=lambda name:np.stack([a[f'{k}|{f}|{s}|{name}'] for s in [3354,3355]]).astype(float)
  base=get('literal_old');target=get('target');gap=target-base;den=(gap**2).sum(axis=(0,2,3,4));nums=[den-((get('remove_'+str(i))-base)*gap).sum(axis=(0,2,3,4)) for i in range(n)];entries=[];heat=[]
  for p in range(4):
   mask=ph==p;stats=[estimate(x,den,mask) for x in nums];vals=[x['value'] for x in stats];competitor=max((i for i in range(n) if i!=selected),key=lambda i:vals[i]);entries.append({'phase':p,'sites':stats,'observed_peak_site':int(np.argmax(vals)),'previous_overall_peak':selected,'posthoc_strongest_other':competitor,'paired_peak_minus_other':estimate(nums[selected]-nums[competitor],den,mask)});heat.append(vals)
  out[k][f]=entries;ax=axs[row,col];im=ax.imshow(np.array(heat),aspect='auto',vmin=-.03,vmax=.65,cmap='YlOrRd');ax.set(title=k+' / '+f,xlabel='Read index',ylabel='Demonstration progress',xticks=range(n),yticks=range(4),yticklabels=['20%','40%','60%','80%'])
  for p in range(4):
   for i in range(n):ax.text(i,p,f'{heat[p][i]:.2f}',ha='center',va='center',fontsize=8,color='black')
fig.colorbar(im,ax=axs.ravel().tolist(),label='Removed signed natural action gap');fig.savefig(W/'phase-map.png',dpi=150)
report={'source_audit':r['audit'],'analysis':'Existing completed full-generation samples, no new GPU run. Phases are positions in successful demonstration trajectories, not diffusion timesteps. Goal/all four ±32XY changes; two seeds3354/3355. 3000 paired source-group bootstrap18groups. Signed projections are neither additive attribution nor success rates.','results':out,'limitations':'Exploratory follow-up on previously diagnosed data; strongest competitor selected posthoc separately in each phase. CIs descriptive without multiple-comparison correction; not blind confirmation. Same source groups across phases.'};(W/'report.json').write_text(json.dumps(report,indent=2));(W/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>DP接口效应：动作阶段</title><style>body{max-width:1100px;margin:40px auto;font:20px/1.6 system-ui;color:#20324c;padding:20px}img{width:100%}</style><h1>最强入口是否只在一个阶段有效？</h1><p>只移动终点：U-Net 的第4号入口、Transformer 的第0号入口，在示范的20%、40%、60%、80%四个阶段都最强。</p><img src="phase-map.png"><p>整体移动场景：Transformer 第0号仍在四个阶段最强；U-Net 的第0、1、4号入口共同影响，40%阶段第0号略高于第4号。不能把整体均值最高的入口视为所有阶段唯一关键入口。</p><p>数字是将一个入口换回原图后，沿自然动作变化方向消掉的响应比例；不是成功率，也不能逐层相加。这里的阶段是示范进度，不是扩散去噪轮次。</p><p>使用已有78个观测、18个源示范组、两个种子的完整20步生成结果，不新增训练或GPU运行。各阶段置信区间、配对比较和原始运行审计见<a href="report.json">报告</a>。这是已诊断数据的探索性分析，不是独立盲测。</p><p><a href="http://127.0.0.1:8788/pusht-interface-read-map/index.html">完整入口扫描</a></p>''')
for k in out:
 for f in out[k]:
  print(k,f,[(x['phase'],x['observed_peak_site'],x['paired_peak_minus_other']) for x in out[k][f]])
