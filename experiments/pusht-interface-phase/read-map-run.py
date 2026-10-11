import ast
from pathlib import Path
nodes=[]
for node in ast.parse(Path('/home/current/work/pusht-physical-history/run.py').read_text()).body:
 if isinstance(node,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='st' for x in node.targets):break
 nodes.append(node)
exec(compile(ast.Module(body=nodes,type_ignores=[]),'setup','exec'),globals())
images=np.load('/home/current/work/pusht-relative-position/images.npz')['images'];meta=json.load(open('/home/current/work/pusht-relative-position/web/contexts.json'));pr=json.load(open('/home/current/work/pusht-relative-position/web/protocol.json'));variants=pr['variants'];offsets=np.array(pr['offsets_px']);arrays={};inventory={};calls=0;sampler_inputs=0;audits={};hook_calls=0
protocol={'hypothesis':'Only some image-conditioning reads may mediate goal/all-layout changes. Layerwise remove and single-read sufficiency may differ and effects need not add.','data':'78boundary-screened observations18source groups, four phases; goal±32XY andall-layout±32XY; frozenfinetunedUnetandTransformer.','intervention':'Each denoising call atoneconditioningread substitutes originalimage condition, other reads usechangedimage. Oppositesufficiency setsallreadsto originalexceptone. U-Net swapslast256of512(time+condition) beforeMish/LinearFiLMencoder; timeunchanged. Transformer swapsmemory argumentofoneDecoderLayer, queriesandselfattention untouched. Oldimageisreference,notnullablation.','references':'target readsallchanged; literal_oldreadsoriginaldirectly; all_oldviahooks mustmatchliteralold<0.002px; originalbaselineusesuntranslatedinitialnoise. Goal uses sameinitialnoise; alllayout usescoupledinitialnoise sqrtalpha95*delta/256.','metric':'Recoverednaturalimageeffect = dot(patch-old,target-old)/||target-old||²; target1,old0; can exceed1ornegative. Relative action reconstruction error and idealtranslationerror separate. No semanticor tasksuccessfraction.','test':'20DDIMfuture8XYcommands, newseeds3354/3355, allsitemasks predeclared; no posttest layerselection, allshown.','limitations':'Fullgenerationcross-contextintervention maybeoffdistribution. No prune/retrain/physics/taskrepair. Depthprofilesdependoncontrast/checkpoint/data; cannosinglelayerdominancegeneralize to VLA. Both modelsread sameCNN4tokensatdifferentblocks, notdifferentCNNdepths.'};(W/'protocol.json').write_text(json.dumps(protocol,indent=2));(W/'contexts.json').write_text(json.dumps(meta,indent=2))
for kind in ['unet','transformer']:
 ck=Path('/home/current/work/pusht-wide-finetune')/f'{kind}-finetuned.pt';sha=hashlib.sha256(ck.read_bytes()).hexdigest();m=Policy(kind).cuda().eval().requires_grad_(False);m.load_state_dict(torch.load(ck,map_location='cuda',weights_only=False)['model']);flat=images.reshape(-1,2,96,96,3);hs=[]
 with torch.no_grad():
  for lo in range(0,len(flat),64):hs.append(m.encode(torch.tensor(flat[lo:lo+64].transpose(0,1,4,2,3).copy(),device='cuda')))
 H=torch.cat(hs).reshape(len(meta),len(variants),72,256)
 if kind=='unet':
  names=[f'down_modules.{i}.{j}.cond_encoder' for i in range(3) for j in range(2)]+[f'mid_modules.{i}.cond_encoder' for i in range(2)]+[f'up_modules.{i}.{j}.cond_encoder' for i in range(2) for j in range(2)];mods=dict(m.unet.net.named_modules());sites=[mods[n] for n in names]
 else:names=[f'blocks.{i}' for i in range(6)];sites=list(m.unet.blocks)
 inventory[kind]=names;positive=0.;checked=0
 for factor in ['goal','all']:
  indexes=[variants.index(f'{factor}_{i}') for i in range(4)];hn=H[:,indexes].reshape(-1,72,256);ho=H[:,:1].expand(-1,4,-1,-1).reshape(-1,72,256)
  if kind=='unet':
   with torch.no_grad():cn=m.unet.pool(hn);co=m.unet.pool(ho)
  else:cn=hn;co=ho
  modes=['target','literal_old','all_old','original']+[f'remove_{i}' for i in range(len(names))]+[f'keep_{i}' for i in range(len(names))]
  for seed in [3354,3355]:
   for mode in modes:
    useold=set()
    if mode=='all_old':useold=set(range(len(names)))
    elif mode.startswith('remove_'):useold={int(mode.split('_')[1])}
    elif mode.startswith('keep_'):useold=set(range(len(names)))-{int(mode.split('_')[1])}
    context={};handles=[]
    def callback(module,args):
     global hook_calls
     hook_calls+=1
     if kind=='unet':
      inp=args[0];assert inp.shape[1]==512;assert torch.equal(inp[:,-256:],context['new']);return (torch.cat([inp[:,:256],context['old']],1),)
     assert torch.equal(args[1],context['new']);return (args[0],context['old'])+args[2:]
    for i in useold:handles.append(sites[i].register_forward_pre_hook(callback))
    out=[];sampler_inputs+=len(cn)
    with torch.no_grad():
     for lo in range(0,len(cn),64):
      new=cn[lo:lo+64];old=co[lo:lo+64];context.update(new=new,old=old);sch=DDIMScheduler(num_train_timesteps=100,beta_schedule='squaredcos_cap_v2',clip_sample=True,prediction_type='epsilon');sch.set_timesteps(20);a=torch.randn(1,16,2,device='cuda',generator=torch.Generator(device='cuda').manual_seed(seed)).expand(len(new),-1,-1).clone()
      if factor=='all' and mode!='original':a+=sch.alphas_cumprod[95].sqrt().to('cuda')*torch.tensor(np.tile(offsets,(len(meta),1))[lo:lo+64],dtype=a.dtype,device='cuda')[:,None]/256
      cond=old if mode in ['literal_old','original'] else new
      for t in sch.timesteps:
       ep=m.unet.net(a,t,global_cond=cond) if kind=='unet' else m.unet(a,t,global_cond=cond);calls+=len(cond);a=sch.step(ep,t,a).prev_sample
      out.append(((a[:,1:9]+1)*256).cpu().numpy())
    for handle in handles:handle.remove()
    arrays[f'{kind}|{factor}|{seed}|{mode}']=np.concatenate(out).reshape(len(meta),4,8,2);status(phase='sampling',model=kind,factor=factor,seed=seed,mode=mode,sampler_inputs=sampler_inputs)
   positive=max(positive,float(np.max(np.abs(arrays[f'{kind}|{factor}|{seed}|all_old']-arrays[f'{kind}|{factor}|{seed}|literal_old']))));checked+=1
  print(kind,factor,'done',flush=True)
 assert positive<.002;audits[kind]={'sha_before':sha,'sha_after':hashlib.sha256(ck.read_bytes()).hexdigest(),'all_old_hook_vs_direct_max_error_px':positive,'positive_checks':checked,'reads':len(names)};assert sha==audits[kind]['sha_after'];del m,H;torch.cuda.empty_cache()
np.savez_compressed(R/'actions.npz',**arrays);audit={'complete':True,'models':audits,'sampler_inputs':sampler_inputs,'denoiser_inputs':calls,'hook_calls':hook_calls,'contexts':len(meta),'sources':len(set(x['source_episode'] for x in meta)),'finite':all(np.isfinite(x).all() for x in arrays.values())};assert audit['finite'];(W/'report.json').write_text(json.dumps({'protocol':protocol,'inventory':inventory,'audit':audit},indent=2));status(phase='complete',**audit);print(json.dumps(audit),flush=True)
