import torch,math
base="/home/current/work/cosmos3/outputs/official-demos/milk-posttrain/gripper-context/deep-dive/milk-shift-mechanism/matched-names/"
d=torch.load(base+"x15_milk_box_V198_A195_native/dispatch-t00-L35.pt",map_location="cpu",weights_only=True)
Q=d["Q"][0].float();K=d["K"][0].float().repeat_interleave(4,1);V=d["V"][0].float().repeat_interleave(4,1)
P=(torch.einsum("qhd,khd->hqk",Q,K)/math.sqrt(128)).softmax(-1)
O=torch.einsum("hqk,khd->qhd",P,V);N=d["native_output"][0].float()
e=(O-N).abs();print("mean abs err",e.mean().item(),"mean abs ref",N.abs().mean().item(),"rel L2",((O-N).norm()/N.norm()).item())
# contribution: weighted-value norm from text keys vs others on action rows
a=P[:,250:266,:]
Vn=V.norm(dim=-1)   # k x h
for n,(lo,hi) in dict(text=(0,122),current=(122,172),future=(172,372),action=(372,388)).items():
    c=torch.einsum("hqk,kh->hq",a[:,:,lo:hi],Vn[lo:hi]).mean().item()
    print(n,"attn-weighted |V| contribution",round(c,4),"mean |V|",Vn[lo:hi].mean().item())
