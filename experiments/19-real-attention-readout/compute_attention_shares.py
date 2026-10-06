import torch, json, math
from transformers import AutoTokenizer
base="/home/current/work/cosmos3/outputs/official-demos/milk-posttrain/gripper-context/deep-dive/milk-shift-mechanism/matched-names/"
tok=AutoTokenizer.from_pretrained("/home/current/work/cosmos3/checkpoints/Cosmos3-Nano-Policy-LIBEROall-5k/text_tokenizer",local_files_only=True)
out={}
for goal in ["milk_box","cream_cheese"]:
    d=torch.load(base+f"x15_{goal}_V198_A195_native/dispatch-t00-L35.pt",map_location="cpu",weights_only=True)
    Q=d["Q"][0].float(); K=d["K"][0].float(); V=d["V"][0].float()   # Q 266x32x128, K 388x8x128
    ix=d["indexes"]
    # GQA: head h uses kv head h//4
    Kx=K.repeat_interleave(4,dim=1); Vx=V.repeat_interleave(4,dim=1)
    logits=torch.einsum("qhd,khd->hqk",Q,Kx)/math.sqrt(128)
    P=logits.softmax(-1)                       # 32 x 266 x 388
    O=torch.einsum("hqk,khd->qhd",P,Vx)
    err=(O-d["native_output"][0].float()).abs().max().item(); ref=d["native_output"][0].float().abs().mean().item()
    act=P[:,250:266,:]                          # action queries
    g={k:ix[k].tolist() for k in ["text_keys","current_keys","future_keys","action_keys"]}
    share={n:act[:,:,idx].sum(-1).mean().item() for n,idx in g.items()}
    tpos=d["target_span"]["target_positions"]
    name=act[:,:,tpos].sum(-1)                  # 32x16
    text_total=act[:,:,g["text_keys"]].sum(-1)
    ids=d["target_span"]["input_ids"].tolist()
    text=tok.decode(ids)
    toks=[tok.decode([i]) for i in ids]
    # per-head: share of text on name tokens, and top heads by text share
    head_text=text_total.mean(1)                # 32
    # per key avg attention (mean over heads, 16 action queries)
    per_key=act.mean(dim=(0,1))
    # also check ranges of key groups
    rng={n:[min(v),max(v)] for n,v in g.items()}
    out[goal]=dict(max_abs_err=err,mean_abs_out=ref,share=share,name_share_of_all=name.mean().item(),
      name_share_of_text=(name.sum()/text_total.sum()).item(),
      uniform_expect_name=2/388,uniform_expect_text=122/388,
      head_text_min=head_text.min().item(),head_text_max=head_text.max().item(),
      per_key_text=per_key[g["text_keys"]].tolist(),ranges=rng,prompt=d["target_span"]["prompt"],full_text=text,tokens=toks,tpos=tpos,
      n_keys={n:len(v) for n,v in g.items()})
json.dump(out,open("/tmp/real/attn.json","w"),ensure_ascii=False,indent=1)
for k,v in out.items():
    print(k,"err",v["max_abs_err"],"/",v["mean_abs_out"]);print(v["share"],v["name_share_of_all"],v["name_share_of_text"],v["ranges"],v["n_keys"])
print(out["milk_box"]["full_text"]);print(out["milk_box"]["tokens"][44:60])
