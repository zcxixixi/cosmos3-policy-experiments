"""pi0.5 (PyTorch, official pi05_libero weights) HTTP server with language-contrast interventions.

Run with the openpi venv and PYTHONPATH=~/work/pi05-pt/overlay. Nothing is written except under --out.
One sampler for every mode (eager, same noise rule), so latencies are comparable across modes:

  mode = {"type": "plain"}
  mode = {"type": "cag", "omega": w, "null_prompt": ""}          # action-level: null + w*(cond-null), batch of 2
  mode = {"type": "mix", "where": "kv"|"h", "ref_prompt": str, "layers": [a, b], "s": float}
         # prefix pass for [target, ref] as batch 2; on prefix layers a..b (inclusive) the target stream
         # is replaced by ref + s*(target-ref)  ("h": residual stream after the layer, propagates to later layers;
         # "kv": only the key/value cache the action expert reads). Denoising runs once for the target.
  s = 1 reproduces plain (up to numerical noise) -> built-in sanity check.
Optional request field "capture_path": saves pooled per-layer features of the plain/target pass.
"""
import argparse
import base64
import copy
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("OPENPI_DATA_HOME", "/home/current/.cache/openpi")

import jax  # noqa: E402
import torch  # noqa: E402

from openpi.models import model as _model  # noqa: E402
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks  # noqa: E402
from openpi.policies import policy_config  # noqa: E402
from openpi.training import config  # noqa: E402

CKPT = "/home/current/work/pi05-pt/pi05_libero_pytorch"
TOKENIZER = "/home/current/.cache/openpi/big_vision/paligemma_tokenizer.model"
LANG0 = 768  # 3 images x 256 tokens precede the language tokens in the prefix
DEV = "cuda"
H, ADIM = 10, 32


class Engine:
    def __init__(self):
        cfg = config.get_config("pi05_libero")
        self.policy = policy_config.create_trained_policy(cfg, CKPT, sample_kwargs={"num_steps": 10},
                                                         pytorch_device=DEV)
        self.model = self.policy._model
        self.lm = self.model.paligemma_with_expert.paligemma.language_model
        self.expert = self.model.paligemma_with_expert.gemma_expert.model
        self.nlayers = len(self.lm.layers)
        self.online_mix = None
        self.online_calls = 0
        self.repair_gain = 0.0
        self.repair_delta = torch.from_numpy(np.load(Path(__file__).parent / "adapter.npz")["delta"]).to(DEV)
        self.repair_calls = 0
        self.expert.layers[16].register_forward_hook(self._repair_hook)
        self.mix = None      # dict(layers=set, s=float) for "h" mixing
        self.cap = None      # dict while capturing
        self.expert_seen = False
        for i, layer in enumerate(self.lm.layers):
            layer.register_forward_hook(self._prefix_hook(i))
        for i, layer in enumerate(self.expert.layers):
            layer.register_forward_hook(self._expert_hook(i))
        self.lock = threading.Lock()
        import sentencepiece
        self.sp = sentencepiece.SentencePieceProcessor(model_file=TOKENIZER)

    def noun_cols(self, prompt, noun):
        """Absolute prefix columns of the object-noun tokens, from the instruction template (deployable)."""
        head = prompt.split(noun)[0]
        n0 = len(self.sp.encode(head.strip(), add_bos=True))
        n1 = len(self.sp.encode((head + noun).strip(), add_bos=True))
        return [LANG0 + i for i in range(n0, n1)]

    def denoise_bias(self, st, pad, pkv, x_t, t, cols, logalpha):
        """model.denoise_step with an additive attention bias on the given prefix key columns."""
        m = self.model
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks as mk
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = m.embed_suffix(st, x_t, t)
        suffix_len, B, plen = suffix_pad_masks.shape[1], pad.shape[0], pad.shape[1]
        full = torch.cat([pad[:, None, :].expand(B, suffix_len, plen), mk(suffix_pad_masks, suffix_att_masks)], dim=2)
        pos = pad.sum(-1)[:, None] + torch.cumsum(suffix_pad_masks, dim=1) - 1
        full4d = m._prepare_attention_masks_4d(full)
        if cols:
            full4d[..., cols] = full4d[..., cols] + logalpha
        m.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
        out, _ = m.paligemma_with_expert.forward(attention_mask=full4d, position_ids=pos, past_key_values=pkv,
                                                 inputs_embeds=[None, suffix_embs], use_cache=False,
                                                 adarms_cond=[None, adarms_cond])
        out = out[1][:, -m.config.action_horizon:].to(torch.float32)
        return m.action_out_proj(out)

    def _repair_hook(self, module, args, output):
        if self.repair_gain == 0.0:
            return None
        hs = output[0] if isinstance(output, tuple) else output
        assert hs.shape[-1] == 1024
        new = (hs.float() + self.repair_gain * self.repair_delta).to(hs.dtype)
        self.repair_calls += 1
        return (new,) + tuple(output[1:]) if isinstance(output, tuple) else new

    def _prefix_hook(self, i):
        def hook(module, args, output):
            hs = output[0] if isinstance(output, tuple) else output
            if self.mix is not None and i in self.mix["layers"] and hs.shape[0] == 2:
                s = self.mix["s"]
                tgt, ref = hs[0].float(), hs[1].float()
                new = (ref + s * (tgt - ref)).to(hs.dtype)
                hs = torch.stack([new, hs[1]], 0)
                return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs
            if self.cap is not None:
                x = hs[0].float()
                m = self.cap["lang_mask"]
                self.cap["base"][i] = x[:256].mean(0).cpu().numpy()
                self.cap["wrist"][i] = x[256:512].mean(0).cpu().numpy()
                lang = x[768:]
                self.cap["lang_mean"][i] = lang[m].mean(0).cpu().numpy()
                self.cap["lang_last"][i] = lang[m][-1].cpu().numpy()
            return None
        return hook

    def _expert_hook(self, i):
        def hook(module, args, output):
            if self.online_mix is not None and i in self.online_mix['layers']:
                hs = output[0] if isinstance(output, tuple) else output
                assert hs.shape[0] == 2
                scale = self.online_mix['s']
                # Both branches receive exactly the same current noisy actions.
                mixed = hs[0] if scale == 1 else (hs[1].float() + scale * (hs[0].float()-hs[1].float())).to(hs.dtype)
                hs = torch.stack([mixed, hs[1]], 0)
                self.online_calls += 1
                return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs
            if self.cap is not None and not self.cap["expert_done"]:
                hs = output[0] if isinstance(output, tuple) else output
                self.cap["expert"][i] = hs[0].float().mean(0).cpu().numpy()
                if i == len(self.expert.layers) - 1:
                    self.cap["expert_done"] = True
            return None
        return hook

    def _batch(self, obs_base, obs_wrist, state, prompts):
        items = []
        for p in prompts:
            obs = {"observation/image": obs_base, "observation/wrist_image": obs_wrist,
                   "observation/state": state, "prompt": p}
            inputs = self.policy._input_transform(jax.tree.map(lambda x: x, obs))
            items.append(jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(DEV)[None, ...], inputs))
        batched = jax.tree.map(lambda *xs: torch.cat(xs, 0), *items)
        return _model.Observation.from_dict(batched)

    @torch.no_grad()
    def infer(self, base, wrist, state, prompt, seed, mode, capture_path=None):
        self.repair_gain = -1.0 if mode.get("repair", False) and "orange juice" in prompt.lower() else 0.0
        self.repair_calls = 0
        m = self.model
        typ = mode.get("type", "plain")
        if typ == "plain":
            prompts = [prompt]
        elif typ == "cag":
            prompts = [prompt, mode.get("null_prompt", "")]
        elif typ in ("mix", "lcag"):
            prompts = [prompt, mode["ref_prompt"]]
        elif typ == "attn":
            prompts = [prompt]
        else:
            raise ValueError(typ)
        noise1 = np.random.default_rng(int(seed)).standard_normal((H, ADIM), dtype=np.float32)
        t_start = time.perf_counter()
        observation = self._batch(base, wrist, state, prompts)
        B = len(prompts)
        noise = torch.from_numpy(noise1).to(DEV)[None].expand(B, -1, -1).contiguous()
        images, img_masks, lang_tokens, lang_masks, st = m._preprocess_observation(observation, train=False)
        embs, pad, att = m.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        att4d = m._prepare_attention_masks_4d(make_att_2d_masks(pad, att))
        pos = torch.cumsum(pad, dim=1) - 1
        m.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        bias_cols, logalpha = [], 0.0
        if typ == "attn":
            valid = (LANG0 + torch.nonzero(lang_masks[0]).flatten()).tolist()
            bias_cols = valid if mode.get("tokens", "lang") == "lang" else self.noun_cols(prompt, mode["noun"])
            if mode.get("tokens") == "obj":
                assert all(c in valid for c in bias_cols) and bias_cols, bias_cols
            logalpha = float(np.log(mode["alpha"]))
            if mode.get("where", "expert") == "both":
                att4d = att4d.clone()
                att4d[..., bias_cols] = att4d[..., bias_cols] + logalpha
        if capture_path:
            self.cap = dict(base={}, wrist={}, lang_mean={}, lang_last={}, expert={}, expert_done=False,
                            lang_mask=lang_masks[0].bool())
        if typ == "mix" and mode["where"] == "h":
            a, b = mode["layers"]
            self.mix = dict(layers=set(range(a, b + 1)), s=float(mode["s"]))
        try:
            _, pkv = m.paligemma_with_expert.forward(attention_mask=att4d, position_ids=pos,
                                                     past_key_values=None, inputs_embeds=[embs, None],
                                                     use_cache=True)
        finally:
            self.mix = None
        if typ == "mix":
            if mode["where"] == "kv":
                a, b = mode["layers"]
                s = float(mode["s"])
                for l in range(len(pkv.key_cache)):
                    k, v = pkv.key_cache[l], pkv.value_cache[l]
                    if a <= l <= b:
                        k0 = (k[1:2].float() + s * (k[0:1].float() - k[1:2].float())).to(k.dtype)
                        v0 = (v[1:2].float() + s * (v[0:1].float() - v[1:2].float())).to(v.dtype)
                    else:
                        k0, v0 = k[0:1], v[0:1]
                    pkv.key_cache[l], pkv.value_cache[l] = k0.contiguous(), v0.contiguous()
            else:
                for l in range(len(pkv.key_cache)):
                    pkv.key_cache[l] = pkv.key_cache[l][0:1].contiguous()
                    pkv.value_cache[l] = pkv.value_cache[l][0:1].contiguous()
            pad, st, noise = pad[0:1], st[0:1], noise[0:1]
        self.online_calls = 0
        self.online_mix = dict(layers=set(mode['layers']), s=float(mode['s'])) if typ == 'lcag' else None
        x = noise
        dt = torch.tensor(-1.0 / 10, dtype=torch.float32, device=DEV)
        t = torch.tensor(1.0, dtype=torch.float32, device=DEV)
        while t >= -dt / 2:
            if typ == "attn":
                v = self.denoise_bias(st, pad, pkv, x, t.expand(x.shape[0]), bias_cols, logalpha)
            else:
                v = m.denoise_step(st, pad, pkv, x, t.expand(x.shape[0]))
            if typ == 'lcag':
                # Reference is recomputed at target x_t, never an independently evolving trajectory.
                x = (x[0:1] + dt * v[0:1]).expand(2, -1, -1).contiguous()
            else:
                x = x + dt * v
            t = t + dt
        if typ == 'lcag':
            assert self.online_calls == 10 * len(mode['layers']), self.online_calls
        self.online_mix = None
        if typ == "cag":
            w = float(mode["omega"])
            x = x[1:2] + w * (x[0:1] - x[1:2])
        else:
            x = x[0:1]
        assert self.repair_calls == (10 if self.repair_gain else 0), self.repair_calls
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t_start) * 1000
        out = {"state": st[0:1][0].float().cpu().numpy(), "actions": x[0].float().cpu().numpy()}
        out = self.policy._output_transform(out)
        if capture_path and self.cap is not None:
            c = self.cap
            arrays = {}
            for key in ("base", "wrist", "lang_mean", "lang_last", "expert"):
                arrays[key] = np.stack([c[key][i] for i in sorted(c[key])]).astype(np.float16)
            np.savez(capture_path, **arrays)
        self.cap = None
        return np.asarray(out["actions"]), ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8941)
    args = ap.parse_args()
    eng = Engine()
    print("engine ready", flush=True)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send({"state": "ready", "model": "pi05_libero_pytorch"})

        def do_POST(self):
            try:
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                base = np.frombuffer(base64.b64decode(req["base"]), np.uint8).reshape(224, 224, 3)
                wrist = np.frombuffer(base64.b64decode(req["wrist"]), np.uint8).reshape(224, 224, 3)
                state = np.asarray(req["state"], dtype=np.float64)
                with eng.lock:
                    actions, ms = eng.infer(base.copy(), wrist.copy(), state, req["prompt"], req["seed"],
                                            req.get("mode", {"type": "plain"}), req.get("capture_path"))
                self._send({"actions": actions.tolist(), "ms": ms})
            except Exception:
                import traceback
                self._send({"error": traceback.format_exc()}, 500)

    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
