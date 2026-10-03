#!/usr/bin/env python3
"""Compare a saved Diffusers first pass with the official Cosmos3 network.

Default mode is CPU metadata inspection only. --run explicitly loads the
existing shards and executes ONE official transformer forward on CUDA.
No VAE, processor, tokenizer, audio decoder, scheduler, FSDP or downloads.

This tests the network on identical saved model-facing tensors. It does not
prove equivalence of the complete official HTTP server or preprocessing.
Time-MLP precision is explicitly FP32; all other parameters use BF16, matching
the earlier Diffusers probe. No native attention implementation is replaced.
"""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--framework", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--states", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--run", action="store_true")
    return p.parse_args()


def official_key_mapper(framework):
    """Use the official pure mapping rules without importing the full loader."""
    source = framework / "cosmos_framework/inference/model.py"
    tree = ast.parse(source.read_text())
    constants = {
        "_DIFFUSERS_DROP_WEIGHT_PATH_RES", "_DIFFUSERS_DROP_KEY_RES",
        "_DIFFUSERS_KEY_MAPPING_RES", "_DIFFUSERS_NET_KEY_PREFIXES",
        "_DIFFUSERS_NET_KEYS",
    }
    functions = {
        "_should_drop_diffusers_weight_path", "_should_drop_diffusers_key",
        "_apply_diffusers_key_mapping", "_is_loadable_diffusers_net_key",
        "_diffusers_to_net_key",
    }
    nodes = []
    for node in tree.body:
        name = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        if name in constants or isinstance(node, ast.FunctionDef) and node.name in functions:
            nodes.append(node)
    ns = {"re": re}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), ns)
    return ns["_diffusers_to_net_key"]


def construct_meta(framework, checkpoint, torch):
    from cosmos_framework.model.generator.mot.unified_mot import (
        Qwen3VLMoTConfig, Qwen3VLTextForCausalLM,
    )
    from cosmos_framework.model.generator.mot.cosmos3_vfm_network import (
        Cosmos3VFMNetworkConfig, Cosmos3VFMNetwork,
    )
    old = json.loads((checkpoint / "config.json").read_text())["model"]["config"]
    expert = old["diffusion_expert_config"]
    qwen_path = framework / "cosmos_framework/model/generator/reasoner/qwen3_vl/configs/Qwen3-VL-8B-Instruct.json"
    base = json.loads(qwen_path.read_text())
    with torch.device("meta"):
        qwen = Qwen3VLMoTConfig(
            base, include_visual=False, qk_norm_for_text=True,
            qk_norm_for_diffusion=True,
        )
        lm = Qwen3VLTextForCausalLM(qwen)
        cfg = Cosmos3VFMNetworkConfig(
            vlm_config=lm.config,
            latent_patch_size=expert["patch_spatial"],
            latent_downsample_factor=old["latent_downsample_factor"],
            latent_channel_size=old["state_ch"],
            max_latent_h=expert["max_vae_latent_side_after_patchify"],
            max_latent_w=expert["max_vae_latent_side_after_patchify"],
            max_latent_t=old["state_t"],
            enable_fps_modulation=expert["enable_fps_modulation"],
            base_fps=expert["base_fps"],
            vision_gen=old["vision_gen"], action_gen=old["action_gen"],
            sound_gen=old["sound_gen"],
            joint_attn_implementation=old["joint_attn_implementation"],
            timestep_scale=expert["timestep_range"] / old["rectified_flow_inference_config"]["num_train_timesteps"],
            action_dim=old["max_action_dim"],
            num_embodiment_domains=old["num_embodiment_domains"],
            sound_dim=old["sound_dim"], sound_latent_fps=old["sound_latent_fps"],
        )
        cfg._attn_implementation_internal = "eager"
        net = Cosmos3VFMNetwork(lm, cfg).to(dtype=torch.bfloat16)
        # Native time embedding consumes FP32 sine/cosine input. Keep its
        # learned MLP in FP32 explicitly and disclose this runtime choice.
        net.time_embedder.float()
    return net


def inventory(net, framework, checkpoint):
    from safetensors import safe_open
    mapper = official_key_mapper(framework)
    wm = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    wm = {k: v for k, v in wm.items() if ".k_norm_und_for_gen." not in k}
    component = checkpoint / "transformer/diffusion_pytorch_model.safetensors.index.json"
    for key, value in json.loads(component.read_text())["weight_map"].items():
        wm.setdefault(key, "transformer/" + value)
    wanted = net.state_dict()
    mapping = {}
    for source, filename in wm.items():
        target = mapper(source, filename)
        if target not in wanted:
            continue
        if target in mapping:
            raise ValueError(f"Multiple source keys map to {target}")
        mapping[target] = (source, filename)
    missing = sorted(set(wanted) - set(mapping))
    if missing:
        raise ValueError(f"Checkpoint missing native parameters: {missing}")
    for target, (source, filename) in mapping.items():
        with safe_open(checkpoint / filename, framework="pt", device="cpu") as reader:
            shape = reader.get_slice(source).get_shape()
        if shape != list(wanted[target].shape):
            raise ValueError(f"Shape mismatch {target}: {shape} vs {list(wanted[target].shape)}")
    return mapping


def native_pack(saved, torch):
    """Adapt the already-packed captured model inputs; add no new tokens."""
    from cosmos_framework.data.generator.sequence_packing import PackedSequence, ModalityData
    inp = saved["input"]
    if saved["mode"] != "off" or inp["und_len"] != 121:
        raise ValueError("This controlled audit requires the native no-system 121-token case")
    if inp["sound_tokens"] is not None:
        raise ValueError("This audit does not support a sound input")
    if not torch.equal(saved["initial_action"].to(torch.bfloat16), inp["action_tokens"][0]):
        raise ValueError("Captured first-pass action does not match saved FP32 initial action")
    if not torch.equal(saved["initial_vision"].to(torch.bfloat16), inp["vision_tokens"][0]):
        raise ValueError("Captured first-pass vision does not match saved FP32 initial vision")

    def modality(prefix):
        return ModalityData(
            sequence_indexes=inp[f"{prefix}_sequence_indexes"].clone(),
            timesteps=inp[f"{prefix}_timesteps"].float(),
            mse_loss_indexes=inp[f"{prefix}_mse_loss_indexes"].clone(),
            token_shapes=[tuple(x) for x in inp[f"{prefix}_token_shapes"]],
            tokens=[x.clone() for x in inp[f"{prefix}_tokens"]],
            noisy_frame_indexes=[x.clone() for x in inp[f"{prefix}_noisy_frame_indexes"]],
            domain_id=([x.clone() for x in inp["action_domain_ids"]] if prefix == "action" else []),
        )
    ntext = int(inp["und_len"])
    length = int(inp["sequence_length"])
    return PackedSequence(
        sample_lens=[length], split_lens=[ntext, length - ntext],
        attn_modes=["causal", "full"], is_image_batch=False,
        uses_single_timestep=True, sequence_length=length,
        text_ids=inp["input_ids"].clone(), text_indexes=inp["text_indexes"].clone(),
        position_ids=inp["position_ids"].float(),
        text_caption_lens=[[ntext]], text_caption_view_ids=[[-1]],
        vision=modality("vision"), action=modality("action"),
    )


def error_metrics(actual, reference, torch):
    delta = actual.float().cpu() - reference.float().cpu()
    rms = float(delta.square().mean().sqrt())
    reference_rms = float(reference.float().square().mean().sqrt())
    return {"rms": rms, "max_abs": float(delta.abs().max()),
            "relative_rms_percent": 100 * rms / max(reference_rms, 1e-12)}


def native_action_rows(pack, action_global_indexes, torch):
    """Map saved packed-sequence action IDs to the native GEN tower rows.

    The native packer appends a separate padding segment even when CUDA-graph
    padding is disabled. Its `_full_indices` contains REAL packed-sequence
    IDs, in GEN tower order, and excludes that padding. Never use `[-16:]`.
    """
    full = pack["full_only_seq"]
    full_ids = pack["_full_indices"]
    if full.ndim != 2 or full_ids.ndim != 1:
        raise ValueError(f"Unexpected native pack shapes: GEN={tuple(full.shape)}, IDs={tuple(full_ids.shape)}")
    if full.shape[0] < full_ids.numel():
        raise ValueError("Native GEN tower is shorter than its real-token IDs")
    if not torch.equal(full_ids, full_ids.sort().values):
        raise ValueError("This controlled native two-way pack requires sorted real GEN IDs")
    action_ids = action_global_indexes.to(device=full_ids.device)
    rows = torch.searchsorted(full_ids, action_ids)
    if torch.any(rows >= full_ids.numel()) or not torch.equal(full_ids[rows], action_ids):
        raise ValueError("Saved action IDs are absent from the native GEN tower")
    return rows


def run_once(net, mapping, checkpoint, saved, packed, torch):
    from safetensors import safe_open
    if torch.backends.cudnn.version() < 92000:
        raise RuntimeError("Official cuDNN backend needs >=9.20; select existing 9.24 libraries before process start")
    if not torch.cuda.is_available():
        raise RuntimeError("--run requires CUDA")
    device = torch.device("cuda")
    # Initialize native nonpersistent RoPE/time-frequency buffers while all
    # parameters are still meta. They must not be left uninitialized.
    net.init_weights(buffer_device=device)
    wanted = net.state_dict()
    files = {}
    for target, (source, filename) in mapping.items():
        files.setdefault(filename, []).append((target, source))
    for filename, keys in sorted(files.items()):
        with safe_open(checkpoint / filename, framework="pt", device="cpu") as reader:
            values = {target: reader.get_tensor(source).to(device=device, dtype=wanted[target].dtype)
                      for target, source in keys}
        net.load_state_dict(values, strict=False, assign=True)
        del values
        print(json.dumps({"loaded_shard": filename, "keys": len(keys)}), flush=True)
    assert not any(p.is_meta for p in net.parameters())
    assert not any(b.is_meta for b in net.buffers())
    net.eval()
    packed.to_cuda()
    boundaries = {}
    readout = {}
    layout = {}
    handles = []
    action_global_indexes = packed.action.sequence_indexes
    if not torch.equal(action_global_indexes, packed.action.mse_loss_indexes):
        raise ValueError("This first-pass audit requires all saved actions to be noisy output tokens in identical order")

    def extract(pack):
        full = pack["full_only_seq"]
        rows = native_action_rows(pack, action_global_indexes, torch)
        if "action_rows" not in layout:
            layout.update(action_rows=rows, gen_shape=list(full.shape),
                          real_gen_tokens=int(pack["_full_indices"].numel()))
        elif not torch.equal(rows, layout["action_rows"]):
            raise ValueError("Action-token routing changed across native block boundaries")
        return full.index_select(0, rows).detach().cpu().clone()

    def first_pre(module, args, kwargs):
        boundaries[0] = extract(args[0] if args else kwargs["input"])

    handles.append(net.language_model.model.layers[0].register_forward_pre_hook(first_pre, with_kwargs=True))
    for index, layer in enumerate(net.language_model.model.layers):
        def post(module, args, out, index=index):
            boundaries[index + 1] = extract(out[0])
        handles.append(layer.register_forward_hook(post))
    def norm_post(module, args, out):
        if out.ndim != 2 or list(out.shape) != layout["gen_shape"]:
            raise ValueError(f"Native final GEN norm has unexpected shape: {tuple(out.shape)}")
        readout["action"] = out.index_select(0, layout["action_rows"]).detach().cpu().clone()
    handles.append(net.language_model.model.norm_moe_gen.register_forward_hook(norm_post))
    start = time.perf_counter()
    with torch.inference_mode():
        output = net(packed_seq=packed)
        torch.cuda.synchronize()
        reference_hidden = saved["readout"][0].to(device)
        domains = packed.action.domain_id[0].expand(16)
        reference_velocity = net.llm2action(reference_hidden, domains).detach().cpu()
    elapsed = time.perf_counter() - start
    for handle in handles:
        handle.remove()
    native_velocity = output["preds_action"][0].detach().cpu()
    hidden_stack = torch.stack([boundaries[i] for i in range(37)])
    metrics = {
        "forward_seconds": elapsed,
        "readout": error_metrics(readout["action"], saved["readout"][0], torch),
        "action_velocity_first10": error_metrics(native_velocity[:, :10], reference_velocity[:, :10], torch),
        "boundaries": [error_metrics(hidden_stack[i], saved["hidden"][0, i], torch) for i in range(37)],
        "native_layout": {"gen_shape_with_padding": layout["gen_shape"],
                          "real_gen_tokens": layout["real_gen_tokens"],
                          "action_global_indexes": action_global_indexes.cpu().tolist(),
                          "action_gen_rows": layout["action_rows"].cpu().tolist()},
        "reference_velocity": "Replay official loaded head on saved Diffusers readout; not a simulator action",
    }
    tensors = {"native_boundaries": hidden_stack, "native_readout": readout["action"],
               "native_velocity": native_velocity, "reference_head_velocity": reference_velocity}
    return metrics, tensors


def main():
    args = arguments()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("COSMOS_TRAINING", "0")
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    os.environ.setdefault("I4_ATTN_BACKENDS", "cudnn")
    if not args.run:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if args.run and args.output is None:
        raise ValueError("--run requires a fresh --output directory")
    if args.output is not None and args.output.exists():
        raise FileExistsError(f"Preserve previous evidence: output already exists: {args.output}")
    sys.path.insert(0, str(args.framework))
    import torch
    net = construct_meta(args.framework, args.checkpoint, torch)
    mapping = inventory(net, args.framework, args.checkpoint)
    saved = torch.load(args.states, map_location="cpu", weights_only=False)
    packed = native_pack(saved, torch)
    report = {
        "scope": "one native transformer forward on saved identical no-system model inputs",
        "full_official_server_equivalence": False,
        "case": saved["case"], "mode": saved["mode"],
        "torch": torch.__version__, "cudnn_version": torch.backends.cudnn.version(),
        "parameter_keys": len(mapping), "missing_keys": 0,
        "sequence_length": packed.sequence_length, "und_tokens": len(packed.text_ids),
        "time_mlp_precision": "float32", "other_parameter_precision": "bfloat16",
        "states": str(args.states), "checkpoint": str(args.checkpoint),
        "checkpoint_config_sha256": hashlib.sha256((args.checkpoint / "config.json").read_bytes()).hexdigest(),
        "status": "metadata_only",
    }
    if args.run:
        metrics, tensors = run_once(net, mapping, args.checkpoint, saved, packed, torch)
        report.update(status="forward_completed", comparisons=metrics)
        args.output.mkdir(parents=True, exist_ok=False)
        torch.save(tensors, args.output / "native_forward.pt")
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
