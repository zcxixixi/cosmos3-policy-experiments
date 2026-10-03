"""Route interventions for the unsharded Diffusers Cosmos3 attention processor.

Usage (run each prediction under torch.inference_mode(), with model.eval()):
    routes = CosmosAttentionRoutes(pipe.transformer)
    routes.begin_capture("recipient")
    try:
        recipient_output = run_recipient_prediction()
    finally:
        recipient_cache = routes.end()
    # Capture donor on the SAME observation/noise/schedule, with another caption.
    routes.begin_patch(recipient_cache, donor_cache, "direct")
    try:
        patched_output = run_recipient_prediction()
    finally:
        report = routes.end()

Always check cache/report["complete"]. end() removes this helper's hooks even
after an incomplete prediction and reports the incomplete session without
masking the runner's exception. The runner verifies output exactness itself.
For a 30-step prediction the runner must also assert cache["n_steps"] == 30;
capture completeness alone describes the forwards that were observed.

Caches are ordinary dicts containing CPU tensors, keyed by (step, layer).
Projection values are captured BEFORE per-head normalization and RoPE. No
UND deduplication is performed. Sessions represent ONE prediction/query;
recapture on each current closed-loop observation, never another trajectory.
Only single-sample, no-audio, no-CFG, unsharded policy packs with all action
tokens noisy are supported. Native framework packs/padding are NOT supported.
"""

from __future__ import annotations

from typing import Any

import torch


# (donor-text query groups, recipient-baseline visual K/V clamp groups)
_MODES = {
    "empty": ((), ()),
    "full": (("action", "future", "clean"), ()),
    "direct": (("action",), ("future", "clean")),
    "indirect": (("future",), ("clean",)),
    "indirect_blocked": (("future",), ("future", "clean")),
    "joint": (("action", "future"), ("clean",)),
    "clamp_self": ((), ("future", "clean")),
}
_PROJECTIONS = {
    "und_k": "to_k", "und_v": "to_v",
    "gen_k": "add_k_proj", "gen_v": "add_v_proj",
}
_STRUCTURAL_KEYS = (
    "und_len", "sequence_length", "input_ids", "text_indexes", "position_ids",
    "vision_sequence_indexes", "vision_mse_loss_indexes", "vision_token_shapes",
    "vision_noisy_frame_indexes", "action_sequence_indexes", "action_mse_loss_indexes",
    "action_token_shapes", "action_noisy_frame_indexes", "action_domain_ids",
    "vision_timesteps", "action_timesteps",
)


def _cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu(x) for x in value)
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    return value


def _equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
                and left.dtype == right.dtype and torch.equal(left.cpu(), right.cpu()))
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return (type(left) is type(right) and len(left) == len(right)
                and all(_equal(a, b) for a, b in zip(left, right)))
    if isinstance(left, dict) or isinstance(right, dict):
        return (isinstance(left, dict) and isinstance(right, dict) and left.keys() == right.keys()
                and all(_equal(left[k], right[k]) for k in left))
    return left == right


def _rows(value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.dtype != torch.long:
        raise ValueError("Sequence indexes must be LongTensors")
    result = value.detach().cpu().flatten()
    if result.unique().numel() != result.numel():
        raise ValueError("Repeated sequence indexes are unsupported")
    return result


def _layout(kwargs: dict[str, Any]) -> dict[str, Any]:
    if kwargs.get("sound_tokens") is not None:
        raise ValueError("Audio packs are unsupported")
    if len(kwargs["vision_tokens"]) != 1 or len(kwargs["action_tokens"]) != 1:
        raise ValueError("Exactly one vision item and one action item are required")
    und_len, length = int(kwargs["und_len"]), int(kwargs["sequence_length"])
    text = _rows(kwargs["text_indexes"])
    if not torch.equal(text, torch.arange(und_len)) or kwargs["input_ids"].numel() != und_len:
        raise ValueError("UND must be one contiguous, unpadded text prefix")
    vision = _rows(kwargs["vision_sequence_indexes"])
    future = _rows(kwargs["vision_mse_loss_indexes"])
    action = _rows(kwargs["action_mse_loss_indexes"])
    if not torch.equal(action, _rows(kwargs["action_sequence_indexes"])):
        raise ValueError("Clean/historical action tokens are unsupported")
    if not bool(torch.isin(future, vision).all()):
        raise ValueError("Noisy vision indexes must belong to the vision segment")
    shapes = kwargs["vision_token_shapes"]
    noisy_frames = kwargs["vision_noisy_frame_indexes"]
    if len(shapes) != 1 or len(noisy_frames) != 1:
        raise ValueError("Multiple vision segments are unsupported")
    t, h, w = (int(x) for x in shapes[0])
    if vision.numel() != t * h * w:
        raise ValueError("Vision grid and index count disagree")
    nf = _rows(noisy_frames[0])
    if bool(((nf < 0) | (nf >= t)).any()):
        raise ValueError("Invalid noisy vision frame")
    offsets = (nf[:, None] * (h * w) + torch.arange(h * w)[None, :]).flatten()
    if not torch.equal(future, vision[offsets]):
        raise ValueError("Future indexes disagree with noisy-frame/grid metadata")
    clean = vision[~torch.isin(vision, future)]
    all_gen = torch.cat((vision, action))
    if not torch.equal(all_gen.sort().values, torch.arange(und_len, length)):
        raise ValueError("A/F/C must cover all GEN rows exactly, without padding/other modalities")
    return {"und_len": und_len, "gen_len": length - und_len,
            "action": action - und_len, "future": future - und_len, "clean": clean - und_len}


class CosmosAttentionRoutes:
    """Capture and replace projected K/V without editing model/library code.

    begin_patch(R, D, mode, gain=1): full/direct/indirect/indirect_blocked/joint
    select the donor-text GEN attention output rows. clamp_self tests baseline
    visual K/V replay without donor text. empty and gain=0 return original
    attention outputs and disable ALL clamps. full with D=R is self-patch.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.attentions = [layer.self_attn for layer in model.layers]
        for attn in self.attentions:
            for name in (*_PROJECTIONS.values(), "processor"):
                if not hasattr(attn, name):
                    raise TypeError(f"Expected Diffusers Cosmos3 attention attribute {name}")
        self._handles: list[Any] = []
        self._kind: str | None = None
        self._recomputing = False

    def _start(self, kind: str) -> None:
        if self._kind is not None:
            raise RuntimeError("End the active session before starting another")
        if self.model.training or torch.is_grad_enabled():
            raise RuntimeError("Use model.eval() and torch.no_grad()/inference_mode()")
        if (getattr(self.model, "_cp_shard_fn", None) is not None
                or getattr(self.model, "_cp_gather_fn", None) is not None
                or any(getattr(a.processor, "_parallel_config", None) is not None for a in self.attentions)):
            raise ValueError("Context-parallel/sharded processors are unsupported")
        self._kind, self._step = kind, -1
        self._visited: set[tuple[int, int]] = set()
        self._finished_steps: set[int] = set()
        self._recompute_calls = 0
        self._recomputing = False
        try:
            self._handles.append(self.model.register_forward_pre_hook(self._before_model, with_kwargs=True))
            self._handles.append(self.model.register_forward_hook(self._after_model))
            for layer, attn in enumerate(self.attentions):
                if kind == "capture":
                    for key, name in _PROJECTIONS.items():
                        self._handles.append(getattr(attn, name).register_forward_hook(
                            self._capture_projection(layer, key)))
                self._handles.append(attn.register_forward_hook(self._after_attention(layer), with_kwargs=True))
        except BaseException:
            self._remove_hooks()
            self._kind = None
            raise

    def begin_capture(self, label: str) -> None:
        self._start("capture")
        self._cache: dict[str, Any] = {"version": 1, "label": label, "n_layers": len(self.attentions),
                                       "steps": [], "frames": {}, "cache_device": "cpu"}

    def begin_patch(self, recipient_cache: dict, donor_cache: dict, mode: str, gain: int = 1) -> None:
        if self._kind is not None:
            raise RuntimeError("End the active session before starting another")
        if mode not in _MODES or gain not in (0, 1):
            raise ValueError("Unknown route mode or gain outside {0, 1}")
        for cache in (recipient_cache, donor_cache):
            if not cache.get("complete") or cache.get("n_layers") != len(self.attentions):
                raise ValueError("An incomplete/incompatible capture cannot be patched")
        if recipient_cache["n_steps"] != donor_cache["n_steps"]:
            raise ValueError("Recipient/donor step counts disagree")
        for step, (r, d) in enumerate(zip(recipient_cache["steps"], donor_cache["steps"])):
            for key in r["metadata"]:
                if key != "input_ids" and not _equal(r["metadata"][key], d["metadata"][key]):
                    raise ValueError(f"Recipient/donor pack or schedule mismatch: step={step}, key={key}")
            if not _equal(r["layout"], d["layout"]):
                raise ValueError("Recipient/donor query layouts disagree")
        if not _equal(recipient_cache["initial_inputs"], donor_cache["initial_inputs"]):
            raise ValueError("Recipient/donor initial observation or noise differs")
        self._recipient, self._donor = recipient_cache, donor_cache
        self._mode, self._gain = mode, gain
        self._start("patch")

    def _before_model(self, module, args, kwargs):
        if self._recomputing:
            raise RuntimeError("Attention recomputation must not recurse through the model")
        if args:
            raise ValueError("Call the transformer with keyword arguments")
        if self._step >= 0 and any((self._step, l) not in self._visited for l in range(len(self.attentions))):
            raise RuntimeError("Previous model forward did not visit every attention layer")
        self._step += 1
        self._current_layout = _layout(kwargs)
        metadata = {key: _cpu(kwargs.get(key)) for key in _STRUCTURAL_KEYS}
        initial = {key: _cpu(kwargs[key]) for key in ("vision_tokens", "action_tokens")}
        if self._kind == "capture":
            if self._step:
                first = self._cache["steps"][0]
                for key in metadata:
                    if key not in ("vision_timesteps", "action_timesteps") and not _equal(
                            metadata[key], first["metadata"][key]):
                        raise ValueError(f"Static capture pack changed at step {self._step}: {key}; CFG unsupported")
            self._cache["steps"].append({"layout": self._current_layout, "metadata": metadata})
            if self._step == 0:
                self._cache["initial_inputs"] = initial
        else:
            if self._step >= self._recipient["n_steps"]:
                raise ValueError("More model forwards than captured; CFG/extra passes are unsupported")
            reference = self._recipient["steps"][self._step]
            if not _equal(metadata, reference["metadata"]):
                raise ValueError(f"Patch call pack/text/schedule differs from R at step {self._step}")
            if self._step == 0 and not _equal(initial, self._recipient["initial_inputs"]):
                raise ValueError("Patch initial observation/noise differs from R")

    def _capture_projection(self, layer: int, key: str):
        def capture(module, args, output):
            if self._recomputing:
                return
            frame = self._cache["frames"].setdefault((self._step, layer), {})
            if key in frame:
                raise RuntimeError("Projection called twice in one step/layer")
            frame[key] = _cpu(output)
        return capture

    def _after_model(self, module, args, output):
        if any((self._step, layer) not in self._visited for layer in range(len(self.attentions))):
            raise RuntimeError("Model completed without visiting all attention layers")
        self._finished_steps.add(self._step)

    def _after_attention(self, layer: int):
        def after(module, args, kwargs, output):
            if self._recomputing:
                return
            pair = (self._step, layer)
            if pair in self._visited:
                raise RuntimeError("Attention called twice in one step/layer")
            if not isinstance(output, tuple) or len(output) != 2:
                raise TypeError("Expected (UND output, GEN output)")
            if output[1].shape[0] != self._current_layout["gen_len"]:
                raise ValueError("Unexpected GEN length; possible padding/sharding")
            if self._kind == "capture":
                if set(self._cache["frames"].get(pair, {})) != set(_PROJECTIONS):
                    raise RuntimeError("Missing raw projection capture")
                self._visited.add(pair)
                return
            if self._gain == 0 or self._mode == "empty":
                self._visited.add(pair)
                return output
            groups, frozen = _MODES[self._mode]
            values = [args[i] if len(args) > i else kwargs[name]
                      for i, name in enumerate(("und_seq", "gen_seq", "rotary_emb"))]
            base = self._recompute(module, values, layer, frozen, False) if frozen else output
            if not groups:
                result = (output[0], base[1])
            else:
                alternative = self._recompute(module, values, layer, frozen, True)
                rows = torch.cat([self._current_layout[g] for g in groups]).to(output[1].device)
                gen = base[1].clone()
                gen.index_copy_(0, rows, alternative[1].index_select(0, rows))
                result = (output[0], gen)
            self._visited.add(pair)
            return result
        return after

    def _recompute(self, attn, values, layer: int, frozen: tuple, donor: bool):
        # Calling processor directly bypasses self_attn.forward hooks. The flag
        # additionally prevents our projection capture hooks seeing this pass.
        handles = []
        pair = (self._step, layer)
        rows = torch.cat([self._current_layout[g] for g in frozen]) if frozen else None

        def replacement(cache_key: str, source: dict, selected: torch.Tensor | None):
            saved = source["frames"][pair][cache_key]
            def replace(module, args, output):
                if saved.shape != output.shape or saved.dtype != output.dtype:
                    raise ValueError(f"Projection shape/dtype mismatch at {pair}/{cache_key}")
                if selected is None:
                    return saved.to(device=output.device)
                result = output.clone()
                indexes = selected.to(device=output.device)
                # Transfer only selected visual rows; preserve live action K/V.
                result.index_copy_(0, indexes, saved.index_select(0, selected).to(output.device))
                return result
            return replace

        self._recomputing = True
        try:
            if rows is not None:
                for key in ("gen_k", "gen_v"):
                    handles.append(getattr(attn, _PROJECTIONS[key]).register_forward_hook(
                        replacement(key, self._recipient, rows)))
            if donor:
                for key in ("und_k", "und_v"):
                    handles.append(getattr(attn, _PROJECTIONS[key]).register_forward_hook(
                        replacement(key, self._donor, None)))
            result = attn.processor(attn, *values)
            self._recompute_calls += 1
            if not isinstance(result, tuple) or len(result) != 2:
                raise TypeError("Processor changed its output contract")
            return result
        finally:
            for handle in reversed(handles):
                handle.remove()
            self._recomputing = False

    def _remove_hooks(self):
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()

    def end(self) -> dict[str, Any]:
        """Remove owned hooks and return capture or patch report, even if incomplete."""
        if self._kind is None:
            raise RuntimeError("No active route session")
        kind = self._kind
        self._remove_hooks()
        self._kind = None
        steps = self._step + 1
        expected = {(s, l) for s in range(steps) for l in range(len(self.attentions))}
        missing = sorted(expected - self._visited)
        unfinished = sorted(set(range(steps)) - self._finished_steps)
        complete = steps > 0 and not missing and not unfinished
        if kind == "capture":
            cache = self._cache
            complete = complete and all(set(cache["frames"].get(pair, {})) == set(_PROJECTIONS)
                                        for pair in expected)
            identical = {}
            for layer in range(len(self.attentions)):
                first = cache["frames"].get((0, layer), {})
                identical[layer] = complete and all(
                    _equal(first.get(k), cache["frames"][(s, layer)][k])
                    for s in range(steps) for k in ("und_k", "und_v"))
            cache.update(complete=complete, n_steps=steps, missing_layers=missing,
                         unfinished_forwards=unfinished,
                         und_identical_across_steps=identical, und_deduplicated=False)
            del self._cache
            return cache
        complete = complete and steps == self._recipient["n_steps"]
        groups, frozen = _MODES[self._mode] if self._gain else ((), ())
        report = {"complete": complete, "mode": self._mode, "gain": self._gain,
                  "n_steps": steps, "n_layers": len(self.attentions), "missing_layers": missing,
                  "unfinished_forwards": unfinished,
                  "expected_steps": self._recipient["n_steps"], "recompute_calls": self._recompute_calls,
                  "donor_query_groups": groups, "frozen_visual_kv_groups": frozen,
                  "layout": self._recipient["steps"][0]["layout"],
                  "output_exactness_checked_by_helper": False}
        del self._recipient, self._donor
        return report
