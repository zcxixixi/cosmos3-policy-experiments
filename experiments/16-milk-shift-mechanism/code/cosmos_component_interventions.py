"""Local interventions on Cosmos3 GEN action attention/MLP increments.

This helper matches the Diffusers decoder in transformer_cosmos3.py: its
attention output is added to gen_seq, then mlp_moe_gen output is added to that
residual. Hooks change only those increments, never an entire residual state.
Rows come from each real model call's action_mse_loss_indexes - und_len. No
assumption is made that action rows are a suffix, contiguous, or all action
tokens. Layer and denoising-forward indexes are zero based. One session means
ONE policy query, not a closed-loop trajectory; the caller resets per query.

    hooks = CosmosComponentInterventions(pipe.transformer)
    with torch.inference_mode():
        hooks.begin_capture("milk15", expected_steps=30)
        try:
            baseline = predict()
        finally:
            recipient = hooks.end()
        # A brief attenuation at one layer and selected denoising forwards:
        hooks.begin_patch(recipient, layer=12, component="attention",
                          steps=[0], gain=0.5)
        try:
            altered = predict_same_observation_and_initial_noise()
        finally:
            report = hooks.end()

For natural donor replacement, supply donor_cache from another observation
with the SAME instruction, sequence layout, schedule and initial action noise.
Visual inputs may differ; this is explicitly reported, not assumed aligned in
meaning. gain=1 copies the donor increment, gain=0 does nothing; intermediate
gains blend current + gain*(donor-current). Without a donor, gain scales only
the selected increment. Check complete and n_steps in every returned result.
The runner must compare actual final outputs for no-op and self-patch.

This is not a path-isolating intervention: every downstream component remains
free to respond. Persistence after removing the hook can reflect denoising or
environment feedback and does not alone establish a neural attractor. No
processor is recomputed. Duplicate component calls fail rather than silently
recording recomputations. Capture/patch assumes eager, eval, no-grad,
single-item, no-audio, no-CFG and unsharded execution. Do not combine with an
attention-route recomputation helper. Post-to_add_out coordinates are NOT
attention heads; head patches need the pre-to_add_out tensor with its verified
num_attention_heads x head_dim layout. MLP-unit patches similarly need the
post-activation input to down_proj. Those sites are intentionally unsupported.

Run --cpu-check in the existing Cosmos3 environment to exercise the ACTUAL
installed decoder with tiny weights, causal/full attention and residuals;
this does not load the checkpoint, create a CUDA context or test MuJoCo.
"""

from __future__ import annotations

import math
from typing import Any

import torch


_COMPONENTS = ("attention", "mlp")
_STRUCTURAL_KEYS = (
    "und_len", "sequence_length", "input_ids", "text_indexes", "position_ids",
    "vision_sequence_indexes", "vision_mse_loss_indexes", "vision_token_shapes",
    "vision_noisy_frame_indexes", "action_sequence_indexes", "action_mse_loss_indexes",
    "action_token_shapes", "action_noisy_frame_indexes", "action_domain_ids",
    "vision_timesteps", "action_timesteps",
)
_TIMESTEP_KEYS = ("vision_timesteps", "action_timesteps")


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
                and left.dtype == right.dtype and left.shape == right.shape
                and torch.equal(left.cpu(), right.cpu()))
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return (type(left) is type(right) and len(left) == len(right)
                and all(_equal(a, b) for a, b in zip(left, right)))
    if isinstance(left, dict) or isinstance(right, dict):
        return (isinstance(left, dict) and isinstance(right, dict) and left.keys() == right.keys()
                and all(_equal(left[k], right[k]) for k in left))
    return left == right


def _indexes(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.dtype != torch.long:
        raise ValueError(f"{name} must be a LongTensor")
    rows = value.detach().cpu().flatten()
    if rows.unique().numel() != rows.numel():
        raise ValueError(f"{name} contains repeated indexes")
    return rows


def _layout(kwargs: dict[str, Any]) -> dict[str, Any]:
    if kwargs.get("sound_tokens") is not None:
        raise ValueError("Audio packs are unsupported")
    if len(kwargs["vision_tokens"]) != 1 or len(kwargs["action_tokens"]) != 1:
        raise ValueError("Exactly one vision item and one action item are required")
    und_len, length = int(kwargs["und_len"]), int(kwargs["sequence_length"])
    if not 0 < und_len < length:
        raise ValueError("Invalid UND/GEN boundary")
    text = _indexes(kwargs["text_indexes"], "text_indexes")
    if not torch.equal(text, torch.arange(und_len)) or kwargs["input_ids"].numel() != und_len:
        raise ValueError("UND must be a contiguous unpadded text prefix")
    vision = _indexes(kwargs["vision_sequence_indexes"], "vision_sequence_indexes")
    action = _indexes(kwargs["action_sequence_indexes"], "action_sequence_indexes")
    targets = _indexes(kwargs["action_mse_loss_indexes"], "action_mse_loss_indexes")
    if targets.numel() == 0 or not bool(torch.isin(targets, action).all()):
        raise ValueError("Nonempty action prediction indexes must belong to action_sequence_indexes")
    if not torch.equal(torch.cat((vision, action)).sort().values, torch.arange(und_len, length)):
        raise ValueError("Vision/action rows must cover GEN exactly; padding/other modalities unsupported")
    return {"und_len": und_len, "gen_len": length - und_len,
            "action_rows": targets - und_len, "global_action_indexes": targets}


def _step_selection(steps: Any) -> tuple[int, ...] | None:
    if steps is None:
        return None
    result = tuple(steps)
    if (not result or any(type(s) is not int or s < 0 for s in result)
            or len(set(result)) != len(result)):
        raise ValueError("steps must contain unique nonnegative integer forward indexes")
    return tuple(sorted(result))


class CosmosComponentInterventions:
    """Capture two component outputs; patch one component at one layer."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.layers = list(model.layers)
        if not self.layers:
            raise ValueError("The model has no decoder layers")
        self.site_specs = {}
        for layer, block in enumerate(self.layers):
            if not hasattr(block, "self_attn") or not hasattr(block, "mlp_moe_gen"):
                raise TypeError("Expected Cosmos3 dual-pathway decoder layers")
            attn, mlp = block.self_attn, block.mlp_moe_gen
            for name in ("processor", "to_add_out", "num_attention_heads", "head_dim"):
                if not hasattr(attn, name):
                    raise TypeError(f"Missing Cosmos3 attention attribute {name}")
            if not hasattr(mlp, "down_proj"):
                raise TypeError("Expected GEN MLP with down_proj")
            self.site_specs[layer] = {
                "attention_output_width": attn.to_add_out.out_features,
                "attention_pre_projection_width": attn.to_add_out.in_features,
                "num_attention_heads": attn.num_attention_heads, "head_dim": attn.head_dim,
                "mlp_output_width": mlp.down_proj.out_features,
                "mlp_intermediate_width": mlp.down_proj.in_features,
            }
        self._kind = None
        self._handles = []

    def _start(self, kind: str) -> None:
        if self._kind is not None:
            raise RuntimeError("End/reset the active session first")
        if self.model.training or torch.is_grad_enabled():
            raise RuntimeError("Use model.eval() and torch.no_grad()/inference_mode()")
        if (getattr(self.model, "_cp_shard_fn", None) is not None
                or getattr(self.model, "_cp_gather_fn", None) is not None
                or any(getattr(b.self_attn.processor, "_parallel_config", None) is not None
                       for b in self.layers)):
            raise ValueError("Context-parallel/sharded execution is unsupported")
        self._kind, self._step, self._inside_forward = kind, -1, False
        self._visited, self._finished = set(), set()
        try:
            self._handles.append(self.model.register_forward_pre_hook(self._before_model, with_kwargs=True))
            self._handles.append(self.model.register_forward_hook(self._after_model))
            for layer, block in enumerate(self.layers):
                self._handles.append(block.self_attn.register_forward_hook(self._after_component(layer, "attention")))
                self._handles.append(block.mlp_moe_gen.register_forward_hook(self._after_component(layer, "mlp")))
        except BaseException:
            self.reset()
            raise

    def begin_capture(self, label: str, *, steps=None, expected_steps: int | None = None) -> None:
        selected = _step_selection(steps)
        if expected_steps is not None and (type(expected_steps) is not int or expected_steps <= 0):
            raise ValueError("expected_steps must be a positive integer")
        if selected is not None and expected_steps is not None and selected[-1] >= expected_steps:
            raise ValueError("Selected capture step is outside expected_steps")
        self._start("capture")
        self._selected, self._expected_steps = selected, expected_steps
        self._cache = {"version": 1, "label": label, "site_specs": _cpu(self.site_specs),
                       "n_layers": len(self.layers), "steps": [], "frames": {}, "cache_device": "cpu",
                       "components": _COMPONENTS, "selected_steps": selected,
                       "index_convention": "zero_based_layer_and_denoising_forward"}

    def begin_patch(self, recipient_cache: dict, *, layer: int, component: str,
                    steps, gain: float, donor_cache: dict | None = None) -> None:
        if self._kind is not None:
            raise RuntimeError("End/reset the active session first")
        selected = _step_selection(steps)
        if selected is None:
            raise ValueError("Explicit patch steps are required")
        if type(layer) is not int or not 0 <= layer < len(self.layers) or component not in _COMPONENTS:
            raise ValueError("Select one valid layer and attention or mlp; head/unit sites are unsupported")
        if isinstance(gain, bool) or not isinstance(gain, (int, float)) or not math.isfinite(gain):
            raise ValueError("gain must be a finite scalar")
        for cache in (recipient_cache,) if donor_cache is None else (recipient_cache, donor_cache):
            if (cache.get("version") != 1 or not cache.get("complete")
                    or cache.get("n_layers") != len(self.layers)
                    or not _equal(cache.get("site_specs"), self.site_specs)):
                raise ValueError("Incomplete/incompatible component capture")
            if selected[-1] >= cache["n_steps"]:
                raise ValueError("Selected patch step was not observed in the capture")
            for step in selected:
                frame = cache["frames"].get((step, layer, component))
                expected = (cache["steps"][step]["layout"]["action_rows"].numel(),
                            self.site_specs[layer][f"{component}_output_width"])
                if (not isinstance(frame, torch.Tensor) or tuple(frame.shape) != expected
                        or not bool(torch.isfinite(frame).all())):
                    raise ValueError(f"Missing/invalid {component} frame at step {step}, layer {layer}")
        if donor_cache is not None:
            if recipient_cache["n_steps"] != donor_cache["n_steps"]:
                raise ValueError("Recipient/donor denoising-forward counts differ")
            for step, (r, d) in enumerate(zip(recipient_cache["steps"], donor_cache["steps"])):
                if not _equal(r, d):
                    raise ValueError(f"Recipient/donor text, indexes, positions or schedule differ at step {step}")
            if not _equal(recipient_cache["first_model_kwargs"]["action_tokens"],
                          donor_cache["first_model_kwargs"]["action_tokens"]):
                raise ValueError("Recipient/donor initial action noise differs")
        self._start("patch")
        self._recipient, self._donor = recipient_cache, donor_cache
        self._selected, self._expected_steps = selected, recipient_cache["n_steps"]
        self._layer, self._component, self._gain = layer, component, float(gain)
        self._changes = {}

    def _before_model(self, module, args, kwargs):
        if args:
            raise ValueError("Call the transformer with keyword arguments")
        if self._inside_forward:
            raise RuntimeError("Nested/repeated transformer forward or an unfinished failed forward")
        self._inside_forward = True
        self._step += 1
        if self._expected_steps is not None and self._step >= self._expected_steps:
            raise ValueError("Extra model forward: check CFG or unexpected recomputation")
        self._layout = _layout(kwargs)
        frame = {"layout": self._layout, "metadata": {k: _cpu(kwargs.get(k)) for k in _STRUCTURAL_KEYS}}
        if self._kind == "capture":
            if self._step:
                first = self._cache["steps"][0]["metadata"]
                for key, value in frame["metadata"].items():
                    if key not in _TIMESTEP_KEYS and not _equal(value, first[key]):
                        raise ValueError(f"Capture pack/text changed at step {self._step}: {key}; CFG unsupported")
            else:
                self._cache["first_model_kwargs"] = _cpu(kwargs)
            self._cache["steps"].append(frame)
        else:
            if not _equal(frame, self._recipient["steps"][self._step]):
                raise ValueError(f"Patch pack/text/schedule differs from recipient at step {self._step}")
            if self._step == 0:
                first = self._recipient["first_model_kwargs"]
                if any(not _equal(kwargs.get(k), first.get(k)) for k in ("vision_tokens", "action_tokens")):
                    raise ValueError("Patch initial visual observation or action noise differs from recipient")

    def _after_model(self, module, args, output):
        expected = {(self._step, l, c) for l in range(len(self.layers)) for c in _COMPONENTS}
        if not expected.issubset(self._visited):
            raise RuntimeError("Transformer completed without every attention/GEN MLP component")
        self._finished.add(self._step)
        self._inside_forward = False

    def _after_component(self, layer: int, component: str):
        def after(module, args, output):
            if not self._inside_forward:
                raise RuntimeError("Component ran outside the tracked transformer forward")
            key = (self._step, layer, component)
            if key in self._visited:
                raise RuntimeError(f"Duplicate component call/recomputation: {key}")
            if component == "attention":
                if not isinstance(output, tuple) or len(output) != 2:
                    raise TypeError("Expected Cosmos3 (UND output, GEN output)")
                gen = output[1]
            else:
                gen = output
            expected_shape = (self._layout["gen_len"], self.site_specs[layer][f"{component}_output_width"])
            if not isinstance(gen, torch.Tensor) or tuple(gen.shape) != expected_shape:
                raise ValueError(f"Unexpected component shape at {key}; padding/sharding unsupported")
            self._visited.add(key)
            selected = self._selected is None or self._step in self._selected
            if self._kind == "capture":
                if selected:
                    rows = self._layout["action_rows"].to(gen.device)
                    self._cache["frames"][key] = _cpu(gen.index_select(0, rows))
                return
            if not selected or layer != self._layer or component != self._component:
                return
            rows = self._layout["action_rows"].to(gen.device)
            current = gen.index_select(0, rows)
            if self._donor is None:
                changed = current if self._gain == 1 else current * self._gain
            else:
                source = self._donor["frames"][key].to(device=gen.device, dtype=gen.dtype)
                if self._gain == 0 or torch.equal(current, source):
                    changed = current
                elif self._gain == 1:
                    changed = source
                else:
                    changed = current + self._gain * (source - current)
            before, after_cpu = _cpu(current), _cpu(changed)
            delta = after_cpu.float() - before.float()
            self._changes[key] = {"step": self._step, "layer": layer, "component": component,
                                  "shape": tuple(current.shape), "before": before, "after": after_cpu,
                                  "exact_no_change": torch.equal(before, after_cpu),
                                  "delta_l2": float(torch.linalg.vector_norm(delta))}
            if torch.equal(before, after_cpu):
                return output
            altered = gen.clone()
            altered.index_copy_(0, rows, changed)
            return (output[0], altered) if component == "attention" else altered
        return after

    def reset(self) -> None:
        """Remove only this helper's hooks and abandon any active session."""
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        self._kind, self._inside_forward = None, False
        for name in ("_cache", "_recipient", "_donor", "_changes"):
            if hasattr(self, name):
                delattr(self, name)

    def end(self) -> dict[str, Any]:
        """Clean up even after an exception; return diagnostics without raising."""
        if self._kind is None:
            raise RuntimeError("No active component session")
        steps = self._step + 1
        expected = {(s, l, c) for s in range(steps) for l in range(len(self.layers)) for c in _COMPONENTS}
        missing = sorted(expected - self._visited)
        unfinished = sorted(set(range(steps)) - self._finished)
        complete = steps > 0 and not missing and not unfinished
        if self._expected_steps is not None:
            complete = complete and steps == self._expected_steps
        selected = tuple(range(steps)) if self._selected is None else self._selected
        if selected and selected[-1] >= steps:
            complete = False
        common = {"complete": complete, "n_steps": steps, "n_layers": len(self.layers),
                  "expected_steps": self._expected_steps, "missing_component_calls": missing,
                  "unfinished_forwards": unfinished, "component_call_count": len(self._visited),
                  "processor_recompute_calls": 0, "output_exactness_checked_by_helper": False}
        if self._kind == "capture":
            result = self._cache
            saved = {(s, l, c) for s in selected for l in range(len(self.layers)) for c in _COMPONENTS}
            common["complete"] = complete and saved == set(result["frames"])
            result.update(common, frame_shapes={k: tuple(v.shape) for k, v in result["frames"].items()},
                          captured_component_count=len(result["frames"]))
        else:
            expected_changes = {(s, self._layer, self._component) for s in selected}
            common["complete"] = complete and expected_changes == set(self._changes)
            donor = self._donor
            result = dict(common, layer=self._layer, component=self._component, steps=selected,
                          gain=self._gain, mode="scale" if donor is None else "donor_blend",
                          changes=self._changes, intervention_count=len(self._changes),
                          recipient_label=self._recipient.get("label"),
                          donor_label=None if donor is None else donor.get("label"),
                          donor_initial_vision_equal=None if donor is None else _equal(
                              self._recipient["first_model_kwargs"]["vision_tokens"],
                              donor["first_model_kwargs"]["vision_tokens"]),
                          action_rows=_cpu(self._recipient["steps"][0]["layout"]["action_rows"]),
                          patch_site="component increment before decoder residual addition")
        self.reset()
        return result


def _cpu_check() -> None:
    """Meaningful controls on the installed Cosmos3 decoder, not a mock."""
    import copy
    import hashlib
    import inspect
    import json
    from pathlib import Path
    from diffusers.models.transformers.transformer_cosmos3 import (
        Cosmos3AttnProcessor, Cosmos3VLTextMoTDecoderLayer,
    )

    class TinyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(16, 8)
            self.layers = torch.nn.ModuleList([Cosmos3VLTextMoTDecoderLayer(
                hidden_size=8, head_dim=4, num_attention_heads=2, num_key_value_heads=1,
                intermediate_size=16, attention_bias=False, rms_norm_eps=1e-6,
            ) for _ in range(3)])
            self.fail = False
            self.repeat_component = None

        def forward(self, **kwargs):
            und = self.embed(kwargs["input_ids"])
            gen = torch.zeros(6, 8)
            gen[kwargs["vision_sequence_indexes"] - kwargs["und_len"]] = kwargs["vision_tokens"][0]
            gen[kwargs["action_sequence_indexes"] - kwargs["und_len"]] = kwargs["action_tokens"][0]
            rope = (torch.ones(3, 4), torch.zeros(3, 4), torch.ones(6, 4), torch.zeros(6, 4))
            for block in self.layers:
                und, gen = block(und, gen, rope)
            if self.repeat_component == "attention":
                self.layers[0].self_attn(und, gen, rope)
            elif self.repeat_component == "mlp":
                self.layers[0].mlp_moe_gen(gen)
            if self.fail:
                raise RuntimeError("deliberate failure after decoder")
            return und, gen

    if torch.cuda.is_initialized():
        raise RuntimeError("CPU check must run before any CUDA initialization")
    torch.manual_seed(19)
    torch.set_num_threads(2)
    checks = []
    with torch.inference_mode():
        model = TinyPolicy().eval()
        hooks = CosmosComponentInterventions(model)
        initial_vision, initial_action = torch.randn(3, 8), torch.randn(3, 8)

        def run(vision=None, n_steps=3):
            v = initial_vision.clone() if vision is None else vision.clone()
            a, outputs = initial_action.clone(), []
            for timestep in [999, 600, 300][:n_steps]:
                output = model(**dict(
                    und_len=3, sequence_length=9, input_ids=torch.tensor([1, 2, 3]),
                    text_indexes=torch.arange(3), position_ids=torch.arange(9),
                    vision_sequence_indexes=torch.tensor([3, 5, 7]), vision_mse_loss_indexes=torch.tensor([5, 7]),
                    vision_token_shapes=[(3, 1, 1)], vision_noisy_frame_indexes=[torch.tensor([1, 2])],
                    action_sequence_indexes=torch.tensor([4, 6, 8]), action_mse_loss_indexes=torch.tensor([8, 4]),
                    action_token_shapes=[(3, 1, 1)], action_noisy_frame_indexes=[torch.tensor([0, 2])],
                    action_domain_ids=[torch.tensor([0])], vision_timesteps=torch.tensor([timestep]),
                    action_timesteps=torch.tensor([timestep]), vision_tokens=[v], action_tokens=[a], sound_tokens=None,
                ))
                outputs.append(tuple(x.clone() for x in output))
                v = v - output[1][torch.tensor([0, 2, 4])] * 0.1
                a = a - output[1][torch.tensor([1, 3, 5])] * 0.1
            return outputs

        def capture(label, vision=None, steps=None):
            hooks.begin_capture(label, steps=steps, expected_steps=3)
            try:
                output = run(vision)
            finally:
                cache = hooks.end()
            assert cache["complete"] and cache["n_steps"] == 3
            return output, cache

        baseline, recipient = capture("R")
        _, donor = capture("D", initial_vision + 0.25)
        assert torch.equal(recipient["steps"][0]["layout"]["action_rows"], torch.tensor([5, 1]))
        assert len(recipient["frames"]) == 18
        checks.append("Actual decoder capture: 3 layers x 3 forwards x 2 components; interleaved/nonmonotonic noisy action rows")

        def patch(component, gain, source=None, n_steps=3):
            hooks.begin_patch(recipient, layer=1, component=component, steps=[0], gain=gain, donor_cache=source)
            try:
                output = run(n_steps=n_steps)
            finally:
                report = hooks.end()
            return output, report

        def exact(left, right):
            return len(left) == len(right) and all(torch.equal(x, y) for a, b in zip(left, right) for x, y in zip(a, b))

        assert exact(run(), baseline)
        checks.append("Capture hooks then end: unhooked repeat is bit-exact at every final UND/GEN output")
        for component in _COMPONENTS:
            for gain, source in [(1, None), (0, donor), (1, recipient)]:
                output, report = patch(component, gain, source)
                assert report["complete"] and exact(output, baseline)
                assert all(v["exact_no_change"] for v in report["changes"].values())
            checks.append(f"{component}: scale-one, donor-zero and self-patch are bit-exact at final outputs over feedback steps")
            for gain, source in [(0, None), (0.5, None), (1.5, None), (1, donor)]:
                output, report = patch(component, gain, source)
                assert report["complete"] and report["intervention_count"] == 1
                assert not exact(output, baseline) and report["changes"][(0, 1, component)]["delta_l2"] > 0
                assert all(torch.equal(x[0], y[0]) for x, y in zip(output, baseline))
            checks.append(f"{component}: brief attenuation/amplification/donor patch changes downstream GEN; UND preserved")
        # Observe component outputs AFTER the intervention hook to verify the
        # immediate rows and actual decoder residual addition, not just norms.
        def observe_run(component=None):
            observations = {k: [] for k in ("input", "attention", "mlp", "layer_output")}
            if component is not None:
                hooks.begin_patch(recipient, layer=1, component=component, steps=[0], gain=0)
            block = model.layers[1]
            probes = [
                block.register_forward_pre_hook(lambda m, a: observations["input"].append(a[1].clone())),
                block.self_attn.register_forward_hook(lambda m, a, o: observations["attention"].append(o[1].clone())),
                block.mlp_moe_gen.register_forward_hook(lambda m, a, o: observations["mlp"].append(o.clone())),
                block.register_forward_hook(lambda m, a, o: observations["layer_output"].append(o[1].clone())),
            ]
            try:
                run()
            finally:
                for probe in probes:
                    probe.remove()
                if component is not None:
                    assert hooks.end()["complete"]
            return observations

        native = observe_run()
        target, other = torch.tensor([5, 1]), torch.tensor([0, 2, 3, 4])
        for component in _COMPONENTS:
            changed = observe_run(component)
            assert torch.equal(changed["input"][0], native["input"][0])
            assert torch.equal(changed[component][0][other], native[component][0][other])
            assert torch.equal(changed[component][0][target], torch.zeros(2, 8))
            residual = changed["input"][0] + changed["attention"][0]
            assert torch.equal(changed["layer_output"][0], residual + changed["mlp"][0])
            if component == "mlp":
                assert torch.equal(changed["layer_output"][0][target], residual[target])
                assert not torch.equal(residual[target], torch.zeros(2, 8))
            checks.append(f"{component}: only selected action rows changed immediately; actual decoder residual sum preserved")

        _, selective = capture("selective", steps=[1])
        assert len(selective["frames"]) == 6 and selective["component_call_count"] == 18
        checks.append("Selective capture saves requested step while validating all calls")
        _, incomplete = patch("mlp", 0, n_steps=2)
        assert not incomplete["complete"] and not hooks._handles
        checks.append("Truncated prediction is incomplete and hooks are removed")
        bad = copy.deepcopy(donor)
        bad["steps"][0]["metadata"]["position_ids"][0] += 1
        try:
            hooks.begin_patch(recipient, layer=1, component="mlp", steps=[0], gain=1, donor_cache=bad)
            raise AssertionError("Misaligned donor accepted")
        except ValueError:
            pass
        assert not hooks._handles
        checks.append("Misaligned donor is rejected before hook registration")
        bad = copy.deepcopy(donor)
        bad["frames"][(0, 1, "mlp")] = bad["frames"][(0, 1, "mlp")][:1]
        try:
            hooks.begin_patch(recipient, layer=1, component="mlp", steps=[0], gain=1, donor_cache=bad)
            raise AssertionError("Wrong donor shape accepted")
        except ValueError:
            pass
        assert not hooks._handles
        checks.append("Wrong donor row count is rejected before hook registration")
        hooks.begin_capture("invalid-index", expected_steps=3)
        bad_kwargs = copy.deepcopy(recipient["first_model_kwargs"])
        bad_kwargs["action_mse_loss_indexes"] = torch.tensor([3])
        try:
            model(**bad_kwargs)
            raise AssertionError("Vision index accepted as action target")
        except ValueError:
            pass
        finally:
            invalid = hooks.end()
        assert not invalid["complete"] and not hooks._handles
        checks.append("Actual model-prehook rejects non-action target indexes and cleanup works before any layer")
        hooks.begin_capture("failure", expected_steps=3)
        model.fail = True
        try:
            run()
            raise AssertionError("Deliberate failure not raised")
        except RuntimeError:
            pass
        finally:
            failed = hooks.end()
            model.fail = False
        assert not failed["complete"] and failed["unfinished_forwards"] == [0]
        assert all(not x._forward_hooks and not x._forward_pre_hooks for x in model.modules())
        checks.append("Failure after complete decoder marks forward incomplete and removes every owned hook")
        for component in _COMPONENTS:
            hooks.begin_capture("duplicate-guard", expected_steps=3)
            model.repeat_component = component
            try:
                run()
                raise AssertionError("Duplicate component call accepted")
            except RuntimeError as exc:
                assert "Duplicate component call" in str(exc)
            finally:
                duplicate = hooks.end()
                model.repeat_component = None
            assert not duplicate["complete"] and not hooks._handles
            checks.append(f"Repeated {component} call/recomputation rejected and hooks removed")
        hooks.begin_capture("reset")
        hooks.reset()
        assert hooks._kind is None and not hooks._handles
        checks.append("reset abandons a session and removes hooks")
    source = Path(inspect.getfile(Cosmos3AttnProcessor))
    print(json.dumps({"status": "passed", "scope": "CPU actual installed Cosmos3 decoder, tiny random weights; no checkpoint/MuJoCo",
                      "torch": torch.__version__, "cuda_initialized": torch.cuda.is_initialized(),
                      "decoder_source": str(source), "decoder_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                      "checks": checks}, ensure_ascii=False))


if __name__ == "__main__":
    import sys
    if sys.argv[1:] != ["--cpu-check"]:
        raise SystemExit("Usage: python cosmos_component_interventions.py --cpu-check")
    _cpu_check()
