"""CPU tests with the actual installed Diffusers Cosmos3 decoder/processor.

Run in the existing Cosmos3 Python environment, beside the helper. This test
uses tiny random weights and three state-feedback steps, not the 8B checkpoint
or MuJoCo. It prints one JSON result; the calling runner may archive stdout.
"""

import copy
import hashlib
import inspect
import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import torch
from diffusers.models.transformers.transformer_cosmos3 import (
    Cosmos3AttnProcessor,
    Cosmos3VLTextMoTDecoderLayer,
)
from cosmos_attention_routes import CosmosAttentionRoutes


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(16, 8)
        self.layers = torch.nn.ModuleList([
            Cosmos3VLTextMoTDecoderLayer(
                hidden_size=8, head_dim=4, num_attention_heads=2,
                num_key_value_heads=1, intermediate_size=16,
                attention_bias=False, rms_norm_eps=1e-6,
            ) for _ in range(3)
        ])
        self.logs = []
        self.fail_after_layers = False

    def forward(self, **kwargs):
        und = self.embed(kwargs["input_ids"])
        gen = torch.cat((kwargs["vision_tokens"][0], kwargs["action_tokens"][0]))
        rope = (torch.ones(3, 4), torch.zeros(3, 4), torch.ones(4, 4), torch.zeros(4, 4))
        log = []
        for layer in self.layers:
            und, gen = layer(und, gen, rope)
            log.append((und.clone(), gen.clone()))
        self.logs.append(log)
        if self.fail_after_layers:
            raise RuntimeError("deliberate final decode failure")
        return und, gen


def main():
    torch.manual_seed(19)
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    model = TinyPolicy().eval()
    routes = CosmosAttentionRoutes(model)
    initial_v, initial_a = torch.randn(2, 8), torch.randn(2, 8)
    checks = []

    def kwargs(noun, timestep, vision, action):
        return dict(
            und_len=3, sequence_length=7, input_ids=torch.tensor([1, noun, 3]),
            text_indexes=torch.arange(3), position_ids=torch.arange(7),
            vision_sequence_indexes=torch.tensor([3, 4]), vision_mse_loss_indexes=torch.tensor([4]),
            vision_token_shapes=[(2, 1, 1)], vision_noisy_frame_indexes=[torch.tensor([1])],
            action_sequence_indexes=torch.tensor([5, 6]), action_mse_loss_indexes=torch.tensor([5, 6]),
            action_token_shapes=[(2, 1, 1)], action_noisy_frame_indexes=[torch.tensor([0, 1])],
            action_domain_ids=[torch.tensor([0])], vision_timesteps=torch.tensor([timestep]),
            action_timesteps=torch.tensor([timestep]), vision_tokens=[vision], action_tokens=[action],
            sound_tokens=None,
        )

    def run(noun, steps=3):
        vision, action = initial_v.clone(), initial_a.clone()
        outputs, model.logs = [], []
        for timestep in [999, 600, 300][:steps]:
            und, gen = model(**kwargs(noun, timestep, vision, action))
            outputs.append((und.clone(), gen.clone()))
            vision = torch.cat((initial_v[:1], vision[1:] - gen[1:2] * .1))
            action = action - gen[2:] * .1
        return outputs, copy.deepcopy(model.logs)

    def capture(label, noun):
        routes.begin_capture(label)
        try:
            output, logs = run(noun)
        finally:
            cache = routes.end()
        assert cache["complete"] and cache["n_steps"] == 3
        return output, logs, cache

    r, rlogs, recipient = capture("R", 2)
    d, dlogs, donor = capture("D", 4)
    assert all(recipient["und_identical_across_steps"].values())
    checks.append("CPU full K/V cache; observed UND equality across steps")

    def patch(mode, source=donor, gain=1, steps=3):
        routes.begin_patch(recipient, source, mode, gain)
        try:
            output, logs = run(2, steps)
        finally:
            report = routes.end()
        return output, logs, report

    def gen_exact(a, b):
        return len(a) == len(b) and all(torch.equal(x[1], y[1]) for x, y in zip(a, b))

    for mode, source, gain in [
        ("empty", donor, 1), ("direct", donor, 0),
        ("full", recipient, 1), ("clamp_self", donor, 1),
    ]:
        out, _, report = patch(mode, source, gain)
        assert report["complete"] and gen_exact(out, r), mode
        checks.append(f"{mode}, gain={gain}: GEN exact R")

    full, flogs, report = patch("full")
    assert report["complete"] and gen_exact(full, d)
    assert all(torch.equal(flogs[s][l][1], dlogs[s][l][1]) for s in range(3) for l in range(3))
    assert all(torch.equal(full[s][0], r[s][0]) for s in range(3))
    checks.append("Full: every layer/step GEN exact D; UND exact R")

    blocked, _, report = patch("indirect_blocked")
    assert report["complete"]
    assert all(torch.equal(blocked[s][1][2:], r[s][1][2:]) for s in range(3))
    assert not gen_exact(blocked, r)
    checks.append("Indirect-blocked: action exact R; future actually changed")

    indirect, ilogs, report = patch("indirect")
    assert report["complete"]
    assert torch.equal(ilogs[0][0][1][2:], rlogs[0][0][1][2:])
    assert not torch.equal(indirect[0][1][2:], r[0][1][2:])
    checks.append("Indirect: first-layer action exact R; later action changed")
    for mode in ("direct", "joint"):
        out, _, report = patch(mode)
        assert report["complete"] and not gen_exact(out, r)
        _, _, incomplete = patch(mode, steps=2)
        assert not incomplete["complete"]
    checks.append("Direct/Joint nonzero; truncated patch reports incomplete")

    bad = copy.deepcopy(donor)
    bad["steps"][0]["metadata"]["position_ids"][0] += 1
    try:
        routes.begin_patch(recipient, bad, "full")
        raise AssertionError("Misaligned pack accepted")
    except ValueError:
        pass
    checks.append("Recipient/donor position mismatch rejected before hooks")

    bad = copy.deepcopy(donor)
    bad["frames"][(0, 0)]["und_k"] = bad["frames"][(0, 0)]["und_k"][:2]
    routes.begin_patch(recipient, bad, "full")
    try:
        run(2)
        raise AssertionError("Bad projection accepted")
    except ValueError:
        pass
    finally:
        incomplete = routes.end()
    assert not incomplete["complete"] and not routes._handles and not routes._recomputing
    checks.append("Projection failure removes temporary and session hooks")

    routes.begin_capture("failure")
    model.fail_after_layers = True
    try:
        run(2)
        raise AssertionError("Expected final decode failure")
    except RuntimeError:
        pass
    finally:
        incomplete = routes.end()
        model.fail_after_layers = False
    assert not incomplete["complete"] and incomplete["unfinished_forwards"] == [0]
    checks.append("Failure after all layers still marks forward incomplete")

    routes.begin_capture("CFG-guard")
    model(**kwargs(2, 999, initial_v, initial_a))
    try:
        model(**kwargs(4, 999, initial_v, initial_a))
        raise AssertionError("Changed caption accepted inside capture")
    except ValueError:
        pass
    finally:
        incomplete = routes.end()
    assert not incomplete["complete"]
    assert all(not x._forward_hooks and not x._forward_pre_hooks for x in model.modules())
    checks.append("Static capture caption change rejected; all owned hooks cleaned up")

    source_file = Path(inspect.getfile(Cosmos3AttnProcessor))
    print(json.dumps(dict(
        status="passed", scope="CPU tiny actual Cosmos3 decoder, 3 layers x 3 feedback steps; no checkpoint/MuJoCo",
        torch=torch.__version__, cuda_initialized=torch.cuda.is_initialized(),
        processor_source=str(source_file), processor_sha256=hashlib.sha256(source_file.read_bytes()).hexdigest(),
        checks=checks,
    ), ensure_ascii=False))


if __name__ == "__main__":
    main()
