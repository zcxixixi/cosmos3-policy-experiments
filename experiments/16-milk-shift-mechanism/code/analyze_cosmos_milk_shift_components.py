"""Analyze nine completed normal milk-position captures on CPU.

This is paired observational vector geometry, not a causal intervention.
No layer residual was saved: attention/MLP delta alignment cannot establish
that an MLP suppressed a target representation or changed object identity.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


SEEDS = (195, 196, 198)
POSITIONS = ("x00", "x06", "x15")
CONTRASTS = (("x00", "x06"), ("x06", "x15"))
COMPONENTS = ("attention", "mlp")
STEPS, LAYERS, ROWS, WIDTH = 30, 36, 16, 4096
INPUT_KEYS = (
    "und_len", "sequence_length", "input_ids", "text_indexes", "position_ids",
    "vision_sequence_indexes", "vision_mse_loss_indexes", "vision_token_shapes",
    "vision_noisy_frame_indexes", "action_sequence_indexes", "action_mse_loss_indexes",
    "action_token_shapes", "action_noisy_frame_indexes", "action_domain_ids",
    "vision_timesteps", "action_timesteps",
)


def require_exact(left, right, label):
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        assert isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor), label
        assert left.dtype == right.dtype and left.shape == right.shape, label
        assert torch.equal(left, right), label
        assert torch.equal(left.contiguous().reshape(-1).view(torch.uint8),
                           right.contiguous().reshape(-1).view(torch.uint8)), label + "/bytes"
    elif isinstance(left, dict) or isinstance(right, dict):
        assert isinstance(left, dict) and isinstance(right, dict) and left.keys() == right.keys(), label
        for key in left:
            require_exact(left[key], right[key], f"{label}/{key}")
    elif isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right), label
        for index, (a, b) in enumerate(zip(left, right)):
            require_exact(a, b, f"{label}/{index}")
    else:
        assert left == right, label


def input_structure(record):
    return {key: record["model_input"].get(key) for key in INPUT_KEYS}


def load_trial(root, position, seed):
    folder = root / "closed-loop" / f"{position}_seed{seed}" / "chunk_00"
    record = torch.load(folder / "states.pt", map_location="cpu", weights_only=True, mmap=True)
    cache = torch.load(folder / "components.pt", map_location="cpu", weights_only=True, mmap=True)
    label = f"{position}_seed{seed}"
    assert cache["version"] == 1 and cache["complete"] is True, label
    assert cache["n_steps"] == len(cache["steps"]) == STEPS, label
    assert cache["n_layers"] == LAYERS and cache["label"] == label, label
    expected = {(s, l, c) for s in range(STEPS) for l in range(LAYERS) for c in COMPONENTS}
    assert set(cache["frames"]) == expected, label + "/frames"
    assert cache["captured_component_count"] == cache["component_call_count"] == len(expected), label
    assert not cache["missing_component_calls"] and not cache["unfinished_forwards"], label
    require_exact(cache["first_model_kwargs"], record["model_input"], label + "/observer_inputs")
    for step in cache["steps"]:
        assert step["layout"]["action_rows"].numel() == ROWS, label + "/action_rows"
    for layer in range(LAYERS):
        for component in COMPONENTS:
            assert cache["site_specs"][layer][f"{component}_output_width"] == WIDTH, label
    for key, frame in cache["frames"].items():
        assert isinstance(frame, torch.Tensor) and frame.shape == (ROWS, WIDTH), (label, key)
        assert torch.isfinite(frame).all(), (label, key)
    for key, shape in (("readout", (STEPS, ROWS, WIDTH)),
                       ("action_velocity", (STEPS, ROWS, 64)), ("actions", (ROWS, 10))):
        assert record[key].shape == shape and torch.isfinite(record[key]).all(), (label, key)
    assert record["timesteps"].numel() == STEPS and torch.isfinite(record["sigmas"]).all(), label
    assert len(record["pure_noise"]) == 2, label
    assert all(value.dtype == torch.float32 for value in record["pure_noise"]), label
    assert record["pure_noise"][0].ndim == 5 and record["pure_noise"][1].shape == (ROWS, 64), label
    return record, cache


def check_pair(left, right, label):
    a, ac = left
    b, bc = right
    require_exact(a["pure_noise"], b["pure_noise"], label + "/both_pure_noise_draws")
    for key in ("timesteps", "sigmas"):
        require_exact(a[key], b[key], label + "/" + key)
    require_exact(input_structure(a), input_structure(b), label + "/text_pack_and_initial_time")
    require_exact(ac["steps"], bc["steps"], label + "/all_step_text_pack_and_time")
    require_exact(ac["site_specs"], bc["site_specs"], label + "/component_sites")
    require_exact(a["model_input"]["action_tokens"], b["model_input"]["action_tokens"],
                  label + "/initial_action_sample")


def geometry(left, right):
    _, a = left
    _, b = right
    result = {name: np.empty((STEPS, LAYERS), dtype=np.float64)
              for name in ("attention_delta_l2", "mlp_delta_l2", "delta_cosine",
                           "increment_delta_sum_l2", "cancellation_ratio", "candidate_score")}
    for step in range(STEPS):
        for layer in range(LAYERS):
            da = (b["frames"][(step, layer, "attention")].float()
                  - a["frames"][(step, layer, "attention")].float()).flatten()
            dm = (b["frames"][(step, layer, "mlp")].float()
                  - a["frames"][(step, layer, "mlp")].float()).flatten()
            an, mn = float(torch.linalg.vector_norm(da)), float(torch.linalg.vector_norm(dm))
            total = float(torch.linalg.vector_norm(da + dm))
            cosine = float(torch.dot(da, dm) / (an * mn)) if an > 0 and mn > 0 else float("nan")
            if np.isfinite(cosine):
                cosine = float(np.clip(cosine, -1, 1))
            values = (an, mn, cosine, total, total / (an + mn) if an + mn > 0 else float("nan"),
                      max(0.0, -cosine) * min(an, mn) if np.isfinite(cosine) else 0.0)
            for name, value in zip(result, values):
                result[name][step, layer] = value
    return result


def finite_number(value):
    return float(value) if np.isfinite(value) else None


def plot_maps(destination, name, matrices, title, colorbar, *, cosine=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plotted = matrices if cosine else np.log10(np.maximum(matrices, 1e-12))
    limits = (-1, 1) if cosine else (float(np.nanmin(plotted)), float(np.nanmax(plotted)))
    if limits[0] == limits[1]:
        limits = (limits[0] - 0.5, limits[1] + 0.5)
    figure, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True, constrained_layout=True)
    for index, axis in enumerate(axes):
        display = axis.imshow(np.ma.masked_invalid(plotted[index]), aspect="auto", origin="lower",
                              extent=(0.5, LAYERS + 0.5, -0.5, STEPS - 0.5),
                              cmap="coolwarm" if cosine else "viridis", vmin=limits[0], vmax=limits[1])
        axis.set(title=f"{CONTRASTS[index][0]} → {CONTRASTS[index][1]}",
                 xlabel="Layer (1–36)", xticks=(1, 6, 12, 18, 24, 30, 36))
    axes[0].set_ylabel("Denoising forward index (0–29)")
    figure.colorbar(display, ax=axes, label=colorbar, shrink=0.9)
    figure.suptitle(title + "\nMedian of paired seeds 195 / 196 / 198; observational only")
    figure.savefig(destination / name, dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Existing experiment root, not analysis folder")
    parser.add_argument("--plot-only", action="store_true", help="Render already saved validated metrics; do not recompute or overwrite them")
    args = parser.parse_args()
    root = args.output
    destination = root / "component-analysis"
    if args.plot_only:
        summary = json.loads((destination / "summary.json").read_text())
        assert summary["complete_captures"] == 9 and len(summary["checks"]) == 6
        with np.load(destination / "component-deltas.npz", allow_pickle=False) as values:
            for name, key, title, label, cosine in (
                ("attention-delta.png", "median_attention_delta_l2", "Attention increment delta magnitude", "log10 L2; zero floor = 1e-12", False),
                ("mlp-delta.png", "median_mlp_delta_l2", "GEN MLP increment delta magnitude", "log10 L2; zero floor = 1e-12", False),
                ("attention-mlp-cosine.png", "median_delta_cosine", "Attention/MLP delta cosine: candidate opposing increments", "Cosine; undefined cells masked", True),
            ):
                assert not (destination / name).exists(), name
                plot_maps(destination, name, values[key], title, label, cosine=cosine)
        (destination / "plots_complete.json").write_text(json.dumps(dict(state="complete", complete_captures=9, figures=3)) + "\n")
        return
    assert not destination.exists(), f"Preserve existing analysis: {destination}"
    for seed in SEEDS:
        for position in POSITIONS:
            for filename in ("states.pt", "components.pt"):
                path = root / "closed-loop" / f"{position}_seed{seed}" / "chunk_00" / filename
                assert path.is_file(), f"Require all nine completed captures first: {path}"
    torch.set_num_threads(4)
    metrics, output_metrics, checks = {}, {}, []
    common = None
    for seed_index, seed in enumerate(SEEDS):
        trials = {position: load_trial(root, position, seed) for position in POSITIONS}
        for position, (record, cache) in trials.items():
            structure = dict(timesteps=record["timesteps"], sigmas=record["sigmas"],
                             model_input=input_structure(record), steps=cache["steps"])
            if common is None:
                common = structure
            else:
                require_exact(structure, common, f"common_schedule_text_pack/{position}/{seed}")
        # All pairing assertions precede any geometry for this seed.
        for first, second in CONTRASTS:
            check_pair(trials[first], trials[second], f"{first}_to_{second}/seed{seed}")
            checks.append(dict(contrast=f"{first}_to_{second}", seed=seed,
                               both_pure_noise_draws_exact=True, schedule_text_pack_all_steps_exact=True))
        for contrast_index, (first, second) in enumerate(CONTRASTS):
            pair = geometry(trials[first], trials[second])
            for name, value in pair.items():
                if name not in metrics:
                    metrics[name] = np.empty((2, 3, STEPS, LAYERS), dtype=np.float64)
                metrics[name][contrast_index, seed_index] = value
            a, b = trials[first][0], trials[second][0]
            readout = (b["readout"].float() - a["readout"].float()).flatten(1).norm(dim=1).numpy()
            velocity = (b["action_velocity"][:, :, :10].float()
                        - a["action_velocity"][:, :, :10].float()).flatten(1).norm(dim=1).numpy()
            actions = (b["actions"].float() - a["actions"].float()).numpy()
            for name, value in (("readout_delta_l2", readout), ("velocity10_delta_l2", velocity),
                                ("final_action10_delta", actions)):
                if name not in output_metrics:
                    output_metrics[name] = np.empty((2, 3) + value.shape, dtype=np.float64)
                output_metrics[name][contrast_index, seed_index] = value
        print(f"Validated and analyzed seed {seed} (3 complete captures)", flush=True)
    medians = {name: np.median(values, axis=1) for name, values in metrics.items()}
    rankings = {}
    for contrast_index, (first, second) in enumerate(CONTRASTS):
        ranked = []
        order = np.argsort(medians["candidate_score"][contrast_index].ravel())[::-1]
        for flat_index in order:
            step, layer = np.unravel_index(int(flat_index), (STEPS, LAYERS))
            score = medians["candidate_score"][contrast_index, step, layer]
            cosine = medians["delta_cosine"][contrast_index, step, layer]
            if score <= 0 or not np.isfinite(cosine) or cosine >= 0:
                continue
            ranked.append(dict(
                denoising_forward_index=int(step), layer_one_based=int(layer + 1),
                timestep=int(common["timesteps"][step]), sigma=float(common["sigmas"][step]),
                median={name: finite_number(values[contrast_index, step, layer]) for name, values in medians.items()},
                seeds=[dict(seed=seed, **{name: finite_number(values[contrast_index, i, step, layer])
                                          for name, values in metrics.items()}) for i, seed in enumerate(SEEDS)],
                all_three_cosines_negative=bool(np.all(metrics["delta_cosine"][contrast_index, :, step, layer] < 0))))
            if len(ranked) == 12:
                break
        rankings[f"{first}_to_{second}"] = ranked
    summary = dict(
        scope="Nine first-query normal baseline captures; observational component delta geometry only.",
        seeds=list(SEEDS), contrasts=[f"{a}_to_{b}" for a, b in CONTRASTS],
        axes="NPZ component arrays: contrast, seed, denoising_forward_index, zero_based_layer",
        checks=checks, complete_captures=9,
        delta_definition="second position minus first position at the same seed, step, layer and actual action rows",
        norm_definition="L2 over all 16 x 4096 GEN action component output coordinates",
        median_definition="Median of three per-seed measurements; undefined cosine at any seed masks its median cell",
        candidate_ranking="Descending median(max(0, -cosine) * min(attention_delta_l2, mlp_delta_l2)); heuristic only",
        limitations=[
            "No residual states were saved: delta_attention + delta_mlp is component vector geometry, not measured residual attenuation.",
            "Negative cosine does not demonstrate MLP overwrite, target selection or causal competition.",
            "Layer and denoising-step dynamics differ; candidate sites require local interventions on paired inputs.",
            "Only three paired seeds and the first policy query; no causal or closed-loop mechanism claim.",
        ],
        candidate_sites=rankings,
        output_differences={f"{a}_to_{b}": [dict(seed=seed,
            final_action10_rmse=float(np.sqrt(np.mean(output_metrics["final_action10_delta"][c, i] ** 2))),
            final_action_xyz_rmse=float(np.sqrt(np.mean(output_metrics["final_action10_delta"][c, i, :, :3] ** 2))),
            first_velocity10_delta_l2=float(output_metrics["velocity10_delta_l2"][c, i, 0]),
            last_velocity10_delta_l2=float(output_metrics["velocity10_delta_l2"][c, i, -1]))
            for i, seed in enumerate(SEEDS)] for c, (a, b) in enumerate(CONTRASTS)},
        action_units="Normalized action coordinates; RMSE is not meters or target probability.",
    )
    destination.mkdir(exist_ok=False)
    np.savez_compressed(destination / "component-deltas.npz", **metrics, **output_metrics,
                        **{f"median_{name}": values for name, values in medians.items()},
                        seeds=np.asarray(SEEDS), timesteps=common["timesteps"].numpy(),
                        sigmas=common["sigmas"].float().numpy(), contrasts=np.asarray(summary["contrasts"]))
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    plot_maps(destination, "attention-delta.png", medians["attention_delta_l2"],
              "Attention increment delta magnitude", "log10 L2; zero floor = 1e-12")
    plot_maps(destination, "mlp-delta.png", medians["mlp_delta_l2"],
              "GEN MLP increment delta magnitude", "log10 L2; zero floor = 1e-12")
    plot_maps(destination, "attention-mlp-cosine.png", medians["delta_cosine"],
              "Attention/MLP delta cosine: candidate opposing increments", "Cosine; undefined cells masked", cosine=True)
    print(f"Saved {destination} (NPZ, summary JSON, 3 PNGs); candidates are observational clues.", flush=True)


if __name__ == "__main__":
    main()
