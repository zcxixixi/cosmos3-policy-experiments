"""Summarize four fixed layouts and three policy-noise seeds from real trajectories."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository", type=Path)
    args = parser.parse_args()
    repo = args.repository
    parent = repo / "experiments/10-identity-position"
    e09 = repo / "experiments/09-system-prompt/closed-loop"
    originals = {x["case"]: x for x in json.loads((e09 / "result.json").read_text())["cases"]}
    swaps = {x["case"]: x for x in json.loads((parent / "result.json").read_text())["cases"]}
    repeats = json.loads((parent / "repeats/result.json").read_text())["cases"]
    rows = [
        ("center_original", 198, e09 / "center_milk_off", originals["center_milk_off"], "reused_experiment09"),
        ("far_original", 198, e09 / "far_milk_off", originals["far_milk_off"], "reused_experiment09"),
        ("center_swapped", 198, parent / "center_cream_butter_swap_milk_off", swaps["center_cream_butter_swap_milk_off"], "initial_experiment10"),
        ("far_swapped", 198, parent / "far_cream_butter_swap_milk_off", swaps["far_cream_butter_swap_milk_off"], "initial_experiment10"),
    ]
    rows += [(x["scene"], x["seed"], parent / "repeats" / x["case"], x, "new_noise_repeat") for x in repeats]
    snapshots = {scene: directory / "initial_state.npy" for scene, seed, directory, _, _ in rows if seed == 198}
    output = []
    for scene, seed, directory, recorded, provenance in rows:
        trajectory = json.loads((directory / "trajectory.json").read_text())
        assert len(trajectory) == 129
        objects = {}
        selected = []
        for name in trajectory[0]["objects"]:
            height = np.array([x["objects"][name][2] for x in trajectory])
            lift = 100 * (height - height[0])
            contact = [i for i, x in enumerate(trajectory) if name in x["grasped"]]
            above = (lift > 2) & np.array([name in x["grasped"] for x in trajectory])
            sustained = next((i for i in range(len(above) - 4) if np.all(above[i:i + 5])), None)
            assert abs(float(lift.max()) - recorded["per_object"][name]["max_lift_cm"]) < 1e-9
            assert len(contact) == recorded["per_object"][name]["grasp_frames"]
            objects[name] = {"first_contact_frame": contact[0] if contact else None,
                             "contact_frames": len(contact), "max_lift_cm": float(lift.max()),
                             "first_2cm_for_5frames": sustained}
            if contact and sustained is not None:
                selected.append(name)
        assert len(selected) == 1, (scene, seed, selected)
        initial_path = snapshots[scene]
        output.append({"scene": scene, "noise_seed_base": seed,
                       "selected_object": selected[0], "source": directory.relative_to(repo).as_posix(),
                       "provenance": provenance, "steps": 128, "state_records": 129,
                       "initial_state_source": initial_path.relative_to(repo).as_posix(),
                       "initial_state_sha256": digest(initial_path),
                       "trajectory_sha256": digest(directory / "trajectory.json"),
                       "final_milk_in_basket": bool(trajectory[-1]["milk_in_basket"]),
                       "per_object": objects})
    groups = []
    for scene in ["center_original", "center_swapped", "far_original", "far_swapped"]:
        items = sorted((x for x in output if x["scene"] == scene), key=lambda x: x["noise_seed_base"])
        identities = [x["selected_object"] for x in items]
        assert len(set(identities)) == 1 and len(items) == 3
        lifts = [x["per_object"][x["selected_object"]]["max_lift_cm"] for x in items]
        groups.append({"scene": scene, "seeds": [x["noise_seed_base"] for x in items],
                       "selected_objects": identities, "max_lift_range_cm": [min(lifts), max(lifts)],
                       "final_milk_in_basket_count": sum(x["final_milk_in_basket"] for x in items)})
    result = {"scope": "Four fixed simulation snapshots, each repeated with three policy-noise seeds. Not twelve independent scenes or a benchmark success-rate estimate.",
              "instruction": "pick up the milk and place it in the basket", "system_prompt": False,
              "selection_criterion": "Both-finger contact plus >2cm lift for >=5 simulation records; recorded full trajectories, not generated videos.",
              "groups": groups, "cases": output}
    (parent / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(groups, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
