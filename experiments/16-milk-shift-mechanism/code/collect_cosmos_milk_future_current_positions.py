"""Capture X+15cm and X+18cm scenes without policy or simulator steps.

Reuse the frozen matched-scene collector. X+0cm is only its CPU restore/photo
reference; it is not an additional policy trial. Accepted prior arrays remain
immutable. The separate producer will test a fixed first-nine-layer cut.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path('/home/current/work/cosmos3')
SOURCES = {
    'simulator': ('rollout_cosmos_milk_shift_mechanism.py', '7060f610814cbf6b13fb79ca2da78af73bba542d479ac2b8a73110097543d845'),
    'collector': ('collect_cosmos_goal_pair.py', 'b06454696a61b2ca9c570cbeac3f3d8149a4bf5179f51dafdb75a5d70cbbc945'),
    'controller': ('rollout_cosmos_goal_pair.py', 'bc18a41d97dcdf7c321ef093faf4eea076678298f269efa5cf8fc9c623157dc7'),
}
SHIFTS = {'x00': 0, 'x15': 15, 'x18': 18}
PROMPT = 'pick up the milk and place it in the basket'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    stage = args.output.resolve() / 'future-current-position-inputs'
    assert args.output.resolve().is_dir()
    assert not stage.exists(), 'No overwrite or partial retry'
    files = {name: ROOT / 'work' / pair[0] for name, pair in SOURCES.items()}
    for name, path in files.items():
        assert sha(path) == SOURCES[name][1], (name, 'frozen source changed')
    stage.mkdir(exist_ok=False)
    protocol = dict(script_sha256=sha(Path(__file__)), source_files={name: dict(path=str(path), sha256=sha(path)) for name, path in files.items()},
        shifts_cm=SHIFTS, policy_scenes=['x15', 'x18'], reference_only_scene='x00', prompt=PROMPT,
        no_env_step_during_collection_or_restore=True, policy_model_forwards=0,
        fixed_followup=dict(vision_source_seeds=[195, 198], action_source_seed=195,
            arms=['native', 'all_allowed', 'early_l1_9'], fresh_q0_predictions=12,
            selected_window=dict(steps=[0, 14], layers=[0, 8]), no_selection_by_outcome=True))
    write(stage / 'protocol.json', protocol)
    try:
        os.environ.setdefault('MUJOCO_GL', 'egl')
        sys.path.insert(0, '/home/current/work/openpi-demo/cosmos-policy')
        sys.path.insert(0, str(ROOT / 'cosmos-framework'))
        simulator = load('position_inputs_frozen_simulator', files['simulator'])
        collector = load('position_inputs_frozen_collector', files['collector'])
        controller = load('position_inputs_frozen_controller', files['controller'])
        assert simulator.PROMPT == PROMPT
        simulator.SHIFTS = SHIFTS.copy()
        simulator.SEEDS = (195,)
        simulator.collect_inputs(stage, collector, controller)
        inputs = stage / 'inputs'
        complete = json.loads((inputs / 'complete.json').read_text())
        metadata = json.loads((inputs / 'metadata.json').read_text())
        assert complete['state'] == 'matched_inputs_verified'
        assert complete['metadata_sha256'] == sha(inputs / 'metadata.json')
        assert complete['source_center_pixels_exact'] is True
        assert complete['additional_warmup_steps'] == 0 and complete['policy_executed'] is False
        assert complete['scenes'] == list(SHIFTS) and metadata['shifts_cm'] == SHIFTS
        for scene in SHIFTS:
            folder = inputs / scene
            for name in ('center_restore_checks', 'restore_checks'):
                rows = json.loads((folder / (name + '.json')).read_text())
                assert all(row['exact'] for row in rows), (scene, name)
            for name in ('controller_checks', 'restored_controller_checks'):
                checks = json.loads((folder / (name + '.json')).read_text())
                assert all(value is True for value in checks.values()), (scene, name)
        write(stage / 'complete.json', dict(state='complete', script_sha256=sha(Path(__file__)),
            protocol_sha256=sha(stage / 'protocol.json'), inputs_complete_sha256=sha(inputs / 'complete.json'),
            inputs_metadata_sha256=sha(inputs / 'metadata.json'), policy_model_forwards=0,
            policy_scenes=['x15', 'x18'], reference_only_scene='x00', additional_warmup_steps=0,
            all_three_scenes_typed_state_controller_pixels_restored_exact=True))
        print('[POSITION-INPUTS] matched x15/x18; x00 reference only; model forwards 0', flush=True)
    except Exception as exc:
        write(stage / 'failed.json', dict(type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
