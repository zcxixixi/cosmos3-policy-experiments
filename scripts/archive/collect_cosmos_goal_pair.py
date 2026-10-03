"""Collect and verify one shared LIBERO Goal input; never run a policy.

Uses Goal task4/init0, ten native no-op steps, then restores the same simulator
state, fixture transforms and cameras after reset in the same environment.
Imports the simulator only inside main(), after argument parsing.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import numpy as np


POLICY_SOURCE = Path('/home/current/work/openpi-demo/cosmos-policy')
TASKS = (
    (4, 'put_the_bowl_on_top_of_the_cabinet', 'akita_black_bowl_1',
     '2ffba859a154f50c3c99ffb3420743fa5aa65c70bf7d4cd26f5bc81d07be5713'),
    (2, 'put_the_wine_bottle_on_top_of_the_cabinet', 'wine_bottle_1',
     '2ef6f403a8451e216af44158c1a6c975cf8e5058ad02d78367ac0be443d096fe'),
)
CAMERAS = ('agentview', 'robot0_eye_in_hand')
CAMERA_FIELDS = ('cam_pos', 'cam_quat', 'cam_fovy', 'cam_bodyid', 'cam_mode')
NOOP = [0., 0., 0., 0., 0., 0., -1.]


def sha_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_record(value):
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def bddl_fields(text):
    """Parse the installed, hash-checked BDDL without importing an environment."""
    stack, root = [], None
    for token in re.findall(r'\(|\)|[^\s()]+', re.sub(r';[^\n]*', '', text)):
        if token == '(':
            value = []
            if stack:
                stack[-1].append(value)
            else:
                assert root is None
                root = value
            stack.append(value)
        elif token == ')':
            stack.pop()
        else:
            stack[-1].append(token)
    assert not stack and root[0] == 'define'
    return {value[0]: value[1:] for value in root[1:]}


def copy_observations(obs):
    arrays = {name: np.array(value, copy=True) for name, value in obs.items()}
    for name, value in arrays.items():
        assert value.dtype.kind in 'buifc', (name, str(value.dtype))
        assert np.isfinite(value).all(), name
    for name in ('robot0_eef_pos', 'robot0_eef_quat', 'robot0_joint_pos',
                 'robot0_gripper_qpos'):
        assert name in arrays, name
    for camera in CAMERAS:
        value = arrays[camera + '_image']
        assert value.shape == (256, 256, 3) and value.dtype == np.uint8, camera
    return arrays


def capture(env, obs):
    en, sim = env.env, env.sim
    state = np.array(env.get_sim_state(), copy=True)
    assert state.shape == (79,) and state.dtype == np.float64
    fixtures = {}
    fixture_bodies = {name: obj.root_body for name, obj in en.fixtures_dict.items()}
    for name, body in fixture_bodies.items():
        body_id = sim.model.body_name2id(body)
        fixtures[name + '__body_pos'] = sim.model.body_pos[body_id].copy()
        fixtures[name + '__body_quat'] = sim.model.body_quat[body_id].copy()
    camera_ids = [sim.model.camera_name2id(name) for name in CAMERAS]
    cameras = {field: np.array(getattr(sim.model, field)[camera_ids], copy=True)
               for field in CAMERA_FIELDS}
    cameras['world_xpos'] = sim.data.cam_xpos[camera_ids].copy()
    cameras['world_xmat'] = sim.data.cam_xmat[camera_ids].copy()
    objects, object_bodies = {}, {}
    for name, obj in en.objects_dict.items():
        object_bodies[name] = obj.root_body
        body_id = sim.model.body_name2id(obj.root_body)
        objects[name + '__world_pos'] = sim.data.body_xpos[body_id].copy()
        objects[name + '__world_quat_wxyz'] = sim.data.body_xquat[body_id].copy()
        for joint in obj.joints:
            objects[name + '__' + joint + '__qpos'] = np.array(
                sim.data.get_joint_qpos(joint), copy=True)
            objects[name + '__' + joint + '__qvel'] = np.array(
                sim.data.get_joint_qvel(joint), copy=True)
    observations = copy_observations(obs)
    rotated = {name: observations[name + '_image'][::-1, ::-1].copy()
               for name in CAMERAS}
    rotated['input'] = np.concatenate([rotated[name] for name in CAMERAS], axis=1)
    assert rotated['input'].shape == (256, 512, 3)
    return dict(state={'sim_state': state}, fixtures=fixtures, cameras=cameras,
                observations=observations, objects=objects, images=rotated,
                fixture_bodies=fixture_bodies, object_bodies=object_bodies)


def restore(env, reference):
    """Restore captured values; do not step or resample after restoring."""
    sim = env.sim
    actual = {name: obj.root_body for name, obj in env.env.fixtures_dict.items()}
    assert actual == reference['fixture_bodies'], 'Fixture names changed on reset'
    for name, body in actual.items():
        body_id = sim.model.body_name2id(body)
        sim.model.body_pos[body_id] = reference['fixtures'][name + '__body_pos']
        sim.model.body_quat[body_id] = reference['fixtures'][name + '__body_quat']
    camera_ids = [sim.model.camera_name2id(name) for name in CAMERAS]
    for field in ('cam_bodyid', 'cam_mode'):
        assert np.array_equal(getattr(sim.model, field)[camera_ids],
                              reference['cameras'][field]), field
    for field in ('cam_pos', 'cam_quat', 'cam_fovy'):
        getattr(sim.model, field)[camera_ids] = reference['cameras'][field]
    return env.set_init_state(reference['state']['sim_state'])


def save_capture(folder, record):
    from PIL import Image

    folder.mkdir(exist_ok=False)
    np.save(folder / 'state.npy', record['state']['sim_state'])
    for category in ('fixtures', 'cameras', 'observations', 'objects'):
        np.savez_compressed(folder / (category + '.npz'), **record[category])
    for name, pixels in record['images'].items():
        Image.fromarray(pixels).save(folder / (name + '.png'))
    write_json(folder / 'arrays.json', {
        category: {name: array_record(value) for name, value in record[category].items()}
        for category in ('state', 'fixtures', 'cameras', 'observations', 'objects', 'images')})


def compare(reference, restored):
    checks = []
    for category in ('state', 'fixtures', 'cameras', 'observations', 'objects', 'images'):
        a, b = reference[category], restored[category]
        assert set(a) == set(b), (category, 'array names differ')
        for name in sorted(a):
            equal = (a[name].shape == b[name].shape and a[name].dtype == b[name].dtype
                     and np.array_equal(a[name], b[name])
                     and a[name].tobytes(order='C') == b[name].tobytes(order='C'))
            checks.append(dict(category=category, array=name, exact=bool(equal),
                               reference=array_record(a[name]), restored=array_record(b[name])))
    for category in ('fixture_bodies', 'object_bodies'):
        checks.append(dict(category=category, exact=reference[category] == restored[category]))
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='New output directory; existing paths are refused.')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    env = None
    try:
        os.environ.setdefault('MUJOCO_GL', 'egl')
        sys.path.insert(0, str(POLICY_SOURCE))
        from libero.libero import benchmark, get_libero_path
        from cosmos_policy.experiments.robot.libero.libero_utils import get_libero_env

        suite = benchmark.get_benchmark_dict()['libero_goal'](task_order_index=0)
        tasks, definitions = [], []
        for index, name, target, expected_sha in TASKS:
            task = suite.get_task(index)
            assert task.name == name, (index, task.name)
            path = Path(suite.get_task_bddl_file_path(index))
            assert sha_file(path) == expected_sha, ('BDDL changed', str(path))
            definition = bddl_fields(path.read_text())
            definitions.append(definition)
            tasks.append(dict(index=index, name=name, native_language=task.language,
                              bddl_path=str(path), bddl_sha256=expected_sha,
                              bddl_language=' '.join(definition[':language']),
                              target_object=target, goal=definition[':goal']))
        physical_fields = ('problem', ':domain', ':regions', ':fixtures', ':objects', ':init')
        assert all(definitions[0][key] == definitions[1][key] for key in physical_fields)
        init_file = Path(get_libero_path('init_states')) / 'libero_goal' / suite.get_task(4).init_states_file
        initial = np.array(suite.get_task_init_states(4)[0], copy=True)
        assert initial.shape == (79,) and initial.dtype == np.float64
        metadata = dict(
            scope='One verified shared input, not a policy rollout or a natural-success result.',
            script_sha256=sha_file(Path(__file__)), suite='libero_goal', task_order_index=0,
            tasks=tasks, shared_init_task_index=4, shared_init_index=0,
            shared_init_file=str(init_file), shared_init_file_sha256=sha_file(init_file),
            shared_init_array=array_record(initial), physical_bddl_fields_equal=list(physical_fields),
            environment_bddl_task_index=4, environment_instances=1, environment_seed=0,
            warmup_steps=10, warmup_action=NOOP, resolution_per_camera=[256, 256],
            camera_order=list(CAMERAS), image_rule='Rotate each RGB by 180 degrees; concatenate agentview then wrist.',
            restore_rule='reset; restore fixture body_pos/body_quat and camera static values; set_init_state; no additional steps.',
            reference_observation_rule='After warmup, regenerate observations from the captured state using set_init_state.',
            future_policy_contract=dict(domain_name='libero', domain_id=5, raw_action_dim=10,
                                        chunk_size=16, denoising_steps=30),
            action_conversion='None. Raw EEF, quaternion, joint and gripper observations are preserved.',
            limit='79D state excludes Python controller memory. Equal inputs do not prove identical subsequent closed-loop behavior.')
        write_json(args.output / 'metadata.json', metadata)
        np.save(args.output / 'native_task4_init0.npy', initial)
        env, description = get_libero_env(suite.get_task(4), 'cosmos', resolution=256)
        assert description == tasks[0]['native_language']
        env.reset()
        obs = env.set_init_state(initial)
        for _ in range(10):
            obs, _, _, _ = env.step(NOOP)
        warmed_state = np.array(env.get_sim_state(), copy=True)
        obs = env.set_init_state(warmed_state)
        reference = capture(env, obs)
        assert np.array_equal(reference['state']['sim_state'], warmed_state)
        metadata['fixture_bodies'] = reference['fixture_bodies']
        metadata['object_bodies'] = reference['object_bodies']
        write_json(args.output / 'metadata.json', metadata)
        save_capture(args.output / 'reference', reference)
        env.reset()
        restored = capture(env, restore(env, reference))
        save_capture(args.output / 'restored', restored)
        checks = compare(reference, restored)
        write_json(args.output / 'checks.json', checks)
        assert all(check['exact'] for check in checks), [
            (check['category'], check.get('array')) for check in checks if not check['exact']]
        for filename in ('agentview.png', 'robot0_eye_in_hand.png', 'input.png'):
            a, b = args.output / 'reference' / filename, args.output / 'restored' / filename
            assert a.read_bytes() == b.read_bytes(), ('PNG bytes', filename)
    except Exception as exc:
        write_json(args.output / 'failed.json', dict(type=type(exc).__name__, error=str(exc)))
        raise
    finally:
        if env is not None:
            env.close()
    write_json(args.output / 'complete.json', dict(
        state='shared_input_verified', exact_checks=len(checks), png_files_bytes_equal=3,
        policy_executed=False, policy_success_evaluated=False,
        shared_input=str(args.output / 'reference' / 'input.png')))
    print(json.dumps(dict(state='shared_input_verified', exact_checks=len(checks),
                          output=str(args.output)), ensure_ascii=False))


if __name__ == '__main__':
    main()
