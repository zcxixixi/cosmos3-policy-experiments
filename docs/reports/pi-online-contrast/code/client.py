"""Closed-loop LIBERO-Object task-7 trials against pi05_server.py; records which object is grasped.

Same scenes (captured snapshots), same grasp rule as the Cosmos experiments (exp20/21):
both fingerpads touch AND object lifted >2 cm for >=5 consecutive records.
pi0.5 conventions follow the official openpi LIBERO caller: 224x224 padded images, 8-d state,
horizon 10 / replan 5, native actions sent unchanged.

Trials file: list of {"id","scene","prompt","seed_base","mode", optional "steps","capture","early_stop"}
Run with the cosmos simulator venv.  Results: <out>/trials/<id>.json (+ .npz); finished trials are skipped.
"""
import argparse
import ast
import base64
import json
import math
import os
import sys
import time
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path('/home/current/work/cosmos3')
OPENPI = Path('/home/current/work/openpi-demo/openpi')
DEEP = ROOT / 'outputs/official-demos/milk-posttrain/gripper-context/deep-dive'
SCENES = {'center': DEEP / 'object-position/inputs/center', 'far15': DEEP / 'milk-far/inputs/xp15',
          **{n: DEEP / f'object-position/inputs/{n}' for n in ('xm', 'xp', 'ym', 'yp')}}
HORIZON, REPLAN = 10, 5


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def func_from_source(path, name, ns):
    fn = next(n for n in ast.parse(Path(path).read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), ns)
    return ns[name]


def quantify(records):
    per_object, selected = {}, []
    for name in records[0]['objects']:
        xyz = np.asarray([r['objects'][name] for r in records])
        lift = xyz[:, 2] - xyz[0, 2]
        both = np.asarray([r['both'][name] for r in records], dtype=bool)
        strict = both & (lift > .02)
        first = next((i for i in range(len(strict) - 4) if np.all(strict[i:i + 5])), None)
        per_object[name] = dict(first_2cm_for_5frames=first, contact_frames=int(both.sum()),
                                max_lift_cm=float(lift.max() * 100),
                                max_xy_displacement_cm=float(np.linalg.norm(xyz[:, :2] - xyz[0, :2], axis=1).max() * 100))
        if first is not None:
            selected.append(name)
    return per_object, selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--url', default='http://127.0.0.1:8941')
    ap.add_argument('--shard', default='0/1')
    args = ap.parse_args()
    shard, nshard = map(int, args.shard.split('/'))
    args.out = args.out.resolve()
    os.environ.setdefault('MUJOCO_GL', 'egl')
    sys.path.insert(0, '/home/current/work/openpi-demo/cosmos-policy')
    import requests
    from PIL import Image
    from libero.libero import benchmark
    from cosmos_policy.experiments.robot.libero.libero_utils import get_libero_env
    image_tools = load_file('image_tools', OPENPI / 'packages/openpi-client/src/openpi_client/image_tools.py')
    quat2axisangle = func_from_source(OPENPI / 'examples/libero/main.py', '_quat2axisangle', {'np': np, 'math': math})
    (args.out / 'trials').mkdir(parents=True, exist_ok=True)
    (args.out / 'feat').mkdir(parents=True, exist_ok=True)
    trials = json.loads(args.trials.read_text())
    trials = [t for i, t in enumerate(trials) if i % nshard == shard]
    suite = benchmark.get_benchmark_dict()['libero_object']()
    pudding_id = next(i for i in range(suite.n_tasks) if 'chocolate_pudding' in suite.get_task(i).bddl_file)
    task_id = pudding_id if trials[0]['family']=='canonical' else 7
    assert all(t['family']==trials[0]['family'] for t in trials)
    task = suite.get_task(task_id)
    initial_states = suite.get_task_init_states(task_id)
    env, task_description = get_libero_env(task, 'cosmos', resolution=256)
    print('TASK',task_id,task_description,flush=True)
    en = env.env
    gripper = en.robots[0].gripper

    def run_trial(t, path):
        began = time.perf_counter()
        env.reset()
        if t['family']=='canonical':
            obs = env.set_init_state(initial_states[t['initial_index']])
            for _ in range(10):
                obs, _, _, _ = env.step([0.]*6+[-1.])
            assert task_description == t['prompt'], (task_description,t['prompt'])
        else:
            obs = env.set_init_state(np.load(SCENES['center'] / 'state.npy'))
        before = en.sim.data.qpos.copy()
        object_before = {n:en.sim.data.get_joint_qpos(o.joints[0]).copy() for n,o in en.objects_dict.items()}
        if t['swap']:
            a='chocolate_pudding_1'; b=t['swap_with']
            qa=object_before[a].copy(); qb=object_before[b].copy()
            qa[:2],qb[:2]=qb[:2].copy(),qa[:2].copy()
            en.sim.data.set_joint_qpos(en.objects_dict[a].joints[0],qa)
            en.sim.data.set_joint_qpos(en.objects_dict[b].joints[0],qb)
        shift=np.asarray(t.get('orange_shift_xy',[0.,0.]))
        qm=en.sim.data.get_joint_qpos(en.objects_dict['orange_juice_1'].joints[0]).copy()
        qm[:2]+=shift
        en.sim.data.set_joint_qpos(en.objects_dict['orange_juice_1'].joints[0],qm)
        obs = env.set_init_state(env.get_sim_state())
        after = en.sim.data.qpos.copy()
        object_after = {n:en.sim.data.get_joint_qpos(o.joints[0]).copy() for n,o in en.objects_dict.items()}
        changed=np.flatnonzero(before!=after).tolist()
        assert not t['swap']
        assert len(changed)==int(np.count_nonzero(shift)),changed
        for n in object_before:
            expected=object_before[n].copy()
            if t['swap'] and n in ['chocolate_pudding_1',t['swap_with']]:
                other=t['swap_with'] if n=='chocolate_pudding_1' else 'chocolate_pudding_1'
                expected[:2]=object_before[other][:2]
            if n=='orange_juice_1': expected[:2]+=shift
            assert np.array_equal(object_after[n],expected),n
        init_check=dict(changed_qpos_indices=changed,task_id=task_id,task_description=task_description,
            objects_before={n:v.tolist() for n,v in object_before.items()},
            objects_after={n:v.tolist() for n,v in object_after.items()})
        np.savez_compressed(args.out/'trials'/f"{t['id']}_initial.npz",qpos_before=before,qpos_after=after,
                            base=obs['agentview_image'][::-1,::-1],wrist=obs['robot0_eye_in_hand_image'][::-1,::-1])
        frames=[]
        input_base=[]; input_wrist=[]; input_state=[]
        steps = int(t.get('steps', 200))

        def measure(step):
            both = {}
            for name, obj in en.objects_dict.items():
                if name == 'basket_1':
                    continue
                l = bool(en.check_contact(gripper.important_geoms['left_fingerpad'], obj.contact_geoms))
                r = bool(en.check_contact(gripper.important_geoms['right_fingerpad'], obj.contact_geoms))
                both[name] = l and r
            return dict(step=step, both=both, gripper_qpos=np.asarray(obs['robot0_gripper_qpos']).tolist(), eef=np.asarray(obs['robot0_eef_pos']).tolist(),
                        objects={n: en.sim.data.get_joint_qpos(o.joints[0])[:3].tolist()
                                 for n, o in en.objects_dict.items() if n != 'basket_1'})

        records = [measure(0)]
        actions_log, ms_log = [], []
        streak, first_sel, success_any = {}, None, False
        q = 0
        while len(records) - 1 < steps:
            frame = np.concatenate([obs['agentview_image'][::-1, ::-1],
                                    obs['robot0_eye_in_hand_image'][::-1, ::-1]], axis=1).copy()
            frames.append(frame.copy())
            base = image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(frame[:, :256]), 224, 224))
            wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(frame[:, 256:]), 224, 224))
            quat = np.asarray(obs['robot0_eef_quat']).copy()
            state = np.concatenate((obs['robot0_eef_pos'], quat2axisangle(quat.copy()), obs['robot0_gripper_qpos']))
            input_base.append(base.copy()); input_wrist.append(wrist.copy()); input_state.append(state.copy())
            req = dict(base=base64.b64encode(base.tobytes()).decode(), wrist=base64.b64encode(wrist.tobytes()).decode(),
                       state=state.tolist(), prompt=t['prompt'], seed=int(t['seed_base']) + q, mode=t['mode'])
            if t.get('capture') and q < int(t.get('capture_queries', 4)):
                req['capture_path'] = str(args.out / 'feat' / f"{t['id']}_q{q}.npz")
            reply = requests.post(args.url, json=req, timeout=600)
            if reply.status_code != 200:
                raise RuntimeError(reply.text)
            rep = reply.json()
            native = np.asarray(rep['actions'])
            ms_log.append(rep['ms'])
            actions_log.append(native.tolist())
            for k in range(REPLAN):
                if len(records) - 1 >= steps:
                    break
                obs, _, done, _ = env.step(native[k].tolist())
                records.append(measure(len(records)))
                success_any = success_any or bool(en._check_success())
                if first_sel is None:
                    for name in records[-1]['both']:
                        z = records[-1]['objects'][name][2] - records[0]['objects'][name][2]
                        ok = records[-1]['both'][name] and z > .02
                        streak[name] = streak.get(name, 0) + 1 if ok else 0
                        if streak[name] >= 5:
                            first_sel = len(records) - 1
            q += 1
            if t['family']=='canonical' and success_any:
                break
            if t.get('early_stop', True) and first_sel is not None and len(records) - 1 >= first_sel + 20:
                break
        per_object, selected = quantify(records)
        eef = np.asarray([r['eef'] for r in records])
        objects = {n: np.asarray([r['objects'][n] for r in records]) for n in records[0]['objects']}
        closest = {n: float(np.linalg.norm(eef - tr, axis=1).min() * 100) for n, tr in objects.items()}
        first_time = min((v['first_2cm_for_5frames'] for v in per_object.values() if v['first_2cm_for_5frames'] is not None),default=None)
        first_objects=[n for n,v in per_object.items() if first_time is not None and v['first_2cm_for_5frames']==first_time]
        result = dict(t, first_objects=first_objects, init_check=init_check, selected_objects=selected, per_object=per_object, closest_approach_cm=closest,
                      success_any=success_any, steps_run=len(records) - 1, queries=q,
                      ms_median=float(np.median(ms_log)), elapsed_s=time.perf_counter() - began)
        np.savez_compressed(args.out / 'trials' / f"{t['id']}.npz", eef=eef, actions=np.asarray(actions_log),
                            ms=np.asarray(ms_log), **{f'obj__{k}': v for k, v in objects.items()})
        np.savez_compressed(args.out/'trials'/f"{t['id']}_frames.npz",frames=np.asarray(frames))
        np.savez_compressed(args.out/'trials'/f"{t['id']}_inputs.npz",base=np.asarray(input_base),wrist=np.asarray(input_wrist),state=np.asarray(input_state))
        (args.out/'trials'/f"{t['id']}.records").write_text(json.dumps(records))
        path.write_text(json.dumps(result, indent=1) + '\n')
        print(t['id'], 'selected', selected, f"{result['elapsed_s']:.0f}s ms={result['ms_median']:.0f}", flush=True)

    try:
        for t in trials:
            path = args.out / 'trials' / f"{t['id']}.json"
            if path.exists():
                continue
            try:
                run_trial(t, path)
            except Exception:
                import traceback
                (args.out / 'trials' / f"{t['id']}.error.txt").write_text(traceback.format_exc())
                print(t['id'], 'ERROR', flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
