"""Normal Cosmos inference for one verified LIBERO Goal pair.

Uses the checked pixel/schedule factory without installing attention routes.
The fixed output must be absent. Four q0 predictions establish two repeat
controls before this localhost server accepts the sixteen closed-loop queries.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer

ROOT = Path('/home/current/work/cosmos3')
INPUT = ROOT / 'outputs/official-demos/goal-pair-input'
OUT = ROOT / 'outputs/official-demos/goal-pair-policy'
PORT = 8923
TASKS = {
    'bowl': (4, 'put_the_bowl_on_top_of_the_cabinet',
             'put the bowl on top of the cabinet', 'akita_black_bowl_1'),
    'wine': (2, 'put_the_wine_bottle_on_top_of_the_cabinet',
             'put the wine bottle on top of the cabinet', 'wine_bottle_1'),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_input():
    complete = json.loads((INPUT / 'complete.json').read_text())
    assert complete['state'] == 'shared_input_verified'
    assert complete['exact_checks'] == 63 and complete['png_files_bytes_equal'] == 3
    assert not (INPUT / 'failed.json').exists()
    checks = json.loads((INPUT / 'checks.json').read_text())
    assert len(checks) == 63 and all(row['exact'] for row in checks)
    metadata = json.loads((INPUT / 'metadata.json').read_text())
    assert metadata['suite'] == 'libero_goal' and metadata['task_order_index'] == 0
    assert metadata['shared_init_task_index'] == 4 and metadata['shared_init_index'] == 0
    assert metadata['warmup_steps'] == 10 and metadata['environment_seed'] == 0
    by_index = {row['index']: row for row in metadata['tasks']}
    collector = load_file('goal_pair_input_contract', ROOT / 'work/collect_cosmos_goal_pair.py')
    assert metadata['script_sha256'] == digest(Path(collector.__file__))
    for key, (index, name, language, target) in TASKS.items():
        row = by_index[index]
        assert (row['name'], row['native_language'], row['target_object']) == (name, language, target), key
        expected_sha = next(item[3] for item in collector.TASKS if item[0] == index)
        assert row['bddl_sha256'] == expected_sha, key
        assert digest(Path(row['bddl_path'])) == expected_sha, key
    for filename in ('agentview.png', 'robot0_eye_in_hand.png', 'input.png'):
        assert (INPUT / 'reference' / filename).read_bytes() == (INPUT / 'restored' / filename).read_bytes()
    return metadata


class NormalRuntime:
    """Only observational hooks; every original tensor is returned unchanged."""

    def __init__(self, factory):
        import torch
        self.torch, self.factory = torch, factory
        self.pipe, self.sampler = factory.make_runtime()
        self.cm = factory.cm

    def cpu(self, value):
        if isinstance(value, self.torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {key: self.cpu(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return type(value)(self.cpu(item) for item in value)
        return value

    def require_exact(self, a, b, label):
        torch = self.torch
        if isinstance(a, torch.Tensor):
            assert isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape, label
            assert torch.equal(a, b), label
            assert torch.equal(a.contiguous().reshape(-1).view(torch.uint8),
                               b.contiguous().reshape(-1).view(torch.uint8)), f'{label}/bytes'
        elif isinstance(a, dict):
            assert isinstance(b, dict) and a.keys() == b.keys(), label
            for key in a:
                self.require_exact(a[key], b[key], f'{label}/{key}')
        elif isinstance(a, (tuple, list)):
            assert type(a) is type(b) and len(a) == len(b), label
            for index, (left, right) in enumerate(zip(a, b)):
                self.require_exact(left, right, f'{label}/{index}')
        else:
            assert a == b, label

    def predict(self, task, image_path, seed, destination):
        from PIL import Image

        torch, pipe, cm = self.torch, self.pipe, self.cm
        destination.mkdir(exist_ok=False)
        record = dict(readout=[], action_velocity=[], action_states=[], pure_noise=[])
        handles, model_calls, prepare_calls = [], [], []
        original_random = cm.randn_tensor
        original_prepare = pipe.prepare_latents
        scheduler_class = self.sampler.FlowUniPCMultistepScheduler
        original_step = scheduler_class.step

        def before_model(module, args, kwargs):
            model_calls.append(len(model_calls))
            if len(model_calls) == 1:
                record['model_input'] = self.cpu(kwargs)

        def before_head(module, args):
            record['readout'].append(self.cpu(args[0]))

        def after_head(module, args, output):
            record['action_velocity'].append(self.cpu(output))

        def observe_random(*args, **kwargs):
            value = original_random(*args, **kwargs)
            record['pure_noise'].append(self.cpu(value))
            return value

        def observe_prepare(*args, **kwargs):
            value = original_prepare(*args, **kwargs)
            prepare_calls.append(1)
            record['prepared_latents_and_masks'] = self.cpu(value)
            return value

        def observe_step(instance, model_output, timestep, sample, *args, **kwargs):
            is_action = tuple(sample.shape) == (1, 16, 64)
            if is_action and not record['action_states']:
                record['action_states'].append(self.cpu(sample))
            value = original_step(instance, model_output, timestep, sample, *args, **kwargs)
            if is_action:
                previous = value.prev_sample if hasattr(value, 'prev_sample') else value[0]
                record['action_states'].append(self.cpu(previous))
            return value

        began = time.perf_counter()
        try:
            handles.append(pipe.transformer.register_forward_pre_hook(before_model, with_kwargs=True))
            handles.append(pipe.transformer.action_proj_out.register_forward_pre_hook(before_head))
            handles.append(pipe.transformer.action_proj_out.register_forward_hook(after_head))
            cm.randn_tensor = observe_random
            pipe.prepare_latents = observe_prepare
            scheduler_class.step = observe_step
            pipe.scheduler = scheduler_class(num_train_timesteps=1000, shift=1.)
            condition = cm.CosmosActionCondition(mode='policy', chunk_size=16, domain_name='libero',
                resolution_tier=256, image=Image.open(image_path).convert('RGB'), view_point='concat_view')
            output = pipe(prompt=TASKS[task][2], action=condition, fps=20, num_inference_steps=30,
                guidance_scale=1., use_system_prompt=False,
                generator=torch.Generator(device='cuda').manual_seed(seed), output_type='latent')
        finally:
            cm.randn_tensor = original_random
            pipe.prepare_latents = original_prepare
            scheduler_class.step = original_step
            for handle in handles:
                handle.remove()
        record.update(actions=self.cpu(output.action[0].float()), video=self.cpu(output.video),
                      timesteps=self.cpu(pipe.scheduler.timesteps), sigmas=self.cpu(pipe.scheduler.sigmas))
        for key in ('readout', 'action_velocity', 'action_states'):
            record[key] = torch.stack(record[key])
        # Save observed tensors before any guard can reject this prediction.
        torch.save(record, destination / 'states.pt')
        write_json(destination / 'normalized_actions.json', record['actions'].tolist())
        steps = record['timesteps'].tolist()
        metadata = dict(task=task, task_index=TASKS[task][0], prompt=TASKS[task][2], seed=seed,
            pixels='official', schedule='official', use_system_prompt=False,
            domain_name='libero', domain_id=5, raw_action_dim=10, chunk_size=16,
            num_inference_steps=30, image_path=str(image_path), input_png_sha256=digest(image_path),
            und_len=int(record['model_input']['und_len']), timesteps=steps,
            model_calls=len(model_calls), prepare_calls=len(prepare_calls),
            pure_noise_shapes=[list(value.shape) for value in record['pure_noise']],
            elapsed_s=time.perf_counter() - began,
            capture='Actual first model kwargs; actual FP32 vision/action random draws before masks/padding; prepared latents/masks; 30 action readouts and velocities; 31 FP32 action states. No attention or GEN layer cache.')
        write_json(destination / 'metadata.json', metadata)
        assert len(model_calls) == 30 and len(prepare_calls) == 1
        assert len(record['pure_noise']) == 2
        assert all(value.dtype == torch.float32 for value in record['pure_noise'])
        assert record['pure_noise'][0].ndim == 5 and record['pure_noise'][1].shape == (16, 64)
        prepared = record['prepared_latents_and_masks']
        assert len(prepared) == 12 and prepared[0].dtype == prepared[2].dtype == torch.float32
        assert prepared[1] is None and prepared[8].tolist() == [5] and prepared[10] == 10
        assert prepared[2].shape == (16, 64) and torch.count_nonzero(prepared[2][:, 10:]) == 0
        assert torch.count_nonzero(prepared[7]) == 0
        assert record['readout'].shape == (30, 16, 4096)
        assert record['action_velocity'].shape == (30, 16, 64)
        assert record['action_states'].shape == (31, 1, 16, 64)
        assert record['action_states'].dtype == torch.float32
        assert record['actions'].shape == (16, 10) and torch.isfinite(record['actions']).all()
        assert len(steps) == 30 and all(type(step) is int for step in steps)
        assert steps[10] == 666 and steps[20] == 333
        return record, metadata


def main():
    # Existing data is never touched, including when this guard rejects a rerun.
    OUT.mkdir(parents=True, exist_ok=False)
    try:
        os.environ['HF_HUB_OFFLINE'] = '1'
        metadata = validate_input()
        factory_path = ROOT / 'work/probe_cosmos_language_routes.py'
        factory = load_file('goal_pair_checked_runtime', factory_path)
        factory.OUT = OUT  # Its status writer must not touch the previous experiment.
        write_json(OUT / 'provenance.json', dict(script_sha256=digest(Path(__file__)),
            runtime_factory_sha256=digest(factory_path), collector_sha256=metadata['script_sha256'],
            shared_input_metadata_sha256=digest(INPUT / 'metadata.json'),
            shared_input_png_sha256=digest(INPUT / 'reference/input.png'),
            pixel_helper_sha256=digest(ROOT / 'work/check_cosmos_official_pixels_cpu.py'),
            scheduler_sha256=digest(factory.SCHEDULER), tasks=TASKS, library_edits=False,
            scope='Two native instructions from one collected Goal4/init0 snapshot. Normal inference; no latent intervention, training, or benchmark-rate claim.',
            noise_seed_rule='198 + query_index, independently reset per task/query.',
            proprioception='No proprioception fields are added to CosmosActionCondition. Raw observations stay in the input/rollout audit files.'))
        runtime = NormalRuntime(factory)
        baseline, baseline_metadata, controls = {}, {}, []
        (OUT / 'baseline').mkdir(exist_ok=False)
        for task in TASKS:
            folder = OUT / 'baseline' / task
            folder.mkdir(exist_ok=False)
            first, first_meta = runtime.predict(task, INPUT / 'reference/input.png', 198, folder / 'q0')
            second, _ = runtime.predict(task, INPUT / 'reference/input.png', 198, folder / 'q0_repeat')
            runtime.require_exact(first, second, f'{task}/same_input_repeat')
            baseline[task], baseline_metadata[task] = first, first_meta
            controls.append(dict(task=task, repeated_prediction_all_saved_tensors_exact=True))
            write_json(OUT / 'baseline_controls.json', controls)
        runtime.require_exact(baseline['bowl']['pure_noise'], baseline['wine']['pure_noise'], 'paired_q0_noise')
        runtime.require_exact(baseline['bowl']['timesteps'], baseline['wine']['timesteps'], 'paired_schedule')
        write_json(OUT / 'baseline_complete.json', dict(state='complete', normal_predictions=4,
            exact_repeat_controls=2, paired_q0_noise_exact=True, native_task_languages=[value[2] for value in TASKS.values()]))
        served, noise_by_query = [], {}
        failed = False

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path != '/health':
                    self.send_error(404)
                    return
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'ready')

            def do_POST(self):
                nonlocal failed
                try:
                    assert self.path == '/predict'
                    request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                    assert set(request) == {'task', 'query', 'seed', 'prompt'}
                    task, query = request['task'], request['query']
                    assert task in TASKS and type(query) is int and 0 <= query < 8
                    assert request['seed'] == 198 + query and request['prompt'] == TASKS[task][2]
                    assert (task, query) not in served, 'Duplicate query; no overwrite or automatic retry.'
                    assert query == sum(item[0] == task for item in served), 'Queries must be ordered per task.'
                    run = OUT / 'closed-loop' / task
                    image = run / f'input_{query:02d}.png'
                    rec, meta = runtime.predict(task, image, 198 + query, run / f'chunk_{query:02d}')
                    assert meta['timesteps'] == baseline_metadata[task]['timesteps']
                    if query == 0:
                        assert image.read_bytes() == (INPUT / 'reference/input.png').read_bytes(), 'q0 PNG bytes'
                        runtime.require_exact(baseline[task], rec, f'{task}/live_q0_baseline')
                    paired = query in noise_by_query
                    if paired:
                        runtime.require_exact(noise_by_query[query], rec['pure_noise'], f'query{query}/paired_noise')
                    else:
                        noise_by_query[query] = rec['pure_noise']
                    served.append((task, query))
                    write_json(OUT / 'served_queries.json', [dict(task=key, query=index, seed=198 + index)
                                                           for key, index in served])
                    value = dict(metadata=meta, actions=rec['actions'].tolist(),
                                 q0_all_saved_tensors_exact=True if query == 0 else None,
                                 paired_noise_checked_exact=paired)
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(value, allow_nan=False).encode())
                except Exception as exc:
                    failed = True
                    write_json(OUT / 'failed.json', dict(stage='server_query', error=str(exc),
                        traceback=traceback.format_exc(), served_queries=served))
                    self.send_error(500, str(exc))

        server = HTTPServer(('127.0.0.1', PORT), Handler)
        server.timeout = 30
        write_json(OUT / 'ready.json', dict(state='ready', url=f'http://127.0.0.1:{PORT}', baseline_controls=controls))
        print(f'[GOAL-PAIR] ready on localhost:{PORT}', flush=True)
        try:
            while not failed and len(served) < 16:
                server.handle_request()
        finally:
            server.server_close()
        assert not failed and len(served) == 16
        write_json(OUT / 'server_complete.json', dict(state='complete', normal_closed_loop_predictions=16,
            q0_matches_baseline_all_saved_tensors_exact=2, paired_noise_checks_exact=8))
    except Exception as exc:
        if not (OUT / 'failed.json').exists():
            write_json(OUT / 'failed.json', dict(stage='server', error=str(exc), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
