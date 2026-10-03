"""Isolate system-message and position-format effects on a frozen Cosmos3 policy.

Preserve the legacy image preparation/scheduler to change one contract item.
After offline controls, retain the same model for explicit on/off rollout requests.
"""
import hashlib
import importlib.util
import json
import os
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
import numpy as np
import torch
from PIL import Image
from diffusers import Cosmos3OmniPipeline
from diffusers.pipelines.cosmos import pipeline_cosmos3_omni as cm

ROOT = Path('/home/current/work/cosmos3')
DATA = ROOT/'outputs/official-demos/milk-posttrain/gripper-context/deep-dive'
OUT = DATA/'runtime-system-prompt'
OUT.mkdir(exist_ok=False)
(OUT/'cases').mkdir()
started = time.perf_counter()
torch.set_grad_enabled(False)
cm._EMBODIMENT_TO_RAW_ACTION_DIM['libero'] = 10
pipe = Cosmos3OmniPipeline.from_pretrained(
    str(ROOT/'checkpoints/Cosmos3-Nano-Policy-LIBEROall-5k'),
    torch_dtype=torch.bfloat16, sound_tokenizer=None,
    enable_safety_checker=False, local_files_only=True)
pipe.to('cuda')
pipe.vae.enable_tiling()
model = pipe.transformer
assert not model.is_cache_enabled
old_builder = pipe._build_action_json_prompt


def builder(*args, **kwargs):
    value = json.loads(old_builder(*args, **kwargs))
    value['actions'][0]['idle_frame'] = '0 out of 16.'
    return json.dumps(value)


pipe._build_action_json_prompt = builder
spec = importlib.util.spec_from_file_location('system_prompt_native_flow',
    ROOT/'cosmos-framework/cosmos_framework/model/generator/diffusion/samplers/fm_solvers_unipc.py')
sampler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sampler)
pipe.register_to_config(use_native_flow_schedule=True)
images = {
    'center': DATA/'object-position/inputs/center/input.png',
    'far': DATA/'milk-far/inputs/xp15/input.png',
}
active = None
step = -1
indexes = None
record = None
position_source = None
native_inputs = {}
arrays = {}
rows = []
checks = []


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, list):
        return [cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu(v) for v in value)
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items()}
    return value


def tensor_hash(value):
    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def exactly_equal(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(exactly_equal(x, y) for x, y in zip(a, b))
    return a == b


def status(stage, **kwargs):
    value = dict(stage=stage, elapsed_s=time.perf_counter()-started, **kwargs)
    (OUT/'status.json').write_text(json.dumps(value))
    print('[SYSTEM-PROMPT] '+json.dumps(value), flush=True)


original_prepare = pipe.prepare_latents


def prepare(*args, **kwargs):
    value = original_prepare(*args, **kwargs)
    if active is not None:
        # Preserve original FP32 initial states, not rounded transformer inputs.
        record['initial_vision'] = cpu(value[0])
        record['initial_action'] = cpu(value[2])
        assert record['initial_vision'].dtype == torch.float32
        assert record['initial_action'].dtype == torch.float32
    return value


pipe.prepare_latents = prepare


def before_model(module, args, kwargs):
    global step, indexes
    if active is None:
        return
    step += 1
    indexes = kwargs['action_sequence_indexes'].flatten()-kwargs['und_len']
    if position_source is not None:
        positions = kwargs['position_ids'].clone()
        u = kwargs['und_len']
        source_u = position_source['und_len']
        positions[..., u:] = position_source['position_ids'][..., source_u:].to(positions)
        if record['mode'] == 'off_all_positions_on':
            assert source_u-u == 19
            assert torch.equal(kwargs['input_ids'], position_source['input_ids'][19:].to(kwargs['input_ids']))
            positions[..., :u] = position_source['position_ids'][..., 19:source_u].to(positions)
        kwargs['position_ids'] = positions
    if step == 0:
        record['input'] = cpu(kwargs)
    if record['capture_hidden'] and step in [0, 29]:
        record['hidden'][step] = []
    return args, kwargs


def before_first(module, args):
    if active is not None and record['capture_hidden'] and step in [0, 29]:
        record['hidden'][step].append(cpu(args[1][indexes]))


def after_layer(module, args, output):
    if active is not None and record['capture_hidden'] and step in [0, 29]:
        record['hidden'][step].append(cpu(output[1][indexes]))


def before_head(module, args):
    if active is not None and record['capture_hidden']:
        record['readout'].append(cpu(args[0]))


handles = [model.register_forward_pre_hook(before_model, with_kwargs=True),
           model.layers[0].register_forward_pre_hook(before_first),
           model.action_proj_out.register_forward_pre_hook(before_head)]
handles.extend(layer.register_forward_hook(after_layer) for layer in model.layers)


def predict(scene, noun, seed, mode, image=None, destination=None, capture=True):
    global active, step, record, position_source
    assert mode in ['on', 'off', 'off_gen_positions_on', 'on_gen_positions_off', 'off_all_positions_on']
    system_on = mode in ['on', 'on_gen_positions_off']
    name = f'{scene}_{noun}_{seed}_{mode}'
    record = dict(case=name, mode=mode, hidden={}, readout=[], capture_hidden=capture)
    if mode in ['off_gen_positions_on', 'off_all_positions_on']:
        position_source = native_inputs[(scene, noun, seed, 'on')]
    elif mode == 'on_gen_positions_off':
        position_source = native_inputs[(scene, noun, seed, 'off')]
    else:
        position_source = None
    active, step = name, -1
    pipe.scheduler = sampler.FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1.)
    path = images[scene] if image is None else image
    condition = cm.CosmosActionCondition(
        mode='policy', chunk_size=16, domain_name='libero', resolution_tier=256,
        image=Image.open(path).convert('RGB'), view_point='concat_view')
    run_started = time.perf_counter()
    try:
        result = pipe(
            prompt=f'pick up the {noun} and place it in the basket', action=condition,
            fps=20, num_inference_steps=30, guidance_scale=1.,
            use_system_prompt=system_on,
            generator=torch.Generator(device='cuda').manual_seed(seed), output_type='latent')
    finally:
        active = None
    assert step == 29, step
    actions = result.action[0].float().cpu().numpy()
    assert actions.shape == (16, 10)
    rec = record
    rec['output_actions'] = torch.from_numpy(actions)
    rec['output_video_latents'] = cpu(result.video)
    rec['sigmas'] = cpu(pipe.scheduler.sigmas)
    rec['timesteps'] = cpu(pipe.scheduler.timesteps)
    rec['prompt'] = f'pick up the {noun} and place it in the basket'
    if capture:
        rec['readout'] = torch.stack(rec['readout'])
        rec['hidden'] = torch.stack([torch.stack(rec['hidden'][s]) for s in [0, 29]])
        assert rec['hidden'].shape == (2, 37, 16, 4096)
        assert rec['readout'].shape == (30, 16, 4096)
    inp = rec['input']
    if mode in ['on', 'off'] and image is None:
        native_inputs[(scene, noun, seed, mode)] = inp
    item = dict(case=name, scene=scene, noun=noun, seed=seed, mode=mode,
        use_system_prompt=system_on, und_len=inp['und_len'],
        input_ids=inp['input_ids'].flatten().tolist(),
        decoded_input=pipe.text_tokenizer.decode(inp['input_ids'].flatten().tolist()),
        fp32_initial_vision_sha256=tensor_hash(rec['initial_vision']),
        fp32_initial_action_sha256=tensor_hash(rec['initial_action']),
        gripper_mean=float(actions[:, 9].mean()), elapsed_s=time.perf_counter()-run_started,
        timesteps=rec['timesteps'].tolist(), sigmas=rec['sigmas'].tolist())
    if destination is not None:
        destination.mkdir(exist_ok=False)
        (destination/'normalized_actions.json').write_text(json.dumps(actions.tolist()))
        (destination/'metadata.json').write_text(json.dumps(item, indent=2))
        if capture:
            torch.save(rec, destination/'states.pt')
    return actions, rec, item


def relative(a, b):
    return float((b.float()-a.float()).square().mean().sqrt()/a.float().square().mean().sqrt().clamp_min(1e-20))


def execute_probe():
    for seed in [195, 196, 198]:
        for scene in images:
            for noun in ['milk', 'butter']:
                local = {}
                modes = ['on', 'off'] + (['off_gen_positions_on', 'on_gen_positions_off', 'off_all_positions_on'] if seed == 198 else [])
                for mode in modes:
                    name = f'{scene}_{noun}_{seed}_{mode}'
                    action, rec, item = predict(scene, noun, seed, mode,
                        destination=OUT/'cases'/name, capture=seed == 198)
                    if mode == 'on':
                        ref = np.load(DATA/'semantic-transport'/f'{scene}_{noun}_{seed}_actions.npy')
                        assert np.array_equal(action, ref), ('legacy_replay', name)
                        item['legacy_action_replay_exact'] = True
                    else:
                        baseline = local['on']
                        assert torch.equal(rec['initial_vision'], baseline['initial_vision'])
                        assert torch.equal(rec['initial_action'], baseline['initial_action'])
                        assert torch.equal(rec['timesteps'], baseline['timesteps'])
                        assert torch.equal(rec['sigmas'], baseline['sigmas'])
                        item['fp32_initial_states_and_schedule_exact'] = True
                        left = rec['input']; right = baseline['input']
                        if mode.startswith('off'):
                            assert right['und_len']-left['und_len'] == 19
                            assert torch.equal(left['input_ids'], right['input_ids'][19:])
                        for key in ['vision_tokens', 'action_tokens', 'vision_timesteps', 'action_timesteps', 'action_domain_ids']:
                            assert exactly_equal(left[key], right[key]), key
                        if mode == 'off_gen_positions_on' or mode == 'off_all_positions_on':
                            assert torch.equal(left['position_ids'][..., left['und_len']:], right['position_ids'][..., right['und_len']:])
                        if mode == 'off_all_positions_on':
                            assert torch.equal(left['position_ids'][..., :left['und_len']], right['position_ids'][..., 19:right['und_len']])
                        if mode == 'on_gen_positions_off':
                            off = local['off']['input']
                            assert torch.equal(left['position_ids'][..., left['und_len']:], off['position_ids'][..., off['und_len']:])
                    local[mode] = rec
                    arrays[name] = action
                    rows.append(item)
                    status('offline_case_complete', completed=len(rows), total=36, case=name)
                baseline = local['on']
                for mode in modes[1:]:
                    rec = local[mode]
                    effect = dict(scene=scene, noun=noun, seed=seed, mode=mode,
                        final_xyz_rmse=float(np.sqrt(np.mean((rec['output_actions'].numpy()[:, :3]-baseline['output_actions'].numpy()[:, :3])**2))),
                        final_action10_rmse=float(np.sqrt(np.mean((rec['output_actions'].numpy()-baseline['output_actions'].numpy())**2))))
                    if seed == 198:
                        effect['first_readout_relative'] = relative(baseline['readout'][0], rec['readout'][0])
                        effect['last_readout_relative'] = relative(baseline['readout'][-1], rec['readout'][-1])
                        effect['layer_profile'] = [dict(step=[0, 29][j], boundary=l,
                            relative=relative(baseline['hidden'][j, l], rec['hidden'][j, l]))
                            for j in range(2) for l in range(37)]
                    checks.append(effect)
                del local
                np.savez_compressed(OUT/'arrays.npz', **arrays)
                (OUT/'result.json').write_text(json.dumps(dict(cases=rows, effects=checks,
                    scope='System on/off with unchanged legacy pixels and native schedule; additional GEN-only and shared-token position controls are offline. Not a full framework equivalence proof or a success-rate benchmark.'), indent=2))
    (OUT/'probe_complete.json').write_text(json.dumps(dict(state='complete', cases=len(rows), elapsed_s=time.perf_counter()-started)))
    status('offline_complete_server_loading')


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'ready')

    def do_POST(self):
        try:
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            q = int(request['chunk'])
            mode = request['mode']
            assert mode in ['on', 'off']
            noun = request['noun']
            assert noun in ['milk', 'butter']
            run = Path(request['output'])
            action, rec, item = predict(request['scene'], noun, int(request['seed']), mode,
                image=run/f'input_{q:02d}.png', destination=run/f'chunk_{q:02d}', capture=False)
            body = json.dumps(dict(actions=action.tolist(), use_system_prompt=item['use_system_prompt'], und_len=item['und_len'])).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)
            print('[SYSTEM-PROMPT] rollout', run.name, q+1, item['elapsed_s'], flush=True)
        except Exception as error:
            traceback.print_exc()
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(error).encode())


try:
    status('loading')
    execute_probe()
    status('server_ready', port=8921)
    HTTPServer(('127.0.0.1', 8921), Handler).serve_forever()
except Exception as error:
    (OUT/'failed.json').write_text(json.dumps(dict(error=str(error), traceback=traceback.format_exc())))
    raise
