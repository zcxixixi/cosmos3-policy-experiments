"""Controlled pixel preparation and integer-timestep audit of existing weights.

No model-library edits or weight downloads. Legacy controls must exactly replay
experiment09 before interpreting the four-factor comparisons. A local HTTP
server retains the same model for explicit corrected-input closed-loop tests.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer

os.environ['HF_HUB_OFFLINE'] = '1'
import numpy as np
import torch
from PIL import Image
from diffusers import Cosmos3OmniPipeline
from diffusers.pipelines.cosmos import pipeline_cosmos3_omni as cm

ROOT = Path('/home/current/work/cosmos3')
DATA = ROOT/'outputs/official-demos/milk-posttrain/gripper-context/deep-dive'
OUT = DATA/'runtime-pixel-schedule'
IMAGES = {'center': DATA/'object-position/inputs/center/input.png',
          'far': DATA/'milk-far/inputs/xp15/input.png'}
TV_SOURCE = ROOT/'.venv-chat/lib/python3.12/site-packages/torchvision/transforms/_functional_tensor.py'
TRANSFORM_SOURCE = ROOT/'cosmos-framework/cosmos_framework/data/generator/action/utils/transforms.py'
SCHEDULER_SOURCE = ROOT/'cosmos-framework/cosmos_framework/model/generator/diffusion/samplers/fm_solvers_unipc.py'
OUT.mkdir(exist_ok=False)
(OUT/'cases').mkdir()
started = time.perf_counter()
torch.set_grad_enabled(False)
torch.set_num_threads(4)


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, (list, tuple)):
        return type(value)(cpu(v) for v in value)
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items()}
    return value


def exact(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return a == b


def error(a, b):
    a, b = a.float(), b.float()
    diff = b-a
    rmse = diff.square().mean().sqrt()
    return {'rmse': float(rmse), 'max_abs': float(diff.abs().max()),
            'relative_rms_percent': float(100*rmse/a.square().mean().sqrt().clamp_min(1e-20)),
            'exact': bool(torch.equal(a, b))}


def status(stage, **kwargs):
    item = dict(stage=stage, elapsed_s=time.perf_counter()-started, **kwargs)
    (OUT/'status.json').write_text(json.dumps(item))
    print('[PIXEL-SCHEDULE] '+json.dumps(item), flush=True)


# Reuse the CPU-checked source extraction, including the real official
# normalize_uint8_item rather than an algebraically equivalent float formula.
pixel_tools_module = load_file('checked_cosmos_cpu_pixel_tools', ROOT/'work/check_cosmos_official_pixels_cpu.py')
pixel_tools = pixel_tools_module.load_official_pixel_tools()
cpu_current = json.loads((ROOT/'work/official_pixels_cpu_current.json').read_text())
cpu_chat = json.loads((ROOT/'work/official_pixels_cpu_chat.json').read_text())
assert cpu_current['helper_sha256'] == source_sha(Path(pixel_tools_module.__file__))
assert cpu_chat['helper_sha256'] == cpu_current['helper_sha256']
assert [x['normalized_pixels_sha256'] for x in cpu_current['checks']] == [
    x['normalized_pixels_sha256'] for x in cpu_chat['checks']]
assert all(x['reference_equal'] and x['reference_metadata_equal'] for x in cpu_chat['checks'])


cm._EMBODIMENT_TO_RAW_ACTION_DIM['libero'] = 10
status('loading_model')
pipe = Cosmos3OmniPipeline.from_pretrained(
    str(ROOT/'checkpoints/Cosmos3-Nano-Policy-LIBEROall-5k'),
    torch_dtype=torch.bfloat16, sound_tokenizer=None,
    enable_safety_checker=False, local_files_only=True)
pipe.to('cuda')
pipe.vae.enable_tiling()
model = pipe.transformer
assert not model.is_cache_enabled
sampler = load_file('pixel_schedule_native_flow', SCHEDULER_SOURCE)
old_builder = pipe._build_action_json_prompt


def builder(*args, **kwargs):
    value = json.loads(old_builder(*args, **kwargs))
    value['actions'][0]['idle_frame'] = '0 out of 16.'
    return json.dumps(value)


pipe._build_action_json_prompt = builder
old_pixels, old_encode, old_prepare = (
    pipe._prepare_action_video_conditioning, pipe._encode_video, pipe.prepare_latents)
active = None
record = None
step = -1
indexes = None
prefix_controls = {}


def prepare_pixels(conditioning_clip, resolution_tier, num_frames, device, dtype):
    baseline = old_pixels(conditioning_clip, resolution_tier, num_frames, device, dtype)
    assert dtype == torch.float32
    if active is None or record['pixels'] == 'legacy':
        result = baseline
    else:
        assert resolution_tier == 256 and num_frames == 17 and len(conditioning_clip) == 1
        result = pixel_tools_module.prepare_official_pixels(
            conditioning_clip[0], num_frames, device, dtype, pixel_tools)
        assert torch.equal(result[1], baseline[1])
    if active is not None and record['capture']:
        record['pixels_first'] = cpu(result[0][:, :, :1])
        record['image_size'] = cpu(result[1])
    return result


def encode_video(value):
    result = old_encode(value)
    if active is not None and record['capture']:
        record['encoded_first_padded'] = cpu(result[:, :, :1])
        key = record['scene']+'_'+record['pixels']
        if key not in prefix_controls:
            prefix = old_encode(value[:, :, :1].contiguous())
            prefix_controls[key] = dict(
                difference=error(result[:, :, :1], prefix),
                prefix_first=cpu(prefix), repeat_first=cpu(result[:, :, :1]),
                pixel_shape=list(value.shape), prefix_pixel_shape=list(value[:, :, :1].shape))
    return result


def prepare(*args, **kwargs):
    value = old_prepare(*args, **kwargs)
    if active is not None:
        record['initial_vision'] = cpu(value[0])
        record['initial_action'] = cpu(value[2])
        assert value[0].dtype == value[2].dtype == torch.float32
    return value


pipe._prepare_action_video_conditioning = prepare_pixels
pipe._encode_video = encode_video
pipe.prepare_latents = prepare


def before_model(module, args, kwargs):
    global step, indexes
    if active is not None:
        step += 1
        indexes = kwargs['action_sequence_indexes'].flatten()-kwargs['und_len']
        if step == 0:
            record['input'] = cpu(kwargs)
        if record['capture'] and step in [0, 29]:
            record['hidden'][step] = []


def before_first(module, args):
    if active is not None and record['capture'] and step in [0, 29]:
        record['hidden'][step].append(cpu(args[1][indexes]))


def after_layer(module, args, output):
    if active is not None and record['capture'] and step in [0, 29]:
        record['hidden'][step].append(cpu(output[1][indexes]))


def before_head(module, args):
    if active is not None and record['capture']:
        record['readout'].append(cpu(args[0]))


handles = [model.register_forward_pre_hook(before_model, with_kwargs=True),
           model.layers[0].register_forward_pre_hook(before_first),
           model.action_proj_out.register_forward_pre_hook(before_head)]
handles.extend(layer.register_forward_hook(after_layer) for layer in model.layers)


def predict(scene, noun, pixels, schedule, seed=198, image=None, destination=None, capture=True):
    global active, record, step
    assert scene in ['center', 'far'] and noun in ['milk', 'butter']
    assert pixels in ['legacy', 'official'] and schedule in ['legacy', 'official']
    case = f'{scene}_{noun}_{seed}_{pixels}_{schedule}'
    record = dict(case=case, scene=scene, pixels=pixels, schedule=schedule,
                  capture=capture, hidden={}, readout=[])
    pipe.register_to_config(use_native_flow_schedule=schedule == 'legacy')
    pipe.scheduler = sampler.FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1.)
    action = cm.CosmosActionCondition(mode='policy', chunk_size=16, domain_name='libero',
        resolution_tier=256, image=Image.open(IMAGES[scene] if image is None else image).convert('RGB'),
        view_point='concat_view')
    active, step = case, -1
    began = time.perf_counter()
    try:
        result = pipe(prompt=f'pick up the {noun} and place it in the basket', action=action,
            fps=20, num_inference_steps=30, guidance_scale=1., use_system_prompt=False,
            generator=torch.Generator(device='cuda').manual_seed(seed), output_type='latent')
    finally:
        active = None
    assert step == 29
    rec = record
    actions = result.action[0].float().cpu()
    rec.update(output_actions=actions, output_video_latents=cpu(result.video),
               timesteps=cpu(pipe.scheduler.timesteps), sigmas=cpu(pipe.scheduler.sigmas))
    if capture:
        rec['readout'] = torch.stack(rec['readout'])
        rec['hidden'] = torch.stack([torch.stack(rec['hidden'][s]) for s in [0, 29]])
        assert rec['readout'].shape == (30, 16, 4096)
        assert rec['hidden'].shape == (2, 37, 16, 4096)
    metadata = dict(case=case, scene=scene, noun=noun, seed=seed, pixels=pixels,
                    schedule=schedule, use_system_prompt=False, und_len=rec['input']['und_len'],
                    input_ids=rec['input']['input_ids'].tolist(), timesteps=rec['timesteps'].tolist(),
                    elapsed_s=time.perf_counter()-began)
    if destination is not None:
        destination.mkdir(exist_ok=False)
        (destination/'normalized_actions.json').write_text(json.dumps(actions.tolist()))
        (destination/'metadata.json').write_text(json.dumps(metadata, indent=2))
        if capture:
            torch.save(rec, destination/'states.pt')
    return rec, metadata


def offline():
    rows, effects, actions, readout_public, vae_public = [], [], {}, {}, {}
    for scene in IMAGES:
        for noun in ['milk', 'butter']:
            ref = torch.load(DATA/f'runtime-system-prompt/cases/{scene}_{noun}_198_off/states.pt',
                             map_location='cpu', weights_only=False)
            baseline = None
            for pixels, schedule in [('legacy', 'legacy'), ('official', 'legacy'),
                                     ('legacy', 'official'), ('official', 'official')]:
                rec, metadata = predict(scene, noun, pixels, schedule)
                if baseline is None:
                    for key in ['initial_vision', 'initial_action', 'readout', 'hidden',
                                'output_actions', 'output_video_latents', 'timesteps', 'sigmas']:
                        assert exact(rec[key], ref[key]), ('legacy_replay', scene, noun, key)
                    for key in ref['input']:
                        assert exact(rec['input'][key], ref['input'][key]), ('input_replay', key)
                    baseline = rec
                    metadata['legacy_replay_exact'] = True
                else:
                    assert exact(rec['initial_vision'][:, :, 1:], baseline['initial_vision'][:, :, 1:])
                    assert exact(rec['initial_action'], baseline['initial_action'])
                    assert exact(rec['input']['input_ids'], baseline['input']['input_ids'])
                    assert exact(rec['input']['position_ids'], baseline['input']['position_ids'])
                    if schedule == 'legacy':
                        assert exact(rec['timesteps'], baseline['timesteps'])
                    else:
                        differences = torch.nonzero(rec['timesteps'] != baseline['timesteps']).flatten().tolist()
                        assert differences == [10, 20], differences
                    if pixels == 'legacy':
                        assert exact(rec['initial_vision'], baseline['initial_vision'])
                    metadata['future_noise_action_noise_text_and_positions_exact'] = True
                effects.append(dict(case=rec['case'],
                    pixels=error(baseline['pixels_first'], rec['pixels_first']),
                    clean_latent=error(baseline['initial_vision'][:, :, :1], rec['initial_vision'][:, :, :1]),
                    action_xyz=error(baseline['output_actions'][:, :3], rec['output_actions'][:, :3]),
                    action10=error(baseline['output_actions'], rec['output_actions']),
                    readout_profile=[error(baseline['readout'][i], rec['readout'][i]) for i in range(30)],
                    layers_first_last=[[error(baseline['hidden'][j,l], rec['hidden'][j,l])
                                       for l in range(37)] for j in range(2)]))
                dest = OUT/'cases'/rec['case'];dest.mkdir()
                torch.save(rec, dest/'states.pt')
                (dest/'metadata.json').write_text(json.dumps(metadata, indent=2))
                actions[rec['case']] = rec['output_actions'].numpy()
                readout_public[rec['case']] = rec['readout'][[0, 29]].float().numpy()
                vae_public[rec['case']+'_clean'] = rec['initial_vision'][:, :, :1].numpy()
                vae_public[rec['case']+'_pixels'] = rec['pixels_first'].numpy()
                rows.append(metadata)
                np.savez_compressed(OUT/'actions.npz', **actions)
                (OUT/'result.json').write_text(json.dumps(dict(cases=rows, effects=effects,
                    scope='Two pictures, two nouns, one paired seed. Pixels and integer timesteps varied separately. Same BF16 VAE and model, FP32 noise. Not full official-server equivalence or new execution.'), indent=2))
                status('offline_case_complete', completed=len(rows), total=16, case=rec['case'])
            del baseline, ref
    np.savez_compressed(OUT/'readouts_first_last.npz', **readout_public)
    np.savez_compressed(OUT/'vae_arrays.npz', **vae_public)
    torch.save(prefix_controls, OUT/'prefix_controls.pt')
    (OUT/'prefix_controls.json').write_text(json.dumps({k:{a:b for a,b in v.items()
        if not isinstance(b, torch.Tensor)} for k,v in prefix_controls.items()}, indent=2))
    (OUT/'provenance.json').write_text(json.dumps(dict(
        script_sha256=source_sha(Path(__file__)), pixel_helper_sha256=cpu_current['helper_sha256'],
        torchvision_tensor_source_sha256=source_sha(TV_SOURCE),
        framework_transform_source_sha256=source_sha(TRANSFORM_SOURCE), scheduler_source_sha256=source_sha(SCHEDULER_SOURCE),
        torch=torch.__version__, vae_dtype=str(pipe.vae.dtype),
        normalization='Unchanged official normalize_uint8_item on resized/padded uint8 pixels.',
        helper_extraction='Unchanged reflection_pad_to_target AST with real torchvision tensor resize/pad adapter; CPU comparison with original torchvision passed.',
        library_edits=False, weight_downloads=False), indent=2))
    (OUT/'probe_complete.json').write_text(json.dumps(dict(cases=16, state='complete')))


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200);self.end_headers();self.wfile.write(b'ready')

    def do_POST(self):
        try:
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            q = int(request['chunk']);run = Path(request['output'])
            rec, item = predict(request['scene'], request['noun'], request['pixels'], request['schedule'],
                seed=int(request['seed']), image=run/f'input_{q:02d}.png',
                destination=run/f'chunk_{q:02d}', capture=False)
            body = json.dumps(dict(actions=rec['output_actions'].tolist(),
                use_system_prompt=False, und_len=item['und_len'], pixels=item['pixels'],
                schedule=item['schedule'], timesteps=item['timesteps'])).encode()
            self.send_response(200);self.end_headers();self.wfile.write(body)
            status('rollout_query', case=run.name, query=q+1, elapsed_query_s=item['elapsed_s'])
        except Exception as exc:
            traceback.print_exc();self.send_response(500);self.end_headers();self.wfile.write(str(exc).encode())


try:
    offline()
    status('server_ready', port=8921)
    HTTPServer(('127.0.0.1', 8921), Handler).serve_forever()
except Exception as exc:
    (OUT/'failed.json').write_text(json.dumps(dict(error=str(exc), traceback=traceback.format_exc())))
    raise
