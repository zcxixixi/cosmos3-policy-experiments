"""Separate language reads from clean/future video mediation at real attention seams.

The exact engineering controls are required before interpreting any effect.
Uses the already checked official pixels/default schedule in the current runner.
No model-library edits, new weights, training or amplification.
"""
import argparse
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
from cosmos_attention_routes import CosmosAttentionRoutes

ROOT = Path('/home/current/work/cosmos3')
DATA = ROOT/'outputs/official-demos/milk-posttrain/gripper-context/deep-dive'
OUT = DATA/'language-route-isolation'
IMAGES = {'center': DATA/'object-position/inputs/center/input.png',
          'far': DATA/'milk-far/inputs/xp15/input.png'}
SCHEDULER = ROOT/'cosmos-framework/cosmos_framework/model/generator/diffusion/samplers/fm_solvers_unipc.py'
started = time.perf_counter()
torch.set_grad_enabled(False)
torch.set_num_threads(4)


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cpu(v):
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().clone()
    if isinstance(v, (list, tuple)):
        return type(v)(cpu(x) for x in v)
    if isinstance(v, dict):
        return {k: cpu(x) for k, x in v.items()}
    return v


def metric(a, b):
    a, b = a.float(), b.float()
    d = b-a
    return dict(exact=bool(torch.equal(a, b)), max_abs=float(d.abs().max()),
                rmse=float(d.square().mean().sqrt()),
                relative_rms_percent=float(100*d.square().mean().sqrt()/a.square().mean().sqrt().clamp_min(1e-20)))


def exact(a, b, label):
    assert torch.equal(a, b), (label, metric(a, b))


def write_json(path, value):
    def convert(v):
        if isinstance(v, torch.Tensor):
            return v.detach().cpu().tolist()
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, dict):
            return {str(k): convert(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [convert(x) for x in v]
        return v
    path.write_text(json.dumps(convert(value), indent=2, allow_nan=False)+'\n')


def status(stage, **kwargs):
    r = dict(kwargs, stage=stage, elapsed_s=time.perf_counter()-started)
    write_json(OUT/'status.json', r)
    print('[LANGUAGE-ROUTES] '+json.dumps(r), flush=True)


def make_runtime():
    cm._EMBODIMENT_TO_RAW_ACTION_DIM['libero'] = 10
    pixels = load_file('routes_checked_pixels', ROOT/'work/check_cosmos_official_pixels_cpu.py')
    pixel_tools = pixels.load_official_pixel_tools()
    verified = json.loads((DATA/'runtime-pixel-schedule/provenance.json').read_text())
    assert verified['pixel_helper_sha256'] == sha(Path(pixels.__file__))
    assert verified['framework_transform_source_sha256'] == sha(
        ROOT/'cosmos-framework/cosmos_framework/data/generator/action/utils/transforms.py')
    status('loading_model')
    pipe = Cosmos3OmniPipeline.from_pretrained(
        str(ROOT/'checkpoints/Cosmos3-Nano-Policy-LIBEROall-5k'),
        torch_dtype=torch.bfloat16, sound_tokenizer=None,
        enable_safety_checker=False, local_files_only=True)
    pipe.to('cuda')
    pipe.vae.enable_tiling()
    assert not pipe.transformer.is_cache_enabled
    sampler = load_file('routes_native_unipc', SCHEDULER)
    original_prompt = pipe._build_action_json_prompt

    def prompt(*args, **kwargs):
        value = json.loads(original_prompt(*args, **kwargs))
        value['actions'][0]['idle_frame'] = '0 out of 16.'
        return json.dumps(value)

    def prepare_pixels(conditioning_clip, resolution_tier, num_frames, device, dtype):
        assert resolution_tier == 256 and num_frames == 17 and len(conditioning_clip) == 1
        assert dtype == torch.float32
        return pixels.prepare_official_pixels(conditioning_clip[0], num_frames, device, dtype, pixel_tools)

    pipe._build_action_json_prompt = prompt
    pipe._prepare_action_video_conditioning = prepare_pixels
    pipe.register_to_config(use_native_flow_schedule=False)
    return pipe, sampler


class Runtime:
    def __init__(self, pipe, sampler):
        self.pipe, self.sampler = pipe, sampler
        self.model = pipe.transformer
        self.routes = CosmosAttentionRoutes(self.model)
        self.record = None
        self.step = -1
        self.index = None
        self.gen_reference = None
        self.handles = [self.model.register_forward_pre_hook(self.before_model, with_kwargs=True),
                        self.model.layers[0].register_forward_pre_hook(self.before_first),
                        self.model.action_proj_out.register_forward_pre_hook(self.before_head),
                        self.model.action_proj_out.register_forward_hook(self.after_head)]
        self.handles.extend(layer.register_forward_hook(self.after_layer) for layer in self.model.layers)
        original_step = sampler.FlowUniPCMultistepScheduler.step

        def scheduler_step(instance, model_output, timestep, sample, *args, **kwargs):
            action_sample = sample.ndim == 3 and tuple(sample.shape) == (1, 16, 64)
            if self.record is not None and action_sample and not self.record['action_states']:
                self.record['action_states'].append(cpu(sample))
            result = original_step(instance, model_output, timestep, sample, *args, **kwargs)
            if self.record is not None and action_sample:
                value = result.prev_sample if hasattr(result, 'prev_sample') else result[0]
                self.record['action_states'].append(cpu(value))
            return result

        sampler.FlowUniPCMultistepScheduler.step = scheduler_step

    def before_model(self, module, args, kwargs):
        if self.record is None:
            return
        self.step += 1
        self.index = kwargs['action_mse_loss_indexes'].flatten()-kwargs['und_len']
        assert self.index.numel() == 16
        self.record['hidden'].append([])
        if self.step == 0:
            self.record['input'] = cpu(kwargs)

    def before_first(self, module, args):
        if self.record is not None:
            self.record['hidden'][self.step].append(cpu(args[1][self.index]))
            self.gen_boundary(args[1], 0)

    def after_layer(self, module, args, output):
        if self.record is not None:
            self.record['hidden'][self.step].append(cpu(output[1][self.index]))
            self.gen_boundary(output[1], len(self.record['hidden'][self.step])-1)

    def gen_boundary(self, value, boundary):
        if self.record['capture_gen']:
            if boundary == 0:
                self.record['gen_hidden'].append([])
            self.record['gen_hidden'][self.step].append(cpu(value))
        if self.gen_reference is not None:
            exact(cpu(value), self.gen_reference[self.step, boundary],
                  f'all_GEN_step{self.step}_boundary{boundary}')
            self.record['gen_exact_boundaries'] += 1

    def before_head(self, module, args):
        if self.record is not None:
            self.record['readout'].append(cpu(args[0]))

    def after_head(self, module, args, output):
        if self.record is not None:
            self.record['action_velocity'].append(cpu(output))

    def predict(self, scene, noun, label, mode='capture', recipient=None, donor=None,
                gain=1., seed=198, image=None, destination=None, capture_gen=False, gen_reference=None):
        self.record = dict(hidden=[], readout=[], action_velocity=[], action_states=[],
                           capture_gen=capture_gen, gen_hidden=[], gen_exact_boundaries=0)
        self.gen_reference = gen_reference
        self.step = -1
        if mode == 'capture':
            self.routes.begin_capture(label)
        else:
            self.routes.begin_patch(recipient, donor, mode, gain=gain)
        self.pipe.scheduler = self.sampler.FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1.)
        condition = cm.CosmosActionCondition(mode='policy', chunk_size=16, domain_name='libero',
            resolution_tier=256, image=Image.open(IMAGES[scene] if image is None else image).convert('RGB'),
            view_point='concat_view')
        began = time.perf_counter()
        try:
            output = self.pipe(prompt=f'pick up the {noun} and place it in the basket', action=condition,
                fps=20, num_inference_steps=30, guidance_scale=1., use_system_prompt=False,
                generator=torch.Generator(device='cuda').manual_seed(seed), output_type='latent')
        finally:
            routing = self.routes.end()
        assert routing['complete'], ('incomplete_route_session', routing.get('missing_layers'))
        if mode == 'capture':
            assert routing['n_steps'] == 30
        rec, self.record = self.record, None
        assert self.step == 29
        rec.update(actions=cpu(output.action[0].float()), video=cpu(output.video),
                   timesteps=cpu(self.pipe.scheduler.timesteps), sigmas=cpu(self.pipe.scheduler.sigmas))
        rec['hidden'] = torch.stack([torch.stack(x) for x in rec['hidden']])
        rec['readout'] = torch.stack(rec['readout'])
        rec['action_velocity'] = torch.stack(rec['action_velocity'])
        rec['action_states'] = torch.stack(rec['action_states'])
        if capture_gen:
            rec['gen_hidden'] = torch.stack([torch.stack(x) for x in rec['gen_hidden']])
        assert rec['hidden'].shape == (30, 37, 16, 4096)
        assert rec['readout'].shape == (30, 16, 4096)
        assert rec['action_states'].shape == (31, 1, 16, 64)
        assert rec['action_states'].dtype == torch.float32
        assert rec['timesteps'][10] == 666 and rec['timesteps'][20] == 333
        meta = dict(label=label, scene=scene, noun=noun, mode=mode, gain=gain,
                    seed=seed, pixels='official', schedule='official', use_system_prompt=False,
                    und_len=rec['input']['und_len'], timesteps=rec['timesteps'].tolist(),
                    all_gen_exact_boundaries_checked=rec['gen_exact_boundaries'],
                    elapsed_s=time.perf_counter()-began)
        if destination is not None:
            destination.mkdir(exist_ok=False)
            write_json(destination/'metadata.json', meta)
            write_json(destination/'normalized_actions.json', rec['actions'].tolist())
            torch.save({k: v for k, v in rec.items() if k != 'gen_hidden'}, destination/'states.pt')
        status('prediction_completed', **meta)
        return rec, routing, meta


def compare(rec, ref):
    return dict(action_xyz=metric(ref['actions'][:, :3], rec['actions'][:, :3]),
                action10=metric(ref['actions'], rec['actions']),
                video=metric(ref['video'], rec['video']),
                readout=[metric(ref['readout'][s], rec['readout'][s]) for s in range(30)],
                hidden_first_last=[[metric(ref['hidden'][s, l], rec['hidden'][s, l])
                                    for l in range(37)] for s in [0, 29]])


def require_match(rec, ref, label, include_video=True):
    for k in ['actions', 'hidden', 'readout', 'action_velocity', 'action_states', 'timesteps', 'sigmas']:
        exact(rec[k], ref[k], label+'/'+k)
    if include_video:
        exact(rec['video'], ref['video'], label+'/video')


def offline(rt, scenes):
    rows, controls, public_actions, public_readout = [], [], {}, {}
    ref_npz = np.load(DATA/'runtime-pixel-schedule/actions.npz')
    for scene in scenes:
        baselines, caches = {}, {}
        for noun in ['milk', 'butter']:
            label = f'{scene}_{noun}_capture'
            rec, cache, meta = rt.predict(scene, noun, label, destination=OUT/'cases'/label, capture_gen=True)
            ref = torch.load(DATA/f'runtime-pixel-schedule/cases/{scene}_{noun}_198_official_official/states.pt',
                             map_location='cpu', weights_only=False)
            exact(rec['actions'], ref['output_actions'], label+'/experiment12_actions')
            exact(rec['readout'], ref['readout'], label+'/experiment12_all_readouts')
            exact(rec['hidden'][[0, 29]], ref['hidden'], label+'/experiment12_hidden_first_last')
            exact(rec['video'], ref['output_video_latents'], label+'/experiment12_video')
            assert np.array_equal(rec['actions'].numpy(), ref_npz[f'{scene}_{noun}_198_official_official'])
            del ref
            baselines[noun], caches[noun] = rec, cache
            torch.save(cache, OUT/'cases'/label/'attention_cache.pt')
            controls.append(dict(test='corrected_experiment12_full_replay', label=label, exact=True))
            public_actions[label] = rec['actions'].numpy()
            public_readout[label] = rec['readout'][[0, 29]].float().numpy()
        a, b = baselines['milk']['input'], baselines['butter']['input']
        assert torch.nonzero(a['input_ids'] != b['input_ids']).flatten().tolist() == [48]
        for k in ['position_ids', 'vision_tokens', 'action_tokens']:
            for x, y in zip(a[k], b[k]) if isinstance(a[k], (list, tuple)) else [(a[k], b[k])]:
                exact(x, y, scene+'/'+k)
        for recipient, donor in [('milk', 'butter'), ('butter', 'milk')]:
            ref, other = baselines[recipient], baselines[donor]
            rcache, dcache = caches[recipient], caches[donor]
            specifications = [('empty', 'empty', dcache, 1.),
                              ('zero_gain', 'full', dcache, 0.),
                              ('self_full', 'full', rcache, 1.),
                              ('clamp_self', 'clamp_self', rcache, 1.),
                              ('full', 'full', dcache, 1.),
                              ('direct', 'direct', dcache, 1.),
                              ('indirect', 'indirect', dcache, 1.),
                              ('indirect_blocked', 'indirect_blocked', dcache, 1.),
                              ('joint', 'joint', dcache, 1.)]
            for name, mode, source, gain in specifications:
                label = f'{scene}_{recipient}_to_{donor}_{name}'
                rec, route_report, meta = rt.predict(scene, recipient, label, mode=mode,
                    recipient=rcache, donor=source, gain=gain, destination=OUT/'cases'/label,
                    gen_reference=(other if name == 'full' else ref)['gen_hidden']
                    if name in ['empty', 'zero_gain', 'self_full', 'clamp_self', 'full'] else None)
                if name in ['empty', 'zero_gain', 'self_full', 'clamp_self']:
                    require_match(rec, ref, label)
                    controls.append(dict(test=name, label=label, exact=True))
                if name == 'full':
                    require_match(rec, other, label)
                    assert rec['gen_exact_boundaries'] == 30*37
                    controls.append(dict(test='full_exact_donor', label=label, exact=True))
                if name == 'indirect_blocked':
                    require_match(rec, ref, label, include_video=False)
                    controls.append(dict(test='blocked_action_exact_recipient', label=label, exact=True,
                                         future_video_changed=not torch.equal(rec['video'], ref['video'])))
                if name == 'indirect':
                    exact(rec['hidden'][0, 1], ref['hidden'][0, 1], label+'/first_step_first_layer_no_future_leak')
                    controls.append(dict(test='first_layer_indirect_no_action_leak', label=label, exact=True))
                rows.append(dict(**meta, relative_to_recipient=compare(rec, ref),
                                 relative_to_donor=compare(rec, other), routing=route_report))
                public_actions[label] = rec['actions'].numpy()
                public_readout[label] = rec['readout'][[0, 29]].float().numpy()
                write_json(OUT/'result.json', dict(cases=rows, controls=controls,
                    scope='Controlled text reads and video K/V clamps at each of 36 layers and all 30 denoising steps. One paired seed, two fixed pictures; no new physical execution in offline results. Not an additive mediation percentage or proof of semantic target encoding.'))
                np.savez_compressed(OUT/'actions.npz', **public_actions)
                np.savez_compressed(OUT/'readouts_first_last.npz', **public_readout)
                del rec
        del baselines, caches
    write_json(OUT/'offline_complete.json', dict(state='complete', scenes=scenes,
               baseline_predictions=2*len(scenes), patched_predictions=len(rows), controls=len(controls)))


def serve(rt):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'ready')

        def do_POST(self):
            try:
                request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                scene, recipient, mode = request['scene'], request['noun'], request['mode']
                assert scene in IMAGES and recipient in ['milk', 'butter']
                assert mode in ['direct', 'indirect', 'indirect_blocked', 'joint', 'full']
                donor = 'butter' if recipient == 'milk' else 'milk'
                q, seed, run = int(request['chunk']), int(request['seed']), Path(request['output'])
                assert seed == 198+q and q in range(8)
                chunk = run/f'chunk_{q:02d}'
                chunk.mkdir(exist_ok=False)
                image = run/f'input_{q:02d}.png'
                r, rc, _ = rt.predict(scene, recipient, 'current_recipient', seed=seed,
                                      image=image, destination=chunk/'recipient', capture_gen=mode == 'full')
                d, dc, _ = rt.predict(scene, donor, 'current_donor', seed=seed,
                                      image=image, destination=chunk/'donor', capture_gen=mode == 'full')
                # Every new observation gets new caches from that exact image.
                patched, report, meta = rt.predict(scene, recipient, 'current_'+mode,
                    mode=mode, recipient=rc, donor=dc, seed=seed, image=image, destination=chunk/'patched',
                    gen_reference=d['gen_hidden'] if mode == 'full' else None)
                if mode == 'full':
                    require_match(patched, d, 'closed_loop_full_exact_donor')
                if mode == 'indirect_blocked':
                    require_match(patched, r, 'closed_loop_blocked_exact_recipient', include_video=False)
                if mode == 'indirect':
                    exact(patched['hidden'][0, 1], r['hidden'][0, 1], 'closed_loop_first_layer_no_leak')
                info = dict(**meta, recipient=recipient, donor=donor,
                    current_observation_sha256=sha(image),
                    relative_to_recipient=compare(patched, r), relative_to_donor=compare(patched, d),
                    routing=report, rebuilt_caches_for_current_observation=True)
                write_json(chunk/'result.json', info)
                body = json.dumps(dict(actions=patched['actions'].tolist(), mode=mode,
                    recipient=recipient, donor=donor, pixels='official', schedule='official',
                    use_system_prompt=False, seed=seed, timesteps=patched['timesteps'].tolist(),
                    rebuilt_caches_for_current_observation=True,
                    current_observation_sha256=sha(image))).encode()
                del r, d, rc, dc, patched
                self.send_response(200)
                self.end_headers()
                self.wfile.write(body)
                status('closed_loop_query_complete', case=run.name, query=q+1, mode=mode)
            except Exception as exc:
                traceback.print_exc()
                self.send_response(500)
                self.end_headers()
                self.wfile.write(str(exc).encode())
                write_json(OUT/'server_failed.json', dict(error=str(exc), traceback=traceback.format_exc()))
    status('server_ready', port=8922)
    HTTPServer(('127.0.0.1', 8922), Handler).serve_forever()


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenes', nargs='+', choices=['center', 'far'], default=['center', 'far'])
    parser.add_argument('--serve', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output_dir
    OUT.mkdir(exist_ok=False)
    (OUT/'cases').mkdir()
    write_json(OUT/'provenance.json', dict(script_sha256=sha(Path(__file__)),
        route_module_sha256=sha(ROOT/'work/cosmos_attention_routes.py'),
        pixel_helper_sha256=sha(ROOT/'work/check_cosmos_official_pixels_cpu.py'),
        scope='Natural gain-1 K/V path interventions, no weights/library edits or training.',
        corrected_reference='runtime-pixel-schedule', seed=198, scenes=args.scenes))
    try:
        pipe, sampler = make_runtime()
        rt = Runtime(pipe, sampler)
        offline(rt, args.scenes)
        status('offline_complete')
        if args.serve:
            serve(rt)
    except Exception as exc:
        write_json(OUT/'failed.json', dict(error=str(exc), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
