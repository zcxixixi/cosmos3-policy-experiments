"""Reproduce all30 saved random head sites in the actual producer Torch runtime.

CPU only, exactly torch2.14.0+cu130, no attention/model/solver/physics call.
Create absent BASE/future-head-random-norm-replay. Hash the immutable c3d
analysis-source.py and its original failed.json as failure evidence.
Seal raw tensor hashes and the producer-exact FP32/Float64 reduction scalars;
the Torch2.7 plotter can independently check the pointwise and BF16 byte chain
without pretending a different reduction implementation is bit-identical.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback


ROOT = Path('/home/current/work/cosmos3')
PROBE_SHA = 'b261366b969a8f33980cbf4d450c381a18a64fa4b16be2e1dd95d6c7220ae898'
FAILED_ANALYZER_SHA = 'c3d707e45545ffaf0c89d230fd25b507e2110782ac25e344716f03ec47391f27'
TORCH_VERSION = '2.14.0+cu130'
STAGE = 'future-head-random-norm-replay'
FP32_KEYS = ('FP32random_Dk_selected', 'FP32random_direction', 'FP32random_unit_direction',
             'FP32random_delta_selected', 'FP32random_selected')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    with path.open('x') as stream:
        stream.write(json.dumps(data, indent=2, allow_nan=False) + '\n')


def raw(tensor, torch):
    return tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def exact(a, b, torch):
    return a.dtype == b.dtype and a.shape == b.shape and raw(a, torch) == raw(b, torch)


def meta(tensor, torch):
    return dict(shape=list(tensor.shape), dtype=str(tensor.dtype), sha256=hashlib.sha256(raw(tensor, torch)).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    base, producer = args.output.resolve(), args.output.resolve() / 'future-head-groups'
    source = ROOT / 'work/probe_cosmos_milk_future_head_groups.py'
    old_source = base / 'future-head-effects/analysis-source.py'
    failed = base / 'future-head-effects/failed.json'
    assert sha(source) == PROBE_SHA and sha(old_source) == FAILED_ANALYZER_SHA
    assert read(failed)['state'] == 'failed'
    names = ('complete', 'results', 'provenance', 'protocol', 'sources', 'primary-prediction')
    docs = {name: read(producer / (name + '.json')) for name in names}
    complete, results = docs['complete'], docs['results']
    assert not (producer / 'failed.json').exists() and complete['state'] == results['state'] == 'complete'
    assert complete['script_sha256'] == docs['provenance']['script_sha256'] == PROBE_SHA
    assert complete['fresh_q0_predictions'] == 18 and complete['fresh_model_forwards'] == 540
    assert complete['actual_selected_site_captures'] == 2430 and complete['total_official_dispatch_calls'] == 57090
    for name in names[1:]:
        assert complete[name.replace('-', '_') + '_sha256'] == sha(producer / (name + '.json'))
    out = base / STAGE
    out.mkdir(exist_ok=False)
    started = time.perf_counter()
    try:
        import torch
        assert str(torch.__version__) == TORCH_VERSION, str(torch.__version__)
        assert not torch.cuda.is_initialized()
        torch.set_num_threads(1)  # Root established2.14 reductions are unchanged at1..32 threads.
        global_before = torch.get_rng_state().clone()
        rows = []
        for v in (195, 198):
            label = f'V{v}_A195_random_G6'
            row = next(r for r in results['cases'] if r['case'] == label)
            folder = producer / label
            for name, expected in row['files_sha256'].items():
                assert sha(folder / name) == expected
            manifest = read(folder / 'site-captures.json')
            assert manifest['state'] == 'complete' and manifest['sites'] == 135
            items = [item for item in manifest['files'] if item['layer_zero_based'] == 0]
            assert [item['step'] for item in items] == list(range(15))
            for item in items:
                path = folder / item['file']
                assert sha(path) == item['sha256'] and path.stat().st_size == item['bytes']
                d = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
                t = item['step']
                assert (d['step'], d['layer_zero_based'], d['case_arm']) == (t, 0, 'random_G6')
                assert d['selected_query_heads'] == list(range(24, 28))
                z = d['endpoints']
                assert {name for name, tensor in z.items() if tensor is not None} == {'Znative', 'Z0', 'Zk', 'Zreal'}
                for tensor in (d['Qcurrent50'], d['Kall'], d['Vall'], *(a for a in z.values() if a is not None)):
                    assert tensor.device.type == 'cpu' and tensor.dtype == torch.bfloat16 and bool(torch.isfinite(tensor).all())
                for name in FP32_KEYS:
                    tensor = d[name]
                    assert tensor.shape == (1, 50, 4, 128) and tensor.dtype == torch.float32 and bool(torch.isfinite(tensor).all())
                heads = torch.tensor(list(range(24, 28)), dtype=torch.long)
                baseline, cut = z['Z0'].index_select(2, heads), z['Zk'].index_select(2, heads)
                natural = cut.float() - baseline.float()
                generator = torch.Generator(device='cpu').manual_seed(901000 + t * 36)
                assert exact(generator.get_state(), d['random_generator_before'], torch)
                direction = torch.randn(natural.shape, dtype=torch.float32, device='cpu', generator=generator)
                assert exact(direction, d['FP32random_direction'], torch)
                assert exact(generator.get_state(), d['random_generator_after'], torch)
                target, raw_norm = natural.norm(), direction.norm()
                unit = direction / raw_norm
                increment = unit * target
                candidate = baseline.float() + increment
                for name, tensor in zip(FP32_KEYS, (natural, direction, unit, increment, candidate)):
                    assert exact(d[name], tensor, torch), (label, t, name)
                actual = torch.where(increment == 0, baseline, candidate.to(torch.bfloat16))
                assert exact(actual, z['Zreal'].index_select(2, heads), torch)
                retained = torch.tensor([h for h in range(32) if h not in range(24, 28)], dtype=torch.long)
                assert exact(z['Zreal'].index_select(2, retained), z['Z0'].index_select(2, retained), torch)
                target_l2 = float((cut.double() - baseline.double()).norm())
                actual_l2 = float((actual.double() - baseline.double()).norm())
                error = 0.0 if target_l2 == actual_l2 == 0 else abs(actual_l2 / target_l2 - 1)
                report = d['random_control']
                assert report == d['event']['random_control'] == manifest['random_control']['checks'][t]
                assert report['target_Dk_FP32_l2'] == float(target)
                assert report['random_delta_FP32_l2'] == float(increment.norm())
                assert report['target_Dk_actual_BF16_l2'] == target_l2 and report['random_actual_BF16_l2'] == actual_l2
                assert report['actual_BF16_relative_norm_error'] == error
                assert report['actual_BF16_norm_ratio'] == (None if target_l2 == 0 else actual_l2 / target_l2)
                assert report['actual_BF16_equal_strength_within_tolerance'] is (error <= .10)
                assert report['actual_BF16_changed_values'] == int(torch.count_nonzero(actual.double() - baseline.double()))
                assert report['BF16_quantization_error_l2'] == float((actual.double() - candidate.double()).norm())
                for device in ('cpu', 'cuda'):
                    assert exact(d['RNG_before'][device], d['RNG_after'][device], torch)
                rows.append(dict(case=label, step=t, layer_zero_based=0, source_file=item['file'], source_sha256=item['sha256'],
                    FP32_raw_norm=float(raw_norm), FP32_target_norm=float(target),
                    FP32_raw_norm_int32_bits=int(raw_norm.view(torch.int32)), FP32_target_norm_int32_bits=int(target.view(torch.int32)),
                    actual_BF16_target_double_L2=target_l2, actual_BF16_random_double_L2=actual_l2,
                    actual_BF16_relative_norm_error=error,
                    source_raw_tensor_hashes={**{name: meta(d[name], torch) for name in FP32_KEYS},
                        **{name: meta(d[name], torch) for name in ('Qcurrent50', 'Kall', 'Vall', 'random_generator_before', 'random_generator_after')},
                        **{name: meta(tensor, torch) for name, tensor in z.items() if tensor is not None}},
                    original_random_control_report=report, all_FP32_witnesses_and_final_BF16_bytes_exact=True,
                    original_double_norm_report_exact=True))
        assert len(rows) == 30 and exact(global_before, torch.get_rng_state(), torch)
        assert not torch.cuda.is_initialized()
        write(out / 'norms.json', dict(state='complete', sites=rows, sites_checked=30, torch_version=str(torch.__version__),
            CPU_threads=torch.get_num_threads(), runtime_scope='Producer-version CPU reductions; every saved pointwise/BF16 witness and original double norm report exactly reproduced.',
            model_calls=0, official_attention_dispatch_calls=0, solver_calls=0, physics_calls=0, CUDA_initialized=False))
        write(out / 'provenance.json', dict(script_sha256=sha(Path(__file__)), producer_script_sha256=PROBE_SHA,
            producer_documents_sha256={name: sha(producer / (name + '.json')) for name in names},
            failed_analysis_source_path=str(old_source), failed_analysis_source_sha256=FAILED_ANALYZER_SHA,
            failed_analysis_artifact_path=str(failed), failed_analysis_artifact_sha256=sha(failed),
            torch_version=str(torch.__version__), CPU_threads=torch.get_num_threads(), private_CPU_generator_only=True,
            global_CPU_RNG_bytes_unchanged=True, all30_actual_source_file_and_raw_tensor_hashes_in_norms=True))
        write(out / 'complete.json', dict(state='complete', script_sha256=sha(Path(__file__)), producer_script_sha256=PROBE_SHA,
            producer_complete_sha256=sha(producer / 'complete.json'), failed_analysis_source_sha256=FAILED_ANALYZER_SHA,
            failed_analysis_artifact_sha256=sha(failed), sites_checked=30, torch_version=str(torch.__version__),
            all30_FP32_pointwise_witnesses_and_final_BF16_bytes_exact=True, all30_original_double_norm_reports_exact=True,
            files_sha256={name: sha(out / name) for name in ('norms.json', 'provenance.json')},
            model_calls=0, official_attention_dispatch_calls=0, solver_calls=0, physics_calls=0, CUDA_initialized=False,
            elapsed_s=time.perf_counter() - started))
        print('[RANDOM-NORMS-REPLAY] complete30 exact sites', flush=True)
    except Exception as exc:
        write(out / 'failed.json', dict(state='failed', error=str(exc), traceback=traceback.format_exc(), model_calls=0, physics_calls=0))
        raise


if __name__ == '__main__':
    main()
