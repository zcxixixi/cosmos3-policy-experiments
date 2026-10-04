"""Reuse the frozen milk baseline entry points for X+0/9/12cm trials.

Pass --output BASE --mode collect|server|simulate. Only collect creates the
absent BASE/position-threshold root. The original X+0/6/15cm baseline and
legacy source files are read-only. Their real __file__ values are preserved.

The wrapper changes only module globals and forwards argv to the original
main functions. Three positions x three paired seeds retain the legacy
server's nine trials, 72 queries and 48 paired-noise checks. Exact X+0 inputs
and saved q0 records provide controls against the original baseline. These
trials narrow the position range of the target switch; they do not localize
object semantics or a neural root cause.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import traceback


ROOT = Path('/home/current/work/cosmos3')
SHIFTS = {'x00': 0, 'x09': 9, 'x12': 12}
SEEDS = (195, 196, 198)
PORT = 8927
URL = f'http://127.0.0.1:{PORT}'
PROMPT = 'pick up the milk and place it in the basket'
SCOPE = ('Nine matched 128-action milk trials at X+0/9/12cm with three paired '
         'noise seeds and the existing warmed snapshot. Narrow the target-switch '
         'position range; no semantic, neural-root-cause or benchmark-rate claim.')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def write_new_json(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frozen_contract(baseline):
    """Verify original recorded source hashes before any stage writes output."""
    for name in ('server/failed.json', 'inputs/failed.json', 'closed-loop/failed.json', 'simulator_failed.json'):
        assert not (baseline / name).exists(), f'Original baseline failure requires review: {name}'
    server = read_json(baseline / 'server/provenance.json')
    inputs = read_json(baseline / 'inputs/metadata.json')
    simulator = read_json(baseline / 'closed-loop/provenance.json')
    input_complete = read_json(baseline / 'inputs/complete.json')
    assert input_complete['state'] == 'matched_inputs_verified'
    assert input_complete['metadata_sha256'] == sha(baseline / 'inputs/metadata.json')
    assert input_complete['source_center_pixels_exact'] is True
    assert inputs['prompt'] == PROMPT and inputs['shifts_cm'] == {'x00': 0, 'x06': 6, 'x15': 15}
    assert inputs['seeds'] == list(SEEDS) and inputs['additional_warmup_steps'] == 0
    assert inputs['script_sha256'] == simulator['script_sha256'], 'Original collection/simulation source differs'
    complete = read_json(baseline / 'closed-loop/complete.json')
    assert complete['state'] == 'complete' and complete['cases'] == 9
    assert complete['queries_per_case'] == 8 and complete['steps_per_case'] == 128
    assert complete['state_records_per_case'] == 129
    expected = {f'{scene}_seed{seed}' for scene in inputs['shifts_cm'] for seed in SEEDS}
    assert set(complete['completed_trials']) == expected
    server_complete = read_json(baseline / 'server/complete.json')
    assert server_complete['state'] == 'complete' and server_complete['predictions'] == 72
    assert server_complete['paired_noise_checks'] == 48
    assert server_complete['observer_free_repeat_all_tensors_exact'] is True
    assert read_json(baseline / 'summary.json') == read_json(baseline / 'closed-loop/summary.json')
    work = Path(__file__).resolve().parent
    files = {
        'server': (work / 'serve_cosmos_milk_shift_mechanism.py', server['script_sha256']),
        'simulator': (work / 'rollout_cosmos_milk_shift_mechanism.py', inputs['script_sha256']),
        'normal_runtime': (ROOT / 'work/serve_cosmos_goal_pair.py', server['normal_runtime_sha256']),
        'factory': (ROOT / 'work/probe_cosmos_language_routes.py', server['factory_sha256']),
        'components': (ROOT / 'work/cosmos_component_interventions.py', server['components_sha256']),
        'pixel_helper': (ROOT / 'work/check_cosmos_official_pixels_cpu.py', server['pixel_helper_sha256']),
        'scheduler': (ROOT / 'cosmos-framework/cosmos_framework/model/generator/diffusion/samplers/fm_solvers_unipc.py', server['scheduler_sha256']),
        'collector': (ROOT / 'work/collect_cosmos_goal_pair.py', inputs['collector_sha256']),
        'controller_helper': (ROOT / 'work/rollout_cosmos_goal_pair.py', inputs['controller_helper_sha256']),
        'normalizer_stats': (ROOT / 'cosmos-framework/cosmos_framework/data/generator/action/normalizer_stats/libero_native_frame_wise_relative_rot6d.json', simulator['stats_sha256']),
        'source_state': (Path(inputs['source_state']), inputs['source_state_sha256']),
        'source_png': (Path(inputs['source_png']), inputs['source_png_sha256']),
    }
    for key, (path, expected_sha) in files.items():
        assert sha(path) == expected_sha, f'Original {key} source/data changed: {path}'
    return dict(wrapper_sha256=sha(Path(__file__)), baseline=str(baseline),
        baseline_server_provenance_sha256=sha(baseline / 'server/provenance.json'),
        baseline_inputs_metadata_sha256=sha(baseline / 'inputs/metadata.json'),
        baseline_simulator_provenance_sha256=sha(baseline / 'closed-loop/provenance.json'),
        baseline_summary_sha256=sha(baseline / 'summary.json'),
        frozen_files={key: dict(path=str(path), sha256=expected_sha) for key, (path, expected_sha) in files.items()},
        shifts_cm=SHIFTS, seeds=list(SEEDS), prompt=PROMPT, port=PORT, url=URL,
        trials=9, queries=72, paired_noise_checks=48, library_edits=False, scope=SCOPE)


def zero_input_control(baseline, output, simulator):
    """Compare raw-array manifests, PNG bytes and actual controller arrays."""
    original, current = baseline / 'inputs/x00', output / 'inputs/x00'
    assert read_json(original / 'arrays.json') == read_json(current / 'arrays.json'), 'X+0 full physical/input array manifests differ'
    for name in ('agentview.png', 'robot0_eye_in_hand.png', 'input.png'):
        assert (original / name).read_bytes() == (current / name).read_bytes(), f'X+0 original PNG bytes differ: {name}'
    with simulator.np.load(original / 'controller.npz', allow_pickle=False) as a, simulator.np.load(current / 'controller.npz', allow_pickle=False) as b:
        assert set(a.files) == set(b.files), 'X+0 controller fields differ'
        for key in a.files:
            left, right = a[key], b[key]
            assert left.dtype == right.dtype and left.shape == right.shape, f'X+0 controller {key} type/shape'
            assert left.tobytes(order='C') == right.tobytes(order='C'), f'X+0 controller {key} bytes'
    return dict(state='exact', original_input=str(original), current_input=str(current),
        all_physical_and_observation_array_manifests_exact=True, all_three_png_bytes_exact=True,
        all_controller_array_bytes_exact=True, model_input_png_bytes_exact=True,
        original_arrays_sha256=sha(original / 'arrays.json'), current_arrays_sha256=sha(current / 'arrays.json'))


def zero_model_control(baseline, output):
    """CPU checks of existing q0 files; no additional policy forward."""
    import torch

    def exact(a, b, label):
        if isinstance(a, torch.Tensor):
            assert isinstance(b, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype, label
            assert torch.equal(a, b), label
            assert torch.equal(a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8)), label + '/bytes'
        elif isinstance(a, dict):
            assert isinstance(b, dict) and a.keys() == b.keys(), label
            for key in a:
                exact(a[key], b[key], f'{label}/{key}')
        elif isinstance(a, (tuple, list)):
            assert type(a) is type(b) and len(a) == len(b), label
            for index, (left, right) in enumerate(zip(a, b)):
                exact(left, right, f'{label}/{index}')
        else:
            assert a == b, label

    controls = []
    for seed in SEEDS:
        trial = f'x00_seed{seed}'
        old = baseline / 'closed-loop' / trial / 'chunk_00/states.pt'
        new = output / 'closed-loop' / trial / 'chunk_00/states.pt'
        original = torch.load(old, map_location='cpu', weights_only=True, mmap=True)
        current = torch.load(new, map_location='cpu', weights_only=True, mmap=True)
        exact(original, current, trial + '/all_saved_normal_runtime_fields')
        controls.append(dict(trial=trial, all_saved_q0_fields_bit_exact=True,
                             original_states_sha256=sha(old), current_states_sha256=sha(new)))
    return dict(state='exact', predictions_compared=3, extra_model_forwards=0, controls=controls)


def legacy_main(module, arguments):
    previous = sys.argv
    sys.argv = [module.__file__, *arguments]
    try:
        module.main()
    finally:
        sys.argv = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Completed original X+0/6/15cm baseline root')
    parser.add_argument('--mode', choices=('collect', 'server', 'simulate'), required=True)
    args = parser.parse_args()
    baseline = args.output.resolve()
    output = baseline / 'position-threshold'
    contract = frozen_contract(baseline)
    assert len(SHIFTS) * len(SEEDS) == 9 and SHIFTS['x00'] == 0
    if args.mode == 'collect':
        output.mkdir(exist_ok=False)
    else:
        assert output.is_dir(), 'Run collect first'
        assert not (output / 'wrapper_failed.json').exists(), 'Wrapper failure requires review; no automatic retry'
        saved = read_json(output / 'provenance.json')
        assert saved['contract'] == contract, 'Frozen wrapper/source/baseline contract changed'
        assert saved['inputs_metadata_sha256'] == sha(output / 'inputs/metadata.json')
        assert saved['zero_input_control_sha256'] == sha(output / 'zero-input-control.json')
    try:
        role = 'server' if args.mode == 'server' else 'simulator'
        source = Path(contract['frozen_files'][role]['path'])
        module = load_file('threshold_frozen_' + role, source)
        assert Path(module.__file__).resolve() == source.resolve(), 'Preserve actual legacy __file__'
        module.SHIFTS, module.SEEDS = dict(SHIFTS), SEEDS
        if role == 'simulator':
            module.TRIALS = {f'{scene}_seed{seed}': (scene, seed) for scene in SHIFTS for seed in SEEDS}
            module.URL, module.SCOPE = URL, SCOPE
            assert str(module.SOURCE / 'state.npy') == contract['frozen_files']['source_state']['path']
        else:
            module.PORT = PORT
        arguments = ['--output', str(output)]
        if args.mode == 'collect':
            arguments.append('--collect-only')
        legacy_main(module, arguments)
        if args.mode == 'collect':
            zero = zero_input_control(baseline, output, module)
            write_new_json(output / 'zero-input-control.json', zero)
            write_new_json(output / 'provenance.json', dict(contract=contract,
                inputs_metadata_sha256=sha(output / 'inputs/metadata.json'),
                zero_input_control_sha256=sha(output / 'zero-input-control.json')))
        elif args.mode == 'server':
            write_new_json(output / 'zero-model-control.json', zero_model_control(baseline, output))
            write_new_json(output / 'wrapper-server-complete.json', dict(state='complete',
                server_complete_sha256=sha(output / 'server/complete.json'),
                zero_model_control_sha256=sha(output / 'zero-model-control.json'),
                provenance_sha256=sha(output / 'provenance.json')))
        else:
            write_new_json(output / 'wrapper-simulator-complete.json', dict(state='complete',
                simulator_complete_sha256=sha(output / 'closed-loop/complete.json'),
                provenance_sha256=sha(output / 'provenance.json')))
        print(f'[POSITION-THRESHOLD] {args.mode} complete: {output}', flush=True)
    except Exception as exc:
        failure = output / 'wrapper_failed.json'
        if not failure.exists():
            write_new_json(failure, dict(mode=args.mode, error=str(exc), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
