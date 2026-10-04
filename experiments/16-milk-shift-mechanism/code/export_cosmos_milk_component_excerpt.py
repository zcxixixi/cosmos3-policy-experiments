"""Export a bounded, byte-exact excerpt from completed immutable captures.

Stores BF16 values as uint16 bit patterns, not rounded decimal approximations.
All full captures remain on the A6000; their hashes and exact source keys are
included. This is packaging of measured values, with no inference or rerun.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    assert json.loads((root/'server/complete.json').read_text())['state'] == 'complete'
    out = root/'component-excerpt'
    out.mkdir(exist_ok=False)
    arrays, entries, sources = {}, [], []
    for position in ('x00', 'x06', 'x15'):
        for seed in (195, 196, 198):
            trial = f'{position}_seed{seed}'
            path = root/'closed-loop'/trial/'chunk_00/components.pt'
            cache = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
            assert cache['complete'] and cache['n_steps'] == 30
            assert len(cache['frames']) == 30*36*2
            sources.append(dict(trial=trial, path=str(path), bytes=path.stat().st_size, sha256=sha(path)))
            for step in (20, 21, 29):
                for layer in (17, 29, 31):
                    for component in ('attention', 'mlp'):
                        value = cache['frames'][(step, layer, component)].contiguous()
                        assert value.shape == (16, 4096) and value.dtype == torch.bfloat16
                        assert torch.isfinite(value).all()
                        key = f'{trial}__step{step:02d}__layer{layer:02d}__{component}'
                        bits = value.view(torch.uint16).numpy().copy()
                        assert torch.equal(torch.from_numpy(bits).view(torch.bfloat16), value)
                        arrays[key] = bits
                        entries.append(dict(key=key, trial=trial, step=step, layer_zero_based=layer,
                            layer_human=layer+1, component=component, shape=list(value.shape),
                            original_dtype='torch.bfloat16', stored_dtype='uint16 BF16 bit pattern',
                            raw_bytes_sha256=hashlib.sha256(bits.tobytes()).hexdigest()))
    np.savez_compressed(out/'actual-bf16-bits.npz', **arrays)
    with np.load(out/'actual-bf16-bits.npz', allow_pickle=False) as saved:
        assert set(saved.files) == set(arrays)
        for key, value in arrays.items():
            assert np.array_equal(saved[key], value)
    manifest = dict(state='complete', sources=sources, entries=entries, count=len(entries),
        export_script_sha256=sha(Path(__file__)), archive_sha256=sha(out/'actual-bf16-bits.npz'),
        archive_bytes=(out/'actual-bf16-bits.npz').stat().st_size,
        decode="bits=np.load(path)[key]; values=torch.from_numpy(bits.copy()).view(torch.bfloat16).float()",
        interpretation='Actual attention/MLP increments on 16 action rows; no residual, identity label or grasp probability.')
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(state='complete', count=len(entries), bytes=manifest['archive_bytes'])))


if __name__ == '__main__':
    main()
