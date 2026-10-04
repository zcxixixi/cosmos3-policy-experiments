"""Read-only CPU extraction of the accepted head experiment's t0 output chain.

Extract actual pre-norm L36, observed normalized readout and observed action
velocity. Render the extracted JSON separately; no pipeline or model forward.
This is exploratory and does not modify the new language experiment's primary.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(base, output):
    import torch
    source = base / 'future-head-groups'
    complete = json.loads((source / 'complete.json').read_text())
    assert complete['state'] == 'complete' and complete['fresh_q0_predictions'] == 18
    assert complete['script_sha256'] == 'b261366b969a8f33980cbf4d450c381a18a64fa4b16be2e1dd95d6c7220ae898'
    result = json.loads((source / 'results.json').read_text())
    assert sha(source / 'results.json') == complete['results_sha256']
    evidence = {row['case']: row for row in result['cases']}
    hashes, loaded = {}, {}
    for v in (195, 198):
        for arm in ('all_allowed', 'head_G6', 'full_hardmask'):
            label = f'V{v}_A195_{arm}'
            folder, row = source / label, evidence[label]
            paths = [folder / 'states.pt', folder / 'boundaries-t00.pt']
            assert sha(paths[0]) == row['files_sha256']['states.pt']
            assert row['boundary_files'][0]['step'] == 0
            assert sha(paths[1]) == row['boundary_files'][0]['sha256']
            record, boundary = [torch.load(p, map_location='cpu', weights_only=True, mmap=True) for p in paths]
            assert boundary['step'] == 0 and boundary['action_hidden'].shape == (37, 16, 4096)
            assert record['readout'].shape == (30, 16, 4096) and record['action_velocity'].shape == (30, 16, 64)
            assert torch.equal(boundary['model_output'][2][0], record['action_velocity'][0])
            loaded[label] = dict(L36=boundary['action_hidden'][36], readout=record['readout'][0],
                XYZ_velocity=record['action_velocity'][0, :, :3], actual10_velocity=record['action_velocity'][0, :, :10],
                final_q0_XYZ=record['actions'][:, :3])
            hashes[label] = {p.name: sha(p) for p in paths}
    def stats(a, b):
        a, b = a.double(), b.double()
        assert a.shape == b.shape and torch.isfinite(a).all() and torch.isfinite(b).all()
        delta = float((a - b).square().mean().sqrt())
        ref = float(b.square().mean().sqrt())
        return dict(delta_rms=delta, reference_rms=ref, relative_delta=None if ref == 0 else delta / ref)
    rows = [dict(V=v, arm=arm, t=0, measures={key: stats(loaded[f'V{v}_A195_{arm}'][key], loaded[f'V{v}_A195_all_allowed'][key])
        for key in loaded[f'V{v}_A195_{arm}']}) for v in (195, 198) for arm in ('head_G6', 'full_hardmask')]
    payload = dict(state='complete', exploratory=True, actual_model_forwards=0, torch_version=str(torch.__version__),
        script_sha256=sha(Path(__file__)), accepted_complete_sha256=sha(source / 'complete.json'),
        source=str(source), actual_sources_sha256=hashes, rows=rows,
        formula='Float64 before subtraction; RMS(arm-sameVallallowed)/RMS(sameVallallowed) at each own position.',
        scope='t0 pre-final-norm L36, actual action_proj_out pre-hook readout and post-hook velocity. Final q0 XYZ is a total sampler effect. Different positions have different scales and reference norms. No information-retention percentage, target probability, semantic unit, new-language result or root-cause claim.')
    with output.open('x') as stream:
        stream.write(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(state='complete', output=str(output), sha256=sha(output), rows=len(rows), model_forwards=0)))


def render(source, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = json.loads(source.read_text())
    assert data['state'] == 'complete' and data['actual_model_forwards'] == 0 and not output.exists()
    keys = ('L36', 'readout', 'XYZ_velocity')
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), sharey=True)
    colors = dict(head_G6='#d47719', full_hardmask='#245bb6')
    labels = dict(head_G6='L0 heads 24-27', full_hardmask='Full first-nine-layer cut')
    for ax, v in zip(axes, (195, 198)):
        for row in data['rows']:
            if row['V'] != v:
                continue
            values = [100 * row['measures'][key]['relative_delta'] for key in keys]
            ax.plot(range(3), values, marker='o', color=colors[row['arm']], label=labels[row['arm']])
            for x, y in enumerate(values):
                ax.annotate(f'{y:.2f}%', (x, y), xytext=(0, 7), textcoords='offset points', ha='center', fontsize=9)
        ax.set_title(f'Future-noise source {v}; first forward only')
        ax.set_xticks(range(3), ['L36\nbefore norm', 'Observed\nnormalized readout', 'Observed\nXYZ velocity'])
        ax.set_ylim(0, 7.2)
        ax.grid(axis='y', alpha=.22)
    axes[0].set_ylabel('RMS difference / own reference RMS (%)')
    axes[0].legend(fontsize=8, loc='upper right')
    fig.suptitle('Actual output chain: changed hidden states need not change target selection', fontsize=12)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    print(json.dumps(dict(figure=str(output), sha256=sha(output), data_sha256=sha(source))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('extract', 'render'), required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    extract(args.source, args.output) if args.mode == 'extract' else render(args.source, args.output)


if __name__ == '__main__':
    main()
