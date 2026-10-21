"""Export GAT attention on random held-out WildfireSpreadTS fires.

Run from the repository root, e.g.:
  python scripts/extract_fire_attention.py --fold 0 --seed 1 --n-fires 10

The output is one compressed NPZ per sampled fire/window, plus manifest.json.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from graphml_wildfire.train import TrainConfig, FireData, build_model, forward
from graphml_wildfire.data.patches import CHANNEL_INDEX


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--n-fires', type=int, default=10)
    p.add_argument('--sample-seed', type=int, default=2026)
    p.add_argument('--protocol', default='wsts')
    p.add_argument('--data-dir', default='')
    p.add_argument('--checkpoint-dir', default='data/checkpoints')
    p.add_argument('--checkpoint', default='', help='Exact .pt path; otherwise discover by fold and seed')
    p.add_argument('--output-dir', default='results/attention_analysis')
    args = p.parse_args()

    config = TrainConfig(model='pixel_gtcnn', fold=args.fold, seed=args.seed,
                         protocol=args.protocol, data_dir=args.data_dir,
                         checkpoint_dir=args.checkpoint_dir, compile=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Loading fold data...', flush=True)
    data = FireData(config)
    model = build_model(config, data).to(device)

    if args.checkpoint:
        ckpt = Path(args.checkpoint)
    else:
        candidates = sorted(Path(args.checkpoint_dir).glob(
            f'{config.experiment_name}_f{args.fold}_s{args.seed}_*.pt'))
        if len(candidates) != 1:
            raise RuntimeError(f'Expected one checkpoint; found {len(candidates)}: {candidates}. '
                               'Supply --checkpoint explicitly.')
        ckpt = candidates[0]
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f'Loaded {ckpt}', flush=True)

    attention_modules = [(name, module) for name, module in model.named_modules()
                         if type(module).__name__ == 'EdgeAttention']
    if not attention_modules:
        raise RuntimeError('No EdgeAttention modules found. Check your model implementation.')
    print('Attention modules:', [name for name, _ in attention_modules], flush=True)

    captures = {}
    handles = []
    for name, module in attention_modules:
        def capture(_module, _inputs, output, name=name):
            if not isinstance(output, torch.Tensor):
                raise TypeError(f'{name} returned {type(output)} instead of a tensor')
            captures[name] = output.detach().to('cpu', dtype=torch.float16)
        handles.append(module.register_forward_hook(capture))

    # Choose distinct fires, then one uniformly random available prediction day per fire.
    by_fire = {}
    for fire_index, day in data.samples['test']:
        by_fire.setdefault(fire_index, []).append(day)
    rng = random.Random(args.sample_seed)
    fire_indices = rng.sample(sorted(by_fire), min(args.n_fires, len(by_fire)))

    output_dir = Path(args.output_dir) / f'fold_{args.fold}_seed_{args.seed}'
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    try:
        for number, fire_index in enumerate(fire_indices):
            day = rng.choice(by_fire[fire_index])
            features, valid, target, _, pixels = data.batch([(fire_index, day)], device)
            captures.clear()
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16,
                                                        enabled=device.type == 'cuda'):
                logits = forward(model, features, valid, data, pixels)
            if len(captures) != len(attention_modules):
                raise RuntimeError(f'Only captured {list(captures)} of {len(attention_modules)} layers')
            attention = torch.stack([captures[name] for name, _ in attention_modules])
            # Each layer: [B*T, heads, 8, H, W]; batch size is one.
            expected = (config.window, config.heads, 8, valid.shape[-2], valid.shape[-1])
            if tuple(attention.shape[1:]) != expected:
                raise ValueError(f'Unexpected attention shape {tuple(attention.shape)}; expected [L, {expected}]')

            def raw_channel(channel):
                i = CHANNEL_INDEX[channel]
                return (features[0, :, i] * float(data.std[i]) + float(data.mean[i])).cpu().numpy()

            # Wind and fire history are on the same node grid as attention.
            arrays = {
                'attention': attention.numpy(),  # [layers, time, heads, neighbor, row, col]
                'valid': valid[0].cpu().numpy().astype(np.uint8),
                'wind_row': raw_channel('wind_row').astype(np.float32),
                'wind_col': raw_channel('wind_col').astype(np.float32),
                'fire_frac': raw_channel('fire_frac').astype(np.float32),
                'burned_ever': raw_channel('burned_ever').astype(np.float32),
                'target_node': target[0].cpu().numpy().astype(np.float32),
                'prediction_node_or_pixel': torch.sigmoid(logits[0].float()).cpu().numpy().astype(np.float16),
            }
            if pixels is not None:
                arrays['target_pixel'] = pixels['target'][0].cpu().numpy().astype(np.uint8)
                arrays['valid_pixel'] = pixels['valid'][0].cpu().numpy().astype(np.uint8)
            path = output_dir / f'fire_{fire_index}_day_{day}.npz'
            np.savez_compressed(path, **arrays)
            manifest.append({'file': path.name, 'fire_index': fire_index, 'day': day,
                             'fold': args.fold, 'seed': args.seed, 'checkpoint': str(ckpt),
                             'attention_modules': [name for name, _ in attention_modules]})
            print(f'[{number+1}/{len(fire_indices)}] {path} attention={tuple(attention.shape)}', flush=True)
    finally:
        for handle in handles:
            handle.remove()
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('Done:', output_dir)


if __name__ == '__main__':
    main()
