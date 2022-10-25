import numpy as np
from tqdm import tqdm
import os
import os.path as osp
import argparse
import torch
from tats import VQGAN, VideoData
from tats.utils import save_video_grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n', type=int, default=512)
    parser.add_argument('-c', '--ckpt', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)
    parser = VideoData.add_data_specific_args(parser)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    data = VideoData(args, shuffle=True)
    test_loader = data.test_dataloader()

    model = VQGAN.load_from_checkpoint(args.ckpt).cuda()

    pbar = tqdm(total=args.n)
    recons, reals = [], []
    for batch in test_loader:
        video = batch['video'].cuda()
        z = model.encode(video)
        recon = model.decode(z)
        recon = torch.clamp(recon + 0.5, 0, 1) # BCTHW

        video = ((video + 0.5) * 255).byte()
        recon = (recon * 255).byte()

        recons.append(recon.cpu().numpy())
        reals.append(video.cpu().numpy())

        pbar.update(args.batch_size)
        total = sum([r.shape[0] for r in recons])
        if total >= args.n:
            break

    recons = np.concatenate(recons)[:args.n]
    reals = np.concatenate(reals)[:args.n]

    print(reals.shape, recons.shape, reals.dtype, recons.dtype)

    folder = args.output
    os.makedirs(folder, exist_ok=True)
    np.savez_compressed(osp.join(folder, 'data.npz'), real=reals, fake=recons)
    print('Saved to', osp.join(folder, 'data.npz'))


if __name__ == '__main__':
    main()


