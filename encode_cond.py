import argparse
import numpy as np
from tqdm import tqdm
import torch
from tats import VQGAN, VideoData
from tats.utils import save_video_grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--ckpt', type=str, required=True)
    parser = VideoData.add_data_specific_args(parser)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    data = VideoData(args, shuffle=False)
    test_loader = data.test_dataloader()

    model = VQGAN.load_from_checkpoint(args.ckpt).cuda()

    encodings = []
    actions = []
    N, B = 512, 32
    total = 0
    pbar = tqdm(total=N)
    for batch in test_loader:
        video = batch['video'][:, :, :36].cuda()
        act = batch['actions']
        z = model.encode(video)
        encodings.append(z.cpu().numpy().astype(np.int32))
        actions.append(act.cpu().numpy().astype(np.int32))
        pbar.update(video.shape[0])
        total += video.shape[0]
        if total >= N:
            break
    encodings = np.concatenate(encodings)[:N]
    actions = np.concatenate(actions)[:N]

    print(encodings.shape, actions.shape)
    np.savez('encoded_cond.npz', encodings=encodings, actions=actions)

    print('done')


if __name__ == '__main__':
    main()


