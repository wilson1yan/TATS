import argparse
import h5py
import torch
from tats import VQGAN
from tats.utils import save_video_grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--ckpt', type=str, required=True)
    parser.add_argument('-b', '--batch_size', type=int, default=4)
    parser.add_argument('-d', '--data_path', type=str, required=True)
    parser.add_argument('-s', '--seq_len', type=int, default=16)
    args = parser.parse_args()

    torch.set_grad_enabled(False)

    data = h5py.File(args.data_path, 'r')
    encodings = data['test_data'][:args.batch_size, 20:20 + args.seq_len]
    encodings = torch.LongTensor(encodings).cuda()
    model = VQGAN.load_from_checkpoint(args.ckpt).cuda()
    recon = model.decode(encodings)
    recon = torch.clamp(recon + 0.5, 0, 1) # BCTHW
    recon = recon.cpu()
    save_video_grid(recon, 'recon_vq.mp4')

    print('done')


if __name__ == '__main__':
    main()


