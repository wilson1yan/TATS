import argparse
import torch
from tats import VQGAN, VideoData
from tats.utils import save_video_grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--ckpt', type=str, required=True)
    parser = VideoData.add_data_specific_args(parser)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    data = VideoData(args, shuffle=True)
    test_loader = data.test_dataloader()
    batch = next(iter(test_loader))
    video = batch['video']
    if len(video.shape) > 5:
        video = video.view(-1, *video.shape[2:])

    model = VQGAN.load_from_checkpoint(args.ckpt).cuda()
    video = video.cuda()

    z = model.encode(video)
    recon = model.decode(z)
    recon = torch.clamp(recon + 0.5, 0, 1) # BCTHW
    print(video.shape, recon.shape)
    
    viz = torch.stack([video + 0.5, recon], dim=1)
    viz = viz.view(-1, *viz.shape[2:])
    viz = viz.cpu()
    save_video_grid(viz, 'recon.mp4')

    print('done')


if __name__ == '__main__':
    main()


