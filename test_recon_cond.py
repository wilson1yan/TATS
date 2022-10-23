import argparse
import torch
from tats import VQGAN, VideoData
from tats.utils import save_video_grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--ckpt', type=str, required=True)
    parser.add_argument('-o', '--open_loop_ctx', type=int, default=36)
    parser = VideoData.add_data_specific_args(parser)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    data = VideoData(args, shuffle=True)
    test_loader = data.test_dataloader()
    batch = next(iter(test_loader))
    video = batch['video']
    if len(video.shape) > 5:
        video = video.view(-1, *video.shape[2:])
    video = video[:1]

    cond = video[:, :, :args.open_loop_ctx]

    model = VQGAN.load_from_checkpoint(args.ckpt).cuda()
    cond, video = cond.cuda(), video.cuda()

    print(video.shape, cond.shape)
    z_cond = model.encode(cond)
    z = model.encode(video)
    import ipdb; ipdb.set_trace()
    z[:, :args.open_loop_ctx // 4] = z_cond
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


