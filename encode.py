import argparse
import h5py
import os.path as osp
import numpy as np
from tqdm import tqdm
import glob
import torch
import torch.multiprocessing as mp
from tats import VQGAN


class NumpyDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, train=True, n_chunks=None, chunk_id=None):
        super().__init__()
        split = 'train' if train else 'test'
        self.files = glob.glob(osp.join(data_file, split, '**', '*.npz'), recursive=True)
        if n_chunks is not None:
            self.files = np.array_split(self.files, n_chunks)[chunk_id].tolist()
            print(n_chunks, chunk_id, len(self.files))
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        fname = self.files[idx]
        data = np.load(fname)
        video = data['video'] # THWC
        video = torch.FloatTensor(video) / 255. - 0.5 # THWC [-0.5, 0.5]
        video = video.movedim(-1, 0)

        actions = data['actions']
        actions = torch.FloatTensor(actions)
        
        return dict(video=video, actions=actions)

        
def worker(i, args, split, queue):
    device = torch.device(f'cuda:{i}')
    model = VQGAN.load_from_checkpoint(args.ckpt).to(device)
    model.eval()
    torch.set_grad_enabled(False)

    dataset = NumpyDataset(args.data_path, train=split == 'train', n_chunks=torch.cuda.device_count(), chunk_id=i)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.batch_size,
        pin_memory=False, shuffle=False, drop_last=False
    )
    for batch in loader:
        video, actions = batch['video'], batch['actions']
        video = video.to(device, non_blocking=True)
        encoding = model.encode(video)
        encoding = encoding.cpu().numpy()
        actions = actions.numpy()
        queue.put((encoding, actions))
    queue.put(None)


def process(split):
    vid_len = 300
    latent_shape = (vid_len // 4, 16, 16)
    n_devices  = torch.cuda.device_count()
    dataset = NumpyDataset(args.data_path, train=split == 'train')
    n_videos = len(dataset)

    args.split = split
    queue = mp.Queue()
    procs = [mp.Process(target=worker, args=(i, args, split, queue))
             for i in range(n_devices)]
    [p.start() for p in procs]

    idx = 0
    hf_file.create_dataset(f'{split}_data', (n_videos, *latent_shape), dtype=np.uint16)
    hf_file.create_dataset(f'{split}_actions', (n_videos, vid_len), dtype=np.int32)

    done = 0
    pbar = tqdm(total=n_videos)
    while done < n_devices:
        out = queue.get()
        if out is None:
            done += 1
            continue
        video, actions = out
        assert len(video) == len(actions)
        for v, a in zip(video, actions):
            hf_file[f'{split}_data'][idx] = v
            hf_file[f'{split}_actions'][idx] = a
            idx += 1
        pbar.update(len(video))
    pbar.close()
    assert idx == n_videos, f'{idx} != {n_videos}'


if __name__ == '__main__':
    mp.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--ckpt', type=str, required=True)
    parser.add_argument('-d', '--data_path', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('-b', '--batch_size', type=int, default=1)
    args = parser.parse_args()

    hf_file = h5py.File(args.output, 'a')
    process('test')
    process('train')
    hf_file.close()


