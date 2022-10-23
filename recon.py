import argparse
from tats import VQGAN, VideoData


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--ckpt', type=str, required=True)
    parser = VideoData.add_data_specific_args(parser)
    args = parser.parse_args()

    data = VideoData(args, shuffle=True)
    test_loader = data.test_dataloader()
    batch = next(iter(test_loader))

    model = VQGAN.load_from_checkpoint(args.ckpt)


if __name__ == '__main__':
    main()


