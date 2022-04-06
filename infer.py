import os
from argparse import ArgumentParser
import torch
import torch.utils.data
from generative_lightning.models.generators import UNETGenerator, WideResnetEncoderDecoder, WideResnetUNET, CustomUNET
from generative_lightning.models.cycle_gan import CycleGAN
from generative_lightning.data.dataloader import MonetDataset
import numpy as np
import PIL
import tqdm


def main(args):
    os.makedirs(args.output_path, exist_ok=True)
    dataroot = args.data_path
    workers = 12
    dataset = MonetDataset(dataroot=dataroot, shuffle=False)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=workers,
        batch_size=1,
    )
    generators = {
        "UNET": UNETGenerator,
        "Resnet": WideResnetEncoderDecoder,
        "WideResnetUNET": WideResnetUNET,
        "CustomUNET": CustomUNET,
    }
    model = CycleGAN(generator=generators[args.gen])
    state_dict = torch.load(args.ckpt_path)["state_dict"]
    model.load_state_dict(state_dict)
    model = model.eval()
    model.cuda()
    with torch.no_grad():
        i = 1
        for _, photo in tqdm.tqdm(dataloader):
            photo = photo.cuda()
            prediction = model.m_gen(photo)
            prediction = (prediction * 127.5 + 127.5)
            prediction = prediction.squeeze().permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            im = PIL.Image.fromarray(prediction)
            im.save("{}/{}.jpg".format(args.output_path, str(i)))
            i += 1
    os.system("cd {}; zip output.zip *.jpg".format(args.output_path))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data-path", metavar="FILE", default=None, required=True)
    parser.add_argument("--gen", required=True, type=str)
    parser.add_argument("--ckpt-path", required=True, type=str)
    parser.add_argument("--output-path", required=True, type=str)
    args = parser.parse_args()
    main(args)