import math
import numbers
import os
import random
import time
from collections import deque

import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as VTF
from torchvision.utils import make_grid, save_image
from PIL import Image

from imagen_pytorch import ImagenTrainer, ElucidatedImagenConfig
from imagen_pytorch import load_imagen_from_checkpoint
from gan_utils import get_images, get_vocab
from data_generator import ImageLabelDataset  


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default=None, help="image source")
    parser.add_argument('--tags_source', type=str, default=None, help="tag files. will use --source if not specified.")
    parser.add_argument('--poses', type=str, default=None)
    parser.add_argument('--tags', type=str, default=None)
    parser.add_argument('--vocab', default=None)
    parser.add_argument('--size', default=256, type=int)
    parser.add_argument('--sample_steps', default=32, type=int)
    parser.add_argument('--num_unets', default=1, type=int, help="additional unet networks")
    parser.add_argument('--vocab_limit', default=None, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--imagen', default="imagen.pth")
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--replace', action='store_true', help="replace the output file")
    parser.add_argument('--unet_dims', default=128, type=int)
    parser.add_argument('--unet2_dims', default=64, type=int)
    parser.add_argument("--start_size", default=64, type=int)

    # training
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--micro_batch_size', default=8, type=int)
    parser.add_argument('--samples_out', default="samples")
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--shuffle_tags', action='store_true')
    parser.add_argument('--train_unet', type=int, default=1)
    parser.add_argument('--random_drop_tags', type=float, default=0.)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--no_text_transform', action='store_true')
    parser.add_argument('--aug', action='store_true', help="additional image augmentations")

    args = parser.parse_args()

    if args.sample_steps is None:
        args.sample_steps = args.size

    if args.tags_source is None:
        args.tags_source = args.source

    if args.vocab is None:
        args.vocab = args.source
    else:
        assert os.path.isfile(args.vocab)

    sample_from_trained(args)

def restore_parts(state_dict_target, state_dict_from):
    for name, param in state_dict_from.items():
        if name not in state_dict_target:
            continue
        # if isinstance(param, Parameter):
        #    param = param.data
        if param.size() == state_dict_target[name].size():
            state_dict_target[name].copy_(param)
        else:
            print(f"layer {name}({param.size()} different than target: {state_dict_target[name].size()}")

    return state_dict_target

def save(imagen, path):
    out = {}
    unets = []
    for unet in imagen.unets:
        unets.append(unet.cpu().state_dict())
    out["unets"] = unets

    out["imagen"] = imagen.cpu().state_dict()

    torch.save(out, path)


def load(path):

    imagen = load_imagen_from_checkpoint(path)

    return imagen


def get_imagen(args, unet_dims=None, unet2_dims=None):

    if unet_dims is None:
        unet_dims = args.unet_dims

    if unet2_dims is None:
        unet2_dims = args.unet2_dims

    if args.poses is not None:
        cond_images_channels = 3
    else:
        cond_images_channels = 0

    # unet for imagen
    unet1 = dict(
        dim=unet_dims,
        cond_dim=512,
        dim_mults=(1, 2, 3, 4),
        cond_images_channels=cond_images_channels,
        num_resnet_blocks=3,
        layer_attns=(False, True, True, True),
        memory_efficient=False
    )

    unets = [unet1]

    for i in range(args.num_unets):

        unet2 = dict(
            dim=unet2_dims // (i + 1),
            cond_dim=512,
            dim_mults=(1, 2, 3, 6),
            cond_images_channels=cond_images_channels,
            num_resnet_blocks=(2, 4, 8, 8),
            layer_attns=(False, False, False, True),
            layer_cross_attns=(False, False, True, True),
            final_conv_kernel_size=1,
            memory_efficient=True
        )

        unets.append(unet2)

    image_sizes = [args.start_size]

    for i in range(0, len(unets)-1):
        image_sizes.append(image_sizes[-1] * 4)

    print(f"image_sizes={image_sizes}")

    sample_steps = [args.sample_steps] * (args.num_unets + 1)

    imagen = ElucidatedImagenConfig(
        unets=unets,
        text_encoder_name='t5-large',
        num_sample_steps=sample_steps,
        # noise_schedules=["cosine", "cosine"],
        # pred_objectives=["noise", "x_start"],
        image_sizes=image_sizes,
        per_sample_random_aug_noise_level=True,
        lowres_sample_noise_level=0.3
    ).create().cuda()

    return imagen


def make_training_samples(poses, trainer, args):
    sample_texts = ['a big brown bear']

    disp_size = min(args.batch_size, 4)
    sample_poses = None

    if poses is not None:
        sample_poses = poses[:disp_size]


    if poses is not None and sample_poses is None:
        sample_poses = poses[:disp_size]

    sample_images = trainer.sample(texts=sample_texts,
                                   cond_images=sample_poses,
                                   cond_scale=7.,
                                   return_all_unet_outputs=True,
                                   stop_at_unet_number=args.train_unet)

    final_samples = None

    if len(sample_images) > 1:
        for si in sample_images:
            sample_images1 = transforms.Resize(args.size)(si)
            if final_samples is None:
                final_samples = sample_images1
                continue

            sample_images1 = transforms.Resize(args.size)(si)
            final_samples = torch.cat([final_samples, sample_images1])
        
        sample_images = final_samples
    else:
        sample_images = sample_images[0]
        sample_images = transforms.Resize(args.size)(sample_images)

    if poses is not None:
        sample_poses0 = transforms.Resize(args.size)(sample_poses)
        sample_images = torch.cat([sample_images.cpu(), sample_poses0.cpu()])

    grid = make_grid(sample_images, nrow=disp_size, normalize=False, range=(-1, 1))
    VTF.to_pil_image(grid).save(os.path.join(args.samples_out, f"imagen_sample.png"))


def sample_from_trained(args):

    imagen = get_imagen(args)

    trainer = ImagenTrainer(imagen, fp16=args.fp16)
    
    if args.imagen is not None and os.path.isfile(args.imagen):
        print(f"Loading model: {args.imagen}")
        trainer.load(args.imagen)
    
    os.makedirs(args.samples_out, exist_ok=True)

    make_training_samples(None, trainer, args)


if __name__ == "__main__":
    main()
