from typing import Optional

import torch
import torch.nn as nn
import torchvision.transforms.functional as ttf
from diffusers import AutoencoderKL
from einops import rearrange
from torchvision.transforms import (
    CenterCrop,
    ColorJitter,
    RandomCrop,
    Resize,
    Normalize,
)


class ToTensor(nn.Module):
    """
    Convert a batch of images from (B, H, W, C) to (B, C, H, W)
    and normalize the pixel values to the range [0, 1].
    """

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs.permute((0, 3, 1, 2)).contiguous().float().div_(255.0)


class AutoRandomCrop(nn.Module):
    """
    Perform random cropping during training and center cropping during eval.
    """

    def __init__(self, size: tuple[int, int]) -> None:
        super().__init__()
        self.size = size
        self.random_crop = RandomCrop(size=size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self.random_crop(inputs)
        else:
            # Take center crop during eval
            return ttf.center_crop(img=inputs, output_size=self.size)


class AutoColorJitter(nn.Module):
    """
    Perform color jittering during training and no-op during eval.
    """

    def __init__(
        self,
        brightness: float,
        contrast: float,
        saturation: float,
        hue: tuple[float],
    ) -> None:
        super().__init__()
        self.color_jitter = ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=tuple(hue),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self.color_jitter(inputs)
        else:
            return inputs  # no-op during eval


class VAEDownsample(nn.Module):
    """
    Downsample images using a pre-trained VAE.
    """

    def __init__(self) -> None:
        super().__init__()
        # Input normalization
        self.norm = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        self.inv_norm = Normalize(mean=[-1, -1, -1], std=[2, 2, 2], inplace=True)

        # Normalization stats (computed from multitask 12)
        shift = torch.tensor([0.0, 0.0, 0.0, 0.0]).view(1, 4, 1, 1)
        scale = torch.tensor([3.0, 3.0, 3.0, 3.0]).view(1, 4, 1, 1)
        self.register_buffer('shift', shift)
        self.register_buffer('scale', scale)

        # Load pre-trained VAE
        self.vae = AutoencoderKL.from_pretrained(
            'stabilityai/sdxl-vae'
            # 'stabilityai/sd-vae-ft-mse'
            # 'stabilityai/stable-diffusion-2-1'
        )  # https://arxiv.org/pdf/2307.08698 uses "stabilityai/sd-vae-ft-mse" for flow matching
        for p in self.vae.parameters():
            p.requires_grad = False
        self.scaling_factor = self.vae.config.scaling_factor

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = self.norm(images)
        feats = self.vae.encode(images).latent_dist.sample()
        feats = feats.mul_(self.scaling_factor)
        feats = feats.sub_(self.shift).div_(self.scale)
        return feats

    def inverse(self, feats: torch.Tensor) -> torch.Tensor:
        feats = feats.mul_(self.scale).add_(self.shift)
        feats = feats.div_(self.scaling_factor)
        images = self.vae.decode(feats).sample
        images = self.inv_norm(images)
        return images


class ImageTransform(nn.Module):
    """
    Apply a sequence of transforms to images.
    """

    def __init__(
        self,
        resize_shape: Optional[tuple[int, int]] = None,
        crop_shape: Optional[tuple[int, int]] = None,
        random_crop: bool = True,
        color_jitter: Optional[dict] = None,
        downsample: bool = False,
        imagenet_norm: bool = True,
    ) -> None:
        super().__init__()
        transform = list()

        # Convert image to tensor format
        transform.append(ToTensor())

        # Resize images
        if resize_shape is not None:
            transform.append(Resize(resize_shape))

        # Apply random crop during training and center crop during eval
        if crop_shape is not None:
            if random_crop:
                transform.append(AutoRandomCrop(crop_shape))
            else:
                transform.append(CenterCrop(crop_shape))

        # Apply color jitter during training
        if color_jitter is not None:
            transform.append(
                AutoColorJitter(
                    brightness=color_jitter['brightness'],
                    contrast=color_jitter['contrast'],
                    saturation=color_jitter['saturation'],
                    hue=tuple(color_jitter['hue']),
                )
            )

        # Normalize using imagenet statistics
        if downsample:
            if imagenet_norm:
                print('Disabling imagenet normalization since downsample is enabled.')
            transform.append(VAEDownsample())
        elif imagenet_norm:
            transform.append(Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True))

        self.transform = nn.Sequential(*transform)

    @property
    def vae(self) -> VAEDownsample:
        assert isinstance(self.transform[-1], VAEDownsample)
        return self.transform[-1]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.transform(images)


class VideoTransform(ImageTransform):
    """
    Flatten videos to images, apply transforms, and reshape back to videos.
    """

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        num_frames = images.shape[1]
        images = rearrange(images, 'b t h w c-> (b t) h w c')
        images = self.transform(images)
        images = rearrange(images, '(b t) c h w-> b c t h w', t=num_frames)
        return images
