from typing import Callable

import timm
import torch
import torch.nn as nn
import torchvision
from einops import rearrange
from torchvision.transforms import Normalize
from typing import Tuple
from transformers import Dinov2WithRegistersModel
from torch import nn
import torch
from math import *


def get_imagenet_norm(inplace=True):
    """
    Construct an ImageNet normalization transform.
    """
    return Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=inplace)


def replace_submodules(
    root_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
    func: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    """
    Recursively replace submodules that satisfy a given predicate.

    Args:
        root_module (nn.Module): The root module to process.
        predicate (Callable[[nn.Module], bool]): A function that takes a module as input and
            returns True if the module should be replaced.
        func (Callable[[nn.Module], nn.Module]): A function that takes a module as input and
            returns a new module to replace it.
        **kwargs: Additional keyword arguments to be passed to the ResNet model constructor.
    """
    if predicate(root_module):
        return func(root_module)

    # Replace all submodules that satisfy the predicate
    module_list = [k.split('.') for k, m in root_module.named_modules(remove_duplicate=True) if predicate(m)]
    for *parent, k in module_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)

    # Verify that all submodules are replaced
    module_list = [k.split('.') for k, m in root_module.named_modules(remove_duplicate=True) if predicate(m)]
    assert len(module_list) == 0
    return root_module


def get_resnet(name, embed_dim, weights=None, replace_batch_norm=True, **kwargs):
    """
    Construct a ResNet model with a custom output embedding dimension and optional batch norm replacement.

    Args:
        name (str): The name of the ResNet architecture to use (e.g., "resnet18", "resnet34", "resnet50").
        embed_dim (int): The dimension of the output embedding.
        weights (Optional[str]): Pre-trained weights to load (e.g., "IMAGENET1K_V1"). If None, no pre-trained weights are used.
        replace_batch_norm (bool, optional): If True, replaces `nn.BatchNorm2d` layers with `nn.GroupNorm` layers. Default is True.
    """
    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)
    resnet.fc = nn.Linear(resnet.fc.in_features, embed_dim)
    if replace_batch_norm:
        resnet = replace_submodules(
            root_module=resnet,
            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
            func=lambda x: nn.GroupNorm(
                num_groups=x.num_features // 16,
                num_channels=x.num_features,
            ),
        )
    return resnet


def get_vit(name, embed_dim, weights=None, **kwargs):
    """
    Construct a Vision Transformer (ViT) model with a custom output embedding dimension.

    Args:
        name (str): The name of the ViT architecture to use (e.g., "vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32", "vit_h_14").
        embed_dim (int): The dimension of the output embedding.
        weights (Optional[str]): Pre-trained weights to load (e.g., "IMAGENET1K_V1"). If None, no pre-trained weights are used.
        **kwargs: Additional keyword arguments to be passed to the ViT model constructor.
    """
    func = getattr(torchvision.models, name)
    vit = func(weights=weights, **kwargs)
    vit.heads = nn.Linear(768, embed_dim)
    return vit


def get_clip(embed_dim, **kwargs):
    """
    Construct a pretrained CLIP encoder with a custom output embedding dimension.

    Args:
        embed_dim (int): The dimension of the output embedding.
        **kwargs: Additional keyword arguments to be passed to the timm model creation function.
    """
    clip = timm.create_model('hf_hub:timm/vit_base_patch32_clip_224.openai', pretrained=True, **kwargs)
    clip.head = nn.Linear(768, embed_dim)
    return clip


def get_dinov3():
    """
    Construct a DINOv3 model with a custom output embedding dimension.

    Args:
        embed_dim (int): The dimension of the output embedding.
        repo_dir (str, optional): Path to the DINOv3 repository directory.
            If None, will try to use environment variable DINO_REPO_DIR or default path.
        weights_path (str, optional): Path to the pretrained weights file.
            If None, will try to use environment variable DINO_WEIGHTS_PATH or default path.
    """
    import os

    # Set default paths
    repo_dir = os.getenv('DINO_REPO_DIR', '/gscratch/weirdlab/hongmm/dinov3')
    weights_path = os.getenv(
        'DINO_WEIGHTS_PATH', '/gscratch/weirdlab/hongmm/dinov3/models/dinov3_vits16_pretrain_lvd1689m-08c60483.pth'
    )
    dinov3 = torch.hub.load(repo_dir, 'dinov3_vits16', source='local', weights=weights_path)

    # repo_dir = os.getenv('DINO_REPO_DIR', '/home/hongmm/dinov3')
    # weights_path = os.getenv(
    #     'DINO_WEIGHTS_PATH', #'/home/hongmm/dinov3/models/dinov3_vits16_pretrain_lvd1689m-08c60483.pth'
    #     '/home/hongmm/dinov3/models/dinov3_vits16_pretrain_lvd1689m-08c60483 (1).pth'
    # )
    # dinov3 = torch.hub.load(repo_dir, 'dinov3_vits16', source='local', weights=weights_path)

    # # Freeze and eval
    # dinov3.eval()
    # for param in dinov3.parameters():
    #     param.requires_grad = False

    return dinov3


class ResNetImageEncoder(nn.Module):
    """
    Multi-view image encoder using a ResNet backbone.

    The input is expected to be a tensor with shape (B, V, C, T, H, W), where:
      - B is the batch size,
      - V is the number of views,
      - C is the number of channels,
      - T is the number of frames,
      - H and W are the image height and width, respectively.
    The encoder reshapes the input to treat each view and frame as an individual image,
    extracts features using the ResNet model, and then concatenates the features across
    all views and frames.

    Args:
        num_views (int): Number of camera views in the input.
        embed_dim (int): Dimension of the output embedding features.
    """

    def __init__(self, num_views: int, embed_dim: int):
        super().__init__()
        self.num_views = num_views
        self.norm = get_imagenet_norm()
        self.model = get_resnet('resnet18', embed_dim, weights='IMAGENET1K_V1')

    def forward(self, imgs: torch.Tensor):
        B, V = imgs.shape[:2]
        imgs = rearrange(imgs, 'b v c t h w -> (b v t) c h w')
        feats = self.model(self.norm(imgs))
        feats = rearrange(feats, '(b v t) c -> b (v t c)', b=B, v=V)
        return feats


class ViTImageEncoder(nn.Module):
    """
    Multi-view image encoder using a Vision Transformer (ViT) backbone.

    Args:
        num_views (int): Number of camera views in the input.
        embed_dim (int): Dimension of the output embedding features.
    """

    def __init__(self, num_views: int, embed_dim: int):
        super().__init__()
        self.num_views = num_views
        self.norm = get_imagenet_norm()
        self.model = get_vit('vit_b_32', embed_dim, weights='IMAGENET1K_V1')

    def forward(self, imgs: torch.Tensor):
        B, V = imgs.shape[:2]
        imgs = rearrange(imgs, 'b v c t h w -> (b v t) c h w')
        imgs = self.norm(imgs)

        # Reshape and permute the input tensor
        x = self.model._process_input(imgs)

        # Expand the class token to the full batch
        batch_cls_token = self.model.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([batch_cls_token, x], dim=1)

        # Get raw tokens
        x = self.model.encoder(x)
        x = self.model.heads(x[:, 0])
        feats = rearrange(x, '(b v t) c -> b (v t c)', b=B, v=V)
        return feats  # (b, v*t, c)


class ViTImagePatchEncoder(nn.Module):
    """
    Multi-view image patch encoder using a Vision Transformer (ViT) backbone with learnable positional embeddings.

    Args:
        num_views (int): Number of camera views in the input.
        num_frames (int): Number of frames per view.
        embed_dim (int): Dimension of the output embedding features.
    """

    def __init__(self, num_views: int, num_frames: int, embed_dim: int):
        super().__init__()
        self.num_views = num_views
        self.norm = get_imagenet_norm()
        self.model = get_vit('vit_b_32', embed_dim, weights='IMAGENET1K_V1')

        # Learnable embeddings
        self.pos_shift = nn.Parameter(
            torch.zeros(1, num_views * num_frames, 1, embed_dim),
            requires_grad=True,
        )
        self.pos_scale = nn.Parameter(
            torch.zeros(1, num_views * num_frames, 1, embed_dim),
            requires_grad=True,
        )

    def forward(self, imgs: torch.Tensor):
        B, V = imgs.shape[:2]
        imgs = rearrange(imgs, 'b v c t h w -> (b v t) c h w')
        imgs = self.norm(imgs)

        # Reshape and permute the input tensor
        x = self.model._process_input(imgs)

        # Expand the class token to the full batch
        batch_cls_token = self.model.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([batch_cls_token, x], dim=1)

        # Get raw tokens
        x = self.model.encoder(x)
        x = self.model.heads(x)

        # Add learned positional embeddings
        feats = rearrange(x, '(b v t) n c -> b (v t) n c', b=B, v=V)
        feats = feats * (1 + self.pos_scale) + self.pos_shift
        return feats.flatten(1, 2)  # (b, v*t*n, c)


class DinoImageEncoder(nn.Module):
    """
    Multi-view image encoder using a DINOv3 backbone.

    The input is expected to be a tensor with shape (B, V, C, T, H, W), where:
      - B is the batch size,
      - V is the number of views,
      - C is the number of channels,
      - T is the number of frames,
      - H and W are the image height and width, respectively.
    The encoder reshapes the input to treat each view and frame as an individual image,
    extracts features using the DINOv3 model, and then concatenates the features across
    all views and frames.

    Args:
        num_views (int): Number of camera views in the input.
        embed_dim (int): Dimension of the output embedding features.
        repo_dir (str): Path to the DINOv3 repository directory.
        weights_path (str): Path to the pretrained weights file.
    """

    def __init__(self, num_views: int, embed_dim: int):
        super().__init__()
        self.num_views = num_views
        self.norm = get_imagenet_norm()

        # --- Load pretrained DINOv3 backbone ---
        self.model = get_dinov3()  # your repo loader
        self.model.eval()
        self.model.requires_grad_(False)

    def forward(self, imgs: torch.Tensor, **kwargs):
        B, V = imgs.shape[:2]

        # Ensure images are in [0, 1] range
        if imgs.max() > 1.0:
            imgs = imgs / 255.0

        imgs = rearrange(imgs, 'b v c t h w -> (b v t) c h w')
        with torch.no_grad():
            feats = self.model(self.norm(imgs))
            feats = rearrange(feats, '(b v t) c -> b (v t c)', b=B, v=V)

        return feats


class Dinov2withNorm(nn.Module):
    def __init__(
        self,
        dinov2_path: str = 'facebook/dinov2-with-registers-base',
        normalize: bool = True,
    ):
        """
        Adapted from: https://github.com/bytetriper/RAE/blob/main/src/stage1/encoders/dinov2.py
        """
        super().__init__()
        # Support both local paths and HuggingFace model IDs
        try:
            self.encoder = Dinov2WithRegistersModel.from_pretrained(dinov2_path, local_files_only=True)
        except (OSError, ValueError, AttributeError):
            self.encoder = Dinov2WithRegistersModel.from_pretrained(dinov2_path, local_files_only=False)
        self.encoder.requires_grad_(False)
        if normalize:
            self.encoder.layernorm.elementwise_affine = False
            self.encoder.layernorm.weight = None
            self.encoder.layernorm.bias = None
        self.patch_size = self.encoder.config.patch_size
        self.hidden_size = self.encoder.config.hidden_size

    def dinov2_forward(self, x: torch.Tensor, output_hidden_states=False) -> torch.Tensor:
        x = self.encoder(x, output_hidden_states=True)
        unused_token_num = 5  # 1 CLS + 4 register tokens
        image_features = x.last_hidden_state[:, unused_token_num:]

        if output_hidden_states:
            return image_features
        else:
            pooled = image_features.mean(dim=1)  # [B, D]
            return pooled

    def forward(self, x: torch.Tensor, output_hidden_states=False) -> torch.Tensor:
        return self.dinov2_forward(x, output_hidden_states=output_hidden_states)


class MultiViewVideoPatchifier(nn.Module):
    def __init__(
        self,
        num_views: int,
        input_shape: Tuple[int, ...] = (8, 224, 224),
        patch_shape: Tuple[int, ...] = (2, 8, 8),
        num_chans: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.num_views = num_views
        iT, iH, iW = input_shape
        pT, pH, pW = patch_shape
        self.T, self.H, self.W = iT // pT, iH // pH, iW // pW
        self.pT, self.pH, self.pW = pT, pH, pW

        self.patch_encoder = nn.Conv3d(
            in_channels=num_chans,
            out_channels=embed_dim,
            kernel_size=patch_shape,
            stride=patch_shape,
        )
        self.patch_decoder = nn.Linear(embed_dim, num_chans * pT * pH * pW)

    def forward(self, imgs):
        return self.patchify(imgs)

    def patchify(self, imgs):
        imgs = rearrange(imgs, 'b v c t h w -> (b v) c t h w')
        feats = self.patch_encoder(imgs)
        feats = rearrange(feats, '(b v) c t h w -> b (v t h w) c', v=self.num_views)
        return feats

    def unpatchify(self, feats):
        imgs = self.patch_decoder(feats)
        imgs = rearrange(
            imgs,
            'b (v t h w) (c pt ph pw) -> b v c (t pt) (h ph) (w pw)',
            v=self.num_views,
            t=self.T,
            h=self.H,
            w=self.W,
            pt=self.pT,
            ph=self.pH,
            pw=self.pW,
        )
        return imgs

    @property
    def num_patches(self):
        return self.num_views * self.T * self.H * self.W
