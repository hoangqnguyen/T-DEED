# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import sys

import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional
import torch.utils.checkpoint as checkpoint

from einops import rearrange
from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

from timm.models.layers import DropPath, to_2tuple
from timm.models.vision_transformer import _load_weights

import torchvision.transforms as T

import math

def init_weights_conv2d(m):
    if isinstance(m, nn.Conv2d):
        # Kaiming He initialization (also known as He normal initialization)
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, kernel_size=1, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.tubelet_size = kernel_size

        self.proj = nn.Conv3d(
            in_chans, embed_dim, 
            kernel_size=(kernel_size, patch_size[0], patch_size[1]),
            stride=(kernel_size, patch_size[0], patch_size[1])
        )

    def forward(self, x):
        x = self.proj(x)
        return x
    

class TransformerSpotCore(nn.Module):
    def __init__(
            self, args = None,
            img_size=224, 
            patch_size=16, 
            embed_dim=192, 
            channels=3, 
            num_classes=1000,
            # tf settings
            num_dec=4, 
            nhead=8, dim_ffd=512,
            drop_rate=0.,
            drop_path_rate=0.1,
            # video
            kernel_size=1, 
            num_frames=8, 
            fc_drop_rate=0., 
            device=None,
            dtype=None,
            # checkpoint
            use_checkpoint=False,
            checkpoint_num=0,
        ):
        factory_kwargs = {"device": device, "dtype": dtype} # follow MambaLMHeadModel
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.checkpoint_num = checkpoint_num
        print(f'Use checkpoint: {use_checkpoint}')
        print(f'Checkpoint number: {checkpoint_num}')
        
        self._radi_displacement = args.radi_displacement
        # pretrain parameters
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, 
            kernel_size=kernel_size,
            in_chans=channels, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        # self.cls_token = nn.Parameter(torch.zeros(1, num_frames, self.embed_dim)) # TODO: test cls token for each frame

        # self.cls_token = nn.Parameter(torch.sin(torch.arange(num_frames).unsqueeze(1) * torch.arange(self.embed_dim).unsqueeze(0) / 10000.0).unsqueeze(0)) # TODO: Test init as Sinusoidal
        self.cls_query = nn.Parameter(torch.randn(1, num_frames, self.embed_dim)) 

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
        self.temporal_pos_embedding = nn.Parameter(torch.zeros(1, num_frames // kernel_size, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.head_drop = nn.Dropout(fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()

        # prediction heads
        self.head_cls = nn.Linear(self.num_features, num_classes+1)
        if self._radi_displacement > 0:
            self.head_displ = nn.Linear(self.num_features, 1)
            self.head_displ.apply(segm_init_weights)

        if args.predict_location:
            self.head_location = nn.Conv2d(in_channels=embed_dim, out_channels=3, kernel_size=1, stride=1, padding='same') # ('has_event_in_that_patch, x , y) = 3 = out_channels
            self.head_location.apply(init_weights_conv2d)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_dec)]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        # mamba blocks
        self.temp_arch = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=embed_dim, nhead=nhead, dim_feedforward=dim_ffd, dropout=drop_rate, batch_first=True),
                num_layers=num_dec
        )
        
        # output head
        self.norm_f = nn.LayerNorm

        # original init
        self.apply(segm_init_weights)
        self.head_cls.apply(segm_init_weights)

        trunc_normal_(self.pos_embed, std=.02)

        
        #Augmentations and crop
        self.augmentation = T.Compose([
            T.RandomApply([T.ColorJitter(hue = 0.2)], p = 0.25),
            T.RandomApply([T.ColorJitter(saturation = (0.7, 1.2))], p = 0.25),
            T.RandomApply([T.ColorJitter(brightness = (0.7, 1.2))], p = 0.25),
            T.RandomApply([T.ColorJitter(contrast = (0.7, 1.2))], p = 0.25),
            T.RandomApply([T.GaussianBlur(5)], p = 0.25),
            torch.nn.Identity() if args.predict_location else T.RandomHorizontalFlip(),
        ])

        #Standarization
        self.standarization = T.Compose([
            T.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)) #Imagenet mean and std
        ])

        #Augmentation at test time
        self.augmentationI = T.Compose([
            torch.nn.Identity() if args.predict_location else T.RandomHorizontalFlip(p = 1.0)
        ])

        #Croping in case of using it
        self.croping = args.crop_dim
        if self.croping != None and args.predict_location == False:
            self.cropT = T.RandomCrop((self.croping, self.croping))
            self.cropI = T.CenterCrop((self.croping, self.croping))
        else:
            self.cropT = torch.nn.Identity()
            self.cropI = torch.nn.Identity()
        
        self.img_size = img_size
        self.resize = T.Resize((img_size, img_size),  antialias=True )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "temporal_pos_embedding"}
    
    def get_num_layers(self):
        return len(self.layers)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward_features(self, x, inference=False, inference_params=None, augment_inference=False):
        # original input should be: B x T x C x H x W
        # print("0. x shape: ", x.shape)
        B, T, C, _, _ = x.shape

        x = self.normalize(x)
        x = self.resize(x.flatten(0, 1)).view(B, T, C, self.img_size, self.img_size)


        if not inference:
            x = self.augment(x) #augmentation per-batch
            x = self.standarize(x) #standarization imagenet stats


        else:
            if augment_inference:
                x = self.augmentI(x)
            x = self.standarize(x)

        x = x.transpose(1, 2)  # B x C x T x H x W
        x = self.patch_embed(x)
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C)
        loc_feat = None
        if hasattr(self, 'head_location'):
            loc_feat = self.head_location(x.view(B * T, C, H, W)).view(B, T, H, W, 3) # B*T*P^2, 3

        # cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # B*T, 1, C        
        
        # cls_token = self.cls_token.expand(B, -1, -1)  # B*T, 1, C
        # print(f"{cls_token.shape=}")  
        # breakpoint()    
        # cls_token = cls_token.reshape(B * T, 1, -1)

        # x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed

        # temporal pos
        # cls_tokens = x[:B, :1, :]  # B, 1, C
        # cls_tokens = x[:, :1, :].reshape(B, T, -1) # B*T, 1, C
        # x = x[:, 1:]
        x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)
        x = x + self.temporal_pos_embedding
        x = rearrange(x, '(b n) t m -> b (t n) m', b=B, t=T)
        # x = torch.cat((cls_tokens, x), dim=1)

        x = self.pos_drop(x)

        cls_query = self.cls_query.expand(B, -1, -1)
        
        # breakpoint()
        hidden_states = self.temp_arch(cls_query, x)


        # return only cls token
        # return hidden_states[:, :T, :], loc_feat
        return hidden_states, loc_feat

    def forward(self, x, y=None, inference_params=None, inference=False, augment_inference=False):
        x, loc_feat = self.forward_features(x, inference_params=inference_params, inference=inference, augment_inference=augment_inference)
        x = self.head_drop(x)
        im_feat = self.head_cls(x)
        displ_feat = None   
        if self._radi_displacement > 0:
            displ_feat = self.head_displ(x).squeeze(-1)
        return {'im_feat': im_feat, 'displ_feat': displ_feat , 'loc_feat': loc_feat}, y

    def normalize(self, x):
        return x / 255.
    
    def augment(self, x):
        for i in range(x.shape[0]):
            x[i] = self.augmentation(x[i])
        return x
    
    def augmentI(self, x):
        for i in range(x.shape[0]):
            x[i] = self.augmentationI(x[i])
        return x

    def standarize(self, x):
        for i in range(x.shape[0]):
            x[i] = self.standarization(x[i])
        return x

    def print_stats(self):
        print('Model params:',
            sum(p.numel() for p in self.parameters()))

def inflate_weight(weight_2d, time_dim, center=True):
    print(f'Init center: {center}')
    if center:
        weight_3d = torch.zeros(*weight_2d.shape)
        weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        middle_idx = time_dim // 2
        weight_3d[:, :, middle_idx, :, :] = weight_2d
    else:
        weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        weight_3d = weight_3d / time_dim
    return weight_3d


def load_state_dict(model, state_dict, center=True):
    state_dict_3d = model.state_dict()
    for k in state_dict.keys():
        if k in state_dict_3d.keys() and state_dict[k].shape != state_dict_3d[k].shape:
            if len(state_dict_3d[k].shape) <= 3:
                print(f'Ignore: {k}')
                continue
            print(f'Inflate: {k}, {state_dict[k].shape} => {state_dict_3d[k].shape}')
            time_dim = state_dict_3d[k].shape[2]
            state_dict[k] = inflate_weight(state_dict[k], time_dim, center=center)
    
    del state_dict['head.weight']
    del state_dict['head.bias']
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)



@register_model
def mambaspotcore_nano(args, pretrained=False, **kwargs):
    model = TransformerSpotCore(
        args=args,
        patch_size=16, 
        embed_dim=192, 
        num_dec=3, 
        dim_ffd=512,
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        print('load pretrained weights')
        state_dict = torch.load(_MODELS["videomamba_t16_in1k"], map_location='cpu')
        load_state_dict(model, state_dict, center=True)
    return model

@register_model
def mambaspotcore_tiny(args, pretrained=False, **kwargs):
    model = TransformerSpotCore(
        args=args,
        patch_size=16, 
        embed_dim=192, 
        num_dec=6, 
        dim_ffd=512,
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        print('load pretrained weights')
        state_dict = torch.load(_MODELS["videomamba_t16_in1k"], map_location='cpu')
        load_state_dict(model, state_dict, center=True)
    return model


@register_model
def mambaspotcore_small(args, pretrained=False, **kwargs):
    model = TransformerSpotCore(
        args=args,
        patch_size=16, 
        embed_dim=384, 
        num_dec=9, 
        dim_ffd=512,
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        print('load pretrained weights')
        state_dict = torch.load(_MODELS["videomamba_s16_in1k"], map_location='cpu')
        load_state_dict(model, state_dict, center=True)
    return model


@register_model
def mambaspotcore_middle(args, pretrained=False, **kwargs):
    model = TransformerSpotCore(
        args=args,
        patch_size=16, 
        embed_dim=576, 
        num_dec=12, 
        dim_ffd=512,
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        print('load pretrained weights')
        state_dict = torch.load(_MODELS["videomamba_m16_in1k"], map_location='cpu')
        load_state_dict(model, state_dict, center=True)
    return model

def isnan(x):
    return x.isnan().any()

if __name__ == '__main__':
    import time
    from torchinfo import summary
    import numpy as np
    from easydict import EasyDict as edict
    args = edict()

    args.radi_displacement = 1
    args.predict_location = True
    args.crop_dim = None

    seed = 4217
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    num_frames = 30
    img_size = 224

    # To evaluate GFLOPs, pleaset set `rms_norm=False` and `fused_add_norm=False`
    model = mambaspotcore_middle(args=args, num_frames=num_frames, num_classes=6).cuda()
    batch = torch.rand(4, num_frames, 3, img_size, img_size).cuda()
    output, y = model(batch)
    for k, v in output.items():
        print(f'{k}: {v.shape}')
    print("Total params:", sum(p.numel() for p in model.parameters()))
    # flops = FlopCountAnalysis(model, torch.rand(1, 3, num_frames, img_size, img_size).cuda())
    # summary(model, input_data=batch)
    s = time.time()
    # print(flop_count_table(flops, max_depth=1))
    # print(time.time()-s)
