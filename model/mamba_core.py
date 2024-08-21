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

from mamba_ssm.modules.mamba_simple import Mamba

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


MODEL_PATH = 'your_model_path'
_MODELS = {
    "videomamba_t16_in1k": os.path.join(MODEL_PATH, "videomamba_t16_in1k_res224.pth"),
    "videomamba_s16_in1k": os.path.join(MODEL_PATH, "videomamba_s16_in1k_res224.pth"),
    "videomamba_m16_in1k": os.path.join(MODEL_PATH, "videomamba_m16_in1k_res224.pth"),
}


class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False,drop_path=0.,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        # print(f'fused_add_norm: {fused_add_norm}')
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None,
        use_checkpoint=False
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        # clamping for numerical stability
        if residual is not None:
            residual = torch.clamp(residual, min=-1e6, max=1e6)
            # residual = torch.nan_to_num(residual, nan=0.0, posinf=1e6, neginf=-1e6)
        hidden_states = torch.clamp(hidden_states, min=-1e6, max=1e6)
        # hidden_states = torch.nan_to_num(hidden_states, nan=0.0, posinf=1e6, neginf=-1e6)

        if isnan(hidden_states):
            print(f'Block: input hidden_states is NaN')
            breakpoint()

            
        if residual is not None and isnan(residual):    
            print(f'Block: input residual is NaN')
            breakpoint()

        if not self.fused_add_norm:
            residual = (residual + self.drop_path(hidden_states)) if residual is not None else hidden_states

            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype)) 
            # residual = residual 
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
            
            if residual is not None:
                residual = torch.clamp(residual, min=-1e6, max=1e6)
            hidden_states = torch.nan_to_num(hidden_states, nan=0.0, posinf=1e6, neginf=-1e6)

            if isnan(hidden_states):
                print(f'not self.fused_add_norm:  hidden_states is NaN')
                breakpoint()

                
            if residual is not None and isnan(residual):
                print(f'not self.fused_add_norm:  residual is NaN')
                breakpoint()
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            # fused_add_norm_fn = layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states if residual is None else self.drop_path(hidden_states),
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )

            # hidden_states = hidden_states 
            # residual = residual 
            if residual is not None:
                residual = torch.clamp(residual, min=-1e6, max=1e6)
            hidden_states = torch.nan_to_num(hidden_states, nan=0.0, posinf=1e6, neginf=-1e6)

            if isnan(hidden_states):
                print(f'yes self.fused_add_norm:  hidden_states is NaN')
                breakpoint()

                
            if residual is not None and isnan(residual):
                print(f'yes self.fused_add_norm:  residual is NaN')
                breakpoint()
        if use_checkpoint:
            hidden_states = checkpoint.checkpoint(self.mixer, hidden_states, inference_params)
        else:
            
            if isnan(hidden_states):
                print(f'before mixer:  hidden_states is NaN')
                breakpoint()

            if residual is not None:
                residual = torch.clamp(residual, min=-1e6, max=1e6)
            hidden_states = self.mixer(hidden_states, inference_params=inference_params)
            # hidden_states = torch.clamp(hidden_states, min=-1e6, max=1e6)
            hidden_states = torch.nan_to_num(hidden_states, nan=0.0, posinf=1e6, neginf=-1e6)

            if isnan(hidden_states):
                print(f'after mixer:  hidden_states is NaN')
                breakpoint()

        if residual is not None:
            residual = torch.clamp(residual, min=-1e6, max=1e6)
        hidden_states = torch.nan_to_num(hidden_states, nan=0.0, posinf=1e6, neginf=-1e6)

        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.,
    rms_norm=False,
    residual_in_fp32=True,
    fused_add_norm=True,
    layer_idx=None,
    bimamba=True,
    device=None,
    dtype=None,
):
    factory_kwargs = {"device": device, "dtype": dtype}
    if ssm_cfg is None:
        ssm_cfg = {}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, bimamba=bimamba, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon)
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


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
    

class MambaSpotCore(nn.Module):
    def __init__(
            self, args = None,
            img_size=224, 
            patch_size=16, 
            depth=24, 
            embed_dim=192, 
            channels=3, 
            num_classes=1000,
            drop_rate=0.,
            drop_path_rate=0.1,
            ssm_cfg=None, 
            norm_epsilon=1e-5, 
            initializer_cfg=None,
            fused_add_norm=True,
            rms_norm=False, 
            residual_in_fp32=True,
            bimamba=True,
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
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
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
        self.cls_token = nn.Parameter(torch.zeros(1, num_frames, self.embed_dim)) # TODO: test cls token for each frame

        # self.cls_token = nn.Parameter(torch.sin(torch.arange(num_frames).unsqueeze(1) * torch.arange(self.embed_dim).unsqueeze(0) / 10000.0).unsqueeze(0)) # TODO: Test init as Sinusoidal


        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
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

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        # mamba blocks
        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    bimamba=bimamba,
                    drop_path=inter_dpr[i],
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )
        
        # output head
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(embed_dim, eps=norm_epsilon, **factory_kwargs)

        # original init
        self.apply(segm_init_weights)
        self.head_cls.apply(segm_init_weights)

        trunc_normal_(self.pos_embed, std=.02)

        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        
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
        
        cls_token = self.cls_token.expand(B, -1, -1)  # B*T, 1, C
        # print(f"{cls_token.shape=}")  
        # breakpoint()    
        cls_token = cls_token.reshape(B * T, 1, -1)

        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed

        # temporal pos
        # cls_tokens = x[:B, :1, :]  # B, 1, C
        cls_tokens = x[:, :1, :].reshape(B, T, -1) # B*T, 1, C
        x = x[:, 1:]
        x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)
        x = x + self.temporal_pos_embedding
        x = rearrange(x, '(b n) t m -> b (t n) m', b=B, t=T)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.pos_drop(x)

        # mamba impl
        residual = None
        hidden_states = x

        for idx, layer in enumerate(self.layers):
            if self.use_checkpoint and idx < self.checkpoint_num:
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params,
                    use_checkpoint=True
                )
            else:
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params
                )
                if isnan(hidden_states):
                    print(f'MAMBA: Layer {idx} hidden_states is NaN')
                    breakpoint()
                
                if isnan(residual):
                    print(f'MAMBA: Layer {idx} residual is NaN')
                    breakpoint()

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))

            if isnan(hidden_states):
                print(f'Not FANorm: Layer {idx} hidden_states is NaN')
                breakpoint()
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
            if isnan(hidden_states):
                print(f'Yes FANorm: Layer {idx} hidden_states is NaN')
                breakpoint()

        # return only cls token
        return hidden_states[:, :T, :], loc_feat

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
    model = MambaSpotCore(
        args=args,
        patch_size=16, 
        embed_dim=192, 
        depth=12, 
        rms_norm=True, 
        residual_in_fp32=True, 
        # fused_add_norm=True, 
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
    model = MambaSpotCore(
        args=args,
        patch_size=16, 
        embed_dim=192, 
        depth=24, 
        rms_norm=True, 
        residual_in_fp32=True, 
        # fused_add_norm=kwargs['fused_add_norm'], 
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
    model = MambaSpotCore(
        args=args,
        patch_size=16, 
        embed_dim=384, 
        depth=24, 
        rms_norm=True, 
        residual_in_fp32=True, 
        # fused_add_norm=True, 
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        print('load pretrained weights')
        state_dict = torch.load(_MODELS["videomamba_s16_in1k"], map_location='cpu')
        load_state_dict(model, state_dict, center=True)
    return model


@register_model
def mambaspotcore_middle(pretrained=False, **kwargs):
    model = MambaSpotCore(
        args=args,
        patch_size=16, 
        embed_dim=576, 
        depth=32, 
        rms_norm=True, 
        residual_in_fp32=True, 
        fused_add_norm=True, 
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
    model = mambaspotcore_small(args, num_frames=num_frames, num_classes=6).cuda()
    batch = torch.rand(4, num_frames, 3, img_size, img_size).cuda()
    output, y = model(batch)
    for k, v in output.items():
        print(f'{k}: {v.shape}')
    # flops = FlopCountAnalysis(model, torch.rand(1, 3, num_frames, img_size, img_size).cuda())
    # summary(model, input_data=batch)
    s = time.time()
    # print(flop_count_table(flops, max_depth=1))
    # print(time.time()-s)
