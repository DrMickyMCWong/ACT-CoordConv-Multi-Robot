# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from detr.util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding

import IPython
e = IPython.embed

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other policy_models than torchvision.policy_models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool,
                 unfreeze_layers: List[str] = None):
        super().__init__()
        
        # Optional selective layer freezing/unfreezing
        # For RGBD finetuning, it's recommended to unfreeze conv1 and layer1
        if unfreeze_layers is not None:
            print(f"[BackboneBase] Applying selective layer unfreezing: {unfreeze_layers}")
            for name, parameter in backbone.named_parameters():
                # First freeze all
                parameter.requires_grad_(False)
                # Then unfreeze specified layers
                for layer_name in unfreeze_layers:
                    if layer_name in name:
                        parameter.requires_grad_(True)
                        break
            
            # Print unfrozen parameter count
            trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in backbone.parameters())
            print(f"[BackboneBase] Trainable: {trainable_params:,} / {total_params:,} params "
                  f"({100*trainable_params/total_params:.1f}%)")
        # Original commented-out code for reference:
        # for name, parameter in backbone.named_parameters(): # only train later layers # TODO do we want this?
        #     if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #         parameter.requires_grad_(False)
        
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor):
        xs = self.body(tensor)
        return xs
        # out: Dict[str, NestedTensor] = {}
        # for name, x in xs.items():
        #     m = tensor_list.mask
        #     assert m is not None
        #     mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        #     out[name] = NestedTensor(x, mask)
        # return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool,
                 in_channels: int = 3,
                 unfreeze_early_layers: bool = False):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d) # pretrained # TODO do we want frozen batch_norm??
        
        # RGBD Support: Modify first conv layer for 4-channel input
        if in_channels != 3:
            print(f"[Backbone] Modifying conv1 for {in_channels}-channel input (RGBD mode)")
            # Save original conv1 weights
            original_conv1 = backbone.conv1
            
            # Create new conv1 with desired input channels
            backbone.conv1 = nn.Conv2d(
                in_channels, 64,
                kernel_size=7, stride=2, padding=3, bias=False
            )
            
            # Initialize new conv1 weights
            with torch.no_grad():
                if in_channels == 4:  # RGBD case
                    # Copy RGB weights from pretrained model
                    backbone.conv1.weight[:, :3, :, :] = original_conv1.weight
                    # Initialize depth channel as average of RGB channels
                    # (standard practice for transfer learning with extra channel)
                    backbone.conv1.weight[:, 3:4, :, :] = original_conv1.weight.mean(dim=1, keepdim=True)
                    print("[Backbone] ✓ Copied RGB weights, initialized depth channel as mean(RGB)")
                else:
                    # For other channel counts, initialize randomly
                    nn.init.kaiming_normal_(backbone.conv1.weight, mode='fan_out', nonlinearity='relu')
                    print(f"[Backbone] ✓ Initialized {in_channels}-channel conv1 with kaiming_normal")
            
            # IMPORTANT: Ensure conv1 is trainable for finetuning the depth channel
            for param in backbone.conv1.parameters():
                param.requires_grad = True
            print("[Backbone] ✓ conv1 set to trainable (requires_grad=True)")
        
        # Determine which layers to unfreeze for RGBD finetuning
        unfreeze_layers = None
        if in_channels == 4 and unfreeze_early_layers:
            # For RGBD: unfreeze conv1 and layer1 for better depth feature learning
            unfreeze_layers = ['conv1', 'bn1', 'layer1']
            print("[Backbone] RGBD mode: Will unfreeze conv1, bn1, and layer1 for finetuning")
        
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers, unfreeze_layers=unfreeze_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    
    # Get in_channels from args (default to 3 for backward compatibility with RGB-only models)
    in_channels = getattr(args, 'in_channels', 3)
    
    # Get unfreeze_early_layers flag (recommended for RGBD finetuning)
    unfreeze_early_layers = getattr(args, 'unfreeze_early_layers', False)
    
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation, 
                       in_channels=in_channels, unfreeze_early_layers=unfreeze_early_layers)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
