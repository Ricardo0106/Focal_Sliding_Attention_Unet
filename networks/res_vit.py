import torch.nn as nn
import torch
from utils import SwinTransformerBlock, PatchEmbed, PatchMerging, FinalPatchExpand_X4
from einops import rearrange
from torch import tensor
import numpy as np

class ResVit(nn.Module):
    def __init__(self):
        super().__init__()
        # in_chans is 3 RGB
        self.embed_dim = 96
        self.patch_size = 4
        self.window_size = 7
        self.num_classes = 3
        self.encoder = PatchEmbed(img_size=224, patch_size=self.patch_size, in_chans=3, embed_dim=self.embed_dim, norm_layer=nn.LayerNorm)
        patches_resolution = self.encoder.patches_resolution
        self.patches_resolution = patches_resolution
        self.dim_scale = 4

        self.block1_1 = SwinTransformerBlock(dim=self.embed_dim, input_resolution=(patches_resolution[0],patches_resolution[1]), num_heads=3, shift_size=0)
        self.block1_2 = SwinTransformerBlock(dim=self.embed_dim, input_resolution=(patches_resolution[0],patches_resolution[1]), num_heads=3, shift_size=self.window_size // 2)
        self.downsample1 = PatchMerging(dim=self.embed_dim, input_resolution=(patches_resolution[0],patches_resolution[1]), norm_layer=nn.LayerNorm)

        self.block2_1 = SwinTransformerBlock(dim=self.embed_dim * 2, input_resolution=(patches_resolution[0] // 2,patches_resolution[1] // 2), num_heads=6, shift_size=0)
        self.block2_2 = SwinTransformerBlock(dim=self.embed_dim * 2, input_resolution=(patches_resolution[0] // 2,patches_resolution[1] // 2), num_heads=6, shift_size=self.window_size // 2)
        self.downsample2 = PatchMerging(dim=self.embed_dim * 2,
                                       input_resolution=(patches_resolution[0] // 2, patches_resolution[1] // 2),
                                       norm_layer=nn.LayerNorm)

        self.block3_1 = SwinTransformerBlock(dim=self.embed_dim * 4, input_resolution=(patches_resolution[0] // 4,patches_resolution[1] // 4), num_heads=12, shift_size=0)
        self.block3_2 = SwinTransformerBlock(dim=self.embed_dim * 4, input_resolution=(patches_resolution[0] // 4,patches_resolution[1] // 4), num_heads=12, shift_size=self.window_size // 2)
        self.downsample3 = PatchMerging(dim=self.embed_dim * 4,
                                       input_resolution=(patches_resolution[0] // 4, patches_resolution[1] // 4),
                                       norm_layer=nn.LayerNorm)

        self.block4_1 = SwinTransformerBlock(dim=self.embed_dim * 8, input_resolution=(patches_resolution[0] // 8,patches_resolution[1] // 8), num_heads=24, shift_size=0)
        self.block4_2 = SwinTransformerBlock(dim=self.embed_dim * 8, input_resolution=(patches_resolution[0] // 8, patches_resolution[1] // 8), num_heads=24, shift_size=self.window_size // 2)
        # self.downsample4 = PatchMerging(dim=self.embed_dim, input_resolution=(patches_resolution[0] // 8, patches_resolution[1] // 8),
        #                                 norm_layer=nn.LayerNorm)
        self.output = nn.Conv2d(in_channels=self.embed_dim // 2, out_channels=self.num_classes, kernel_size=3, bias=False)
        self.norm = nn.LayerNorm(self.embed_dim // 2)

    def forward(self, x):
        x = self.encoder(x)
        x = self.block1_1(x)
        x = self.block1_2(x)
        x = self.downsample1(x)

        x = self.block2_1(x)
        x = self.block2_2(x)
        x = self.downsample2(x)

        x = self.block3_1(x)
        x = self.block3_2(x)
        x = self.downsample3(x)

        x = self.block4_1(x)
        x = self.block4_2(x)
        # x.shape = B L C
        # H, W = self.patches_resolution // 8

        B, L, C = x.shape
        x = x.view(B, 7, 7, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        x = self.norm(x)
        _, H, W, _ = x.shape
        # x = x.view(B,4*H,4*W,-1)

        x = x.permute(0,3,1,2) #B,C,H,W
        x = self.output(x)
        out = x

if __name__ == '__main__':
    net = ResVit()
    x = torch.randn(2, 3, 224, 224)
    x = net(x)