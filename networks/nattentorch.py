from torch import nn
import torch
from torch.nn.functional import unfold, pad
from timm.models.layers import trunc_normal_
import warnings


class LegacyNeighborhoodAttention(nn.Module):
    def __init__(self, dim, kernel_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 mode=1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        assert kernel_size > 1 and kernel_size % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        self.kernel_size = kernel_size
        self.win_size = kernel_size // 2
        self.mid_cell = kernel_size - 1
        self.rpb_size = 2 * kernel_size - 1

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mode = mode
        self.rpb = nn.Parameter(torch.zeros(num_heads, self.rpb_size, self.rpb_size))
        trunc_normal_(self.rpb, std=.02)
        # RPB implementation by @qwopqwop200
        self.idx_h = torch.arange(0, kernel_size)
        self.idx_w = torch.arange(0, kernel_size)
        self.idx_k = ((self.idx_h.unsqueeze(-1) * self.rpb_size) + self.idx_w).view(-1)
        warnings.warn("This is the legacy version of NAT -- it uses unfold+pad to produce NAT, and is highly inefficient.")

    def apply_pb(self, attn, height, width):
        num_repeat_h = torch.ones(self.kernel_size,dtype=torch.long)
        num_repeat_w = torch.ones(self.kernel_size,dtype=torch.long)
        num_repeat_h[self.kernel_size//2] = height - (self.kernel_size-1)
        num_repeat_w[self.kernel_size//2] = width - (self.kernel_size-1)
        bias_hw = (self.idx_h.repeat_interleave(num_repeat_h).unsqueeze(-1) * (2*self.kernel_size-1)) + self.idx_w.repeat_interleave(num_repeat_w)
        bias_idx = bias_hw.unsqueeze(-1) + self.idx_k
        bias_idx = torch.flip(bias_idx.reshape(-1, self.kernel_size**2), [0])
        return attn + self.rpb.flatten(1, 2)[:, bias_idx].reshape(self.num_heads, height * width, 1, self.kernel_size ** 2).transpose(0, 1)

    def forward(self, x):
        B, H, W, C = x.shape
        N = H * W
        num_tokens = int(self.kernel_size ** 2)
        pad_l = pad_t = pad_r = pad_b = 0
        Ho, Wo = H, W
        if N <= num_tokens:
            if self.kernel_size > W:
                pad_r = self.kernel_size - W
            if self.kernel_size > H:
                pad_b = self.kernel_size - H
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            B, H, W, C = x.shape
            N = H * W
            assert N == num_tokens, f"Something went wrong. {N} should equal {H} x {W}!"
        x = self.qkv(x).reshape(B, H, W, 3 * C)
        q, x = x[:, :, :, :C], x[:, :, :, C:]
        q = q.reshape(B, N, self.num_heads, C // self.num_heads, 1).transpose(3, 4) * self.scale
        pd = self.kernel_size - 1
        pdr = pd // 2
        if self.mode == 0:
            x = x.permute(0, 3, 1, 2).flatten(0, 1)
            x = x.unfold(1, self.kernel_size, 1).unfold(2, self.kernel_size, 1).permute(0, 3, 4, 1, 2)
            x = pad(x, (pdr, pdr, pdr, pdr, 0, 0), 'replicate')
            x = x.reshape(B, 2, self.num_heads, C // self.num_heads, num_tokens, N)
            x = x.permute(1, 0, 5, 2, 4, 3)
        elif self.mode == 1:
            Hr, Wr = H - pd, W - pd
            x = unfold(x.permute(0, 3, 1, 2),
                       kernel_size=(self.kernel_size, self.kernel_size),
                       stride=(1, 1),
                       padding=(0, 0)).reshape(B, 2 * C * num_tokens, Hr, Wr)
            x = pad(x, (pdr, pdr, pdr, pdr), 'replicate').reshape(
                B, 2, self.num_heads, C // self.num_heads, num_tokens, N)
            x = x.permute(1, 0, 5, 2, 4, 3)
        else:
            raise NotImplementedError(f'Mode {self.mode} not implemented for NeighborhoodAttention2D.')
        k, v = x[0], x[1]

        attn = (q @ k.transpose(-2, -1))  # B x N x H x 1 x num_tokens
        attn = self.apply_pb(attn, H, W)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v)  # B x N x H x 1 x C
        x = x.reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Ho, :Wo, :]
        return self.proj_drop(self.proj(x))

class LegacyNeighborhoodAttentionDownSampler(nn.Module):
    def __init__(self, dim, kernel_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 mode=1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        assert kernel_size > 1 and kernel_size % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        self.kernel_size = kernel_size
        self.win_size = kernel_size // 2
        self.mid_cell = kernel_size - 1
        self.rpb_size = 2 * kernel_size - 1

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.roj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mode = mode
        self.rpb = nn.Parameter(torch.zeros(num_heads, self.rpb_size, self.rpb_size))
        trunc_normal_(self.rpb, std=.02)
        # RPB implementation by @qwopqwop200
        self.idx_h = torch.arange(0, kernel_size)
        self.idx_w = torch.arange(0, kernel_size)
        self.idx_k = ((self.idx_h.unsqueeze(-1) * self.rpb_size) + self.idx_w).view(-1)
        warnings.warn("This is the legacy version of NAT -- it uses unfold+pad to produce NAT, and is highly inefficient.")

    def apply_pb(self, attn, height, width):
        num_repeat_h = torch.ones(self.kernel_size,dtype=torch.long)
        num_repeat_w = torch.ones(self.kernel_size,dtype=torch.long)
        num_repeat_h[self.kernel_size//2] = height - (self.kernel_size-1)
        num_repeat_w[self.kernel_size//2] = width - (self.kernel_size-1)
        bias_hw = (self.idx_h.repeat_interleave(num_repeat_h).unsqueeze(-1) * (2*self.kernel_size-1)) + self.idx_w.repeat_interleave(num_repeat_w)
        bias_idx = bias_hw.unsqueeze(-1) + self.idx_k
        bias_idx = torch.flip(bias_idx.reshape(-1, self.kernel_size**2), [0])
        return attn + self.rpb.flatten(1, 2)[:, bias_idx].reshape(self.num_heads, height * width, 1, self.kernel_size ** 2).transpose(0, 1)

    def forward(self, x):
        B, H, W, C = x.shape
        N = H * W
        num_tokens = int(self.kernel_size ** 2)
        pad_l = pad_t = pad_r = pad_b = 0
        Ho, Wo = H, W
        if N <= num_tokens:
            if self.kernel_size > W:
                pad_r = self.kernel_size - W
            if self.kernel_size > H:
                pad_b = self.kernel_size - H
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            B, H, W, C = x.shape
            N = H * W
            assert N == num_tokens, f"Something went wrong. {N} should equal {H} x {W}!"
        x = self.qkv(x).reshape(B, H, W, 3 * C)
        q, x = x[:, :, :, :C], x[:, :, :, C:]
        q = q.reshape(B, N, self.num_heads, C // self.num_heads, 1).transpose(3, 4) * self.scale
        pd = self.kernel_size - 1
        pdr = pd // 2
        if self.mode == 0:
            x = x.permute(0, 3, 1, 2).flatten(0, 1)
            x = x.unfold(1, self.kernel_size, 1).unfold(2, self.kernel_size, 1).permute(0, 3, 4, 1, 2)
            x = pad(x, (pdr, pdr, pdr, pdr, 0, 0), 'replicate')
            x = x.reshape(B, 2, self.num_heads, C // self.num_heads, num_tokens, N)
            x = x.permute(1, 0, 5, 2, 4, 3)
        elif self.mode == 1:
            Hr, Wr = H - pd, W - pd
            x = unfold(x.permute(0, 3, 1, 2),
                       kernel_size=(self.kernel_size, self.kernel_size),
                       stride=(2, 2),
                       padding=(0, 0))
            x = x.reshape(B, 2 * C * num_tokens, Hr // 2, Wr // 2)
            _, _ ,new_H, _ = x.shape
            pdr_ = (H // 2 - new_H) // 2
            x = pad(x, (pdr_, pdr_, pdr_, pdr_), 'replicate')
            if (H // 2 -  new_H) % 2 != 0:
                x = pad(x, (1, 0, 1, 0), 'replicate')
            x = x.reshape(B, 2, self.num_heads, C // self.num_heads, num_tokens, N // 4)
            x = x.permute(1, 0, 5, 2, 4, 3)
        else:
            raise NotImplementedError(f'Mode {self.mode} not implemented for NeighborhoodAttention2D.')
        k, v = x[0], x[1]

        attn = (q @ k.transpose(-2, -1))  # B x N x H x 1 x num_tokens
        attn = self.apply_pb(attn, H // 2, W // 2)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v)  # B x N x H x 1 x C
        x = x.reshape(B, H // 2, W // 2, C)
        if pad_r or pad_b:
            x = x[:, :Ho, :Wo, :]
        return self.proj_drop(self.proj(x))

class LegacyNeighborhoodAttentionDownSampler_v2(nn.Module):
    def __init__(self, dim, kernel_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 mode=1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        assert kernel_size > 1 and kernel_size % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        self.kernel_size = kernel_size
        self.win_size = kernel_size // 2
        self.mid_cell = kernel_size - 1
        self.rpb_size = 2 * kernel_size - 1

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.roj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mode = mode
        self.rpb = nn.Parameter(torch.zeros(num_heads, self.rpb_size, self.rpb_size))
        trunc_normal_(self.rpb, std=.02)
        # RPB implementation by @qwopqwop200
        self.idx_h = torch.arange(0, kernel_size)
        self.idx_w = torch.arange(0, kernel_size)
        self.idx_k = ((self.idx_h.unsqueeze(-1) * self.rpb_size) + self.idx_w).view(-1)
        warnings.warn("This is the legacy version of NAT -- it uses unfold+pad to produce NAT, and is highly inefficient.")

    def apply_pb(self, attn, height, width):
        num_repeat_h = torch.ones(self.kernel_size,dtype=torch.long)
        num_repeat_w = torch.ones(self.kernel_size,dtype=torch.long)
        num_repeat_h[self.kernel_size//2] = height - (self.kernel_size-1)
        num_repeat_w[self.kernel_size//2] = width - (self.kernel_size-1)
        bias_hw = (self.idx_h.repeat_interleave(num_repeat_h).unsqueeze(-1) * (2*self.kernel_size-1)) + self.idx_w.repeat_interleave(num_repeat_w)
        bias_idx = bias_hw.unsqueeze(-1) + self.idx_k
        bias_idx = torch.flip(bias_idx.reshape(-1, self.kernel_size**2), [0])
        return attn + self.rpb.flatten(1, 2)[:, bias_idx].reshape(self.num_heads, height * width, 1, self.kernel_size ** 2).transpose(0, 1)

    def forward(self, x):
        B, H, W, C = x.shape
        N = H * W
        num_tokens = int(self.kernel_size ** 2)
        pad_l = pad_t = pad_r = pad_b = 0
        Ho, Wo = H, W
        if N <= num_tokens:
            if self.kernel_size > W:
                pad_r = self.kernel_size - W
            if self.kernel_size > H:
                pad_b = self.kernel_size - H
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            B, H, W, C = x.shape
            N = H * W
            assert N == num_tokens, f"Something went wrong. {N} should equal {H} x {W}!"
        x = self.qkv(x).reshape(B, H, W, 3 * C)
        q, x = x[:, :, :, :C], x[:, :, :, C:]
        q = q.reshape(B, N, self.num_heads, C // self.num_heads, 1).transpose(3, 4) * self.scale
        pd = self.kernel_size - 1
        pdr = pd // 2
        if self.mode == 0:
            x = x.permute(0, 3, 1, 2).flatten(0, 1)
            x = x.unfold(1, self.kernel_size, 1).unfold(2, self.kernel_size, 1).permute(0, 3, 4, 1, 2)
            x = pad(x, (pdr, pdr, pdr, pdr, 0, 0), 'replicate')
            x = x.reshape(B, 2, self.num_heads, C // self.num_heads, num_tokens, N)
            x = x.permute(1, 0, 5, 2, 4, 3)
        elif self.mode == 1:
            Hr, Wr = H - pd, W - pd
            x = unfold(x.permute(0, 3, 1, 2),
                       kernel_size=(self.kernel_size, self.kernel_size),
                       stride=(1, 1),
                       padding=(0, 0)).reshape(B, 2 * C * num_tokens, Hr, Wr)
            x = pad(x, (pdr, pdr, pdr, pdr), 'replicate').reshape(
                B, 2, self.num_heads, C // self.num_heads, num_tokens, N)
            x = x.permute(1, 0, 5, 2, 4, 3)
        else:
            raise NotImplementedError(f'Mode {self.mode} not implemented for NeighborhoodAttention2D.')
        k, v = x[0], x[1]

        attn = (q @ k.transpose(-2, -1))  # B x N x H x 1 x num_tokens
        attn = self.apply_pb(attn, H, W)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v)  # B x N x H x 1 x C
        x = x.reshape(B, H, W, C)

        if pad_r or pad_b:
            x = x[:, :Ho, :Wo, :]
        return self.proj_drop(self.proj(x))

class recombinationDownSampler(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        B, H, W, C = x.shape
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x