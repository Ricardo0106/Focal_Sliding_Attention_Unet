import torch
def split_flatten_t(x):
    B, C, H, W = x.shape
    split_x_b = x.split(1, dim=0)
    c = split_x_b[0]
    for i in range(B):
        split_x_b_c = split_x_b[i].split(1, dim=1)
        b = split_x_b_c[0][0]
        for j in range(C):
            a = split_x_b_c[j].squeeze(0).squeeze(0).t().flatten().unsqueeze(0)
            if j == 0:
                b = a.clone()
            else:
                b = torch.cat([b, a], dim=0)
        b = b.unsqueeze(0)
        if i == 0:
            c = b.clone()
        else:
            c = torch.cat([c, b], dim=0)
    return c


if __name__ == '__main__':
    x = torch.rand(24,3,4,4)
    c_ = split_flatten_t(x)
