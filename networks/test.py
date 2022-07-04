import torch
import numpy as np

if __name__ == '__main__':
    x = torch.tensor(np.random.randn(1, 1, 4, 4))
    print(x)
    x = x.permute(0, 2, 3, 1)
    x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
    x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
    x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
    x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
    x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
    x = x.permute(0, 3, 2, 1)
    y = x.reshape(1, 4, 2, 2)
    print(y)