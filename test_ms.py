import tensor
import numpy as np
import mindspore


def init_myweights(self, val):   #通过这个方法将模型各种算子初始化都统一成固定值的常数初始化，能够保证mindspore和torch从算子里出来的值更一致(测试的时候使用)
    conv_constants = getattr(nn.Conv2d(1, 1, 1), '__constants__')

    def all_in(x, y):
        for _x in x:
            if _x not in y:
                return False
        return True

    for name, module in self.named_modules():
        is_script_conv = False
        if 'Script' in type(module).__name__:
            # 1.4 workaround: now there's an original_name member so just use that
            if hasattr(module, 'original_name'):
                is_script_conv = 'Conv' in module.original_name
            # 1.3 workaround: check if this has the same constants as a conv module
            else:
                is_script_conv = (
                        all_in(module.__dict__['_constants_set'], conv_constants)
                        and all_in(conv_constants, module.__dict__['_constants_set']))
        is_conv_layer = isinstance(module, nn.Conv2d) or is_script_conv
        if is_conv_layer:
            nn.init.constant_(module.weight.data, val)
            if module.bias is not None:
                nn.init.constant_(module.bias.data, val)

if __name__ == '__main__':
    np.save('data', data)

    img = torch.from_numpy(np.load("/home/yzb/Datasets/np_save/data.npy"))