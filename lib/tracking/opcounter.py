from torchvision.models import resnet50
from thop import profile
import torch
from torch import nn
from lib.models.trackingmambav2.trackingmambav2 import build_trackingmambav2
from lib.config.trackingmambav2.config import cfg, update_config_from_file

model = build_trackingmambav2(cfg).cuda()
var2 = torch.randn(1, 3, 256, 256).cuda()
var1 = torch.randn(1, 3, 128, 128).cuda()
tensor1 = torch.randn(1, 1, 384)  # 标准正态分布
tensor2 = torch.randn(1, 1, 384)
tensor3 = torch.randn(1, 1, 384)

# 将张量存入列表
tensor_list = []


macs, params = profile(model, inputs=(var1, var2, False, False, tensor_list))
                        #custom_ops={YourModule: count_your_model})

# print(macs)
# print(params)
print('FLOPs = ' + str(macs / 1000 ** 3) + 'G')
print('Params = ' + str(params / 1000 ** 2) + 'M')