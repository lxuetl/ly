import torch
from safetensors.torch import save_file

print("正在读取 bin 文件...")
# 1. 加载你下载的旧格式权重
state_dict = torch.load('./weights/depth_anything_vits14/pytorch_model.bin', map_location='cpu')

print("正在转换为 safetensors...")
# 2. 直接存成新格式
save_file(state_dict, './weights/depth_anything_vits14/model.safetensors')

print("转换完成！")