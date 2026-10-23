"""Exact channelwise Conv2d execution to bound large dilated DW workspaces."""
import types
import torch
from torch import nn
import torch.nn.functional as F


def channelwise(module,x,chunk=32):
    outputs=[]
    for start in range(0,module.in_channels,chunk):
        stop=min(start+chunk,module.in_channels)
        bias=None if module.bias is None else module.bias[start:stop]
        outputs.append(F.conv2d(x[:,start:stop],module.weight[start:stop],bias,module.stride,
            module.padding,module.dilation,stop-start))
    return torch.cat(outputs,1)


def install(net):
    count=0
    for module in net.modules():
        if isinstance(module,nn.Conv2d) and module.groups==module.in_channels==module.out_channels and module.in_channels>=64 and module.kernel_size!=(1,1) and module.padding_mode=='zeros':
            module.forward=types.MethodType(lambda self,x:channelwise(self,x),module);count+=1
    return count
