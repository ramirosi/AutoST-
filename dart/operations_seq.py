import torch
import torch.nn as nn
import torch.nn.functional as F


OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce3D(C, C, affine=affine),
    'sep_conv_3x3': lambda C, stride, affine: SepConv3D(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv3D(C, C, 5, stride, 2, affine=affine),
    'conv_3x3': lambda C, stride, affine: ReLUConvBN3D(C, C, kernel_size=3, stride=stride, padding=1, affine=affine),
    'conv_5x5': lambda C, stride, affine: ReLUConvBN3D(C, C, kernel_size=5, stride=stride, padding=2, affine=affine),
}

class Conv3x3(nn.Module):
    def __init__(self, C_in, C_out, stride, padding, affine=True):
        super(Conv3x3, self).__init__()
        self.op = ReLUConvBN3D(C_in, C_out, kernel_size=3, stride=stride, padding=padding, affine=affine)

    def forward(self, x):
        return self.op(x)

class Conv5x5(nn.Module):
    def __init__(self, C_in, C_out, stride, padding, affine=True):
        super(Conv5x5, self).__init__()
        self.op = ReLUConvBN3D(C_in, C_out, kernel_size=5, stride=stride, padding=padding, affine=affine)

    def forward(self, x):
        return self.op(x)

class DilConv(nn.Module):
    
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = ReLUConvBN3D(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, affine=affine)
        self.op.add_module('pointwise_conv', nn.Conv3d(C_in, C_out, kernel_size=1, padding=0, bias=False))
        self.op.add_module('batch_norm', nn.BatchNorm3d(C_out, affine=affine))

    def forward(self, x):
        return self.op(x)

class SepConv3D(nn.Module):
    
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv3D, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv3d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv3d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm3d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv3d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv3d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm3d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)

class ReLUConvBN3D(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(ReLUConvBN3D, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv3d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm3d(C_out, affine=affine)
    )

  def forward(self, x):
    return self.op(x)

class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


class Zero(nn.Module):

  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride

  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    return x[:,:,::self.stride,::self.stride].mul(0.)

class FactorizedReduce3D(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce3D, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv3d(C_in, C_out // 2, kernel_size=1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv3d(C_in, C_out // 2, kernel_size=1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm3d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out1 = self.conv_1(x)
        out2 = self.conv_2(x[:, :, 1:, 1:, 1:])
        
        # Ensure the sizes match along dimension 1
        size_diff = out2.size(2) - out1.size(2)
        out1 = F.pad(out1, (0, 0, 0, 0, 0, size_diff))
        
        out = torch.cat([out1, out2], dim=1)
        out = self.bn(out)
        return out


