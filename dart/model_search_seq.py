import torch
import torch.nn as nn
import torch.nn.functional as F
from operations_seq import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype
import torch.nn.functional as F

class MixedOp3D(nn.Module):

  def __init__(self, C, stride):
    super(MixedOp3D, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in PRIMITIVES:
      op = OPS[primitive](C, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm3d(C, affine=False))
      self._ops.append(op)

  def forward(self, x, weights):
    return sum(w * op(x) for w, op in zip(weights, self._ops))

# class MixedOp3D(nn.Module):
#     def __init__(self, C, stride):
#         super(MixedOp3D, self).__init__()
#         self._ops = nn.ModuleList()
#         self.common_size = (22, 44)  # Choose a common size
#         for primitive in PRIMITIVES:
#             op = OPS[primitive](C, stride, False)
#             if 'pool' in primitive:
#                 op = nn.Sequential(op, nn.BatchNorm3d(C, affine=False))
#             self._ops.append(op)
            
#     def forward(self, x, weights):
#         intermediate_tensors = []
#         for w, op in zip(weights, self._ops):
#             tensor = op(x)
#             if tensor.shape[3:] != self.common_size:
#                 tensor = F.interpolate(tensor, size=self.common_size, mode='nearest')
#             # print(f"Intermediate tensor shape: {tensor.shape}")
#             intermediate_tensors.append(w * tensor)

#         result = sum(intermediate_tensors)
#         # print(f"Output tensor shape: {result.shape}")
#         return result

class Cell3D(nn.Module):
    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell3D, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce3D(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN3D(C_prev_prev, C, kernel_size=1, stride=1, padding=0, affine=False)

        self.preprocess1 = ReLUConvBN3D(C_prev, C, kernel_size=1, stride=1, padding=0, affine=False)

        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()

        for i in range(self._steps):
            for j in range(2+i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp3D(C, stride)
                self._ops.append(op)

    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            for j in range(2 + i):  # Ensure that the loop structure matches the creation of ops
                s = self._ops[offset + j](states[j], weights[offset + j])
                states.append(s)
            offset += 2 + i  # Adjust the offset based on the loop structure

        concatenated_states = torch.cat(states[-self._multiplier:], dim=1)
        return concatenated_states


class Network(nn.Module):
    def __init__(self, C, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
        super(Network, self).__init__()
    
        self._C = C  # Num init channels
        self._layers = layers  # Total number of layers
        self._criterion = criterion  # Loss function
        self._steps = steps  # number of intermediate nodes in the cell
        self._multiplier = multiplier  # number of intermediate nodes that contribute to the output

        # Multiplier for the amount of C (initial filters)
        C_curr = stem_multiplier * C

        self.stem = nn.Sequential(
            nn.Conv3d(1, C_curr, (3, 3, 3), padding=(1, 1, 1), bias=False),  # Set input channels to 1
            nn.BatchNorm3d(C_curr, affine=False)
        )
    
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C

        self.cells = nn.ModuleList()  # Basically a PyTorch container to place cells within

        reduction_prev = False

        for i in range(layers):  # Loop over the max amount of layers of the total network
            # Check if the current layer is 1/3 or 2/3 of the total network, if so make it a reduction cell.
            if i in [layers//3, 2*layers//3]:
                C_curr *= 2  # Double the amount of filters typically done with reduction
                reduction = True
            else:
                reduction = False

            # Create a new reduction or non-reduction cell
            cell = Cell3D(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)

            reduction_prev = reduction  # Update reduction parameter

            # Add the new cell to all other cells in the container
            self.cells += [cell]

            # Not sure?
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Linear(C_prev, 1)

        self._initialize_alphas()

    def new(self):
        model_new = Network(self._C, self._layers, self._criterion).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input):
        print(input.shape)
        input = F.interpolate(input, size=(44, 44), mode='bilinear', align_corners=False)
        input = input.view(input.size(0), 1, input.size(1), input.size(2), input.size(3))
        print(input.shape)

        s0 = s1 = self.stem(input)

        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = F.softmax(self.alphas_reduce, dim=-1)
            else:
                weights = F.softmax(self.alphas_normal, dim=-1)
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)

    def _initialize_alphas(self):
        # Calculate the number of alphas needed based on the amount of intermediate nodes in a cell
        k = sum(1 for i in range(self._steps) for n in range(2+i))
        # How many alphas node connection?
        num_ops = len(PRIMITIVES)

        self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
        self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
        self._arch_parameters = [self.alphas_normal, self.alphas_reduce,]

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):
        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

        concat = range(2+self._steps-self._multiplier, self._steps+2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype
