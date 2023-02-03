import torch
from torch import nn


def generate_mask(pooling_counts, N):
    mask2d = torch.zeros(N, N, dtype=torch.bool)
    mask2d[range(N), range(N)] = 1

    stride, offset = 1, 0
    maskij = []
    for c in pooling_counts:
        for _ in range(c):
            # fill a diagonal line
            offset += stride
            i, j = range(0, N - offset, stride), range(offset, N, stride)
            mask2d[i, j] = 1
            maskij.append((i, j))
        stride *= 2
    return mask2d

class SparseMaxPool(nn.Module):
    def __init__(self, pooling_counts, N):
        super(SparseMaxPool, self).__init__()
        mask2d = torch.zeros(N, N, dtype=torch.bool)
        mask2d[range(N), range(N)] = 1

        stride, offset = 1, 0
        maskij = []
        for c in pooling_counts:
            for _ in range(c):
                # fill a diagonal line
                offset += stride
                i, j = range(0, N - offset, stride), range(offset, N, stride)
                mask2d[i, j] = 1
                maskij.append((i, j))
            stride *= 2

        poolers = [nn.MaxPool1d(2, 1) for _ in range(pooling_counts[0])]
        for c in pooling_counts[1:]:
            poolers.extend(
                [nn.MaxPool1d(3, 2)] + [nn.MaxPool1d(2, 1) for _ in range(c - 1)]
            )

        self.mask2d = mask2d.to("cuda")
        self.maskij = maskij
        self.poolers = poolers

    def forward(self, x):
        B, D, N = x.shape
        map2d = x.new_zeros(B, D, N, N)
        map2d[:, :, range(N), range(N)] = x
        for pooler, (i, j) in zip(self.poolers, self.maskij):
            x = pooler(x)
            map2d[:, :, i, j] = x
        return map2d


def mask2weight(mask2d, mask_kernel, padding=0):
    # from the feat2d.py,we can know the mask2d is 4-d
    weight = torch.conv2d(mask2d[None, None, :, :].float(),
                          mask_kernel, padding=padding)[0, 0]
    weight[weight > 0] = 1 / weight[weight > 0]
    return weight


class Predictor(nn.Module):
    def __init__(self, input_size, hidden_size, k, num_stack_layers, mask2d):
        super(Predictor, self).__init__()

        # Padding to ensure the dimension of the output map2d
        mask_kernel = torch.ones(1, 1, k, k).to(mask2d.device)
        first_padding = (k - 1) * num_stack_layers // 2

        self.weights = [
            mask2weight(mask2d, mask_kernel, padding=first_padding)
        ]
        self.convs = nn.ModuleList(
            [nn.Conv2d(input_size, hidden_size, k, padding=first_padding)]
        )

        for _ in range(num_stack_layers - 1):
            self.weights.append(mask2weight(self.weights[-1] > 0, mask_kernel))
            self.convs.append(nn.Conv2d(hidden_size, hidden_size, k))
        self.pred = nn.Conv2d(hidden_size, 1, 1)

    def forward(self, x):
        for conv, weight in zip(self.convs, self.weights):
            x = conv(x).relu() * weight
        x = self.pred(x).squeeze_()
        return x