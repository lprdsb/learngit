import torch


a = torch.rand(10, 10)

a[a <= 0.5] = 1

print(a)