import torch

ls = [torch.tensor((x, x)) for x in range(5)]
lss = torch.cat(torch.tensor((7, 7)), ls, dim=0)

print(lss)
