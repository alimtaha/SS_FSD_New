import torch

print(torch.cuda.is_available())

print(torch.cuda.device_count())

print((torch.device('cuda:0')))
