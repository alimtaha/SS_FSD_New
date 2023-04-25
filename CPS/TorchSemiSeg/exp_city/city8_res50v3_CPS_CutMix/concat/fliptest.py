import torch
import cv2
import numpy as np
# tensor = torch.arange(8).reshape((2,2,2)).unsqueeze(0)
# tensor = tensor.unsqueeze(0)
# #tensorflip = torch.flip(tensor, dims=[2])
# tensorflip = tensor[:, :, :, ::-1]
# print(tensorflip)
# tensorlr = torch.fliplr(tensor)
# print(tensorlr)

sample_img = '/mnt/Dataset/leftImg8bit_trainvaltest_blurred/leftImg8bit/train/aachen/aachen_000000_000019.jpg'
img = cv2.imread(sample_img)

save_path = '/home/ubuntu/SS_FSD_New/SS_FSD_New/CPS/TorchSemiSeg/exp_city/city8_res50v3_CPS_CutMix/concat'



tensor_img = torch.Tensor(img)
#flipped_img = torch.flip(tensor_img, dims=[2])
print(tensor_img.shape)
flipped_img = tensor_img.numpy()[:, ::-1, :]
#flipped_lr_img = torch.fliplr(tensor_img)
flipped_img = torch.from_numpy(np.copy(flipped_img)).squeeze()

cv2.imwrite(save_path + '/flip.png', img=flipped_img.numpy())
#cv2.imwrite(save_path + '/flip_lr.png', img=flipped_lr_img.numpy())