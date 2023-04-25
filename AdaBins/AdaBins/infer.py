import glob
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import model_io
import utils
from models import UnetAdaptiveBins
from loss import SILogLoss, BinsChamferLoss


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


class ToTensor(object):
    def __init__(self):
        self.normalize = transforms.Normalize(
            mean=[
                0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225])

    def __call__(self, image, target_size=(640, 480)):
        # image = image.resize(target_size)
        image = self.to_tensor(image)
        image = self.normalize(image)
        return image

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(
                    pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img


class InferenceHelper:
    def __init__(self, dataset='cityscapes', device='cuda:0'):  # change dataset to kitti
        self.toTensor = ToTensor()
        self.device = device
        if dataset == 'nyu':
            self.min_depth = 1e-3
            self.max_depth = 10
            self.saving_factor = 1000  # used to save in 16 bit
            model = UnetAdaptiveBins.build(
                n_bins=256, min_val=self.min_depth, max_val=self.max_depth)
            pretrained_path = "./pretrained/AdaBins_nyu.pt"
        elif dataset == 'kitti':
            self.min_depth = 1e-3
            self.max_depth = 80
            self.saving_factor = 256
            model = UnetAdaptiveBins.build(
                n_bins=256, min_val=self.min_depth, max_val=self.max_depth)
            pretrained_path = "./pretrained/AdaBins_kitti.pt"
        elif dataset == 'cityscapes':
            self.min_depth = 1e-3
            self.max_depth = 256
            self.saving_factor = 256
            model = UnetAdaptiveBins.build(
                n_bins=256, min_val=self.min_depth, max_val=self.max_depth)
            pretrained_path = "./pretrained/AdaBins_cityscapes.pt"
        else:
            raise ValueError(
                "dataset can be either 'nyu' or 'kitti' but got {}".format(dataset))

        self.model, _, _ = model_io.load_checkpoint(pretrained_path, model)
        model.eval()
        #self.model = model.to(self.device)

    @torch.no_grad()
    def predict_pil(self, pil_image, visualized=False):
        # pil_image = pil_image.resize((640, 480))
        img = np.asarray(pil_image) / 255.

        img = self.toTensor(img).unsqueeze(0).float()  # .to(self.device)
        print('Inference Image Dimensions', img.shape)
        bin_centers, pred, bins = self.predict(img)

        if visualized:
            viz = utils.colorize(
                torch.from_numpy(pred).unsqueeze(0),
                vmin=None,
                vmax=None,
                cmap='magma')
            # pred = np.asarray(pred*1000, dtype='uint16')
            # unspoported type passed into Image.fromarray function
            viz = Image.fromarray(viz)
            return bin_centers, pred, viz
        return bin_centers, pred, bins

    @torch.no_grad()
    def predict(self, image):
        bins, pred = self.model(image)  # non flipped image producing depth map
        # clipped so values outside max are clipped to max and values below min
        # are set to the min
        pred = np.clip(pred.cpu().numpy(), self.min_depth, self.max_depth)

        # Flip
        print("image shape", image.shape)
        # .to(self.device) #the -1 means start from the end, so the image is flipped in the last dimension, in this case the weight
        image = torch.Tensor(np.array(image.cpu().numpy())[..., ::-1].copy())
        print("image shape after numpy operation", image.shape)
        # taking last element from whats returned- only depth map needed from
        # flipped image
        pred_lr = self.model(image)[-1]
        # after the pred_lr is derived from the flipped image, the pred_lr is
        # flipped again so it matches the orientaion of the original prediction
        # before additions/combination of both
        pred_lr = np.clip(pred_lr.cpu().numpy()
                          [..., ::-1], self.min_depth, self.max_depth)

        # Take average of original and mirror
        final = 0.5 * (pred + pred_lr)
        print("final image shape [-2:]", image.shape[-2:])
        # final = nn.functional.interpolate(torch.Tensor(final), image.shape[-2:],
        # mode='bilinear', align_corners=True).cpu()#.numpy() #produced depth
        # map is only H/2 and W/2, use bilinear interpolation for restoring to
        # original dimensions

        print("final image prediction size after interpolation", final.shape)
        final[final < self.min_depth] = self.min_depth
        final[final > self.max_depth] = self.max_depth
        final[np.isinf(final)] = self.max_depth
        final[np.isnan(final)] = self.min_depth
        # 4 lines above basically same as clipping after the interpolation

        centers = 0.5 * (bins[:, 1:] + bins[:, :-1])
        centers = centers.cpu().squeeze().numpy()
        centers = centers[centers > self.min_depth]
        centers = centers[centers < self.max_depth]

        return centers, final, bins

    @torch.no_grad()
    def predict_dir(self, test_dir, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        transform = ToTensor()
        all_files = glob.glob(os.path.join(test_dir, "*"))
        self.model.eval()
        for f in tqdm(all_files):
            image = np.asarray(Image.open(f), dtype='float32') / 255.
            image = transform(image).unsqueeze(0).to(self.device)

            centers, final = self.predict(image)
            # final = final.squeeze().cpu().numpy()

            final = (final * self.saving_factor).astype('uint16')
            basename = os.path.basename(f).split('.')[0]
            save_path = os.path.join(out_dir, basename + ".png")

            Image.fromarray(final.squeeze()).save(save_path)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from time import time

    fig = plt.figure(figsize=(10, 7))
    img = Image.open(
        "/media/taha_a/T7/leftImg8bit_sequence/val/depth_training/frankfurt_000000_000279_leftImg8bit.png")
    fig.add_subplot(4, 1, 1)
    plt.imshow(img)
    #img = img.resize((int(img.width / 2), int(img.height / 2)))
    start = time()
    inferHelper = InferenceHelper()
    criterion_ueff = SILogLoss()
    criterion_bins = BinsChamferLoss()
    centers, pred, bins = inferHelper.predict_pil(img)
    print(pred.squeeze().shape)
    depth_gt = Image.open(
        "/media/taha_a/T7/Datasets/cityscapes/DVPS_Depth/train/frankfurt_000000_000279_depth.png")
    #depth_gt = depth_gt.resize((int(depth_gt.width / 2), int(depth_gt.height / 2)))

    '''
    img = Image.open("../dataset/kitti/raw/2011_09_26/2011_09_26_drive_0002_sync/image_02/data/0000000005.png")
    start = time()
    inferHelper = InferenceHelper()
    centers, pred = inferHelper.predict_pil(img)
    print(pred.squeeze().shape)
    depth_gt = Image.open("../dataset/kitti/gts/2011_09_26_drive_0002_sync/proj_depth/groundtruth/image_02/0000000005.png")
    '''
    image = np.asarray(img, dtype=np.float32) / 255.0
    depth_gt = np.asarray(depth_gt, dtype=np.float32)
    print("depth image size before squeeze", depth_gt.shape)
    depth_gt = depth_gt.squeeze() / 256.0
    depth_gt = np.expand_dims(depth_gt, axis=2)
    print("depth image size after expand dims", depth_gt.shape)
    image = torch.from_numpy(image.transpose((2, 0, 1)))
    depth_gt = torch.from_numpy(depth_gt.transpose((2, 0, 1)))
    print("depth image size after transpose", depth_gt.shape)
    depth_gt = depth_gt.unsqueeze(0)
    pred = torch.Tensor(pred)
    mask_min = depth_gt > 0.001
    mask_max = depth_gt < 80.001
    mask = mask_max & mask_min
    l_dense = criterion_ueff(
        pred, depth_gt, mask=mask.to(
            torch.bool), interpolate=True)
    l_chamfer = criterion_bins(bins, depth_gt)
    loss = l_dense + 0.1 * l_chamfer

    pred = nn.functional.interpolate(torch.Tensor(pred), image.shape[-2:],
                                     mode='bilinear', align_corners=True).cpu()
    print(f"took :{time() - start}s")

    input = pred * mask.int().float()
    target = depth_gt * mask.int().float()
    print("depth image size", depth_gt.shape)
    print("mask shape", mask.shape)
    print("input shape", input)
    g = torch.log(input) - torch.log(target)
    print(l_dense)
    print("loss shape", g.shape)
    #plt.imshow(depth_gt, cmap='magma_r')
    fig.add_subplot(4, 1, 2)
    plt.imshow(pred.squeeze(), cmap='magma_r')
    plt.colorbar()
    fig.add_subplot(4, 1, 3)
    plt.imshow(depth_gt.squeeze(), cmap='magma_r')
    plt.colorbar()
    fig.add_subplot(4, 1, 4)
    g = pred - depth_gt
    plt.imshow(g.squeeze(), cmap='magma_r')
    plt.colorbar()
    plt.show()
    #plt.imshow(pred.squeeze(), cmap='magma_r')
    # plt.colorbar()
    # plt.show()
