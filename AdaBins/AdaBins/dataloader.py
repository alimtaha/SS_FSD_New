# This file is mostly taken from BTS; author: Jin Han Lee, with only
# slight modifications

import os
import random
import json

import numpy as np
import torch
import torch.utils.data.distributed
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def _is_pil_image(img):  # returns whether the image is a PIL object
    """
    Check if the input is a PIL image. If it is, return true. Otherwise, return false.
    @param img - the input image
    @return True if the input is a PIL image, False otherwise
    """
    return isinstance(img, Image.Image)


def _is_numpy_image(img):  # returns whether the image is a numpy object
    """
    Check if the input is a numpy image.
    @param img - the input image
    @returns True if the input is a numpy image, False otherwise.
    """
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


# preprocessing to transform image to a tensor
def preprocessing_transforms(mode):
    """
    Create the transforms for the data. This is used to convert the data to a format that can be used by the network.
    @param mode - the mode of the data that is being used. Either train, online_eval or test.
    @returns the transforms
    """
    return transforms.Compose([
        # a class instance is instantiated with the correct mode and the
        # instance is returned (the instance is assigned to 'transform' in line
        # 53)
        ToTensor(mode=mode)
    ])


# PROCESS FLOW: Dataset defined through dataset class -> data loader defined which references dataset class. When dataloader is called, it used the initialised
# (I THINK) dataset provides one image only per call of _get_item_, and dataloader (depending on batch size), returns a batch of n samples by calling _get_item_ n times, this ensures every sample even within the same batch is tarnsformed differently

class DepthDataLoader(
        object):  # loads the dataset and stores the PyTorch DataLoader in self.data - called from the train.py and evaluate.py files
    """
    Load the depth data from the file.
    @param filename - the file name of the depth data file.
    @returns the depth data as a numpy array.
    """
    """
    Initialize the dataloader for training or testing.
    @param args - the arguments from the command line interface.
    @param mode - the mode we are in, either train or test.
    @returns the data loader for the training and test data
    """

    # mode derived also from class called (called from the train.py and
    # evaluate.py files)
    def __init__(self, args, mode):

        if mode == 'train':
            self.training_samples = DataLoadPreprocess(
                args,
                mode,
                transform=preprocessing_transforms(mode))  # data is sent to data load preprocess for
            if args.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                    self.training_samples)
            else:
                self.train_sampler = None

            self.data = DataLoader(self.training_samples, args.batch_size,  # the dataloader object takes in a dataset as an input, and provides an iterable over the given dataset
                                   shuffle=(self.train_sampler is None),
                                   num_workers=args.num_threads,
                                   pin_memory=True,
                                   sampler=self.train_sampler)

        elif mode == 'online_eval':
            self.testing_samples = DataLoadPreprocess(
                args, mode, transform=preprocessing_transforms(mode))
            if args.distributed:  # redundant. here only for readability and to be more explicit
                # Give whole test set to all processes (and perform/report
                # evaluation only on one) regardless
                self.eval_sampler = None
            else:
                self.eval_sampler = None
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=False,
                                   num_workers=1,
                                   pin_memory=False,
                                   sampler=self.eval_sampler)

        elif mode == 'test':
            self.testing_samples = DataLoadPreprocess(
                args, mode, transform=preprocessing_transforms(mode))
            self.data = DataLoader(
                self.testing_samples,
                1,
                shuffle=False,
                num_workers=1)

        else:
            print(
                'mode should be one of \'train, test, online_eval\'. Got {}'.format(mode))


def remove_leading_slash(s):
    if s[0] == '/' or s[0] == '\\':
        return s[1:]
    return s


class DataLoadPreprocess(Dataset):

    """
    The DataLoadPreprocess class is used to load the data and preprocess it. It inherits from the torch.utils.data.Dataset class.
    Basically the class is called once to initialise, then for every image to be loaded the getitem method is called
    The DataLoadPreprocess class returns an image and depth
    """
    """
    The __len__ method is used to return the length of the dataset.
    @returns the length of the dataset
    """

    # uses torch.utils.data class

    def __init__(self, args, mode, transform=None, is_for_online_eval=False):
        '''
        initialise the class with the required arguments for preprocessing
        open the file with the filenames and paths for the images and save them in self.filenames
        '''

        # args from arg parses (terminal) - default KITTI values already
        # defined
        self.args = args
        if mode == 'online_eval':  # if running in online eval mode, open the eval file
            with open(args.filenames_file_eval, 'r') as f:
                # filenames_file =
                # ./train_test_inputs/kitti_eigen_train_files_with_gt.txt
                self.filenames = f.readlines()
        # readlines() method returns a list containing each line in the file as
        # a list item.
        else:
            with open(args.filenames_file, 'r') as f:
                self.filenames = f.readlines()  # reding the image names into self.filenames list

        self.mode = mode  # mode derived from whatever called DataLoadPreprocess
        # transform here is passed in from the DepthDataLoader class where it
        # instantiates an instance of the ToTensor class (bottom of this file)
        self.transform = transform
        # NOT USED? since the transform parameter for the function is passed
        # from the instance of the class when defined.
        self.to_tensor = ToTensor
        self.is_for_online_eval = is_for_online_eval

    def __getitem__(self, idx):
        """
        Load the image and depth data from the filepaths. If the mode is 'train' and the dataset is 'kitti' and the use_right flag is set, then we will randomly select whether to use the left or right image.
        @param idx - the filepath to the sample image and depth gt pair
        @returns the image and depth data as a tensor since pass into the ToTensor class first (which is instantiated to 'transform')
        """

        # getting a specific image and depth path from the self.filenames list
        # (derived using readlines() above)
        sample_path = self.filenames[idx]
        # getting the (focal length? in pixels?) - lives in the last column in
        # the train/test filename_file
        focal = float(sample_path.split()[
                      2]) if self.args.dataset != 'cityscapes' else 0

        if self.mode == 'train':
            if self.args.dataset == 'kitti' and self.args.use_right is True and random.random() > 0.5:
                # image_path = os.path.join(self.args.data_path,
                # remove_leading_slash(sample_path.split()[3]))  #path for the
                # image formed by joining working directory path plus path from
                # filename
                image_path = os.path.join(
                    self.args.data_path, remove_leading_slash(
                        sample_path.split()[3]))
                
                depth_path = os.path.join(
                    self.args.gt_path, remove_leading_slash(
                        sample_path.split()[4]))
            
            else:
                image_path = os.path.join(
                    self.args.data_path, remove_leading_slash(
                        sample_path.split()[0])).replace('.png', '.jpg')
                
                if self.args.disparity:
                    depth_path = os.path.join(
                        self.args.disparity_path, remove_leading_slash(
                            sample_path.split()[1])).replace('.png', '_disparity.png')

                    camera_path = os.path.join(self.args.camera_path, remove_leading_slash(
                            sample_path.split()[2]))
                else:
                    depth_path = os.path.join(
                        self.args.gt_path, remove_leading_slash(
                            sample_path.split()[1]))

            """
            Image and depth gt opened below
            """
            image = Image.open(
                image_path)  # open image and depth map gt using the image path
            depth_gt = Image.open(depth_path)

            """
            transforms of the image below
            """

            # because the depth and kitti image are of different sizes, the function below makes them both the same size
            # image dimensions used as the basis for cropping
            # the paper defines a crop size of 704 X 352 aspect ratio is 2:1
            if self.args.do_kb_crop is True:
                height = image.height  # gt is 1242 x 375, the image height and depth in args file is 1241 x 376, some depth gt height and width are slightly different
                width = image.width
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                # top and left margin normalise the image size (1216 x 352)to
                # be the same regardless of the image size?
                depth_gt = depth_gt.crop(
                    (left_margin,
                     top_margin,
                     left_margin +
                     1216,
                     top_margin +
                     352))  # cropping to a size of 1216 x 352 according to this
                image = image.crop(
                    (left_margin, top_margin, left_margin + 1216, top_margin + 352))

            # To avoid blank boundaries due to pixel registration
            if self.args.dataset == 'nyu':
                depth_gt = depth_gt.crop((43, 45, 608, 472))
                image = image.crop((43, 45, 608, 472))

            if self.args.dataset == 'cityscapes':
                top_margin = int(image.height - 1024)
                left_margin = int(image.width - 2048)
                depth_gt = depth_gt.crop(
                    (left_margin,
                     top_margin,
                     left_margin +
                     2048,
                     top_margin +
                     1024))  # cropping to a size of 1216 x 352 according to this
                image = image.crop(
                    (left_margin, top_margin, left_margin + 2048, top_margin + 1024))
                #depth_gt = depth_gt.resize((int(depth_gt.width / 2), int(depth_gt.height / 2)))
                #image = image.resize((int(image.width / 2), int(image.height / 2)))

            if self.args.do_random_rotate is True:
                random_angle = (random.random() - 0.5) * 2 * self.args.degree
                # image rotated at random angle
                image = self.rotate_image(image, random_angle)
                depth_gt = self.rotate_image(
                    depth_gt, random_angle, flag=Image.NEAREST)

            """
                Normalize the image and depth data. Also add a third dimension to the depth data.
                @param image - the image data
                @param depth_gt - the depth data
                @return the normalized image and depth data
            """

            # converting image to array, normalise to value between 0 & 1
            # (since images have 8 bit colour values between 0-255)
            image = np.asarray(image, dtype=np.float32) / 255.0
            # converting depth image to array
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(
                depth_gt, axis=2)  # like torch.unsqueeze()

            if self.args.dataset == 'nyu':
                depth_gt = depth_gt / 1000.0
            else:
                if self.args.disparity:
                    with open(camera_path, 'r') as f:   # load camera parameters and convert to metres
                        camera_params = json.loads(f.read())
                        depth_gt[depth_gt > 0] = (depth_gt[depth_gt > 0] - 1) / 256
                        depth_gt[depth_gt > 0] = camera_params['extrinsic']['baseline'] * (
                            (camera_params['intrinsic']['fx'] + camera_params['intrinsic']['fy']) / 2) / depth_gt[depth_gt > 0]
                        depth_gt[depth_gt == np.inf] = 0
                        depth_gt[depth_gt == np.nan] = 0
                else:
                    depth_gt = depth_gt / 256.0  # convert to metres

            # random crop as defined in args file - for training cropped to 704
            # X 352
            image, depth_gt = self.random_crop(
                image, depth_gt, self.args.input_height, self.args.input_width)
            image, depth_gt = self.train_preprocess(image, depth_gt)
            sample = {
                'image': image,
                'depth': depth_gt,
                'focal': focal,
                'image_path': sample_path.split()[0],
                'depth_path': sample_path.split()[1]}  # code packed into dictionary for returning to model through eval or train file
            # add code here to look into the image dimensions, and visualise the images
            # image transformation pipeline: open image -> kb_crop -> random_rotate -> np.asarray
            #  -> normalise values (divide by 255 image and 256 depth) -> random crop -> train_preprocess (flip, augmenting, etc)

        else:
            if self.mode == 'online_eval':
                data_path = self.args.data_path_eval
            else:
                data_path = self.args.data_path

            image_path = os.path.join(
                data_path, remove_leading_slash(
                    sample_path.split()[0])).replace('.png', '.jpg')
            image = Image.open(image_path)

            if self.args.dataset == 'cityscapes':
                top_margin = int(image.height - 1024)
                left_margin = int(image.width - 2048)
                image = image.crop(
                    (left_margin, top_margin, left_margin + 2048, top_margin + 1024))
                #image = image.resize((int(image.width / 2), int(image.height / 2)))

            image = np.asarray(image, dtype=np.float32) / 255.0

            if self.mode == 'online_eval':
                if self.args.disparity:
                    depth_path = os.path.join(
                        self.args.disparity_path, remove_leading_slash(
                            sample_path.split()[1])).replace('.png', '_disparity.png')

                    camera_path = os.path.join(self.args.camera_path, remove_leading_slash(
                            sample_path.split()[2]))
                else:
                    gt_path = self.args.gt_path_eval
                    depth_path = os.path.join(
                        gt_path, remove_leading_slash(
                            sample_path.split()[1]))
                    has_valid_depth = False
                
                try:
                    depth_gt = Image.open(depth_path)
                    has_valid_depth = True
                except IOError:
                    depth_gt = False
                    print('Missing gt for {}'.format(image_path))

                if has_valid_depth:
                    if self.args.dataset == 'cityscapes':
                        depth_gt = depth_gt.crop(
                            (left_margin,
                             top_margin,
                             left_margin +
                             2048,
                             top_margin +
                             1024))  # cropping to a size of 1216 x 352 according to this
                        #depth_gt = depth_gt.resize((int(depth_gt.width / 2), int(depth_gt.height / 2)))
                    depth_gt = np.asarray(depth_gt, dtype=np.float32)
                    # third dimension added to depth (to make it match the
                    # image which has three dimensions?)
                    depth_gt = np.expand_dims(depth_gt, axis=2)
                    
                    if self.args.dataset == 'nyu':
                        depth_gt = depth_gt / 1000.0
                    else:
                        if self.args.disparity:
                            with open(camera_path, 'r') as f:   # load camera parameters and convert to metres
                                camera_params = json.loads(f.read())
                                depth_gt[depth_gt > 0] = (depth_gt[depth_gt > 0] - 1) / 256
                                depth_gt[depth_gt > 0] = camera_params['extrinsic']['baseline'] * (
                                    (camera_params['intrinsic']['fx'] + camera_params['intrinsic']['fy']) / 2) / depth_gt[depth_gt > 0]
                                depth_gt[depth_gt == np.inf] = 0
                                depth_gt[depth_gt == np.nan] = 0
                        else:
                            depth_gt = depth_gt / 256.0  
                        
            if self.args.do_kb_crop is True:
                height = image.shape[0]
                width = image.shape[1]
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                image = image[top_margin:top_margin + 352,
                              left_margin:left_margin + 1216, :]
                if self.mode == 'online_eval' and has_valid_depth:
                    depth_gt = depth_gt[top_margin:top_margin +
                                        352, left_margin:left_margin + 1216, :]

            if self.mode == 'online_eval':
                sample = {
                    'image': image,
                    'depth': depth_gt,
                    'focal': focal,
                    'has_valid_depth': has_valid_depth,
                    'image_path': sample_path.split()[0],
                    'depth_path': sample_path.split()[1]}
            else:
                sample = {'image': image, 'focal': focal}

        if self.transform:  # after all preprocessing done on the image, pass the image to the ToTensor class
            sample = self.transform(sample)

        #print('sample shape', len(sample))
        #print( '\nsample image dimension shape', type(sample['image']), '\ndepth image dimension shape', type(sample['depth']) )
        return sample

    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def random_crop(self, img, depth, height, width):
        """
        Randomly crop the image and depth to the specified size.
        @param img - the image to crop
        @param depth - the depth to crop
        @param height - the height of the crop
        @param width - the width of the crop
        @return the cropped image and depth
        """

        # the assert statement makes sure the shape of the image being passed
        # is greater than the crop height requested, otherwise will raise an
        # exception
        assert img.shape[0] >= height
        # the assert statement makes sure the shape of the image being passed
        # is greater than the crop height requested,
        assert img.shape[1] >= width
        # assert makes sure again that the height of the image is the same as
        # the height of the depth image
        assert img.shape[0] == depth.shape[0]
        # assert makes sure again that the height of the image is the same as
        # the height of the depth image
        assert img.shape[1] == depth.shape[1]
        # pick a random starting point between 0 and (width-crop size), for
        # example, if you're random cropping a 1000px image to a size of 700px,
        # then you can start the crop anywhere from the 0th pixel all the way
        # to the 300th to maintain a cropped image size of 700px
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)  # same as above
        #print('image shape', img.shape)
        # is the last dimension number of images? so the entire batch, or is it
        # the colours?
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]
        return img, depth

    def train_preprocess(self, image, depth_gt):
        # Random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()

        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)

        return image, depth_gt

    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        if self.args.dataset == 'nyu':
            brightness = random.uniform(0.75, 1.25)
        else:
            brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug

    def __len__(self):
        return len(self.filenames)  # returns the length of the batch


class ToTensor(object):
    """
    Convert the image to a tensor. Also normalize the image.
    @param sample - the image itself
    @returns the image in a tensor format
    """

    def __init__(self, mode):
        self.mode = mode
        self.normalize = transforms.Normalize(
            mean=[
                0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225])  # loading the PyTorch normalize method into self.normalize

    def __call__(self, sample):
        image, focal = sample['image'], sample['focal']
        image = self.to_tensor(image)
        image = self.normalize(image)

        if self.mode == 'test':
            return {'image': image, 'focal': focal}

        depth = sample['depth']
        if self.mode == 'train':
            # depth gt doesn't get converted to a tensor above like image - so
            # if mode is train then depth also needs to be converted to a
            # tensor, but not normalized
            depth = self.to_tensor(depth)
            return {
                'image': image,
                'depth': depth,
                'focal': focal,
                'image_path': sample['image_path'],
                'depth_path': sample['depth_path']}
        else:  # if mode is online eval
            has_valid_depth = sample['has_valid_depth']
            return {
                'image': image,
                'depth': depth,
                'focal': focal,
                'has_valid_depth': has_valid_depth,
                'image_path': sample['image_path'],
                'depth_path': sample['depth_path']}

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        # handle PIL Image
        if pic.mode == 'I':
            # if it's a PIL image, convert to a numpy array before to a torch
            # tensor
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
