import math
import pdb
import numpy as np
import scipy.stats
import torch
import torch.nn.functional as F


class MaskGenerator (object):
    """
    Mask Generator
    """

    def generate_params(self, n_masks, mask_shape, rng=None):
        raise NotImplementedError('Abstract')

    def append_to_batch(self, *batch):
        x = batch[0]
        params = self.generate_params(len(x), x.shape[2:4])
        return batch + (params,)

    def torch_masks_from_params(self, t_params, mask_shape, torch_device):
        raise NotImplementedError('Abstract')


class BoxMaskGenerator (MaskGenerator):
    def __init__(
            self,
            prop_range,
            n_boxes=1,
            random_aspect_ratio=True,
            prop_by_area=True,
            within_bounds=True,
            invert=False):
        if isinstance(prop_range, float):
            prop_range = (prop_range, prop_range)
        self.prop_range = prop_range
        self.n_boxes = n_boxes
        self.random_aspect_ratio = random_aspect_ratio
        self.prop_by_area = prop_by_area
        self.within_bounds = within_bounds
        self.invert = invert

    def generate_depth_masks(
            self,
            n_masks,
            mask_shape,
            depths,
            uns_img_1,
            uns_img_2,
            rng=None):
        """
        Box masks can be generated quickly on the CPU so do it there.
        >>> boxmix_gen = BoxMaskGenerator((0.25, 0.25))
        >>> params = boxmix_gen.generate_params(256, (32, 32))
        >>> t_masks = boxmix_gen.torch_masks_from_params(params, (32, 32), 'cuda:0')
        :param n_masks: number of masks to generate (batch size)
        :param mask_shape: Mask shape as a `(height, width)` tuple
        :param rng: [optional] np.random.RandomState instance
        :return: masks: masks as a `(N, 1, H, W)` array
        """

        if rng is None:
            rng = np.random

        # Choose the proportion of each mask that should be above the threshold
        # shape of returned random floats is (n_boxes X n_masks - equal to
        # batch size (256 X 3)
        mask_props = rng.uniform(
            self.prop_range[0], self.prop_range[1], size=(
                n_masks, self.n_boxes))

        # Zeros will cause NaNs, so detect and suppres them
        zero_mask = mask_props == 0.0

        if self.random_aspect_ratio:
            # with DepthMix, bias is for the cut area to be larger in the
            # x-direction than y-direction since intention is to avoid areas
            # like the sky
            x_props = np.exp(
                rng.uniform(
                    low=1.0,
                    high=1.0,
                    size=(
                        n_masks,
                        self.n_boxes)) *
                np.log(mask_props))
            y_props = mask_props / x_props
        else:
            y_props = x_props = np.sqrt(mask_props)
        fac = np.sqrt(1.0 / self.n_boxes)  # reducing size of crops
        y_props *= fac
        x_props *= fac

        y_props[zero_mask] = 0
        x_props[zero_mask] = 0

        # proportions are converted to image/pixel sizes using the mask shape
        # parameter passed into the function initially
        sizes = np.round(np.stack([y_props, x_props], axis=2)
                         * np.array(mask_shape)[None, None, :])
        # sizes is now an array with the x,y sizes in pixels of the CutMix boxes for all images in the batch, in this case there are 3 boxes per image
        # [None, None, :] adds two extra dimensions at the start of the arrays so they become (1 x 1 x n_batch x no_of_boxes x 2) - 2 here is because the x and y were stacked together

        y_shape, x_shape = mask_shape
        # bias_shape = np.array((y_shape-200, x_shape))    #to enable effective
        # depth mixing, the top and bottom 100 pixels are exlcuded from the
        # crop, this is done by subtracting 200 from the allowed shape, then
        # adding 100 after the positions are generated.Therefore the max top
        # shape of the max is 100 from the top of the image, and the min is
        # always 100px from the bottom

        # for i in range(n_masks):
        #    sizes[i,:,0] = np.min((sizes[i,:,0], y_shape-200))

        #assert (bias_shape[0] >= sizes[:,:,0]).all()

        if self.within_bounds:

            # for y positions, bias is needed to avoid extremities of the image
            # since little value is achieved from sky and road augmentation (in
            # theory)
            positions = np.round(
                (mask_shape -
                 sizes) *
                rng.uniform(
                    low=0.0,
                    high=1.0,
                    size=sizes.shape))
            # positions[:,:,0] = positions[:,:,0]+100     #100 pix offset added
            # to the positions of the box
            rectangles = np.append(positions, positions + sizes, axis=2)

        else:
            centres = np.round(
                np.array(mask_shape) *
                rng.uniform(
                    low=0.0,
                    high=1.0,
                    size=sizes.shape))
            rectangles = np.append(
                centres - sizes * 0.5,
                centres + sizes * 0.5,
                axis=2)

        self.invert = True

        if self.invert:
            masks = np.zeros((n_masks, 1) + mask_shape)
        else:
            masks = np.ones((n_masks, 1) + mask_shape)

        for i, sample_rectangles in enumerate(
                rectangles):  # enumerate returns a loop iterable that contains both the value and the index of the returned iterable
            for y0, x0, y1, x1 in sample_rectangles:  # get initial mask for entire area
                masks[i, 0, int(y0):int(y1), int(x0):int(x1)] = 1 - \
                    masks[i, 0, int(y0):int(y1), int(x0):int(x1)]

        d_mask = np.zeros_like(masks)
        epsilon = 0.1

        for i in range(n_masks):
            k = uns_img_2[i, 3, ...] < (uns_img_1[i, 3, ...] + epsilon)
            d_mask[i, 0, :, :] = k.cpu().numpy()

        combined_masks = np.logical_and(masks, d_mask)

        return combined_masks

    def generate_params(self, n_masks, mask_shape, rng=None):
        """
        Box masks can be generated quickly on the CPU so do it there.
        >>> boxmix_gen = BoxMaskGenerator((0.25, 0.25))
        >>> params = boxmix_gen.generate_params(256, (32, 32))
        >>> t_masks = boxmix_gen.torch_masks_from_params(params, (32, 32), 'cuda:0')
        :param n_masks: number of masks to generate (batch size)
        :param mask_shape: Mask shape as a `(height, width)` tuple
        :param rng: [optional] np.random.RandomState instance
        :return: masks: masks as a `(N, 1, H, W)` array
        """
        if rng is None:
            rng = np.random

        if self.prop_by_area:
            # Choose the proportion of each mask that should be above the
            # threshold
            # shape of returned random floats is (n_boxes X n_masks - equal to
            # batch size (256 X 3)
            mask_props = rng.uniform(
                self.prop_range[0], self.prop_range[1], size=(
                    n_masks, self.n_boxes))

            # Zeros will cause NaNs, so detect and suppres them
            zero_mask = mask_props == 0.0

            if self.random_aspect_ratio:
                y_props = np.exp(
                    rng.uniform(
                        low=0.0,
                        high=1.0,
                        size=(
                            n_masks,
                            self.n_boxes)) *
                    np.log(mask_props))
                x_props = mask_props / y_props
            else:
                y_props = x_props = np.sqrt(mask_props)
            fac = np.sqrt(1.0 / self.n_boxes)  # reducing size of crops
            y_props *= fac
            x_props *= fac

            y_props[zero_mask] = 0
            x_props[zero_mask] = 0
        else:
            if self.random_aspect_ratio:
                y_props = rng.uniform(
                    self.prop_range[0], self.prop_range[1], size=(
                        n_masks, self.n_boxes))
                x_props = rng.uniform(
                    self.prop_range[0], self.prop_range[1], size=(
                        n_masks, self.n_boxes))
            else:
                x_props = y_props = rng.uniform(
                    self.prop_range[0], self.prop_range[1], size=(
                        n_masks, self.n_boxes))
            fac = np.sqrt(1.0 / self.n_boxes)
            y_props *= fac
            x_props *= fac

        # proportions are converted to image/pixel sizes using the mask shape
        # parameter passed into the function initially
        sizes = np.round(np.stack([y_props, x_props], axis=2)
                         * np.array(mask_shape)[None, None, :])
        # sizes is now an array with the x,y sizes in pixels of the CutMix boxes for all images in the batch, in this case there are 3 boxes per image
        # [None, None, :] adds two extra dimensions at the start of the arrays so they become (1 x 1 x n_batch x no_of_boxes x 2) - 2 here is because the x and y were stacked together

        if self.within_bounds:
            positions = np.round(
                (np.array(mask_shape) -
                 sizes) *
                rng.uniform(
                    low=0.0,
                    high=1.0,
                    size=sizes.shape))
            rectangles = np.append(positions, positions + sizes, axis=2)
        else:
            centres = np.round(
                np.array(mask_shape) *
                rng.uniform(
                    low=0.0,
                    high=1.0,
                    size=sizes.shape))
            rectangles = np.append(
                centres - sizes * 0.5,
                centres + sizes * 0.5,
                axis=2)

        if self.invert:
            masks = np.zeros((n_masks, 1) + mask_shape)
        else:
            masks = np.ones((n_masks, 1) + mask_shape)
        for i, sample_rectangles in enumerate(
                rectangles):  # enumerate returns a loop iterable that contains both the value and the index of the returned iterable
            for y0, x0, y1, x1 in sample_rectangles:
                masks[i, 0, int(y0):int(y1), int(x0):int(x1)] = 1 - \
                    masks[i, 0, int(y0):int(y1), int(x0):int(x1)]
        return masks

    def torch_masks_from_params(self, t_params, mask_shape, torch_device):
        return t_params


class AddMaskParamsToBatch (object):
    """
    We add the cut-and-paste parameters to the mini-batch within the collate function,
    (we pass it as the `batch_aug_fn` parameter to the `SegCollate` constructor)
    as the collate function pads all samples to a common size
    """

    def __init__(self, mask_gen):
        self.mask_gen = mask_gen

    def __call__(self, batch):
        sample = batch[0]
        # returns shape of first and second dimension (h x w) or (w x h)?
        mask_size = sample['data'].shape[1:3]
        params = self.mask_gen.generate_params(len(batch), mask_size)
        for sample, p in zip(batch, params):
            sample['mask_params'] = p.astype(
                np.float32)  # mask params appended to the batch
        return batch
