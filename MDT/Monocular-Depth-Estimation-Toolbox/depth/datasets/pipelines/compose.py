# Copyright (c) OpenMMLab. All rights reserved.
import collections
from ..builder import PIPELINES
from mmcv.utils import build_from_cfg
#from matplotlib import pyplot as plt


@PIPELINES.register_module()
class Compose(object):
    """Compose multiple transforms sequentially.

    Args:
        transforms (Sequence[dict | callable]): Sequence of transform object or
            config dict to be composed.
    """

    def __init__(self, transforms):
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')

    def __call__(self, data):
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """

        for t in self.transforms:
            data = t(data)
            if data is None:
                return None

        #fig = plt.figure()
        #plt.imshow(data['depth_gt']._data[0].squeeze(), cmap = 'magma_r')
        # fig.suptitle('Compose')
        # plt.colorbar()
        # plt.show()
        # print(data.keys())
        # print(data['depth_gt']._data[0].squeeze().shape)

        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string
