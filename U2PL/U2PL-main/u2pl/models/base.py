import torch
import torch.nn as nn
from torch.nn import functional as F


def get_syncbn():
    # Due to not using distributed compute, this is uncommented and the below
    # is commented
    return nn.BatchNorm2d
    # return nn.SyncBatchNorm


class ASPP(nn.Module):
    """
    Reference:
        Chen, Liang-Chieh, et al. *"Rethinking Atrous Convolution for Semantic Image Segmentation."*
    """

    def __init__(
        self,
        in_planes,
        inner_planes=256,
        sync_bn=False,
        dilations=(
            12,
            24,
            36)):
        super(ASPP, self).__init__()

        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm2d
        self.conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(
                in_planes,
                inner_planes,
                kernel_size=1,
                padding=0,
                dilation=1,
                bias=False,
            ),
            # norm_layer(inner_planes),
            # nn.LeakyReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_planes,
                inner_planes,
                kernel_size=1,
                padding=0,
                dilation=1,  # changed to 0
                bias=False,
            ),
            # norm_layer(inner_planes),
            # nn.LeakyReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_planes,
                inner_planes,
                kernel_size=3,
                padding=dilations[0],
                dilation=dilations[0],
                bias=False,
            ),
            # norm_layer(inner_planes),
            # nn.LeakyReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_planes,
                inner_planes,
                kernel_size=3,
                padding=dilations[1],
                dilation=dilations[1],
                bias=False,
            ),
            # norm_layer(inner_planes),
            # nn.LeakyReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_planes,
                inner_planes,
                kernel_size=3,
                padding=dilations[2],
                dilation=dilations[2],
                bias=False,
            ),
            # norm_layer(inner_planes),
            # nn.LeakyReLU(inplace=True),
        )

        self.depth_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(
                (1, 1)), nn.Conv2d(
                256, inner_planes, kernel_size=1, bias=False))

        self.out_planes = (len(dilations) + 2) * inner_planes

        self.leak_relu = nn.LeakyReLU()

        self.map_bn = norm_layer(self.out_planes)

        self.depth_downsample = nn.Sequential(
            nn.Conv2d(
                1,
                256,
                kernel_size=7,
                stride=4,
                padding=2,
                bias=False),
            norm_layer(256),
            nn.LeakyReLU())

        self.depth_map_convs = nn.ModuleList([
            nn.Conv2d(256, inner_planes, 1, bias=False),
            nn.Conv2d(256, inner_planes, 3, bias=False, dilation=dilations[0],
                      padding=dilations[0]),
            nn.Conv2d(256, inner_planes, 3, bias=False, dilation=dilations[1],
                      padding=dilations[1]),
            nn.Conv2d(256, inner_planes, 3, bias=False, dilation=dilations[2],
                      padding=dilations[2])
        ])

        # the final convolution ins ASPP is done in the decoder head unlike CPS
        # - self.red_conv = nn.Conv2d(inner_planes * 4, out_channels, 1,
        # bias=False)

    def get_outplanes(self):
        return self.out_planes

    def forward(self, x, d=None):
        _, _, h, w = x.size()
        feat1 = F.interpolate(
            self.conv1(x), size=(h, w), mode="bilinear", align_corners=True
        )
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        aspp_out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)
        aspp_out = self.map_bn(aspp_out)
        aspp_out = self.leak_relu(aspp_out)

        if d is not None:
            d = self.depth_downsample(d)
            _, _, h_d, w_d = d.size()
            depth_1 = F.interpolate(self.depth_pool(d), size=(
                h_d, w_d), mode="bilinear", align_corners=True)
            depth_aspp = [m(d) for m in self.depth_map_convs]
            depth_aspp.insert(0, depth_1)
            depth_aspp = torch.cat(depth_aspp, dim=1)
            depth = self.map_bn(depth_aspp)
            depth = self.leak_relu(depth)
        else:
            depth = None

        return aspp_out, depth
