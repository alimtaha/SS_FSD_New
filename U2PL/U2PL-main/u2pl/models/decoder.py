import torch
import torch.nn as nn
from torch.nn import functional as F

from .base import ASPP, get_syncbn


class dec_deeplabv3(nn.Module):
    def __init__(
        self,
        in_planes,
        num_classes=19,
        inner_planes=256,
        sync_bn=False,
        dilations=(12, 18, 24),
    ):
        super(dec_deeplabv3, self).__init__()

        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm2d

        self.aspp = ASPP(
            in_planes,
            inner_planes=inner_planes,
            sync_bn=sync_bn,
            dilations=dilations)
        self.head = nn.Sequential(
            nn.Conv2d(
                self.aspp.get_outplanes(),
                256,
                kernel_size=3,
                padding=1,
                dilation=1,
                bias=False,
            ),
            norm_layer(256),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(0.1),
            nn.Conv2d(
                256,
                num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True),
        )

    def forward(self, x):
        aspp_out = self.aspp(x)
        res = self.head(aspp_out)
        return res


class dec_deeplabv3_plus(nn.Module):
    def __init__(
        self,
        in_planes,
        num_classes=19,
        inner_planes=256,
        sync_bn=False,
        dilations=(12, 18, 24),
        rep_head=True,
    ):
        super(dec_deeplabv3_plus, self).__init__()

        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm2d
        self.rep_head = rep_head

        self.low_conv = nn.Sequential(
            nn.Conv2d(
                256, 256, kernel_size=1), norm_layer(256), nn.ReLU(
                inplace=True))

        self.aspp = ASPP(
            in_planes,
            inner_planes=inner_planes,
            sync_bn=sync_bn,
            dilations=dilations)

        self.head = nn.Sequential(
            nn.Conv2d(
                self.aspp.get_outplanes(),
                256,
                kernel_size=1,  # changed from 3 to a 1 x 1 conv
                padding=0,  # changed from 1
                dilation=1,  # changed from 1
                bias=False,
            ),
            # norm_layer(256),
            nn.LeakyReLU(inplace=True),
            # nn.Dropout2d(0.1),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=True),
            norm_layer(256),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(0.1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            norm_layer(256),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(0.1),
            nn.Conv2d(
                256,
                num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True),
        )

        if self.rep_head:

            self.representation = nn.Sequential(
                nn.Conv2d(
                    512,
                    256,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True),
                norm_layer(256),
                nn.ReLU(inplace=True),
                # nn.Dropout2d(0.1),
                nn.Conv2d(
                    256,
                    256,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True),
                norm_layer(256),
                nn.ReLU(inplace=True),
                # nn.Dropout2d(0.1),
                nn.Conv2d(
                    256,
                    256,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True),
            )

    def forward(self, x):
        x1, x2, x3, x4 = x
        aspp_out, _ = self.aspp(x4)
        low_feat = self.low_conv(x1)
        print(self.aspp.get_outplanes())
        print(aspp_out.shape)
        # this is the same as the red_conv in CPS - but done in decoder head vs
        # ASPP module un CPS
        aspp_out = self.head(aspp_out)
        h, w = low_feat.size()[-2:]
        aspp_out = F.interpolate(
            aspp_out, size=(h, w), mode="bilinear", align_corners=True
        )
        aspp_out = torch.cat((low_feat, aspp_out), dim=1)

        res = {"pred": self.classifier(aspp_out)}

        if self.rep_head:
            res["rep"] = self.representation(aspp_out)

        return res


class dec_deeplabv3_plus_depth(nn.Module):
    def __init__(
        self,
        in_planes,
        num_classes=19,
        inner_planes=256,
        sync_bn=False,
        dilations=(12, 18, 24),
        rep_head=True,
    ):
        super(dec_deeplabv3_plus_depth, self).__init__()

        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm2d
        self.rep_head = rep_head

        self.low_conv = nn.Sequential(
            nn.Conv2d(
                256, 256, kernel_size=1), norm_layer(256), nn.ReLU(
                inplace=True))

        self.aspp = ASPP(
            in_planes,
            inner_planes=inner_planes,
            sync_bn=sync_bn,
            dilations=dilations)

        self.head = nn.Sequential(
            nn.Conv2d(
                self.aspp.get_outplanes(),
                256,
                kernel_size=1,  # changed from 3 to a 1 x 1 conv
                padding=0,  # changed from 1
                dilation=1,  # changed from 1
                bias=False,
            ),
            # norm_layer(256),
            nn.LeakyReLU(inplace=True),
            # nn.Dropout2d(0.1),
        )

        self.depth_head = nn.Sequential(
            nn.Conv2d(
                self.aspp.get_outplanes(),
                256,
                kernel_size=1,
                padding=0,
                dilation=1,
                bias=False,
            ),
            # norm_layer(256),
            nn.LeakyReLU(inplace=True),
            # nn.Dropout2d(0.1),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(
                768,
                256,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            norm_layer(256),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(0.1),
            nn.Conv2d(
                256,
                256,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            norm_layer(256),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(0.1),
            nn.Conv2d(
                256,
                num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True),
        )

        if self.rep_head:

            self.representation = nn.Sequential(
                nn.Conv2d(
                    768,
                    256,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True),
                norm_layer(256),
                nn.ReLU(inplace=True),
                # nn.Dropout2d(0.1),
                nn.Conv2d(
                    256,
                    256,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True),
                norm_layer(256),
                nn.ReLU(inplace=True),
                # nn.Dropout2d(0.1),
                nn.Conv2d(
                    256,
                    256,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True),
            )

    def forward(self, x, d):
        x1, x2, x3, x4 = x
        aspp_out, d_out = self.aspp(x4, d)
        low_feat = self.low_conv(x1)
        # this is the same as the red_conv in CPS - but done in decoder head vs
        # ASPP module un CPS
        aspp_out = self.head(aspp_out)
        d_out = self.depth_head(d_out)
        h, w = low_feat.size()[-2:]
        aspp_out = F.interpolate(
            aspp_out, size=(h, w), mode="bilinear", align_corners=True
        )

        d_out = F.interpolate(
            d_out, size=(h, w), mode="bilinear", align_corners=True
        )

        aspp_out = torch.cat((low_feat, d_out, aspp_out), dim=1)

        res = {"pred": self.classifier(aspp_out)}

        if self.rep_head:
            res["rep"] = self.representation(aspp_out)

        return res
