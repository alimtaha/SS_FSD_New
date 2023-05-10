# encoding: utf-8

from furnace.seg_opr.loss_opr import SigmoidFocalLoss, ProbOhemCrossEntropy2d, bce2d
from furnace.engine.engine import Engine
from furnace.engine.lr_policy import WarmUpPolyLR
from furnace.utils.init_func import init_weight, group_weight
from dataloader_depth_concat import CityScape, get_train_loader
from furnace.base_model import resnet50
from collections import OrderedDict
from functools import partial
from config import config
import torch.nn.functional as F
import torch.nn as nn
import torch
import sys
import path

sys.path.append('../../../')

#from furnace.seg_opr.sync_bn import DataParallelModel, Reduce, BatchNorm2d


class NetworkFullResnet(nn.Module):
    def __init__(
            self,
            num_classes,
            criterion,
            norm_layer,
            pretrained_model=None,
            depth_only=False,
            **kwargs):
        super(NetworkFullResnet, self).__init__()
        

        self.branch1 = SingleNetworkFullResnet(
            num_classes, criterion, norm_layer, pretrained_model, full_depth_resnet=True, depth_only=depth_only)
        self.branch2 = SingleNetworkFullResnet(
            num_classes, criterion, norm_layer, pretrained_model, full_depth_resnet=True, depth_only=depth_only)

    def forward(self, data, step=1, feat=False):
        if not self.training:
            pred1 = self.branch1(data)
            return pred1

        if step == 1:
            return self.branch1(data)
        elif step == 2:
            return self.branch2(data)


class Network(nn.Module):
    def __init__(
            self,
            num_classes,
            criterion,
            norm_layer,
            pretrained_model=None,
            depth_only=False,
            **kwargs):
        super(Network, self).__init__()


        self.branch1 = SingleNetwork(
            num_classes, criterion, norm_layer, pretrained_model, full_depth_resnet=False, depth_only=depth_only)
        self.branch2 = SingleNetwork(
            num_classes, criterion, norm_layer, pretrained_model, full_depth_resnet=False, depth_only=depth_only)

    def forward(self, data, step=1, feat=False):
        if not self.training:
            pred1 = self.branch1(data)
            return pred1

        if step == 1:
            return self.branch1(data)
        elif step == 2:
            return self.branch2(data)


class SingleNetwork(nn.Module):
    def __init__(
            self,
            num_classes,
            criterion,
            norm_layer,
            pretrained_model=None,
            full_depth_resnet=False,
            depth_only=False):
        super(SingleNetwork, self).__init__()
        self.full_depth_resnet = full_depth_resnet
        self.depth_only = depth_only
        
        print('Full Depth ResNet Status: ', full_depth_resnet)
        self.backbone = resnet50(pretrained_model, norm_layer=norm_layer,
                                 bn_eps=config.bn_eps,
                                 bn_momentum=config.bn_momentum,
                                 deep_stem=True, stem_width=64, inplace=False)

        # self.conv_depth = nn.Sequential(nn.Conv2d(1,256,3), nn.BatchNorm2d(256,bn_eps=config.bn_eps,
        #                          bn_momentum=config.bn_momentum),nn.LeakyReLU)

        self.stem_width = 64

        self.dilate = 2
        for m in self.backbone.layer4.children():
            m.apply(partial(self._nostride_dilate, dilate=self.dilate))  # the apply function only takes in references to functions and then calls the function after, it expects whatever function is passed to it to have only one parameter, m, as the model. It then calls the function with m as the parameter. The reason partial is used is because the _nostride_dilate function has more than one input argument, so to get over that, the partial class takes in a function, and a partial subset of it's input arguments, and wraps a new function around them where the new function's input parameters are what remains (what wasn't defined in the partial call). So the apply function only sees a function with one input argument, m, so it works.
            self.dilate *= 2  # this explains the apply fucntion https://stackoverflow.com/questions/55613518/how-does-the-applyfn-function-in-pytorch-work-with-a-function-without-return-s

        self.head = Head(num_classes, norm_layer, config.bn_momentum, full_depth_resnet)
        self.business_layer = []
        self.business_layer.append(self.head)
        self.criterion = criterion

        # num_classes is number of classes we're predicting, the input to the
        # classifier is a 256 channel feature map
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1, bias=True)
        self.business_layer.append(self.classifier)

    def forward(self, data):
        blocks = self.backbone(data[:, :3, :, :])
        depth = data[:, 3:, :, :]                           #Extracting Depth Maps only from Image Input (4th channel)
        
        v3plus_feature, feat = self.head(blocks, depth)      # (b, c, h, w)
        b, c, h, w = v3plus_feature.shape

        pred = self.classifier(v3plus_feature)

        b, c, h, w = data.shape
        pred = F.interpolate(
            pred,
            size=(
                h,
                w),
            mode='bilinear',
            align_corners=True)

        if self.training:
            return v3plus_feature, pred
        return pred, feat, v3plus_feature #v3_plus is output of concatenated maps + conv, feat is pre conv

    # @staticmethod
    def _nostride_dilate(self, m, dilate):
        if isinstance(m, nn.Conv2d):
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

class SingleNetworkFullResnet(nn.Module):
    def __init__(
            self,
            num_classes,
            criterion,
            norm_layer,
            pretrained_model=None,
            full_depth_resnet=False,
            depth_only=False):
        super(SingleNetworkFullResnet, self).__init__()
        self.full_depth_resnet = full_depth_resnet
        self.depth_only = depth_only
        
        print('Full Depth ResNet Status: ', full_depth_resnet)
        self.backbone = resnet50(pretrained_model, norm_layer=norm_layer,
                                 bn_eps=config.bn_eps,
                                 bn_momentum=config.bn_momentum,
                                 deep_stem=True, stem_width=64, inplace=False)

        # self.conv_depth = nn.Sequential(nn.Conv2d(1,256,3), nn.BatchNorm2d(256,bn_eps=config.bn_eps,
        #                          bn_momentum=config.bn_momentum),nn.LeakyReLU)
        self.dilate = 2
        for m in self.backbone.layer4.children():
            m.apply(partial(self._nostride_dilate, dilate=self.dilate))  # the apply function only takes in references to functions and then calls the function after, it expects whatever function is passed to it to have only one parameter, m, as the model. It then calls the function with m as the parameter. The reason partial is used is because the _nostride_dilate function has more than one input argument, so to get over that, the partial class takes in a function, and a partial subset of it's input arguments, and wraps a new function around them where the new function's input parameters are what remains (what wasn't defined in the partial call). So the apply function only sees a function with one input argument, m, so it works.
            self.dilate *= 2  # this explains the apply fucntion https://stackoverflow.com/questions/55613518/how-does-the-applyfn-function-in-pytorch-work-with-a-function-without-return-s

        self.stem_width = 64

        if full_depth_resnet == True:
            self.depth_backbone = resnet50(pretrained_model, norm_layer=norm_layer,
                                 bn_eps=config.bn_eps,
                                 bn_momentum=config.bn_momentum,
                                 deep_stem=True, stem_width=64, inplace=False)
            
            self.depth_backbone.conv1 = nn.Sequential(
            nn.Conv2d(1, self.stem_width, kernel_size=3, stride=2, padding=1,
                      bias=False),
            norm_layer(self.stem_width, eps=config.bn_eps, momentum=config.bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.stem_width, self.stem_width, kernel_size=3, stride=1,
                      padding=1,
                      bias=False),
            norm_layer(self.stem_width, eps=config.bn_eps, momentum=config.bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.stem_width, self.stem_width * 2, kernel_size=3, stride=1,
                      padding=1,
                      bias=False),
            )

            self.dilate = 2
            for m in self.depth_backbone.layer4.children():
                m.apply(partial(self._nostride_dilate, dilate=self.dilate))  # the apply function only takes in references to functions and then calls the function after, it expects whatever function is passed to it to have only one parameter, m, as the model. It then calls the function with m as the parameter. The reason partial is used is because the _nostride_dilate function has more than one input argument, so to get over that, the partial class takes in a function, and a partial subset of it's input arguments, and wraps a new function around them where the new function's input parameters are what remains (what wasn't defined in the partial call). So the apply function only sees a function with one input argument, m, so it works.
                self.dilate *= 2  # this explains the apply fucntion https://stackoverflow.com/questions/55613518/how-does-the-applyfn-function-in-pytorch-work-with-a-function-without-return-s

        self.head = Head(num_classes, norm_layer, config.bn_momentum, full_depth_resnet, depth_only)
        self.business_layer = []
        self.business_layer.append(self.head)
        self.criterion = criterion

        # num_classes is number of classes we're predicting, the input to the
        # classifier is a 256 channel feature map
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1, bias=True)
        self.business_layer.append(self.classifier)

    def forward(self, data):
        if self.depth_only == False:
            blocks = self.backbone(data[:, :3, :, :])
        depth = data[:, 3:, :, :]                           #Extracting Depth Maps only from Image Input (4th channel)
        
        if self.full_depth_resnet == True:
            depth_blocks = self.depth_backbone(depth)
            if self.depth_only:
                v3plus_feature, feat = self.head(None, depth_blocks)
            else:
                v3plus_feature, feat = self.head(blocks, depth_blocks)
        else:
            if self.depth_only:
                v3plus_feature, feat = self.head(None, depth)
            else:
                v3plus_feature, feat = self.head(blocks, depth)      # (b, c, h, w)
        
        b, c, h, w = v3plus_feature.shape

        pred = self.classifier(v3plus_feature)

        b, c, h, w = data.shape
        pred = F.interpolate(
            pred,
            size=(
                h,
                w),
            mode='bilinear',
            align_corners=True)

        if self.training:
            return v3plus_feature, pred
        return pred, feat, v3plus_feature #v3_plus is output of concatenated maps + conv, feat is pre conv

    # @staticmethod
    def _nostride_dilate(self, m, dilate):
        if isinstance(m, nn.Conv2d):
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)


class ASPP(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dilation_rates=(12, 24, 36),
                 hidden_channels=256,
                 norm_act=nn.BatchNorm2d,
                 pooling_size=None,
                 full_depth_resnet=False,
                 depth_only=False):
        super(ASPP, self).__init__()
        self.pooling_size = pooling_size
        self.full_depth_resnet = full_depth_resnet
        self.depth_only = depth_only
        # add dilated convs for depth, change  last conv number of input dimensions to suit
        # add initial conv to append to low level features, both depth streams
        # will be appended to the same areas

        self.depth_downsample = nn.Sequential(      #Depth change, downsample
            nn.Conv2d(
                1,
                in_channels,
                kernel_size=7,
                stride=4,
                padding=2,
                bias=False),
            norm_act(in_channels),
            nn.LeakyReLU())

        self.depth_map_convs = nn.ModuleList([      #Depth change, add conv
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilation_rates[0],
                      padding=dilation_rates[0]),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilation_rates[1],
                      padding=dilation_rates[1]),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilation_rates[2],
                      padding=dilation_rates[2])
        ])

        self.pool_depth = nn.Sequential(            #Depth change - pooling
            nn.AdaptiveAvgPool2d(
                (1, 1)), nn.Conv2d(
                in_channels, hidden_channels, kernel_size=1, bias=False))

        self.pool_u2pl = nn.Sequential(
            nn.AdaptiveAvgPool2d(
                (1, 1)), nn.Conv2d(
                in_channels, hidden_channels, kernel_size=1, bias=False))

        self.map_convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels,
                    hidden_channels,
                    1,
                    bias=False),
                nn.Conv2d(
                    in_channels,
                    hidden_channels,
                    3,
                    bias=False,
                    dilation=dilation_rates[0],
                    padding=dilation_rates[0]),
                nn.Conv2d(
                    in_channels,
                    hidden_channels,
                    3,
                    bias=False,
                    dilation=dilation_rates[1],
                    padding=dilation_rates[1]),
                nn.Conv2d(
                    in_channels,
                    hidden_channels,
                    3,
                    bias=False,
                    dilation=dilation_rates[2],
                    padding=dilation_rates[2])])
        self.map_bn = norm_act(hidden_channels * 5)

        self.global_pooling_conv = nn.Conv2d(
            in_channels, hidden_channels, 1, bias=False)
        self.global_pooling_bn = norm_act(hidden_channels)

        self.red_conv = nn.Conv2d(
            hidden_channels * 5, out_channels, 1, bias=False)
        #if full_depth_resnet:
        self.depth_red_conv = nn.Conv2d(
            hidden_channels * 5, out_channels, 1, bias=False)
        #unused (pool red conv)
        self.pool_red_conv = nn.Conv2d(
            hidden_channels, out_channels, 1, bias=False)
        self.red_bn = norm_act(out_channels)

        self.leak_relu = nn.LeakyReLU()

    def forward(self, x, d):
        # Map convolutions
        if self.depth_only == False:
            _, _, h, w = x.size()
            out1 = F.interpolate(
                self.pool_u2pl(x), size=(h, w), mode="bilinear", align_corners=True
            )
            aspp_list = [m(x) for m in self.map_convs]
            aspp_list.insert(0, out1)
            out = torch.cat(aspp_list, dim=1)
            out = self.map_bn(out)
            out = self.leak_relu(out)       # add activation layer
            out = self.red_conv(out)
            out = self.leak_relu(out)
        else:
            out = None

        #Depth operations - downsample, bilinear, aspp, concat and then activation functions
        if self.full_depth_resnet == False:
            d = self.depth_downsample(d)
        _, _, h_d, w_d = d.size()
        depth_1 = F.interpolate(self.pool_depth(d), size=(
            h_d, w_d), mode="bilinear", align_corners=True)
        depth_aspp = [m(d) for m in self.depth_map_convs]
        depth_aspp.insert(0, depth_1)
        depth_aspp = torch.cat(depth_aspp, dim=1)
        depth = self.map_bn(depth_aspp)
        depth = self.leak_relu(depth)       # add activation layer
        #if self.full_depth_resnet:
        depth = self.depth_red_conv(depth)
        #else:
        #    depth = self.red_conv(depth)
        depth = self.leak_relu(depth)

        # Global pooling - Pooling changed to match U2PL - added to map convs
        #pool = self._global_pooling(x)
        #pool = self.global_pooling_conv(pool)
        #pool = self.global_pooling_bn(pool)

        # pool = self.leak_relu(pool)  # add activation layer

        #pool = self.pool_red_conv(pool)
        # if self.training or self.pooling_size is None:
        #    pool = pool.repeat(1, 1, x.size(2), x.size(3))

        # out = out #+ pool
        #out = self.red_bn(out)
        # out = self.leak_relu(out)  # add activation layer
        return out, depth

    def _global_pooling(self, x):
        if self.training or self.pooling_size is None:
            pool = x.view(x.size(0), x.size(1), -1).mean(dim=-1)
            pool = pool.view(x.size(0), x.size(1), 1, 1)
        else:  # pooling size is always none so does not play a role here
            pooling_size = (min(try_index(self.pooling_size, 0), x.shape[2]),
                            min(try_index(self.pooling_size, 1), x.shape[3]))
            padding = (
                (pooling_size[1] -
                 1) //
                2,
                (pooling_size[1] -
                 1) //
                2 if pooling_size[1] %
                2 == 1 else (
                    pooling_size[1] -
                    1) //
                2 +
                1,
                (pooling_size[0] -
                 1) //
                2,
                (pooling_size[0] -
                 1) //
                2 if pooling_size[0] %
                2 == 1 else (
                    pooling_size[0] -
                    1) //
                2 +
                1)

            pool = nn.functional.avg_pool2d(x, pooling_size, stride=1)
            pool = nn.functional.pad(pool, pad=padding, mode="replicate")
        return pool


class Head(nn.Module):
    def __init__(
            self,
            classify_classes,
            norm_act=nn.BatchNorm2d,
            bn_momentum=0.0003,
            full_depth_resnet=False,
            depth_only=False):
        super(Head, self).__init__()

        self.full_depth_resnet = full_depth_resnet
        self.depth_only = depth_only
        
        self.classify_classes = classify_classes
        self.aspp = ASPP(2048, 256, [12, 18, 24], norm_act=norm_act, full_depth_resnet=full_depth_resnet, depth_only=depth_only)

        self.reduce = nn.Sequential(
            nn.Conv2d(256, 256, 1, bias=False),
            norm_act(256, momentum=bn_momentum),
            nn.ReLU(),
        )

        if full_depth_resnet:
            self.depth_reduce = nn.Sequential(
            nn.Conv2d(256, 256, 1, bias=False),
            norm_act(256, momentum=bn_momentum),
            nn.ReLU(),
            )
        
        #In channels are 768 vs normal 512 since concatenating backbone features, aspp image features and depth aspp features
        if full_depth_resnet:
            if self.depth_only:
                d_channels = 512
            else:
                d_channels = 1024
        else:
            if self.depth_only:
                d_channels = 256
            else:
                d_channels = 768

        self.last_conv = nn.Sequential(
            nn.Conv2d(
                d_channels, 256, kernel_size=3, stride=1, padding=1, bias=False), norm_act(
                256, momentum=bn_momentum), nn.ReLU(), nn.Conv2d(
                256, 256, kernel_size=3, stride=1, padding=1, bias=False), norm_act(
                    256, momentum=bn_momentum), nn.ReLU(), )

    def forward(self, f_list, depth):

        if self.depth_only == False:
            f = f_list[-1]

        if self.full_depth_resnet == True:
            depth1 = depth[-1]
            if self.depth_only:
                _, depth1 = self.aspp(None, depth1)
            else:
                f1, depth1 = self.aspp(f, depth1)
        else:
            if self.depth_only:
                _, depth1 = self.aspp(None, depth)
            else:
                f1, depth1 = self.aspp(f, depth)
            
        
        if self.depth_only == False:
            low_level_features = f_list[0]
            low_h, low_w = low_level_features.size(2), low_level_features.size(3)
            low_level_features = self.reduce(low_level_features)
            f2 = F.interpolate(
                f1,
                size=(
                    low_h,
                    low_w),
                mode='bilinear',
                align_corners=True)

        if self.full_depth_resnet == True:
            depth_low_level  = depth[0]
            d_low_h, d_low_w = depth_low_level.size(2), depth_low_level.size(3)
            depth_low_level = self.depth_reduce(depth_low_level)
            d2 = F.interpolate(
                    depth1,
                    size=(
                    d_low_h,
                    d_low_w),
                    mode='bilinear',
                    align_corners=True)
            if self.depth_only:
                f3 = torch.cat((d2, depth_low_level), dim=1)
                embed_feat = d2
            else:
                f3 = torch.cat((f2, d2, low_level_features, depth_low_level), dim=1)
                embed_feat = torch.concat((f2, d2), dim=1)
        else:
            if self.depth_only:
                f3 = depth1
                embed_feat = depth1
            else:
                f3 = torch.cat((f2, depth1, low_level_features), dim=1)
                embed_feat = torch.concat((f2, depth1), dim=1) #embed feats is aspp and depth aspp feats concatenated
          
        f4 = self.last_conv(f3)

        return f4, embed_feat


def count_params(model):
    return sum(p.numel() for p in model.parameters())  # if p.requires_grad)


if __name__ == '__main__':

    criterion = ProbOhemCrossEntropy2d(ignore_label=255, thresh=0.7,
                                       min_kept=50000, use_weight=False)

    model = Network(2, pretrained_model=None, criterion=criterion,  # change number of classes to free space only, 2 classes since freespace and non free space,
                    norm_layer=nn.BatchNorm2d)

    # print(model)

    # for module in model.branch1.modules():
    #    print(f"Module:", module, "\n")

    print("Number of Params", count_params(model))

    '''
    model = Network(40, criterion=nn.CrossEntropyLoss(),
                    pretrained_model=None,
                    norm_layer=nn.BatchNorm2d)
    left = torch.randn(2, 3, 128, 128)
    right = torch.randn(2, 3, 128, 128)

    print(model.backbone)

    out = model(left)
    print(out.shape)
    '''
