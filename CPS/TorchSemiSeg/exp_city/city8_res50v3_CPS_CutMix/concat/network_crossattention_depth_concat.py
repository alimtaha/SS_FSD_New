# encoding: utf-8

from furnace.seg_opr.loss_opr import SigmoidFocalLoss, ProbOhemCrossEntropy2d, bce2d
from furnace.engine.engine import Engine
from furnace.engine.lr_policy import WarmUpPolyLR
from furnace.utils.init_func import init_weight, group_weight
from furnace.base_model import resnet50
from collections import OrderedDict
from functools import partial
from exp_city.city8_res50v3_CPS_CutMix.config import config as conzeft
import torch.nn.functional as F
import torch.nn as nn
import torch
import sys
import path

sys.path.append('../../../')

#from furnace.seg_opr.sync_bn import DataParallelModel, Reduce, BatchNorm2d

class Network(nn.Module):
    def __init__(
            self,
            num_classes,
            criterion,
            norm_layer,
            pretrained_model=None,
            heads_config=4,
            **kwargs):
        super(Network, self).__init__()
        self.branch1 = SingleNetwork(
            num_classes, criterion, norm_layer, pretrained_model, heads_config=heads_config)
        self.branch2 = SingleNetwork(
            num_classes, criterion, norm_layer, pretrained_model, heads_config=heads_config)

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
            heads_config=4):
        super(SingleNetwork, self).__init__()
        
        self.backbone = resnet50(pretrained_model, norm_layer=norm_layer,
                                 bn_eps=conzeft.bn_eps,
                                 bn_momentum=conzeft.bn_momentum,
                                 deep_stem=True, stem_width=64, inplace=False)

        # self.conv_depth = nn.Sequential(nn.Conv2d(1,256,3), nn.BatchNorm2d(256,bn_eps=conzeft.bn_eps,
        #                          bn_momentum=conzeft.bn_momentum),nn.LeakyReLU)

        self.stem_width = 64

        self.dilate = 2
        for m in self.backbone.layer4.children():
            m.apply(partial(self._nostride_dilate, dilate=self.dilate))  # the apply function only takes in references to functions and then calls the function after, it expects whatever function is passed to it to have only one parameter, m, as the model. It then calls the function with m as the parameter. The reason partial is used is because the _nostride_dilate function has more than one input argument, so to get over that, the partial class takes in a function, and a partial subset of it's input arguments, and wraps a new function around them where the new function's input parameters are what remains (what wasn't defined in the partial call). So the apply function only sees a function with one input argument, m, so it works.
            self.dilate *= 2  # this explains the apply fucntion https://stackoverflow.com/questions/55613518/how-does-the-applyfn-function-in-pytorch-work-with-a-function-without-return-s

        self.head = Head(num_classes, norm_layer, conzeft.bn_momentum)
        self.business_layer = []
        self.business_layer.append(self.head)
        self.criterion = criterion
        
        self.cross_attention = CrossAttention(embed=256, heads=heads_config, need_weights=False, mode='dual_patch_patch', in_channels=256, patch_size=4)

        # num_classes is number of classes we're predicting, the input to the
        # classifier is a 256 channel feature map
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1, bias=True)
        self.business_layer.append(self.classifier)

    def forward(self, data):
        blocks = self.backbone(data[:, :3, :, :])
        depth = data[:, 3:, :, :]
        v3plus_feature, feat = self.head(blocks, depth, self.cross_attention)      # (b, c, h, w)
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
        return pred, feat, v3plus_feature

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

class CrossAttention(nn.Module):
    def __init__(self,
                embed=256,
                heads=6,
                need_weights=False,
                mode='combined_patch',
                in_channels=256,
                patch_size=4,
                norm_layer=nn.BatchNorm2d):
        super(CrossAttention, self).__init__()
        self.embed =  embed
        self.heads = heads
        self.need_weights = need_weights
        self.mode = mode
        self.patch_size = patch_size
        self.in_channels = in_channels

        self.im_embedding_convPxP = nn.Conv2d(self.in_channels, self.embed,  # this is the convolution done on the patches before feeding into the transformer
                                           kernel_size=self.patch_size, stride=self.patch_size, padding=0)

        self.im_positional_encodings = nn.Parameter(torch.rand(8192, embed), requires_grad=True)        

        self.d_attn_im_query = nn.MultiheadAttention(embed_dim=self.embed, num_heads=self.heads)#, batch_first=True)
        
        self.norm_layer = norm_layer(self.embed)

        if self.mode in ['dual_patch_patch', 'dual_token_patch']:
            
            self.d_embedding_convPxP = nn.Conv2d(self.in_channels, self.embed,  # this is the convolution done on the patches before feeding into the transformer
                                           kernel_size=self.patch_size, stride=self.patch_size, padding=0)


            self.im_attn_d_query = nn.MultiheadAttention(embed_dim=self.embed, num_heads=self.heads)#, batch_first=True)
            
            self.d_positional_encodings = nn.Parameter(torch.rand(8192, embed), requires_grad=True)

        else:
            raise NotImplementedError()
        

    def forward(self, im_feat, d_feat, im_token, d_token):

        n, c, h, w = im_feat.shape      #dep and im same shape since fed from same area (post biliner interpolation before concat in decoder)

        need_w = self.need_weights

        
        if self.mode in ['image_patch_patch', 'dual_patch_patch']:
            
            if self.mode == 'image_patch_patch':
                            
                #RGB depth dembeddings
                embed_im_feat_conv = self.im_embedding_convPxP(im_feat.clone())
                embed_im_flat =  embed_im_feat_conv.flatten(2)
                embed_im_feat = embed_im_flat + self.im_positional_encodings.clone()[:embed_im_flat.shape[2], :].T.unsqueeze(0)
                embed_im_feat = embed_im_feat.permute(2, 0, 1)

                #depth patch embeddings                
                embed_d_feat_conv = self.d_embedding_convPxP(d_feat.clone())
                embed_d_flat = embed_d_feat_conv.flatten(2)
                embed_d_feat = embed_d_flat + self.d_positional_encodings.clone()[:embed_d_flat.shape[2], :].T.unsqueeze(0)
                embed_d_feat = embed_d_feat.permute(2, 0, 1)     

                d_attn_im_query, _ = self.d_attn_im_query(embed_im_feat.clone(), embed_d_feat.clone(), embed_d_feat.clone(), need_weights=need_w)

                #L,N,E to N,E,L
                d_attn_im_query = d_attn_im_query.permute(1,2,0)

                d_attn_spatial = embed_d_feat_conv.clone()

                #reconstruct to 2D image from 1D sequence
                sqrt_d = embed_im_feat_conv.shape[-2]
                for i in range(sqrt_d):
                    if sqrt_d == 64 :
                        j = 2
                    else:
                        j = 1
                    
                    d_attn_spatial[:,:,i:(i+1),:] = (d_attn_im_query[:,:,(i*sqrt_d*j):((i+1)*sqrt_d*j)].unsqueeze(2))
      

                d_attn_spatial = F.interpolate(
                    d_attn_spatial, size=(h, w), mode="bilinear", align_corners=True
                    ) 


                #Skip Connections
                d_attn_spatial += d_feat


                #Layer Norm (Experiment with Layer Norm)
                d_attn_spatial = self.norm_layer(d_attn_spatial)

                return  im_feat, d_attn_spatial

            if self.mode == 'dual_patch_patch':

                #RGB depth dembeddings
                embed_im_feat_conv = self.im_embedding_convPxP(im_feat.clone())
                embed_im_flat =  embed_im_feat_conv.flatten(2)
                embed_im_feat = embed_im_flat + self.im_positional_encodings.clone()[:embed_im_flat.shape[2], :].T.unsqueeze(0)
                embed_im_feat = embed_im_feat.permute(2, 0, 1)

                #depth patch embeddings                
                embed_d_feat_conv = self.d_embedding_convPxP(d_feat.clone())
                embed_d_flat = embed_d_feat_conv.flatten(2)
                embed_d_feat = embed_d_flat + self.d_positional_encodings.clone()[:embed_d_flat.shape[2], :].T.unsqueeze(0)
                embed_d_feat = embed_d_feat.permute(2, 0, 1)     

                d_attn_im_query, _ = self.d_attn_im_query(embed_im_feat.clone(), embed_d_feat.clone(), embed_d_feat.clone(), need_weights=need_w)
                im_attn_d_query, _ = self.im_attn_d_query(embed_d_feat.clone(), embed_im_feat.clone(), embed_im_feat.clone(), need_weights=need_w)                


                d_attn_im_query = d_attn_im_query.permute(1,2,0)
                im_attn_d_query = im_attn_d_query.permute(1,2,0)

                d_attn_spatial = embed_d_feat_conv.clone()
                im_attn_spatial = embed_im_feat_conv.clone()

                #reconstruct to 2D image from 1D sequence
                sqrt_d = embed_im_feat_conv.shape[-2]
                for i in range(sqrt_d):
                    if sqrt_d == 64 :
                        j = 2
                    else:
                        j = 1
                    
                    d_attn_spatial[:,:,i:(i+1),:] = (d_attn_im_query[:,:,(i*sqrt_d*j):((i+1)*sqrt_d*j)].unsqueeze(2))
                    im_attn_spatial[:,:,i:(i+1),:] = (im_attn_d_query[:,:,(i*sqrt_d*j):((i+1)*sqrt_d*j)].unsqueeze(2))

                d_attn_spatial = F.interpolate(
                    d_attn_spatial, size=(h, w), mode="bilinear", align_corners=True
                    )       

                im_attn_spatial = F.interpolate(
                    im_attn_spatial, size=(h, w), mode="bilinear", align_corners=True
                    ) 


                #Skip Connections
                d_attn_spatial += d_feat
                im_attn_spatial += im_feat


                #Layer Norm (Experiment with Layer Norm)
                d_attn_spatial = self.norm_layer(d_attn_spatial)
                im_attn_spatial = self.norm_layer(im_attn_spatial)

                return im_attn_spatial, d_attn_spatial

            elif self.mode == 'image_token_patch':
                
                #This mode uses the image max pooling feature map as the token and query for cross attention with depth patches and values

                #image max pool used as token - dimensions aligned so convolution not required
                im_token = im_token.permute(2,0,1)


                #depth patch embeddings                
                embed_d_feat_conv = self.d_embedding_convPxP(d_feat.clone())
                embed_d_flat = embed_d_feat_conv.flatten(2)
                embed_d_feat = embed_d_flat + self.d_positional_encodings.clone()[:embed_d_flat.shape[2], :].T.unsqueeze(0)
                embed_d_feat = embed_d_feat.permute(2, 0, 1)     

                d_attn_im_query, _ = self.d_attn_im_query(im_token.clone(), embed_d_feat.clone(), embed_d_feat.clone(), need_weights=need_w)

                #L,N,E to N,E,L
                d_attn_im_query = d_attn_im_query.permute(1,2,0)

                #will be used for unpacking attention sequence in for loop
                d_attn_spatial = embed_d_feat_conv.clone()

                #reconstruct to 2D image from 1D sequence
                sqrt_d = embed_d_feat_conv.shape[-2]
                for i in range(sqrt_d):
                    if sqrt_d == 64 :
                        j = 2
                    else:
                        j = 1
                    
                    d_attn_spatial[:,:,i:(i+1),:] = (d_attn_im_query[:,:,(i*sqrt_d*j):((i+1)*sqrt_d*j)].unsqueeze(2))
      

                d_attn_spatial = F.interpolate(
                    d_attn_spatial, size=(h, w), mode="bilinear", align_corners=True
                    ) 

                #Skip Connections
                d_attn_spatial += d_feat


                #Layer Norm (Experiment with Layer Norm)
                d_attn_spatial = self.norm_layer(d_attn_spatial)

                return  im_feat, d_attn_spatial

            
            elif self.mode == 'dual_token_patch':
                
                #image max pool used as token - dimensions aligned so convolution not required
                im_token = im_token.permute(2,0,1)

                #image max pool used as token - dimensions aligned so convolution not required
                d_token = d_token.permute(2,0,1)

                #RGB depth dembeddings
                embed_im_feat_conv = self.im_embedding_convPxP(im_feat.clone())
                embed_im_flat =  embed_im_feat_conv.flatten(2)
                embed_im_feat = embed_im_flat + self.im_positional_encodings.clone()[:embed_im_flat.shape[2], :].T.unsqueeze(0)
                embed_im_feat = embed_im_feat.permute(2, 0, 1)

                #depth patch embeddings                
                embed_d_feat_conv = self.d_embedding_convPxP(d_feat.clone())
                embed_d_flat = embed_d_feat_conv.flatten(2)
                embed_d_feat = embed_d_flat + self.d_positional_encodings.clone()[:embed_d_flat.shape[2], :].T.unsqueeze(0)
                embed_d_feat = embed_d_feat.permute(2, 0, 1)     

                d_attn_im_query, _ = self.d_attn_im_query(im_token.clone(), embed_d_feat.clone(), embed_d_feat.clone(), need_weights=need_w) 
                im_attn_d_query, _ = self.im_attn_d_query(d_token.clone(), embed_im_feat.clone(), embed_im_feat.clone(), need_weights=need_w) 


                d_attn_im_query = d_attn_im_query.permute(1,2,0)
                im_attn_d_query = im_attn_d_query.permute(1,2,0)

                d_attn_spatial = embed_d_feat_conv.clone()
                im_attn_spatial = embed_im_feat_conv.clone()

                #reconstruct to 2D image from 1D sequence
                sqrt_d = embed_im_feat_conv.shape[-2]
                for i in range(sqrt_d):
                    if sqrt_d == 64 :
                        j = 2
                    else:
                        j = 1
                    
                    d_attn_spatial[:,:,i:(i+1),:] = (d_attn_im_query[:,:,(i*sqrt_d*j):((i+1)*sqrt_d*j)].unsqueeze(2))
                    im_attn_spatial[:,:,i:(i+1),:] = (im_attn_d_query[:,:,(i*sqrt_d*j):((i+1)*sqrt_d*j)].unsqueeze(2))

                d_attn_spatial = F.interpolate(
                    d_attn_spatial, size=(h, w), mode="bilinear", align_corners=True
                    )       

                im_attn_spatial = F.interpolate(
                    im_attn_spatial, size=(h, w), mode="bilinear", align_corners=True
                    ) 


                #Skip Connections
                d_attn_spatial += d_feat
                im_attn_spatial += im_feat


                #Layer Norm (Experiment with Layer Norm)
                d_attn_spatial = self.norm_layer(d_attn_spatial)
                im_attn_spatial = self.norm_layer(im_attn_spatial)

                return im_attn_spatial, d_attn_spatial


            elif self.mode == 'token_token':
                
                #image max pool used as token - dimensions aligned so convolution not required
                im_token = im_token.permute(2,0,1)

                #image max pool used as token - dimensions aligned so convolution not required
                d_token = d_token.permute(2,0,1)    

                #output will be L,E,1
                d_attn_im_query, _ = self.d_attn_im_query(im_token.clone(), d_token.clone(), d_token.clone(), need_weights=need_w)
                im_attn_d_query, _ = self.im_attn_d_query(d_token.clone(), im_token.clone(), im_token.clone(), need_weights=need_w)


                d_attn_im_query = d_attn_im_query.permute(1,2,0)
                im_attn_d_query = im_attn_d_query.permute(1,2,0)

                d_attn_spatial = embed_d_feat_conv.clone()
                im_attn_spatial = embed_im_feat_conv.clone()


                #token attention used as weights
                d_attn_spatial = d_feat * d_attn_im_query
                im_attn_spatial = im_feat * im_attn_d_query

                #Skip Connections
                d_attn_spatial += d_feat
                im_attn_spatial += im_feat

                #Layer Norm (Experiment with Layer Norm)
                d_attn_spatial = self.norm_layer(d_attn_spatial)
                im_attn_spatial = self.norm_layer(im_attn_spatial)

                return im_attn_spatial, d_attn_spatial
        


class ASPP(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dilation_rates=(12, 24, 36),
                 hidden_channels=256,
                 norm_act=nn.BatchNorm2d,
                 pooling_size=None):
        super(ASPP, self).__init__()
        self.pooling_size = pooling_size

        # add dilated convs for depth, change  last conv number of input dimensions to suit
        # add initial conv to append to low level features, both depth streams
        # will be appended to the same areas

        self.depth_downsample = nn.Sequential(
            nn.Conv2d(
                1,
                256,
                kernel_size=7,
                stride=4,
                padding=2,
                bias=False),
            norm_act(256),
            nn.LeakyReLU())

        self.depth_map_convs = nn.ModuleList([
            nn.Conv2d(256, hidden_channels, 1, bias=False),
            nn.Conv2d(256, hidden_channels, 3, bias=False, dilation=dilation_rates[0],
                      padding=dilation_rates[0]),
            nn.Conv2d(256, hidden_channels, 3, bias=False, dilation=dilation_rates[1],
                      padding=dilation_rates[1]),
            nn.Conv2d(256, hidden_channels, 3, bias=False, dilation=dilation_rates[2],
                      padding=dilation_rates[2])
        ])

        self.pool_depth = nn.Sequential(
            nn.AdaptiveAvgPool2d(
                (1, 1)), nn.Conv2d(
                256, hidden_channels, kernel_size=1, bias=False))

        self.pool_img = nn.Sequential(
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
        self.pool_red_conv = nn.Conv2d(
            hidden_channels, out_channels, 1, bias=False)
        self.red_bn = norm_act(out_channels)

        self.leak_relu = nn.LeakyReLU()

    def forward(self, x, d):
        # Map convolutions
        _, _, h, w = x.size()
        img_pool = F.interpolate(
            self.pool_img(x), size=(h, w), mode="bilinear", align_corners=True
        )
        aspp_list = [m(x) for m in self.map_convs]
        aspp_list.insert(0, img_pool)
        out = torch.cat(aspp_list, dim=1)
        out = self.map_bn(out)
        out = self.leak_relu(out)       # add activation layer
        out = self.red_conv(out)
        out = self.leak_relu(out)

        d = self.depth_downsample(d)
        _, _, h_d, w_d = d.size()
        depth_pool = F.interpolate(self.pool_depth(d), size=(
            h_d, w_d), mode="bilinear", align_corners=True)
        depth_aspp = [m(d) for m in self.depth_map_convs]
        depth_aspp.insert(0, depth_pool)
        depth_aspp = torch.cat(depth_aspp, dim=1)
        depth = self.map_bn(depth_aspp)
        depth = self.leak_relu(depth)       # add activation layer
        depth = self.red_conv(depth)
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
        return out, depth, img_pool, depth_pool

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
            bn_momentum=0.0003):
        super(Head, self).__init__()

        self.classify_classes = classify_classes
        self.aspp = ASPP(2048, 256, [12, 18, 24], norm_act=norm_act)

        self.reduce = nn.Sequential(
            nn.Conv2d(256, 256, 1, bias=False),
            norm_act(256, momentum=bn_momentum),
            nn.ReLU(),
        )
        self.last_conv = nn.Sequential(
            nn.Conv2d(
                768, 256, kernel_size=3, stride=1, padding=1, bias=False), norm_act(
                256, momentum=bn_momentum), nn.ReLU(), nn.Conv2d(
                256, 256, kernel_size=3, stride=1, padding=1, bias=False), norm_act(
                    256, momentum=bn_momentum), nn.ReLU(), )

        #for name, parameters in self.cross_attention.named_parameters:
        #    print(name)

        #for name, parameters in self.cross_attention.named_children:
        #    print(name)

    def forward(self, f_list, depth, cross_attention):
        f = f_list[-1]
        f1, depth1, img_pool, depth_pool = self.aspp(f, depth)

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
        cross_attn_im, cross_attn_d = cross_attention(f2, depth1, img_pool, depth_pool)
        f3 = torch.cat((cross_attn_d, cross_attn_im, low_level_features), dim=1)
        embed_feat = torch.concat((cross_attn_d, cross_attn_im), dim=1)
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
