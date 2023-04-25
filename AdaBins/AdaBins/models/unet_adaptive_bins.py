import torch
import torch.nn as nn
import torch.nn.functional as F

from .miniViT import mViT


# When this file is called - UnetAdaptiveBins is called, particularly the build method and the pre-trained model is loaded
# into the 'model' variable,

class UpSampleBN(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSampleBN, self).__init__()

        self._net = nn.Sequential(
            nn.Conv2d(
                skip_input,
                output_features,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.BatchNorm2d(output_features),
            nn.LeakyReLU(),
            nn.Conv2d(
                output_features,
                output_features,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.BatchNorm2d(output_features),
            nn.LeakyReLU())

    def forward(self, x, concat_with):
        up_x = F.interpolate(
            x,
            size=[
                concat_with.size(2),
                concat_with.size(3)],
            mode='bilinear',
            align_corners=True)
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)


class DecoderBN(nn.Module):
    def __init__(
            self,
            num_features=2048,
            num_classes=1,
            bottleneck_features=2048):
        super(DecoderBN, self).__init__()
        features = int(num_features)

        self.conv2 = nn.Conv2d(
            bottleneck_features,
            features,
            kernel_size=1,
            stride=1,
            padding=1)

        self.up1 = UpSampleBN(
            skip_input=features // 1 + 112 + 64,
            output_features=features // 2)
        self.up2 = UpSampleBN(
            skip_input=features // 2 + 40 + 24,
            output_features=features // 4)
        self.up3 = UpSampleBN(
            skip_input=features // 4 + 24 + 16,
            output_features=features // 8)
        self.up4 = UpSampleBN(
            skip_input=features // 8 + 16 + 8,
            output_features=features // 16)

        #         self.up5 = UpSample(skip_input=features // 16 + 3, output_features=features//16)
        self.conv3 = nn.Conv2d(
            features // 16,
            num_classes,
            kernel_size=3,
            stride=1,
            padding=1)
        # self.act_out = nn.Softmax(dim=1) if output_activation == 'softmax' else nn.Identity()

    def forward(self, features):

        #print("input to decoder size", len(features))
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[
            4], features[5], features[6], features[8], features[11]

        #print("features 4, 5, 6, 8, 11 shapes", x_block0.shape, x_block1.shape, x_block2.shape, x_block3.shape, x_block4.shape )
        x_d0 = self.conv2(x_block4)

        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)
        #         x_d5 = self.up5(x_d4, features[0])
        out = self.conv3(x_d4)
        # out = self.act_out(out)
        # if with_features:
        #     return out, features[-1]
        # elif with_intermediate:
        # return out, [x_block0, x_block1, x_block2, x_block3, x_block4, x_d1,
        # x_d2, x_d3, x_d4]
        return out


class Encoder(nn.Module):
    def __init__(self, backend):
        super(Encoder, self).__init__()
        self.original_model = backend

    def forward(self, x):
        features = [x]  # initialising a list of value x (same as input)
        # print(self.original_model._modules.items()) #encoder modules?
        for k, v in self.original_model._modules.items():
            if (k == 'blocks'):
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))  # appending to the end
            else:
                features.append(v(features[-1]))  # appending to the end
        return features


class UnetAdaptiveBins(nn.Module):
    def __init__(
            self,
            backend,
            n_bins=100,
            min_val=0.1,
            max_val=10,
            norm='linear'):
        super(UnetAdaptiveBins, self).__init__()
        self.num_classes = n_bins
        self.min_val = min_val  # max and min val are the depth intervals to be used for a particular dataset, default here of 10 is for NYU, KITTI used 80 metres
        self.max_val = max_val
        self.encoder = Encoder(backend)
        self.n_query_channels = 128
        #print('query channels', n_bins)
        self.adaptive_bins_layer = mViT(
            128,
            n_query_channels=self.n_query_channels,
            patch_size=16,
            dim_out=n_bins,
            embedding_dim=128,
            norm=norm)
        print()
        self.decoder = DecoderBN(num_classes=128)
        self.conv_out = nn.Sequential(
            nn.Conv2d(
                self.n_query_channels,
                n_bins,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.Softmax(
                dim=1))

    def forward(self, x, **kwargs):
        #print("image shape going into model encoder", x.shape)
        unet_out = self.decoder(
            self.encoder(x),
            **kwargs)  # h x w x Cd (where Cd is?)
        #print("image shape output from decoder", unet_out.shape)
        bin_widths_normed, range_attention_maps = self.adaptive_bins_layer(
            unet_out)
        #print("bin widths normed", bin_widths_normed.shape, "range attentuion maps", range_attention_maps.shape)

        out = self.conv_out(range_attention_maps)
        #print("final conv", out.shape)

        # Post process
        # n, c, h, w = out.shape
        # hist = torch.sum(out.view(n, c, h * w), dim=2) / (h * w)  # not used
        # for training

        bin_widths = (self.max_val - self.min_val) * \
            bin_widths_normed  # .shape = N, dim_out
        bin_widths = nn.functional.pad(
            bin_widths, (1, 0), mode='constant', value=self.min_val)
        # cumilitive sum, so for a vector of dimension N, output will also be N
        # dimensions, element 0 will be the same, element [1] will be [0+1],
        # etc
        bin_edges = torch.cumsum(bin_widths, dim=1)

        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        n, dout = centers.size()
        centers = centers.view(n, dout, 1, 1)

        pred = torch.sum(out * centers, dim=1, keepdim=True)
        #print("final prediction size",pred.shape)
        #print("final bin edges size",bin_edges.shape)

        return bin_edges, pred   # pred is h x w x 1

    def get_1x_lr_params(self):  # lr/10 learning rate
        return self.encoder.parameters()

    def get_10x_lr_params(self):  # lr learning rate
        modules = [self.decoder, self.adaptive_bins_layer, self.conv_out]
        for m in modules:
            yield from m.parameters()

    # defining a class method instead of an instance method so 'cls' is passed
    # instead of self, self is the instance method (so when you define a class
    # object) and
    @classmethod
    def build(cls, n_bins, **kwargs):  # class methods are usually defined to prevent defining multiple instances with the same constructor multiple times, for example see https://realpython.com/instance-class-and-static-methods-demystified/ when he references pizza ingredients
        # in this case, it was used so the backend for the encoder/decoder
        # could be passed in as a parameter for defining instances. The reason
        # this works is because you can call class methods before calling the
        # class constructor (which is not the case for instace methods because
        # you need to pass in the self parameter for them)
        basemodel_name = 'tf_efficientnet_b5_ap'
        # therefore in the last function (if __name__ = main), class method was
        # called, the backend was loaded from PyTorch hub loaded and modified,
        # and then this was used to call the class ('cls' function in line 140)
        # and create an instance with the backend being passed in as a
        # parameter

        print('Loading base model ()...'.format(basemodel_name), end='')
        # loading efficientnet_b5 from github repo (it's a tf-tensorflow model
        # weirdly enough)
        basemodel = torch.hub.load(
            'rwightman/gen-efficientnet-pytorch',
            basemodel_name,
            pretrained=True)
        print('Done.')

        # Remove last layer
        print('Removing last two layers (global_pool & classifier).')
        # replacing the final two layers with the idnetity function
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()

        # Building Encoder-Decoder model
        print('Building Encoder-Decoder model..', end='')
        # calling 'cls' is basically the same as calling the class directly,
        # like calling 'UnetAdaptiveBins', reason this works is basically how
        # self works in an instance method, cls is automatically passed in a
        # class method which is the actual class
        m = cls(basemodel, n_bins=n_bins, **kwargs)
        print('Done.')
        return m


if __name__ == '__main__':
    model = UnetAdaptiveBins.build(100)  # n. of bins
    # seems like a self check to check the model outputs the right dimensions
    x = torch.rand(2, 3, 480, 640)
    # what's fed into it is a random set of two images, and the bins and
    # predictions are checked
    bins, pred = model(x)
    print(bins.shape, pred.shape)
