import importlib

import torch.nn as nn
from torch.nn import functional as F

#from .decoder import Aux_Module


def __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                  **kwargs):
    for name, m in feature.named_modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            conv_init(m.weight, **kwargs)

        elif isinstance(m, norm_layer):
            m.eps = bn_eps
            m.momentum = bn_momentum
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                **kwargs):
    if isinstance(module_list, list):
        for feature in module_list:
            __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                          **kwargs)
    else:
        __init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                      **kwargs)


class ModelBuilder(nn.Module):
    def __init__(self, net_cfg, depth=None):
        super(ModelBuilder, self).__init__()
        self._sync_bn = net_cfg["sync_bn"]
        self._num_classes = net_cfg["num_classes"]

        self.encoder = self._build_encoder(net_cfg["encoder"])
        self.decoder = self._build_decoder(net_cfg["decoder"])

        self.depth_concat = True if net_cfg["decoder"]["type"] \
            == 'u2pl.models.decoder.dec_deeplabv3_plus_depth' else False

        self._use_auxloss = True if net_cfg.get("aux_loss", False) else False
        print("Aux Loss should be set to false", self._use_auxloss)
        self.fpn = True if net_cfg["encoder"]["kwargs"].get(
            "fpn", False) else False
        if self._use_auxloss:
            cfg_aux = net_cfg["aux_loss"]
            self.loss_weight = cfg_aux["loss_weight"]
            self.auxor = Aux_Module(
                cfg_aux["aux_plane"], self._num_classes, self._sync_bn
            )
        init_weight(self.decoder, nn.init.kaiming_normal_,
                    nn.BatchNorm2d, 1e-5, 0.1,
                    mode='fan_in', nonlinearity='relu')

    def _build_encoder(self, enc_cfg):
        enc_cfg["kwargs"].update({"sync_bn": self._sync_bn})
        encoder = self._build_module(enc_cfg["type"], enc_cfg["kwargs"])
        return encoder

    def _build_decoder(self, dec_cfg):
        dec_cfg["kwargs"].update(
            {
                "in_planes": self.encoder.get_outplanes(),
                "sync_bn": self._sync_bn,
                "num_classes": self._num_classes,
            }
        )
        decoder = self._build_module(dec_cfg["type"], dec_cfg["kwargs"])
        return decoder

    # explanation below for decoder - applies to encoder as well
    def _build_module(self, mtype, kwargs):
        # returns a list with ["u2pl.models.decoder", "dec_deeplabv3_plus"]
        # since rsplit starts from the right and maxsplit is limited to 1
        module_name, class_name = mtype.rsplit(".", 1)
        # therefore this statement acts like an import statement, to import the
        # decoder module/file which is in U2PL -> models folder
        module = importlib.import_module(module_name)
        # The class dec_deeplabv3_plus is an attribute of the imported
        # module/file, so it is assigned to cls - cls is now the class
        cls = getattr(module, class_name)
        return cls(**kwargs)  # the initialisation parameters are passed into the class - this is like dec_deeplabv3_plus(num_classes, etc) so the _init_ method is executed and the instantiated decoder is now returned

    def forward(self, x):
        if self._use_auxloss:
            if self.fpn:
                # feat1 used as dsn loss as default, f1 is layer2's output as
                # default
                f1, f2, feat1, feat2 = self.encoder(x)
                outs = self.decoder([f1, f2, feat1, feat2])
            else:
                feat1, feat2 = self.encoder(x)
                outs = self.decoder(feat2)

            pred_aux = self.auxor(feat1)

            outs.update({"aux": pred_aux})
            return outs
        else:
            if self.fpn:

                if self.depth_concat:
                    f1, f2, feat1, feat2 = self.encoder(x[:, :3, :, :])
                    depth = x[:, 3:, :, :]
                    outs = self.decoder([f1, f2, feat1, feat2], depth)

                else:
                    f1, f2, feat1, feat2 = self.encoder(x)
                    outs = self.decoder([f1, f2, feat1, feat2])

            else:
                feat = self.encoder(x)
                outs = self.decoder(feat)

            return outs
