from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo

from .utils import pyramidPooling, conv2DBatchNormRelu

# code adopted from https://github.com/nianticlabs/monodepth2/blob/master/networks/resnet_encoder.py

class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False) #, dilation=2
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

    if pretrained:
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


class FeatExactor(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers=18, pretrained=True):
        super(FeatExactor, self).__init__()

        num_input_images = 1

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4


        self.pyramid_pooling = pyramidPooling(512, None, fusion_mode='sum', model_name='icnet')
        # Iconvs
        self.upconv6 = nn.Sequential(nn.Upsample(scale_factor=2),
                                     conv2DBatchNormRelu(in_channels=512, k_size=3, n_filters=256))
        self.iconv5 = conv2DBatchNormRelu(in_channels=512, k_size=3, n_filters=256)

        self.upconv5 = nn.Sequential(nn.Upsample(scale_factor=2),
                                     conv2DBatchNormRelu(in_channels=256, k_size=3, n_filters=128))
        self.iconv4 = conv2DBatchNormRelu(in_channels=256, k_size=3, n_filters=128,
                                          padding=1, stride=1, bias=False)

        self.upconv4 = nn.Sequential(nn.Upsample(scale_factor=2),
                                     conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64))
        self.iconv3 = conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64)

        self.proj6 = conv2DBatchNormRelu(in_channels=512, k_size=1, n_filters=128, padding=0, stride=1, bias=False)
        self.proj5 = conv2DBatchNormRelu(in_channels=256, k_size=1, n_filters=64, padding=0, stride=1, bias=False)
        self.proj4 = conv2DBatchNormRelu(in_channels=128, k_size=1, n_filters=32, padding=0, stride=1, bias=False)
        self.proj3 = conv2DBatchNormRelu(in_channels=64, k_size=1, n_filters=16, padding=0, stride=1, bias=False)

    def forward(self, input_image):
        self.features = []
        x = input_image.clone()

        # H, W -> H/2, W/2
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)

        ## H/2, W/2 -> H/4, W/4
        pool1 = self.encoder.maxpool(x)

        # H/4, W/4 -> H/16, W/16
        conv3 = self.encoder.layer1(pool1)
        conv4 = self.encoder.layer2(conv3)
        conv5 = self.encoder.layer3(conv4)
        conv6 = self.encoder.layer4(conv5)
        conv6 = self.pyramid_pooling(conv6)


        concat5 = torch.cat((conv5, self.upconv6(conv6)), dim=1)
        conv5 = self.iconv5(concat5)

        concat4 = torch.cat((conv4, self.upconv5(conv5)), dim=1)
        conv4 = self.iconv4(concat4)

        concat3 = torch.cat((conv3, self.upconv4(conv4)), dim=1)
        conv3 = self.iconv3(concat3)

        proj6 = self.proj6(conv6)
        proj5 = self.proj5(conv5)
        proj4 = self.proj4(conv4)
        proj3 = self.proj3(conv3)

        return proj6,proj5,proj4,proj3
