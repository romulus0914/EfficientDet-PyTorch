import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

from efficientnet import _BuildEfficientNet
from config import efficientdet_model_params

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)

class NearestUpsampling(nn.Module):
    def __init__(self, scale):
        super(NearestUpsampling, self).__init__()

        self.scale = scale

    def forward(self, x):
        bs, c, h, w = x.shape
        x = torch.reshape(x, (bs, c, h, 1, w, 1)) * torch.ones(1, 1, 1, self.scale, 1, self.scale)

        return torch.reshape(x, (bs, c, h*self.scale, w*self.scale))

def _BatchNorm(channels, eps=1e-4, momentum=0.003):
    return nn.BatchNorm2d(channels, eps=eps, momentum=momentum)

def _SepconvBnReLU(in_channels, out_channels, kernel_size=3, relu_last=True):
    if relu_last:
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, 1, kernel_size//2, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
            _BatchNorm(out_channels),
            Swish()
        )
    else:
        return nn.Sequential(
            Swish(),
            nn.Conv2d(in_channels, in_channels, kernel_size, 1, kernel_size//2, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
            _BatchNorm(out_channels)
        )

def _ResampleFeatureMap(width, num_channels, target_width, target_num_channels):
    resample = [nn.Identity()]

    if num_channels != target_num_channels:
        resample.append(nn.Sequential(
            nn.Conv2d(num_channels, target_num_channels, 1, 1, 0),
            _BatchNorm(target_num_channels)
        ))

    if width > target_width:
        stride = int(width//target_width)
        resample.append(nn.MaxPool2d(stride+1, stride, stride//2))
    elif width < target_width:
        scale = target_width // width
        resample.append(NearestUpsampling(scale))

    return nn.Sequential(*resample)

class ClassNet(nn.Module):
    def __init__(self, model_params, num_classes=90, num_anchors=9):
        super(ClassNet, self).__init__()

        fpn_num_channels = model_params['fpn_num_channels']
        box_class_repeats = model_params['box_class_repeats']
        num_features = model_params['num_features']

        class_conv = []
        for _ in range(num_features):
            conv = []
            for _ in range(box_class_repeats):
                conv.append(_SepconvBnReLU(fpn_num_channels, fpn_num_channels))
            conv.append(_SepconvBnReLU(fpn_num_channels, num_classes*num_anchors))

            class_conv.append(nn.Sequential(*conv))

        self.class_conv = nn.Sequential(*class_conv)

        self._initialize_weights(fpn_num_channels)

    def _initialize_weights(self, fpn_num_channels):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                nn.init.normal_(m.weight, 0, math.sqrt(1.0/n))
                if m.out_channels == fpn_num_channels:
                    nn.init.zeros_(m.bias)
                else:
                    nn.init.constant_(m.bias, -np.log((1-0.01)/0.01))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, features):
        class_predicts = [class_conv(feature) for class_conv, feature in zip(self.class_conv, features)]

        return class_predicts

class BoxNet(nn.Module):
    def __init__(self, model_params, num_anchors=9):
        super(BoxNet, self).__init__()

        fpn_num_channels = model_params['fpn_num_channels']
        box_class_repeats = model_params['box_class_repeats']
        num_features = model_params['num_features']

        box_conv = []
        for _ in range(num_features):
            conv = []
            for _ in range(box_class_repeats):
                conv.append(_SepconvBnReLU(fpn_num_channels, fpn_num_channels))
            conv.append(_SepconvBnReLU(fpn_num_channels, 4*num_anchors))

            box_conv.append(nn.Sequential(*conv))

        self.box_conv = nn.Sequential(*box_conv)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                nn.init.normal_(m.weight, 0, math.sqrt(1.0/n))
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, features):
        box_predicts = [box_conv(feature) for box_conv, feature in zip(self.box_conv, features)]

        return box_predicts

class BiFPN(nn.Module):
    fpn_config = [
        {'width_ratio': 1.0/(2**3), 'inputs': []},
        {'width_ratio': 1.0/(2**4), 'inputs': []},
        {'width_ratio': 1.0/(2**5), 'inputs': []},
        {'width_ratio': 1.0/(2**6), 'inputs': []},
        {'width_ratio': 1.0/(2**7), 'inputs': []},
        {'width_ratio': 1.0/(2**6), 'inputs': [3, 4]},
        {'width_ratio': 1.0/(2**5), 'inputs': [2, 5]},
        {'width_ratio': 1.0/(2**4), 'inputs': [1, 6]},
        {'width_ratio': 1.0/(2**3), 'inputs': [0, 7]},
        {'width_ratio': 1.0/(2**4), 'inputs': [1, 7, 8]},
        {'width_ratio': 1.0/(2**5), 'inputs': [2, 6, 9]},
        {'width_ratio': 1.0/(2**6), 'inputs': [3, 5, 10]},
        {'width_ratio': 1.0/(2**7), 'inputs': [4, 11]},
    ]

    def __init__(self, model_params, features_num_channels):
        super(BiFPN, self).__init__()

        image_size = model_params['image_size']
        fpn_num_channels = model_params['fpn_num_channels']
        fpn_cell_repeats = model_params['fpn_cell_repeats']
        num_features = model_params['num_features']

        # downsample feature from backbone model to desirable width and channels
        feature_width = image_size // (2**5)
        self.resample6 = _ResampleFeatureMap(feature_width, features_num_channels[-1], feature_width//2, fpn_num_channels)
        self.resample7 = _ResampleFeatureMap(feature_width//2, fpn_num_channels, feature_width//4, fpn_num_channels)

        # build fpn cells
        fpn_cells = []
        fpn_cells_resample = []

        fpn_nodes_width = [int(image_size*node_config['width_ratio']) for node_config in self.fpn_config]
        for cell_idx in range(fpn_cell_repeats):
            fpn_layers = []
            fpn_layers_resample = []
            for node_idx, node_config in enumerate(self.fpn_config[num_features:]):
                # resample input features
                input_nodes_resample = []
                for input_node in node_config['inputs']:
                    input_nodes_resample.append(_ResampleFeatureMap(
                        fpn_nodes_width[input_node],
                        features_num_channels[input_node] if cell_idx == 0 and input_node < num_features - 2 else fpn_num_channels,
                        fpn_nodes_width[node_idx+num_features],
                        fpn_num_channels
                    ))
                fpn_layers_resample.append(nn.Sequential(*input_nodes_resample))

                # depthwise separable convolution for feature fusion
                fpn_layers.append(_SepconvBnReLU(fpn_num_channels, fpn_num_channels, relu_last=False))

            fpn_cells.append(nn.Sequential(*fpn_layers))
            fpn_cells_resample.append(nn.Sequential(*fpn_layers_resample))

        self.fpn_cells = nn.Sequential(*fpn_cells)
        self.fpn_cells_resample = nn.Sequential(*fpn_cells_resample)

        # weight method for input nodes
        if model_params['weight_method'] == 'fastattn':
            total_input_nodes = sum([len(node_config['inputs']) for node_config in self.fpn_config[num_features:]])
            self.fastattn_weights = nn.Parameter(torch.ones(fpn_cell_repeats, total_input_nodes), requires_grad=True)

        self.model_params = model_params

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, features):
        features.append(self.resample6(features[-1]))
        features.append(self.resample7(features[-1]))

        for cell_idx, (fpn_layers, fpn_layers_resample) in enumerate(zip(self.fpn_cells, self.fpn_cells_resample)):
            weights_offset = 0
            for node_idx, node_config in enumerate(self.fpn_config[self.model_params['num_features']:]):
                num_inputs = len(node_config['inputs'])
                feature_inputs = [resample(features[i]) for i, resample in zip(node_config['inputs'], fpn_layers_resample[node_idx])]
                if self.model_params['weight_method'] == 'fastattn':
                    weights = [self.fastattn_weights[cell_idx][weights_offset+i] for i in range(num_inputs)]
                    weights_sum = sum(weights)
                    feature_inputs = [feature*F.relu(weight)/(weights_sum+0.0001) for feature, weight in zip(feature_inputs, weights)]

                feature = sum(feature_inputs)
                feature = fpn_layers[node_idx](feature)
                features.append(feature)

                weights_offset += num_inputs

            features = features[-self.model_params['num_features']:]

        return features

class EfficientDet(nn.Module):
    def __init__(self, model_params):
        super(EfficientDet, self).__init__()

        model_params['num_features'] = 5

        # EfficientNet backbone
        self.backbone_model, features_num_channels = _BuildEfficientNet(model_params['backbone_type'], model_params['num_features']-2)

        # BiFPN
        self.fpn = BiFPN(model_params, features_num_channels)

        # box and class net
        self.class_net = ClassNet(model_params)
        self.box_net = BoxNet(model_params)

        self.num_features = model_params['num_features']

    def forward(self, images):
        _ = self.backbone_model(images)
        features = [self.backbone_model.layers['reduction_{}'.format(i)] for i in range(8-self.num_features, 6)]
        features = self.fpn(features)
        class_predicts = self.class_net(features)
        box_predicts = self.box_net(features)

        return class_predicts, box_predicts

if __name__ == '__main__':
    model = EfficientDet(efficientdet_model_params['efficientdet-d0'])
    image = nn.Parameter(torch.randn(1, 3, 512, 512), requires_grad=False)
    predicts = model(image)
