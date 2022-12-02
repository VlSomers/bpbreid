from __future__ import absolute_import
from __future__ import division

# Source: https://github.com/hh23333/PVPM

__all__ = ['pcb_p6', 'pcb_p4', 'pose_resnet50_256_p4', 'pose_resnet50_256_p6',
    'pose_resnet50_256_p6_pscore_reg', 'pose_resnet50_256_p4_pscore_reg']

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import torch.utils.model_zoo as model_zoo
from .osnet import ConvLayer, Conv1x1, Conv1x1Linear, Conv3x3, LightConv3x3, OSBlock

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DimReduceLayer(nn.Module):

    def __init__(self, in_channels, out_channels, nonlinear):
        super(DimReduceLayer, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        if nonlinear == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif nonlinear == 'leakyrelu':
            layers.append(nn.LeakyReLU(0.1))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class PCB(nn.Module):
    """Part-based Convolutional Baseline.

    Reference:
        Sun et al. Beyond Part Models: Person Retrieval with Refined
        Part Pooling (and A Strong Convolutional Baseline). ECCV 2018.

    Public keys:
        - ``pcb_p4``: PCB with 4-part strips.
        - ``pcb_p6``: PCB with 6-part strips.
    """

    def __init__(self, num_classes, loss, block, layers,
                 parts=6,
                 reduced_dim=256,
                 nonlinear='relu',
                 **kwargs):
        self.inplanes = 64
        super(PCB, self).__init__()
        self.loss = loss
        self.parts = parts
        self.feature_dim = 512 * block.expansion

        # backbone network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)

        # pcb layers
        self.parts_avgpool = nn.AdaptiveAvgPool2d((self.parts, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.em = nn.ModuleList( # TODO: before = DimReduceLayer
            [self._construct_em_layer(reduced_dim, 512 * block.expansion) for _ in range(self.parts)])
        self.feature_dim = reduced_dim
        self.classifier = nn.ModuleList(
            [nn.Linear(self.feature_dim, num_classes, bias=False) for _ in range(self.parts)])

        self._init_params()

    def _construct_em_layer(self, fc_dims, input_dim, dropout_p=0.5): # TODO new
        """
        Construct fully connected layer

        - fc_dims (list or tuple): dimensions of fc layers, if None,
                                   no fc layers are constructed
        - input_dim (int): input dimension
        - dropout_p (float): dropout probability, if None, dropout is unused
        """
        if fc_dims is None:
            self.feature_dim = input_dim
            return None

        layers = []

        # layers.append(nn.Linear(input_dim, fc_dims))
        layers.append(nn.Conv2d(input_dim, fc_dims, 1, stride=1, padding=0))
        layers.append(nn.BatchNorm2d(fc_dims))
        layers.append(nn.ReLU(inplace=True))
        # layers.append(nn.Dropout(p=dropout_p))

        # self.feature_dim = fc_dims[-1]

        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.001) # TODO Learning rate of pre-trained layers: 0.1 x base learning rate ?
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x):
        f = self.featuremaps(x)
        # vis_featmat_Kmeans(f) # TODO
        # vis_featmat_DBSCAN(f) # TODO
        v_g = self.parts_avgpool(f)  # nn.AdaptiveAvgPool2d((self.parts, 1))

        if not self.training:
            v_g = F.normalize(v_g, p=2, dim=1)
            return v_g.view(v_g.size(0), -1)

        # v_g = self.dropout(v_g)
        # v_h = self.conv5(v_g)

        y = []
        v = []
        # v_g.shape = [n, 2048, 6, 1] ?
        for i in range(self.parts): # TODO new
            v_g_i = v_g[:, :, i, :].view(v_g.size(0), -1, 1, 1)
            v_g_i = self.em[i](v_g_i)  # fully connected layer, Conv2d-BatchNorm2d-ReLU
            v_h_i = v_g_i.view(v_g_i.size(0), -1)
            y_i = self.classifier[i](v_h_i)
            y.append(y_i)
            v.append(v_g_i)

        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            v_g = F.normalize(v_g, p=2, dim=1) # TODO
            return y, v_g.view(v_g.size(0), -1)
        else:
            raise KeyError('Unsupported loss: {}'.format(self.loss))


def init_pretrained_weights(model, model_url):
    """Initializes model with pretrained weights.

    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    pretrain_dict = model_zoo.load_url(model_url)
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)


def pcb_p6(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = PCB(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        last_stride=1,
        parts=6,
        reduced_dim=256,
        nonlinear='relu',
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model


def pcb_p4(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = PCB(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        last_stride=1,
        parts=4,
        reduced_dim=256,
        nonlinear='relu',
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model


class Conv1x1_att(nn.Module):
    """1x1 convolution + bn + relu."""

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(Conv1x1_att, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0,
                              bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class score_embedding(nn.Module):
    """1x1 convolution + bn + relu."""

    def __init__(self, in_channels, out_channels):
        super(score_embedding, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.reg = nn.Linear(in_channels, out_channels, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.reg(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class Pose_Subnet(nn.Module): # TODO
    '''
    PVP and PGA
    '''
    def __init__(self, blocks, in_channels, channels, att_num=1, IN=False, matching_score_reg=False):
        super(Pose_Subnet, self).__init__()
        num_blocks = len(blocks)
        self.conv1 = ConvLayer(in_channels, channels[0], 7, stride=1, padding=3, IN=IN)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(blocks[0], 1, channels[0], channels[1], reduce_spatial_size=True)
        self.conv3 = self._make_layer(blocks[1], 1, channels[1], channels[2], reduce_spatial_size=False)
        self.conv4 = Conv3x3(channels[2], channels[2])
        # PGA
        self.conv_out = Conv1x1_att(channels[2], att_num)
        # PVP
        self.matching_score_reg = matching_score_reg
        if self.matching_score_reg:
            self.conv_score = score_embedding(channels[2], att_num)

        self._init_params()

    def _make_layer(self, block, layer, in_channels, out_channels, reduce_spatial_size, IN=False):
        layers = []
        layers.append(block(in_channels, out_channels, IN=IN, gate_reduction=4))
        for i in range(1, layer):
            layers.append(block(out_channels, out_channels, IN=IN, gate_reduction=4))

        if reduce_spatial_size:
            layers.append(
                nn.Sequential(
                    Conv1x1(out_channels, out_channels),
                    nn.AvgPool2d(2, stride=2)
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x_ = self.conv4(x)
        x = self.conv_out(x_)
        _, max_index = x.max(dim=1, keepdim=True)
        onehot_index = torch.zeros_like(x).scatter_(1, max_index, 1)
        if self.matching_score_reg:
            score = self.conv_score(x_)
            return x, score, onehot_index
        else:
            return x, onehot_index

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class pose_guide_att_Resnet(PCB):
    def __init__(self, num_classes, loss, block, layers, last_stride=2, parts=4, reduced_dim=None,
                 nonlinear='relu', pose_inchannel=56, part_score_reg=False, **kwargs):
        super(pose_guide_att_Resnet, self).__init__(num_classes, loss, block, layers,
                                                    last_stride=last_stride, parts=parts,
                                                    reduced_dim=reduced_dim,
                                                    nonlinear=nonlinear, **kwargs)
        self.part_score_reg = part_score_reg
        self.pose_subnet = Pose_Subnet(blocks=[OSBlock, OSBlock], in_channels=pose_inchannel,
                                       channels=[32, 32, 32], att_num=parts, matching_score_reg=part_score_reg)
        self.pose_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.parts_avgpool = nn.ModuleList([nn.AdaptiveAvgPool2d((1, 1)) for _ in range(self.parts)]) # TODO why not use the same?????

    def forward(self, x, pose_map):
        f = self.featuremaps(x)
        if self.part_score_reg:
            pose_att, part_score, onehot_index = self.pose_subnet(pose_map) # TODO
        else:
            pose_att, onehot_index = self.pose_subnet(pose_map)
        pose_att = pose_att * onehot_index
        pose_att_pool = self.pose_pool(pose_att) # AdaptiveAvgPool2d -> (1,1)
        v_g = []
        for i in range(self.parts):
            v_g_i = f * pose_att[:, i, :, :].unsqueeze(1) / (pose_att_pool[:, i, :, :].unsqueeze(1) + 1e-6) # TODO why divide by average?
            v_g_i = self.parts_avgpool[i](v_g_i)
            v_g.append(v_g_i)

        if not self.training:
            v_g = torch.cat(v_g, dim=2)
            v_g = F.normalize(v_g, p=2, dim=1) # TODO apply normalize myself?
            if self.part_score_reg:
                return v_g.squeeze(), part_score
            else:
                return v_g.view(v_g.size(0), -1)
        y = []
        v = []
        for i in range(self.parts): # add final fc layer
            v_g_i = self.em[i](v_g[i])
            v_h_i = v_g_i.view(v_g_i.size(0), -1)
            y_i = self.classifier[i](v_h_i)
            y.append(y_i)
            v.append(v_g_i)

        if self.loss == 'softmax':
            if self.training:
                if self.part_score_reg:
                    return y, pose_att, part_score, v_g
                else:
                    return y, pose_att
            else:
                return y
        elif self.loss == 'triplet':
            return y, v
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


def pose_resnet50_256_p4(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = pose_guide_att_Resnet(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        parts=4,
        reduced_dim=256,
        nonlinear='relu',
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model


def pose_resnet50_256_p6(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = pose_guide_att_Resnet(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        parts=6,
        reduced_dim=256,
        nonlinear='relu',
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model


def pose_resnet50_256_p6_pscore_reg(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = pose_guide_att_Resnet(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        parts=6,
        reduced_dim=256,
        nonlinear='relu',
        part_score_reg=True,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model


def pose_resnet50_256_p4_pscore_reg(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = pose_guide_att_Resnet(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        parts=4,
        reduced_dim=256,
        nonlinear='relu',
        part_score_reg=True,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model
