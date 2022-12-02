from __future__ import division, absolute_import

import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from torchreid import models
from torchreid.utils.constants import *

__all__ = [
    'bpbreid'
]


class BPBreID(nn.Module):
    """Posed based feature extraction network
    """
    def __init__(self, num_classes, pretrained, loss, model_cfg, horizontal_stripes=False, **kwargs):
        super(BPBreID, self).__init__()

        # Init config
        self.model_cfg = model_cfg
        # number of training classes/identities
        self.num_classes = num_classes
        # number of parts K
        self.parts_num = self.model_cfg.masks.parts_num
        # whether to perform horizontal stripes pooling similar to PCB
        self.horizontal_stripes = horizontal_stripes
        # use shared weights/parameters between each part branch for the identity classifier
        self.shared_parts_id_classifier = self.model_cfg.shared_parts_id_classifier
        # at test time, perform a 'soft' or 'hard' merging of the learned attention maps with the external part masks
        self.test_use_target_segmentation = self.model_cfg.test_use_target_segmentation
        # use continuous or binary visibility scores at train time:
        self.training_binary_visibility_score = self.model_cfg.training_binary_visibility_score
        # use continuous or binary visibility scores at test time:
        self.testing_binary_visibility_score = self.model_cfg.testing_binary_visibility_score

        # Init backbone feature extractor
        self.backbone_appearance_feature_extractor = models.build_model(self.model_cfg.backbone,
                                                                        num_classes,
                                                                        loss=loss,
                                                                        pretrained=pretrained,
                                                                        last_stride=self.model_cfg.last_stride,
                                                                        enable_dim_reduction=(self.model_cfg.dim_reduce=='before_pooling'),
                                                                        dim_reduction_channels=self.model_cfg.dim_reduce_output,
                                                                        pretrained_path=self.model_cfg.hrnet_pretrained_path
                                                                        )
        self.spatial_feature_size = self.backbone_appearance_feature_extractor.feature_dim

        # Init dim reduce layers
        self.init_dim_reduce_layers(self.model_cfg.dim_reduce,
                                    self.spatial_feature_size,
                                    self.model_cfg.dim_reduce_output)

        # Init pooling layers
        self.global_pooling_head = nn.AdaptiveAvgPool2d(1)
        self.foreground_attention_pooling_head = GlobalAveragePoolingHead(self.dim_reduce_output)
        self.background_attention_pooling_head = GlobalAveragePoolingHead(self.dim_reduce_output)
        self.parts_attention_pooling_head = init_part_attention_pooling_head(self.model_cfg.normalization,
                                                                             self.model_cfg.pooling,
                                                                             self.dim_reduce_output)

        # Init parts classifier
        self.learnable_attention_enabled = self.model_cfg.learnable_attention_enabled
        self.pixel_classifier = PixelToPartClassifier(self.spatial_feature_size, self.parts_num)

        # Init id classifier
        self.global_identity_classifier = BNClassifier(self.dim_reduce_output, self.num_classes)
        self.background_identity_classifier = BNClassifier(self.dim_reduce_output, self.num_classes)
        self.foreground_identity_classifier = BNClassifier(self.dim_reduce_output, self.num_classes)
        self.concat_parts_identity_classifier = BNClassifier(self.parts_num * self.dim_reduce_output, self.num_classes)
        if self.shared_parts_id_classifier:
            # the same identity classifier weights are used for each part branch
            self.parts_identity_classifier = BNClassifier(self.dim_reduce_output, self.num_classes)
        else:
            # each part branch has its own identity classifier
            self.parts_identity_classifier = nn.ModuleList(
                [
                    BNClassifier(self.dim_reduce_output, self.num_classes)
                    for _ in range(self.parts_num)
                ]
            )

    def init_dim_reduce_layers(self, dim_reduce_mode, spatial_feature_size, dim_reduce_output):
        self.dim_reduce_output = dim_reduce_output
        self.after_pooling_dim_reduce = False
        self.before_pooling_dim_reduce = None

        if dim_reduce_mode == 'before_pooling':
            self.before_pooling_dim_reduce = BeforePoolingDimReduceLayer(spatial_feature_size, dim_reduce_output)
            self.spatial_feature_size = dim_reduce_output
        elif dim_reduce_mode == 'after_pooling':
            self.after_pooling_dim_reduce = True
            self.global_after_pooling_dim_reduce = AfterPoolingDimReduceLayer(spatial_feature_size, dim_reduce_output)
            self.foreground_after_pooling_dim_reduce = AfterPoolingDimReduceLayer(spatial_feature_size, dim_reduce_output)
            self.background_after_pooling_dim_reduce = AfterPoolingDimReduceLayer(spatial_feature_size, dim_reduce_output)
            self.parts_after_pooling_dim_reduce = AfterPoolingDimReduceLayer(spatial_feature_size, dim_reduce_output)
        elif dim_reduce_mode == 'before_and_after_pooling':
            self.before_pooling_dim_reduce = BeforePoolingDimReduceLayer(spatial_feature_size, dim_reduce_output * 2)
            spatial_feature_size = dim_reduce_output * 2
            self.spatial_feature_size = spatial_feature_size
            self.after_pooling_dim_reduce = True
            self.global_after_pooling_dim_reduce = AfterPoolingDimReduceLayer(spatial_feature_size, dim_reduce_output)
            self.foreground_after_pooling_dim_reduce = AfterPoolingDimReduceLayer(spatial_feature_size, dim_reduce_output)
            self.background_after_pooling_dim_reduce = AfterPoolingDimReduceLayer(spatial_feature_size, dim_reduce_output)
            self.parts_after_pooling_dim_reduce = AfterPoolingDimReduceLayer(spatial_feature_size, dim_reduce_output)
        elif dim_reduce_mode == 'after_pooling_with_dropout':
            self.after_pooling_dim_reduce = True
            self.global_after_pooling_dim_reduce = AfterPoolingDimReduceLayer(spatial_feature_size, dim_reduce_output, 0.5)
            self.foreground_after_pooling_dim_reduce = AfterPoolingDimReduceLayer(spatial_feature_size, dim_reduce_output, 0.5)
            self.background_after_pooling_dim_reduce = AfterPoolingDimReduceLayer(spatial_feature_size, dim_reduce_output, 0.5)
            self.parts_after_pooling_dim_reduce = AfterPoolingDimReduceLayer(spatial_feature_size, dim_reduce_output, 0.5)
        else:
            self.dim_reduce_output = spatial_feature_size

    def forward(self, images, external_parts_masks=None):
        """
        :param images: images tensor of size [N, C, Hi, Wi], where N is the batch size, C channel depth (3 for RGB), and
            (Hi, Wi) are the image height and width.
        :param external_parts_masks: masks tensor of size [N, K+1, Hm, Wm], where N is the batch size, K is the number
            parts, and (Hm, Wm) are the masks height and width. The first index (index 0) along the parts K+1 dimension
            is the background by convention. The masks are expected to have values in the range [0, 1]. Spatial entry at
            location external_parts_masks[i, k+1, h, w] is the probability that the pixel at location (h, w) belongs to
            part k for batch sample i. The masks are NOT expected to be of the same size as the images.
        :return:
        """

        # Global spatial_features
        spatial_features = self.backbone_appearance_feature_extractor(images)  # [N, D, Hf, Wf]
        N, _, Hf, Wf = spatial_features.shape

        if self.before_pooling_dim_reduce is not None \
                and spatial_features.shape[1] != self.dim_reduce_output:  # When HRNet used as backbone, already done
            spatial_features = self.before_pooling_dim_reduce(spatial_features)  # [N, dim_reduce_output, Hf, Wf]

        # Pixels classification and parts attention weights
        if self.horizontal_stripes:
            pixels_cls_scores = None
            feature_map_shape = (Hf, Wf)
            stripes_range = np.round(np.arange(0, self.parts_num + 1) * feature_map_shape[0] / self.parts_num).astype(int)
            pcb_masks = torch.zeros((self.parts_num, feature_map_shape[0], feature_map_shape[1]))
            for i in range(0, stripes_range.size - 1):
                pcb_masks[i, stripes_range[i]:stripes_range[i + 1], :] = 1
            pixels_parts_probabilities = pcb_masks
            pixels_parts_probabilities.requires_grad = False
        elif self.learnable_attention_enabled:
            pixels_cls_scores = self.pixel_classifier(spatial_features)  # [N, K, Hf, Wf]
            pixels_parts_probabilities = F.softmax(pixels_cls_scores, dim=1)
        else:
            pixels_cls_scores = None
            assert external_parts_masks is not None
            external_parts_masks = external_parts_masks.type(spatial_features.dtype)
            pixels_parts_probabilities = nn.functional.interpolate(external_parts_masks, (Hf, Wf), mode='bilinear', align_corners=True)
            pixels_parts_probabilities.requires_grad = False
            assert pixels_parts_probabilities.max() <= 1 and pixels_parts_probabilities.min() >= 0

        background_masks = pixels_parts_probabilities[:, 0]
        parts_masks = pixels_parts_probabilities[:, 1:]

        # Explicit pixels segmentation of re-id target using external part masks
        if not self.training and self.test_use_target_segmentation == 'hard':
            assert external_parts_masks is not None
            # hard masking
            external_parts_masks = nn.functional.interpolate(external_parts_masks, (Hf, Wf), mode='bilinear',
                                                                   align_corners=True)
            target_segmentation_mask = external_parts_masks[:, 1::].max(dim=1)[0] > external_parts_masks[:, 0]
            background_masks = ~target_segmentation_mask
            parts_masks[background_masks.unsqueeze(1).expand_as(parts_masks)] = 1e-12

        if not self.training and self.test_use_target_segmentation == 'soft':
            assert external_parts_masks is not None
            # soft masking
            external_parts_masks = nn.functional.interpolate(external_parts_masks, (Hf, Wf), mode='bilinear',
                                                                   align_corners=True)
            parts_masks = parts_masks * external_parts_masks[:, 1::]

        # foreground_masks = parts_masks.sum(dim=1)
        foreground_masks = parts_masks.max(dim=1)[0]
        global_masks = torch.ones_like(foreground_masks)

        # Parts visibility
        if (self.training and self.training_binary_visibility_score) or (not self.training and self.testing_binary_visibility_score):
            pixels_parts_predictions = pixels_parts_probabilities.argmax(dim=1)  # [N, Hf, Wf]
            pixels_parts_predictions_one_hot = F.one_hot(pixels_parts_predictions, self.parts_num + 1).permute(0, 3, 1, 2)  # [N, K+1, Hf, Wf]
            parts_visibility = pixels_parts_predictions_one_hot.amax(dim=(2, 3)).to(torch.bool)  # [N, K+1]
        else:
            parts_visibility = pixels_parts_probabilities.amax(dim=(2, 3))  # [N, K+1]
        background_visibility = parts_visibility[:, 0]  # [N]
        foreground_visibility = parts_visibility.amax(dim=1)  # [N]
        parts_visibility = parts_visibility[:, 1:]  # [N, K]
        concat_parts_visibility = foreground_visibility
        global_visibility = torch.ones_like(foreground_visibility)  # [N]

        # Global embedding
        global_embeddings = self.global_pooling_head(spatial_features).view(N, -1)  # [N, D]

        # Foreground and background embeddings
        foreground_embeddings = self.foreground_attention_pooling_head(spatial_features, foreground_masks.unsqueeze(1)).flatten(1, 2)  # [N, D]
        background_embeddings = self.background_attention_pooling_head(spatial_features, background_masks.unsqueeze(1)).flatten(1, 2)  # [N, D]

        # Part features
        parts_embeddings = self.parts_attention_pooling_head(spatial_features, parts_masks)  # [N, K, D]

        # Dim reduction
        if self.after_pooling_dim_reduce:
            global_embeddings = self.global_after_pooling_dim_reduce(global_embeddings)  # [N, D]
            foreground_embeddings = self.foreground_after_pooling_dim_reduce(foreground_embeddings)  # [N, D]
            background_embeddings = self.background_after_pooling_dim_reduce(background_embeddings)  # [N, D]
            parts_embeddings = self.parts_after_pooling_dim_reduce(parts_embeddings)  # [N, M, D]

        # Concatenated part features
        concat_parts_embeddings = parts_embeddings.flatten(1, 2)  # [N, K*D]

        # Identity classification scores
        bn_global_embeddings, global_cls_score = self.global_identity_classifier(global_embeddings)  # [N, D], [N, num_classes]
        bn_background_embeddings, background_cls_score = self.background_identity_classifier(background_embeddings)  # [N, D], [N, num_classes]
        bn_foreground_embeddings, foreground_cls_score = self.foreground_identity_classifier(foreground_embeddings)  # [N, D], [N, num_classes]
        bn_concat_parts_embeddings, concat_parts_cls_score = self.concat_parts_identity_classifier(concat_parts_embeddings)  # [N, K*D], [N, num_classes]
        bn_parts_embeddings, parts_cls_score = self.parts_identity_classification(self.dim_reduce_output, N, parts_embeddings)  # [N, K, D], [N, K, num_classes]

        # Outputs
        embeddings = {
            GLOBAL: global_embeddings,  # [N, D]
            BACKGROUND: background_embeddings,  # [N, D]
            FOREGROUND: foreground_embeddings,  # [N, D]
            CONCAT_PARTS: concat_parts_embeddings,  # [N, K*D]
            PARTS: parts_embeddings,  # [N, K, D]
            BN_GLOBAL: bn_global_embeddings,  # [N, D]
            BN_BACKGROUND: bn_background_embeddings,  # [N, D]
            BN_FOREGROUND: bn_foreground_embeddings,  # [N, D]
            BN_CONCAT_PARTS: bn_concat_parts_embeddings,  # [N, K*D]
            BN_PARTS: bn_parts_embeddings,  #  [N, K, D]
        }

        visibility_scores = {
            GLOBAL: global_visibility,  # [N]
            BACKGROUND: background_visibility,  # [N]
            FOREGROUND: foreground_visibility,  # [N]
            CONCAT_PARTS: concat_parts_visibility,  # [N]
            PARTS: parts_visibility,  # [N, K]
        }

        id_cls_scores = {
            GLOBAL: global_cls_score,  # [N, num_classes]
            BACKGROUND: background_cls_score,  # [N, num_classes]
            FOREGROUND: foreground_cls_score,  # [N, num_classes]
            CONCAT_PARTS: concat_parts_cls_score,  # [N, num_classes]
            PARTS: parts_cls_score,  # [N, K, num_classes]
        }

        masks = {
            GLOBAL: global_masks,  # [N, Hf, Wf]
            BACKGROUND: background_masks,  # [N, Hf, Wf]
            FOREGROUND: foreground_masks,  # [N, Hf, Wf]
            CONCAT_PARTS: foreground_masks,  # [N, Hf, Wf]
            PARTS: parts_masks,  # [N, K, Hf, Wf]
        }

        return embeddings, visibility_scores, id_cls_scores, pixels_cls_scores, spatial_features, masks

    def parts_identity_classification(self, D, N, parts_embeddings):
        if self.shared_parts_id_classifier:
            # apply the same classifier on each part embedding, classifier weights are therefore shared across parts
            parts_embeddings = parts_embeddings.flatten(0, 1)  # [N*K, D]
            bn_part_embeddings, part_cls_score = self.parts_identity_classifier(parts_embeddings)
            bn_part_embeddings = bn_part_embeddings.view([N, self.parts_num, D])
            part_cls_score = part_cls_score.view([N, self.parts_num, -1])
        else:
            # apply K classifiers on each of the K part embedding, each part has therefore it's own classifier weights
            scores = []
            embeddings = []
            for i, parts_identity_classifier in enumerate(self.parts_identity_classifier):
                bn_part_embeddings, part_cls_score = parts_identity_classifier(parts_embeddings[:, i])
                scores.append(part_cls_score.unsqueeze(1))
                embeddings.append(bn_part_embeddings.unsqueeze(1))
            part_cls_score = torch.cat(scores, 1)
            bn_part_embeddings = torch.cat(embeddings, 1)

        return bn_part_embeddings, part_cls_score


########################################
#    Dimensionality reduction layers   #
########################################

class BeforePoolingDimReduceLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BeforePoolingDimReduceLayer, self).__init__()
        layers = []
        layers.append(
            nn.Conv2d(
                input_dim, output_dim, 1, stride=1, padding=0
            )
        )
        layers.append(nn.BatchNorm2d(output_dim))
        layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)
        self._init_params()

    def forward(self, x):
        return self.layers(x)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
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


class AfterPoolingDimReduceLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_p=None):
        super(AfterPoolingDimReduceLayer, self).__init__()
        # dim reduction used in ResNet and PCB
        layers = []
        layers.append(
            nn.Linear(
                input_dim, output_dim, bias=True
            )
        )
        layers.append(nn.BatchNorm1d(output_dim))
        layers.append(nn.ReLU(inplace=True))
        if dropout_p is not None:
            layers.append(nn.opout(p=dropout_p))

        self.layers = nn.Sequential(*layers)
        self._init_params()

    def forward(self, x):
        if len(x.size()) == 3:
            N, K, _ = x.size()  # [N, K, input_dim]
            x = x.flatten(0, 1)
            x = self.layers(x)
            x = x.view(N, K, -1)
        else:
            x = self.layers(x)
        return x

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
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


########################################
#             Classifiers              #
########################################

class PixelToPartClassifier(nn.Module):
    def __init__(self, dim_reduce_output, parts_num):
        super(PixelToPartClassifier, self).__init__()
        self.bn = torch.nn.BatchNorm2d(dim_reduce_output)
        self.classifier = nn.Conv2d(in_channels=dim_reduce_output, out_channels=parts_num + 1, kernel_size=1, stride=1, padding=0)
        self._init_params()

    def forward(self, x):
        x = self.bn(x)
        return self.classifier(x)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.001)  # ResNet = 0.01, Bof and ISP-reid = 0.001
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class BNClassifier(nn.Module):
    # Source: https://github.com/upgirlnana/Pytorch-Person-REID-Baseline-Bag-of-Tricks
    def __init__(self, in_dim, class_num):
        super(BNClassifier, self).__init__()

        self.in_dim = in_dim
        self.class_num = class_num

        self.bn = nn.BatchNorm1d(self.in_dim)
        self.bn.bias.requires_grad_(False)  # BoF: this doesn't have a big impact on perf according to author on github
        self.classifier = nn.Linear(self.in_dim, self.class_num, bias=False)

        self._init_params()

    def forward(self, x):
        feature = self.bn(x)
        cls_score = self.classifier(feature)
        return feature, cls_score

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.001)  # ResNet = 0.01, Bof and ISP-reid = 0.001
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


########################################
#            Pooling heads             #
########################################

def init_part_attention_pooling_head(normalization, pooling, dim_reduce_output):
    if pooling == 'gap':
        parts_attention_pooling_head = GlobalAveragePoolingHead(dim_reduce_output, normalization)
    elif pooling == 'gmp':
        parts_attention_pooling_head = GlobalMaxPoolingHead(dim_reduce_output, normalization)
    elif pooling == 'gwap':
        parts_attention_pooling_head = GlobalWeightedAveragePoolingHead(dim_reduce_output, normalization)
    else:
        raise ValueError('pooling type {} not supported'.format(pooling))
    return parts_attention_pooling_head


class GlobalMaskWeightedPoolingHead(nn.Module):
    def __init__(self, depth, normalization='identity'):
        super().__init__()
        if normalization == 'identity':
            self.normalization = nn.Identity()
        elif normalization == 'batch_norm_3d':
            self.normalization = torch.nn.BatchNorm3d(depth, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        elif normalization == 'batch_norm_2d':
            self.normalization = torch.nn.BatchNorm2d(depth, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        elif normalization == 'batch_norm_1d':
            self.normalization = torch.nn.BatchNorm1d(depth, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        else:
            raise ValueError('normalization type {} not supported'.format(normalization))

    def forward(self, features, part_masks):
        part_masks = torch.unsqueeze(part_masks, 2)
        features = torch.unsqueeze(features, 1)
        parts_features = torch.mul(part_masks, features)

        N, M, _, _, _ = parts_features.size()
        parts_features = parts_features.flatten(0, 1)
        parts_features = self.normalization(parts_features)
        parts_features = self.global_pooling(parts_features)
        parts_features = parts_features.view(N, M, -1)
        return parts_features

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.001)  # ResNet = 0.01, Bof and ISP-reid = 0.001
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class GlobalMaxPoolingHead(GlobalMaskWeightedPoolingHead):
    global_pooling = nn.AdaptiveMaxPool2d((1, 1))


class GlobalAveragePoolingHead(GlobalMaskWeightedPoolingHead):
    global_pooling = nn.AdaptiveAvgPool2d((1, 1))


class GlobalWeightedAveragePoolingHead(GlobalMaskWeightedPoolingHead):
    def forward(self, features, part_masks):
        part_masks = torch.unsqueeze(part_masks, 2)
        features = torch.unsqueeze(features, 1)
        parts_features = torch.mul(part_masks, features)

        N, M, _, _, _ = parts_features.size()
        parts_features = parts_features.flatten(0, 1)
        parts_features = self.normalization(parts_features)
        parts_features = torch.sum(parts_features, dim=(-2, -1))
        part_masks_sum = torch.sum(part_masks.flatten(0, 1), dim=(-2, -1))
        part_masks_sum = torch.clamp(part_masks_sum, min=1e-6)
        parts_features_avg = torch.div(parts_features, part_masks_sum)
        parts_features = parts_features_avg.view(N, M, -1)
        return parts_features


########################################
#             Constructors             #
########################################

def bpbreid(num_classes, loss='part_based', pretrained=True, config=None, **kwargs):
    model = BPBreID(
        num_classes,
        pretrained,
        loss,
        config.model.bpbreid,
        **kwargs
    )
    return model


def pcb(num_classes, loss='part_based', pretrained=True, config=None, **kwargs):
    config.model.bpbreid.learnable_attention_enabled = False
    model = BPBreID(
        num_classes,
        pretrained,
        loss,
        config.model.bpbreid,
        horizontal_stipes=True,
        config=config,
        **kwargs
    )
    return model


def bot(num_classes, loss='part_based', pretrained=True, config=None, **kwargs):
    config.model.bpbreid.masks.parts_num = 1
    config.model.bpbreid.learnable_attention_enabled = False
    model = BPBreID(
        num_classes,
        pretrained,
        loss,
        config.model.bpbreid,
        horizontal_stipes=True,
        config=config,
        **kwargs
    )
    return model
