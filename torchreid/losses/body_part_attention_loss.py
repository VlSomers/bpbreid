from __future__ import division, absolute_import

import torch.nn as nn
from collections import OrderedDict
from monai.losses import FocalLoss, DiceLoss
from torch.nn import CrossEntropyLoss
from torchmetrics import Accuracy
from torchreid.utils.constants import PIXELS


class BodyPartAttentionLoss(nn.Module):
    """ A body part attention loss as described in our paper
    'Somers V. & al, Body Part-Based Representation Learning for Occluded Person Re-Identification, WACV23'.
    Source: https://github.com/VlSomers/bpbreid
    """

    def __init__(self, loss_type='cl', label_smoothing=0.1, use_gpu=False):
        super().__init__()
        self.pred_accuracy = Accuracy(top_k=1)
        if use_gpu:
            self.pred_accuracy = self.pred_accuracy.cuda()
        if loss_type == 'cl':
            self.part_prediction_loss_1 = CrossEntropyLoss(label_smoothing=label_smoothing)
        elif loss_type == 'fl':
            self.part_prediction_loss = FocalLoss(to_onehot_y=True, gamma=1.0)
        elif loss_type == 'dl':
            self.part_prediction_loss = DiceLoss(to_onehot_y=True, softmax=True)
        else:
            raise ValueError("Loss {} for part prediction is not supported".format(loss_type))

    def forward(self, pixels_cls_scores, targets):
        """ Compute loss for body part attention prediction.
            Args:
                pixels_cls_scores [N, K, H, W]
                targets [N, H, W]
            Returns:
        """
        loss_summary = {}
        loss_summary[PIXELS] = OrderedDict()
        pixels_cls_loss, pixels_cls_accuracy = self.compute_pixels_cls_loss(pixels_cls_scores, targets)
        loss_summary[PIXELS]['c'] = pixels_cls_loss
        loss_summary[PIXELS]['a'] = pixels_cls_accuracy
        return pixels_cls_loss, loss_summary

    def compute_pixels_cls_loss(self, pixels_cls_scores, targets):
        if pixels_cls_scores.is_cuda:
            targets = targets.cuda()
        pixels_cls_score_targets = targets.flatten()  # [N*Hf*Wf]
        pixels_cls_scores = pixels_cls_scores.permute(0, 2, 3, 1).flatten(0, 2)  # [N*Hf*Wf, M]
        loss = self.part_prediction_loss_1(pixels_cls_scores, pixels_cls_score_targets)
        accuracy = self.pred_accuracy(pixels_cls_scores, pixels_cls_score_targets)
        return loss, accuracy.item()
