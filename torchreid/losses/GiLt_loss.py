from __future__ import division, absolute_import

import torch
import torch.nn as nn
from collections import OrderedDict
from torchmetrics import Accuracy
from torchreid.losses import init_part_based_triplet_loss, CrossEntropyLoss
from torchreid.utils.constants import GLOBAL, FOREGROUND, CONCAT_PARTS, PARTS


class GiLtLoss(nn.Module):
    """ The Global-identity Local-triplet 'GiLt' loss as described in our paper:
    'Somers V. & al, Body Part-Based Representation Learning for Occluded Person Re-Identification, WACV23'.
    Source: https://github.com/VlSomers/bpbreid
    The default weights for the GiLt strategy (as described in the paper) are provided in 'default_losses_weights': the
    identity loss is applied only on holistic embeddings and the triplet loss is applied only on part-based embeddings.
    'tr' denotes 'triplet' for the triplet loss and 'id' denotes 'identity' for the identity cross-entropy loss.
    """

    default_losses_weights = {
        GLOBAL: {'id': 1., 'tr': 0.},
        FOREGROUND: {'id': 1., 'tr': 0.},
        CONCAT_PARTS: {'id': 1., 'tr': 0.},
        PARTS: {'id': 0., 'tr': 1.}
    }

    def __init__(self,
                 losses_weights=None,
                 use_visibility_scores=False,
                 triplet_margin=0.3,
                 loss_name='part_averaged_triplet_loss',
                 use_gpu=False,
                 writer=None):
        super().__init__()
        if losses_weights is None:
            losses_weights = self.default_losses_weights
        self.pred_accuracy = Accuracy(top_k=1)
        if use_gpu:
            self.pred_accuracy = self.pred_accuracy.cuda()
        self.losses_weights = losses_weights
        self.part_triplet_loss = init_part_based_triplet_loss(loss_name, margin=triplet_margin, writer=writer)
        self.identity_loss = CrossEntropyLoss(label_smooth=True)
        self.use_visibility_scores = use_visibility_scores

    def forward(self, embeddings_dict, visibility_scores_dict, id_cls_scores_dict, pids):
        """
        Keys in the input dictionaries are from {'globl', 'foreg', 'conct', 'parts'} and correspond to the different
        types of embeddings. In the documentation below, we denote the batch size by 'N' and the number of parts by 'K'.
        :param embeddings_dict: a dictionary of embeddings, where the keys are the embedding types and the values are
            Tensors of size [N, D] or [N, K*D] or [N, K, D].
        :param visibility_scores_dict: a dictionary of visibility scores, where the keys are the embedding types and the
            values are Tensors of size [N] or [N, K].
        :param id_cls_scores_dict: a dictionary of identity classification scores, where the keys are the embedding types
            and the values are Tensors of size [N, num_classes] or [N, K, num_classes]
        :param pids: A Tensor of size [N] containing the person IDs.
        :return: a tupel with the total combined loss and a dictionnary with performance information for each individual
            loss.
        """
        loss_summary = {}
        losses = []
        # global, foreground and parts embeddings id loss
        for key in [GLOBAL, FOREGROUND, CONCAT_PARTS, PARTS]:
            loss_info = OrderedDict() if key not in loss_summary else loss_summary[key]
            ce_w = self.losses_weights[key]['id']
            if ce_w > 0:
                parts_id_loss, parts_id_accuracy = self.compute_id_cls_loss(id_cls_scores_dict[key],
                                                                            visibility_scores_dict[key], pids)
                losses.append((ce_w, parts_id_loss))
                loss_info['c'] = parts_id_loss
                loss_info['a'] = parts_id_accuracy

            loss_summary[key] = loss_info

        # global, foreground and parts embeddings triplet loss
        for key in [GLOBAL, FOREGROUND, CONCAT_PARTS, PARTS]:
            loss_info = OrderedDict() if key not in loss_summary else loss_summary[key]
            tr_w = self.losses_weights[key]['tr']
            if tr_w > 0:
                parts_triplet_loss, parts_trivial_triplets_ratio, parts_valid_triplets_ratio = \
                    self.compute_triplet_loss(embeddings_dict[key], visibility_scores_dict[key], pids)
                losses.append((tr_w, parts_triplet_loss))
                loss_info['t'] = parts_triplet_loss
                loss_info['tt'] = parts_trivial_triplets_ratio
                loss_info['vt'] = parts_valid_triplets_ratio

            loss_summary[key] = loss_info

        # weighted sum of all losses
        if len(losses) == 0:
            return torch.tensor(0., device=(pids.get_device() if pids.is_cuda else None)), loss_summary
        else:
            loss = torch.stack([weight * loss for weight, loss in losses]).sum()
            return loss, loss_summary

    def compute_triplet_loss(self, embeddings, visibility_scores, pids):
        if self.use_visibility_scores:
            visibility = visibility_scores if len(visibility_scores.shape) == 2 else visibility_scores.unsqueeze(1)
        else:
            visibility = None
        embeddings = embeddings if len(embeddings.shape) == 3 else embeddings.unsqueeze(1)
        triplet_loss, trivial_triplets_ratio, valid_triplets_ratio = self.part_triplet_loss(embeddings, pids,
                                                                                            parts_visibility=visibility)
        return triplet_loss, trivial_triplets_ratio, valid_triplets_ratio

    def compute_id_cls_loss(self, id_cls_scores, visibility_scores, pids):
        if len(id_cls_scores.shape) == 3:
            M = id_cls_scores.shape[1]
            id_cls_scores = id_cls_scores.flatten(0, 1)
            pids = pids.unsqueeze(1).expand(-1, M).flatten(0, 1)
            visibility_scores = visibility_scores.flatten(0, 1)
        weights = None
        if self.use_visibility_scores and visibility_scores.dtype is torch.bool:
            id_cls_scores = id_cls_scores[visibility_scores]
            pids = pids[visibility_scores]
        elif self.use_visibility_scores and visibility_scores.dtype is not torch.bool:
            weights = visibility_scores
        cls_loss = self.identity_loss(id_cls_scores, pids, weights)
        accuracy = self.pred_accuracy(id_cls_scores, pids)
        return cls_loss, accuracy
