from __future__ import division, absolute_import

import torch

from torchreid.losses.part_averaged_triplet_loss import PartAveragedTripletLoss
from torchreid.utils.tensortools import replace_values


class PartMinTripletLoss(PartAveragedTripletLoss):

    def __init__(self, **kwargs):
        super(PartMinTripletLoss, self).__init__(**kwargs)

    def _combine_part_based_dist_matrices(self, part_based_pairwise_dist, valid_part_based_pairwise_dist_mask, labels):
        if valid_part_based_pairwise_dist_mask is not None:
            max_value = torch.finfo(part_based_pairwise_dist.dtype).max
            valid_part_based_pairwise_dist = replace_values(part_based_pairwise_dist, ~valid_part_based_pairwise_dist_mask, max_value)
            self.writer.update_invalid_part_based_pairwise_distances_count(valid_part_based_pairwise_dist_mask)
        else:
            valid_part_based_pairwise_dist = part_based_pairwise_dist

        pairwise_dist, part_id = valid_part_based_pairwise_dist.min(0)

        if valid_part_based_pairwise_dist_mask is not None:
            invalid_pairwise_dist_mask = valid_part_based_pairwise_dist_mask.sum(dim=0) == 0
            pairwise_dist = replace_values(pairwise_dist, invalid_pairwise_dist_mask, -1)

        parts_count = part_based_pairwise_dist.shape[0]
        if part_based_pairwise_dist.shape[0] > 1:
            self.writer.used_parts_statistics(parts_count, part_id)

        return pairwise_dist
