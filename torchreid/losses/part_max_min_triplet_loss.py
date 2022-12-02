from __future__ import division, absolute_import

import torch

from torchreid.losses.part_averaged_triplet_loss import PartAveragedTripletLoss
from torchreid.utils.tensortools import replace_values


class PartMaxMinTripletLoss(PartAveragedTripletLoss):

    def __init__(self, **kwargs):
        super(PartMaxMinTripletLoss, self).__init__(**kwargs)

    def _combine_part_based_dist_matrices(self, part_based_pairwise_dist, valid_part_based_pairwise_dist_mask, labels):
        if valid_part_based_pairwise_dist_mask is not None:
            valid_part_based_pairwise_dist_for_max = replace_values(part_based_pairwise_dist, ~valid_part_based_pairwise_dist_mask, -1)
            self.writer.update_invalid_part_based_pairwise_distances_count(valid_part_based_pairwise_dist_mask)
        else:
            valid_part_based_pairwise_dist_for_max = part_based_pairwise_dist

        max_pairwise_dist, part_id_for_max = valid_part_based_pairwise_dist_for_max.max(0)

        if valid_part_based_pairwise_dist_mask is not None:
            max_value = torch.finfo(part_based_pairwise_dist.dtype).max
            valid_part_based_pairwise_dist_for_min = replace_values(part_based_pairwise_dist,
                                                                   ~valid_part_based_pairwise_dist_mask, max_value)
            self.writer.update_invalid_part_based_pairwise_distances_count(valid_part_based_pairwise_dist_mask)
        else:
            valid_part_based_pairwise_dist_for_min = part_based_pairwise_dist

        min_pairwise_dist, part_id_for_min = valid_part_based_pairwise_dist_for_min.min(0)

        labels_equal_mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1))
        pairwise_dist = max_pairwise_dist * labels_equal_mask + min_pairwise_dist * ~labels_equal_mask
        part_id = part_id_for_max * labels_equal_mask + part_id_for_min * ~labels_equal_mask

        if valid_part_based_pairwise_dist_mask is not None:
            invalid_pairwise_dist_mask = valid_part_based_pairwise_dist_mask.sum(dim=0) == 0
            pairwise_dist = replace_values(pairwise_dist, invalid_pairwise_dist_mask, -1)

        if part_based_pairwise_dist.shape[0] > 1:
            self.writer.used_parts_statistics(part_based_pairwise_dist.shape[0], part_id)

        return pairwise_dist
