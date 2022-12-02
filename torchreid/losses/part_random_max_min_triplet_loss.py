from __future__ import division, absolute_import

import torch

from torchreid.losses.part_averaged_triplet_loss import PartAveragedTripletLoss
from torchreid.utils.tensortools import replace_values


class PartRandomMaxMinTripletLoss(PartAveragedTripletLoss):

    def __init__(self, **kwargs):
        super(PartRandomMaxMinTripletLoss, self).__init__(**kwargs)

    def _combine_part_based_dist_matrices(self, part_based_pairwise_dist, valid_part_based_pairwise_dist_mask, labels):
        if valid_part_based_pairwise_dist_mask is None:
            valid_part_based_pairwise_dist_mask = torch.ones(part_based_pairwise_dist.shape, dtype=torch.bool, device=(labels.get_device() if labels.is_cuda else None))
        # put some random entries to 0 (20%)
        dropout_mask = torch.rand(size=valid_part_based_pairwise_dist_mask.shape, device=(labels.get_device() if labels.is_cuda else None)) > 0.5  # FIXME should be symmetric?
        # dropout_mask = torch.rand(size=valid_body_part_pairwise_dist_mask.shape[0:2], device=(labels.get_device() if labels.is_cuda else None)) > 0.3
        # dropout_mask = dropout_mask.unsqueeze(1) * dropout_mask.unsqueeze(2)

        valid_part_based_pairwise_dist_mask *= dropout_mask

        valid_part_based_pairwise_dist_for_max = replace_values(part_based_pairwise_dist, ~valid_part_based_pairwise_dist_mask, -1)
        self.writer.update_invalid_part_based_pairwise_distances_count(valid_part_based_pairwise_dist_mask)
        max_pairwise_dist, part_id_for_max = valid_part_based_pairwise_dist_for_max.max(0)

        max_value = torch.finfo(part_based_pairwise_dist.dtype).max
        valid_part_based_pairwise_dist_for_min = replace_values(part_based_pairwise_dist,
                                                               ~valid_part_based_pairwise_dist_mask, max_value)
        self.writer.update_invalid_part_based_pairwise_distances_count(valid_part_based_pairwise_dist_mask)
        min_pairwise_dist, part_id_for_min = valid_part_based_pairwise_dist_for_min.min(0)

        labels_equal_mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1))
        pairwise_dist = max_pairwise_dist * labels_equal_mask + min_pairwise_dist * ~labels_equal_mask
        part_id = part_id_for_max * labels_equal_mask + part_id_for_min * ~labels_equal_mask

        invalid_pairwise_dist_mask = valid_part_based_pairwise_dist_mask.sum(dim=0) == 0
        pairwise_dist = replace_values(pairwise_dist, invalid_pairwise_dist_mask, -1)

        if part_based_pairwise_dist.shape[0] > 1:
            self.writer.used_parts_statistics(part_based_pairwise_dist.shape[0], part_id)

        return pairwise_dist
