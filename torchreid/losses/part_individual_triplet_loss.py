from __future__ import division, absolute_import

from torchreid.losses.part_averaged_triplet_loss import PartAveragedTripletLoss
from torchreid.utils.tensortools import replace_values


class PartIndividualTripletLoss(PartAveragedTripletLoss):
    """A triplet loss applied individually for each part, without considering the global/combined distance
        between two training samples. If the model outputs K embeddings (for K parts), this loss will compute the
        batch-hard triplet loss K times and output the average of them. With the part-averaged triplet loss, the global
        distance between two training samples is used in the triplet loss equation: that global distance is obtained by
        combining all K part-based distance between two samples into one value ('combining' = mean, max, min, etc).
        With the part-individual triplet loss, the triplet loss is applied only on local distance individually, i.e.,
        the distance between two local parts is used in the triplet loss equation. This part-individual triplet loss is
        therefore more sensitive to occluded parts (if 'valid_part_based_pairwise_dist_mask' is not used) and to
        non-discriminative parts, i.e. parts from two different identities having similar appearance.
        'Somers V. & al, Body Part-Based Representation Learning for Occluded Person Re-Identification, WACV23'.
        Source: https://github.com/VlSomers/bpbreid
        """
    def __init__(self, **kwargs):
        super(PartIndividualTripletLoss, self).__init__(**kwargs)

    def _combine_part_based_dist_matrices(self, part_based_pairwise_dist, valid_part_based_pairwise_dist_mask, labels):
        """Do not combine part-based distance, simply return the input part-based pairwise distances, and optionally
        replace non-valid part-based distance with -1"""
        if valid_part_based_pairwise_dist_mask is not None:
            valid_part_based_pairwise_dist = replace_values(part_based_pairwise_dist, ~valid_part_based_pairwise_dist_mask, -1)
            self.writer.update_invalid_part_based_pairwise_distances_count(valid_part_based_pairwise_dist_mask)
        else:
            valid_part_based_pairwise_dist = part_based_pairwise_dist

        return valid_part_based_pairwise_dist
