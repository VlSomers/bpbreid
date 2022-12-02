from __future__ import division, absolute_import
from torchreid.losses.part_averaged_triplet_loss import PartAveragedTripletLoss
import torch


class InterPartsTripletLoss(PartAveragedTripletLoss):

    def __init__(self, **kwargs):
        super(InterPartsTripletLoss, self).__init__(**kwargs)

    def forward(self, body_parts_features, targets, n_iter=0, parts_visibility=None):
        # body_parts_features.shape = [M, N, N]
        body_parts_dist_matrices = self.compute_mixed_body_parts_dist_matrices(body_parts_features)
        return self.hard_mine_triplet_loss(body_parts_dist_matrices, targets)

    def compute_mixed_body_parts_dist_matrices(self, body_parts_features):
        body_parts_features = body_parts_features.flatten(start_dim=0, end_dim=1).unsqueeze(1)
        body_parts_dist_matrices = self._part_based_pairwise_distance_matrix(body_parts_features, False, self.epsilon).squeeze()
        return body_parts_dist_matrices

    def hard_mine_triplet_loss(self, dist, targets): # TODO extract code for mask generation into separate method
        # TODO cleanup
        nm = dist.shape[0]
        n = targets.size(0)
        m = int(nm / n)
        expanded_targets = targets.repeat(m).expand(nm, -1)
        pids_mask = expanded_targets.eq(expanded_targets.t())

        body_parts_targets = []
        for i in range(0, m):
            body_parts_targets.append(torch.full_like(targets, i))
        body_parts_targets = torch.cat(body_parts_targets)
        expanded_body_parts_targets = body_parts_targets.expand(nm, -1)
        body_parts_mask = expanded_body_parts_targets.eq(expanded_body_parts_targets.t())

        mask_p = torch.logical_and(pids_mask, body_parts_mask)
        mask_n = pids_mask == 0

        # Create two big mask of size BxB with B = M*N (one entry per embedding).
        # positive mask cell = true if embedding of same identity and same body part
        # to create that, compute AND between classic pids M*N mask and a new body part mask :
        # 110000 112233.expand().eq(112233.expand().t())
        # 110000
        # 001100
        # 001100
        # 000011
        # 000011
        # negative mask cell = true if embeddings from different ids

        dist_ap, dist_an = [], []
        for i in range(nm):
            i_pos_dist = dist[i][mask_p[i]]
            dist_ap.append(i_pos_dist.max().unsqueeze(0))
            i_neg_dist = dist[i][mask_n[i]]
            assert i_neg_dist.nelement() != 0, "embedding %r should have at least one negative counterpart" % i
            dist_an.append(i_neg_dist.min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)
