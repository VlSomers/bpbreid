from __future__ import division, absolute_import
import torch
import torch.nn as nn


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    
    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.
    
    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """

        # Compute pairwise distance
        dist = self.compute_dist_matrix(inputs)

        # For each anchor, find the hardest positive and negative
        return self.compute_hard_mine_triplet_loss(dist, inputs, targets)

    def compute_hard_mine_triplet_loss(self, dist, inputs, targets):
        n = inputs.size(0)
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)

    def compute_dist_matrix(self, inputs):
        n = inputs.size(0)
        # dist(a, b) = sqrt(sum((a_i - b_i)^2)) = sqrt(sum(a_i^2) + sum(b_i^2) - 2*sum(a_i*b_i))
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()  # sum(a_i^2) + sum(b_i^2)
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)  # sum(a_i^2) + sum(b_i^2) - 2*sum(a_i*b_i)
        dist = dist.clamp(min=1e-12)  # for numerical stability
        dist = dist.sqrt()  # sqrt(sum(a_i^2) + sum(b_i^2) - 2*sum(a_i*b_i))
        return dist
