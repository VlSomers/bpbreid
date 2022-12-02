from __future__ import division, print_function, absolute_import

from .inter_parts_triplet_loss import InterPartsTripletLoss
from .part_averaged_triplet_loss import PartAveragedTripletLoss
from .part_max_triplet_loss import PartMaxTripletLoss
from .part_max_min_triplet_loss import PartMaxMinTripletLoss
from .part_averaged_triplet_loss import PartAveragedTripletLoss
from .part_min_triplet_loss import PartMinTripletLoss
from .part_random_max_min_triplet_loss import PartRandomMaxMinTripletLoss
from .part_individual_triplet_loss import PartIndividualTripletLoss
from .cross_entropy_loss import CrossEntropyLoss
from .hard_mine_triplet_loss import TripletLoss

__body_parts_losses = {
    'part_averaged_triplet_loss': PartAveragedTripletLoss,  # Part-Averaged triplet loss described in the paper
    'part_max_triplet_loss': PartMaxTripletLoss,
    'part_min_triplet_loss': PartMinTripletLoss,
    'part_max_min_triplet_loss': PartMaxMinTripletLoss,
    'part_random_max_min_triplet_loss': PartRandomMaxMinTripletLoss,
    'inter_parts_triplet_loss': InterPartsTripletLoss,
    'intra_parts_triplet_loss': PartIndividualTripletLoss,
}

def init_part_based_triplet_loss(name, **kwargs):
    """Initializes the part based triplet loss based on the part-based distance combination strategy."""
    avai_body_parts_losses = list(__body_parts_losses.keys())
    if name not in avai_body_parts_losses:
        raise ValueError(
            'Invalid loss name. Received "{}", '
            'but expected to be one of {}'.format(name, avai_body_parts_losses)
        )
    return __body_parts_losses[name](**kwargs)


def deep_supervision(criterion, xs, y):
    """DeepSupervision

    Applies criterion to each element in a list.

    Args:
        criterion: loss function
        xs: tuple of inputs
        y: ground truth
    """
    loss = 0.
    for x in xs:
        loss += criterion(x, y)
    loss /= len(xs)
    return loss
