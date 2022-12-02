

def replace_values(input, mask, value):
    # TODO test perfs
    output = input * (~mask) + mask * value
    # input[mask] = value
    # output = input
    # output = torch.where(mask, input, torch.tensor(value, dtype=input.dtype, device=(input.get_device() if input.is_cuda else None)))
    return output


def masked_mean(input, mask):
    # TODO CHECK ON RANKING GRID IF IT WORK WITH CONTINUOUS VISIBILITY
    """ output -1 where mean couldn't be computed """
    valid_input = input * mask
    mean_weights = mask.sum(0)
    mean_weights = mean_weights + (mean_weights == 0)  # to avoid division by 0
    pairwise_dist = valid_input.sum(0) / mean_weights
    invalid_pairs = (mask.sum(dim=0) == 0)
    valid_pairwise_dist = replace_values(pairwise_dist, invalid_pairs, -1)
    return valid_pairwise_dist
