from __future__ import division, print_function, absolute_import
import torch
from torch.nn import functional as F

from torchreid.utils.writer import Writer
from torchreid.utils.tensortools import replace_values, masked_mean


def compute_distance_matrix(input1, input2, metric='euclidean'):
    """A wrapper function for computing distance matrix.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.
        metric (str, optional): "euclidean" or "cosine".
            Default is "euclidean".

    Returns:
        torch.Tensor: distance matrix.

    Examples::
       >>> from torchreid import metrics
       >>> input1 = torch.rand(10, 2048)
       >>> input2 = torch.rand(100, 2048)
       >>> distmat = metrics.compute_distance_matrix(input1, input2)
       >>> distmat.size() # (10, 100)
    """
    # check input
    assert isinstance(input1, torch.Tensor)
    assert isinstance(input2, torch.Tensor)
    assert input1.dim() == 2, 'Expected 2-D tensor, but got {}-D'.format(
        input1.dim()
    )
    assert input2.dim() == 2, 'Expected 2-D tensor, but got {}-D'.format(
        input2.dim()
    )
    assert input1.size(1) == input2.size(1)

    if metric == 'euclidean':
        distmat = euclidean_squared_distance(input1, input2)
    elif metric == 'cosine':
        distmat = cosine_distance(input1, input2)
    else:
        raise ValueError(
            'Unknown distance metric: {}. '
            'Please choose either "euclidean" or "cosine"'.format(metric)
        )

    return distmat


def euclidean_squared_distance(input1, input2):
    """Computes euclidean squared distance.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.

    Returns:
        torch.Tensor: distance matrix.
    """
    # dist(a, b) = sum((a_i - b_i)^2) = sum(a_i^2) + sum(b_i^2) - 2*sum(a_i*b_i)
    m, n = input1.size(0), input2.size(0)
    mat1 = torch.pow(input1, 2).sum(dim=1, keepdim=True).expand(m, n) # sum(a_i^2)
    mat2 = torch.pow(input2, 2).sum(dim=1, keepdim=True).expand(n, m).t() # sum(b_i^2)
    distmat = mat1 + mat2 # sum(a_i^2) + sum(b_i^2)
    distmat.addmm_(input1, input2.t(), beta=1, alpha=-2) # sum(a_i^2) + sum(b_i^2) - 2*sum(a_i*b_i)
    return distmat


def cosine_distance(input1, input2):
    """Computes cosine distance.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.

    Returns:
        torch.Tensor: distance matrix.
    """
    input1_normed = F.normalize(input1, p=2, dim=1)
    input2_normed = F.normalize(input2, p=2, dim=1)
    distmat = 1 - torch.mm(input1_normed, input2_normed.t())
    return distmat


def compute_distance_matrix_using_bp_features(qf, gf, qf_parts_visibility=None, gf_parts_visibility=None, dist_combine_strat='mean', batch_size_pairwise_dist_matrix=5000, use_gpu=False, metric='euclidean'):
    """Computes distance matrix between each pair of samples using their part-based features. 3 implementations here: without visibility scores, with boolean/binary visibility scores and with continuous [0, 1] visibility scores."""
    # TODO keep only one generic implementation
    if qf_parts_visibility is not None and gf_parts_visibility is not None:
        if qf_parts_visibility.dtype is torch.bool and gf_parts_visibility.dtype is torch.bool:
            # boolean visibility scores
            return _compute_distance_matrix_using_bp_features_and_masks(qf, gf, qf_parts_visibility, gf_parts_visibility, dist_combine_strat, batch_size_pairwise_dist_matrix, use_gpu, metric)
        else:
            # continuous visibility scores
            return _compute_distance_matrix_using_bp_features_and_visibility_scores(qf, gf, qf_parts_visibility, gf_parts_visibility, dist_combine_strat, batch_size_pairwise_dist_matrix, use_gpu, metric)
    else:
        # no visibility scores
        return _compute_distance_matrix_using_bp_features(qf, gf, dist_combine_strat, batch_size_pairwise_dist_matrix, use_gpu, metric)


def _compute_distance_matrix_using_bp_features(qf, gf, dist_combine_strat, batch_size_pairwise_dist_matrix, use_gpu, metric):
    if use_gpu:
        qf = qf.cuda()

    pairwise_dist_, body_part_pairwise_dist_ = [], []
    for batch_gf in torch.split(gf, batch_size_pairwise_dist_matrix):
        if use_gpu:
            batch_gf = batch_gf.cuda()
        batch_body_part_pairwise_dist = _compute_body_parts_dist_matrices(qf, batch_gf, metric)
        if dist_combine_strat == 'max':
            batch_pairwise_dist, _ = batch_body_part_pairwise_dist.max(dim=0)
        elif dist_combine_strat == 'mean':
            batch_pairwise_dist = batch_body_part_pairwise_dist.mean(dim=0)
        else:
            raise ValueError('Body parts distance combination strategy "{}" not supported'.format(dist_combine_strat))
        batch_body_part_pairwise_dist = batch_body_part_pairwise_dist.cpu()
        body_part_pairwise_dist_.append(batch_body_part_pairwise_dist)

        batch_pairwise_dist = batch_pairwise_dist.cpu()
        pairwise_dist_.append(batch_pairwise_dist)

    pairwise_dist = torch.cat(pairwise_dist_, 1)
    body_part_pairwise_dist = torch.cat(body_part_pairwise_dist_, 2)

    if Writer.current_writer() is not None:
        Writer.current_writer().qg_pairwise_dist_statistics(pairwise_dist, body_part_pairwise_dist, None, None)

    return pairwise_dist, body_part_pairwise_dist

def _compute_distance_matrix_using_bp_features_and_masks(qf, gf, qf_parts_visibility, gf_parts_visibility, dist_combine_strat, batch_size_pairwise_dist_matrix, use_gpu, metric):
    batch_gf_list = torch.split(gf, batch_size_pairwise_dist_matrix)
    batch_gf_parts_visibility_list = torch.split(gf_parts_visibility, batch_size_pairwise_dist_matrix)

    qf_parts_visibility_cpu = qf_parts_visibility
    if use_gpu:
        qf = qf.cuda()
        qf_parts_visibility = qf_parts_visibility.cuda()

    qf_parts_visibility = qf_parts_visibility.t()
    pairwise_dist_, body_part_pairwise_dist_ = [], []
    for batch_gf, batch_gf_parts_visibility in zip(batch_gf_list, batch_gf_parts_visibility_list):
        if use_gpu:
            batch_gf = batch_gf.cuda()
            batch_gf_parts_visibility = batch_gf_parts_visibility.cuda()

        batch_body_part_pairwise_dist = _compute_body_parts_dist_matrices(qf, batch_gf, metric)
        assert qf_parts_visibility.dtype is torch.bool and batch_gf_parts_visibility.dtype is torch.bool
        batch_gf_parts_visibility = batch_gf_parts_visibility.t()
        valid_body_part_pairwise_dist_mask = qf_parts_visibility.unsqueeze(2) * batch_gf_parts_visibility.unsqueeze(1)

        if dist_combine_strat == 'max':
            valid_body_part_pairwise_dist = replace_values(batch_body_part_pairwise_dist,
                                                           ~valid_body_part_pairwise_dist_mask, -1)
            batch_pairwise_dist, _ = valid_body_part_pairwise_dist.max(dim=0)
        elif dist_combine_strat == 'mean':
            batch_pairwise_dist = masked_mean(batch_body_part_pairwise_dist, valid_body_part_pairwise_dist_mask)
            valid_body_part_pairwise_dist = replace_values(batch_body_part_pairwise_dist,
                                                           ~valid_body_part_pairwise_dist_mask, -1)
        else:
            raise ValueError('Body parts distance combination strategy "{}" not supported'.format(dist_combine_strat))

        valid_body_part_pairwise_dist = valid_body_part_pairwise_dist.cpu()
        body_part_pairwise_dist_.append(valid_body_part_pairwise_dist)
        batch_pairwise_dist = batch_pairwise_dist.cpu()
        pairwise_dist_.append(batch_pairwise_dist)
    pairwise_dist = torch.cat(pairwise_dist_, 1)
    body_part_pairwise_dist = torch.cat(body_part_pairwise_dist_, 2)

    if Writer.current_writer() is not None:
        Writer.current_writer().qg_pairwise_dist_statistics(pairwise_dist, body_part_pairwise_dist, qf_parts_visibility_cpu, gf_parts_visibility)

    max_value = body_part_pairwise_dist.max() + 1 # FIXME not clean with cosine dist
    valid_pairwise_dist_mask = (pairwise_dist != float(-1))
    pairwise_dist = replace_values(pairwise_dist, ~valid_pairwise_dist_mask, max_value)
    body_part_pairwise_dist = replace_values(body_part_pairwise_dist, (body_part_pairwise_dist == -1), max_value)

    return pairwise_dist, body_part_pairwise_dist


def _compute_distance_matrix_using_bp_features_and_visibility_scores(qf, gf, qf_parts_visibility, gf_parts_visibility, dist_combine_strat, batch_size_pairwise_dist_matrix, use_gpu, metric):
    batch_gf_list = torch.split(gf, batch_size_pairwise_dist_matrix)
    batch_gf_parts_visibility_list = torch.split(gf_parts_visibility, batch_size_pairwise_dist_matrix)

    qf_parts_visibility_cpu = qf_parts_visibility
    if use_gpu:
        qf = qf.cuda()
        qf_parts_visibility = qf_parts_visibility.cuda()

    qf_parts_visibility = qf_parts_visibility.t()
    pairwise_dist_, body_part_pairwise_dist_ = [], []
    for batch_gf, batch_gf_parts_visibility in zip(batch_gf_list, batch_gf_parts_visibility_list):
        if use_gpu:
            batch_gf = batch_gf.cuda()
            batch_gf_parts_visibility = batch_gf_parts_visibility.cuda()

        batch_body_part_pairwise_dist = _compute_body_parts_dist_matrices(qf, batch_gf, metric)
        batch_gf_parts_visibility = batch_gf_parts_visibility.t()
        valid_body_part_pairwise_dist_mask = torch.sqrt(qf_parts_visibility.unsqueeze(2) * batch_gf_parts_visibility.unsqueeze(1))

        batch_pairwise_dist = masked_mean(batch_body_part_pairwise_dist, valid_body_part_pairwise_dist_mask)
        valid_body_part_pairwise_dist = batch_body_part_pairwise_dist

        valid_body_part_pairwise_dist = valid_body_part_pairwise_dist.cpu()
        body_part_pairwise_dist_.append(valid_body_part_pairwise_dist)
        batch_pairwise_dist = batch_pairwise_dist.cpu()
        pairwise_dist_.append(batch_pairwise_dist)
    pairwise_dist = torch.cat(pairwise_dist_, 1)
    body_part_pairwise_dist = torch.cat(body_part_pairwise_dist_, 2)

    # TODO check if still valid:
    if Writer.current_writer() is not None:
        Writer.current_writer().qg_pairwise_dist_statistics(pairwise_dist, body_part_pairwise_dist, qf_parts_visibility_cpu, gf_parts_visibility)

    max_value = body_part_pairwise_dist.max() + 1
    valid_pairwise_dist_mask = (pairwise_dist != float(-1))
    pairwise_dist = replace_values(pairwise_dist, ~valid_pairwise_dist_mask, max_value)

    return pairwise_dist, body_part_pairwise_dist


def _compute_body_parts_dist_matrices(qf, gf, metric='euclidean'):
    """
    gf, qf shapes = (N, M, C)
    ||a-b||^2 = |a|^2 - 2*<a,b> + |b|^2
    """
    if metric == 'euclidean':
        qf = qf.transpose(1, 0)
        gf = gf.transpose(1, 0)
        dot_product = torch.matmul(qf, gf.transpose(2, 1))
        qf_square_sum = qf.pow(2).sum(dim=-1)
        gf_square_sum = gf.pow(2).sum(dim=-1)

        distances = qf_square_sum.unsqueeze(2) - 2 * dot_product + gf_square_sum.unsqueeze(1)
        distances = F.relu(distances)
        distances = torch.sqrt(distances)
    elif metric == 'cosine':
        qf = qf.transpose(1, 0)
        gf = gf.transpose(1, 0)
        distances = 1 - torch.matmul(qf, gf.transpose(2, 1))
    else:
        raise ValueError(
            'Unknown distance metric: {}. '
            'Please choose either "euclidean" or "cosine"'.format(metric)
        )

    return distances
