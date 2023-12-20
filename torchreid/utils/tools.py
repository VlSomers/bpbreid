from __future__ import division, print_function, absolute_import
import os
import sys
import json
import time
import errno
import cv2
import numpy as np
import random
import os.path as osp
import warnings
import PIL
import torch
from PIL import Image
from torchreid.utils.constants import bn_correspondants

__all__ = [
    'mkdir_if_missing', 'check_isfile', 'read_json', 'write_json',
    'set_random_seed', 'download_url', 'read_image', 'read_masks', 'collect_env_info', 'perc'
]


def mkdir_if_missing(dirname):
    """Creates dirname if it is missing."""
    if not osp.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def check_isfile(fpath):
    """Checks if the given path is a file.

    Args:
        fpath (str): file path.

    Returns:
       bool
    """
    isfile = osp.isfile(fpath)
    if not isfile:
        warnings.warn('No file found at "{}"'.format(fpath))
    return isfile


def read_json(fpath):
    """Reads json file from a path."""
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    """Writes to a json file."""
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def download_url(url, dst):
    """Downloads file from a url to a destination.

    Args:
        url (str): url to download file.
        dst (str): destination path.
    """
    from six.moves import urllib
    print('* url="{}"'.format(url))
    print('* destination="{}"'.format(dst))

    def _reporthook(count, block_size, total_size):
        global start_time
        if count == 0:
            start_time = time.time()
            return
        duration = time.time() - start_time
        progress_size = int(count * block_size)
        speed = int(progress_size / (1024*duration))
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(
            '\r...%d%%, %d MB, %d KB/s, %d seconds passed' %
            (percent, progress_size / (1024*1024), speed, duration)
        )
        sys.stdout.flush()

    urllib.request.urlretrieve(url, dst, _reporthook)
    sys.stdout.write('\n')


def read_image(path):
    """Reads image from path using ``PIL.Image``.

    Args:
        path (str): path to an image.

    Returns:
        PIL image
    """
    got_img = False
    if not osp.exists(path):
        raise IOError('"{}" does not exist'.format(path))
    while not got_img:
        try:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            got_img = True
        except IOError:
            print(
                'IOError incurred when reading "{}". Will redo. Don\'t worry. Just chill.'
                .format(path)
            )
    return img


def read_masks(masks_path):
    """Reads part-based masks information from path.

    Args:
        path (str): path to an image. Part-based masks information is stored in a .npy file with image name as prefix

    Returns:
        Numpy array of size N x H x W where N is the number of part-based masks
    """

    got_masks = False
    if not osp.exists(masks_path):
        raise IOError('Masks file"{}" does not exist'.format(masks_path))
    while not got_masks:
        try:
            masks = np.load(masks_path)
            masks = np.transpose(masks, (1, 2, 0))
            got_masks = True
        except IOError:
            print(
                'IOError incurred when reading "{}". Will redo. Don\'t worry. Just chill.'
                .format(masks_path)
            )
    return masks


def collect_env_info():
    """Returns env info as a string.

    Code source: github.com/facebookresearch/maskrcnn-benchmark
    """
    from torch.utils.collect_env import get_pretty_env_info
    env_str = get_pretty_env_info()
    env_str += '\n        Pillow ({})'.format(PIL.__version__)
    return env_str


def perc(val, decimals=2):
    return np.around(val*100, decimals)

def extract_test_embeddings(model_output, test_embeddings):
    embeddings, visibility_scores, id_cls_scores, pixels_cls_scores, spatial_features, parts_masks = model_output
    embeddings_list = []
    visibility_scores_list = []
    embeddings_masks_list = []

    for test_emb in test_embeddings:
        embds = embeddings[test_emb]
        embeddings_list.append(embds if len(embds.shape) == 3 else embds.unsqueeze(1))
        if test_emb in bn_correspondants:
            test_emb = bn_correspondants[test_emb]
        vis_scores = visibility_scores[test_emb]
        visibility_scores_list.append(vis_scores if len(vis_scores.shape) == 2 else vis_scores.unsqueeze(1))
        pt_masks = parts_masks[test_emb]
        embeddings_masks_list.append(pt_masks if len(pt_masks.shape) == 4 else pt_masks.unsqueeze(1))

    assert len(embeddings) != 0

    embeddings = torch.cat(embeddings_list, dim=1)  # [N, P+2, D]
    visibility_scores = torch.cat(visibility_scores_list, dim=1)  # [N, P+2]
    embeddings_masks = torch.cat(embeddings_masks_list, dim=1)  # [N, P+2, Hf, Wf]

    return embeddings, visibility_scores, embeddings_masks, pixels_cls_scores

