from __future__ import division, print_function, absolute_import

import cv2
import torch
import numpy as np
from albumentations import (
    Resize, Compose, Normalize, ColorJitter, HorizontalFlip, CoarseDropout, RandomCrop, PadIfNeeded
)
from albumentations.pytorch import ToTensorV2
from torchreid.data.masks_transforms import masks_preprocess_all, AddBackgroundMask, ResizeMasks, PermuteMasksDim, \
    RemoveBackgroundMask
from torchreid.data.data_augmentation import RandomOcclusion


class NpToTensor(object):
    def __call__(self, masks):
        assert isinstance(masks, np.ndarray)
        return torch.as_tensor(masks)

    def __repr__(self):
        return self.__class__.__name__ + '()'


def build_transforms(
    height,
    width,
    config,
    mask_scale=4,
    transforms='random_flip',
    norm_mean=[0.485, 0.456, 0.406],
    norm_std=[0.229, 0.224, 0.225],
    remove_background_mask=False,
    masks_preprocess = 'none',
    softmax_weight = 0,
    mask_filtering_threshold = 0.3,
    background_computation_strategy = 'threshold',
    **kwargs
):
    """Builds train and test transform functions.

    Args:
        height (int): target image height.
        width (int): target image width.
        transforms (str or list of str, optional): transformations applied to model training.
            Default is 'random_flip'.
        norm_mean (list or None, optional): normalization mean values. Default is ImageNet means.
        norm_std (list or None, optional): normalization standard deviation values. Default is
            ImageNet standard deviation values.
    """
    if transforms is None:
        transforms = []

    if isinstance(transforms, str):
        transforms = [transforms]

    if not isinstance(transforms, list):
        raise ValueError(
            'transforms must be a list of strings, but found to be {}'.format(
                type(transforms)
            )
        )

    if len(transforms) > 0:
        transforms = [t.lower() for t in transforms]

    if norm_mean is None or norm_std is None:
        norm_mean = [0.485, 0.456, 0.406] # imagenet mean
        norm_std = [0.229, 0.224, 0.225] # imagenet std
    normalize = Normalize(mean=norm_mean, std=norm_std)

    print('Building train transforms ...')
    transform_tr = []

    print('+ resize to {}x{}'.format(height, width))
    transform_tr += [Resize(height, width)]

    if 'random_occlusion' in transforms or 'ro' in transforms:
        print('+ random occlusion')
        transform_tr += [RandomOcclusion(path=config.data.ro.path,
                                         im_shape=[config.data.height, config.data.width],
                                         p=config.data.ro.p,
                                         n=config.data.ro.n,
                                         min_overlap=config.data.ro.min_overlap,
                                         max_overlap=config.data.ro.max_overlap,
                                         )]

    if 'random_flip' in transforms or 'rf' in transforms:
        print('+ random flip')
        transform_tr += [HorizontalFlip()]

    if 'random_crop' in transforms or 'rc' in transforms:
        print('+ random crop')
        pad_size = 10
        transform_tr += [PadIfNeeded(min_height=height+pad_size*2, min_width=width+pad_size*2, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1),
                         RandomCrop(height, width, p=1)]

    if 'color_jitter' in transforms or 'cj' in transforms:
        print('+ color jitter')
        transform_tr += [
            ColorJitter(brightness=config.data.cj.brightness,
                        contrast=config.data.cj.contrast,
                        saturation=config.data.cj.saturation,
                        hue=config.data.cj.hue,
                        always_apply=config.data.cj.always_apply,
                        p=config.data.cj.p,
                        )
        ]

    print('+ normalization (mean={}, std={})'.format(norm_mean, norm_std))
    transform_tr += [normalize]

    if 'random_erase' in transforms or 're' in transforms:
        print('+ random erase')
        transform_tr += [CoarseDropout(min_holes=1, max_holes=1,
                                       min_height=int(height*0.15), max_height=int(height*0.65),
                                       min_width=int(width*0.15), max_width=int(width*0.65),
                                       fill_value=[0.485, 0.456, 0.406], mask_fill_value=0, always_apply=False, p=0.5)]

    print('+ to torch tensor of range [0, 1]')
    transform_tr += [ToTensorV2()]

    print('Building test transforms ...')
    print('+ resize to {}x{}'.format(height, width))
    print('+ to torch tensor of range [0, 1]')
    print('+ normalization (mean={}, std={})'.format(norm_mean, norm_std))

    transform_te = [
        Resize(height, width),
        normalize,
        ToTensorV2()
    ]

    transform_tr += [PermuteMasksDim()]
    transform_te += [PermuteMasksDim()]

    if remove_background_mask:  # ISP masks
        print('+ use remove background mask')
        # remove background before performing other transforms
        transform_tr = [RemoveBackgroundMask()] + transform_tr
        transform_te = [RemoveBackgroundMask()] + transform_te

        # Derive background mask from all foreground masks once other tasks have been performed
        print('+ use add background mask')
        transform_tr += [AddBackgroundMask('sum')]
        transform_te += [AddBackgroundMask('sum')]
    else:  # Pifpaf confidence based masks
        if masks_preprocess != 'none':
            print('+ masks preprocess = {}'.format(masks_preprocess))
            masks_preprocess_transform = masks_preprocess_all[masks_preprocess]
            transform_tr += [masks_preprocess_transform()]
            transform_te += [masks_preprocess_transform()]

        print('+ use add background mask')
        transform_tr += [AddBackgroundMask(background_computation_strategy, softmax_weight, mask_filtering_threshold)]
        transform_te += [AddBackgroundMask(background_computation_strategy, softmax_weight, mask_filtering_threshold)]

    transform_tr += [ResizeMasks(height, width, mask_scale)]
    transform_te += [ResizeMasks(height, width, mask_scale)]

    transform_tr = Compose(transform_tr, is_check_shapes=False)
    transform_te = Compose(transform_te, is_check_shapes=False)

    return transform_tr, transform_te
