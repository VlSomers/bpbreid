from __future__ import absolute_import
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from torchreid.utils import (
    check_isfile, load_pretrained_weights, compute_model_complexity
)
from torchreid.models import build_model
from torchreid.data.transforms import build_transforms


class FeatureExtractor(object):
    """A simple API for feature extraction.

    FeatureExtractor can be used like a python function, which
    accepts input of the following types:
        - a list of strings (image paths)
        - a list of numpy.ndarray each with shape (H, W, C)
        - a single string (image path)
        - a single numpy.ndarray with shape (H, W, C)
        - a torch.Tensor with shape (B, C, H, W) or (C, H, W)

    Returned is a torch tensor with shape (B, D) where D is the
    feature dimension.

    Args:
        model_name (str): model name.
        model_path (str): path to model weights.
        image_size (sequence or int): image height and width.
        pixel_mean (list): pixel mean for normalization.
        pixel_std (list): pixel std for normalization.
        pixel_norm (bool): whether to normalize pixels.
        device (str): 'cpu' or 'cuda' (could be specific gpu devices).
        verbose (bool): show model details.

    Examples::

        from torchreid.utils import FeatureExtractor

        extractor = FeatureExtractor(
            model_name='osnet_x1_0',
            model_path='a/b/c/model.pth.tar',
            device='cuda'
        )

        image_list = [
            'a/b/c/image001.jpg',
            'a/b/c/image002.jpg',
            'a/b/c/image003.jpg',
            'a/b/c/image004.jpg',
            'a/b/c/image005.jpg'
        ]

        features = extractor(image_list)
        print(features.shape) # output (5, 512)
    """

    def __init__(
        self,
        cfg,
        model_path='',
        image_size=(256, 128),
        pixel_mean=[0.485, 0.456, 0.406],
        pixel_std=[0.229, 0.224, 0.225],
        pixel_norm=True,
        device='cuda',
        num_classes=1,
        verbose=True,
        model=None
    ):
        # Build model
        if model is None:
            print("building model on device {}".format(device))
            model = build_model(
                name=cfg.model.name,
                loss=cfg.loss.name,
                pretrained=cfg.model.pretrained,
                num_classes=num_classes,
                use_gpu=device.startswith('cuda'),
                pooling=cfg.model.bpbreid.pooling,
                normalization=cfg.model.bpbreid.normalization,
                last_stride=cfg.model.bpbreid.last_stride,
                config=cfg
            )

        model.eval()

        if verbose:
            num_params, flops = compute_model_complexity(
                model, cfg
            )
            print('Model: {}'.format(cfg.model.name))
            print('- params: {:,}'.format(num_params))
            print('- flops: {:,}'.format(flops))

        if model_path and check_isfile(model_path):
            load_pretrained_weights(model, model_path)

        # Build transform functions
        _, preprocess = build_transforms(image_size[0],
                                         image_size[1],
                                         cfg,
                                         transforms=None,
                                         norm_mean=pixel_mean,
                                         norm_std=pixel_std,
                                         masks_preprocess=cfg.model.bpbreid.masks.preprocess,
                                         softmax_weight=cfg.model.bpbreid.masks.softmax_weight,
                                         background_computation_strategy=cfg.model.bpbreid.masks.background_computation_strategy,
                                         mask_filtering_threshold=cfg.model.bpbreid.masks.mask_filtering_threshold,
                                         )

        to_pil = T.ToPILImage()

        device = torch.device(device)
        model.to(device)

        # Class attributes
        self.model = model
        self.preprocess = preprocess
        self.to_pil = to_pil
        self.device = device

    def __call__(self, input, external_parts_masks=None):
        if isinstance(input, list):
            images = []
            masks = []

            for i, element in enumerate(input):
                if isinstance(element, str):
                    image = Image.open(element).convert('RGB')

                elif isinstance(element, np.ndarray):
                    image = self.to_pil(element)

                else:
                    raise TypeError(
                        'Type of each element must belong to [str | numpy.ndarray]'
                    )

                transf_args = {}
                if external_parts_masks is not None:
                    transf_args['mask'] = external_parts_masks[i].transpose(1, 2, 0)
                transf_args['image'] = np.array(image)
                result = self.preprocess(**transf_args)
                images.append(result['image'])
                if external_parts_masks is not None:
                    masks.append(result['mask'])

            images = torch.stack(images, dim=0)
            images = images.to(self.device)
            if external_parts_masks is not None:
                masks = torch.stack(masks, dim=0)
                masks = masks.to(self.device)

        elif isinstance(input, str):
            image = Image.open(input).convert('RGB')
            transf_args = {}
            if external_parts_masks is not None:
                transf_args['mask'] = external_parts_masks.transpose(1, 2, 0)
            transf_args['image'] = np.array(image)
            result = self.preprocess(**transf_args)
            images = result['image'].unsqueeze(0).to(self.device)
            masks = result['mask'].unsqueeze(0).to(self.device)

        elif isinstance(input, np.ndarray):
            image = input
            transf_args = {}
            if external_parts_masks is not None:
                transf_args['mask'] = external_parts_masks.transpose(1, 2, 0)
            transf_args['image'] = np.array(image)
            result = self.preprocess(**transf_args)
            images = result['image'].unsqueeze(0).to(self.device)
            masks = result['mask'].unsqueeze(0).to(self.device)

        elif isinstance(input, torch.Tensor):
            if input.dim() == 3:
                input = input.unsqueeze(0)
                external_parts_masks = external_parts_masks.unsqueeze(0)
            images = input.to(self.device)

            masks = external_parts_masks.to(self.device)

        else:
            raise NotImplementedError

        with torch.no_grad():
            features = self.model(images, external_parts_masks=masks)

        return features
