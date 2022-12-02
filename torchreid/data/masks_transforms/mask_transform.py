import torch
from torch import nn
from albumentations import DualTransform
import torch.nn.functional as F


class MaskTransform(DualTransform):
    def __init__(self):
        super(MaskTransform, self).__init__(always_apply=True, p=1)

    def apply(self, img, **params):
        return img

    def apply_to_bbox(self, bbox, **params):
        raise NotImplementedError("Method apply_to_bbox is not implemented in class " + self.__class__.__name__)

    def apply_to_keypoint(self, keypoint, **params):
        raise NotImplementedError("Method apply_to_keypoint is not implemented in class " + self.__class__.__name__)


class MaskGroupingTransform(MaskTransform):

    def __init__(self, parts_grouping, parts_map, combine_mode='max'):
        super().__init__()
        self.parts_grouping = parts_grouping
        self.parts_map = parts_map
        self.parts_names = list(parts_grouping.keys())
        self.parts_num = len(self.parts_names)
        self.combine_mode = combine_mode

    def apply_to_mask(self, masks, **params):
        parts_masks = []
        for i, part in enumerate(self.parts_names):
            if self.combine_mode == 'sum':
                parts_masks.append(masks[[self.parts_map[k] for k in self.parts_grouping[part]]].sum(dim=0).clamp(0, 1))
            else:
                parts_masks.append(masks[[self.parts_map[k] for k in self.parts_grouping[part]]].max(dim=0)[0].clamp(0, 1))
        return torch.stack(parts_masks)


class PermuteMasksDim(MaskTransform):
    def apply_to_mask(self, masks, **params):
        return masks.permute(2, 0, 1)


class ResizeMasks(MaskTransform):
    def __init__(self, height, width, mask_scale):
        super(ResizeMasks, self).__init__()
        self._size = (int(height/mask_scale), int(width/mask_scale))

    def apply_to_mask(self, masks, **params):
        return nn.functional.interpolate(masks.unsqueeze(0), self._size, mode='nearest').squeeze(0)  # Best perf with nearest here and bilinear in parts engine


class RemoveBackgroundMask(MaskTransform):
    def apply_to_mask(self, masks, **params):
        return masks[:, :, 1::]


class AddBackgroundMask(MaskTransform):
    def __init__(self, background_computation_strategy='sum', softmax_weight=0, mask_filtering_threshold=0.3):
        super().__init__()
        self.background_computation_strategy = background_computation_strategy
        self.softmax_weight = softmax_weight
        self.mask_filtering_threshold = mask_filtering_threshold

    def apply_to_mask(self, masks, **params):
        if self.background_computation_strategy == 'sum':
            background_mask = 1 - masks.sum(dim=0)
            background_mask = background_mask.clamp(0, 1)
            masks = torch.cat([background_mask.unsqueeze(0), masks])
        elif self.background_computation_strategy == 'threshold':
            background_mask = masks.max(dim=0)[0] < self.mask_filtering_threshold
            masks = torch.cat([background_mask.unsqueeze(0), masks])
        elif self.background_computation_strategy == 'diff_from_max':
            background_mask = 1 - masks.max(dim=0)[0]
            background_mask = background_mask.clamp(0, 1)
            masks = torch.cat([background_mask.unsqueeze(0), masks])
        else:
            raise ValueError('Background mask combine strategy {} not supported'.format(self.background_computation_strategy))
        if self.softmax_weight > 0:
            masks = F.softmax(masks * self.softmax_weight, dim=0)
        else:
            masks = masks / masks.sum(dim=0)
        return masks


class IdentityMask(MaskTransform):
    parts_names = ['id']
    parts_num = 1
    def apply_to_mask(self, masks, **params):
        return torch.ones((1, masks.shape[1], masks.shape[2]))
