import numpy as np
import torch

from torchreid.data.masks_transforms.mask_transform import MaskTransform


class PCBMasks(MaskTransform):
    def apply_to_mask(self, masks, **params):
        self._size = masks.shape[1:3]
        self.stripe_height = self._size[0] / self.parts_num

        self.pcb_masks = torch.zeros((self.parts_num, self._size[0], self._size[1]))

        stripes_range = np.round(np.arange(0, self.parts_num + 1) * self._size[0] / self.parts_num).astype(int)
        for i in range(0, stripes_range.size-1):
            self.pcb_masks[i, stripes_range[i]:stripes_range[i+1], :] = 1

        return self.pcb_masks


class PCBMasks2(PCBMasks):
    parts_num = 2
    parts_names = ["p{}".format(p) for p in range(1, parts_num+1)]


class PCBMasks3(PCBMasks):
    parts_num = 3
    parts_names = ["p{}".format(p) for p in range(1, parts_num+1)]


class PCBMasks4(PCBMasks):
    parts_num = 4
    parts_names = ["p{}".format(p) for p in range(1, parts_num+1)]


class PCBMasks5(PCBMasks):
    parts_num = 5
    parts_names = ["p{}".format(p) for p in range(1, parts_num+1)]


class PCBMasks6(PCBMasks):
    parts_num = 6
    parts_names = ["p{}".format(p) for p in range(1, parts_num+1)]


class PCBMasks7(PCBMasks):
    parts_num = 7
    parts_names = ["p{}".format(p) for p in range(1, parts_num+1)]


class PCBMasks8(PCBMasks):
    parts_num = 8
    parts_names = ["p{}".format(p) for p in range(1, parts_num+1)]
