from __future__ import print_function, absolute_import

from .mask_transform import *
from .pcb_transforms import *
from .pifpaf_mask_transform import *
from .coco_keypoints_transforms import *
from torchreid.data.datasets import get_image_dataset

masks_preprocess_pifpaf = {
    'full': CombinePifPafIntoFullBodyMask,
    'bs_fu': AddFullBodyMaskToBaseMasks,
    'bs_fu_bb': AddFullBodyMaskAndFullBoundingBoxToBaseMasks,
    'mu_sc': CombinePifPafIntoMultiScaleBodyMasks,
    'one': CombinePifPafIntoOneBodyMasks,
    'two_v': CombinePifPafIntoTwoBodyMasks,
    'three_v': CombinePifPafIntoThreeBodyMasks,
    'four': CombinePifPafIntoFourBodyMasks,
    'four_no': CombinePifPafIntoFourBodyMasksNoOverlap,
    'four_v': CombinePifPafIntoFourVerticalParts,
    'four_v_pif': CombinePifPafIntoFourVerticalPartsPif,
    'five_v': CombinePifPafIntoFiveVerticalParts,
    'five': CombinePifPafIntoFiveBodyMasks,
    'six': CombinePifPafIntoSixBodyMasks,
    'six_v': CombinePifPafIntoSixVerticalParts,
    'six_no': CombinePifPafIntoSixBodyMasksSum,
    'six_new': CombinePifPafIntoSixBodyMasksSimilarToEight,
    'seven_v': CombinePifPafIntoSevenVerticalBodyMasks,
    'seven_new': CombinePifPafIntoSevenBodyMasksSimilarToEight,
    'eight': CombinePifPafIntoEightBodyMasks,
    'eight_v': CombinePifPafIntoEightVerticalBodyMasks,
    'ten_ms': CombinePifPafIntoTenMSBodyMasks,
    'eleven': CombinePifPafIntoElevenBodyMasks,
    'fourteen': CombinePifPafIntoFourteenBodyMasks,
}

masks_preprocess_coco = {
    'cc6': CocoToSixBodyMasks
}

masks_preprocess_fixed = {
    'id': IdentityMask,
    'strp_2': PCBMasks2,
    'strp_3': PCBMasks3,
    'strp_4': PCBMasks4,
    'strp_5': PCBMasks5,
    'strp_6': PCBMasks6,
    'strp_7': PCBMasks7,
    'strp_8': PCBMasks8,
}

masks_preprocess_transforms = {**masks_preprocess_pifpaf, **masks_preprocess_coco}
masks_preprocess_all = {**masks_preprocess_pifpaf, **masks_preprocess_fixed, **masks_preprocess_coco}


def compute_parts_num_and_names(cfg):
    mask_config = get_image_dataset(cfg.data.sources[0]).get_masks_config(cfg.model.bpbreid.masks.dir)
    if cfg.loss.name == 'part_based':
        if (mask_config is not None and mask_config[1]) or cfg.model.bpbreid.masks.preprocess == 'none':
            # ISP masks or no transform
            cfg.model.bpbreid.masks.parts_num = mask_config[0]
            cfg.model.bpbreid.masks.parts_names = mask_config[3] if 3 in mask_config else ["p{}".format(p) for p in range(1, cfg.data.parts_num+1)]
        else:
            masks_transform = masks_preprocess_all[cfg.model.bpbreid.masks.preprocess]()
            cfg.model.bpbreid.masks.parts_num = masks_transform.parts_num
            cfg.model.bpbreid.masks.parts_names = masks_transform.parts_names