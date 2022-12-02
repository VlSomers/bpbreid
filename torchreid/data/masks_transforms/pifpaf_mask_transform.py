from __future__ import division, print_function, absolute_import

import torch
from torchreid.data.masks_transforms.mask_transform import MaskGroupingTransform

PIFPAF_KEYPOINTS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
                    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
                    "right_knee", "left_ankle", "right_ankle"]

PIFPAF_JOINTS = ["left_ankle_to_left_knee", "left_knee_to_left_hip", "right_ankle_to_right_knee",
                 "right_knee_to_right_hip", "left_hip_to_right_hip", "left_shoulder_to_left_hip",
                 "right_shoulder_to_right_hip", "left_shoulder_to_right_shoulder", "left_shoulder_to_left_elbow",
                 "right_shoulder_to_right_elbow", "left_elbow_to_left_wrist", "right_elbow_to_right_wrist",
                 "left_eye_to_right_eye", "nose_to_left_eye", "nose_to_right_eye", "left_eye_to_left_ear",
                 "right_eye_to_right_ear", "left_ear_to_left_shoulder", "right_ear_to_right_shoulder"]

PIFPAF_PARTS = PIFPAF_KEYPOINTS + PIFPAF_JOINTS
PIFPAF_SINGLE_GROUPS = {k:k for k in PIFPAF_PARTS}
PIFPAF_PARTS_MAP = {k: i for i, k in enumerate(PIFPAF_PARTS)}


class CombinePifPafIntoFullBodyMask(MaskGroupingTransform):
    parts_grouping = {
        "full_body": PIFPAF_PARTS
    }

    def __init__(self):
        super().__init__(self.parts_grouping, PIFPAF_PARTS_MAP)


class AddFullBodyMaskToBaseMasks(MaskGroupingTransform):
    parts_grouping = {**PIFPAF_SINGLE_GROUPS,
        **{
            "full_body": PIFPAF_PARTS
        }
    }

    def __init__(self):
        super().__init__(self.parts_grouping, PIFPAF_PARTS_MAP)


class AddFullBodyMaskAndFullBoundingBoxToBaseMasks(MaskGroupingTransform):
    parts_num = 38
    parts_names = ["p{}".format(p) for p in range(1, parts_num+1)]

    def apply_to_mask(self, masks, **params):
        full_body_mask = torch.max(masks, 0, keepdim=True)[0]

        full_bounding_box = torch.ones(masks.shape[1:3]).unsqueeze(0)

        return torch.cat([masks,
                            full_body_mask,
                            full_bounding_box
                          ])


class CombinePifPafIntoMultiScaleBodyMasks(MaskGroupingTransform):
    parts_grouping = {**PIFPAF_SINGLE_GROUPS, **{
            "head_mask": ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_eye_to_right_eye",
                                             "nose_to_left_eye", "nose_to_right_eye", "left_eye_to_left_ear",
                                             "right_eye_to_right_ear", "left_ear_to_left_shoulder",
                                             "right_ear_to_right_shoulder"],
            "arms_mask": ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist",
                                             "right_wrist", "left_shoulder_to_left_elbow", "right_shoulder_to_right_elbow",
                                             "left_elbow_to_left_wrist", "right_elbow_to_right_wrist"],
            "torso_mask": ["left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_hip_to_right_hip",
                                              "left_shoulder_to_left_hip", "right_shoulder_to_right_hip",
                                              "left_shoulder_to_right_shoulder"],
            "legs_mask": ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
                                             "left_ankle_to_left_knee", "left_knee_to_left_hip", "right_ankle_to_right_knee",
                                             "right_knee_to_right_hip", "left_hip_to_right_hip"],
            "feet_mask": ["left_ankle", "right_ankle"],

            "upper_body": ["torso_mask", "arms_mask", "head_mask"],

            "lower_body": ["legs_mask", "feet_mask"],

            "full_body_mask": PIFPAF_PARTS
        }
    }

    def __init__(self):
        super().__init__(self.parts_grouping, PIFPAF_PARTS_MAP)


class CombinePifPafIntoOneBodyMasks(MaskGroupingTransform):
    parts_grouping = {
        "full": PIFPAF_PARTS
    }

    def __init__(self):
        super().__init__(self.parts_grouping, PIFPAF_PARTS_MAP)


class CombinePifPafIntoTwoBodyMasks(MaskGroupingTransform):
    parts_grouping = {
        "torso_arms_head": ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_eye_to_right_eye",
                                                "nose_to_left_eye", "nose_to_right_eye", "left_eye_to_left_ear",
                                                "right_eye_to_right_ear", "left_ear_to_left_shoulder",
                                                "right_ear_to_right_shoulder", "left_shoulder", "right_shoulder",
                                                "left_shoulder_to_left_hip", "right_shoulder_to_right_hip",
                                                "left_shoulder_to_right_shoulder",
                                                "left_elbow", "right_elbow", "left_wrist",
                                                "right_wrist", "left_shoulder_to_left_elbow", "right_shoulder_to_right_elbow",
                                                "left_elbow_to_left_wrist", "right_elbow_to_right_wrist"],
        "legs": ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
                                     "left_ankle_to_left_knee", "left_knee_to_left_hip", "right_ankle_to_right_knee",
                                     "right_knee_to_right_hip", "left_hip_to_right_hip"]
    }

    def __init__(self):
        super().__init__(self.parts_grouping, PIFPAF_PARTS_MAP)


class CombinePifPafIntoThreeBodyMasks(MaskGroupingTransform):
    parts_grouping = {
        "head_mask": ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_eye_to_right_eye",
                                         "nose_to_left_eye", "nose_to_right_eye", "left_eye_to_left_ear",
                                         "right_eye_to_right_ear", "left_ear_to_left_shoulder",
                                         "right_ear_to_right_shoulder"],
        "torso_arms_mask": ["left_shoulder", "right_shoulder",
                                           "left_shoulder_to_left_hip", "right_shoulder_to_right_hip",
                                           "left_shoulder_to_right_shoulder",
                                           "left_elbow", "right_elbow", "left_wrist",
                                           "right_wrist", "left_shoulder_to_left_elbow", "right_shoulder_to_right_elbow",
                                           "left_elbow_to_left_wrist", "right_elbow_to_right_wrist"],
        "legs_mask": ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
                                     "left_ankle_to_left_knee", "left_knee_to_left_hip", "right_ankle_to_right_knee",
                                     "right_knee_to_right_hip", "left_hip_to_right_hip"]
    }

    def __init__(self):
        super().__init__(self.parts_grouping, PIFPAF_PARTS_MAP)


class CombinePifPafIntoFourBodyMasks(MaskGroupingTransform):
    parts_grouping = {
        "head_mask": ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_eye_to_right_eye",
                                         "nose_to_left_eye", "nose_to_right_eye", "left_eye_to_left_ear",
                                         "right_eye_to_right_ear", "left_ear_to_left_shoulder",
                                         "right_ear_to_right_shoulder"],
        "arms_mask": ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist",
                                     "right_wrist", "left_shoulder_to_left_elbow", "right_shoulder_to_right_elbow",
                                     "left_elbow_to_left_wrist", "right_elbow_to_right_wrist"],
        "torso_mask": ["left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_hip_to_right_hip",
                                      "left_shoulder_to_left_hip", "right_shoulder_to_right_hip",
                                      "left_shoulder_to_right_shoulder"],
        "legs_mask": ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
                                     "left_ankle_to_left_knee", "left_knee_to_left_hip", "right_ankle_to_right_knee",
                                     "right_knee_to_right_hip", "left_hip_to_right_hip"]
    }

    def __init__(self):
        super().__init__(self.parts_grouping, PIFPAF_PARTS_MAP)


class CombinePifPafIntoFourBodyMasksNoOverlap(MaskGroupingTransform):
    parts_grouping = {
        "head_mask": ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_eye_to_right_eye",
                                         "nose_to_left_eye", "nose_to_right_eye", "left_eye_to_left_ear",
                                         "right_eye_to_right_ear", "left_ear_to_left_shoulder",
                                         "right_ear_to_right_shoulder"],
        "arms_mask": ["left_elbow", "right_elbow", "left_wrist",
                                         "right_wrist", "left_shoulder_to_left_elbow", "right_shoulder_to_right_elbow",
                                         "left_elbow_to_left_wrist", "right_elbow_to_right_wrist"],
        "torso_mask": ["left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_hip_to_right_hip",
                                          "left_shoulder_to_left_hip", "right_shoulder_to_right_hip",
                                          "left_shoulder_to_right_shoulder"],
        "legs_mask": ["left_knee", "right_knee", "left_ankle", "right_ankle",
                                         "left_ankle_to_left_knee", "left_knee_to_left_hip", "right_ankle_to_right_knee",
                                         "right_knee_to_right_hip", "left_hip_to_right_hip"]
    }

    def __init__(self):
        super().__init__(self.parts_grouping, PIFPAF_PARTS_MAP)


class CombinePifPafIntoFourVerticalParts(MaskGroupingTransform):
    parts_grouping = {
        "head_mask": ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_eye_to_right_eye",
                                         "nose_to_left_eye", "nose_to_right_eye", "left_eye_to_left_ear",
                                         "right_eye_to_right_ear", "left_ear_to_left_shoulder",
                                         "right_ear_to_right_shoulder"],
        "arms_torso_mask": ["left_elbow", "right_elbow", "left_wrist",
                                         "right_wrist", "left_shoulder_to_left_elbow", "right_shoulder_to_right_elbow",
                                         "left_elbow_to_left_wrist", "right_elbow_to_right_wrist", "left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_hip_to_right_hip",
                                          "left_shoulder_to_left_hip", "right_shoulder_to_right_hip",
                                          "left_shoulder_to_right_shoulder"],
        "legs_mask": ["left_hip", "right_hip", "left_knee", "right_knee",
                                         "left_ankle_to_left_knee", "left_knee_to_left_hip", "right_ankle_to_right_knee",
                                         "right_knee_to_right_hip"],
        "feet_mask": ["left_ankle", "right_ankle"],
    }

    def __init__(self):
        super().__init__(self.parts_grouping, PIFPAF_PARTS_MAP)


class CombinePifPafIntoFourVerticalPartsPif(MaskGroupingTransform):
    parts_grouping = {
        "head_mask": ["nose", "left_eye", "right_eye", "left_ear", "right_ear"],
        "arms_torso_mask": ["left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_shoulder", "right_shoulder", "left_hip", "right_hip"],
        "legs_mask": ["left_hip", "right_hip", "left_knee", "right_knee"],
        "feet_mask": ["left_ankle", "right_ankle"],
    }

    def __init__(self):
        super().__init__(self.parts_grouping, PIFPAF_PARTS_MAP)


class CombinePifPafIntoFiveVerticalParts(MaskGroupingTransform):
    parts_grouping = {
        "head_mask": ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_eye_to_right_eye",
                                         "nose_to_left_eye", "nose_to_right_eye", "left_eye_to_left_ear",
                                         "right_eye_to_right_ear", "left_ear_to_left_shoulder",
                                         "right_ear_to_right_shoulder"],
        "upper_arms_torso_mask": ["left_elbow", "right_elbow",
                                                     "left_shoulder_to_left_elbow", "right_shoulder_to_right_elbow",
                                                     "left_shoulder", "right_shoulder",
                                                     "left_shoulder_to_right_shoulder"],
        "lower_arms_torso_mask": ["left_wrist", "right_wrist",
                                                     "left_elbow_to_left_wrist", "right_elbow_to_right_wrist",
                                                     "left_hip", "right_hip",
                                                     "right_shoulder_to_right_hip"],
        "legs_mask": ["left_hip", "right_hip", "left_knee", "right_knee",
                                         "left_ankle_to_left_knee", "left_knee_to_left_hip", "right_ankle_to_right_knee",
                                         "right_knee_to_right_hip"],
        "feet_mask": ["left_ankle", "right_ankle"],
    }

    def __init__(self):
        super().__init__(self.parts_grouping, PIFPAF_PARTS_MAP)


class CombinePifPafIntoFiveBodyMasks(MaskGroupingTransform):
    parts_grouping = {
        "head_mask": ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_eye_to_right_eye",
                                         "nose_to_left_eye", "nose_to_right_eye", "left_eye_to_left_ear",
                                         "right_eye_to_right_ear", "left_ear_to_left_shoulder",
                                         "right_ear_to_right_shoulder"],

        "arms_mask": ["left_shoulder", "left_elbow", "left_wrist", "left_shoulder_to_left_elbow",
                                             "left_elbow_to_left_wrist", "right_shoulder", "right_elbow", "right_wrist", "right_shoulder_to_right_elbow",
                                              "right_elbow_to_right_wrist"],

        "torso_mask": ["left_hip", "right_hip", "left_hip_to_right_hip",
                                          "left_shoulder_to_left_hip", "right_shoulder_to_right_hip",
                                          "left_shoulder_to_right_shoulder"],

        "legs_mask": ["left_hip_to_right_hip", "left_hip", "right_hip", "left_knee", "right_knee",
                                         "left_ankle_to_left_knee", "left_knee_to_left_hip", "right_ankle_to_right_knee",
                                         "right_knee_to_right_hip"],

        "feet_mask": ["left_ankle", "right_ankle"],
    }

    def __init__(self):
        super().__init__(self.parts_grouping, PIFPAF_PARTS_MAP)


class CombinePifPafIntoSixVerticalParts(MaskGroupingTransform):
    parts_grouping = {
        "head_mask": ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_eye_to_right_eye",
                                         "nose_to_left_eye", "nose_to_right_eye", "left_eye_to_left_ear",
                                         "right_eye_to_right_ear", "left_ear_to_left_shoulder",
                                         "right_ear_to_right_shoulder"],
        "arms_mask": ["left_shoulder", "left_elbow", "left_wrist", "left_shoulder_to_left_elbow",
                                             "left_elbow_to_left_wrist", "right_shoulder", "right_elbow", "right_wrist", "right_shoulder_to_right_elbow",
                                              "right_elbow_to_right_wrist"],
        "upper_torso_mask": ["left_shoulder_to_left_hip", "right_shoulder_to_right_hip",
                                          "left_shoulder_to_right_shoulder"],
        "lower_torso_mask": ["left_hip", "right_hip", "left_hip_to_right_hip"],
        "legs_mask": ["left_hip", "right_hip", "left_knee", "right_knee",
                                         "left_ankle_to_left_knee", "left_knee_to_left_hip", "right_ankle_to_right_knee",
                                         "right_knee_to_right_hip"],
        "feet_mask": ["left_ankle", "right_ankle"],
    }

    def __init__(self):
        super().__init__(self.parts_grouping, PIFPAF_PARTS_MAP)


class CombinePifPafIntoSixBodyMasks(MaskGroupingTransform):
    parts_grouping = {
        "head_mask": ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_eye_to_right_eye",
                                         "nose_to_left_eye", "nose_to_right_eye", "left_eye_to_left_ear",
                                         "right_eye_to_right_ear", "left_ear_to_left_shoulder",
                                         "right_ear_to_right_shoulder"],
        "left_arm_mask": ["left_shoulder", "left_elbow", "left_wrist", "left_shoulder_to_left_elbow",
                                             "left_elbow_to_left_wrist"],
        "right_arm_mask": ["right_shoulder", "right_elbow", "right_wrist", "right_shoulder_to_right_elbow",
                                              "right_elbow_to_right_wrist"],
        "torso_mask": ["left_hip", "right_hip", "left_hip_to_right_hip",
                                          "left_shoulder_to_left_hip", "right_shoulder_to_right_hip",
                                          "left_shoulder_to_right_shoulder"],
        "left_leg_mask": ["left_knee", "left_ankle", "left_ankle_to_left_knee",
                                             "left_knee_to_left_hip", "left_hip_to_right_hip"],
        "right_leg_mask": ["right_knee", "right_ankle", "right_ankle_to_right_knee",
                                         "right_knee_to_right_hip"],
    }

    def __init__(self):
        super().__init__(self.parts_grouping, PIFPAF_PARTS_MAP)


class CombinePifPafIntoSixBodyMasksSum(MaskGroupingTransform):
    parts_grouping = {
        "head_mask": ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_eye_to_right_eye",
                                         "nose_to_left_eye", "nose_to_right_eye", "left_eye_to_left_ear",
                                         "right_eye_to_right_ear", "left_ear_to_left_shoulder",
                                         "right_ear_to_right_shoulder"],
        "left_arm_mask": ["left_shoulder", "left_elbow", "left_wrist", "left_shoulder_to_left_elbow",
                                         "left_elbow_to_left_wrist"],
        "right_arm_mask": ["right_shoulder", "right_elbow", "right_wrist", "right_shoulder_to_right_elbow",
                                          "right_elbow_to_right_wrist"],
        "torso_mask": ["left_hip", "right_hip", "left_hip_to_right_hip",
                                      "left_shoulder_to_left_hip", "right_shoulder_to_right_hip",
                                      "left_shoulder_to_right_shoulder"],
        "left_leg_mask": ["left_knee", "left_ankle", "left_ankle_to_left_knee",
                                         "left_knee_to_left_hip", "left_hip_to_right_hip"],
        "right_leg_mask": ["right_knee", "right_ankle", "right_ankle_to_right_knee",
                                          "right_knee_to_right_hip"],
    }

    def __init__(self):
        super().__init__(self.parts_grouping, PIFPAF_PARTS, 'sum')


class CombinePifPafIntoSixBodyMasksSimilarToEight(MaskGroupingTransform):
    parts_grouping = {
        "head_mask": ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_eye_to_right_eye",
                                         "nose_to_left_eye", "nose_to_right_eye", "left_eye_to_left_ear",
                                         "right_eye_to_right_ear", "left_ear_to_left_shoulder",
                                         "right_ear_to_right_shoulder"],
        "torso_mask": ["left_hip", "right_hip", "left_hip_to_right_hip",
                                          "left_shoulder_to_left_hip", "right_shoulder_to_right_hip",
                                          "left_shoulder_to_right_shoulder"],
        "left_arm_mask": ["left_shoulder", "left_elbow", "left_wrist", "left_shoulder_to_left_elbow",
                                             "left_elbow_to_left_wrist"],
        "right_arm_mask": ["right_shoulder", "right_elbow", "right_wrist", "right_shoulder_to_right_elbow",
                                              "right_elbow_to_right_wrist"],
        "leg_mask": ["left_knee", "left_ankle_to_left_knee",
                                             "left_knee_to_left_hip", "left_hip_to_right_hip", "right_knee", "right_ankle_to_right_knee",
                                         "right_knee_to_right_hip"],
        "feet_mask": ["left_ankle", "right_ankle"],
    }

    def __init__(self):
        super().__init__(self.parts_grouping, PIFPAF_PARTS_MAP)


class CombinePifPafIntoEightBodyMasks(MaskGroupingTransform):
    parts_grouping = {
        "head_mask": ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_eye_to_right_eye",
                                         "nose_to_left_eye", "nose_to_right_eye", "left_eye_to_left_ear",
                                         "right_eye_to_right_ear", "left_ear_to_left_shoulder",
                                         "right_ear_to_right_shoulder"],
        "left_arm_mask": ["left_shoulder", "left_elbow", "left_wrist", "left_shoulder_to_left_elbow",
                                             "left_elbow_to_left_wrist"],
        "right_arm_mask": ["right_shoulder", "right_elbow", "right_wrist", "right_shoulder_to_right_elbow",
                                              "right_elbow_to_right_wrist"],
        "torso_mask": ["left_hip", "right_hip", "left_hip_to_right_hip",
                                          "left_shoulder_to_left_hip", "right_shoulder_to_right_hip",
                                          "left_shoulder_to_right_shoulder"],
        "left_leg_mask": ["left_knee", "left_ankle_to_left_knee",
                                             "left_knee_to_left_hip", "left_hip_to_right_hip"],
        "right_leg_mask": ["right_knee", "right_ankle_to_right_knee",
                                         "right_knee_to_right_hip"],
        "left_feet_mask": ["left_ankle"],
        "right_feet_mask": ["right_ankle"],
    }

    def __init__(self):
        super().__init__(self.parts_grouping, PIFPAF_PARTS_MAP)



class CombinePifPafIntoEightVerticalBodyMasks(MaskGroupingTransform):
    parts_grouping = {
        "head_mask": ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_eye_to_right_eye",
                                         "nose_to_left_eye", "nose_to_right_eye", "left_eye_to_left_ear",
                                         "right_eye_to_right_ear", "left_ear_to_left_shoulder",
                                         "right_ear_to_right_shoulder"],
        "left_arm_mask": ["left_shoulder", "left_elbow", "left_wrist", "left_shoulder_to_left_elbow",
                                             "left_elbow_to_left_wrist"],
        "right_arm_mask": ["right_shoulder", "right_elbow", "right_wrist", "right_shoulder_to_right_elbow",
                                              "right_elbow_to_right_wrist"],
        "torso_mask": ["left_hip", "right_hip", "left_hip_to_right_hip",
                                          "left_shoulder_to_left_hip", "right_shoulder_to_right_hip",
                                          "left_shoulder_to_right_shoulder"],
        "left_leg_mask": ["left_knee", "left_ankle_to_left_knee",
                                             "left_knee_to_left_hip", "left_hip_to_right_hip"],
        "right_leg_mask": ["right_knee", "right_ankle_to_right_knee",
                                         "right_knee_to_right_hip"],
        "left_feet_mask": ["left_ankle"],
        "right_feet_mask": ["right_ankle"],
    }

    def __init__(self):
        super().__init__(self.parts_grouping, PIFPAF_PARTS_MAP)


class CombinePifPafIntoTenMSBodyMasks(MaskGroupingTransform):
    parts_grouping = {
        "head_mask": ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_eye_to_right_eye",
                                         "nose_to_left_eye", "nose_to_right_eye", "left_eye_to_left_ear",
                                         "right_eye_to_right_ear", "left_ear_to_left_shoulder",
                                         "right_ear_to_right_shoulder"],
        "left_arm_mask": ["left_shoulder", "left_elbow", "left_wrist", "left_shoulder_to_left_elbow",
                                             "left_elbow_to_left_wrist"],
        "right_arm_mask": ["right_shoulder", "right_elbow", "right_wrist", "right_shoulder_to_right_elbow",
                                              "right_elbow_to_right_wrist"],
        "torso_mask": ["left_hip", "right_hip", "left_hip_to_right_hip",
                                          "left_shoulder_to_left_hip", "right_shoulder_to_right_hip",
                                          "left_shoulder_to_right_shoulder"],
        "left_leg_mask": ["left_knee", "left_ankle_to_left_knee",
                                             "left_knee_to_left_hip", "left_hip_to_right_hip"],
        "right_leg_mask": ["right_knee", "right_ankle_to_right_knee",
                                         "right_knee_to_right_hip"],
        "left_feet_mask": ["left_ankle"],
        "right_feet_mask": ["right_ankle"],

        "upper_body_mask": ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_eye_to_right_eye",
                                         "nose_to_left_eye", "nose_to_right_eye", "left_eye_to_left_ear",
                                         "right_eye_to_right_ear", "left_ear_to_left_shoulder",
                                         "right_ear_to_right_shoulder",
                            "left_shoulder", "left_elbow", "left_wrist", "left_shoulder_to_left_elbow",
                            "left_elbow_to_left_wrist",
                            "right_shoulder", "right_elbow", "right_wrist", "right_shoulder_to_right_elbow",
                            "right_elbow_to_right_wrist",
                            "left_hip", "right_hip", "left_hip_to_right_hip",
                            "left_shoulder_to_left_hip", "right_shoulder_to_right_hip",
                            "left_shoulder_to_right_shoulder"],
        "lower_body_mask": ["left_knee", "left_ankle_to_left_knee",
                                             "left_knee_to_left_hip", "left_hip_to_right_hip",
                            "right_knee", "right_ankle_to_right_knee",
                            "right_knee_to_right_hip",
                            "left_ankle",
                            "right_ankle"],
    }

    def __init__(self):
        super().__init__(self.parts_grouping, PIFPAF_PARTS_MAP)


class CombinePifPafIntoSevenVerticalBodyMasks(MaskGroupingTransform):
    parts_grouping = {
        "head_mask": ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_eye_to_right_eye",
                                         "nose_to_left_eye", "nose_to_right_eye", "left_eye_to_left_ear",
                                         "right_eye_to_right_ear", "left_ear_to_left_shoulder",
                                         "right_ear_to_right_shoulder"],
        "shoulders_mask": ["left_shoulder", "right_shoulder", "left_shoulder_to_right_shoulder"],
        "elbow_mask": ["left_elbow", "right_elbow"],
        "wrist_mask": ["left_wrist", "right_wrist"],
        "hip_mask": ["left_hip", "right_hip", "left_hip_to_right_hip"],
        "knee_mask": ["left_knee", "right_knee"],
        "ankle_mask": ["left_ankle", "right_ankle"],
    }

    def __init__(self):
        super().__init__(self.parts_grouping, PIFPAF_PARTS_MAP)


class CombinePifPafIntoSevenBodyMasksSimilarToEight(MaskGroupingTransform):
    parts_grouping = {
        "head_mask": ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_eye_to_right_eye",
                                         "nose_to_left_eye", "nose_to_right_eye", "left_eye_to_left_ear",
                                         "right_eye_to_right_ear", "left_ear_to_left_shoulder",
                                         "right_ear_to_right_shoulder"],
        "left_arm_mask": ["left_shoulder", "left_elbow", "left_wrist", "left_shoulder_to_left_elbow",
                                             "left_elbow_to_left_wrist"],
        "right_arm_mask": ["right_shoulder", "right_elbow", "right_wrist", "right_shoulder_to_right_elbow",
                                              "right_elbow_to_right_wrist"],
        "upper_torso_mask": ["left_shoulder_to_left_hip", "right_shoulder_to_right_hip",
                                          "left_shoulder_to_right_shoulder"],
        "lower_torso_mask": ["left_hip", "right_hip", "left_hip_to_right_hip"],
        "leg_mask": ["left_knee", "left_ankle_to_left_knee",
                                             "left_knee_to_left_hip", "left_hip_to_right_hip", "right_knee", "right_ankle_to_right_knee",
                                         "right_knee_to_right_hip"],
        "feet_mask": ["left_ankle", "right_ankle"],
    }

    def __init__(self):
        super().__init__(self.parts_grouping, PIFPAF_PARTS_MAP)



class CombinePifPafIntoElevenBodyMasks(MaskGroupingTransform):
    parts_grouping = {
        "head_mask": ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_eye_to_right_eye",
                                         "nose_to_left_eye", "nose_to_right_eye", "left_eye_to_left_ear",
                                         "right_eye_to_right_ear", "left_ear_to_left_shoulder",
                                         "right_ear_to_right_shoulder"],
        "left_elbow_mask": ["left_shoulder", "left_elbow", "left_shoulder_to_left_elbow"],
        "left_wrist_mask": ["left_wrist", "left_elbow_to_left_wrist"],

        "right_elbow_mask": ["right_shoulder", "right_elbow", "right_shoulder_to_right_elbow"],
        "right_wrist_mask": ["right_wrist", "right_elbow_to_right_wrist"],

        "upper_torso_mask": ["left_shoulder_to_left_hip", "right_shoulder_to_right_hip", "left_shoulder_to_right_shoulder"],

        "lower_torso_mask": ["left_hip", "right_hip", "left_hip_to_right_hip"],
        "left_leg_mask": ["left_knee", "left_knee_to_left_hip", "left_hip_to_right_hip"],
        "right_leg_mask": ["right_knee", "right_knee_to_right_hip"],
        "left_feet_mask": ["left_ankle_to_left_knee", "left_ankle"],
        "right_feet_mask": ["right_ankle_to_right_knee", "right_ankle"],
    }

    def __init__(self):
        super().__init__(self.parts_grouping, PIFPAF_PARTS_MAP)


class CombinePifPafIntoFourteenBodyMasks(MaskGroupingTransform):
    parts_grouping = {
        "head_mask": ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_eye_to_right_eye",
                                         "nose_to_left_eye", "nose_to_right_eye", "left_eye_to_left_ear",
                                         "right_eye_to_right_ear"],
        "neck_mask": ["left_ear_to_left_shoulder", "right_ear_to_right_shoulder"],
        "left_elbow_mask": ["left_shoulder", "left_elbow", "left_shoulder_to_left_elbow"],
        "left_wrist_mask": ["left_wrist", "left_elbow_to_left_wrist"],
        "right_elbow_mask": ["right_shoulder", "right_elbow", "right_shoulder_to_right_elbow"],
        "right_wrist_mask": ["right_wrist", "right_elbow_to_right_wrist"],
        "upper_torso_mask": ["left_shoulder_to_left_hip", "right_shoulder_to_right_hip", "left_shoulder_to_right_shoulder"],
        "lower_torso_mask": ["left_hip", "right_hip", "left_hip_to_right_hip"],
        "left_leg_mask": ["left_knee", "left_knee_to_left_hip", "left_hip_to_right_hip"],
        "right_leg_mask": ["right_knee", "right_knee_to_right_hip"],
        "left_tibia_mask": ["left_ankle_to_left_knee"],
        "right_tibia_mask": ["right_ankle_to_right_knee"],
        "left_feet_mask": ["left_ankle"],
        "right_feet_mask": ["right_ankle"],
    }

    def __init__(self):
        super().__init__(self.parts_grouping, PIFPAF_PARTS_MAP)
