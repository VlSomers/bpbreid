from torchreid.data.masks_transforms.mask_transform import MaskGroupingTransform

COCO_KEYPOINTS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
                  "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
                  "right_knee", "left_ankle", "right_ankle"]

COCO_KEYPOINTS_MAP = {k: i for i, k in enumerate(COCO_KEYPOINTS)}


class CocoToSixBodyMasks(MaskGroupingTransform):
    parts_grouping = {
        "head": ["nose", "left_eye", "right_eye", "left_ear", "right_ear"],
        "torso": ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
        "left_arm": ["left_shoulder", "left_elbow", "left_wrist"],
        "right_arm": ["right_shoulder", "right_elbow", "right_wrist"],
        "left_leg": ["left_hip", "left_knee", "left_ankle"],
        "right_leg": ["right_hip", "right_knee", "right_ankle"]
    }

    def __init__(self):
        super().__init__(self.parts_grouping, COCO_KEYPOINTS_MAP)