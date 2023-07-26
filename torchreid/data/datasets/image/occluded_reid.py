from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import os.path as osp
import glob
import warnings
from ..dataset import ImageDataset

# Sources :
# https://github.com/hh23333/PVPM
# Zhuo, J., Chen, Z., Lai, J., & Wang, G. (2018). Occluded Person Re-identification.


class OccludedReID(ImageDataset):
    dataset_dir = 'Occluded_REID'
    masks_base_dir = 'masks'

    masks_dirs = {
        # dir_name: (parts_num, masks_stack_size, contains_background_mask)
        'pifpaf': (36, False, '.tif.confidence_fields.npy'),
        'pifpaf_maskrcnn_filtering': (36, False, '.npy'),
    }

    @staticmethod
    def get_masks_config(masks_dir):
        if masks_dir not in OccludedReID.masks_dirs:
            return None
        else:
            return OccludedReID.masks_dirs[masks_dir]

    def infer_masks_path(self, img_path):
        masks_path = os.path.join(self.dataset_dir, self.masks_base_dir, self.masks_dir, os.path.basename(os.path.dirname(os.path.dirname(img_path))), os.path.splitext(os.path.basename(img_path))[0] + self.masks_suffix)
        return masks_path

    def __init__(self, root='', masks_dir=None, **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.masks_dir = masks_dir
        if self.masks_dir in self.masks_dirs:
            self.masks_parts_numbers, self.has_background, self.masks_suffix = self.masks_dirs[self.masks_dir]
        else:
            self.masks_parts_numbers, self.has_background, self.masks_suffix = None, None, None
        # allow alternative directory structure
        if not osp.isdir(self.dataset_dir):
            warnings.warn(
                'The current data structure is deprecated. Please '
                'put data folders such as "bounding_box_train" under '
                '"Market-1501-v15.09.15".'
            )
        self.query_dir = osp.join(self.dataset_dir, 'occluded_body_images')
        self.gallery_dir = osp.join(self.dataset_dir, 'whole_body_images')

        train = []
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False, is_query=False)
        super(OccludedReID, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, relabel=False, is_query=True):
        img_paths = glob.glob(osp.join(dir_path, '*', '*.tif'))
        if is_query:
            camid = 0
        else:
            camid = 1
        pid_container = set()
        for img_path in img_paths:
            img_name = img_path.split('/')[-1]
            pid = int(img_name.split('_')[0])
            pid_container.add(pid)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            img_name = img_path.split('/')[-1]
            pid = int(img_name.split('_')[0])
            if relabel:
                pid = pid2label[pid]
            masks_path = self.infer_masks_path(img_path)
            data.append({'img_path': img_path,
                         'pid': pid,
                         'masks_path': masks_path,
                         'camid': camid})
        return data
