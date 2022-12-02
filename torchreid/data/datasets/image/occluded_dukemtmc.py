from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os.path as osp
import glob
import re
from ..dataset import ImageDataset

# Sources :
# https://github.com/hh23333/PVPM
# https://github.com/lightas/Occluded-DukeMTMC-Dataset
# Miao, J., Wu, Y., Liu, P., DIng, Y., & Yang, Y. (2019). "Pose-guided feature alignment for occluded person re-identification". ICCV 2019

class OccludedDuke(ImageDataset):
    dataset_dir = 'Occluded_Duke'
    masks_base_dir = 'masks'

    masks_dirs = {
        # dir_name: (parts_num, masks_stack_size, contains_background_mask)
        'pifpaf': (36, False, '.jpg.confidence_fields.npy'),
        'pifpaf_maskrcnn_filtering': (36, False, '.jpg.confidence_fields.npy'),
        'isp_6_parts': (5, True, '.jpg.confidence_fields.npy', ["p{}".format(p) for p in range(1, 5+1)])
    }

    @staticmethod
    def get_masks_config(masks_dir):
        if masks_dir not in OccludedDuke.masks_dirs:
            return None
        else:
            return OccludedDuke.masks_dirs[masks_dir]

    def __init__(self, root='', masks_dir=None, **kwargs):
        self.masks_dir = masks_dir
        if self.masks_dir in self.masks_dirs:
            self.masks_parts_numbers, self.has_background, self.masks_suffix = self.masks_dirs[self.masks_dir]
        else:
            self.masks_parts_numbers, self.has_background, self.masks_suffix = None, None, None
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        required_files = [
            self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)

        super(OccludedDuke, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            assert 1 <= camid <= 8
            camid -= 1  # index starts from 0
            if relabel:
                pid = pid2label[pid]
            masks_path = self.infer_masks_path(img_path)
            data.append({'img_path': img_path,
                         'pid': pid,
                         'masks_path': masks_path,
                         'camid': camid})

        return data
