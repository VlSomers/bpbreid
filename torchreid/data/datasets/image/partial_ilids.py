from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os.path as osp
import glob
import warnings

from ..dataset import ImageDataset

# Sources :
# https://github.com/hh23333/PVPM
# Lingxiao He, Jian Liang, Haiqing Li, and Zhenan Sun, "Deep spatial feature reconstruction for partial person reidentification: Alignment-free approach", 2018

class Partial_iLIDS(ImageDataset):
    dataset_dir = 'Partial_iLIDS'

    def __init__(self, root='', **kwargs):
        self.root=osp.abspath(osp.expanduser(root))
        # self.dataset_dir = self.root
        data_dir = osp.join(self.root, self.dataset_dir)
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn('The current data structure is deprecated.')
        self.query_dir = osp.join(self.data_dir, 'partial_body_images')
        self.gallery_dir = osp.join(self.data_dir, 'whole_body_images')

        train = []
        query = self.process_dir(self.query_dir)
        gallery = self.process_dir(self.gallery_dir, is_query=False)
        super(Partial_iLIDS, self).__init__(train, query, gallery, **kwargs)
        self.load_pose = isinstance(self.transform, tuple)
        if self.load_pose:
            if self.mode == 'query':
                self.pose_dir = osp.join(self.data_dir, 'occluded_body_pose')
            elif self.mode == 'gallery':
                self.pose_dir = osp.join(self.data_dir, 'whole_body_pose')
            else:
                self.pose_dir = ''

    def process_dir(self, dir_path, is_query=True):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        if is_query:
            camid = 0
        else:
            camid = 1

        data = []
        for img_path in img_paths:
            img_name = img_path.split('/')[-1]
            pid = int(img_name.split('.')[0])
            data.append({'img_path': img_path, 'pid': pid, 'camid': camid})
        return data
