from __future__ import division, print_function, absolute_import
import copy
import os

import numpy as np
import os.path as osp
import tarfile
import zipfile
import torch

from torchreid.utils import read_masks, read_image, download_url, mkdir_if_missing


class Dataset(object):
    """An abstract class representing a Dataset.

    This is the base class for ``ImageDataset`` and ``VideoDataset``.

    Args:
        train (list): contains tuples of (img_path(s), pid, camid).
        query (list): contains tuples of (img_path(s), pid, camid).
        gallery (list): contains tuples of (img_path(s), pid, camid).
        transform: transform function.
        mode (str): 'train', 'query' or 'gallery'.
        combineall (bool): combines train, query and gallery in a
            dataset for training.
        verbose (bool): show information.
    """
    _junk_pids = [
    ] # contains useless person IDs, e.g. background, false detections

    masks_base_dir = None
    eval_metric = 'default'  # default to market101

    def gallery_filter(self, q_pid, q_camid, q_ann, g_pids, g_camids, g_anns):
        """ Remove gallery samples that have the same pid and camid as the query sample, since ReID is a cross-camera
        person retrieval task for most datasets. However, we still keep samples from the same camera but of different
        identity as distractors."""
        remove = (g_camids == q_camid) & (g_pids == q_pid)
        return remove

    def infer_masks_path(self, img_path):
        masks_path = os.path.join(self.dataset_dir, self.masks_base_dir, self.masks_dir, os.path.basename(os.path.dirname(img_path)), os.path.splitext(os.path.basename(img_path))[0] + self.masks_suffix)
        return masks_path

    def __init__(
        self,
        train,
        query,
        gallery,
        config=None,
        transform_tr=None,
        transform_te=None,
        mode='train',
        combineall=False,
        verbose=True,
        use_masks=False,
        masks_dir=None,
        masks_base_dir=None,
        load_masks=False,
        **kwargs
    ):
        self.train = train
        self.query = query
        self.gallery = gallery
        self.transform_tr = transform_tr
        self.transform_te = transform_te
        self.cfg = config
        self.mode = mode
        self.combineall = combineall
        self.verbose = verbose
        self.use_masks = use_masks
        self.masks_dir = masks_dir
        self.load_masks = load_masks
        if masks_base_dir is not None:
            self.masks_base_dir = masks_base_dir

        self.num_train_pids = self.get_num_pids(self.train)
        self.num_train_cams = self.get_num_cams(self.train)

        if self.combineall:
            self.combine_all()

        if self.verbose:
            self.show_summary()

    def transforms(self, mode):
        """Returns the transforms of a specific mode."""
        if mode == 'train':
            return self.transform_tr
        elif mode == 'query':
            return self.transform_te
        elif mode == 'gallery':
            return self.transform_te
        else:
            raise ValueError("Invalid mode. Got {}, but expected to be "
                             "'train', 'query' or 'gallery'".format(mode))

    def data(self, mode):
        """Returns the data of a specific mode.

        Args:
            mode (str): 'train', 'query' or 'gallery'.

        Returns:
            list: contains tuples of (img_path(s), pid, camid).
        """
        if mode == 'train':
            return self.train
        elif mode == 'query':
            return self.query
        elif mode == 'gallery':
            return self.gallery
        else:
            raise ValueError("Invalid mode. Got {}, but expected to be "
                             "'train', 'query' or 'gallery'".format(mode))

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):  # kept for backward compatibility
        return self.len(self.mode)

    def len(self, mode):
        return len(self.data(mode))

    def __add__(self, other):
        """Adds two datasets together (only the train set)."""
        train = copy.deepcopy(self.train)

        for sample in other.train:
            sample['pid'] += self.num_train_pids
            train.append(sample)

        ###################################
        # Things to do beforehand:
        # 1. set verbose=False to avoid unnecessary print
        # 2. set combineall=False because combineall would have been applied
        #    if it was True for a specific dataset, setting it to True will
        #    create new IDs that should have been included
        ###################################


        # FIXME find better implementation for combining datasets and masks
        assert self.use_masks == other.use_masks

        if isinstance(self, ImageDataset):
            return ImageDataset(
                train,
                self.query,
                self.gallery,
                transform=self.transform,
                mode=self.mode,
                combineall=False,
                verbose=False,
                use_masks=self.use_masks,
                masks_base_dir=self.masks_base_dir,
            )
        else:
            return VideoDataset(
                train,
                self.query,
                self.gallery,
                transform=self.transform,
                mode=self.mode,
                combineall=False,
                verbose=False,
                seq_len=self.seq_len,
                sample_method=self.sample_method
            )

    def __radd__(self, other):
        """Supports sum([dataset1, dataset2, dataset3])."""
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def parse_data(self, data):
        """Parses data list and returns the number of person IDs
        and the number of camera views.

        Args:
            data (list): contains tuples of (img_path(s), pid, camid)
        """
        pids = set()
        cams = set()
        for i, sample in enumerate(data):
            pids.add(sample['pid'])
            cams.add(sample['camid'])
        return len(pids), len(cams)

    def get_num_pids(self, data):
        """Returns the number of training person identities."""
        return self.parse_data(data)[0]

    def get_num_cams(self, data):
        """Returns the number of training cameras."""
        return self.parse_data(data)[1]

    def show_summary(self):
        """Shows dataset statistics."""
        pass

    def combine_all(self):
        """Combines train, query and gallery in a dataset for training."""
        combined = copy.deepcopy(self.train)

        # relabel pids in gallery (query shares the same scope)
        g_pids = set()
        for sample in self.gallery:
            pid = sample['pid']
            if pid in self._junk_pids:
                continue
            g_pids.add(pid)
        pid2label = {pid: i for i, pid in enumerate(g_pids)}

        def _combine_data(data):
            for sample in data:
                pid = sample['pid']
                if pid in self._junk_pids:
                    continue
                sample['pid'] = pid2label[pid] + self.num_train_pids
                combined.append(sample)

        _combine_data(self.query)
        _combine_data(self.gallery)

        self.train = combined
        self.num_train_pids = self.get_num_pids(self.train)

    def download_dataset(self, dataset_dir, dataset_url):
        """Downloads and extracts dataset.

        Args:
            dataset_dir (str): dataset directory.
            dataset_url (str): url to download dataset.
        """
        if osp.exists(dataset_dir):
            return

        if dataset_url is None:
            raise RuntimeError(
                '{} dataset needs to be manually '
                'prepared, please follow the '
                'document to prepare this dataset'.format(
                    self.__class__.__name__
                )
            )

        print('Creating directory "{}"'.format(dataset_dir))
        mkdir_if_missing(dataset_dir)
        fpath = osp.join(dataset_dir, osp.basename(dataset_url))

        print(
            'Downloading {} dataset to "{}"'.format(
                self.__class__.__name__, dataset_dir
            )
        )
        download_url(dataset_url, fpath)

        print('Extracting "{}"'.format(fpath))
        try:
            tar = tarfile.open(fpath)
            tar.extractall(path=dataset_dir)
            tar.close()
        except:
            zip_ref = zipfile.ZipFile(fpath, 'r')
            zip_ref.extractall(dataset_dir)
            zip_ref.close()

        print('{} dataset is ready'.format(self.__class__.__name__))

    def check_before_run(self, required_files):
        """Checks if required files exist before going deeper.

        Args:
            required_files (str or list): string file name(s).
        """
        if isinstance(required_files, str):
            required_files = [required_files]

        for fpath in required_files:
            if not osp.exists(fpath):
                raise RuntimeError('"{}" is not found'.format(fpath))

    def __repr__(self):
        num_train_pids, num_train_cams = self.parse_data(self.train)
        num_query_pids, num_query_cams = self.parse_data(self.query)
        num_gallery_pids, num_gallery_cams = self.parse_data(self.gallery)

        msg = '  ----------------------------------------\n' \
              '  subset   | # ids | # items | # cameras\n' \
              '  ----------------------------------------\n' \
              '  train    | {:5d} | {:7d} | {:9d}\n' \
              '  query    | {:5d} | {:7d} | {:9d}\n' \
              '  gallery  | {:5d} | {:7d} | {:9d}\n' \
              '  ----------------------------------------\n' \
              '  items: images/tracklets for image/video dataset\n'.format(
                  num_train_pids, len(self.train), num_train_cams,
                  num_query_pids, len(self.query), num_query_cams,
                  num_gallery_pids, len(self.gallery), num_gallery_cams
              )

        return msg


class ImageDataset(Dataset):
    """A base class representing ImageDataset.

    All other image datasets should subclass it.

    ``__getitem__`` returns an image given index.
    It will return ``img``, ``pid``, ``camid`` and ``img_path``
    where ``img`` has shape (channel, height, width). As a result,
    data in each batch has shape (batch_size, channel, height, width).
    """

    def __init__(self, train, query, gallery, **kwargs):
        super(ImageDataset, self).__init__(train, query, gallery, **kwargs)

    def __getitem__(self, index):  # kept for backward compatibility
        return self.getitem(index, self.mode)

    def getitem(self, index, mode):
        # BPBreID can work with None masks
        # list all combination: source vs target, merged/joined vs not, cross domain or not, load from disk vs fixed for BoT/PBP transform vs None,
        # need masks when available for pixel accuracy prediction
        sample = self.data(mode)[index]
        transf_args = {"image": read_image(sample['img_path'])}
        if self.use_masks:
            if self.load_masks and 'masks_path' in sample:
                transf_args["mask"] = read_masks(sample['masks_path'])
            elif not self.load_masks:
                # hack for BoT and PCB masks that are generated in transform().
                # FIXME BoT and PCB masks should not be generated here, but later in BPBreID model with a config
                transf_args["mask"] = np.ones((1, 2, 2))
            else:
                pass
        result = self.transforms(mode)(**transf_args)
        sample.update(result)
        return sample

    def show_summary(self):
        num_train_pids, num_train_cams = self.parse_data(self.train)
        num_query_pids, num_query_cams = self.parse_data(self.query)
        num_gallery_pids, num_gallery_cams = self.parse_data(self.gallery)

        print('=> Loaded {}'.format(self.__class__.__name__))
        print('  ----------------------------------------')
        print('  subset   | # ids | # images | # cameras')
        print('  ----------------------------------------')
        print(
            '  train    | {:5d} | {:8d} | {:9d}'.format(
                num_train_pids, len(self.train), num_train_cams
            )
        )
        print(
            '  query    | {:5d} | {:8d} | {:9d}'.format(
                num_query_pids, len(self.query), num_query_cams
            )
        )
        print(
            '  gallery  | {:5d} | {:8d} | {:9d}'.format(
                num_gallery_pids, len(self.gallery), num_gallery_cams
            )
        )
        print('  ----------------------------------------')


class VideoDataset(Dataset):
    """A base class representing VideoDataset.

    All other video datasets should subclass it.

    ``__getitem__`` returns an image given index.
    It will return ``imgs``, ``pid`` and ``camid``
    where ``imgs`` has shape (seq_len, channel, height, width). As a result,
    data in each batch has shape (batch_size, seq_len, channel, height, width).
    """

    def __init__(
        self,
        train,
        query,
        gallery,
        seq_len=15,
        sample_method='evenly',
        **kwargs
    ):
        super(VideoDataset, self).__init__(train, query, gallery, **kwargs)
        self.seq_len = seq_len
        self.sample_method = sample_method

        if self.transform is None:
            raise RuntimeError('transform must not be None')

    def getitem(self, index, mode):
        img_paths, pid, camid = self.data(mode)[index]  # FIXME new format
        num_imgs = len(img_paths)

        if self.sample_method == 'random':
            # Randomly samples seq_len images from a tracklet of length num_imgs,
            # if num_imgs is smaller than seq_len, then replicates images
            indices = np.arange(num_imgs)
            replace = False if num_imgs >= self.seq_len else True
            indices = np.random.choice(
                indices, size=self.seq_len, replace=replace
            )
            # sort indices to keep temporal order (comment it to be order-agnostic)
            indices = np.sort(indices)

        elif self.sample_method == 'evenly':
            # Evenly samples seq_len images from a tracklet
            if num_imgs >= self.seq_len:
                num_imgs -= num_imgs % self.seq_len
                indices = np.arange(0, num_imgs, num_imgs / self.seq_len)
            else:
                # if num_imgs is smaller than seq_len, simply replicate the last image
                # until the seq_len requirement is satisfied
                indices = np.arange(0, num_imgs)
                num_pads = self.seq_len - num_imgs
                indices = np.concatenate(
                    [
                        indices,
                        np.ones(num_pads).astype(np.int32) * (num_imgs-1)
                    ]
                )
            assert len(indices) == self.seq_len

        elif self.sample_method == 'all':
            # Samples all images in a tracklet. batch_size must be set to 1
            indices = np.arange(num_imgs)

        else:
            raise ValueError(
                'Unknown sample method: {}'.format(self.sample_method)
            )

        imgs = []
        for index in indices:
            img_path = img_paths[int(index)]
            img = read_image(img_path)
            if self.transform is not None:
                img = self.transform(img)
            img = img.unsqueeze(0) # img must be torch.Tensor
            imgs.append(img)
        imgs = torch.cat(imgs, dim=0)

        return imgs, pid, camid

    def show_summary(self):
        num_train_pids, num_train_cams = self.parse_data(self.train)
        num_query_pids, num_query_cams = self.parse_data(self.query)
        num_gallery_pids, num_gallery_cams = self.parse_data(self.gallery)

        print('=> Loaded {}'.format(self.__class__.__name__))
        print('  -------------------------------------------')
        print('  subset   | # ids | # tracklets | # cameras')
        print('  -------------------------------------------')
        print(
            '  train    | {:5d} | {:11d} | {:9d}'.format(
                num_train_pids, len(self.train), num_train_cams
            )
        )
        print(
            '  query    | {:5d} | {:11d} | {:9d}'.format(
                num_query_pids, len(self.query), num_query_cams
            )
        )
        print(
            '  gallery  | {:5d} | {:11d} | {:9d}'.format(
                num_gallery_pids, len(self.gallery), num_gallery_cams
            )
        )
        print('  -------------------------------------------')
