from __future__ import print_function, absolute_import

import copy

from .image import (
    GRID, PRID, CUHK01, CUHK02, CUHK03, MSMT17, VIPeR, SenseReID, Market1501,
    DukeMTMCreID, iLIDS, OccludedDuke, OccludedReID, Partial_iLIDS, Partial_REID, PDukemtmcReid,
    P_ETHZ
)
from .video import PRID2011, Mars, DukeMTMCVidReID, iLIDSVID
from .dataset import Dataset, ImageDataset, VideoDataset

__image_datasets = {
    'market1501': Market1501,
    'cuhk03': CUHK03,
    'dukemtmcreid': DukeMTMCreID,
    'msmt17': MSMT17,
    'viper': VIPeR,
    'grid': GRID,
    'cuhk01': CUHK01,
    'ilids': iLIDS,
    'sensereid': SenseReID,
    'prid': PRID,
    'cuhk02': CUHK02,
    'occluded_duke': OccludedDuke,
    'occluded_reid': OccludedReID,
    'partial_reid': Partial_REID,
    'partial_ilids': Partial_iLIDS,
    'p_ETHZ': P_ETHZ,
    'p_dukemtmc_reid': PDukemtmcReid,
}

__datasets_nicknames = {
    'market1501': 'mk',
    'cuhk03': 'c03',
    'dukemtmcreid': 'du',
    'msmt17': 'ms',
    'viper': 'vi',
    'grid': 'gr',
    'cuhk01': 'c01',
    'ilids': 'il',
    'sensereid': 'se',
    'prid': 'pr',
    'cuhk02': 'c02',
    'occluded_duke': 'od',
    'occluded_reid': 'or',
    'partial_reid': 'pr',
    'partial_ilids': 'pi',
    'p_ETHZ': 'pz',
    'p_dukemtmc_reid': 'pd',
}

__video_datasets = {
    'mars': Mars,
    'ilidsvid': iLIDSVID,
    'prid2011': PRID2011,
    'dukemtmcvidreid': DukeMTMCVidReID
}

__datasets_cache = {}


def configure_dataset_class(clazz, **ext_kwargs):
    """
    Wrapper function to provide the class with args external to torchreid
    """
    class ClazzWrapper(clazz):
        def __init__(self, **kwargs):
            self.__name__ = clazz.__name__
            super(ClazzWrapper, self).__init__(**{**kwargs, **ext_kwargs})

    ClazzWrapper.__name__ = clazz.__name__

    return ClazzWrapper


def get_dataset_nickname(name):
    return __datasets_nicknames.get(name, name)


def get_image_dataset(name):
    avai_datasets = list(__image_datasets.keys())
    if name not in avai_datasets:
        raise ValueError(
            'Invalid dataset name. Received "{}", '
            'but expected to be one of {}'.format(name, avai_datasets)
        )
    return __image_datasets[name]


def init_image_dataset(name, mode='train', **kwargs):
    """
    Initializes an image dataset.
    The copy.copy() was introduced to fix Torchreid implementing multiple times the same dataset.
    In Datamanager, each dataset was instantiated multiple times via 'init_image_dataset': one for train, one for query
    and one for gallery. Each instance had its own 'data' field containing either train, query or gallery set, based on
    the 'mode' field passed as argument, and its own transforms, to perform training time or test time data transformation.
    However, instantiating the same dataset multiple times is not efficient, as it requires to load the dataset metadata from
    disk multiple times. Moreover, other printing (such as dataset summary) are displayed multiple times.
    To fix this, we copy the dataset class but not its contained objects (such as train/query/gallery) and set a new 'mode' on each copy.
    Thanks to that hack, the data list is created only once, and only the Dataset class is instantiated multiple times
    (for each 'mode'). Therefore, each Dataset uses the same data lists in the background, switching
    between train, query and gallery based on the 'mode' field.
    """
    if name in __datasets_cache:
        print("Using cached dataset {}.".format(name))
        dataset = __datasets_cache[name]
    else:
        print("Creating new dataset {} and add it to the datasets cache.".format(name))
        dataset = get_image_dataset(name)(mode=mode, **kwargs)
        __datasets_cache[name] = dataset
    mode_dataset = copy.copy(dataset)
    mode_dataset.mode = mode
    return mode_dataset


def init_video_dataset(name, **kwargs):
    """Initializes a video dataset."""
    avai_datasets = list(__video_datasets.keys())
    if name not in avai_datasets:
        raise ValueError(
            'Invalid dataset name. Received "{}", '
            'but expected to be one of {}'.format(name, avai_datasets)
        )
    return __video_datasets[name](**kwargs)


def register_image_dataset(name, dataset, nickname=None):
    """Registers a new image dataset.

    Args:
        name (str): key corresponding to the new dataset.
        dataset (Dataset): the new dataset class.

    Examples::
        
        import torchreid
        import NewDataset
        torchreid.data.register_image_dataset('new_dataset', NewDataset)
        # single dataset case
        datamanager = torchreid.data.ImageDataManager(
            root='reid-data',
            sources='new_dataset'
        )
        # multiple dataset case
        datamanager = torchreid.data.ImageDataManager(
            root='reid-data',
            sources=['new_dataset', 'dukemtmcreid']
        )
    """
    global __image_datasets
    curr_datasets = list(__image_datasets.keys())
    if name in curr_datasets:
        raise ValueError(
            'The given name already exists, please choose '
            'another name excluding {}'.format(curr_datasets)
        )
    __image_datasets[name] = dataset
    __datasets_nicknames[name] = nickname if nickname is not None else name


def register_video_dataset(name, dataset):
    """Registers a new video dataset.

    Args:
        name (str): key corresponding to the new dataset.
        dataset (Dataset): the new dataset class.

    Examples::
        
        import torchreid
        import NewDataset
        torchreid.data.register_video_dataset('new_dataset', NewDataset)
        # single dataset case
        datamanager = torchreid.data.VideoDataManager(
            root='reid-data',
            sources='new_dataset'
        )
        # multiple dataset case
        datamanager = torchreid.data.VideoDataManager(
            root='reid-data',
            sources=['new_dataset', 'ilidsvid']
        )
    """
    global __video_datasets
    curr_datasets = list(__video_datasets.keys())
    if name in curr_datasets:
        raise ValueError(
            'The given name already exists, please choose '
            'another name excluding {}'.format(curr_datasets)
        )
    __video_datasets[name] = dataset
