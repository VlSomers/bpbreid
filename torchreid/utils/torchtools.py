from __future__ import division, print_function, absolute_import

import itertools
import pickle
import shutil
import os.path as osp
import warnings
from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn

from .tools import mkdir_if_missing
import collections
from torch._six import string_classes
import re

__all__ = [
    'save_checkpoint', 'load_checkpoint', 'resume_from_checkpoint',
    'open_all_layers', 'open_specified_layers', 'count_num_param',
    'load_pretrained_weights'
]


def save_checkpoint(
    state, save_dir, job_id=None, is_best=False, remove_module_from_keys=False
):
    r"""Saves checkpoint.

    Args:
        state (dict): dictionary.
        save_dir (str): directory to save checkpoint.
        is_best (bool, optional): if True, this checkpoint will be copied and named
            ``model-best.pth.tar``. Default is False.
        remove_module_from_keys (bool, optional): whether to remove "module."
            from layer names. Default is False.

    Examples::
        >>> state = {
        >>>     'state_dict': model.state_dict(),
        >>>     'epoch': 10,
        >>>     'rank1': 0.5,
        >>>     'optimizer': optimizer.state_dict()
        >>> }
        >>> save_checkpoint(state, 'log/my_model')
    """
    mkdir_if_missing(save_dir)
    if remove_module_from_keys:
        # remove 'module.' in state_dict's keys
        state_dict = state['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            new_state_dict[k] = v
        state['state_dict'] = new_state_dict
    # save
    epoch = state['epoch']
    fpath = osp.join(save_dir, 'job-{}_{}_model.pth.tar'.format(job_id, str(epoch)))
    torch.save(state, fpath)
    print('Checkpoint saved to "{}"'.format(fpath))
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'model-best.pth.tar'))


def load_checkpoint(fpath):
    r"""Loads checkpoint.

    ``UnicodeDecodeError`` can be well handled, which means
    python2-saved files can be read from python3.

    Args:
        fpath (str): path to checkpoint.

    Returns:
        dict

    Examples::
        >>> from torchreid.utils import load_checkpoint
        >>> fpath = 'log/my_model/model.pth.tar-10'
        >>> checkpoint = load_checkpoint(fpath)
    """
    if fpath is None:
        raise ValueError('File path is None')
    if not osp.exists(fpath):
        raise FileNotFoundError('File is not found at "{}"'.format(fpath))
    map_location = None if torch.cuda.is_available() else 'cpu'
    try:
        checkpoint = torch.load(fpath, map_location=map_location)
    except UnicodeDecodeError:
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        checkpoint = torch.load(
            fpath, pickle_module=pickle, map_location=map_location
        )
    except Exception:
        print('Unable to load checkpoint from "{}"'.format(fpath))
        raise
    return checkpoint


def resume_from_checkpoint(fpath, model, optimizer=None, scheduler=None):
    r"""Resumes training from a checkpoint.

    This will load (1) model weights and (2) ``state_dict``
    of optimizer if ``optimizer`` is not None.

    Args:
        fpath (str): path to checkpoint.
        model (nn.Module): model.
        optimizer (Optimizer, optional): an Optimizer.
        scheduler (LRScheduler, optional): an LRScheduler.

    Returns:
        int: start_epoch.

    Examples::
        >>> from torchreid.utils import resume_from_checkpoint
        >>> fpath = 'log/my_model/model.pth.tar-10'
        >>> start_epoch = resume_from_checkpoint(
        >>>     fpath, model, optimizer, scheduler
        >>> )
    """
    print('Loading checkpoint from "{}"'.format(fpath))
    checkpoint = load_checkpoint(fpath)
    model.load_state_dict(checkpoint['state_dict'])
    print('Loaded model weights')
    if optimizer is not None and 'optimizer' in checkpoint.keys():
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('Loaded optimizer')
    if scheduler is not None and 'scheduler' in checkpoint.keys():
        scheduler.load_state_dict(checkpoint['scheduler'])
        print('Loaded scheduler')
    start_epoch = checkpoint['epoch']
    print('Last epoch = {}'.format(start_epoch))
    if 'rank1' in checkpoint.keys():
        print('Last rank1 = {:.1%}'.format(checkpoint['rank1']))
    return start_epoch


def adjust_learning_rate(
    optimizer,
    base_lr,
    epoch,
    stepsize=20,
    gamma=0.1,
    linear_decay=False,
    final_lr=0,
    max_epoch=100
):
    r"""Adjusts learning rate.

    Deprecated.
    """
    if linear_decay:
        # linearly decay learning rate from base_lr to final_lr
        frac_done = epoch / max_epoch
        lr = frac_done*final_lr + (1.-frac_done) * base_lr
    else:
        # decay learning rate by gamma for every stepsize
        lr = base_lr * (gamma**(epoch // stepsize))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def set_bn_to_eval(m):
    r"""Sets BatchNorm layers to eval mode."""
    # 1. no update for running mean and var
    # 2. scale and shift parameters are still trainable
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def open_all_layers(model):
    r"""Opens all layers in model for training.

    Examples::
        >>> from torchreid.utils import open_all_layers
        >>> open_all_layers(model)
    """
    model.train()
    for p in model.parameters():
        p.requires_grad = True


def open_specified_layers(model, open_layers):
    r"""Opens specified layers in model for training while keeping
    other layers frozen.

    Args:
        model (nn.Module): neural net model.
        open_layers (str or list): layers open for training.

    Examples::
        >>> from torchreid.utils import open_specified_layers
        >>> # Only model.classifier will be updated.
        >>> open_layers = 'classifier'
        >>> open_specified_layers(model, open_layers)
        >>> # Only model.fc and model.classifier will be updated.
        >>> open_layers = ['fc', 'classifier']
        >>> open_specified_layers(model, open_layers)
    """
    if isinstance(model, nn.DataParallel):
        model = model.module

    if isinstance(open_layers, str):
        open_layers = [open_layers]

    for layer in open_layers:
        assert hasattr(
            model, layer
        ), '"{}" is not an attribute of the model, please provide the correct name'.format(
            layer
        )

    for name, module in model.named_children():
        if name in open_layers:
            module.train()
            for p in module.parameters():
                p.requires_grad = True
        else:
            module.eval()
            for p in module.parameters():
                p.requires_grad = False


def count_num_param(model):
    r"""Counts number of parameters in a model while ignoring ``self.classifier``.

    Args:
        model (nn.Module): network model.

    Examples::
        >>> from torchreid.utils import count_num_param
        >>> model_size = count_num_param(model)

    .. warning::
        
        This method is deprecated in favor of
        ``torchreid.utils.compute_model_complexity``.
    """
    warnings.warn(
        'This method is deprecated and will be removed in the future.'
    )

    num_param = sum(p.numel() for p in model.parameters())

    if isinstance(model, nn.DataParallel):
        model = model.module

    if hasattr(model,
               'classifier') and isinstance(model.classifier, nn.Module):
        # we ignore the classifier because it is unused at test time
        num_param -= sum(p.numel() for p in model.classifier.parameters())

    return num_param


def load_pretrained_weights(model, weight_path):
    r"""Loads pretrained weights to model.

    Features::
        - Incompatible layers (unmatched in name or size) will be ignored.
        - Can automatically deal with keys containing "module.".

    Args:
        model (nn.Module): network model.
        weight_path (str): path to pretrained weights.

    Examples::
        >>> from torchreid.utils import load_pretrained_weights
        >>> weight_path = 'log/my_model/model-best.pth.tar'
        >>> load_pretrained_weights(model, weight_path)
    """
    checkpoint = load_checkpoint(weight_path)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:] # discard module.

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    if len(matched_layers) == 0:
        warnings.warn(
            'The pretrained weights "{}" cannot be loaded, '
            'please check the key names manually '
            '(** ignored and continue **)'.format(weight_path)
        )
    else:
        print(
            'Successfully loaded pretrained weights from "{}"'.
            format(weight_path)
        )
        if len(discarded_layers) > 0:
            print(
                '** The following layers are discarded '
                'due to unmatched keys or layer size: {}'.
                format(discarded_layers)
            )


# Copied from torch.utils.data._utils.collate.default_collate
np_str_obj_array_pattern = re.compile(r'[SaUO]')
default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")
def collate(batch):
    r"""
        Function that takes in a batch of data and puts the elements within the batch
        into a tensor with an additional outer dimension - batch size. The exact output type can be
        a :class:`torch.Tensor`, a `Sequence` of :class:`torch.Tensor`, a
        Collection of :class:`torch.Tensor`, or left unchanged, depending on the input type.
        This is used as the default function for collation when
        `batch_size` or `batch_sampler` is defined in :class:`~torch.utils.data.DataLoader`.

        Here is the general input type (based on the type of the element within the batch) to output type mapping:
        * :class:`torch.Tensor` -> :class:`torch.Tensor` (with an added outer dimension batch size)
        * NumPy Arrays -> :class:`torch.Tensor`
        * `float` -> :class:`torch.Tensor`
        * `int` -> :class:`torch.Tensor`
        * `str` -> `str` (unchanged)
        * `bytes` -> `bytes` (unchanged)
        * `Mapping[K, V_i]` -> `Mapping[K, default_collate([V_1, V_2, ...])]`
        * `NamedTuple[V1_i, V2_i, ...]` -> `NamedTuple[default_collate([V1_1, V1_2, ...]), default_collate([V2_1, V2_2, ...]), ...]`
        * `Sequence[V1_i, V2_i, ...]` -> `Sequence[default_collate([V1_1, V1_2, ...]), default_collate([V2_1, V2_2, ...]), ...]`

        Args:
            batch: a single batch to be collated

        Examples:
            >>> # Example with a batch of `int`s:
            >>> default_collate([0, 1, 2, 3])
            tensor([0, 1, 2, 3])
            >>> # Example with a batch of `str`s:
            >>> default_collate(['a', 'b', 'c'])
            ['a', 'b', 'c']
            >>> # Example with `Map` inside the batch:
            >>> default_collate([{'A': 0, 'B': 1}, {'A': 100, 'B': 100}])
            {'A': tensor([  0, 100]), 'B': tensor([  1, 100])}
            >>> # Example with `NamedTuple` inside the batch:
            >>> Point = namedtuple('Point', ['x', 'y'])
            >>> default_collate([Point(0, 0), Point(1, 1)])
            Point(x=tensor([0, 1]), y=tensor([0, 1]))
            >>> # Example with `Tuple` inside the batch:
            >>> default_collate([(0, 1), (2, 3)])
            [tensor([0, 2]), tensor([1, 3])]
            >>> # Example with `List` inside the batch:
            >>> default_collate([[0, 1], [2, 3]])
            [tensor([0, 2]), tensor([1, 3])]
    """
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage).resize_(len(batch), *list(elem.size()))
        return torch.cat(batch, dim=0, out=out).numpy() # changed to cat from original code
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return collate([b for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch).numpy()
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64).numpy()
    elif isinstance(elem, int):
        return torch.tensor(batch).numpy()
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        try:
            return elem_type({key: collate([d[key] for d in batch]) for key in elem})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        return list(itertools.chain.from_iterable(batch))
        # it = iter(batch)
        # elem_size = len(next(it))
        # # if not all(len(elem) == elem_size for elem in it):
        # #     raise RuntimeError('each element in list of batch should be of equal size')
        # transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.
        #
        # if isinstance(elem, tuple):
        #     return [collate(samples) for samples in transposed]  # Backwards compatibility.
        # else:
        #     try:
        #         return elem_type([collate(samples) for samples in transposed])
        #     except TypeError:
        #         # The sequence type may not support `__init__(iterable)` (e.g., `range`).
        #         return [collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))
