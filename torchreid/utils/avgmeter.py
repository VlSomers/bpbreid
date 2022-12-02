from __future__ import division, absolute_import

import datetime
import warnings
from collections import defaultdict, OrderedDict
import time
import torch
import numpy as np

__all__ = ['AverageMeter', 'MetricMeter', 'TimeMeter', 'TorchTimeMeter', 'EpochMetricsMeter']

from torchreid.utils.engine_state import EngineStateListener


class AverageMeter(object):
    """Computes and stores the average and current value.

    Examples::
        >>> # Initialize a meter to record loss
        >>> losses = AverageMeter()
        >>> # Update meter after every minibatch update
        >>> losses.update(loss_value, batch_size)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class BatchMeter(object):
    def __init__(self, epoch_count, batch_count):
        self.epoch_count = epoch_count
        self.batch_count = batch_count
        self.reset()

    def reset(self):
        self.last_val = None
        self.values = np.zeros((self.epoch_count, self.batch_count))

    def update(self, epoch, batch, val):
        self.last_val = val
        self.values[epoch, batch] = val

    def total_for_epoch(self, epoch):
        self.values[epoch].sum()

    def avg_for_epoch(self, epoch):
        self.values[epoch].mean()

    def batch_avg(self):
        self.values.mean()

    def epoch_avg(self):
        self.values.mean(axis=1).mean()

    def total(self):
        self.values.sum()


class SingleMeter(EngineStateListener):
    def __init__(self, engine_state):
        self.engine_state = engine_state
        self.engine_state.add_listener(self)
        self.reset()
        self.is_empty = True

    def reset(self):
        self.total = 0
        self.val = 0

    def update(self, val, total):
        if torch.is_tensor(val):
            val = val.item()
        if torch.is_tensor(total):
            total = total.item()

        self.val = val
        self.total = total
        self.is_empty = False

    def ratio(self):
        if self.total == 0:
            return 0
        return self.val / self.total


class EpochMeter(EngineStateListener):
    # With RandomSample, number of batches might change from one epoch to another
    def __init__(self, engine_state):
        self.engine_state = engine_state
        self.engine_state.add_listener(self)
        self.min = np.zeros(self.engine_state.max_epoch)
        self.mean = np.zeros(self.engine_state.max_epoch)
        self.max = np.zeros(self.engine_state.max_epoch)
        self.batch_size = np.zeros(self.engine_state.max_epoch)
        self.sum = np.zeros(self.engine_state.max_epoch)
        self.total = np.zeros(self.engine_state.max_epoch)
        self.epoch_sum = []
        self.epoch_total = []
        self.is_empty = True

    def update(self, val, total=1.):
        self.is_empty = False
        if torch.is_tensor(val):
            val = val.item()
        if torch.is_tensor(total):
            total = total.item()

        self.epoch_sum.append(val)
        self.epoch_total.append(total)

    # Listeners
    def epoch_completed(self):
        if not self.is_empty:
            self.epoch_sum = np.array(self.epoch_sum)
            self.epoch_total = np.array(self.epoch_total)
            ratio = self.epoch_sum / self.epoch_total

            self.min[self.engine_state.epoch] = ratio.min()
            self.mean[self.engine_state.epoch] = ratio.mean()
            self.max[self.engine_state.epoch] = ratio.max()
            self.batch_size[self.engine_state.epoch] = len(self.epoch_sum)
            self.sum[self.engine_state.epoch] = self.epoch_sum.sum()
            self.total[self.engine_state.epoch] = self.epoch_total.sum()

            self.epoch_sum = []
            self.epoch_total = []
        self.is_empty = True

    # Utils
    def last_val(self):
        return self.epoch_sum[-1]

    def epoch_ratio(self, epoch):
        return self.mean[epoch]

    def total_ratio(self):
        return self.mean.mean()


class EpochArrayMeter(EngineStateListener):
    # With RandomSample, number of batches might change from one epoch to another
    def __init__(self, engine_state, array_size):
        self.engine_state = engine_state
        self.engine_state.add_listener(self)
        self.min = np.zeros((self.engine_state.max_epoch, array_size))
        self.mean = np.zeros((self.engine_state.max_epoch, array_size))
        self.max = np.zeros((self.engine_state.max_epoch, array_size))
        self.batch_size = np.zeros((self.engine_state.max_epoch, array_size))
        self.sum = np.zeros((self.engine_state.max_epoch, array_size))
        self.total = np.zeros((self.engine_state.max_epoch, array_size))
        self.epoch_sum = []
        self.epoch_total = []
        self.is_empty = True

    def update(self, val, total):
        self.is_empty = False

        if torch.is_tensor(val):
            if val.is_cuda:
                val = val.cpu()
            val = val.numpy()
        if torch.is_tensor(total):
            if val.is_cuda:
                val = val.cpu()
            val = val.numpy()

        self.epoch_sum.append(val)
        self.epoch_total.append(total)

    # Listeners
    def epoch_completed(self):
        if not self.is_empty:
            self.epoch_sum = np.array(self.epoch_sum)
            self.epoch_total = np.array(self.epoch_total)
            ratio = self.epoch_sum / self.epoch_total

            self.min[self.engine_state.epoch] = ratio.min(axis=0)
            self.mean[self.engine_state.epoch] = ratio.mean(axis=0)
            self.max[self.engine_state.epoch] = ratio.max(axis=0)
            self.batch_size[self.engine_state.epoch] = self.epoch_sum.shape[0]
            self.sum[self.engine_state.epoch] = self.epoch_sum.sum(axis=0)
            self.total[self.engine_state.epoch] = self.epoch_total.sum(axis=0)

            self.epoch_sum = []
            self.epoch_total = []

    # Utils
    def epoch_ratio(self, epoch):
        return self.mean[epoch]

    def total_ratio(self):
        return self.mean.mean(axis=0)


class TimeMeter(AverageMeter):
    """Computes and stores the average time and current time value.
    """

    def __init__(self, name):
        super().__init__()
        self.name = name
        self.tic = None

    def _format_time(self, time):
        return str(datetime.timedelta(milliseconds=round(time)))

    def total_time(self):
        return self._format_time(self.sum)

    def average_time(self):
        return self._format_time(self.avg)

    def start(self):
        self.tic = self._current_time_ms()

    def stop(self):
        if self.tic is not None:
            self.update(self._current_time_ms() - self.tic)
            self.tic = None
            return self.val
        else:
            warnings.warn("{0}.start() should be called before {0}.stop()".format(self.__class__.__name__, RuntimeWarning))
            return 0

    @staticmethod
    def _current_time_ms():
        return time.time() * 1000


class TorchTimeMeter(TimeMeter):
    """Computes and stores the average time and current time value.
    """

    def __init__(self, name, plot=True):
        super().__init__(name)
        self.start_event = None
        self.end_event = None
        self.cuda = torch.cuda.is_available()
        self.plot = plot

    def start(self):
        if self.cuda:
            self._start_cuda()
        else:
            super().start()

    def _start_cuda(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.start_event.record()

    def stop(self):
        if self.cuda:
            return self._stop_cuda()
        else:
            return super().stop()

    def _stop_cuda(self):
        if self.start_event is not None:
            self.end_event.record()
            torch.cuda.synchronize()  # TODO Check if slows down computation
            self.update(self.start_event.elapsed_time(self.end_event))
            self.start_event = None
            self.end_event = None
            return self.val
        else:
            warnings.warn("{0}.start() should be called before {0}.stop()".format(self.__class__.__name__),
                          RuntimeWarning)
            return 0


class EpochMetricsMeter(object):
    """A collection of metrics.

    Source: https://github.com/KaiyangZhou/Dassl.pytorch

    Examples::
        >>> # 1. Create an instance of MetricMeter
        >>> metric = EpochMetricsMeter()
        >>> # 2. Update using a dictionary as input
        >>> input_dict = {'loss_1': value_1, 'loss_2': value_2}
        >>> metric.update(input_dict)
        >>> # 3. Convert to string and print
        >>> print(str(metric.summary(epoch)))
    """

    def __init__(self, engine_state, delimiter='\t'):
        self.engine_state = engine_state
        self.meters = {}
        self.delimiter = delimiter

    def update(self, input_dict):
        if input_dict is None:
            return

        if not isinstance(input_dict, dict):
            raise TypeError(
                'Input to MetricMeter.update() must be a dictionary'
            )

        for k, v in input_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            if k not in self.meters.keys():
                self.meters[k] = EpochMeter(self.engine_state)
            self.meters[k].update(v)

    def summary(self, epoch):
        output_str = []
        for name, meter in self.meters.items():
            output_str.append(
                '{} {:.3f} [{:.2f}, {:.2f}]'.format(name, meter.mean[epoch], meter.min[epoch], meter.max[epoch])
            )
        return self.delimiter.join(output_str)

class LossEpochMetricsMeter(object):
    def __init__(self, engine_state, delimiter='\t'):
        self.engine_state = engine_state
        self.meters = OrderedDict()
        self.delimiter = delimiter

    def update(self, input_dict):
        if input_dict is None:
            return

        if not isinstance(input_dict, dict):
            raise TypeError(
                'Input to MetricMeter.update() must be a dictionary'
            )

        for k1, v1 in input_dict.items():
            if k1 not in self.meters.keys():
                self.meters[k1] = OrderedDict()
            for k2, v2 in v1.items():
                if isinstance(v2, torch.Tensor):
                    v2 = v2.item()
                if k2 not in self.meters[k1].keys():
                    self.meters[k1][k2] = EpochMeter(self.engine_state)
                self.meters[k1][k2].update(v2)

    def summary(self, epoch):
        final_str = ""
        for name, dict in self.meters.items():
            if dict:
                output_str = ["\n\t" + name + ": "]
                for key, meter in dict.items():
                    output_str.append(
                        '{} {:.3f} [{:.2f}, {:.2f}]'.format(key, meter.mean[epoch], meter.min[epoch], meter.max[epoch])
                    )
                final_str += self.delimiter.join(output_str)
        return final_str


class MetricMeter(object):
    """A collection of metrics.

    Source: https://github.com/KaiyangZhou/Dassl.pytorch

    Examples::
        >>> # 1. Create an instance of MetricMeter
        >>> metric = MetricMeter()
        >>> # 2. Update using a dictionary as input
        >>> input_dict = {'loss_1': value_1, 'loss_2': value_2}
        >>> metric.update(input_dict)
        >>> # 3. Convert to string and print
        >>> print(str(metric))
    """

    def __init__(self, delimiter='\t'):
        self.meters = defaultdict(AverageMeter)
        self.delimiter = delimiter

    def update(self, input_dict):
        if input_dict is None:
            return

        if not isinstance(input_dict, dict):
            raise TypeError(
                'Input to MetricMeter.update() must be a dictionary'
            )

        for k, v in input_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.meters[k].update(v)

    def __str__(self):
        output_str = []
        for name, meter in self.meters.items():
            output_str.append(
                '{} {:.4f} ({:.4f})'.format(name, meter.val, meter.avg)
            )
        return self.delimiter.join(output_str)
