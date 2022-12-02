from __future__ import division, absolute_import
import torch
import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    r"""Cross entropy loss with label smoothing regularizer.
    
    Reference:
        Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.

    With label smoothing, the label :math:`y` for a class is computed by
    
    .. math::
        \begin{equation}
        (1 - \eps) \times y + \frac{\eps}{K},
        \end{equation}

    where :math:`K` denotes the number of classes and :math:`\eps` is a weight. When
    :math:`\eps = 0`, the loss function reduces to the normal cross entropy.
    
    Args:
        num_classes (int): number of classes.
        eps (float, optional): weight. Default is 0.1.
        use_gpu (bool, optional): whether to use gpu devices. Default is True.
        label_smooth (bool, optional): whether to apply label smoothing. Default is True.
    """

    def __init__(self, eps=0.1, label_smooth=True):
        super(CrossEntropyLoss, self).__init__()
        self.eps = eps if label_smooth else 0
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets, weights=None):
        """
        Args:
            inputs (torch.Tensor): prediction matrix (before softmax) with
                shape (batch_size, num_classes).
            targets (torch.LongTensor): ground truth labels with shape (batch_size).
                Each position contains the label index.
        """
        assert inputs.shape[0] == targets.shape[0]
        num_classes = inputs.shape[1]
        log_probs = self.logsoftmax(inputs)
        zeros = torch.zeros(log_probs.size())
        targets = zeros.scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if inputs.is_cuda:
            targets = targets.cuda()
        targets = (1 - self.eps) * targets + self.eps / num_classes
        if weights is not None:
            result = (-targets * log_probs).sum(dim=1)
            result = result * nn.functional.normalize(weights, p=1, dim=0)
            result = result.sum()
        else:
            result = (-targets * log_probs).mean(0).sum()
        return result
