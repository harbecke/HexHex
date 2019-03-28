import torch
from torch._jit_internal import weak_module, weak_script_method
from torch.nn.modules.loss import _Loss

@weak_module
class LQLoss(_Loss):
    # type: (Tensor, Tensor, Optional[bool], Optional[bool], str) -> Tensor
    r"""lq_loss(input, target, q, reduction='mean') -> Tensor

    https://papers.nips.cc/paper/8094-generalized-cross-entropy-loss-for-training-deep-neural-networks-with-noisy-labels.pdf
    """
    def __init__(self, q, reduction='mean'):
        super(LQLoss, self).__init__(size_average=None, reduce=None, reduction=reduction)
        self.q = q

    @weak_script_method
    def forward(self, input, target):
        ret = (1-(1+1e-8-torch.abs(input-target))**self.q)/self.q
        if self.reduction != 'none':
            ret = torch.mean(ret) if self.reduction == 'mean' else torch.sum(ret)

        return ret

