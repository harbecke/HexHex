import torch
from torch.nn import _reduction as _Reduction

@torch._jit_internal.weak_script
def lq_loss(input, target, q, reduction='mean'):
    # type: (Tensor, Tensor, Optional[bool], Optional[bool], str) -> Tensor
    r"""lq_loss(input, target, q, reduction='mean') -> Tensor

    https://papers.nips.cc/paper/8094-generalized-cross-entropy-loss-for-training-deep-neural-networks-with-noisy-labels.pdf
    """

    ret = (1-(1-torch.abs(input-target))**q)/q
    if reduction != 'none':
        ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)

    return ret