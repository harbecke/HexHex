import torch
from torch.distributions.pareto import Pareto
from hex.utils.utils import device

def singh_maddala_onto_output(output_tensor, noise_alpha, noise_beta, noise_lambda):
    '''
    one value of the output_tensor gets increased by a sampled value of singh_maddala
    https://en.wikipedia.org/wiki/Burr_distribution
    alpha=k, beta=c
    '''
    batch_size = output_tensor.shape[0]
    output_tensor[torch.arange(0, batch_size).long(), torch.randint(torch.numel(output_tensor[0]), 
        (batch_size,))] += noise_lambda*(Pareto(1, noise_alpha).sample((batch_size,)).to(device)-1)**(1/noise_beta)
    return output_tensor


def uniform_noise_onto_output(output_tensor, noise_p):
    """
    Adds constant to each output value with probability noise_p
    """
    return output_tensor + (torch.rand_like(output_tensor) < noise_p).type(torch.float) * 1000
