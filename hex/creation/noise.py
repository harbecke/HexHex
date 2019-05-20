import torch
from torch.distributions.pareto import Pareto


def singh_maddala_onto_output(output_tensor, noise_alpha, noise_beta, noise_lambda, device):
    '''
    one value of the output_tensor gets multplied with (1 + a sampled value of singh_maddala)
    https://en.wikipedia.org/wiki/Burr_distribution
    alpha=k, beta=c
    '''
    batch_size = output_tensor.shape[0]
    output_tensor[torch.arange(0, batch_size).long(), torch.randint(torch.numel(output_tensor[0]), 
        (batch_size,))] += torch.log(1 + noise_lambda*(Pareto(1, noise_alpha).sample((batch_size,)).to(device)-1)**(1/noise_beta))
    return output_tensor
