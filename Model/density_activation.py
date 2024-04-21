import torch

class TruncatedExponential(torch.autograd.Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, min=-15, max=15))


trunc_exp = TruncatedExponential.apply


def init_density_activation(activation_type: str):
    # Implementation from k-planes
    # https://github.com/sarafridov/K-Planes/blob/main/plenoxels/ops/activations.py
    if activation_type == 'trunc_exp':
        return lambda x: trunc_exp(x - 1)
    elif activation_type == 'relu':
        return torch.nn.functional.relu
    else:
        raise ValueError(activation_type)