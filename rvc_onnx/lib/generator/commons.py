import torch

def init_weights(m, mean=0.0, std=0.01):
    """Initializes weights of a given module."""
    if m.__class__.__name__.find("Conv") != -1: m.weight.data.normal_(mean, std)

def get_padding(kernel_size, dilation=1):
    """Calculates padding for a given kernel size and dilation."""
    return int((kernel_size * dilation - dilation) / 2)

@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    """Performs fused add, tanh, sigmoid, and multiply operations."""
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    return torch.tanh(in_act[:, :n_channels_int, :]) * torch.sigmoid(in_act[:, n_channels_int:, :])

def sequence_mask(length, max_length = None):
    """Creates a sequence mask."""
    if max_length is None: max_length = length.max()
    return torch.arange(max_length, dtype=length.dtype, device=length.device).unsqueeze(0) < length.unsqueeze(1)

def clip_grad_value(parameters, clip_value, norm_type=2):
    """Clips gradients by value."""
    if isinstance(parameters, torch.Tensor): parameters = [parameters]
    norm_type = float(norm_type)

    if clip_value is not None: clip_value = float(clip_value)
    total_norm = 0

    for p in list(filter(lambda p: p.grad is not None, parameters)):
        total_norm += (p.grad.data.norm(norm_type)).item() ** norm_type
        if clip_value is not None: p.grad.data.clamp_(min=-clip_value, max=clip_value)

    return total_norm ** (1.0 / norm_type)


