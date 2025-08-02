import os
import math
import torch
import librosa

import numpy as np
import soundfile as sf
import torch.nn.functional as F

from torch import nn, einsum
from functools import partial
from einops import rearrange, repeat, pack, unpack
from torch.nn.utils.parametrizations import weight_norm

os.environ["LRU_CACHE_CAPACITY"] = "3"

def exists(val):
    """Checks if a value exists (is not None)."""
    return val is not None

def default(value, d):
    """Returns value if it exists, otherwise returns default."""
    return value if exists(value) else d

def max_neg_value(tensor):
    """Returns the maximum negative value for a given tensor's dtype."""
    return -torch.finfo(tensor.dtype).max

def l2norm(tensor):
    """Performs L2 normalization on a tensor."""
    return F.normalize(tensor, dim = -1).type(tensor.dtype)

def pad_to_multiple(tensor, multiple, dim=-1, value=0):
    """Pads a tensor to a multiple of a given length along a dimension."""
    seqlen = tensor.shape[dim]
    m = seqlen / multiple
    if m.is_integer(): return False, tensor
    return True, F.pad(tensor, (*((0,) * (-1 - dim) * 2), 0, (math.ceil(m) * multiple - seqlen)), value = value)

def look_around(x, backward = 1, forward = 0, pad_value = -1, dim = 2):
    """Pads a tensor to allow 'looking around' for local attention."""
    t = x.shape[1]
    dims = (len(x.shape) - dim) * (0, 0)
    padded_x = F.pad(x, (*dims, backward, forward), value = pad_value)
    return torch.cat([padded_x[:, ind:(ind + t), ...] for ind in range(forward + backward + 1)], dim = dim)

def rotate_half(x):
    """Rotates half of the input tensor for complex numbers."""
    x1, x2 = rearrange(x, \'b ... (r d) -> b ... r d\', r = 2).unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)

def apply_rotary_pos_emb(q, k, freqs, scale = 1):
    """Applies rotary positional embeddings to queries and keys."""
    q_len = q.shape[-2]
    q_freqs = freqs[..., -q_len:, :]
    inv_scale = scale ** -1
    if scale.ndim == 2: scale = scale[-q_len:, :]
    q = (q * q_freqs.cos() * scale) + (rotate_half(q) * q_freqs.sin() * scale)
    k = (k * freqs.cos() * inv_scale) + (rotate_half(k) * freqs.sin() * inv_scale)
    return q, k

class LocalAttention(nn.Module):
    """Local Attention module."""
    def __init__(self, window_size, causal = False, look_backward = 1, look_forward = None, dropout = 0., shared_qk = False, rel_pos_emb_config = None, dim = None, autopad = False, exact_windowsize = False, scale = None, use_rotary_pos_emb = True, use_xpos = False, xpos_scale_base = None):
        super().__init__()
        look_forward = default(look_forward, 0 if causal else 1)
        assert not (causal and look_forward > 0)
        self.scale = scale
        self.window_size = window_size
        self.autopad = autopad
        self.exact_windowsize = exact_windowsize
        self.causal = causal
        self.look_backward = look_backward
        self.look_forward = look_forward
        self.dropout = nn.Dropout(dropout)
        self.shared_qk = shared_qk
        self.rel_pos = None
        self.use_xpos = use_xpos

        if use_rotary_pos_emb and (exists(rel_pos_emb_config) or exists(dim)): 
            if exists(rel_pos_emb_config): dim = rel_pos_emb_config[0]
            self.rel_pos = SinusoidalEmbeddings(dim, use_xpos = use_xpos, scale_base = default(xpos_scale_base, window_size // 2))

    def forward(self, q, k, v, mask = None, input_mask = None, attn_bias = None, window_size = None):
        mask = default(mask, input_mask)
        assert not (exists(window_size) and not self.use_xpos)

        _, autopad, pad_value, window_size, causal, look_backward, look_forward, shared_qk = q.shape, self.autopad, -1, default(window_size, self.window_size), self.causal, self.look_backward, self.look_forward, self.shared_qk
        (q, packed_shape), (k, _), (v, _) = map(lambda t: pack([t], \'* n d\'), (q, k, v))

        if autopad:
            orig_seq_len = q.shape[1]
            (_, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, self.window_size, dim = -2), (q, k, v))

        b, n, dim_head, device, dtype = *q.shape, q.device, q.dtype
        scale = default(self.scale, dim_head ** -0.5)
        assert (n % window_size) == 0
        windows = n // window_size
        if shared_qk: k = l2norm(k)

        seq = torch.arange(n, device = device)
        b_t = rearrange(seq, \'(w n) -> 1 w n\', w = windows, n = window_size)
        bq, bk, bv = map(lambda t: rearrange(t, \'b (w n) d -> b w n d\', w = windows), (q, k, v))
        bq = bq * scale
        look_around_kwargs = dict(backward =  look_backward, forward =  look_forward, pad_value = pad_value)
        bk = look_around(bk, **look_around_kwargs)
        bv = look_around(bv, **look_around_kwargs)

        if exists(self.rel_pos):
            pos_emb, xpos_scale = self.rel_pos(bk)
            bq, bk = apply_rotary_pos_emb(bq, bk, pos_emb, scale = xpos_scale)

        bq_t = b_t
        bq_k = look_around(b_t, **look_around_kwargs)
        bq_t = rearrange(bq_t, \'... i -> ... i 1\')
        bq_k = rearrange(bq_k, \'... j -> ... 1 j\')
        pad_mask = bq_k == pad_value
        sim = einsum(\'b h i e, b h j e -> b h i j\', bq, bk)

        if exists(attn_bias):
            heads = attn_bias.shape[0]
            assert (b % heads) == 0
            attn_bias = repeat(attn_bias, \'h i j -> (b h) 1 i j\', b = b // heads)
            sim = sim + attn_bias

        mask_value = max_neg_value(sim)

        if shared_qk:
            self_mask = bq_t == bq_k
            sim = sim.masked_fill(self_mask, -5e4)
            del self_mask

        if causal:
            causal_mask = bq_t < bq_k
            if self.exact_windowsize: causal_mask = causal_mask | (bq_t > (bq_k + (self.window_size * self.look_backward)))
            sim = sim.masked_fill(causal_mask, mask_value)
            del causal_mask

        sim = sim.masked_fill(((bq_k - (self.window_size * self.look_forward)) > bq_t) | (bq_t > (bq_k + (self.window_size * self.look_backward))) | pad_mask, mask_value) if not causal and self.exact_windowsize else sim.masked_fill(pad_mask, mask_value)

        if exists(mask):
            batch = mask.shape[0]
            assert (b % batch) == 0
            h = b // mask.shape[0]
            if autopad: _, mask = pad_to_multiple(mask, window_size, dim = -1, value = False)
            mask = repeat(rearrange(look_around(rearrange(mask, \'... (w n) -> (...) w n\', w = windows, n = window_size), **{**look_around_kwargs, \'pad_value\': False}), \'... j -> ... 1 j\'), \'b ... -> (b h) ...\', h = h)
            sim = sim.masked_fill(~mask, mask_value)
            del mask

        out = rearrange(einsum(\'b h i j, b h j e -> b h i e\', self.dropout(sim.softmax(dim = -1)), bv), \'b w n d -> b (w n) d\')
        if autopad: out = out[:, :orig_seq_len, :]
        out, *_ = unpack(out, packed_shape, \'* n d\')
        return out
    
class SinusoidalEmbeddings(nn.Module):
    """Sinusoidal positional embeddings."""
    def __init__(self, dim, scale_base = None, use_xpos = False, theta = 10000):
        super().__init__()
        inv_freq = 1. / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer(\'inv_freq\', inv_freq)
        self.use_xpos = use_xpos
        self.scale_base = scale_base
        assert not (use_xpos and not exists(scale_base))
        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.register_buffer(\'scale\', scale, persistent = False)

    def forward(self, x):
        seq_len, device = x.shape[-2], x.device
        t = torch.arange(seq_len, device = x.device).type_as(self.inv_freq)
        freqs = torch.einsum(\'i , j -> i j\', t, self.inv_freq)
        freqs =  torch.cat((freqs, freqs), dim = -1)

        if not self.use_xpos: return freqs, torch.ones(1, device = device)

        power = (t - (seq_len // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, \'n -> n 1\')
        return freqs, torch.cat((scale, scale), dim = -1)

def load_wav_to_torch(full_path, target_sr=None, return_empty_on_exception=False):
    """Loads a WAV file into a PyTorch tensor."""
    try:
        data, sample_rate = sf.read(full_path, always_2d=True)
    except Exception as e:
        print(f"{full_path}: {e}")

        if return_empty_on_exception: return [], sample_rate or target_sr or 48000
        else: raise

    data = data[:, 0] if len(data.shape) > 1 else data
    assert len(data) > 2

    max_mag = (-np.iinfo(data.dtype).min if np.issubdtype(data.dtype, np.integer) else max(np.amax(data), -np.amin(data)))
    data = torch.FloatTensor(data.astype(np.float32)) / ((2**31) + 1 if max_mag > (2**15) else ((2**15) + 1 if max_mag > 1.01 else 1.0))

    if (torch.isinf(data) | torch.isnan(data)).any() and return_empty_on_exception: return [], sample_rate or target_sr or 48000

    if target_sr is not None and sample_rate != target_sr:
        data = torch.from_numpy(librosa.core.resample(data.numpy(), orig_sr=sample_rate, target_sr=target_sr))
        sample_rate = target_sr

    return data, sample_rate

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """Applies dynamic range compression."""
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)

def dynamic_range_decompression(x, C=1):
    """Applies dynamic range decompression."""
    return np.exp(x) / C

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """Applies dynamic range compression to a PyTorch tensor."""
    return torch.log(torch.clamp(x, min=clip_val) * C)

def dynamic_range_decompression_torch(x, C=1):
    """Applies dynamic range decompression to a PyTorch tensor."""
    return torch.exp(x) / C

class STFT:
    """Short-Time Fourier Transform (STFT) class."""
    def __init__(self, sr=22050, n_mels=80, n_fft=1024, win_size=1024, hop_length=256, fmin=20, fmax=11025, clip_val=1e-5):
        self.target_sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_size = win_size
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.clip_val = clip_val
        self.mel_basis = {}
        self.hann_window = {}

    def get_mel(self, y, keyshift=0, speed=1, center=False, train=False):
        n_fft = self.n_fft
        win_size = self.win_size
        hop_length = self.hop_length
        fmax = self.fmax
        factor = 2 ** (keyshift / 12)
        win_size_new = int(np.round(win_size * factor))
        hop_length_new = int(np.round(hop_length * speed))
        mel_basis = self.mel_basis if not train else {}
        hann_window = self.hann_window if not train else {}
        mel_basis_key = str(fmax) + "_" + str(y.device)

        if mel_basis_key not in mel_basis:
            from librosa.filters import mel as librosa_mel_fn
            mel_basis[mel_basis_key] = torch.from_numpy(librosa_mel_fn(sr=self.target_sr, n_fft=n_fft, n_mels=self.n_mels, fmin=self.fmin, fmax=fmax)).float().to(y.device)

        keyshift_key = str(keyshift) + "_" + str(y.device)
        if keyshift_key not in hann_window: hann_window[keyshift_key] = torch.hann_window(win_size_new).to(y.device)

        pad_left = (win_size_new - hop_length_new) // 2
        pad_right = max((win_size_new - hop_length_new + 1) // 2, win_size_new - y.size(-1) - pad_left)
        spec = torch.stft(torch.nn.functional.pad(y.unsqueeze(1), (pad_left, pad_right), mode="reflect" if pad_right < y.size(-1) else "constant").squeeze(1), int(np.round(n_fft * factor)), hop_length=hop_length_new, win_length=win_size_new, window=hann_window[keyshift_key], center=center, pad_mode="reflect", normalized=False, onesided=True, return_complex=True)
        spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + (1e-9))

        if keyshift != 0:
            size = n_fft // 2 + 1
            resize = spec.size(1)
            spec = (F.pad(spec, (0, 0, 0, size - resize)) if resize < size else spec[:, :size, :]) * win_size / win_size_new

        return dynamic_range_compression_torch(torch.matmul(mel_basis[mel_basis_key], spec), clip_val=self.clip_val)

    def __call__(self, audiopath):
        audio, _ = load_wav_to_torch(audiopath, target_sr=self.target_sr)
        return self.get_mel(audio.unsqueeze(0)).squeeze(0)

stft = STFT()

def softmax_kernel(data, *, projection_matrix, is_query, normalize_data=True, eps=1e-4, device=None):
    """Computes the softmax kernel for attention."""
    b, h, *_ = data.shape
    
    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.0
    ratio = projection_matrix.shape[0] ** -0.5

    data_dash = torch.einsum("...id,...jd->...ij", (data_normalizer * data), repeat(projection_matrix, "j d -> b h j d", b=b, h=h).type_as(data))
    diag_data = ((torch.sum(data**2, dim=-1) / 2.0) * (data_normalizer**2)).unsqueeze(dim=-1)

    return (ratio * (torch.exp(data_dash - diag_data - torch.max(data_dash, dim=-1, keepdim=True).values) + eps) if is_query else ratio * (torch.exp(data_dash - diag_data + eps))).type_as(data)

def orthogonal_matrix_chunk(cols, qr_uniform_q=False, device=None):
    """Generates an orthogonal matrix chunk."""
    unstructured_block = torch.randn((cols, cols), device=device)

    q, r = torch.linalg.qr(unstructured_block.cpu(), mode="reduced")
    q, r = map(lambda t: t.to(device), (q, r))

    if qr_uniform_q:
        d = torch.diag(r, 0)
        q *= d.sign()

    return q.t()

def empty(tensor):
    """Checks if a tensor is empty."""
    return tensor.numel() == 0

def cast_tuple(val):
    """Casts a value to a tuple if it's not already one."""
    return (val,) if not isinstance(val, tuple) else val

class PCmer(nn.Module):
    """PCmer module."""
    def __init__(self, num_layers, num_heads, dim_model, dim_keys, dim_values, residual_dropout, attention_dropout):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim_model = dim_model
        self.dim_values = dim_values
        self.dim_keys = dim_keys
        self.residual_dropout = residual_dropout
        self.attention_dropout = attention_dropout
        self._layers = nn.ModuleList([_EncoderLayer(self) for _ in range(num_layers)])

    def forward(self, phone, mask=None):
        for layer in self._layers:
            phone = layer(phone, mask)

        return phone

class _EncoderLayer(nn.Module):
    """Encoder layer for PCmer."""
    def __init__(self, parent: PCmer):
        super().__init__()
        self.conformer = ConformerConvModule(parent.dim_model)
        self.norm = nn.LayerNorm(parent.dim_model)
        self.dropout = nn.Dropout(parent.residual_dropout)
        self.attn = SelfAttention(dim=parent.dim_model, heads=parent.num_heads, causal=False)

    def forward(self, phone, mask=None):
        phone = phone + (self.attn(self.norm(phone), mask=mask))
        return phone + (self.conformer(phone))

def calc_same_padding(kernel_size):
    """Calculates padding for 'same' convolution."""
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)

class Swish(nn.Module):
    """Swish activation function."""
    def forward(self, x):
        return x * x.sigmoid()

class Transpose(nn.Module):
    """Transposes a tensor."""
    def __init__(self, dims):
        super().__init__()
        assert len(dims) == 2, "dims == 2"

        self.dims = dims

    def forward(self, x):
        return x.transpose(*self.dims)

class GLU(nn.Module):
    """Gated Linear Unit (GLU) activation function."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()

class DepthWiseConv1d(nn.Module):
    """Depthwise 1D convolution."""
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups=chan_in)

    def forward(self, x):
        return self.conv(F.pad(x, self.padding))

class ConformerConvModule(nn.Module):
    """Conformer convolutional module."""
    def __init__(self, dim, causal=False, expansion_factor=2, kernel_size=31, dropout=0.0):
        super().__init__()
        inner_dim = dim * expansion_factor
        self.net = nn.Sequential(nn.LayerNorm(dim), Transpose((1, 2)), nn.Conv1d(dim, inner_dim * 2, 1), GLU(dim=1), DepthWiseConv1d(inner_dim, inner_dim, kernel_size=kernel_size, padding=(calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0))), Swish(), nn.Conv1d(inner_dim, dim, 1), Transpose((1, 2)), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)

def linear_attention(q, k, v):
    """Performs linear attention."""
    return torch.einsum("...ed,...nd->...ne", k, q) if v is None else torch.einsum("...de,...nd,...n->...ne", torch.einsum("...nd,...ne->...de", k, v), q, 1.0 / (torch.einsum("...nd,...d->...n", q, k.sum(dim=-2).type_as(q)) + 1e-8))

def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling=0, qr_uniform_q=False, device=None):
    """Generates a Gaussian orthogonal random matrix."""
    nb_full_blocks = int(nb_rows / nb_columns)
    block_list = []

    for _ in range(nb_full_blocks):
        block_list.append(orthogonal_matrix_chunk(nb_columns, qr_uniform_q=qr_uniform_q, device=device))

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0: block_list.append(orthogonal_matrix_chunk(nb_columns, qr_uniform_q=qr_uniform_q, device=device)[:remaining_rows])

    if scaling == 0: multiplier = torch.randn((nb_rows, nb_columns), device=device).norm(dim=1)
    elif scaling == 1: multiplier = math.sqrt((float(nb_columns))) * torch.ones((nb_rows,), device=device)
    else: raise ValueError(f"{scaling} != 0, 1")

    return torch.diag(multiplier) @ torch.cat(block_list)

class FastAttention(nn.Module):
    """Fast Attention module."""
    def __init__(self, dim_heads, nb_features = 256, ortho_scaling = 0, causal = False, generalized_attention = False, kernel_fn = nn.ReLU(), no_projection = False):
        super().__init__()
        nb_features = nb_features
        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling
        self.causal = causal
        self.generalized_attention = generalized_attention
        self.kernel_fn = kernel_fn
        self.no_projection = no_projection

        if not no_projection:
            self.create_projection = partial(gaussian_orthogonal_random_matrix, nb_rows = nb_features, nb_columns = dim_heads, scaling = ortho_scaling)
            projection_matrix = self.create_projection()
            self.register_buffer(\'projection_matrix\', projection_matrix)

    def forward(self, q, k, v):
        device = q.device

        if self.no_projection:
            q = self.kernel_fn(q).relu()
            k = self.kernel_fn(k).relu()
        else:
            create_projection = partial(self.create_projection, device = device)
            projection_matrix = default(self.projection_matrix, create_projection())
            q = softmax_kernel(q, projection_matrix=projection_matrix, is_query=True, device=device)
            k = softmax_kernel(k, projection_matrix=projection_matrix, is_query=False, device=device)

        if self.causal:
            # TODO: Implement causal attention
            out = linear_attention(q, k, v)
        else:
            out = linear_attention(q, k, v)

        return out

class SelfAttention(nn.Module):
    """Self-attention module."""
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., causal=False, generalized_attention=False, kernel_fn=nn.ReLU(), no_projection=False):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)
        self.fast_attention = FastAttention(dim_head, causal=causal, generalized_attention=generalized_attention, kernel_fn=kernel_fn, no_projection=no_projection)

    def forward(self, x, mask=None):
        h = self.heads
        q = self.to_q(x)
        k, v = self.to_kv(x).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, \'b n (h d) -> b h n d\', h=h), (q, k, v))

        out = self.fast_attention(q, k, v)
        out = rearrange(out, \'b h n d -> b n (h d)\')
        return self.to_out(out)

class FCPE(nn.Module):
    """FCPE pitch estimation model."""
    def __init__(self, model_path, device=None, providers=None, onnx=False):
        super().__init__()
        self.onnx = onnx

        if self.onnx:
            import onnxruntime as ort

            sess_options = ort.SessionOptions()
            sess_options.log_severity_level = 3

            self.model = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
        else:
            self.model = PCmer(num_layers=4, num_heads=4, dim_model=256, dim_keys=256, dim_values=256, residual_dropout=0.1, attention_dropout=0.1)
            self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
            self.model.eval()
            self.model = self.model.to(device)

        self.device = device

    def extract_f0(self, audio, sr, hop_length):
        audio = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
        mel = stft.get_mel(audio, train=False)

        if self.onnx:
            f0 = self.model.run(None, {"mel": mel.cpu().numpy()})[0]
        else:
            f0 = self.model(mel).squeeze(0).cpu().numpy()

        return f0, None


if __name__ == "__main__":
    # Example usage (replace with actual values)
    # model_path = "path/to/your/fcpe_model.pth" # or .onnx
    # audio_path = "path/to/your/audio.wav"
    # sr = 16000
    # hop_length = 160

    # fcpe_model = FCPE(model_path, device="cpu", onnx=False)
    # audio, _ = load_wav_to_torch(audio_path, target_sr=sr)
    # f0, _ = fcpe_model.extract_f0(audio.numpy(), sr, hop_length)
    # print(f"Extracted F0 shape: {f0.shape}")
    print("This is a module, not meant to be run directly. Use the functions within your own scripts.")


