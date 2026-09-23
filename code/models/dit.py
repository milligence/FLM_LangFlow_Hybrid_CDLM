import math
import typing

import einops
import flash_attn
import flash_attn.layers.rotary
import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.nn.attention import SDPBackend, sdpa_kernel


env_val = os.getenv('DIT_USE_COMPILE', '0').lower()
USE_COMPILE = env_val in ['1', 'true', 'yes', 'on']
print(f"DIT: USE_COMPILE={USE_COMPILE}")

if USE_COMPILE:
    torch_compile_deco = torch.compile(model=None, mode=None, dynamic=False, options={"max_autotune": True, "triton.cudagraphs": False})
    jit_deco = lambda x: x
else:
    torch_compile_deco = lambda x: x
    jit_deco = lambda x: x
    # jit_deco = torch.jit.script

# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

def bias_dropout_add_scale(
        x: torch.Tensor,
        bias: typing.Optional[torch.Tensor],
        scale: torch.Tensor,
        residual: typing.Optional[torch.Tensor],
        prob: float,
        training: bool) -> torch.Tensor:
    if bias is not None:
        out = scale * F.dropout(x + bias, p=prob, training=training)
    else:
        out = scale * F.dropout(x, p=prob, training=training)

    if residual is not None:
        out = residual + out
    return out


def get_bias_dropout_add_scale(training):
    def _bias_dropout_add(x, bias, scale, residual, prob):
        return bias_dropout_add_scale(
            x, bias, scale, residual, prob, training)

    return _bias_dropout_add


# function overload
def modulate(x: torch.Tensor,
             shift: torch.Tensor,
             scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


def linear_with_jvp(layer, x, x_jvp):
    """Apply a linear layer and its exact input-direction JVP."""
    return layer(x), F.linear(x_jvp, layer.weight, None)


def silu_with_jvp(x, x_jvp):
    output = F.silu(x)
    sigmoid = torch.sigmoid(x)
    derivative = sigmoid * (1.0 + x * (1.0 - sigmoid))
    return output, x_jvp * derivative


def gelu_tanh_with_jvp(x, x_jvp):
    """Exact derivative of PyTorch's tanh-approximate GELU."""
    output = F.gelu(x, approximate='tanh')
    coefficient = math.sqrt(2.0 / math.pi)
    inner = coefficient * (x + 0.044715 * x.pow(3))
    tanh_inner = torch.tanh(inner)
    derivative = (
        0.5 * (1.0 + tanh_inner)
        + 0.5 * x * (1.0 - tanh_inner.square())
        * coefficient * (1.0 + 3.0 * 0.044715 * x.square()))
    return output, x_jvp * derivative


def rms_norm_with_jvp(layer, x, x_jvp):
    """Apply ``nn.RMSNorm`` and propagate one exact input tangent."""
    output = layer(x)
    eps = layer.eps
    if eps is None:
        eps = torch.finfo(x.dtype).eps
    with torch.amp.autocast(device_type=x.device.type, enabled=False):
        x_float = x.float()
        tangent_float = x_jvp.float()
        inverse_rms = torch.rsqrt(
            x_float.square().mean(dim=-1, keepdim=True) + eps)
        normalized_jvp = (
            tangent_float * inverse_rms
            - x_float * inverse_rms.pow(3)
            * (x_float * tangent_float).mean(dim=-1, keepdim=True))
        tangent = normalized_jvp * layer.weight.float()
    return output, tangent.to(output.dtype)


def dropout_with_jvp(x, x_jvp, probability, training):
    if not training or probability == 0.0:
        return x, x_jvp
    output, mask = torch.ops.aten.native_dropout(
        x, float(probability), True)
    tangent = x_jvp * mask.to(x_jvp.dtype) / (1.0 - probability)
    return output, tangent


def scaled_residual_with_jvp(
        x, x_jvp, scale, scale_jvp, residual, residual_jvp,
        probability, training):
    dropped, dropped_jvp = dropout_with_jvp(
        x, x_jvp, probability, training)
    output = residual + scale * dropped
    tangent = (
        residual_jvp + scale * dropped_jvp + scale_jvp * dropped)
    return output, tangent


@jit_deco
def bias_dropout_add_scale_fused_train(
        x: torch.Tensor,
        bias: typing.Optional[torch.Tensor],
        scale: torch.Tensor,
        residual: typing.Optional[torch.Tensor],
        prob: float) -> torch.Tensor:
    return bias_dropout_add_scale(
        x, bias, scale, residual, prob, True)


@jit_deco
def bias_dropout_add_scale_fused_inference(
        x: torch.Tensor,
        bias: typing.Optional[torch.Tensor],
        scale: torch.Tensor,
        residual: typing.Optional[torch.Tensor],
        prob: float) -> torch.Tensor:
    return bias_dropout_add_scale(
        x, bias, scale, residual, prob, False)


@jit_deco
def modulate_fused(x: torch.Tensor,
                   shift: torch.Tensor,
                   scale: torch.Tensor) -> torch.Tensor:
    return modulate(x, shift, scale)


class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        # Lightning validation runs under torch.inference_mode().  A rotary
        # cache created there is an inference tensor and cannot later
        # participate in the first training backward pass.  Rebuild that
        # cache when grad-enabled training resumes.
        cached_inference_tensor = (
            self.cos_cached is not None and self.cos_cached.is_inference())
        if (seq_len != self.seq_len_cached
                or (torch.is_grad_enabled() and cached_inference_tensor)):
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim],
                             device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            # dims are: batch, seq_len, qkv, head, dim
            self.cos_cached = emb.cos(
            )[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            self.sin_cached = emb.sin(
            )[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            # This makes the transformation on v an identity.
            self.cos_cached[:, :, 2, :, :].fill_(1.)
            self.sin_cached[:, :, 2, :, :].fill_(0.)

        return self.cos_cached, self.sin_cached


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def split_and_apply_rotary_pos_emb(qkv, rotary_cos_sin,):
    with torch.amp.autocast(device_type=qkv.device.type, enabled=False):
        cos, sin = rotary_cos_sin
        cos = cos.to(qkv.dtype)
        sin = sin.to(qkv.dtype)
        cos = cos[0, :, 0, 0, :cos.shape[-1]//2]
        sin = sin[0, :, 0, 0, :sin.shape[-1]//2]
        q, k, v = qkv.chunk(3, dim=2)
        q = flash_attn.layers.rotary.apply_rotary_emb_torch(
            q.squeeze(dim=2), cos, sin)
        k = flash_attn.layers.rotary.apply_rotary_emb_torch(
            k.squeeze(dim=2), cos, sin)
        v = v.squeeze(dim=2)
    return q, k, v


def apply_rotary_pos_emb(qkv, cos, sin, use_flash=True):
    cos = cos[0, :, 0, 0, :cos.shape[-1]//2]
    sin = sin[0, :, 0, 0, :sin.shape[-1]//2]

    # flash-attn 2.8.x implements rotary embeddings through
    # torch.library.wrap_triton, which is unavailable in torch 2.5. Keep the
    # fused attention kernel enabled while falling back to the equivalent
    # PyTorch rotary implementation on older supported torch versions.
    can_use_flash_rotary = (
        use_flash
        and hasattr(torch.library, 'wrap_triton')
    )
    if can_use_flash_rotary:
        return flash_attn.layers.rotary.apply_rotary_emb_qkv_(qkv, cos, sin)
    else:
        q, k, v = qkv.unbind(dim=2)
        def apply_rotary(x, cos, sin):
            
            cos = cos.unsqueeze(0).unsqueeze(2)  
            sin = sin.unsqueeze(0).unsqueeze(2)      
            cos = torch.cat([cos, cos], dim=-1)  
            sin = torch.cat([sin, sin], dim=-1)  
            
            return x * cos + rotate_half(x) * sin
        
        q_rotated = apply_rotary(q, cos, sin)
        k_rotated = apply_rotary(k, cos, sin)
        
        return torch.stack([q_rotated, k_rotated, v], dim=2)



def regular_attention_multi_headed(q, k, v, tq=None, tk=None, tv=None):

    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        attention_output = F.scaled_dot_product_attention(
            query=q.transpose(1, 2),
            key=k.transpose(1, 2),
            value=v.transpose(1, 2),
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False)
    # [batch_size, seq_len, num_heads, head_dim]
    attention_output = attention_output.transpose(1, 2)
    return einops.rearrange(attention_output, 'b s h d -> b s (h d)')

class LearnableLossWeighting(nn.Module):
    def __init__(self, cond_dim, is_flow=True, hidden_dim=128):
        super().__init__()
        
        self.s_embed = TimestepEmbedder(cond_dim)
        if not is_flow:
            self.t_embed = TimestepEmbedder(cond_dim)
        else:
            self.t_embed = None
        
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Initialize the last layer to zero so that initially e^-w = 1
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, s, t=None):
        emb = self.s_embed(s)
        if t is not None and self.t_embed is not None:
            emb_t = self.t_embed(t)
            emb = emb + emb_t
        return self.mlp(emb).squeeze(-1)
#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x):
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]

    def forward_with_jvp(self, x, x_jvp):
        output = self.forward(x)
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x_float = x.float()
            tangent_float = x_jvp.float()
            centered = x_float - x_float.mean(dim=-1, keepdim=True)
            inverse_std = torch.rsqrt(
                centered.square().mean(dim=-1, keepdim=True) + 1e-5)
            normalized = centered * inverse_std
            normalized_jvp = (
                tangent_float
                - tangent_float.mean(dim=-1, keepdim=True)
                - normalized
                * (normalized * tangent_float).mean(dim=-1, keepdim=True)
            ) * inverse_std
            tangent = normalized_jvp * self.weight[None, None, :]
        return output, tangent


def residual_linear(x, W, x_skip, residual_scale):
    """x_skip + residual_scale * W @ x"""
    dim_out, dim_in = W.shape[0], W.shape[1]
    return torch.addmm(
        x_skip.view(-1, dim_out),
        x.view(-1, dim_in),
        W.T,
        alpha=residual_scale).view(*x.shape[:-1], dim_out)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True))
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            - math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding,
                 torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

    def forward_with_jvp(self, t, t_jvp):
        half = self.frequency_embedding_size // 2
        frequencies = torch.exp(
            -math.log(10000)
            * torch.arange(
                start=0, end=half, dtype=torch.float32, device=t.device)
            / half)
        arguments = t[:, None].float() * frequencies[None]
        argument_jvp = t_jvp[:, None].float() * frequencies[None]
        embedding = torch.cat(
            [torch.cos(arguments), torch.sin(arguments)], dim=-1)
        embedding_jvp = torch.cat([
            -torch.sin(arguments) * argument_jvp,
            torch.cos(arguments) * argument_jvp,
        ], dim=-1)
        if self.frequency_embedding_size % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            embedding_jvp = torch.cat(
                [embedding_jvp, torch.zeros_like(embedding_jvp[:, :1])],
                dim=-1)
        hidden, hidden_jvp = linear_with_jvp(
            self.mlp[0], embedding, embedding_jvp)
        hidden, hidden_jvp = silu_with_jvp(hidden, hidden_jvp)
        return linear_with_jvp(self.mlp[2], hidden, hidden_jvp)


class FiniteTimeConditioner(nn.Module):
    """Contracted G(r, eta) branch with an exact input-direction JVP."""

    def __init__(self, condition_width):
        super().__init__()
        self.input = nn.Linear(2, condition_width)
        self.output = nn.Linear(condition_width, condition_width)
        self.output.weight.data.zero_()
        self.output.bias.data.zero_()

    def forward(self, features):
        return self.output(F.silu(self.input(features.float())))

    def forward_with_jvp(self, features, features_jvp):
        hidden, hidden_jvp = linear_with_jvp(
            self.input, features.float(), features_jvp.float())
        hidden, hidden_jvp = silu_with_jvp(hidden, hidden_jvp)
        return linear_with_jvp(self.output, hidden, hidden_jvp)

class SquaredReLU(nn.Module):
    """
    Squared ReLU activation function: f(x) = max(0, x)^2
    """
    def forward(self, x):
        return torch.pow(torch.relu(x), 2)
    
class TimestepEmbedderSquaredReLU(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            SquaredReLU(),
            nn.Linear(hidden_size, hidden_size, bias=True))
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            - math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding,
                 torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class PosteriorTVMTimeConditioner(nn.Module):
    """Smooth two-time conditioner used only by from-scratch posterior TVM."""

    def __init__(self, cond_dim):
        super().__init__()
        self.source_r = TimestepEmbedder(cond_dim)
        self.source_gamma = TimestepEmbedder(cond_dim)
        self.gap_eta = TimestepEmbedder(cond_dim)
        self.gap_dt = TimestepEmbedder(cond_dim)
        self.gap_dgamma = TimestepEmbedder(cond_dim)
        self.source_mlp = nn.Sequential(
            nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.gap_mlp = nn.Sequential(
            nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.output_norm = nn.RMSNorm(cond_dim)

    def forward(self, features):
        if features.ndim != 2 or features.shape[-1] != 5:
            raise ValueError(
                'Posterior-TVM time features must have shape [B, 5], got '
                f'{tuple(features.shape)}.')
        r, gamma_r, eta, dt, delta_gamma = features.float().unbind(dim=-1)
        source = self.source_mlp(
            self.source_r(r) + self.source_gamma(gamma_r))
        gap = self.gap_mlp(
            self.gap_eta(eta)
            + self.gap_dt(dt)
            + self.gap_dgamma(delta_gamma))
        return self.output_norm(source + eta[:, None] * gap)

    def forward_with_jvp(self, features, features_jvp):
        if features.ndim != 2 or features.shape[-1] != 5:
            raise ValueError(
                'Posterior-TVM time features must have shape [B, 5], got '
                f'{tuple(features.shape)}.')
        if features_jvp.shape != features.shape:
            raise ValueError(
                'Posterior-TVM feature tangent must match the primal shape, '
                f'got {tuple(features_jvp.shape)} and {tuple(features.shape)}.')
        primal = features.float().unbind(dim=-1)
        tangent = features_jvp.float().unbind(dim=-1)
        r, gamma_r, eta, dt, delta_gamma = primal
        r_jvp, gamma_r_jvp, eta_jvp, dt_jvp, delta_gamma_jvp = tangent

        source_r, source_r_jvp = self.source_r.forward_with_jvp(r, r_jvp)
        source_gamma, source_gamma_jvp = self.source_gamma.forward_with_jvp(
            gamma_r, gamma_r_jvp)
        source, source_jvp = silu_with_jvp(
            source_r + source_gamma, source_r_jvp + source_gamma_jvp)
        source, source_jvp = linear_with_jvp(
            self.source_mlp[1], source, source_jvp)

        gap_eta, gap_eta_jvp = self.gap_eta.forward_with_jvp(eta, eta_jvp)
        gap_dt, gap_dt_jvp = self.gap_dt.forward_with_jvp(dt, dt_jvp)
        gap_gamma, gap_gamma_jvp = self.gap_dgamma.forward_with_jvp(
            delta_gamma, delta_gamma_jvp)
        gap, gap_jvp = silu_with_jvp(
            gap_eta + gap_dt + gap_gamma,
            gap_eta_jvp + gap_dt_jvp + gap_gamma_jvp)
        gap, gap_jvp = linear_with_jvp(
            self.gap_mlp[1], gap, gap_jvp)

        combined = source + eta[:, None] * gap
        combined_jvp = (
            source_jvp + eta_jvp[:, None] * gap + eta[:, None] * gap_jvp)
        return rms_norm_with_jvp(
            self.output_norm, combined, combined_jvp)


class FiniteMapSCProjector(nn.Module):
    """Zero-initialized auxiliary cache projection for finite map queries."""

    def __init__(self, hidden_size, cond_dim, projection_hidden):
        super().__init__()
        self.previous_eta = TimestepEmbedder(cond_dim)
        self.projection = nn.Sequential(
            nn.Linear(hidden_size + cond_dim + 1, projection_hidden),
            nn.SiLU(),
            nn.Linear(projection_hidden, hidden_size))
        self.projection[-1].weight.data.zero_()
        self.projection[-1].bias.data.zero_()

    def forward(self, cache, previous_eta, valid):
        if cache.ndim != 3:
            raise ValueError('Finite-map SC cache must have shape [B, L, D].')
        previous = self.previous_eta(previous_eta.float())
        previous = previous[:, None, :].expand(
            cache.shape[0], cache.shape[1], previous.shape[-1])
        valid_feature = valid.float()[:, None, None].expand(
            cache.shape[0], cache.shape[1], 1)
        return self.projection(torch.cat(
            [cache.float(), previous.float(), valid_feature], dim=-1))


class LabelEmbedder(nn.Module):
    """Embeds class labels into vector representations.

    Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, cond_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, cond_size)
        self.num_classes = num_classes

        # TODO think of initializing with 0.02 std deviation like in original DiT paper

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core Model                                    #
#################################################################################

class DDiTBlockCausal(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)
        
        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate='tanh'),
            nn.Linear(mlp_ratio * dim, dim, bias=True))
        self.dropout2 = nn.Dropout(dropout)
        self.dropout = dropout

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference

    def forward(self, x, rotary_cos_sin, **kwargs):
        del kwargs
        batch_size, seq_len = x.shape[0], x.shape[1]

        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        # attention operation
        x_skip = x
        x = self.norm1(x)

        qkv = self.attn_qkv(x)
        qkv = einops.rearrange(
            qkv,
            'b s (three h d) -> b s three h d',
            three=3,
            h=self.n_heads)
        with torch.amp.autocast(device_type=qkv.device.type, enabled=False):
            cos, sin = rotary_cos_sin
            qkv = apply_rotary_pos_emb(
                qkv, cos.to(qkv.dtype), sin.to(qkv.dtype)
            )
        qkv = einops.rearrange(qkv, 'b s ... -> (b s) ...')
        cu_seqlens = torch.arange(
            0, (batch_size + 1) * seq_len,
            step=seq_len, dtype=torch.int32, device=qkv.device)
        x = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
            qkv, cu_seqlens, seq_len, 0.0, causal=True)

        x = einops.rearrange(x, '(b s) h d -> b s (h d)',
                             b=batch_size)

        scale = torch.ones(1, device=x.device, dtype=x.dtype)
        x = bias_dropout_scale_fn(
            self.attn_out(x), None, scale, x_skip, self.dropout)

        # mlp operation
        x = bias_dropout_scale_fn(
            self.mlp(self.norm2(x)), None, scale, x, self.dropout)
        return x


class DDiTBlock(nn.Module):
    def __init__(self, dim, n_heads, adaLN,
                 cond_dim=None, mlp_ratio=4,
                 dropout=0.1, qk_norm=False,
                 attention_jvp_backend='reference'):
        super().__init__()
        self.n_heads = n_heads
        self.adaLN = adaLN
        self.softcap=50
        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        head_dim = dim // n_heads
        self.q_norm = nn.RMSNorm(head_dim) if qk_norm else None
        self.k_norm = nn.RMSNorm(head_dim) if qk_norm else None
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate='tanh'),
            nn.Linear(mlp_ratio * dim, dim, bias=True))
        self.dropout2 = nn.Dropout(dropout)
        self.dropout = dropout
        self.attention_jvp_backend = str(attention_jvp_backend)
        if self.attention_jvp_backend not in {'reference', 'bmm'}:
            raise ValueError(
                'attention_jvp_backend must be reference or bmm, got '
                f'{self.attention_jvp_backend!r}.')

        if self.adaLN:
            self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim)
            self.adaLN_modulation.weight.data.zero_()
            self.adaLN_modulation.bias.data.zero_()

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference
    
    def custom_sdpa(self, q, k, v, softcap=-1.0):
        B, H, S, D = q.shape
        q = q / (D ** 0.5)
        attn_weights = torch.einsum('bhid,bhjd->bhij', q, k)  # (B, H, S, S)
        if softcap > 0.0:
            attn_weights = softcap * torch.tanh(attn_weights / softcap)
        attn_probs = torch.softmax(attn_weights, dim=-1)  # F.softmax
        output = torch.einsum('bhij,bhjd->bhid', attn_probs, v)  # (B, H, S, D)
        return output

    def bmm_sdpa(self, q, k, v, softcap=-1.0):
        """Exact softcap attention expressed as batched matrix multiplies."""
        batch, heads, sequence, head_dim = q.shape
        q_flat = q.reshape(batch * heads, sequence, head_dim)
        k_flat = k.reshape(batch * heads, sequence, head_dim)
        v_flat = v.reshape(batch * heads, sequence, head_dim)
        scores = torch.bmm(
            q_flat / (head_dim ** 0.5), k_flat.transpose(1, 2))
        if softcap > 0.0:
            scores = softcap * torch.tanh(scores / softcap)
        probabilities = torch.softmax(scores, dim=-1)
        output = torch.bmm(probabilities, v_flat)
        return output.reshape(batch, heads, sequence, head_dim)

    def bmm_sdpa_with_jvp(
            self, q, q_jvp, k, k_jvp, v, v_jvp, softcap=-1.0):
        """Exact softcap attention and its analytic directional derivative."""
        batch, heads, sequence, head_dim = q.shape
        flat_shape = (batch * heads, sequence, head_dim)
        q_flat = q.reshape(flat_shape)
        q_jvp_flat = q_jvp.reshape(flat_shape)
        k_flat = k.reshape(flat_shape)
        k_jvp_flat = k_jvp.reshape(flat_shape)
        v_flat = v.reshape(flat_shape)
        v_jvp_flat = v_jvp.reshape(flat_shape)
        inverse_scale = head_dim ** -0.5
        scores = torch.bmm(
            q_flat * inverse_scale, k_flat.transpose(1, 2))
        scores_jvp = (
            torch.bmm(
                q_jvp_flat * inverse_scale, k_flat.transpose(1, 2))
            + torch.bmm(
                q_flat * inverse_scale, k_jvp_flat.transpose(1, 2)))
        if softcap > 0.0:
            tanh_scores = torch.tanh(scores / softcap)
            scores = softcap * tanh_scores
            scores_jvp = scores_jvp * (1.0 - tanh_scores.square())
        probabilities = torch.softmax(scores, dim=-1)
        probabilities_jvp = probabilities * (
            scores_jvp
            - (probabilities * scores_jvp).sum(dim=-1, keepdim=True))
        output = torch.bmm(probabilities, v_flat)
        output_jvp = (
            torch.bmm(probabilities_jvp, v_flat)
            + torch.bmm(probabilities, v_jvp_flat))
        output_shape = (batch, heads, sequence, head_dim)
        return output.reshape(output_shape), output_jvp.reshape(output_shape)

    def forward_with_jvp(
            self, x, x_jvp, rotary_cos_sin=None, c=None, c_jvp=None):
        """Block forward that explicitly propagates one input tangent."""
        x_skip, x_skip_jvp = x, x_jvp
        x, x_jvp = self.norm1.forward_with_jvp(x, x_jvp)

        if self.adaLN:
            modulation, modulation_jvp = linear_with_jvp(
                self.adaLN_modulation, c, c_jvp)
            primal_chunks = modulation[:, None].chunk(6, dim=2)
            tangent_chunks = modulation_jvp[:, None].chunk(6, dim=2)
            (shift_msa, scale_msa, gate_msa, shift_mlp,
             scale_mlp, gate_mlp) = primal_chunks
            (shift_msa_jvp, scale_msa_jvp, gate_msa_jvp, shift_mlp_jvp,
             scale_mlp_jvp, gate_mlp_jvp) = tangent_chunks
            normalized = x
            x = modulate_fused(normalized, shift_msa, scale_msa)
            x_jvp = (
                x_jvp * (1.0 + scale_msa)
                + normalized * scale_msa_jvp + shift_msa_jvp)

        qkv, qkv_jvp = linear_with_jvp(self.attn_qkv, x, x_jvp)
        qkv = einops.rearrange(
            qkv, 'b s (three h d) -> b s three h d',
            three=3, h=self.n_heads)
        qkv_jvp = einops.rearrange(
            qkv_jvp, 'b s (three h d) -> b s three h d',
            three=3, h=self.n_heads)
        if self.q_norm is not None:
            q, k, v = qkv.unbind(dim=2)
            q_jvp, k_jvp, v_jvp = qkv_jvp.unbind(dim=2)
            q, q_jvp = rms_norm_with_jvp(
                self.q_norm, q, q_jvp)
            k, k_jvp = rms_norm_with_jvp(
                self.k_norm, k, k_jvp)
            qkv = torch.stack((q, k, v), dim=2)
            qkv_jvp = torch.stack((q_jvp, k_jvp, v_jvp), dim=2)
        with torch.amp.autocast(device_type=qkv.device.type, enabled=False):
            cos, sin = rotary_cos_sin
            qkv = apply_rotary_pos_emb(
                qkv, cos.to(qkv.dtype), sin.to(qkv.dtype), use_flash=False)
            qkv_jvp = apply_rotary_pos_emb(
                qkv_jvp, cos.to(qkv_jvp.dtype), sin.to(qkv_jvp.dtype),
                use_flash=False)

        q, k, v = qkv.unbind(dim=2)
        q_jvp, k_jvp, v_jvp = qkv_jvp.unbind(dim=2)
        x, x_jvp = self.bmm_sdpa_with_jvp(
            q.transpose(1, 2), q_jvp.transpose(1, 2),
            k.transpose(1, 2), k_jvp.transpose(1, 2),
            v.transpose(1, 2), v_jvp.transpose(1, 2),
            softcap=self.softcap)
        x = x.transpose(1, 2)
        x_jvp = x_jvp.transpose(1, 2)
        x = einops.rearrange(x, 'b s h d -> b s (h d)')
        x_jvp = einops.rearrange(x_jvp, 'b s h d -> b s (h d)')

        attention, attention_jvp = linear_with_jvp(
            self.attn_out, x, x_jvp)
        if self.adaLN:
            x, x_jvp = scaled_residual_with_jvp(
                attention, attention_jvp, gate_msa, gate_msa_jvp,
                x_skip, x_skip_jvp, self.dropout, self.training)
            normalized, normalized_jvp = self.norm2.forward_with_jvp(
                x, x_jvp)
            modulated = modulate_fused(
                normalized, shift_mlp, scale_mlp)
            modulated_jvp = (
                normalized_jvp * (1.0 + scale_mlp)
                + normalized * scale_mlp_jvp + shift_mlp_jvp)
            hidden, hidden_jvp = linear_with_jvp(
                self.mlp[0], modulated, modulated_jvp)
            hidden, hidden_jvp = gelu_tanh_with_jvp(hidden, hidden_jvp)
            hidden, hidden_jvp = linear_with_jvp(
                self.mlp[2], hidden, hidden_jvp)
            return scaled_residual_with_jvp(
                hidden, hidden_jvp, gate_mlp, gate_mlp_jvp,
                x, x_jvp, self.dropout, self.training)

        unit_scale = torch.ones(1, device=x.device, dtype=x.dtype)
        zero_scale_jvp = torch.zeros_like(unit_scale)
        x, x_jvp = scaled_residual_with_jvp(
            attention, attention_jvp, unit_scale, zero_scale_jvp,
            x_skip, x_skip_jvp, self.dropout, self.training)
        normalized, normalized_jvp = self.norm2.forward_with_jvp(x, x_jvp)
        hidden, hidden_jvp = linear_with_jvp(
            self.mlp[0], normalized, normalized_jvp)
        hidden, hidden_jvp = gelu_tanh_with_jvp(hidden, hidden_jvp)
        hidden, hidden_jvp = linear_with_jvp(
            self.mlp[2], hidden, hidden_jvp)
        return scaled_residual_with_jvp(
            hidden, hidden_jvp, unit_scale, zero_scale_jvp,
            x, x_jvp, self.dropout, self.training)
    
    def forward(self, x, rotary_cos_sin=None, c=None, seqlens=None, exclude_last_token=False, use_jvp_attn=False):

        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        x_skip = x
        x = self.norm1(x)

        if self.adaLN:
            (shift_msa, scale_msa, gate_msa, shift_mlp,
             scale_mlp, gate_mlp) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
            x = modulate_fused(x, shift_msa, scale_msa)
        
        qkv = self.attn_qkv(x)
        qkv = einops.rearrange(
            qkv,
            'b s (three h d) -> b s three h d',
            three=3,
            h=self.n_heads)
        if self.q_norm is not None:
            q, k, v = qkv.unbind(dim=2)
            qkv = torch.stack((self.q_norm(q), self.k_norm(k), v), dim=2)
        with torch.cuda.amp.autocast(enabled=False):
            cos, sin = rotary_cos_sin
            qkv = apply_rotary_pos_emb(
                qkv, cos.to(qkv.dtype), sin.to(qkv.dtype), use_flash= not use_jvp_attn
            )
        
        if use_jvp_attn: #custom attention for JVP support
            q, k, v = qkv.unbind(dim=2) 
            q = q.transpose(1, 2)  
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            if self.attention_jvp_backend == 'reference':
                x = self.custom_sdpa(q, k, v, softcap=self.softcap)
            else:
                x = self.bmm_sdpa(q, k, v, softcap=self.softcap)
            x = x.transpose(1, 2)
        else:
            x = flash_attn.flash_attn_qkvpacked_func(
                qkv, 0.0, causal=False,
                softcap=self.softcap,
                )

        x = einops.rearrange(x, 'b s h d -> b s (h d)',)
        

        if self.adaLN:
            x = bias_dropout_scale_fn(self.attn_out(x),
                                      None,
                                      gate_msa,
                                      x_skip,
                                      self.dropout)
            x = bias_dropout_scale_fn(
                self.mlp(modulate_fused(
                    self.norm2(x), shift_mlp, scale_mlp)),
                None, gate_mlp, x, self.dropout)
        else:
            scale = torch.ones(1, device=x.device, dtype=x.dtype)
            x = bias_dropout_scale_fn(
                self.attn_out(x), None, scale, x_skip, self.dropout)
            x = bias_dropout_scale_fn(
                self.mlp(self.norm2(x)), None, scale, x, self.dropout)
        return x


class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim, normalize=False):
        super().__init__()
        self.dim = dim
        self.normalize = normalize
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def normalized_weight(self):
        """Return the embedding table used by embedding-space models.

        Normalization is computed on every use so gradients still update the
        underlying unconstrained parameter while the ODE always sees rows with
        norm sqrt(hidden_size).
        """
        if not self.normalize:
            return self.embedding
        weight = F.normalize(self.embedding.float(), dim=-1)
        return (weight * math.sqrt(self.dim)).to(self.embedding.dtype)

    def forward(self, x):
        weight = self.normalized_weight()
        if x.ndim == 2:
            return weight[x]
        assert x.ndim == 3
        return torch.einsum(
            "blv,ve->ble",
            x.float(),
            weight.float()).to(x.dtype)


class DDiTFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, cond_dim,
                 adaLN, bias: bool = True):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels, bias=bias)
        self.linear.weight.data.zero_()
        if self.linear.bias is not None:
            self.linear.bias.data.zero_()
        self.adaLN = adaLN
        if self.adaLN:
            self.adaLN_modulation = nn.Linear(cond_dim,
                                              2 * hidden_size,
                                              bias=True)
            self.adaLN_modulation.weight.data.zero_()
            self.adaLN_modulation.bias.data.zero_()

    def forward(self, x, c, return_features=False):
        x = self.norm_final(x)
        if self.adaLN:
            shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
            x = modulate_fused(x, shift, scale)
        features = x
        output = self.linear(features)
        if return_features:
            return output, features
        return output

    def forward_with_jvp(
            self, x, x_jvp, c, c_jvp, return_features=False):
        x, x_jvp = self.norm_final.forward_with_jvp(x, x_jvp)
        if self.adaLN:
            modulation, modulation_jvp = linear_with_jvp(
                self.adaLN_modulation, c, c_jvp)
            shift, scale = modulation[:, None].chunk(2, dim=2)
            shift_jvp, scale_jvp = modulation_jvp[:, None].chunk(2, dim=2)
            normalized = x
            x = modulate_fused(normalized, shift, scale)
            x_jvp = (
                x_jvp * (1.0 + scale)
                + normalized * scale_jvp + shift_jvp)
        features, features_jvp = x, x_jvp
        output, output_jvp = linear_with_jvp(
            self.linear, features, features_jvp)
        if return_features:
            return (output, features), (output_jvp, features_jvp)
        return output, output_jvp


class DIT(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    def __init__(self, config, vocab_size: int):
        super().__init__()
        if type(config) == dict:
            config = omegaconf.OmegaConf.create(config)
        self.causal = config.algo.causal_attention
        self.adaLN = not self.causal
        self.config = config
        self.vocab_size = vocab_size
        dim = config.model.hidden_size
        cond_dim = config.model.cond_dim
        self.embedding_state = bool(getattr(
            config.algo, 'embedding_state', False))
        self.self_conditioning = bool(getattr(
            config.algo, 'self_conditioning', False))
        self.vocab_embed = EmbeddingLayer(
            dim,
            vocab_size,
            normalize=bool(getattr(
                config.algo, 'normalize_embeddings', False)))
        # Freeze the physical codebook before TrainerBase constructs the EMA.
        # EMA filters parameters by requires_grad, so changing this flag later
        # would shift every subsequent shadow/parameter pair.
        if str(getattr(
                config.algo, 'codebook_gradient_mode', 'all')) == 'frozen':
            self.vocab_embed.embedding.requires_grad_(False)
        # The independent Gaussian-bias matrix is copied from the physical
        # codebook without drawing random numbers. Keeping it on the backbone
        # makes it part of the optimizer, checkpoint, and EMA before
        # TrainerBase constructs any of them. The physical codebook may remain
        # trainable; the two parameters then update independently from the same
        # initialization.
        prototype_mode = str(getattr(
            config.algo, 'classification_prototype_mode', 'shared'))
        if prototype_mode == 'independent':
            self.classification_prototype = nn.Parameter(
                self.vocab_embed.embedding.detach().clone())
        else:
            self.register_parameter('classification_prototype', None)
        if self.self_conditioning:
            self.self_cond_proj = nn.Linear(2 * dim, dim, bias=False)
            self.self_cond_proj.weight.data.zero_()
        else:
            self.self_cond_proj = None
        if not self.causal:
            self.sigma_map = TimestepEmbedder(cond_dim)
        if bool(getattr(self.config.algo, 'double_temb', False)):
            self.sigma_map_prime = TimestepEmbedder(cond_dim)
        else:
            self.sigma_map_prime = None
        if bool(getattr(
                self.config.algo, 'posterior_tvm_time_conditioning', False)):
            self.posterior_tvm_time_conditioner = (
                PosteriorTVMTimeConditioner(cond_dim))
        else:
            self.posterior_tvm_time_conditioner = None
        self.rotary_emb = Rotary(dim // config.model.n_heads)
        
        if getattr(config.algo, 'learnable_loss_weighting', False):
            self.learnable_loss_weighting = LearnableLossWeighting(cond_dim=cond_dim)
        
        blocks = []
        for _ in range(config.model.n_blocks):
            if self.causal:
                block = DDiTBlockCausal(
                    dim=dim,
                    n_heads=config.model.n_heads,
                    dropout=config.model.dropout)
            else:
                block = DDiTBlock(
                    dim=dim,
                    n_heads=config.model.n_heads,
                    cond_dim=cond_dim,
                    adaLN=self.adaLN,
                    dropout=config.model.dropout,
                    qk_norm=bool(getattr(config.model, 'qk_norm', False)),
                    attention_jvp_backend=str(getattr(
                        config.model, 'attention_jvp_backend', 'reference')))
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)
        
        self.output_layer = DDiTFinalLayer(
            hidden_size=dim,
            out_channels=vocab_size,
            cond_dim=cond_dim,
            adaLN=self.adaLN,
)
        if bool(getattr(
                self.config.algo, 'finite_sc_residual_head', False)):
            residual_hidden = int(getattr(
                self.config.algo, 'finite_sc_residual_hidden', 256))
            self.sc_residual_head = nn.Sequential(
                nn.Linear(dim, residual_hidden),
                nn.GELU(),
                nn.Linear(residual_hidden, dim))
            self.sc_residual_head[-1].weight.data.zero_()
            self.sc_residual_head[-1].bias.data.zero_()
        else:
            self.sc_residual_head = None

        # Constructed after all common parameters so A/B common initialization
        # remains identical even though only B owns this extra module.
        if bool(getattr(
                self.config.algo, 'finite_map_only_sc', False)):
            projection_hidden = int(getattr(
                self.config.algo, 'finite_map_sc_hidden', 256))
            self.finite_map_sc_projector = FiniteMapSCProjector(
                dim, cond_dim, projection_hidden)
        else:
            self.finite_map_sc_projector = None

        # These modules are constructed after every legacy module and inside a
        # forked RNG scope.  Adding F/G therefore cannot perturb the shared
        # random initialization of the legacy backbone or the other line.
        if bool(getattr(
                self.config.algo, 'task1_finite_time_conditioning', False)):
            finite_seed = int(getattr(
                self.config.algo, 'task1_finite_init_seed', 0))
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(finite_seed)
                self.finite_time_conditioner = FiniteTimeConditioner(cond_dim)
                if str(getattr(
                        self.config.algo, 'task1_tvm_final_line', '')).upper() == 'F':
                    self.finite_correction_head = nn.Linear(
                        dim, vocab_size, bias=True)
                    self.finite_correction_head.weight.data.zero_()
                    self.finite_correction_head.bias.data.zero_()
                else:
                    self.finite_correction_head = None
        else:
            self.finite_time_conditioner = None
            self.finite_correction_head = None
        
        self.sigma = 1e-5
        self.scale_by_sigma = config.model.scale_by_sigma
        if "is_di4c" in config:
            self.is_di4c = config.is_di4c
        else:
            self.is_di4c = config.is_di4c = False

        if "is_di4c_deterministic" in config:
            self.is_di4c_deterministic = config.is_di4c_deterministic
        else:
            self.is_di4c_deterministic = config.is_di4c_deterministic = False

        if self.is_di4c:
            print("Using Di4C")
            # Added for Di4C:
            self.latent_feature_dim = 128
            self.latent_projection = nn.Sequential(
                nn.Linear(in_features=self.latent_feature_dim,
                          out_features=self.latent_feature_dim*4),
                nn.GELU(),
                nn.Linear(self.latent_feature_dim*4, config.model.hidden_size)
            )

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference
    
    @torch_compile_deco
    def forward(self, x, sigma, sigma_prime=None, use_jvp_attn=False,
                inputs_are_embeddings=False, x_self_cond=None,
                return_hidden=False, posterior_time_features=None,
                finite_time_features=None, return_output_features=False,
                finite_sc_cache=None, finite_sc_previous_eta=None,
                finite_sc_valid=None, finite_sc_eta=None,
                finite_sc_amplitude=1.0):
        if inputs_are_embeddings:
            if x.ndim != 3 or x.shape[-1] != self.config.model.hidden_size:
                raise ValueError(
                    'Embedding-state input must have shape [B, L, hidden_size], '
                    f'got {tuple(x.shape)}.')
        else:
            x = self.vocab_embed(x)

        if self.self_cond_proj is not None:
            if x_self_cond is None:
                x_self_cond = torch.zeros_like(x)
            if x_self_cond.shape != x.shape:
                raise ValueError(
                    'Self-conditioning tensor must match the embedding-state '
                    f'shape, got {tuple(x_self_cond.shape)} and {tuple(x.shape)}.')
            x = self.self_cond_proj(torch.cat([x, x_self_cond], dim=-1)) + x

        if self.finite_map_sc_projector is not None:
            batch_size = x.shape[0]
            if finite_sc_cache is None:
                finite_sc_cache = torch.zeros_like(x)
            if finite_sc_previous_eta is None:
                finite_sc_previous_eta = torch.zeros(
                    batch_size, device=x.device, dtype=torch.float32)
            if finite_sc_valid is None:
                finite_sc_valid = torch.zeros(
                    batch_size, device=x.device, dtype=torch.float32)
            if finite_sc_eta is None:
                raise ValueError(
                    'finite_sc_eta is required for finite-map-only SC.')
            sc_update = self.finite_map_sc_projector(
                finite_sc_cache, finite_sc_previous_eta, finite_sc_valid)
            eta_gate = finite_sc_eta.float().view(-1, 1, 1)
            valid_gate = finite_sc_valid.float().view(-1, 1, 1)
            x = x + (
                float(finite_sc_amplitude)
                * eta_gate * valid_gate * sc_update).to(x.dtype)
            
        if self.causal:
            t_cond = None
        else:
            if finite_time_features is not None:
                if self.finite_time_conditioner is None:
                    raise ValueError(
                        'finite_time_features require the contracted G branch.')
                if finite_time_features.shape != (x.shape[0], 2):
                    raise ValueError('finite_time_features must have shape [B, 2].')
                t_emb = self.sigma_map(sigma)
                legacy = F.silu(t_emb)
                eta = finite_time_features[:, 1:2].float()
                gap = self.finite_time_conditioner(finite_time_features)
                t_cond = legacy + eta * gap
            elif posterior_time_features is not None:
                if self.posterior_tvm_time_conditioner is None:
                    raise ValueError(
                        'posterior_time_features require the dedicated '
                        'posterior-TVM conditioner.')
                t_cond = self.posterior_tvm_time_conditioner(
                    posterior_time_features)
            else:
                t_emb = self.sigma_map(sigma)
                if sigma_prime is not None:
                    if self.sigma_map_prime is not None:
                        t_prime_emb = self.sigma_map_prime(sigma_prime)
                    else:
                        t_prime_emb = self.sigma_map(sigma_prime)
                    t_emb = t_emb + t_prime_emb
                t_cond = F.silu(t_emb)

        rotary_cos_sin = self.rotary_emb(x)
        
        with torch.amp.autocast(device_type=x.device.type, dtype=torch.bfloat16):
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, c=t_cond,
                                    seqlens=None, exclude_last_token=self.is_di4c, 
                                    use_jvp_attn=use_jvp_attn)
            hidden = x
            output = self.output_layer(
                hidden, c=t_cond, return_features=return_output_features)
            if return_output_features:
                x, output_features = output
            else:
                x = output
            
        if return_output_features:
            return x, output_features
        if return_hidden:
            return x, hidden
        return x

    def forward_with_jvp(
            self, x, x_jvp, sigma, sigma_jvp=None, sigma_prime=None,
            sigma_prime_jvp=None, inputs_are_embeddings=False,
            x_self_cond=None, x_self_cond_jvp=None, return_hidden=False,
            posterior_time_features=None,
            posterior_time_features_jvp=None,
            finite_time_features=None, finite_time_features_jvp=None,
            return_output_features=False,
            finite_sc_cache=None, finite_sc_previous_eta=None,
            finite_sc_valid=None, finite_sc_eta=None,
            finite_sc_eta_jvp=None, finite_sc_amplitude=1.0):
        """Forward plus one explicit layerwise JVP, without ``torch.func``."""
        if self.causal:
            raise ValueError('Layer JVP is implemented for non-causal DDiT only.')
        if inputs_are_embeddings:
            if x.ndim != 3 or x.shape[-1] != self.config.model.hidden_size:
                raise ValueError(
                    'Embedding-state input must have shape [B, L, hidden_size], '
                    f'got {tuple(x.shape)}.')
        else:
            x = self.vocab_embed(x)
            x_jvp = torch.zeros_like(x)
        if x_jvp.shape != x.shape:
            raise ValueError('Input tangent must match the DIT input shape.')

        if self.self_cond_proj is not None:
            if x_self_cond is None:
                x_self_cond = torch.zeros_like(x)
            if x_self_cond_jvp is None:
                x_self_cond_jvp = torch.zeros_like(x_self_cond)
            projected, projected_jvp = linear_with_jvp(
                self.self_cond_proj,
                torch.cat([x, x_self_cond], dim=-1),
                torch.cat([x_jvp, x_self_cond_jvp], dim=-1))
            x, x_jvp = x + projected, x_jvp + projected_jvp

        if self.finite_map_sc_projector is not None:
            batch_size = x.shape[0]
            if finite_sc_cache is None:
                finite_sc_cache = torch.zeros_like(x)
            if finite_sc_previous_eta is None:
                finite_sc_previous_eta = torch.zeros(
                    batch_size, device=x.device, dtype=torch.float32)
            if finite_sc_valid is None:
                finite_sc_valid = torch.zeros(
                    batch_size, device=x.device, dtype=torch.float32)
            if finite_sc_eta is None or finite_sc_eta_jvp is None:
                raise ValueError(
                    'finite_sc_eta and its tangent are required for '
                    'finite-map-only SC layer JVP.')
            sc_update = self.finite_map_sc_projector(
                finite_sc_cache, finite_sc_previous_eta, finite_sc_valid)
            valid_gate = finite_sc_valid.float().view(-1, 1, 1)
            eta_gate = finite_sc_eta.float().view(-1, 1, 1)
            eta_gate_jvp = finite_sc_eta_jvp.float().view(-1, 1, 1)
            amplitude = float(finite_sc_amplitude)
            x = x + (amplitude * eta_gate * valid_gate * sc_update).to(x.dtype)
            x_jvp = x_jvp + (
                amplitude * eta_gate_jvp * valid_gate * sc_update
            ).to(x_jvp.dtype)

        if finite_time_features is not None:
            if self.finite_time_conditioner is None:
                raise ValueError(
                    'finite_time_features require the contracted G branch.')
            if finite_time_features_jvp is None:
                raise ValueError(
                    'finite_time_features_jvp is required for layer JVP.')
            if finite_time_features.shape != (x.shape[0], 2):
                raise ValueError('finite_time_features must have shape [B, 2].')
            if sigma_jvp is None:
                sigma_jvp = torch.zeros_like(sigma)
            t_emb, t_emb_jvp = self.sigma_map.forward_with_jvp(
                sigma, sigma_jvp)
            legacy, legacy_jvp = silu_with_jvp(t_emb, t_emb_jvp)
            gap, gap_jvp = self.finite_time_conditioner.forward_with_jvp(
                finite_time_features, finite_time_features_jvp)
            eta = finite_time_features[:, 1:2].float()
            eta_jvp = finite_time_features_jvp[:, 1:2].float()
            t_cond = legacy + eta * gap
            t_cond_jvp = legacy_jvp + eta_jvp * gap + eta * gap_jvp
        elif posterior_time_features is not None:
            if self.posterior_tvm_time_conditioner is None:
                raise ValueError(
                    'posterior_time_features require the dedicated '
                    'posterior-TVM conditioner.')
            if posterior_time_features_jvp is None:
                raise ValueError(
                    'posterior_time_features_jvp is required for layer JVP.')
            t_cond, t_cond_jvp = (
                self.posterior_tvm_time_conditioner.forward_with_jvp(
                    posterior_time_features, posterior_time_features_jvp))
        else:
            if sigma_jvp is None:
                sigma_jvp = torch.zeros_like(sigma)
            t_emb, t_emb_jvp = self.sigma_map.forward_with_jvp(
                sigma, sigma_jvp)
            if sigma_prime is not None:
                if sigma_prime_jvp is None:
                    sigma_prime_jvp = torch.zeros_like(sigma_prime)
                embedder = (
                    self.sigma_map_prime
                    if self.sigma_map_prime is not None else self.sigma_map)
                prime, prime_jvp = embedder.forward_with_jvp(
                    sigma_prime, sigma_prime_jvp)
                t_emb, t_emb_jvp = t_emb + prime, t_emb_jvp + prime_jvp
            t_cond, t_cond_jvp = silu_with_jvp(t_emb, t_emb_jvp)

        rotary_cos_sin = self.rotary_emb(x)
        with torch.amp.autocast(device_type=x.device.type, dtype=torch.bfloat16):
            for block in self.blocks:
                x, x_jvp = block.forward_with_jvp(
                    x, x_jvp, rotary_cos_sin,
                    c=t_cond, c_jvp=t_cond_jvp)
            hidden, hidden_jvp = x, x_jvp
            output, output_jvp = self.output_layer.forward_with_jvp(
                hidden, hidden_jvp, t_cond, t_cond_jvp,
                return_features=return_output_features)
            if return_output_features:
                x, output_features = output
                x_jvp, output_features_jvp = output_jvp
            else:
                x, x_jvp = output, output_jvp
        if return_output_features:
            return (x, output_features), (x_jvp, output_features_jvp)
        if return_hidden:
            return (x, hidden), (x_jvp, hidden_jvp)
        return x, x_jvp


class NoSCDIT(DIT):
    """DDiT backbone whose public interface and state exclude SC entirely."""

    def __init__(self, config, vocab_size: int):
        if bool(getattr(config.algo, 'self_conditioning', False)):
            raise ValueError('NoSCDIT requires self_conditioning=false.')
        super().__init__(config, vocab_size)
        if self.self_cond_proj is not None:
            raise RuntimeError('NoSCDIT unexpectedly constructed an SC module.')

    def forward(self, x, sigma, sigma_prime=None, use_jvp_attn=False,
                inputs_are_embeddings=False, return_hidden=False,
                posterior_time_features=None):
        return super().forward(
            x, sigma, sigma_prime=sigma_prime,
            use_jvp_attn=use_jvp_attn,
            inputs_are_embeddings=inputs_are_embeddings,
            x_self_cond=None,
            return_hidden=return_hidden,
            posterior_time_features=posterior_time_features,
            finite_sc_cache=None,
            finite_sc_previous_eta=None,
            finite_sc_valid=None,
            finite_sc_eta=None,
            finite_sc_amplitude=1.0)

# From https://github.com/yang-song/score_sde_pytorch/ which is from
#  https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py


def transformer_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
    half_dim = embedding_dim // 2
    # magic number 10000 is from transformers
    emb = math.log(max_positions) / (half_dim - 1)
    # emb = math.log(2.) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32,
                    device=timesteps.device) * -emb)
    # emb = tf.range(num_embeddings, dtype=jnp.float32)[:, None] * emb[None, :]
    # emb = tf.cast(timesteps, dtype=jnp.float32)[:, None] * emb[None, :]
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb
