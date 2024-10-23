# coding=utf-8
""" PyTorch Yasa model."""

from typing import Any, Optional, Tuple, Union, List

import dataclasses
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from transformers.configuration_utils import PretrainedConfig
import torch.distributed as dist
import os
from flash_attn import flash_attn_varlen_func

logger = logging.get_logger(__name__)

MPU: "MPUClass" = None


@dataclasses.dataclass
class MPUClass:
    world_size: int
    mp_size: int
    rank: int
    mp_group: dist.ProcessGroup
    dp_group: dist.ProcessGroup
    mp_groups: List[dist.ProcessGroup]
    dp_groups: List[dist.ProcessGroup]

    @property
    def dp_rank(self):
        return self.rank // self.mp_size

    @property
    def mp_rank(self):
        return self.rank % self.mp_size

    @property
    def dp_size(self):
        return self.world_size // self.mp_size

    @property
    def first_rank_in_mp_group(self):
        return self.rank // self.mp_size * self.mp_size


def init_distributed(mp_size) -> MPUClass:
    global MPU
    if os.environ.get("WORLD_SIZE", "0") != "0":
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        mp_groups = []
        for i in range(world_size // mp_size):
            group_ranks = [i * mp_size + mp_rank for mp_rank in range(mp_size)]
            group = dist.new_group(group_ranks)
            mp_groups.append(group)
            if rank in group_ranks:
                mp_group = group
        dp_groups = []
        for mp_rank in range(mp_size):
            group_ranks = [j * mp_size + mp_rank for j in range(world_size // mp_size)]
            group = dist.new_group(group_ranks)
            dp_groups.append(group)
            if rank in group_ranks:
                dp_group = group
        MPU = MPUClass(
            world_size=world_size,
            mp_size=mp_size,
            rank=rank,
            mp_group=mp_group,
            mp_groups=mp_groups,
            dp_group=dp_group,
            dp_groups=dp_groups,
        )
    else:
        MPU = MPUClass(world_size=world_size, mp_size=1, rank=0, mp_group=None)
    return MPU

class RMSNorm(torch.nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class YasaConfig(PretrainedConfig):
    r"""Yasa model config.
    ```"""
    model_type = "yasa_model"

    def __init__(
        self,
        vocab_size=50432,
        hidden_size=6144,
        num_hidden_layers=44,
        num_attention_heads=64,
        intermediate_size=24576,
        num_moe_experts=1,
        moe_router_topk=2,
        moe_router_load_balancing_type="sinkhorn",
        hidden_act="gelu",
        rotary_pct=0.25,
        rotary_emb_base=10000,
        max_position_embeddings=2048,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        use_cache=True,
        bos_token_id=0,
        eos_token_id=2,
        tie_word_embeddings=False,
        use_parallel_residual=True,
        add_position_embedding=False,
        mp_size=1,
        unpad=False,
        rotary_seq_len_interpolation_factor=1,
        **kwargs,
    ):
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.num_moe_experts = num_moe_experts
        self.moe_router_topk = moe_router_topk
        self.moe_router_load_balancing_type = moe_router_load_balancing_type
        self.hidden_act = hidden_act
        self.rotary_pct = rotary_pct
        self.rotary_emb_base = rotary_emb_base
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.use_parallel_residual = use_parallel_residual
        self.add_position_embedding = add_position_embedding
        self.mp_size = mp_size
        self.unpad = unpad
        self.rotary_seq_len_interpolation_factor = rotary_seq_len_interpolation_factor


class YasaConfigSmall(PretrainedConfig):
    r"""Yasa config for small model.
    ```"""
    model_type = "yasa_tiny"

    def __init__(
        self,
        vocab_size=50432,
        hidden_size=1024,
        num_hidden_layers=4,
        num_attention_heads=64,
        intermediate_size=1024,
        hidden_act="gelu",
        rotary_pct=0.25,
        rotary_emb_base=10000,
        max_position_embeddings=2048,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        use_cache=True,
        bos_token_id=0,
        eos_token_id=2,
        tie_word_embeddings=False,
        use_parallel_residual=True,
        add_position_embedding=False,
        mp_size=1,
        rotary_seq_len_interpolation_factor=1,
        **kwargs,
    ):
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.rotary_pct = rotary_pct
        self.rotary_emb_base = rotary_emb_base
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.use_parallel_residual = use_parallel_residual
        self.add_position_embedding = add_position_embedding
        self.mp_size = mp_size
        self.rotary_seq_len_interpolation_factor = rotary_seq_len_interpolation_factor



class YasaConfigLarge(PretrainedConfig):
    r"""Yasa config for large model.
    ```"""
    model_type = "yasa_large"

    def __init__(
        self,
        vocab_size=50432,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        intermediate_size=1024,
        hidden_act="gelu",
        rotary_pct=0.25,
        rotary_emb_base=10000,
        max_position_embeddings=2048,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        use_cache=True,
        bos_token_id=0,
        eos_token_id=2,
        tie_word_embeddings=False,
        use_parallel_residual=True,
        add_position_embedding=False,
        mp_size=1,
        rotary_seq_len_interpolation_factor=1,
        **kwargs,
    ):
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.rotary_pct = rotary_pct
        self.rotary_emb_base = rotary_emb_base
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.use_parallel_residual = use_parallel_residual
        self.add_position_embedding = add_position_embedding
        self.mp_size = mp_size
        self.rotary_seq_len_interpolation_factor = rotary_seq_len_interpolation_factor


class SwiGluActivation(nn.Module):
    def forward(self, x):
        x = torch.chunk(x, 2, dim=-1)
        return nn.functional.silu(x[0]) * x[1]


ACT2FN["swiglu"] = SwiGluActivation


class YasaPreTrainedModel(PreTrainedModel):
    """An abstract class to handle weights initialization and a simple interface
    for downloading and loading pretrained models.
    """

    config_class = YasaConfig
    base_model_prefix = "gpt_neox"
    supports_gradient_checkpointing = True
    _no_split_modules = ["YasaLayer"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, YasaModel):
            module.gradient_checkpointing = value


class ForwardAllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Because we are saving one of the inputs use `save_for_backward`
        # Save non-tensors and non-inputs/non-outputs directly on ctx
        if MPU is not None and MPU.mp_size > 1:
            dist.all_reduce(x, group=MPU.mp_group)
        return x

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out


class BackwardAllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Because we are saving one of the inputs use `save_for_backward`
        # Save non-tensors and non-inputs/non-outputs directly on ctx
        return x

    @staticmethod
    def backward(ctx, grad_out):
        if MPU is not None and MPU.mp_size > 1:
            temp = torch.empty_like(grad_out, dtype=torch.float32)
            temp.copy_(grad_out)
            dist.all_reduce(temp, group=MPU.mp_group)
            return temp.to(dtype=grad_out.dtype)
        return grad_out


class GPTNeoXAttention(nn.Module):
    def __init__(self, config: YasaConfig):
        super().__init__()
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.num_attention_heads_per_device = (
            config.num_attention_heads // config.mp_size
        )

        self.hidden_size = config.hidden_size
        self.head_size = config.hidden_size // config.num_attention_heads
        self.rotary_ndims = int(self.head_size * config.rotary_pct)
        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(
                torch.ones((max_positions, max_positions), dtype=torch.bool)
            ).view(1, 1, max_positions, max_positions),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e9))
        self.rotary_emb = RotaryEmbedding(
            self.rotary_ndims,
            config.max_position_embeddings,
            base=config.rotary_emb_base,
            interpolation=config.rotary_seq_len_interpolation_factor
        )
        self.norm_factor = torch.sqrt(
            torch.tensor(self.head_size, dtype=torch.float32)
        ).to(torch.get_default_dtype())

        self.num_query_groups = getattr(config, 'num_query_groups', self.num_attention_heads)
        self.num_query_groups_per_device = self.num_query_groups // config.mp_size
        kv_size = self.num_query_groups_per_device * self.head_size

        self.query_key_value = nn.Linear(
            config.hidden_size, 2 * kv_size + self.num_attention_heads_per_device * self.head_size
        )
        self.dense = nn.Linear(
            self.num_attention_heads_per_device * self.head_size, config.hidden_size
        )

    def _forward_no_unpad(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        position_ids: torch.LongTensor,
        head_mask: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ):
        has_layer_past = layer_past is not None

        # Compute QKV
        # Attention heads [batch, seq_len, hidden_size]
        #   --> [batch, seq_len, (np * 3 * head_size)]
        qkv = self.query_key_value(hidden_states)

        # [batch, seq_len, (num_heads + 2 * num_groups) * head_size)]
        #   --> [batch, seq_len, num_groups, (num_heads/num_groups + 2) * head_size]
        new_qkv_shape = qkv.size()[:-1] + (
            self.num_query_groups_per_device,
            (self.num_attention_heads_per_device // self.num_query_groups_per_device + 2) * self.head_size,
        )
        qkv = qkv.view(*new_qkv_shape)

        # [batch, seq_len, num_groups, (num_heads/num_groups + 2) * head_size] -->
        # 3 [batch, nun_groups or num_heads, seq_len, head_size]
        q_off = self.num_attention_heads_per_device // self.num_query_groups_per_device * self.head_size
        query = qkv[..., : q_off].reshape(qkv.size(0), qkv.size(1), -1, self.head_size).permute(0, 2, 1, 3)
        key = qkv[..., q_off : q_off + self.head_size].permute(0, 2, 1, 3)
        value = qkv[..., q_off + self.head_size :].permute(0, 2, 1, 3)

        # Compute rotary embeddings on rotary_ndims
        query_rot = query[..., : self.rotary_ndims]
        query_pass = query[..., self.rotary_ndims :]
        key_rot = key[..., : self.rotary_ndims]
        key_pass = key[..., self.rotary_ndims :]

        # Compute token offset for rotary embeddings (when decoding)
        seq_len = key.shape[-2]
        if has_layer_past:
            seq_len += layer_past[0].shape[-2]
        cos, sin = self.rotary_emb(value, seq_len=seq_len)
        query, key = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)
        query = torch.cat((query, query_pass), dim=-1)
        key = torch.cat((key, key_pass), dim=-1)

        # Cache QKV values
        if has_layer_past:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        present = (key, value) if use_cache else None

        query = query.reshape(query.size(0) * key.size(1), -1, query.size(2), query.size(3))
        key = key.reshape(key.size(0) * key.size(1), 1, key.size(2), key.size(3))
        value = value.reshape(value.size(0) * value.size(1), 1, value.size(2), value.size(3))

        # Options no longer support in favor of memory efficiency
        if head_mask:
            raise RuntimeError("head mask is no longer supported.")
        if output_attentions:
            raise RuntimeError("output_attentions is no longer supported.")

        # Compute attention
        if attention_mask is not None:
            # merge causal mask and padding mask
            query_length = query.size(-2)
            key_length = key.size(-2)
            causal_mask = self.bias[
                :, :, key_length - query_length : key_length, :key_length
            ] * attention_mask
            causal_mask = causal_mask.to(query.dtype)
            causal_mask = (1.0 - causal_mask) * -10000.0
            causal_mask = causal_mask.expand(-1, self.num_query_groups_per_device, -1, -1)
            causal_mask = causal_mask.reshape(causal_mask.size(0) * causal_mask.size(1), 1, causal_mask.size(2), causal_mask.size(3))
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=causal_mask)
        else:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, is_causal=True)

        # Reshape outputs
        attn_output = attn_output.view(qkv.size(0), -1, attn_output.size(2), attn_output.size(3))
        attn_output = self._merge_heads(
            attn_output, self.num_attention_heads_per_device, self.head_size
        )
        attn_output = self.dense(attn_output)

        outputs = (attn_output, present)

        return outputs

    def _forward_unpad(
        self,
        hidden_states,
        cu_seqlens,
        max_seqlen,
        cos,
        sin,
    ):
        # Do not support MQA now
        # Compute QKV
        # Attention heads [total, hidden_size]
        qkv = self.query_key_value(hidden_states)

        heads_per_group = self.num_attention_heads_per_device // self.num_query_groups_per_device
        #   --> [total, ng, (np / ng + 2), head_size]
        qkv = qkv.view(
            qkv.shape[0],
            self.num_query_groups_per_device,
            heads_per_group + 2,
            self.head_size)

        # [total, np, head]
        q = qkv[:,:,:heads_per_group,:].reshape(qkv.shape[0], self.num_attention_heads_per_device, self.head_size)
        # [total, ng, head]
        k = qkv[:,:,-2,:]
        v = qkv[:,:,-1,:]

        # no shape change
        q, k = apply_rotary_unpacked(q, k, cos, sin)

        # MQA flash -> [total, np, head]
        attn_output = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            causal=True,
        )

        # output proj
        attn_output = self.dense(attn_output.view(attn_output.shape[0], -1))

        outputs = (attn_output, None)
        return outputs

    def forward(
        self,
        *args,
        **kwargs,
    ):
        if self.config.unpad:
            return self._forward_unpad(*args, **kwargs)
        else:
            return self._forward_no_unpad(*args, **kwargs)

    @classmethod
    def _split_heads(cls, tensor, num_attention_heads, attn_head_size):
        """
        Splits hidden dim into attn_head_size and num_attention_heads
        """
        # tensor: [bs, seq_len, hidden_size]
        new_shape = tensor.size()[:-1] + (num_attention_heads, attn_head_size)
        # -> [bs, seq_len, num_attention_heads, attn_head_size]
        tensor = tensor.view(new_shape)
        # -> [bs, num_attention_heads, seq_len, attn_head_size]
        tensor = tensor.permute(0, 2, 1, 3)
        return tensor

    @classmethod
    def _merge_heads(cls, tensor, num_attention_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden dim
        """
        # tensor [bs, num_attention_heads, seq_len, attn_head_size]
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        # -> [bs, seq_len, num_attention_heads, attn_head_size]
        tensor = tensor.view(
            tensor.size(0), tensor.size(1), num_attention_heads * attn_head_size
        )
        # -> [bs, seq_len, hidden_size]
        return tensor

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        # q, k, v: [bs, num_attention_heads, seq_len, attn_head_size]
        # compute causal mask from causal mask buffer
        batch_size, num_attention_heads, query_length, attn_head_size = query.size()
        key_length = key.size(-2)

        causal_mask = self.bias[
            :, :, key_length - query_length : key_length, :key_length
        ]

        query = query.view(
            batch_size * num_attention_heads, query_length, attn_head_size
        )
        key = key.view(batch_size * num_attention_heads, key_length, attn_head_size)
        attn_scores = torch.zeros(
            batch_size * num_attention_heads,
            query_length,
            key_length,
            dtype=query.dtype,
            device=key.device,
        )
        attn_scores = torch.baddbmm(
            attn_scores,
            query,
            key.transpose(1, 2),
            beta=1.0,
            alpha=(
                torch.tensor(
                    1.0, dtype=self.norm_factor.dtype, device=self.norm_factor.device
                )
                / self.norm_factor
            ),
        )
        attn_scores = attn_scores.view(
            batch_size, num_attention_heads, query_length, key_length
        )

        mask_value = torch.finfo(attn_scores.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected
        # scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y
        # to be on the same device`
        mask_value = torch.tensor(mask_value, dtype=attn_scores.dtype).to(
            attn_scores.device
        )
        attn_scores = torch.where(causal_mask, attn_scores, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_scores = attn_scores + attention_mask

        attn_weights = nn.functional.softmax(attn_scores, dim=-1)
        attn_weights = attn_weights.to(value.dtype)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights


def attention_mask_func(attention_scores, ltor_mask):
    attention_scores.masked_fill_(~ltor_mask, torch.finfo(attention_scores.dtype).min)
    return attention_scores


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings, base=10000, device=None,
                    dtype=torch.bfloat16, interpolation=1.0):
        super().__init__()
        self.dtype = dtype
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(
            self.max_seq_len_cached,
            device=self.inv_freq.device,
            dtype=self.inv_freq.dtype,
        )
        if interpolation is not None:
            t *= 1 / interpolation
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to
        # obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos().to(dtype)[None, None, :, :]
        self.sin_cached = emb.sin().to(dtype)[None, None, :, :]

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in
        # `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(
                self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype
            )
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in
            # order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos().to(self.dtype)[None, None, :, :]
            self.sin_cached = emb.sin().to(self.dtype)[None, None, :, :]
        return self.cos_cached[:seq_len, ...].to(x.device), self.sin_cached[
            :seq_len, ...
        ].to(x.device)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    gather_indices = position_ids[:, None, :, None]  # [bs, 1, seq_len, 1]
    gather_indices = gather_indices.repeat(1, cos.shape[1], 1, cos.shape[3])
    cos = torch.gather(cos.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    sin = torch.gather(sin.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_unpacked(q, k, cos, sin):
    # q(k): [total, np(ng), head],
    # cos, sin: [total, head]
    cos = cos[:, None, :]
    sin = sin[:, None, :]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class GPTNeoXMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_act != "swiglu":
            before_activation_size = config.intermediate_size
        else:
            before_activation_size = config.intermediate_size * 2
        self.dense_h_to_4h = nn.Linear(
            config.hidden_size, before_activation_size // config.mp_size
        )
        self.dense_4h_to_h = nn.Linear(
            config.intermediate_size // config.mp_size, config.hidden_size
        )
        self.act = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dense_4h_to_h(hidden_states)
        return hidden_states

class GPTNeoXRoutedMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.router = TopKRouter(config)
        self.router_topk = config.moe_router_topk
        self.experts = nn.ModuleList([GPTNeoXMLP(config) for _ in range(config.num_moe_experts)])
        self.num_moe_experts = config.num_moe_experts

    def token_permutation(
        self, hidden_states: torch.Tensor, max_prob: torch.Tensor, max_ind: torch.Tensor
    ):
        """Dispatch tokens to local experts. It's composed of two stages:
        (1) Permute the tokens across the expert parallel devices. After this stage,
        each device receives all of the tokens assigned to its local set of experts
        in its local HBM.
        (2) Permute the tokens locally so that they are grouped by their expert
        assignment. After the stage (1), the tokens are grouped by which device
        they came from. We re-order them locally for subsequent efficient computation.

        Args:
            hidden_states: input tokens of shape [SeqLen/TP, MBS, HiddenSize]
            max_prob: probs of token assignment to local experts.
            max_ind: token assignment to local experts.

        Returns:
            permuted_local_hidden_states: Permutation of tokens to local experts group.
            tokens_per_expert: the number of tokens each local expert to process.
            indices: The indices of `local_indices` (which holds the un-sorted expert
            indices of tokens that local expert can process) that give its sorted order along dim 0.
            global_local_map (optional): 2D tensor. A mask of mapping between global and flocal tokens where each
            element is True if it's between the local_expert_indices. Only useful
            when cross device token permutation is enabled and **AllGahter** is performed.
        """
        self.hidden_shape = hidden_states.shape
        # [S/TP, B, H] -> [S*B/TP, H]
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])


        if self.router_topk > 1:
            global_local_map = torch.ones_like(max_ind).bool()
            local_indices = max_ind.masked_select(global_local_map)
            local_probs = max_prob.masked_select(global_local_map)
            global_local_map = global_local_map.nonzero()[:, 0]
            global_local_map = global_local_map.view(-1, 1).expand(-1, hidden_states.shape[-1])
            local_hidden_states = torch.gather(hidden_states, 0, global_local_map)
        else:
            local_indices = max_ind
            local_probs = max_prob
            local_hidden_states = hidden_states
            global_local_map = None

        with torch.no_grad():
            # The indices of local_indices that give its sorted order along dim 0.
            indices = torch.argsort(local_indices, dim=0)
            tokens_per_expert = torch.histc(
                local_indices.float(),
                bins=self.num_moe_experts,
                min=0,
                max=self.num_moe_experts - 1,
            )
            tokens_per_expert = tokens_per_expert.cpu().to(torch.long)

        # Stage2: permute the tokens locally so that they are grouped by their expert assignment
        # Reshape indices to be compatible with Tensor.gather
        indices = indices.view(-1, 1).expand(-1, hidden_states.shape[-1])
        permuted_local_hidden_states = torch.gather(local_hidden_states, 0, indices)
        return (
            permuted_local_hidden_states,
            tokens_per_expert,
            local_probs,
            indices,
            global_local_map,
        )

    def token_unpermutation(
        self,
        hidden_states: torch.Tensor,
        scores: torch.Tensor,
        indices: torch.Tensor,
        global_local_map: torch.Tensor = None,
    ):
        """
        Reverse process of `dispatch()` which permutes the ouput of local
        experts locallay and across expert parallel rank into the original order to
        produce the final output.

        Args:
            hidden_states: 2D tensor of shape [sum_tokens_of_all_local_experts, HiddenSize],
            ouput of local experts.
            scores: 2D tensor of the probs of token assignment to local experts.
            indices: 2D tensor of the indices of `local_indices` (which holds the un-sorted expert
            indices of tokens that local expert can process) that give its sorted order along dim 0.
            global_local_map (optional): 2D tensor, a mask of mapping between global and local tokens where each
            element is True if it's between the local_expert_indices. Only useful
            when cross device token permutation is enabled and **AllGather** is performed.

        Returns:
            output_total: un-permuted updated hidden states output from all local experts
            with shape of [SeqLen/TP, MBS, HiddenSize]
        """
        # Stage1: unpermute the tokens and bias locally respectively.

        scores = scores.to(dtype=hidden_states.dtype)
        unpermuted_local_hidden = torch.zeros_like(hidden_states)
        assert indices.shape == hidden_states.shape
        unpermuted_local_hidden = unpermuted_local_hidden.scatter(0, indices, hidden_states)

        # Scale the expert output prior to reduction and subsequent to local unpermutation if k > 1.
        if self.router_topk > 1:
            unpermuted_local_hidden = unpermuted_local_hidden * scores.view(-1, 1)


        output_total = unpermuted_local_hidden

        if self.router_topk > 1:
            global_num_tokens = self.hidden_shape[0] * self.hidden_shape[1]
            global_hidden_shape = [global_num_tokens, hidden_states.shape[-1]]
            unpermuted_global_hidden = torch.zeros(
                global_hidden_shape,
                dtype=hidden_states.dtype,
                # device=torch.cuda.current_device(),
                device=unpermuted_local_hidden.device, #AITOR: CPU bugfix
            )
            output_total = unpermuted_global_hidden.scatter_add(
                0, global_local_map, unpermuted_local_hidden
            )

        if self.router_topk == 1:
            output_total = output_total * scores
        output_total = output_total.view(self.hidden_shape)


        return output_total

    def forward(self, hidden_states):
        scores, indices = self.router(hidden_states)
        (
            dispatched_input,
            tokens_per_expert,
            scores,
            indices,
            global_local_map,
        ) = self.token_permutation(hidden_states, scores, indices)

        #TODO: Sequential MLP
        permuted_local_hidden_states = dispatched_input
        expert_output = torch.zeros_like(permuted_local_hidden_states)

        cumsum_num_tokens = torch.cumsum(tokens_per_expert, dim=0)
        # Insert zero at the begining for offset index's convenience
        zero_tensor = torch.zeros(1, dtype=torch.long)
        cumsum_num_tokens = torch.cat((zero_tensor, cumsum_num_tokens))
        for expert_num, expert in enumerate(self.experts):
            start = cumsum_num_tokens[expert_num]
            end = cumsum_num_tokens[expert_num + 1]
            hidden = permuted_local_hidden_states[start:end]
            output = expert(hidden)

            expert_output[start:end] = output


        output = self.token_unpermutation(
            expert_output, scores, indices, global_local_map
        )

        return output


class TopKRouter(nn.Module):
    """Top-k Router class"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts = self.config.num_moe_experts
        self.topk = self.config.moe_router_topk
        self.routing_type = self.config.moe_router_load_balancing_type
        # Initialize the gate weights.
        self.linear = torch.nn.Linear(
            self.config.hidden_size, self.num_experts, bias=False
        )

    def sinkhorn_load_balancing(self, logits: torch.Tensor):
        """Apply sinkhorn routing to the logits tensor.

        Args:
            logits (torch.Tensor): The logits tensor.

        Returns:
            torch.Tensor: The logits tensor after applying sinkhorn routing.
        """

        def _sinkhorn_activation(logits):
            if self.topk == 1:
                logits = torch.sigmoid(logits)
            else:  # k > 1
                logits = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
            return logits


        logits = _sinkhorn_activation(logits)
        scores, indices = torch.topk(logits, k=self.topk, dim=1)
        return scores, indices

    def aux_loss_load_balancing(self, logits: torch.Tensor):
        """Apply loss-based load balancing to the logits tensor.

        Args:
            logits (torch.Tensor): The logits tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The scores and the indices tensor after applying load balancing.
        """
        top_logits, indices = torch.topk(logits, k=self.topk, dim=1)
        scores = torch.softmax(top_logits, dim=-1, dtype=torch.float32).type_as(logits)
        # Apply load balancing loss
        return scores, indices


    def gating(self, input: torch.Tensor):
        """Forward pass of the router gate.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Logits tensor.
        """
        logits = self.linear(input)
        return logits

    def routing(self, logits: torch.Tensor):
        """Top-k routing function

        Args:
            logits (torch.Tensor): Logits tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Probs and the indices tensor.
        """
        logits = logits.view(-1, self.config.num_moe_experts)

        # # Apply Z-Loss
        # logits = self.apply_z_loss(logits)
        # # Apply input jitter
        # logits = self.apply_input_jitter(logits)
        if self.routing_type == "sinkhorn":
            scores, indices = self.sinkhorn_load_balancing(logits)
        elif self.routing_type == "aux_loss":
            scores, indices = self.aux_loss_load_balancing(logits)
        elif self.routing_type is None:
            # A naive top-k routing without load balancing
            top_logits, indices = torch.topk(logits, k=self.k, dim=1)
            scores = torch.softmax(top_logits, dim=-1, dtype=torch.float32).type_as(logits)

        return scores, indices


    def forward(self, input: torch.Tensor):
        """
        Forward pass of the router.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: scores and indices.
        """
        self.hidden = input.shape[-1]

        logits = self.gating(input)
        logits = logits.view(-1, self.config.num_moe_experts)

        scores, indices = self.routing(logits)

        return scores, indices




class YasaLayer(nn.Module):
    def __init__(self, config: YasaConfig):
        super().__init__()
        self.config = config
        self.use_parallel_residual = config.use_parallel_residual
        if getattr(config, 'normalization', 'LayerNorm') == 'LayerNorm':
            self.input_layernorm = nn.LayerNorm(
                config.hidden_size, eps=config.layer_norm_eps
            )
            self.post_attention_layernorm = nn.LayerNorm(
                config.hidden_size, eps=config.layer_norm_eps
            )
        else:
            self.input_layernorm = RMSNorm(
                config.hidden_size, eps=config.layer_norm_eps
            )
            self.post_attention_layernorm = RMSNorm(
                config.hidden_size, eps=config.layer_norm_eps
            )
        self.attention = GPTNeoXAttention(config)
        if config.num_moe_experts == 1:
            self.mlp = GPTNeoXMLP(config)
        else:
            self.mlp = GPTNeoXRoutedMLP(config)

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor],
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        cu_seqlens: Optional[torch.IntTensor] = None,
        max_seqlen: Optional[int] = None,
        rot_embs: Optional[list] = None,
    ):
        if self.config.unpad:
            attention_layer_outputs = self.attention(
                BackwardAllReduce.apply(self.input_layernorm(hidden_states)),
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                cos=rot_embs[0],
                sin=rot_embs[1],
            )
        else:
            attention_layer_outputs = self.attention(
                BackwardAllReduce.apply(self.input_layernorm(hidden_states)),
                attention_mask=attention_mask,
                position_ids=position_ids,
                layer_past=layer_past,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
        attn_output = attention_layer_outputs[
            0
        ]  # output_attn: attn_output, present, (attn_weights)
        outputs = attention_layer_outputs[1:]

        if self.use_parallel_residual:
            # pseudocode:
            # x = x + attn(ln1(x)) + mlp(ln2(x))
            mlp_output = self.mlp(
                BackwardAllReduce.apply(self.post_attention_layernorm(hidden_states))
            )
            hidden_states = (
                ForwardAllReduce.apply(mlp_output + attn_output) + hidden_states
            )
        else:
            # pseudocode:
            # x = x + attn(ln1(x))
            # x = x + mlp(ln2(x))
            attn_output = ForwardAllReduce.apply(attn_output) + hidden_states
            mlp_output = self.mlp(
                BackwardAllReduce.apply(self.post_attention_layernorm(attn_output))
            )
            hidden_states = ForwardAllReduce.apply(mlp_output) + attn_output

        if use_cache:
            outputs = (
                hidden_states,
            ) + outputs  # hidden_states, present, (attn_weights)
        else:
            outputs = (hidden_states,) + outputs[1:]  # hidden_states, (attn_weights)

        return outputs


class YasaModel(YasaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embed_in = nn.Embedding(config.vocab_size, config.hidden_size)
        if config.add_position_embedding:
            self.embed_pos = nn.Embedding(
                config.max_position_embeddings, config.hidden_size
            )
        else:
            self.embed_pos = None

        self.rotary_emb = RotaryEmbedding(
            config.hidden_size // config.num_attention_heads,
            config.max_position_embeddings,
            base=config.rotary_emb_base,
        )

        self.layers = nn.ModuleList(
            [YasaLayer(config) for _ in range(config.num_hidden_layers)]
        )

        if getattr(config, 'normalization', 'LayerNorm') == 'LayerNorm':
            self.final_layer_norm = nn.LayerNorm(
                config.hidden_size, eps=config.layer_norm_eps
            )
        else:
            self.final_layer_norm = RMSNorm(
                config.hidden_size, eps=config.layer_norm_eps
            )

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_in

    def set_input_embeddings(self, value):
        self.embed_in = value

    def pack_emb(self, input_emb, mask):
        # [batch, seq, hidden] --> [total, hidden]
        packed_emb = input_emb[mask.bool()]
        # allocate batch+1 array flash_attn needs
        cu_seqlens = torch.zeros(mask.shape[0]+1, dtype=torch.int32, device=mask.device)
        seq_lens = mask.sum(dim=1).to(torch.int32)
        max_seqlen = seq_lens.max().item()
        torch.cumsum(seq_lens, 0, out=cu_seqlens[1:])

        # pack rot
        # TODO: why currnt RotaryEmbedding slice at return but on dim 0?
        # [1, 1, max_seq, head] --> [batch, max_seq, head]
        cos, sin = self.rotary_emb(input_emb, seq_len=input_emb.shape[1])
        cos = cos.squeeze(0).expand(seq_lens.shape[0], -1, -1)
        sin = sin.squeeze(0).expand(seq_lens.shape[0], -1, -1)
        # We assume there is only padding on left or right of sequence
        # generate a mask that is packed to the left to get correct rotary
        pos = torch.arange(cos.shape[1], dtype=torch.int32, device=cos.device)
        # [batch, max_seq]
        left_mask = pos[None,:] < seq_lens[:, None]
        # [total, head]
        cos = cos[left_mask]
        sin = sin[left_mask]

        return packed_emb, cu_seqlens, max_seqlen, cos, sin

    def scatter_logits(self, logits, mask):
        # [total, hidden] --> [batch, seq, hidden]
        out = torch.zeros(mask.shape+logits.shape[-1:], dtype=logits.dtype, device=logits.device)
        out[mask.bool()] += logits
        return out

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        r"""forward function."""
        # For now, we do pack/unpack here only for input_embeds mode
        # this should help mmlm training by removing pad token during main network
        # this is cleaner because all change will be contained, at the cost of not
        # support input_ids, input embedding and output embedding
        if self.config.unpad:
            use_cache = False
            if inputs_embeds is not None:
                inputs_embeds, cu_seqlens, max_seqlen, rot_cos, rot_sin = self.pack_emb(inputs_embeds, attention_mask)
            if input_ids is not None:
                input_ids, cu_seqlens, max_seqlen, rot_cos, rot_sin = self.pack_emb(input_ids, attention_mask)

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * self.config.num_hidden_layers)
        else:
            past_length = past_key_values[0][0].size(-2)

        if not self.config.unpad:
            batch_size, seq_length = input_shape
            if position_ids is None:
                device = (
                    input_ids.device
                    if input_ids is not None
                    else inputs_embeds.device
                )
                position_ids = torch.arange(
                    past_length,
                    seq_length + past_length,
                    dtype=torch.long,
                    device=device,
                )
                position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
            else:
                position_ids = position_ids.view(-1, seq_length).long()

        # Attention mask.
        if attention_mask is not None and not self.config.unpad:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.to(torch.bool).view(batch_size, -1)
            #attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to
            # [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of
            # causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast
            # dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and
            # 0.0 formasked positions, this operation will create a tensor
            # which is 0.0 for positions we want to attend and the dtype's
            # smallest value for masked positions.  Since we are adding it to
            # the raw scores before the softmax, this is effectively the same
            # as removing these entirely.
            #attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            #attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # Prepare head mask if needed 1.0 in head_mask indicate we keep the
        # head attention_probs has shape bsz x n_heads x N x N input head_mask
        # has shape [num_heads] or [num_hidden_layers x num_heads] and
        # head_mask is converted to shape [num_hidden_layers x batch x
        # num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        if inputs_embeds is None:
            inputs_embeds = self.embed_in(input_ids)

        if self.embed_pos:
            inputs_embeds = inputs_embeds + self.embed_pos(position_ids)

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning(
                    "`use_cache=True` is incompatible with gradient"
                    "checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        presents = () if use_cache else None
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, (layer, layer_past) in enumerate(zip(self.layers, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for layer_past
                        return module(*inputs, use_cache, None, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    head_mask[i],
                )
            else:
                if self.config.unpad:
                    outputs = layer(
                        hidden_states,
                        cu_seqlens=cu_seqlens,
                        max_seqlen=max_seqlen,
                        rot_embs=[rot_cos, rot_sin],
                    )
                else:
                    outputs = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        head_mask=head_mask[i],
                        layer_past=layer_past,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                    )
            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)
            if output_attentions:
                all_attentions = all_attentions + (outputs[2 if use_cache else 1],)

        hidden_states = self.final_layer_norm(hidden_states)
        if self.config.unpad:
            hidden_states = self.scatter_logits(hidden_states, attention_mask)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_attentions]
                if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class YasaCausalLM(YasaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"position_ids",
        r"predictions.decoder.bias",
        r"layers.(\d+).attention.(bias|masked_bias|rotary_emb.inv_freq)",
    ]

    def __init__(self, config):
        super().__init__(config)

        self.gpt_neox = YasaModel(config)
        self.embed_out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.gpt_neox.get_input_embeddings()

    def get_output_embeddings(self):
        return self.embed_out

    def set_output_embeddings(self, new_embeddings):
        self.embed_out = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""forward function.
        ```"""
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.gpt_neox(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        lm_logits = self.embed_out(hidden_states)

        lm_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and
            # input ids by one
            shift_logits = lm_logits[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)).to(device=labels.device),
                labels.view(-1),
            )

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithPast(
            loss=lm_loss,
            logits=lm_logits.to(
                device=(input_ids if input_ids is not None else inputs_embeds).device
            ),  # Hack for ensuring same GPU decoding,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        input_shape = input_ids.shape

        # cut decoder_input_ids if past is used
        if past_key_values and past_key_values[0] is not None:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if model is used as a decoder in encoder-decoder model, the decoder
        # attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
        )

        return model_inputs

    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx)
                    for past_state in layer_past[:2]
                )
                + layer_past[2:],
            )
        return reordered_past

    def slice_and_load(self, state_dict, **kwargs):
        config = self.config
        num_attention_heads_per_device = config.num_attention_heads // config.mp_size
        per_head_size = config.hidden_size // config.num_attention_heads
        gate_dim = 2 if config.hidden_act == "swiglu" else 1
        intermediate_per_device = config.intermediate_size // config.mp_size

        def _slice(k, v):
            if k.endswith("attention.query_key_value.weight"):
                return v.reshape(
                    (config.num_attention_heads, 3, per_head_size, config.hidden_size)
                )[
                    MPU.mp_rank
                    * num_attention_heads_per_device : (MPU.mp_rank + 1)
                    * num_attention_heads_per_device
                ].reshape(
                    (3 * config.hidden_size // config.mp_size, config.hidden_size)
                )
            elif k.endswith("attention.query_key_value.bias"):
                return v.reshape((config.num_attention_heads, 3, per_head_size))[
                    MPU.mp_rank
                    * num_attention_heads_per_device : (MPU.mp_rank + 1)
                    * num_attention_heads_per_device
                ].reshape((3 * config.hidden_size // config.mp_size,))
            elif k.endswith("attention.dense.weight"):
                return v[
                    :,
                    MPU.mp_rank
                    * num_attention_heads_per_device
                    * per_head_size : (MPU.mp_rank + 1)
                    * num_attention_heads_per_device
                    * per_head_size,
                ]
            elif k.endswith("attention.dense.bias"):
                return v / config.mp_size
            elif k.endswith("dense_h_to_4h.weight"): #  k.endswith("mlp.dense_h_to_4h.weight"): AITOR: Remove mlp to also match experts
                return v.reshape(
                    (gate_dim, config.intermediate_size, config.hidden_size)
                )[
                    :,
                    MPU.mp_rank
                    * intermediate_per_device : (MPU.mp_rank + 1)
                    * intermediate_per_device,
                ].reshape(
                    (gate_dim * intermediate_per_device, config.hidden_size)
                )
            elif k.endswith("dense_h_to_4h.bias"): # k.endswith("mlp.dense_h_to_4h.bias"):
                return v.reshape((gate_dim, config.intermediate_size))[
                    :,
                    MPU.mp_rank
                    * intermediate_per_device : (MPU.mp_rank + 1)
                    * intermediate_per_device,
                ].reshape((gate_dim * intermediate_per_device,))
            elif k.endswith("dense_4h_to_h.weight"): # k.endswith("mlp.dense_4h_to_h.weight")
                return v[
                    :,
                    MPU.mp_rank
                    * intermediate_per_device : (MPU.mp_rank + 1)
                    * intermediate_per_device,
                ]
            elif k.endswith("dense_4h_to_h.bias"): # k.endswith("mlp.dense_4h_to_h.bias"):
                return v / config.mp_size
            return v

        state_dict = {k: _slice(k, v) for k, v in state_dict.items()}
        return self.load_state_dict(state_dict, **kwargs)
