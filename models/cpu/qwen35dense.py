import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from tqdm import tqdm

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

logger = logging.getLogger("Qwen3_5")


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    x1, x2 = torch.chunk(x.to(torch.float32), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    # @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = positions.size(0)
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        query = apply_rotary_emb(query, cos, sin).view(query_shape)
        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        key = apply_rotary_emb(key, cos, sin).view(key_shape)
        return query, key


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    assert rope_scaling is None
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb


def add_rms_norm(x, residual, weight, eps) -> tuple[torch.Tensor, torch.Tensor]:
    orig_dtype = x.dtype
    if residual is not None:
        x += residual
    residual = x
    x = x.to(torch.float32)
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x.mul_(torch.rsqrt(var + eps))
    x = x.to(orig_dtype).mul_(weight)
    return x, residual


def rms_norm(x, weight, eps) -> torch.Tensor:
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x.mul_(torch.rsqrt(var + eps))
    x = x.to(orig_dtype).mul_(weight)
    return x


@dataclass
class RopeConfig:
    mrope_interleaved: bool
    mrope_section: list[int]
    rope_type: str
    rope_theta: int
    partial_rotary_factor: float


@dataclass
class Qwen3_5Config:
    head_dim: int
    hidden_size: int
    hidden_act: str
    intermediate_size: int
    max_position_embeddings: int
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    attn_output_gate: bool
    vocab_size: int
    # rope_theta: int
    rms_norm_eps: float
    rope_parameters: RopeConfig

    linear_conv_kernel_dim: int
    linear_key_head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_value_head_dim: int

    layer_types: list[str]


class FullAtention(nn.Module):
    def __init__(self, layer_idx: int, config: Qwen3_5Config):
        super().__init__()
        self.layer_idx = layer_idx
        self.q_proj = nn.Parameter(
            torch.empty(
                config.num_attention_heads * config.head_dim * (1 + config.attn_output_gate), config.hidden_size
            )
        )
        self.k_proj = nn.Parameter(torch.empty(config.num_key_value_heads * config.head_dim, config.hidden_size))
        self.v_proj = nn.Parameter(torch.empty(config.num_key_value_heads * config.head_dim, config.hidden_size))
        self.o_proj = nn.Parameter(torch.empty(config.hidden_size, config.num_attention_heads * config.head_dim))

        self.q_norm = nn.Parameter(torch.empty(config.head_dim))
        self.k_norm = nn.Parameter(torch.empty(config.head_dim))

        self.register_buffer(
            "kv_cache", torch.zeros(2, config.num_key_value_heads, config.max_position_embeddings, config.head_dim)
        )

    def load_weight(self, f):
        self.q_proj.data.copy_(f.get_tensor(f"model.language_model.layers.{self.layer_idx}.self_attn.q_proj.weight"))
        self.k_proj.data.copy_(f.get_tensor(f"model.language_model.layers.{self.layer_idx}.self_attn.k_proj.weight"))
        self.v_proj.data.copy_(f.get_tensor(f"model.language_model.layers.{self.layer_idx}.self_attn.v_proj.weight"))
        self.o_proj.data.copy_(f.get_tensor(f"model.language_model.layers.{self.layer_idx}.self_attn.o_proj.weight"))

        self.q_norm.data.copy_(f.get_tensor(f"model.language_model.layers.{self.layer_idx}.self_attn.q_norm.weight"))
        self.k_norm.data.copy_(f.get_tensor(f"model.language_model.layers.{self.layer_idx}.self_attn.k_norm.weight"))

    def forward(self, hidden_state: torch.Tensor, position: torch.Tensor):
        return hidden_state


class LinearAttention(nn.Module):
    def __init__(self, layer_idx: int, config: Qwen3_5Config):
        super().__init__()
        self.layer_idx = layer_idx
        key_dim = config.linear_num_key_heads * config.linear_key_head_dim
        value_dim = config.linear_num_value_heads * config.linear_value_head_dim
        conv_dim = key_dim * 2 + value_dim
        self.A_log = nn.Parameter(torch.empty(config.linear_num_value_heads, dtype=torch.float32))
        self.conv1d = nn.Parameter(torch.empty(conv_dim, 1, config.linear_conv_kernel_dim))
        self.dt_biase = nn.Parameter(torch.ones(config.linear_num_value_heads))
        self.in_proj_a = nn.Parameter(torch.empty(config.linear_num_value_heads, config.hidden_size))
        self.in_proj_b = nn.Parameter(torch.empty(config.linear_num_value_heads, config.hidden_size))
        self.in_proj_qkv = nn.Parameter(torch.empty(sum([key_dim, key_dim, value_dim]), config.hidden_size))
        self.in_proj_z = nn.Parameter(torch.empty(value_dim, config.hidden_size))
        self.norm = nn.Parameter(torch.empty(config.linear_value_head_dim, dtype=torch.float32))
        self.out_proj = nn.Parameter(torch.empty(config.hidden_size, value_dim))

    def load_weight(self, f):
        self.A_log.data.copy_(f.get_tensor(f"model.language_model.layers.{self.layer_idx}.linear_attn.A_log"))
        self.conv1d.data.copy_(f.get_tensor(f"model.language_model.layers.{self.layer_idx}.linear_attn.conv1d.weight"))
        self.dt_biase.data.copy_(f.get_tensor(f"model.language_model.layers.{self.layer_idx}.linear_attn.dt_bias"))
        self.in_proj_a.data.copy_(
            f.get_tensor(f"model.language_model.layers.{self.layer_idx}.linear_attn.in_proj_a.weight")
        )
        self.in_proj_b.data.copy_(
            f.get_tensor(f"model.language_model.layers.{self.layer_idx}.linear_attn.in_proj_b.weight")
        )
        self.in_proj_qkv.data.copy_(
            f.get_tensor(f"model.language_model.layers.{self.layer_idx}.linear_attn.in_proj_qkv.weight")
        )
        self.in_proj_z.data.copy_(
            f.get_tensor(f"model.language_model.layers.{self.layer_idx}.linear_attn.in_proj_z.weight")
        )
        self.norm.data.copy_(f.get_tensor(f"model.language_model.layers.{self.layer_idx}.linear_attn.norm.weight"))
        self.out_proj.data.copy_(
            f.get_tensor(f"model.language_model.layers.{self.layer_idx}.linear_attn.out_proj.weight")
        )

    def forward(self, hidden_state: torch.Tensor):
        return hidden_state


class MLP(nn.Module):
    def __init__(self, layer_idx: int, config: Qwen3_5Config):
        super().__init__()
        self.layer_idx = layer_idx
        self.mlp_gate_proj = nn.Parameter(torch.empty(config.intermediate_size, config.hidden_size))
        self.mlp_up_proj = nn.Parameter(torch.empty(config.intermediate_size, config.hidden_size))
        self.mlp_down_proj = nn.Parameter(torch.empty(config.hidden_size, config.intermediate_size))

    def load_weight(self, f):
        self.mlp_gate_proj.data.copy_(
            f.get_tensor(f"model.language_model.layers.{self.layer_idx}.mlp.gate_proj.weight")
        )
        self.mlp_up_proj.data.copy_(f.get_tensor(f"model.language_model.layers.{self.layer_idx}.mlp.up_proj.weight"))
        self.mlp_down_proj.data.copy_(
            f.get_tensor(f"model.language_model.layers.{self.layer_idx}.mlp.down_proj.weight")
        )

    def forward(self, hidden_state: torch.Tensor):
        hidden_state_gate = torch.einsum("bsh,oh->bso", hidden_state, self.mlp_gate_proj.data)
        hidden_state_up = torch.einsum("bsh,oh->bso", hidden_state, self.mlp_up_proj.data)
        hidden_state = F.silu(hidden_state_gate) * hidden_state_up
        hidden_state = torch.einsum("bsh, oh->bso", hidden_state, self.mlp_down_proj.data)
        return hidden_state


class Layer(nn.Module):
    def __init__(self, layer_idx: int, config: Qwen3_5Config):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.config = config
        self.input_layernorm = nn.Parameter(torch.empty(config.hidden_size))
        self.post_attention_layernorm = nn.Parameter(torch.empty(config.hidden_size))

        if self.layer_type == "full_attention":
            self.self_attn = FullAtention(layer_idx, config)
        elif self.layer_type == "linear_attention":
            self.linear_attn = LinearAttention(layer_idx, config)
        self.mlp = MLP(layer_idx, config)

    def load_weight(self, f):
        self.input_layernorm.data.copy_(
            f.get_tensor(f"model.language_model.layers.{self.layer_idx}.input_layernorm.weight")
        )
        self.post_attention_layernorm.data.copy_(
            f.get_tensor(f"model.language_model.layers.{self.layer_idx}.post_attention_layernorm.weight")
        )

        if self.layer_type == "full_attention":
            self.self_attn.load_weight(f)
        elif self.layer_type == "linear_attention":
            self.linear_attn.load_weight(f)
        self.mlp.load_weight(f)

    def forward(self, hidden_state: torch.Tensor, position: torch.Tensor, residual: torch.Tensor | None):
        hidden_state, residual = add_rms_norm(
            hidden_state, residual, self.input_layernorm.data, self.config.rms_norm_eps
        )
        if self.layer_type == "full_attention":
            hidden_state = self.self_attn(hidden_state, position)
        elif self.layer_type == "linear_attention":
            hidden_state = self.linear_attn(hidden_state)
        hidden_state, residual = add_rms_norm(
            hidden_state, residual, self.post_attention_layernorm.data, self.config.rms_norm_eps
        )
        hidden_state = self.mlp(hidden_state)
        return hidden_state, residual


class Qwen3_5(nn.Module):
    def __init__(self, config: Qwen3_5Config, device="cpu"):
        super().__init__()
        torch.set_default_device(device)
        self.config = config
        self.embed_tokens = nn.Embedding(
            num_embeddings=config.vocab_size, embedding_dim=config.hidden_size, _freeze=True
        )
        self.layers = nn.ModuleList([Layer(layer_idx, config) for layer_idx in range(config.num_hidden_layers)])
        self.model_norm = nn.Parameter(torch.empty(config.hidden_size))

        # self.rope = get_rope(
        #     config.head_dim,
        #     rotary_dim=config.head_dim,
        #     max_position=config.max_position_embeddings,
        #     base=config.rope_parameters.rope_theta
        # )

    def load_weight(self, path):
        # currently only support single file
        logger.info("Model Loading...")
        with safe_open(path, "pt", "cpu") as f:
            self.embed_tokens.weight.copy_(f.get_tensor("model.language_model.embed_tokens.weight"))
            for i in tqdm(range(self.config.num_hidden_layers)):
                self.layers[i].load_weight(f)
            self.model_norm.data.copy_(f.get_tensor("model.language_model.norm.weight"))

        logger.info("Model Loaded")

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill=False):
        rms_norm_eps = self.config.rms_norm_eps
        hidden_states = self.embed_tokens(input_ids)
        residual = None

        batch_size, seqlen, _ = hidden_states.shape
        assert batch_size == 1, "Currently only support singual request"

        for layer_idx in range(self.config.num_hidden_layers):
            hidden_states, residual = self.layers[layer_idx](hidden_states, positions, residual)

        hidden_states, _ = add_rms_norm(hidden_states, residual, self.model_norm.data, rms_norm_eps)
        hidden_states = (hidden_states[:, -1, :]).squeeze(1)
        logits = torch.einsum("bh,vh->bv", hidden_states, self.embed_tokens.weight)

        return logits.argmax(dim=-1)
