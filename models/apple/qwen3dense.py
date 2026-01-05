import logging
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from safetensors import safe_open
from tqdm import tqdm

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("Qwen3")


def apply_rotary_emb(
    x: mx.array,
    cos: mx.array,
    sin: mx.array,
) -> mx.array:
    cos = mx.expand_dims(cos, axis=-2)
    sin = mx.expand_dims(sin, axis=-2)
    x1, x2 = mx.split(x.astype(mx.float32), 2, axis=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return mx.concatenate((y1, y2), axis=-1).astype(x.dtype)


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
        inv_freq = 1.0 / (base ** (mx.arange(0, rotary_dim, 2, dtype=mx.float32) / rotary_dim))
        t = mx.arange(max_position_embeddings, dtype=mx.float32)
        freqs = mx.einsum("i,j -> ij", t, inv_freq)
        cos = mx.cos(freqs)
        sin = mx.sin(freqs)
        cache = mx.concatenate((cos, sin), axis=-1)
        self.cos_sin_cache = cache

    def forward(
        self,
        positions: mx.array,
        query: mx.array,
        key: mx.array,
    ) -> tuple[mx.array, mx.array]:
        num_tokens = positions.shape[0]
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = mx.split(cos_sin, 2, axis=-1)
        query_shape = query.shape
        query = query.reshape(num_tokens, -1, self.head_size)
        query = apply_rotary_emb(query, cos, sin).reshape(query_shape)
        key_shape = key.shape
        key = key.reshape(num_tokens, -1, self.head_size)
        key = apply_rotary_emb(key, cos, sin).reshape(key_shape)
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


def add_rms_norm(x, residual, weight, eps) -> tuple[mx.array, mx.array]:
    if residual is not None:
        x = x + residual
    residual = x
    x = mx.fast.rms_norm(x, weight, eps)
    return x, residual


@dataclass
class Qwen3Config:
    head_dim: int
    hidden_size: int
    hidden_act: str
    intermediate_size: int
    max_position_embeddings: int
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    vocab_size: int
    rope_theta: int
    rms_norm_eps: float


class Qwen3(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            num_embeddings=config.vocab_size,
            dims=config.hidden_size
        )

        self.input_layernorm = [
            mx.zeros((config.hidden_size,)) for _ in range(config.num_hidden_layers)
        ]

        self.q_projs = [
            mx.zeros((config.num_attention_heads * config.head_dim, config.hidden_size))
            for _ in range(config.num_hidden_layers)
        ]
        self.k_projs = [
            mx.zeros((config.num_key_value_heads * config.head_dim, config.hidden_size))
            for _ in range(config.num_hidden_layers)
        ]
        self.v_projs = [
            mx.zeros((config.num_key_value_heads * config.head_dim, config.hidden_size))
            for _ in range(config.num_hidden_layers)
        ]
        self.o_projs = [
            mx.zeros((config.hidden_size, config.num_attention_heads * config.head_dim))
            for _ in range(config.num_hidden_layers)
        ]

        self.q_norm = [
            mx.zeros((config.head_dim,)) for _ in range(config.num_hidden_layers)
        ]
        self.k_norm = [
            mx.zeros((config.head_dim,)) for _ in range(config.num_hidden_layers)
        ]

        self.post_attention_norm = [
            mx.zeros((config.hidden_size,)) for _ in range(config.num_hidden_layers)
        ]

        self.mlp_gate_proj = [
            mx.zeros((config.intermediate_size, config.hidden_size))
            for _ in range(config.num_hidden_layers)
        ]
        self.mlp_up_proj = [
            mx.zeros((config.intermediate_size, config.hidden_size))
            for _ in range(config.num_hidden_layers)
        ]
        self.mlp_down_proj = [
            mx.zeros((config.hidden_size, config.intermediate_size))
            for _ in range(config.num_hidden_layers)
        ]

        self.model_norm = mx.zeros((config.hidden_size,))
        self.lm_head = mx.zeros((config.vocab_size, config.hidden_size))

        # KV cache initialization
        self.kv_cache = [
            [
                mx.zeros((config.num_key_value_heads, config.max_position_embeddings, config.head_dim)),
                mx.zeros((config.num_key_value_heads, config.max_position_embeddings, config.head_dim))
            ]
            for _ in range(config.num_hidden_layers)
        ]

        self.rope = get_rope(
            config.head_dim,
            rotary_dim=config.head_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta
        )

    def load_weight(self, path):
        # currently only support single file
        logger.info("Model Loading...")
        with safe_open(path, framework="pt") as f:
            self.embed_tokens.weight = mx.array(f.get_tensor("model.embed_tokens.weight"))
            self.lm_head = mx.array(f.get_tensor("lm_head.weight"))
            for i in tqdm(range(self.config.num_hidden_layers)):
                self.input_layernorm[i] = mx.array(f.get_tensor(f"model.layers.{i}.input_layernorm.weight"))

                self.q_projs[i] = mx.array(f.get_tensor(f"model.layers.{i}.self_attn.q_proj.weight"))
                self.k_projs[i] = mx.array(f.get_tensor(f"model.layers.{i}.self_attn.k_proj.weight"))
                self.v_projs[i] = mx.array(f.get_tensor(f"model.layers.{i}.self_attn.v_proj.weight"))
                self.o_projs[i] = mx.array(f.get_tensor(f"model.layers.{i}.self_attn.o_proj.weight"))

                self.q_norm[i] = mx.array(f.get_tensor(f"model.layers.{i}.self_attn.q_norm.weight"))
                self.k_norm[i] = mx.array(f.get_tensor(f"model.layers.{i}.self_attn.k_norm.weight"))

                self.post_attention_norm[i] = mx.array(f.get_tensor(f"model.layers.{i}.post_attention_layernorm.weight"))

                self.mlp_gate_proj[i] = mx.array(f.get_tensor(f"model.layers.{i}.mlp.gate_proj.weight"))
                self.mlp_up_proj[i] = mx.array(f.get_tensor(f"model.layers.{i}.mlp.up_proj.weight"))
                self.mlp_down_proj[i] = mx.array(f.get_tensor(f"model.layers.{i}.mlp.down_proj.weight"))

            self.model_norm = mx.array(f.get_tensor("model.norm.weight"))
            self.lm_head = mx.array(f.get_tensor("lm_head.weight"))

        logger.info("Model Loaded")

    def __call__(self, input_ids: mx.array, positions: mx.array, is_prefill=False):
        return self.forward(input_ids, positions, is_prefill)

    def forward(self, input_ids: mx.array, positions: mx.array, is_prefill=False):
        rms_norm_eps = self.config.rms_norm_eps
        hidden_states = self.embed_tokens(input_ids)
        residual = None

        batch_size, seqlen, _ = hidden_states.shape
        assert batch_size == 1, "Currently only support single request"
        head_dim = self.config.head_dim
        num_attention_heads = self.config.num_attention_heads
        num_key_value_heads = self.config.num_key_value_heads

        for i in range(self.config.num_hidden_layers):
            # Input Norm
            hidden_states, residual = add_rms_norm(
                hidden_states, residual, self.input_layernorm[i], rms_norm_eps
            )

            # Attention
            q = mx.einsum('bsh,oh->bso', hidden_states, self.q_projs[i])
            k = mx.einsum('bsh,oh->bso', hidden_states, self.k_projs[i])
            v = mx.einsum('bsh,oh->bso', hidden_states, self.v_projs[i])
            q = q.reshape(batch_size, seqlen, num_attention_heads, head_dim)
            k = k.reshape(batch_size, seqlen, num_key_value_heads, head_dim)
            v = v.reshape(batch_size, seqlen, num_key_value_heads, head_dim)

            q = mx.fast.rms_norm(q, self.q_norm[i], rms_norm_eps)
            k = mx.fast.rms_norm(k, self.k_norm[i], rms_norm_eps)
            q, k = self.rope.forward(positions, q, k)

            # batch, head_cnt, seq_len, head_dim
            q = mx.transpose(q, (0, 2, 1, 3))
            k = mx.transpose(k, (0, 2, 1, 3))
            v = mx.transpose(v, (0, 2, 1, 3))

            # Update KV cache
            k_cache, v_cache = self.kv_cache[i]
            for pos_idx, pos in enumerate(positions):
                k_cache[:, pos, :] = k[0, :, pos_idx, :]
                v_cache[:, pos, :] = v[0, :, pos_idx, :]

            # Retrieve cached keys and values
            max_pos = int(positions[-1]) + 1
            k_cached = k_cache[:, :max_pos, :]
            v_cached = v_cache[:, :max_pos, :]

            # Scaled dot product attention using fast kernel
            # q: (batch, num_heads, seqlen, head_dim)
            # k_cached: (num_kv_heads, max_pos, head_dim)
            # v_cached: (num_kv_heads, max_pos, head_dim)
            # Note: mx.fast.scaled_dot_product_attention natively supports GQA

            # Add batch dimension to k and v
            k_cached = mx.expand_dims(k_cached, axis=0)
            v_cached = mx.expand_dims(v_cached, axis=0)

            # Create causal mask for prefill
            mask = None
            if is_prefill:
                q_len = q.shape[2]
                mask = mx.triu(mx.full((q_len, max_pos), -mx.inf), k=1)
                mask = mx.expand_dims(mask, axis=(0, 1))

            o = mx.fast.scaled_dot_product_attention(
                q, k_cached, v_cached,
                scale=1/head_dim**0.5,
                mask=mask
            )

            hidden_states = mx.einsum(
                'bhsd,ohd->bso',
                o,
                self.o_projs[i].reshape(-1, num_attention_heads, head_dim)
            )

            # Post Attention Norm
            hidden_states, residual = add_rms_norm(
                hidden_states, residual, self.post_attention_norm[i], rms_norm_eps
            )

            # MLP
            hidden_states_gate = mx.einsum('bsh,oh->bso', hidden_states, self.mlp_gate_proj[i])
            hidden_states_up = mx.einsum('bsh,oh->bso', hidden_states, self.mlp_up_proj[i])
            hidden_states = nn.silu(hidden_states_gate) * hidden_states_up
            hidden_states = mx.einsum('bsh,oh->bso', hidden_states, self.mlp_down_proj[i])

        hidden_states, _ = add_rms_norm(
            hidden_states, residual, self.model_norm, rms_norm_eps
        )

        # Compute Logits
        hidden_states = hidden_states[:, -1, :]
        # hidden_states = mx.squeeze(hidden_states, axis=1)
        logits = mx.einsum('bh,vh->bv', hidden_states, self.lm_head)

        # greedy sample
        return mx.argmax(logits, axis=-1)
