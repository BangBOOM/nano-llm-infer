import glob
import json
import logging
import math
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu
from safetensors import safe_open
from tqdm import tqdm

from models.npu.utils import ParallelCommunicationGroup, ParallelConfig, add_rms_norm

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("Qwen3")

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
    tie_word_embeddings: bool


class Qwen3(nn.Module):
    def __init__(self, config:Qwen3Config, parallel_config:ParallelConfig, parallel_comm:ParallelCommunicationGroup):
        super().__init__()
        self.config = config
        self.parallel_config = parallel_config
        self.parallel_comm = parallel_comm
        self.embed_tokens = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.hidden_size, _freeze=True)
        self.input_layernorm = nn.ParameterList([nn.Parameter(torch.empty(config.hidden_size)) for _ in range(config.num_hidden_layers)])

        self.qkv_projs = nn.ParameterList([nn.Parameter(torch.empty(config.hidden_size, config.num_attention_heads * config.head_dim + 2 * config.num_key_value_heads * config.head_dim)) for _ in range(config.num_hidden_layers)])
        self.o_projs = nn.ParameterList([nn.Parameter(torch.empty(config.num_attention_heads*config.head_dim, config.hidden_size)) for _ in range(config.num_hidden_layers)])

        self.q_norm = nn.ParameterList([nn.Parameter(torch.empty(config.head_dim)) for _ in range(config.num_hidden_layers)])
        self.k_norm = nn.ParameterList([nn.Parameter(torch.empty(config.head_dim)) for _ in range(config.num_hidden_layers)])

        self.post_attention_norm = nn.ParameterList([nn.Parameter(torch.empty(config.hidden_size)) for _ in range(config.num_hidden_layers)])

        self.mlp_gate_up_proj = nn.ParameterList([nn.Parameter(torch.empty(config.hidden_size, 2 * config.intermediate_size)) for _ in range(config.num_hidden_layers)])
        self.mlp_down_proj = nn.ParameterList([nn.Parameter(torch.empty(config.intermediate_size, config.hidden_size)) for _ in range(config.num_hidden_layers)])

        self.model_norm = nn.Parameter(torch.empty(config.hidden_size))
        self.lm_head = nn.Parameter(torch.empty(config.hidden_size, config.vocab_size))

        self.atten_mask = nn.Parameter(~torch.tril(torch.ones(2048, 2048, dtype=torch.bool)).unsqueeze(0), requires_grad=False)
        self.attention_scale = 1/math.sqrt(config.head_dim)

    def load_weight(self, model_path):
        if os.path.exists(os.path.join(model_path, "model.safetensors.index.json")):
            self._load_weight_multi_file(model_path)
        elif os.path.exists(os.path.join(model_path, "model.safetensors")):
            self._load_weight_single_file(os.path.join(model_path, "model.safetensors"))
        else:
            raise ValueError("model path invalid!!!")

    def _load_weight_single_file(self, path):
        logger.info("Model Loading...")
        with safe_open(path, "pt", "cpu") as f:
            self.embed_tokens.weight.copy_(f.get_tensor("model.embed_tokens.weight"))
            self.lm_head.data.copy_(f.get_tensor("lm_head.weight").permute(1, 0))
            for i in tqdm(range(self.config.num_hidden_layers)):
                self.input_layernorm[i].data.copy_(f.get_tensor(f"model.layers.{i}.input_layernorm.weight"))

                q = f.get_tensor(f"model.layers.{i}.self_attn.q_proj.weight").permute(1, 0)
                k = f.get_tensor(f"model.layers.{i}.self_attn.k_proj.weight").permute(1, 0)
                v = f.get_tensor(f"model.layers.{i}.self_attn.v_proj.weight").permute(1, 0)
                self.qkv_projs[i].data.copy_(torch.cat([q, k, v], dim=-1))
                self.o_projs[i].data.copy_(f.get_tensor(f"model.layers.{i}.self_attn.o_proj.weight").permute(1, 0))

                self.q_norm[i].data.copy_(f.get_tensor(f"model.layers.{i}.self_attn.q_norm.weight"))
                self.k_norm[i].data.copy_(f.get_tensor(f"model.layers.{i}.self_attn.k_norm.weight"))

                self.post_attention_norm[i].data.copy_(f.get_tensor(f"model.layers.{i}.post_attention_layernorm.weight"))

                mlp_gate_proj = f.get_tensor(f"model.layers.{i}.mlp.gate_proj.weight").permute(1, 0)
                mlp_up_proj = f.get_tensor(f"model.layers.{i}.mlp.up_proj.weight").permute(1, 0)
                self.mlp_gate_up_proj[i].data.copy_(torch.cat([mlp_gate_proj, mlp_up_proj], dim=-1))
                self.mlp_down_proj[i].data.copy_(f.get_tensor(f"model.layers.{i}.mlp.down_proj.weight").permute(1, 0))

            self.model_norm.data.copy_(f.get_tensor("model.norm.weight"))

    def _load_weight_multi_file(self, folder):
        with open(os.path.join(folder, "model.safetensors.index.json")) as f:
            weight_map = json.load(f)["weight_map"]

        safe_tensor_map = dict()

        for filename in glob.glob(os.path.join(folder, "*.safetensors")):
            safe_tensor_map[filename.split("/")[-1]] = safe_open(filename, "pt", "cpu")

        def get_tensor(tensor_name):
            return safe_tensor_map[weight_map[tensor_name]].get_tensor(tensor_name)


        logger.info("Model Loading...")
        self.embed_tokens.weight.copy_(get_tensor("model.embed_tokens.weight"))
        if self.config.tie_word_embeddings:
            self.lm_head.data.copy_(self.embed_tokens.weight.permute(1, 0))
        else:
            self.lm_head.data.copy_(get_tensor("lm_head.weight").permute(1, 0))
        self.model_norm.data.copy_(get_tensor("model.norm.weight"))

        for i in tqdm(range(self.config.num_hidden_layers)):
            self.input_layernorm[i].data.copy_(get_tensor(f"model.layers.{i}.input_layernorm.weight"))

            q = get_tensor(f"model.layers.{i}.self_attn.q_proj.weight").permute(1, 0)
            k = get_tensor(f"model.layers.{i}.self_attn.k_proj.weight").permute(1, 0)
            v = get_tensor(f"model.layers.{i}.self_attn.v_proj.weight").permute(1, 0)
            self.qkv_projs[i].data.copy_(torch.cat([q, k, v], dim=-1))
            self.o_projs[i].data.copy_(get_tensor(f"model.layers.{i}.self_attn.o_proj.weight").permute(1, 0))

            self.q_norm[i].data.copy_(get_tensor(f"model.layers.{i}.self_attn.q_norm.weight"))
            self.k_norm[i].data.copy_(get_tensor(f"model.layers.{i}.self_attn.k_norm.weight"))

            self.post_attention_norm[i].data.copy_(get_tensor(f"model.layers.{i}.post_attention_layernorm.weight"))

            mlp_gate_proj = get_tensor(f"model.layers.{i}.mlp.gate_proj.weight").permute(1, 0)
            mlp_up_proj = get_tensor(f"model.layers.{i}.mlp.up_proj.weight").permute(1, 0)
            self.mlp_gate_up_proj[i].data.copy_(torch.cat([mlp_gate_proj, mlp_up_proj], dim=-1))
            self.mlp_down_proj[i].data.copy_(get_tensor(f"model.layers.{i}.mlp.down_proj.weight").permute(1, 0))



    @torch.no_grad()
    def forward(self,
        input_ids:torch.Tensor,
        cos:torch.Tensor,
        sin:torch.Tensor,
        positions:torch.Tensor,
        kv_cache:list,
        actual_seq_lengths_kv:list[int]|None=None,
        is_prefill=False,
        inference_mode="eager"
    ):
        assert (inference_mode == "eager") == is_prefill, "prefill only support eager mode"

        rms_norm_eps = self.config.rms_norm_eps
        hidden_states = self.embed_tokens(input_ids)
        residual = None

        batch_size, seqlen, _ = hidden_states.shape
        assert batch_size == 1, "Currently only support singual request"
        head_dim = self.config.head_dim
        num_attention_heads = self.config.num_attention_heads
        num_key_value_heads = self.config.num_key_value_heads

        if inference_mode == "dynamo":
            cos = cos.repeat(1, 2).reshape(batch_size, seqlen, 1, -1)
            sin = sin.repeat(1, 2).reshape(batch_size, seqlen, 1, -1)

        for i in range(self.config.num_hidden_layers):
            # Input Norm
            hidden_states, residual = add_rms_norm(hidden_states, residual, self.input_layernorm[i].data, rms_norm_eps)

            # Attention
            qkv = torch.matmul(hidden_states, self.qkv_projs[i].data)
            q, k, v = torch.split(qkv, [num_attention_heads * head_dim, num_key_value_heads * head_dim, num_key_value_heads * head_dim], dim=-1)
            q = q.view(batch_size, seqlen, num_attention_heads, head_dim)
            k = k.view(batch_size, seqlen, num_key_value_heads, head_dim)
            v = v.view(batch_size, seqlen, num_key_value_heads, head_dim)

            q = torch_npu.npu_rms_norm(q, self.q_norm[i].data, rms_norm_eps)[0]
            k = torch_npu.npu_rms_norm(k, self.k_norm[i].data, rms_norm_eps)[0]

            if inference_mode == "dynamo":
                # rope torch version
                q = torch_npu.npu_rotary_mul(q, cos, sin)
                k = torch_npu.npu_rotary_mul(k, cos, sin)
            else:
                # rope triton version
                q = torch.ops.my_ops.apply_rotary_emb_triton_block(q.squeeze(0), cos, sin).unsqueeze(0)
                k = torch.ops.my_ops.apply_rotary_emb_triton_block(k.squeeze(0), cos, sin).unsqueeze(0)

            # batch, head_cnt, seq_len, head_dim
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)

            k = torch_npu.scatter_update_(kv_cache[i][0], positions, k, axis=-2)
            v = torch_npu.scatter_update_(kv_cache[i][1], positions, v, axis=-2)

            if is_prefill:
                o = torch_npu.npu_fused_infer_attention_score(
                    q, k, v,
                    input_layout="BNSD",
                    num_heads=self.config.num_attention_heads,
                    num_key_value_heads=self.config.num_key_value_heads,
                    scale=self.attention_scale,
                    sparse_mode=2,
                    atten_mask=self.atten_mask.data,
                    next_tokens=0,
                    actual_seq_lengths_kv=[seqlen,],
                )[0]
            else:
                o = torch_npu.npu_fused_infer_attention_score(
                    q, k, v,
                    input_layout="BNSD",
                    num_heads=self.config.num_attention_heads,
                    num_key_value_heads=self.config.num_key_value_heads,
                    actual_seq_lengths_kv=actual_seq_lengths_kv,
                    scale=self.attention_scale,
                )[0]

            o = o.permute(0, 2, 1, 3).reshape(batch_size, seqlen, -1)
            hidden_states = torch.matmul(o, self.o_projs[i].data)

            # Post Attention Norm
            hidden_states, residual = add_rms_norm(hidden_states, residual, self.post_attention_norm[i].data, rms_norm_eps)

            # MLP
            hidden_states_gate, hidden_states_up = torch.matmul(hidden_states, self.mlp_gate_up_proj[i].data).chunk(2, dim=-1)
            hidden_states = F.silu(hidden_states_gate) * hidden_states_up
            hidden_states = torch.matmul(hidden_states, self.mlp_down_proj[i].data)

        hidden_states, _ = add_rms_norm(hidden_states, residual, self.model_norm.data, rms_norm_eps)

        # Compute Logits
        hidden_states = (hidden_states[:, -1, :]).squeeze(1)
        logits = torch.matmul(hidden_states, self.lm_head.data)

        # greedy sample
        return logits.argmax(dim=-1)
