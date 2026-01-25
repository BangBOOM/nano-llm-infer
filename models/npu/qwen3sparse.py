import os
import glob
import json
import logging
import math
import random
from dataclasses import dataclass

import torch
import torch_npu
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from tqdm import tqdm

from models.npu.utils import ParallelCommunicationGroup, ParallelConfig
from models.npu.utils import add_rms_norm

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("Qwen3MoE")

@dataclass
class Qwen3MoEConfig:
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
    # MoE Configuration
    moe_intermediate_size: int
    num_experts: int
    num_experts_per_tok: int
    norm_topk_prob: bool



class Qwen3MoE(nn.Module):
    def __init__(self, config:Qwen3MoEConfig, parallel_config:ParallelConfig, parallel_comm:ParallelCommunicationGroup) -> None:
        super().__init__()
        self.rank = dist.get_rank()
        self.config = config
        self.parallel_config = parallel_config
        self.parallel_comm = parallel_comm
        self.ep_hcomm_info = parallel_comm.ep_hcomm_info(self.rank)
        self.local_num_experts = config.num_experts // parallel_config.ep
        self.embed_tokens = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.hidden_size, _freeze=True)
        self.input_layernorm = nn.ParameterList([nn.Parameter(torch.empty(config.hidden_size)) for _ in range(config.num_hidden_layers)])

        self.qkv_projs = nn.ParameterList([nn.Parameter(torch.empty(config.hidden_size, config.num_attention_heads * config.head_dim + 2 * config.num_key_value_heads * config.head_dim)) for _ in range(config.num_hidden_layers)])
        self.o_projs = nn.ParameterList([nn.Parameter(torch.empty(config.num_attention_heads*config.head_dim, config.hidden_size)) for _ in range(config.num_hidden_layers)])

        self.q_norm = nn.ParameterList([nn.Parameter(torch.empty(config.head_dim)) for _ in range(config.num_hidden_layers)])
        self.k_norm = nn.ParameterList([nn.Parameter(torch.empty(config.head_dim)) for _ in range(config.num_hidden_layers)])

        self.post_attention_norm = nn.ParameterList([nn.Parameter(torch.empty(config.hidden_size)) for _ in range(config.num_hidden_layers)])
        self.gates = nn.ParameterList([nn.Parameter(torch.empty(config.hidden_size, config.num_experts)) for _ in range(config.num_hidden_layers)])
        self.mlp_gate_up_projs = nn.ParameterList(
            [
                nn.ParameterList(
                    [
                        nn.Parameter(torch.empty(config.hidden_size, 2 * config.moe_intermediate_size // self.parallel_config.tp))
                        for _ in range(self.local_num_experts)
                    ]
                )
                for _ in range(config.num_hidden_layers)
            ]
        )

        self.mlp_down_projs = nn.ParameterList(
            [
                nn.ParameterList(
                    [
                        nn.Parameter(torch.empty(config.moe_intermediate_size // self.parallel_config.tp, config.hidden_size))
                        for _ in range(self.local_num_experts)
                    ]
                )
                for _ in range(config.num_hidden_layers)
            ]
        )

        self.model_norm = nn.Parameter(torch.empty(config.hidden_size))
        self.lm_head = nn.Parameter(torch.empty(config.hidden_size, config.vocab_size))

        self.atten_mask = nn.Parameter(~torch.tril(torch.ones(2048, 2048, dtype=torch.bool)).unsqueeze(0), requires_grad=False)
        self.attention_scale = 1/math.sqrt(config.head_dim)
        self.expert_offset = self.local_num_experts * self.parallel_comm.get_ep_idx()
        logger.info(f"Expert Offset {self.expert_offset}")

    def load_weight(self, model_path):
        with open(os.path.join(model_path, "model.safetensors.index.json")) as f:
            weight_map = json.load(f)["weight_map"]

        safe_tensor_map = dict()
        for filename in glob.glob(os.path.join(model_path, "*.safetensors")):
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

        layers = list(range(self.config.num_hidden_layers))
        random.shuffle(layers)
        for i in tqdm(layers):
            self.input_layernorm[i].data.copy_(get_tensor(f"model.layers.{i}.input_layernorm.weight"))

            q = get_tensor(f"model.layers.{i}.self_attn.q_proj.weight").permute(1, 0)
            k = get_tensor(f"model.layers.{i}.self_attn.k_proj.weight").permute(1, 0)
            v = get_tensor(f"model.layers.{i}.self_attn.v_proj.weight").permute(1, 0)
            self.qkv_projs[i].data.copy_(torch.cat([q, k, v], dim=-1))
            self.o_projs[i].data.copy_(get_tensor(f"model.layers.{i}.self_attn.o_proj.weight").permute(1, 0))

            self.q_norm[i].data.copy_(get_tensor(f"model.layers.{i}.self_attn.q_norm.weight"))
            self.k_norm[i].data.copy_(get_tensor(f"model.layers.{i}.self_attn.k_norm.weight"))

            self.post_attention_norm[i].data.copy_(get_tensor(f"model.layers.{i}.post_attention_layernorm.weight"))
            self.gates[i].data.copy_(get_tensor(f"model.layers.{i}.mlp.gate.weight").permute(1, 0))

            # cannot use tp and ep together
            if self.parallel_config.ep > 1:
                for j in range(self.local_num_experts):
                    expert_idx = self.expert_offset + j
                    mlp_gate_proj = get_tensor(f"model.layers.{i}.mlp.experts.{expert_idx}.gate_proj.weight").permute(1, 0)
                    mlp_up_proj = get_tensor(f"model.layers.{i}.mlp.experts.{expert_idx}.up_proj.weight").permute(1, 0)
                    self.mlp_gate_up_projs[i][j].data.copy_(torch.cat([mlp_gate_proj, mlp_up_proj], dim=-1))
                    self.mlp_down_projs[i][j].data.copy_(get_tensor(f"model.layers.{i}.mlp.experts.{expert_idx}.down_proj.weight").permute(1, 0))
            elif self.parallel_config.tp > 1:
                for j in range(self.config.num_experts):
                    expert_idx = self.expert_offset + j
                    local_moe_intermediate_size = self.config.moe_intermediate_size // self.parallel_config.tp
                    tp_idx = self.parallel_comm.get_tp_idx()
                    mlp_gate_proj = get_tensor(f"model.layers.{i}.mlp.experts.{expert_idx}.gate_proj.weight").permute(1, 0)[:, local_moe_intermediate_size * tp_idx: local_moe_intermediate_size * (tp_idx+1)]
                    mlp_up_proj = get_tensor(f"model.layers.{i}.mlp.experts.{expert_idx}.up_proj.weight").permute(1, 0)[:, local_moe_intermediate_size * tp_idx: local_moe_intermediate_size * (tp_idx+1)]
                    self.mlp_gate_up_projs[i][j].data.copy_(torch.cat([mlp_gate_proj, mlp_up_proj], dim=-1))
                    self.mlp_down_projs[i][j].data.copy_(get_tensor(f"model.layers.{i}.mlp.experts.{expert_idx}.down_proj.weight").permute(1, 0)[local_moe_intermediate_size * tp_idx: local_moe_intermediate_size * (tp_idx+1), :])

        dist.barrier()
        torch.npu.synchronize()

    @torch.no_grad()
    def forward(
        self,
        input_ids:torch.Tensor,
        cos:torch.Tensor,
        sin:torch.Tensor,
        positions:torch.Tensor,
        kv_cache:list,
        actual_seq_lengths_kv:list[int]|None=None,
        is_prefill=False,
        inference_mode="eager"
    ):
        rms_norm_eps = self.config.rms_norm_eps
        hidden_states = self.embed_tokens(input_ids)
        residual = None

        batch_size, seqlen, _ = hidden_states.shape
        assert batch_size == 1, "Currently only support singual request"

        head_dim = self.config.head_dim
        hidden_size = self.config.hidden_size
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
                    num_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
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
                    num_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                    actual_seq_lengths_kv=actual_seq_lengths_kv,
                    scale=self.attention_scale,
                )[0]

            o = o.permute(0, 2, 1, 3).reshape(batch_size, seqlen, -1)
            hidden_states = torch.matmul(o, self.o_projs[i].data)

            # Post Attention Norm
            hidden_states, residual = add_rms_norm(hidden_states, residual, self.post_attention_norm[i].data, rms_norm_eps)


            top_k = self.config.num_experts_per_tok
            hidden_states = hidden_states.view(-1, hidden_size)
            router_logits = torch.matmul(hidden_states, self.gates[i].data)

            routing_weights, selected_experts, _ = torch_npu.npu_moe_gating_top_k_softmax(router_logits, k=top_k)
            if self.config.norm_topk_prob:
                routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

            # USE EP
            if is_prefill:
                flat_expert_ids = selected_experts.flatten()
                flat_tokens = hidden_states.unsqueeze(1).expand(-1, top_k, -1).flatten(0, 1) # [n_tokens *top_k, hidden_size]
                target_devices = flat_expert_ids // self.local_num_experts
                sorted_indices = torch.argsort(target_devices)
                sorted_tokens = flat_tokens[sorted_indices]
                sorted_expert_ids = flat_expert_ids[sorted_indices]

                send_counts = torch.bincount(target_devices, minlength=self.parallel_config.ep)
                recv_counts = torch.zeros_like(send_counts)
                dist.all_to_all_single(recv_counts, send_counts, group=self.parallel_comm.ep_group)

                total_recv = recv_counts.sum()
                recv_tokens = torch.empty(total_recv, hidden_size, dtype=hidden_states.dtype, device=hidden_states.device)
                recv_expert_ids = torch.empty(total_recv, dtype=flat_expert_ids.dtype, device=flat_expert_ids.device)
                recv_counts_list = recv_counts.tolist()
                send_counts_list = send_counts.tolist()
                dist.all_to_all_single(recv_tokens, sorted_tokens, output_split_sizes=recv_counts_list, input_split_sizes=send_counts_list, group=self.parallel_comm.ep_group)
                dist.all_to_all_single(recv_expert_ids, sorted_expert_ids, output_split_sizes=recv_counts_list, input_split_sizes=send_counts_list, group=self.parallel_comm.ep_group)

                recv_sorted_slice = torch.argsort(recv_expert_ids)
                sorted_recv_tokens = recv_tokens[recv_sorted_slice]
                expert_token_cnt = torch.bincount(recv_expert_ids-self.expert_offset, minlength=self.local_num_experts)
                expert_token_cusum = torch.cumsum(expert_token_cnt, dim=-1)

                hidden_states_gate, hidden_states_up = torch_npu.npu_grouped_matmul(
                        x=[sorted_recv_tokens],
                        weight=[item.data for item in self.mlp_gate_up_projs[i]],
                        group_list=expert_token_cusum,
                        split_item=2,
                        group_list_type=0,
                        group_type=0
                )[0].chunk(2, dim=-1)

                hidden_states = F.silu(hidden_states_gate) * hidden_states_up

                sorted_recv_tokens = torch_npu.npu_grouped_matmul(
                    x=[hidden_states],
                    weight=[item.data for item in self.mlp_down_projs[i]],
                    group_list=expert_token_cusum,
                    split_item=2,
                    group_list_type=0,
                    group_type=0
                )[0]

                # Combine
                recv_tokens[recv_sorted_slice] = sorted_recv_tokens
                dist.all_to_all_single(
                    sorted_tokens, recv_tokens,
                    output_split_sizes=send_counts_list,
                    input_split_sizes=recv_counts_list,
                    group=self.parallel_comm.ep_group
                )

                flat_tokens[sorted_indices] = sorted_tokens
                hidden_states = (flat_tokens.view(-1, top_k, hidden_size) * routing_weights.unsqueeze(-1)).sum(-2)
                hidden_states = hidden_states.view(-1, seqlen, hidden_size)
            else:
                expand_x, _, assist_info_for_combine, expert_token_nums, ep_recv_counts, tp_recv_counts, _ = torch_npu.npu_moe_distribute_dispatch_v2(
                    x=hidden_states,
                    expert_ids=selected_experts,
                    group_ep=self.ep_hcomm_info,
                    ep_world_size=self.parallel_config.ep,
                    ep_rank_id=self.parallel_comm.get_ep_idx(),
                    moe_expert_num=self.config.num_experts,
                    expert_token_nums_type=0    # cumsum
                )
                hidden_states_gate, hidden_states_up = torch_npu.npu_grouped_matmul(
                        x=[expand_x],
                        weight=[item.data for item in self.mlp_gate_up_projs[i]],
                        group_list=expert_token_nums,
                        split_item=2,
                        group_list_type=0,
                        group_type=0
                )[0].chunk(2, dim=-1)

                expand_x = F.silu(hidden_states_gate) * hidden_states_up

                expand_x = torch_npu.npu_grouped_matmul(
                    x=[expand_x],
                    weight=[item.data for item in self.mlp_down_projs[i]],
                    group_list=expert_token_nums,
                    split_item=2,
                    group_list_type=0,
                    group_type=0
                )[0]

                hidden_states = torch_npu.npu_moe_distribute_combine_v2(
                    expand_x=expand_x,
                    expert_ids=selected_experts,
                    assist_info_for_combine=assist_info_for_combine,
                    ep_send_counts=ep_recv_counts,
                    tp_send_counts=tp_recv_counts,
                    expert_scales=routing_weights.float(),
                    group_ep=self.ep_hcomm_info,
                    ep_world_size=self.parallel_config.ep,
                    ep_rank_id=self.parallel_comm.get_ep_idx(),
                    moe_expert_num=self.config.num_experts,
                )
                hidden_states = hidden_states.view(batch_size, 1, -1)

        hidden_states, _ = add_rms_norm(hidden_states, residual, self.model_norm.data, rms_norm_eps)

        # Compute Logits
        hidden_states = (hidden_states[:, -1, :]).squeeze(1)
        logits = torch.matmul(hidden_states, self.lm_head.data)

        # greedy sample
        return logits.argmax(dim=-1)
