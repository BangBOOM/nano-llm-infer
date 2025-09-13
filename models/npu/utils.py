from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch_npu


@dataclass
class ParallelConfig:
    ep: int
    dp: int
    tp: int
    world_size: int

    def get_ep_group(self, rank):
        return list(range(self.ep))

    def get_dp_group(self, rank):
        return list(range(self.dp))

    def get_tp_group(self, rank):
        return list(range(self.tp))


class ParallelCommunicationGroup:
    def __init__(self, dp_group=None, tp_group=None, ep_group=None):
        self.dp_group = dp_group
        self.tp_group = tp_group
        self.ep_group = ep_group

    def get_ep_idx(self, rank):
        return dist.get_rank(group=self.ep_group) if self.ep_group else 0

    def get_tp_idx(self, rank):
        return dist.get_rank(group=self.tp_group) if self.tp_group else 0

    def get_dp_idx(self, rank):
        return dist.get_rank(group=self.dp_group) if self.dp_group else 0

    def gep_hcomm_info(self, rank):
        assert self.ep_group is not None, "ep_group is not initialized"
        return self.ep_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)

def add_rms_norm(x, residual, weight, eps) -> tuple[torch.Tensor, torch.Tensor]:
    if residual is not None:
        x, _, residual = torch_npu.npu_add_rms_norm(residual, x, weight, eps)
    else:
        residual = x
        x = torch_npu.npu_rms_norm(x, weight, eps)[0]
    return x, residual
