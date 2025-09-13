import json
import logging
import os

import ray
import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from models.npu.utils import ParallelCommunicationGroup, ParallelConfig

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("ModelRunner")

@ray.remote(resources={"NPU": 1})
class ModelRunner:
    def __init__(self, rank, model_path, config_class, model_class, parallel_config:ParallelConfig, max_seqlen: int, bsz: int=1, inference_mode:str= "eager") -> None:
        assert inference_mode in ["eager", "dynamo"], "Invalid inference mode"
        torch.npu.set_device(f"npu:{rank}")
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        self.max_seqlen = max_seqlen
        self.bsz = bsz
        self.inference_mode = inference_mode
        self.parallel_config = parallel_config
        self.rank = rank
        # init dist communication
        dist.init_process_group(backend="hccl", world_size=parallel_config.world_size, rank=rank)
        self.parallel_com_group = ParallelCommunicationGroup(
            ep_group=dist.new_group(parallel_config.get_ep_group(rank)) if parallel_config.ep else None,
            dp_group=dist.new_group(parallel_config.get_dp_group(rank)) if parallel_config.dp > 1 else None,
            tp_group=dist.new_group(parallel_config.get_tp_group(rank)) if parallel_config.tp > 1 else None
        )

        # init model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        with open(os.path.join(model_path, "config.json"), encoding="utf-8") as f:
            self.config = config_class.from_dict(json.load(f))
        # DEBUG
        # self.config.num_hidden_layers = 2

        self.model = model_class(self.config, self.parallel_config, self.parallel_com_group).bfloat16().npu().eval()
        self.model.load_weight(model_path)

        # init cache
        self.rope_cache = get_rope_cache(self. config. head_dim, self.max_seqlen, self.config.rope_theta)
        logger.info("Initing KV cache...")
        self.kv_cache = [
            (
                torch.zeros((self.config.num_heads, self.max_seqlen, self.config.head_dim), dtype=torch.bfloat16, device="npu"),
                torch.zeros((self.config.num_heads, self.max_seqlen, self.config.head_dim), dtype=torch.bfloat16, device="npu")
            )
            for _ in range(self.config.num_hidden_layers)
        ]
        logger.info("KV cache inited!!!")
        if self.inference_mode == "eager":
            self.decoder = self.model.forward
        elif self.inference_mode == "dynamo":
            # must import torchair after import ray
            import torchair
            config = torchair.CompilerConfig()
            config.node = "reduce-overhead"
            npu_backend = torchair.get_npu_backend(compiler_config=config)
            self.decoder = torch.compile(self.model.forward, backend=npu_backend)
        elif self.inference_mode == "torch":
            self.decoder = self.model.forward

    def generate(self, prompts:list[str], ignore_eos:bool=False, enable_thinking:bool=False, is_stream:bool=False):
        def stream_print(msg):
            if is_stream and self.rank == 0:
                print(msg, end="", flush=True)

        # currently length of prompts must be 1
        prompts = [
            self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
            for prompt in prompts
        ]
        input_ids = self.tokenizer(prompts[0], return_tensors="pt")["input_ids"]
        # do prefill
        positions = list(range(input_ids.shape[-1]))
        cos, sin = self.rope_cache[positions].chunk(2, dim=-1)
        predict_token = self.model(input_ids.npu(), cos.npu(), sin.npu(),
            torch.zeros(1, dtype=torch.int32, device="npu"), # prefill start position is 0
            self.kv_cache, is_prefill=True
        )
        generated_token = self.tokenizer.decode(predict_token)
        res = generated_token

        dist.barrier()
        torch.npu.synchronize(0)
        logger.info("Done Prefi11")
        stream_print(res)
        position_id = positions[-1]
        start_event = torch.npu.Event(enable_timing=True)
        end_event = torch.npu.Event(enable_timing=True)
        step = 0
        while position_id < self.max_seqlen - 1:
            position_id += 1
            output = [generated_token]
            input_ids = self.tokenizer(output, return_tensors="pt")["input_ids"]
            positions = [position_id,]
            cos, sin = self.rope_cache[positions].chunk(2, dim=-1)
            actual_seq_lengths_kv = [position_id+1,]
            inputs = {
                "input_ids": input_ids.npu(),
                "cos": cos.npu(),
                "sin": sin.npu(),
                "positions": torch.tensor(positions, dtype=torch.int32).npu(),
                "kv_cache": self.kv_cache,
                "actual_seq_lengths_kv": actual_seq_lengths_kv,
                "inference_mode": self.inference_mode
            }
            if self.inference_mode == "dynamo":
                torch._dynamo.mark_static(inputs["input_ids"])
                torch._dynamo.mark_static(inputs["cos"])
                torch._dynamo.mark_static(inputs["sin"])
                torch._dynamo.mark_static(inputs["positions"])

            start_event.record()
            predict_tokens = self.decoder(**inputs)
            end_event.record()
            generated_token = self.tokenizer.decode(predict_tokens)
            torch.npu.synchronize()
            elapsed_time_ms = start_event.elapsed_time(end_event)
            logger.info(f"Step ({step}) Time elapsed: {elapsed_time_ms:.2f} ms")
            res += generated_token
            if not ignore_eos and predict_tokens == self.tokenizer.eos_token_id:
                break
            stream_print(generated_token)
            step += 1
        stream_print ("\n")
        logger.info("Done Decoding!!!")
        return res
