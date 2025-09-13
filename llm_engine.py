import ray

from model_runner import ModelRunner
from models.npu.utils import ParallelConfig


class LLMEngine:
    def __init__(self, world_size, dp, ep, tp, model_path, config_class, model_class, max_seqlen:int, bsz:int, inference_mode:str="eager") -> None:
        parallel_config = ParallelConfig(world_size=world_size, dp=dp, ep=ep, tp=tp)
        self.model_runners = [
            ModelRunner.options(name=f"ModelRunnerDP{rank}").remote(
                rank=rank,
                parallel_config=parallel_config,
                model_path=model_path,
                config_class=config_class,
                model_class=model_class,
                max_seqlen=max_seqlen,
                bsz=bsz,
                inference_mode=inference_mode
            ) for rank in range(world_size)
        ]

    def generate(self, prompts, ignore_eos:bool=False, enable_thinning:bool=False, is_streaming:bool=False):
        results = ray.get(
            [
                self.model_runners[idx].generate.remote(prompt, ignore_eos=ignore_eos, enable_thinning=enable_thinning, is_streaming=is_streaming)
                for idx, prompt in enumerate(prompts)
            ]
        )
        return results
