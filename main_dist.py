from llm_engine import LLMEngine
from models.npu.qwen3dense import Qwen3, Qwen3Config


def main(npus, model_path, max_seqlen):
    engine = LLMEngine(
        world_size=npus,
        dp=npus,
        ep=1,
        tp=1,
        model_path=model_path,
        model_class=Qwen3,
        config_class=Qwen3Config,
        max_seqlen=max_seqlen,
        bsz=1,
        # inference_mode="dynamo"
    )
    prompts = ["What is the weather like today?"] * npus
    results = engine.generate(prompts)
    print(results)

if __name__ == "__main__":
    main(npus=1, model_path="path/to/model", max_seqlen=1024)
