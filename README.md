# NANO-LLM-Inference

Development efficiency is the first citizen in this project

## v0.0.1
no kvcache, recompute during each inference step

process one batch a time

## TODO
1. PageAttention KVCache Support
2. Batch Prefill
3. Large model support
4. Distributed Inference with TP
5. Distributed Inference with EP

## Usage

1. init the environment with uv
2. download Qwen3 0.6B model from huggingface
3. edit `main.py` change the path of base path to the model
```python
from model import Qwen3, Qwen3Config

base_path = "xxx/Qwen3-0.6B/"
config_path = base_path + "config.json"
model_path = base_path + "model.safetensors"
```
