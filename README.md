# NANO-LLM-Inference

Development efficiency is the first citizen in this project

## v0.0.2
use kvcache, do not need to recompute the previous tokens

only process one batch a time

## v0.0.1
no kvcache, recompute during each inference step

process one batch a time

## TODO

### Stage 1
1. Support multi round chat
2. Support multi batch inference: a sequence manager is needed to manage the position of a sequence and kvcache

### Stage 2
1. PageAttention KVCache Support

### Stage 3
1. Distributed Inference with TP
2. Distributed Inference with EP

## Usage

1. init the environment with uv
2. download Qwen3 0.6B model from huggingface
3. edit `main.py` change the path of base path to the model, currently only support Qwen3 Dense Model.
```python
from model import Qwen3, Qwen3Config

base_path = "xxx/Qwen3-0.6B/"
config_path = base_path + "config.json"
model_path = base_path + "model.safetensors"
```
