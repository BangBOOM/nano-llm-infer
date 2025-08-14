import json
from dataclasses import fields

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from model import Qwen3, Qwen3Config

base_path = "/Users/bangboom/Documents/models/Qwen3-0.6B/"
config_path = base_path + "config.json"
model_path = base_path + "model.safetensors"

tokenizer = AutoTokenizer.from_pretrained(base_path)
with open(config_path, "r", encoding="utf-8") as f:
    qwen3_config = Qwen3Config(**{k:v for k,v in json.load(f).items() if k in [field.name for field in fields(Qwen3Config)]})

model = Qwen3(qwen3_config).bfloat16()
model.load_weight(model_path)
generated_token = ""
prompts = ["list all prime numbers within 100"]
prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        for prompt in prompts
    ]
for _ in tqdm(range(100)):
    prompts[0] += generated_token

    input_ids = tokenizer(prompts, return_tensors="pt")["input_ids"]
    positions = torch.tensor(list(range(input_ids.shape[-1])), dtype=torch.int)
    predict_tokens = model(input_ids, positions)
    generated_token = tokenizer.decode(predict_tokens)

print(prompts[0])
