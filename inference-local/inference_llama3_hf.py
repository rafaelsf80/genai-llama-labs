""" Simple local inference of Llama 3-8B instruct model, using 4-bit model
    Requires T4 GPU in Colab
    Requires to accept license (gated repo) on Hugging Face
    https://huggingface.co/blog/llama3
"""
#!pip3 install -U accelerate transformers bitsandbytes
# RESTART SESSION to get latest bitsandbytes package

import transformers
import torch

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={
        "torch_dtype": torch.float16,
        "quantization_config": {"load_in_4bit": True},
        "low_cpu_mem_usage": True,
    },
    token = ""  # <---- Insert your HF_TOKEN HERE
)

messages = [
    {"role": "system", "content": "Eres un sevillano que habla en lenguaje andaluz"},
    {"role": "user", "content": "Quien eres?"},
]

prompt = pipeline.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
)

# As per: https://huggingface.co/blog/llama3
# Assistant responses may end with the special token <|eot_id|>, but we must also stop generation if the regular EOS token is found. We can stop generation early by providing a list of terminators in the eos_token_id parameter.
terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipeline(
    prompt,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)

print(outputs[0]["generated_text"][len(prompt):])