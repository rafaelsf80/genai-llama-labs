""" Sample script to train a Llama 2 7B model with QLoRA
    Tuning takes around 1h with a T4 GPU in Colab
    If OOM error, then reduce `per_device_train_batch_size` to 1
"""

#!pip install -U accelerate peft bitsandbytes transformers trl datasets

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

# The model that you want to train from the Hugging Face hub
model_name = "NousResearch/Llama-2-7b-chat-hf"

# Dataset
dataset_name = "mlabonne/guanaco-llama2-1k"
dataset = load_dataset(dataset_name, split="train")

# Load tokenizer and model with QLoRA configuration
#compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_compute_dtype=torch.float16, 
    bnb_4bit_use_double_quant=False, 
)

# Load the entire model on the GPU 0
device_map = {"": 0}

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map
)

# Parametros específicos de Llama
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/configuration_llama.py
model.config.use_cache = False # do not use cache from attention layers
#model.config.pretraining_tp = 1


# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training ERROR PUSE padding_size


# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=16, 
    lora_dropout=0.1, 
    r=64, 
    bias="none",
    task_type="CAUSAL_LM",
)


# Set SFT training parameters
training_arguments = TrainingArguments(
    output_dir='/qlora_model',
    num_train_epochs=1, 
    per_device_train_batch_size=4, # OOM ERROR. BATCH_SIZE=1 WILL FIT IN T4 GPU IN COLAB
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=0,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001, 
    fp16=False, 
    bf16=False, 
    max_grad_norm=0.3,
    max_steps=-1, 
    warmup_ratio=0.03, 
    group_by_length=True, 
    lr_scheduler_type= "cosine", 
    #report_to="tensorboard"
)

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    #max_seq_length=None, ## Maximum sequence length to use
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False, # Pack multiple short examples in the same input sequence to increase efficiency
)

# Train model
trainer.train()

# Save trained model
new_model = "my_qlora_layers"
trainer.model.save_pretrained(new_model)

# Empty VRAM
del model
del trainer
import gc
gc.collect()
gc.collect()

# Reload model in FP16 and merge it with LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)
model = PeftModel.from_pretrained(base_model, "/content/my_qlora_layers")  # Replace if not Colab
model = model.merge_and_unload()

# Run text generation pipeline with our next model
prompt = "What is a large language model?"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])