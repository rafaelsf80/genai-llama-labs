""" Sample code to train a Llama 2-7B-chat model with QLoRA
    Uploads final merged model to Vertex AI Model Registry
    Must run on a V100, T4, L4 or above
"""

import glob
import json
import logging
import os

import datasets
import transformers
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

from google.cloud import storage


logging.info(f"Runtime: {'GPU' if torch.cuda.is_available() else 'CPU'}")
logging.info(f"PyTorch version : {torch.__version__}")
logging.info(f"Transformers version : {transformers.__version__}")
logging.info(f"Datasets version : {datasets.__version__}")

print(f"Notebook runtime: {'GPU' if torch.cuda.is_available() else 'CPU'}")
print(f"PyTorch version : {torch.__version__}")
print(f"Transformers version : {transformers.__version__}")
#print(f"Datasets version : {datasets.__version__}")
output_directory = os.environ['AIP_MODEL_DIR']
print(f"AIP_MODEL_DIR: {output_directory}")

# Model to finetune
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
model.config.use_cache = False
model.config.pretraining_tp = 1


# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training


# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=16, 
    lora_dropout=0.1,
    r=16, 
    bias="none",
    task_type="CAUSAL_LM",
)


# Set SFT training parameters
training_arguments = TrainingArguments(
    output_dir='/qlora_model',
    num_train_epochs=1, 
    per_device_train_batch_size=1, 
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
    report_to="tensorboard"
)

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=None, ## Maximum sequence length to use
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False, # Pack multiple short examples in the same input sequence to increase efficiency
)

# Train model
logging.info("Starting training ....")
trainer.train()

# Save trained model
logging.info("Saving LoRA layers ....")
new_model = "qlora_layers"
trainer.model.save_pretrained(new_model)


# Empty VRAM, avoids OOM
del model
del trainer
import gc
gc.collect()
gc.collect()

# Reload model in FP16 and MERGE it with LoRA weights
logging.info("Merging layers ....")
print("Merging layers ....")

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    temperature=0.1,
    do_sample=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)
model = PeftModel.from_pretrained(base_model, "./qlora_layers")
model = model.merge_and_unload()


# Run text generation pipeline with our next model
prompt = "What is a large language model?"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
result = pipe(f"<s>[INST] {prompt} [/INST]")
logging.info(result[0]['generated_text'])
print(result[0]['generated_text'])




# Save tokenizer, metrics and model locally
# TODO: check if this is LoRA layers only or full model
logging.info("Saving final model and tokenizer locally ....")
tokenizer.save_pretrained(f'model_tokenizer')
model.save_pretrained(f'model_output')

# TODO: METRICS
#logging.info('Saving metrics...')
#with open(os.path.join(f'model_output', 'metrics.json'), 'w') as f:
#    json.dump(metrics, f, indent=2)

output_directory = os.environ['AIP_MODEL_DIR']
logging.info("Saving model and tokenizer to GCS ....")
logging.info(f'Exporting SavedModel to: {output_directory}')

# extract GCS bucket_name from AIP_MODEL_DIR, ex: argolis-vertex-europewest4
bucket_name = output_directory.split("/")[2] # without gs://

# extract GCS object_name from AIP_MODEL_DIR, ex: aiplatform-custom-training-2023-02-22-16:31:12.167/model/
object_name = "/".join(output_directory.split("/")[3:])

directory_path = "model_output" # local directory
# Upload model to GCS
client = storage.Client()
rel_paths = glob.glob(directory_path + '/**', recursive=True)
bucket = client.get_bucket(bucket_name)
for local_file in rel_paths:
    remote_path = f'{object_name}{"/".join(local_file.split(os.sep)[1:])}'
    logging.info(remote_path)
    if os.path.isfile(local_file):
        blob = bucket.blob(remote_path)
        blob.upload_from_filename(local_file)


directory_path = "model_tokenizer" # local directory
# Upload tokenizer to GCS
client = storage.Client()
rel_paths = glob.glob(directory_path + '/**', recursive=True)
bucket = client.get_bucket(bucket_name)
for local_file in rel_paths:
    remote_path = f'{object_name}{"/".join(local_file.split(os.sep)[1:])}'
    logging.info(remote_path)
    if os.path.isfile(local_file):
        blob = bucket.blob(remote_path)
        blob.upload_from_filename(local_file)

# TODO: Upload metrics to Vertex AI Experiments
