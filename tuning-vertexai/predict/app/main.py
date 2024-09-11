""" FastAPI app that serves AIP_PREDICT_ROUTE and calls a Llama 2  model stored locally and downloaded from GCS
    Healthcheck at AIP_HEALTH_ROUTE
"""

import os
import shutil
import logging

from tqdm import tqdm
from google.cloud import storage
from google.cloud import aiplatform
from fastapi import FastAPI, Request
from fastapi.logger import logger

from starlette.responses import JSONResponse
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM

import torch


app = FastAPI()

PROJECT_ID = "argolis-rafaelsanchez-ml-dev"
AIP_PROJECT_NUMBER=os.getenv("AIP_PROJECT_NUMBER")
AIP_PREDICT_ROUTE=os.getenv("AIP_PREDICT_ROUTE", "/predict")
AIP_HEALTH_ROUTE=os.getenv("AIP_HEALTH_ROUTE", "/health")
AIP_STORAGE_URI=os.getenv("AIP_STORAGE_URI")
LOCAL_MODEL_DIR="llama-2-qlora/"


gunicorn_logger = logging.getLogger('gunicorn.error')
logger.handlers = gunicorn_logger.handlers

if __name__ != "main":
    logger.setLevel(gunicorn_logger.level)
else:
    logger.setLevel(logging.INFO)

logger.info(f"Is CUDA available: {torch.cuda.is_available()}")
logger.info(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
device = "cuda:0" if torch.cuda.is_available() else "cpu"

torch.cuda.empty_cache()

logger.info(f"Model stored at {AIP_STORAGE_URI}")

os.mkdir(LOCAL_MODEL_DIR)

aiplatform.init(project=PROJECT_ID)

storage_client = storage.Client(AIP_PROJECT_NUMBER)
bucket = storage_client.bucket(AIP_STORAGE_URI.split("/")[2])
blobs = bucket.list_blobs(prefix=AIP_STORAGE_URI.split("/")[3])
for blob in blobs:
    logger.info(f"Free Disk: {shutil.disk_usage(__file__)[2]/1024/1024/1024}")
    logger.info(blob.name.split("/")[-1])
    if blob.name.split("/")[-1] != "":
        filename = blob.name.split("/")[-1]
        with open(LOCAL_MODEL_DIR+filename, "wb") as in_file:
            with tqdm.wrapattr(in_file, "write", total=blob.size, miniters=1, desc="Downloading") as destination_file_name:
                storage_client.download_blob_to_file(blob, destination_file_name) 

logger.info(f"Loading model {LOCAL_MODEL_DIR}. This takes some time ...")

tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR, use_fast=True)

model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_DIR)
#model = model.to(device)

logger.info(f"Loading model DONE")

    
@app.get(AIP_HEALTH_ROUTE, status_code=200)
def health():
    return dict(status="healthy")

@app.post(AIP_PREDICT_ROUTE)
async def predict(request: Request, status_code=200):
    body = await request.json()
    prompt = body["instances"]
    
    system_message = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    prompt_template=f'''[INST] <<SYS>>
    {system_message}
    <</SYS>>

    {prompt} [/INST]'''

    inputs = tokenizer(prompt_template, return_tensors='pt').input_ids#.to(device)#cuda()
    generated_ids = model.generate(inputs=inputs, temperature=0.7, max_new_tokens=254).to(device)
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    return JSONResponse({"predictions": response})