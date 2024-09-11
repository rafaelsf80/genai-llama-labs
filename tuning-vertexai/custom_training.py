""" Custom training pipeline for Llama 2-7B-chat (PEFT), with script located at 'trainer.py'
"""

from datetime import datetime
from google.cloud import aiplatform

BUCKET = 'gs://argolis-vertex-europewest4'
PROJECT_ID = 'argolis-rafaelsanchez-ml-dev'
LOCATION = 'europe-west4'
SERVICE_ACCOUNT = 'tensorboard-sa@argolis-rafaelsanchez-ml-dev.iam.gserviceaccount.com'
TENSORBOARD_RESOURCE_NAME = 'projects/989788194604/locations/europe-west4/tensorboards/8884581718011412480'
TRAINING_IMAGE="europe-west4-docker.pkg.dev/argolis-rafaelsanchez-ml-dev/ml-pipelines-repo/llama2-qlora-peft:latest"
TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")

aiplatform.init(project=PROJECT_ID, staging_bucket=BUCKET, location=LOCATION)

job = aiplatform.CustomContainerTrainingJob(
    display_name="llama2_guanaco_qlora_gpu_" + TIMESTAMP,
    container_uri=TRAINING_IMAGE,
    command=["python3", "trainer.py"],
    model_serving_container_image_uri="europe-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.1-12:latest"
)
 

model = job.run(
    model_display_name="llama2_guanaco_qlora_gpu_" + TIMESTAMP,
    replica_count=1,
    service_account = SERVICE_ACCOUNT,
    tensorboard = TENSORBOARD_RESOURCE_NAME,
    machine_type="g2-standard-12",
    accelerator_type="NVIDIA_L4",
    accelerator_count = 1,
)
print(model)


# Deploy endpoint
# endpoint = model.deploy(machine_type='n1-standard-4',
#     accelerator_type= "NVIDIA_TESLA_T4",
#     accelerator_count = 1)
# print(endpoint.resource_name)


