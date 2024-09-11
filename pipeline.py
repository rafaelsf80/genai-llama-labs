""" WORK-IN-PROGRESS
    Pipeline for LLM training and deployment
    Performs training and deployment:
    1. Custom training pipeline (PEFT) and deployment in Vertex AI
    Refer to training script located at 'trainer_script.py'

    2. Deploy image with Uvicorn server containing FastAPI app model in Vertex AI
    The deployment uses a g2-standard-24 machine type with 2xL4 GPU
"""

from google.cloud import aiplatform
from datetime import datetime
from google.cloud.aiplatform import Model, Endpoint


BUCKET          = 'gs://argolis-vertex-europewest4'
PROJECT_ID      = 'argolis-rafaelsanchez-ml-dev'
LOCATION        = 'europe-west4'
SERVICE_ACCOUNT = 'tensorboard-sa@argolis-rafaelsanchez-ml-dev.iam.gserviceaccount.com'
TENSORBOARD_RESOURCE_NAME = 'projects/989788194604/locations/europe-west4/tensorboards/8884581718011412480'
TRAIN_IMAGE     = 'europe-west4-docker.pkg.dev/argolis-rafaelsanchez-ml-dev/ml-pipelines-repo/llama2-qlora:latest'
TIMESTAMP       = datetime.now().strftime("%Y%m%d%H%M%S")

DEPLOY_IMAGE    = 'europe-west4-docker.pkg.dev/argolis-rafaelsanchez-ml-dev/ml-pipelines-repo/llama2-qlora-predict'

HEALTH_ROUTE = "/health"
PREDICT_ROUTE = "/predict"
SERVING_CONTAINER_PORTS = [7080]


aiplatform.init(project=PROJECT_ID, staging_bucket=BUCKET, location=LOCATION)
                
# Launch Training pipeline, a type of Vertex Training Job.
# A Training pipeline integrates three steps into one job: Accessing a Managed Dataset (not used here), Training, and Model Upload. 

job = aiplatform.CustomContainerTrainingJob(
    display_name="llama2_guanaco_qlora_gpu_" + TIMESTAMP,
    container_uri=TRAIN_IMAGE,
    model_serving_container_image_uri=DEPLOY_IMAGE,
    model_serving_container_predict_route=PREDICT_ROUTE,
    model_serving_container_health_route=HEALTH_ROUTE,
    model_serving_container_ports=SERVING_CONTAINER_PORTS,
    labels={"peft":"qlora"},
    location="europe-west4",


)


model = job.run(
    model_display_name="llama2_guanaco_qlora_gpu",
    replica_count=1,
    service_account = SERVICE_ACCOUNT,
    tensorboard = TENSORBOARD_RESOURCE_NAME,
    machine_type="n1-standard-4",
    accelerator_type="NVIDIA_TESLA_V100",
    accelerator_count = 1,
)
print(model)


    


# model = Model.upload(
#     display_name="llama2-70B-chat-gcs", 
#     description=f'llama2-70B-chat with Uvicorn and FastAPI',
#     serving_container_image_uri=DEPLOY_IMAGE,
#     serving_container_predict_route=PREDICT_ROUTE,
#     serving_container_health_route=HEALTH_ROUTE,
#     serving_container_ports=SERVING_CONTAINER_PORTS,
#     artifact_uri=ARTIFACT_URI, # sets AIP_STORAGE_URI env variable, nothing else
#     location="europe-west4",
#     upload_request_timeout=1800,
#     sync=True,
#     )

# Retrieve a Model on Vertex
model = Model(model.resource_name)

# Deploy model
endpoint = model.deploy(
    machine_type="g2-standard-24",
    accelerator_type="NVIDIA_L4",
    # The issue is the SERVICE_ACCOUNT does not have access to the GCS bucket where the PEFT lora is stored. After I granted the access to the service_account, the issue goes away.
    service_account="cloud-run-llm@argolis-rafaelsanchez-ml-dev.iam.gserviceaccount.com", # must hace GCS read access
    accelerator_count=2,
    traffic_split={"0": 100}, 
    min_replica_count=1,
    max_replica_count=1,
    traffic_percentage=100,
    deploy_request_timeout=1800,
    sync=True,
)
endpoint.wait()









