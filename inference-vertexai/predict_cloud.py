""" Basic inference of an LLM model deployed in Vertex AI Prediction
"""

from google.cloud import aiplatform

# import google
# import vertexai


# LOCATION = "europe-west4"
# PROJECT_ID = "argolis-rafaelsanchez-ml-dev"

# credentials, _ = google.auth.default(quota_project_id=PROJECT_ID)
# vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)

endpoint = aiplatform.Endpoint(
    endpoint_name="projects/989788194604/locations/europe-west4/endpoints/6866621639275053056"  # <---- CHANGE THIS !!!!
)

PROMPT = "what can I see in Murcia ?"

response = endpoint.predict([[PROMPT]])
print(response.predictions[0])

# Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0! (when checking argument for argument index in method wrapper_CUDA__index_select)