#!/bin/sh

curl -v \
-X POST \
-H "Authorization: Bearer $(gcloud auth print-access-token)" \
-H "Content-Type: application/json" \
https://europe-west4-aiplatform.googleapis.com/v1/projects/989788194604/locations/europe-west4/endpoints/6866621639275053056:predict \
-d @instances_test.json