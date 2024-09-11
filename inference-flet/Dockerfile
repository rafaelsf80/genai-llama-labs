FROM python:3.9

RUN pip install flet 
RUN pip install google-cloud-aiplatform
# RUN pip install protobuf==3.19.6  --user

COPY ./app /app

WORKDIR /app

EXPOSE 7080

CMD ["python", "app.py"]