import mlflow
import os

#docker network connect bridge mlflow_nginx
#docker network connect mlflow_docker_setup_frontend awesome_gates
#docker network connect mlflow_docker_setup_storage awesome_gates
# set environment variables to datbricks uri
os.environ['MLFLOW_TRACKING_URI'] = 'http://mlflow_nginx:80'
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"

# download >> run_model >> regiater_model