from email.policy import default
import mlflow
import os
import logging
import click
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

@click.command(help="This program downloads data for training deep learning model")
@click.option("--download_url", default="https://github.com/amoat7/GCP-APIs/blob/master/dataset.csv", help="remote url for downloading training data")
@click.option("--local_folder", default="./data", help="This is a local data folder")
@click.option("--pipeline_run_name", default="pipeline", help="This is the mlflow run name")
def task(download_url, local_folder, pipeline_run_name):
    with mlflow.start_run(run_name=pipeline_run_name) as mlrun:
        logger.info(f"Downloading data from {download_url}")
        r = requests.get(download_url, allow_redirects=True)
        open('data.csv', 'wb').write(r.content)
        mlflow.log_param("download_url", download_url)
        mlflow.log_param("local_folder", local_folder)
        mlflow.log_param("mlflow run id", mlflow.info.run_id)
        mlflow.set_tag('pipeline_step', __file__)
        mlflow.log_artifacts(local_folder, artifact_path="data")

    logger.info(f"Finished downloading data to {local_folder}")

if __name__ == "__main__":
    task()