import mlflow
import os
import logging
import click
import requests
import os
import pandas as pd


EXPERIMENT_NAME = '/Users/d.amoateng110@gmail.com/dl_model_pipeline'
mlflow.set_experiment(EXPERIMENT_NAME)

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

@click.command(help="This program downloads data for training deep learning model")
@click.option("--download_url", default="https://raw.githubusercontent.com/amoat7/GCP-APIs/master/dataset.csv", help="remote url for downloading training data")
@click.option("--local_folder", default="./data", help="This is a local data folder")
@click.option("--pipeline_run_name", default="pipeline", help="This is the mlflow run name")

def task(download_url, local_folder, pipeline_run_name):
    with mlflow.start_run(run_name=pipeline_run_name) as mlrun:
        logger.info(f"Downloading data from {download_url}")
        r = requests.get(download_url, allow_redirects=True)
        os.makedirs(local_folder, exist_ok=True)
        df = pd.read_csv(download_url)
        df.to_csv(f"{local_folder}/training_data.csv", index=False)
        mlflow.log_param("download_url", download_url)
        mlflow.log_param("local_folder", local_folder)
        mlflow.log_param("mlflow run id", mlrun.info.run_id)
        mlflow.set_tag('pipeline_step', __file__)
        mlflow.log_artifacts(local_folder, artifact_path="data")

    logger.info(f"Finished downloading data to {local_folder}")

if __name__ == "__main__":
    task()