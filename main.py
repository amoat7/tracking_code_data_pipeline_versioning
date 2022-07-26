import mlflow
import os
import logging
import click

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

_steps = [
    "download_data",
    "training_model",
    "register_model"
]

@click.command()
@click.option("--steps", default="all", type=str)
def run_pipeline(steps):
    os.environ['MLFLOW_TRACKING_URI'] = 'http://mlflow_nginx:80'
    os.environ["AWS_ACCESS_KEY_ID"] = "minio"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"

    EXPERIMENT_NAME = 'dl_model_pipeline'
    mlflow.set_experiment(EXPERIMENT_NAME)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    logger.info(f"pipeline experiment_id: {experiment.experiment_id}")

    print(steps)

    # Steps to execute
    active_steps = steps.split(",") if steps!="all" else _steps
    logger.info(f"pipeline active steps to execute in this run :{active_steps}")

    with mlflow.start_run(run_name='pipeline', nested=True) as active_run:
        if "download_data" in active_steps:
            download_run = mlflow.run(".", "download_data", parameters={})
            download_run = mlflow.tracking.MlflowClient().get_run(download_run.run_id)
            
if __name__ == "__main__":
    run_pipeline()
