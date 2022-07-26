import mlflow
import os
import logging
import click

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

_steps = [
    "download_data",
    "tuning",
    "train_model",
    "register_model"

]

@click.command()
@click.option("--steps", default="all", type=str)
def run_pipeline(steps):
    EXPERIMENT_NAME = '/Users/d.amoateng110@gmail.com/dl_model_pipeline'
    mlflow.set_experiment(EXPERIMENT_NAME)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    logger.info(f"pipeline experiment_id: {experiment.experiment_id}")


    # Steps to execute
    active_steps = steps.split(",") if steps!="all" else _steps
    logger.info(f"pipeline active steps to execute in this run :{active_steps}")

    with mlflow.start_run(run_name='pipeline', nested=True) as active_run:
    
        if "download_data" in active_steps:
            download_run = mlflow.run("https://github.com/amoat7/tracking_code_data_pipeline_versioning.git", "download_data", parameters={}, env_manager="local")
            download_run = mlflow.tracking.MlflowClient().get_run(download_run.run_id)
            file_path_uri = download_run.data.params['local_folder']
            logger.info(f'downloaded data is located locally in folder: {file_path_uri}')
            logger.info(download_run)

        if "tuning" in active_steps:
            tuning_run = mlflow.run("https://github.com/amoat7/tracking_code_data_pipeline_versioning.git", "tuning", parameters={}, env_manager="local")
            tuning_run = mlflow.tracking.MlflowClient().get_run(tuning_run.run_id)
            logger.info(tuning_run)
        
        

        if "train_model" in active_steps:
            training_model_run = mlflow.run("https://github.com/amoat7/tracking_code_data_pipeline_versioning.git", "train_model", parameters={"data_path": file_path_uri},env_manager="local")
            training_model_run_id =  training_model_run.run_id
            training_model_run = mlflow.tracking.MlflowClient().get_run(training_model_run_id)
            logger.info(training_model_run)

        if "register_model" in active_steps:
            if training_model_run_id is not None and training_model_run_id!= 'None':
                register_model_run = mlflow.run("https://github.com/amoat7/tracking_code_data_pipeline_versioning.git","register_model", parameters={"mlflow_run_id": training_model_run_id},env_manager="local")
                register_model_run = mlflow.tracking.MlflowClient().get_run(register_model_run.run_id)
                logger.info(register_model_run)
            else:
                logger.info("no model to register since no trained model run id.")
    logger.info('finished mlflow pipeline run with a run_id = %s', active_run.info.run_id)

if __name__ == "__main__":
    run_pipeline()