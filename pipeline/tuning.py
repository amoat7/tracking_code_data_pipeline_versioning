# Objective function
# Function-based APIs and class-based APIs
# Trials
# Search space
# Suggest - Algorithm
# Scheduler - covers early stopping, pruning

import mlflow
import logging
import click
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import os
from ray import tune  
from ray.tune.integration.mlflow import mlflow_mixin
from ray.tune.integration.keras import TuneReportCallback

EXPERIMENT_NAME = '/Users/d.amoateng110@gmail.com/dl_model_pipeline'
mlflow.set_experiment(EXPERIMENT_NAME)
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

@mlflow_mixin
def finetuning_dl_model(config, pipeline_run_name = "pipeline"):
    mlflow.tensorflow.autolog()
    with mlflow.start_run(run_name=pipeline_run_name) as mlrun:
        # first neural network with keras tutorial
        # load the dataset
        dataset = pd.read_csv("https://raw.githubusercontent.com/amoat7/GCP-APIs/master/dataset.csv")
        # split into input (X) and output (y) variables
        X = dataset.iloc[:,0:8]
        y = dataset.iloc[:,8]

        # define the keras model
        model = Sequential()
        model.add(Dense(config['layer1'], input_shape=(8,), activation='relu'))
        model.add(Dense(config['layer2'], activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        opt = tf.keras.optimizers.Adam(learning_rate=config['lr'])
        # compile the keras model
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

        # fit the keras model on the dataset
        metrics = {"accuracy": "val_accuracy"}
        model.fit(X, y, epochs=150, batch_size=config['batch_size'], validation_split=0.1, callbacks=[TuneReportCallback(metrics, on="validation_end")])

        # evaluate the keras model
        _, accuracy = model.evaluate(X, y)
        logger.info(f"Accuracy: %.2f' % {(accuracy*100)}")

        run_id = mlrun.info.run_id
        logger.info("run_id: {}; lifecycle_stage: {}".format(run_id,
                                                             mlflow.get_run(run_id).info.lifecycle_stage))
        mlflow.log_param("fine_tuning_mlflow_run_id", run_id)
        mlflow.set_tag('pipeline_step', __file__)


def run_hpo_dl_model(tracking_uri, experiment_name, num_samples=10, num_epochs=3, gpus_per_trial=0):
    import ray
    ray.init(local_mode=True)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([16, 32, 64, 128]),
        "layer1": tune.choice([4,8,16,32,64]), 
        "layer2":tune.choice([4,8,16,32,64]),
        "mlflow": {
            "experiment_name": EXPERIMENT_NAME,
            "tracking_uri": mlflow.get_tracking_uri()
        },
    }

    trainable = tune.with_parameters(
        finetuning_dl_model,
        num_epochs=num_epochs
    )

    analysis = tune.run(
        trainable,
        resources_per_trial={
            "cpu": 1,
            "gpu": gpus_per_trial
        },
        metric="accuracy",
        mode="max",
        config=config,
        num_samples=num_samples,
        name="hpo_tuning"  
    )
    logger.info("Best hyperparameters found were: %s", analysis.best_config)

def task():
    run_hpo_dl_model(num_samples=10,
                     num_epochs=3,
                     gpus_per_trial=0,
                     tracking_uri="databricks",
                     experiment_name=EXPERIMENT_NAME)


if __name__ == "__main__":
    task()