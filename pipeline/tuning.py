import mlflow
import logging
import click
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import os
import keras_tuner

EXPERIMENT_NAME = '/Users/d.amoateng110@gmail.com/dl_model_pipeline'
mlflow.set_experiment(EXPERIMENT_NAME)
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def build_model(hp):
     # define the keras model
        model = Sequential()
        model.add(Dense(hp.Choice('layer1', [4, 8, 16, 32]), input_shape=(8,), activation='relu'))
        model.add(Dense(hp.Choice('layer1', [4, 8, 16, 32]), activation='relu'))
        model.add(Dense(1, activation='sigmoid'))


        # compile the keras model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model


def task(pipeline_run_name="pipeline"):
    mlflow.tensorflow.autolog()
    with mlflow.start_run(run_name=pipeline_run_name) as mlrun:
        # first neural network with keras tutorial
        # load the dataset
        dataset = pd.read_csv("https://raw.githubusercontent.com/amoat7/GCP-APIs/master/dataset.csv")
        # split into input (X) and output (y) variables
        X = dataset.iloc[:,0:8]
        y = dataset.iloc[:,8]

       
        tuner = keras_tuner.RandomSearch(
            build_model,
            objective='val_loss',
            max_trials=5)
        # fit the keras model on the dataset
        tuner.search(X, y, epochs=150, batch_size=10)

        best_model = tuner.get_best_models()[0]
        
        run_id = mlrun.info.run_id
        logger.info("run_id: {}; lifecycle_stage: {}".format(run_id,
                                                             mlflow.get_run(run_id).info.lifecycle_stage))
        mlflow.log_param("fine_tuning_mlflow_run_id", run_id)
        mlflow.set_tag('pipeline_step', __file__)



if __name__ == "__main__":
    task()