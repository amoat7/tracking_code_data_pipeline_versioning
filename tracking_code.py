
# %%
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
EXPERIMENT_NAME = '/Users/d.amoateng110@gmail.com/dl_model_pipeline'

mlflow.set_experiment(EXPERIMENT_NAME)
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
print("experiment_id:", experiment.experiment_id)

# %%
# enable autologgin

mlflow.tensorflow.autolog()
with mlflow.start_run(experiment_id=experiment.experiment_id, run_name='keras_model'):
    # first neural network with keras tutorial
    from numpy import loadtxt
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense

    # load the dataset
    dataset = loadtxt('pima-indians.txt', delimiter=',')
    # split into input (X) and output (y) variables
    X = dataset[:,0:8]
    y = dataset[:,8]

    # define the keras model
    model = Sequential()
    model.add(Dense(12, input_shape=(8,), activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))


    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit the keras model on the dataset
    model.fit(X, y, epochs=150, batch_size=10)

    # evaluate the keras model
    _, accuracy = model.evaluate(X, y)
    print('Accuracy: %.2f' % (accuracy*100))
# %%
