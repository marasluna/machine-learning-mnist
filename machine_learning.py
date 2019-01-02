import pandas as pd
import numpy as np
import boto3

from flask import Flask
from sklearn.externals import joblib
from flask import request
from flask import json

app = Flask(__name__)

BUCKET_NAME = 'machine-learning-mnist'
MODEL_FILE_NAME = 'logistic_regression.pkl'
MODEL_LOCAL_PATH = '/tmp/' + MODEL_FILE_NAME


@app.route('/')
def index():
    return "A machine learning engine for classifying MNIST dataset."


@app.route('/predict', methods=['POST'])
def predict():
    # get payload
    payload = json.loads(request.get_data())
    # transform payload to panda data
    data = json_to_data(payload)
    # predict
    prediction = predict_model(data)
    # transform prediction to json
    response = prediction_to_json(data.index, prediction)
    return json.dumps(response)


@app.route('/score', methods=['POST'])
def score():
    # get payload
    payload = json.loads(request.get_data())
    # transform payload to dataset
    dataset = json_to_dataset(payload)
    # get target and coefficients
    data = dataset.iloc[:, 1:]
    target = dataset.iloc[:, 0]
    # score
    score = score_model(data, target)
    # transform score to json
    response = score_to_json(score)
    return json.dumps(response)


def json_to_dataset(payload):
    panda_data = pd.DataFrame.from_dict(
        payload["payload"]["dataset"], orient="index")
    panda_data.columns = pd.to_numeric(panda_data.columns)
    panda_data.sort_index(axis=1, inplace=True)

    return panda_data


def json_to_data(payload):
    panda_data = pd.DataFrame.from_dict(
        payload["payload"]["data"], orient="index")
    panda_data.columns = pd.to_numeric(panda_data.columns)
    panda_data.sort_index(axis=1, inplace=True)

    return panda_data


def prediction_to_json(index, prediction):
    df = pd.DataFrame(index=index,
                      data=prediction, columns=["prediction"])
    return json.loads(df.to_json())


def score_to_json(score):
    return {"score": score}


def load_model():
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(BUCKET_NAME)
    obj = bucket.Object(MODEL_FILE_NAME)
    with open(MODEL_LOCAL_PATH, 'wb') as data:
        obj.download_fileobj(data)
    return joblib.load(MODEL_LOCAL_PATH)


def predict_model(data):
    clf = load_model()
    return clf.predict(data).tolist()


def score_model(data, target):
    clf = load_model()
    return clf.score(data, target)
