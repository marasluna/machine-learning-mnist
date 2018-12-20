import pandas as pd
import numpy as np

from flask import Flask
from sklearn.externals import joblib
from flask import request
from flask import json

app = Flask(__name__)

BUCKET_NAME = ''
MODEL_FILE_NAME = 'logistic_regression.pkl'
MODEL_LOCAL_PATH = '/Users/marasluna/projects/machine-learning-mnist/model'


@app.route('/')
def index():
    return "A machine learning engine for classifying MNIST dataset."


@app.route('/predict/', methods=['POST'])
def predict():
    # get payload
    payload = json.loads(request.get_data())
    # predict
    prediction = predict_model(payload["payload"])
    # get response
    data = {}
    data['data'] = prediction
    return json.dumps(data)


@app.route('/test/', methods=['POST'])
def test():
    return 'Test'


def load_model():
    # conn = S3Connection()
    # bucket = conn.create_bucket(BUCKET_NAME)
    # key_obj = Key(bucket)
    # key_obj.key = MODEL_FILE_NAME

    # contents = key_obj.get_contents_to_filename(MODEL_LOCAL_PATH)
    # return joblib.load(MODEL_LOCAL_PATH)
    return joblib.load(MODEL_LOCAL_PATH + "/" + MODEL_FILE_NAME)


def predict_model(data):
	panda_data = pd.DataFrame.from_records(data)
	panda_data.columns = pd.to_numeric(panda_data.columns)
	panda_data.sort_index(axis=1, inplace=True)

	clf = load_model()
	return clf.predict(panda_data).tolist()

