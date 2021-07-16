import sklearn
import joblib
import pandas as pd
import numpy
from preprocessing_functions import extract_first_letter

from flask import Flask, request

#load model
pipeline = joblib.load('titanic.model')
#print(pipeline)

#app
app = Flask('__name__')

@app.route('/')
def index():
  return "<h1>Bienvenue sur notre API. Utiliser /predict en POST pour faire une prédictions sur le dataset du titanic</h1>"

@app.route('/ping', methods=['GET'])
def ping():
  return('pong', 200)

@app.route('/predict', methods=['POST'])
def predict():
  #print(request.json)
  df = pd.DataFrame(request.json)
  prediction = pipeline.predict(df)[0]
  return (str(prediction), 201)


if __name__ == "__main__":
  app.run(host='0.0.0.0')