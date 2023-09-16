"""
This module contains a basic Flask web application for the deployment
of a Deep Learning model.
"""
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import joblib
import numpy as np


def return_prediction(model, scaler, sample_json):
    s_len = sample_json['sepal_length']
    s_wid = sample_json['sepal_width']
    p_len = sample_json['petal_length']
    p_wid = sample_json['petal_width']
    
    flower = [[s_len, s_wid, p_len, p_wid]]
    flower = scaler.transform(flower)
    
    prediction = model.predict(flower)  
    classes_index = np.argmax(prediction, axis=-1)

    classes = np.array(['setosa', 'versicolor', 'virginica'])
    return classes[classes_index]



app = Flask(__name__)

@app.route("/")
def index():
    return "<h1>Flask app running</h1>"

flower_model = load_model('final_iris_model.h5')
flower_scaler = joblib.load('iris_scaler.pkl')

@app.route("/api/flower", methods=['POST'])
def flower_prediction():
    content = request.json
    print(content)
    results = return_prediction(flower_model, flower_scaler, content)
    print(results)
    print (jsonify(results))
    return jsonify(results)
    
    
if __name__ == '__main__':
    app.run()
