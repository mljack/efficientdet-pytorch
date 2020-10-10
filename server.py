import sys
import os
import infer
import warnings
import torch

torch.backends.cudnn.benchmark = True
warnings.simplefilter("ignore")

from flask import Flask, jsonify, request
app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello World!'

@app.route('/predict', methods=["GET","POST"])
def predict():
    if request.method == 'POST':
        if 'path' in request.values:
            return jsonify(infer.predict(path = request.values["path"]))
        else:
            file = request.files['file']
            img_bytes = file.read()
            return jsonify(infer.predict(img_bytes = img_bytes))
    else:
        return jsonify(infer.predict(path = "drone/test/00167_512_3.jpg"))

