import sys
import os
import infer
import warnings
import torch

if torch.cuda.device_count() > 1:
    torch.cuda.set_device(0)
    print("Select [%s]" % torch.cuda.get_device_name(torch.cuda.current_device()))

torch.backends.cudnn.benchmark = True
warnings.simplefilter("ignore")

from flask import Flask, jsonify, request
app = Flask(__name__)
app.debug = True

net, config = infer.init_net()

@app.route('/')
def hello():
    return 'Hello World!'

@app.route('/detect', methods=["GET", "POST"])
def predict():
    if request.method == 'POST':
        if 'path' in request.values:
            return jsonify(infer.predict_obb(net, config, path=request.values["path"]))
        else:
            file = request.files['file']
            img_bytes = file.read()
            return jsonify(infer.predict_obb(net, config, img_bytes=img_bytes))
    else:
        return jsonify(infer.predict_obb(net, config, img_path="_datasets/test/test000.jpg"))

