from flask import Flask, render_template , request , jsonify
from PIL import Image
import os , io , sys
import numpy as np
import cv2
import base64
import requests
import json

app = Flask(__name__)

@app.route('/maskImage' , methods=['POST'])
def mask_image():
    file = request.files['image'].read() # byte file
    url = # INSERT YOUR FUNCTION URL HERE 

    file_str = {'img': file}
    x = requests.post(url, files = file_str)
    req_obj = x.json()
    return jsonify({'status': req_obj['img']})

@app.route("/")
def hello_world():
    return render_template('home.html', title='Banana Cloud', description='Check state of your Bananas!')

@app.after_request
def after_request(response):
    print("log: setting cors" , file = sys.stderr)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response
