
import json
import os
import base64
import cv2

from flask import Request, send_file, jsonify
from .core import detect
import numpy as np
import logging
from PIL import Image
import io
logging.basicConfig(level=logging.DEBUG)
# from .core import utils


function_root = os.environ.get("function_root")

# Now  pre-load the model, e.g.
# from .core import model


def handle(req: Request):
    """handle a request to the function.

    Your response is immediately passed to the caller, unmodified.
    This allows you full control of the response, e.g. you can set
    the status code by returning a tuple (str, int). A detailed
    description of how responses are handled is found here:

    http://flask.pocoo.org/docs/1.0/quickstart/#about-responses

    Args:
        req (Request): Flask request object
    """
    if not req:
        return json.dumps({"error": "No input provided", "code": 400})

    img_f = req.files['img'].read()
    
    #convert str to numpy img
    npimg = np.fromstring(img_f, np.uint8)
    img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
    print('img shape: ', img.shape, flush=True)
    
    result = detect.detect(img)

    result = cv2.cvtColor(result,cv2.COLOR_BGR2RGB)

    print('result: ', result.shape, flush=True)

    img = Image.fromarray(result.astype("uint8"))
    rawBytes = io.BytesIO()
    img.save(rawBytes, "JPEG")
    rawBytes.seek(0)
    img_base64 = base64.b64encode(rawBytes.read())

    return jsonify({'img': str(img_base64), 'code': 200})