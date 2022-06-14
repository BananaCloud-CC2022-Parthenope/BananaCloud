
import json
import os

from flask import Request

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

    return json.dumps({"echo": req.get_data().decode('utf-8')})
