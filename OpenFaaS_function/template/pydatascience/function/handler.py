import json
import os

# from .core import utils


function_root = os.environ.get("function_root")

# Now  pre-load the model, e.g.
# from .core import model


def handle(req: bytes) -> str:
    """handle a request to the function
    Args:
        req (bytes): request body
    """

    return json.dumps({"echo": req)
