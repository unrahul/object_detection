#!usr/bin/env python
"""a rest api for object detection"""
import io

import flask
from flask import Flask, make_response, request, send_file
from flask_cors import CORS
from gevent.pywsgi import WSGIServer
from PIL import Image

from handler import boxed_img

app = Flask(__name__)
CORS(app)


banner = {
    "what": "object detection API",
    "usage": {
        "client": "curl -i  -X POST -F img=@data/cat.jpg 'http://localhost:5059/recog'",
        "server": "docker run -d -p 5059:5059 stacks_obj_detection",
    },
}


@app.before_request
def fix_transfer_encoding():
    """
    Sets the "wsgi.input_terminated" environment flag, thus enabling
    Werkzeug to pass chunked requests as streams.  The gunicorn server
    should set this, but it's not yet been implemented.
    """

    transfer_encoding = request.headers.get("Transfer-Encoding", None)
    if transfer_encoding == u"chunked":
        request.environ["wsgi.input_terminated"] = True


@app.route("/usage", methods=["GET"])
def usage():
    return flask.jsonify({"info": banner}), 201


@app.route("/detect", methods=["POST"])
def detect():
    if flask.request.files.get("img"):
        img = flask.request.files["img"].read()
        img = Image.open(io.BytesIO(img))
        y_hat = boxed_img(img)
        response = make_response(
            send_file(io.BytesIO(y_hat), mimetype="image/jpg"), 201
        )
        response.headers["Content-Transfer-Encoding"] = "base64"
        return response
    return flask.jsonify({"status": "not an image file"})


@app.errorhandler(404)
def not_found(error):
    return flask.make_response(flask.jsonify({"error": "Not found"}), 404)


if __name__ == "__main__":
    # app.run(host="0.0.0.0", port=5059)
    http_server = WSGIServer(("", 5059), app)
    http_server.serve_forever()
