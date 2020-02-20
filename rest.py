#!usr/bin/env python
"""a rest api for object detection"""
import io

from PIL import Image

import quart
from gevent.pywsgi import WSGIServer
from main import boxed_img
from quart_cors import cors

app = quart.Quart(__name__)
cors(app)


banner = {
    "what": "object detection API",
    "usage": {
        "client": "curl -i  -X POST -F img=@data/cat.jpg 'http://localhost:5059/recog'",
        "server": "docker run -d -p 5059:5059 stacks_obj_detection",
    },
}


@app.route("/usage", methods=["GET"])
def usage():
    return quart.jsonify({"info": banner}), 201


@app.route("/detect", methods=["POST"])
def detect():
    if quart.request.files.get("img"):
        img = quart.request.files["img"].read()
        img = Image.open(io.BytesIO(img))
        y_hat = boxed_img(img)
        return quart.jsonify({"class": y_hat[0], "prob": y_hat[1]}), 201
    return quart.jsonify({"status": "not an image file"})


@app.errorhandler(404)
def not_found(error):
    return quart.make_response(quart.jsonify({"error": "Not found"}), 404)


if __name__ == "__main__":
    #app.run(host="0.0.0.0", port=5059)
    http_server = WSGIServer(('', 5059), app)
    http_server.serve_forever()
