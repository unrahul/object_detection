#!usr/bin/env python
"""a rest api for object detection"""
#
#
#
import io

from PIL import Image

import quart
from quart_cors import cors

app = quart.Quart(__name__)
cors(app)


banner = {
    "what": "object detection API",
    "usage": {
        "client": "curl -i  -X POST -F img=@data/cat.jpg 'http://localhost:5000/recog'",
        "server": "docker run -d -p 5000:5000 stacks_img_recog",
    },
}


@app.route("/usage", methods=["GET"])
def usage():
    return quart.jsonify({"info": banner}), 201


@app.route("/recog", methods=["POST"])
def recog():
    if quart.request.files.get("img"):
        img = quart.request.files["img"].read()
        img = Image.open(io.BytesIO(img))
        y_hat = classify(img)
        return quart.jsonify({"class": y_hat[0], "prob": y_hat[1]}), 201
    return quart.jsonify({"status": "not an image file"})


@app.errorhandler(404)
def not_found(error):
    return quart.make_response(quart.jsonify({"error": "Not found"}), 404)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5059)
