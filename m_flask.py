from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import cv2
import json
import numpy as np
from tqdm import tqdm
from flask import Flask, request, Response


from m_config import *
from m_face_recognition import * # myFaceRecognize, compare

# Initialize the Flask application
app = Flask(__name__)
mFR = MFaceRecognize(MODEL_FACE_RECOGNIZE_PATH)
img = np.random.randint(0, 256, size = (112, 112, 3), dtype=np.uint8)
for _ in tqdm(range(30)):
    mFR.get_feature(img)

del(img)

# route http posts to this method
@app.route('/predictions/m_server', methods=['POST'])
def m():
    r = request
    if r.method == "POST":
        nparr = np.fromstring(r.data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            emb = mFR.get_feature(aligned=img)
            response = {'embedding': emb.tolist()}
            response_pickled = json.dumps(response)
        else:
            response = {'embedding': None}
            response_pickled = json.dumps(response)
    else:
        response = {'embedding': None}
        response_pickled = json.dumps(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")


# start flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8022)

