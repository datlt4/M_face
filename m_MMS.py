from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import io
import cv2
import argparse
import base64
import numpy as np
import mxnet as mx
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image

class MFaceRecognize:
    def __init__(self, model_dir):
        self.initialized = False
        self.model_dir = model_dir

    def initialize(self, context):
        self.initialized = True
        properties= context.system_properties
        self.ctx = self.gpu_device()
        self.model = self.get_model(self.ctx, self.model_dir)

    def gpu_device(self, gpu_number=0):
        try:
            _ = mx.nd.array([1, 2, 3], ctx=mx.gpu(gpu_number))
            return mx.gpu()
        except mx.MXNetError:
            return mx.cpu()

    def get_model(self, ctx, model_dir):
        sym, arg_params, aux_params = mx.model.load_checkpoint(model_dir, 0)
        all_layers = sym.get_internals()
        sym = all_layers['fc1_output']
        model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
        model.bind(data_shapes=[('data', (1, 3, 112, 112))])
        model.set_params(arg_params, aux_params)
        return model

    def get_feature(self, aligned):     # face aligned in <BGR> format with shape (112, 112, 3)
        # aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        aligned = np.transpose(aligned, (2,0,1))
        input_blob = np.expand_dims(aligned, axis=0)
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        self.model.forward(db, is_train=False)
        embedding = self.model.get_outputs()[0].asnumpy()
        embedding = embedding.reshape((512,))
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    def preprocess(self, data):
        image_path = data[0].get("data")
        if image_path is None:
            image_path = data[0].get("body")
        image = Image.open(io.BytesIO(image_path))
        return np.array(image)

    def postprocess(self, inference_output):
        return [{"embedding": [inference_output.tolist()]}]

    def handle(self, data, context):
        model_input = self.preprocess(data)
        model_out = self.get_feature(model_input)
        return self.postprocess(model_out)


_service = MFaceRecognize("models/insightface-r100-ii/model")

def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)

# ===================================

# import requests
# import base64
# from threading import Thread
# URL = "http://127.0.0.1:8022/predictions/vehicle_counting"

# def yolo_predict(img):
#     retval, buf = cv2.imencode(".png", img)
#     data = base64.b64encode(buf)
#     files = {'data': data}
#     response = requests.post(URL, files=files)  # , data=values
#     objects = []
#     if response.ok:
#         r = response.json()
#         for vtype, conf, x, y, w, h in zip(r["type"], r["conf"], r["x"], r["y"], r["w"], r["h"]):
#             objects.append([vtype, conf, [x, y, w, h]])
#     return objects



# import requests
# from threading import Thread
# from tqdm import tqdm
# import cv2
# URL = "http://127.0.0.1:8022/predictions/vehicle_counting"

# def yolo_predict(img):
#     headers = {'content-type': 'image/jpeg'}
#     _, img_encoded = cv2.imencode('.jpg', img)
#     response = requests.post(URL, data=img_encoded.tostring(), headers=headers)
#     objects = []
#     if response.ok:
#         r = response.json()
#         for vtype, conf, x, y, w, h in zip(r["type"], r["conf"], r["x"], r["y"], r["w"], r["h"]):
#             objects.append([vtype, conf, [x, y, w, h]])
#     return objects

# img = cv2.imread("traffic.png")

# def func(img):
#     for _ in tqdm(range(50)):
#         print(yolo_predict(img))

# thread1 = Thread(target=func, args=(img,))
# thread2 = Thread(target=func, args=(img,))
# thread1.start()
# thread2.start()
