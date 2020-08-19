from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import cv2
import numpy as np
import mxnet as mx
import numpy as np
from glob import glob
from tqdm import tqdm
from m_config import *
import requests

m_URL = "http://{}:{}/predictions/m_server".format(HOST, PORT)

def distance(a, b):
    diff = np.subtract(a, b)  # (512) vs (512,); (n, 512) vs (512,)
    dist = np.sum(np.square(diff), axis=-1)     # (1,); (n,)
    # dist = 100.
    return np.min(dist)

def m_get_embedding_MMS(aligned):
    retval, img_encoded = cv2.imencode(".png", aligned)
    data = img_encoded.tostring()
    files = {'data': data}
    response = requests.post(m_URL, files=files)  # , data=values
    if response.ok:
        r = response.json()
        embedding = r["embedding"][0]
        embedding = np.array(embedding, dtype=np.float32)
        return embedding
    else:
        return None

def m_get_embedding_Flask(aligned):
    retval, img_encoded = cv2.imencode(".png", aligned)
    data = img_encoded.tostring()
    files = {'data': data}
    response = requests.post(m_URL, data=data, headers={'content-type': 'image/png'})
    if response.ok:
        r = response.json()
        embedding = r["embedding"]
        embedding = np.array(embedding, dtype=np.float32)
        return embedding
    else:
        return None


class MFaceRecognize:
    def __init__(self, model_dir):
        self.ctx = self.gpu_device()
        self.get_model(self.ctx, model_dir)

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
        self.model = mx.mod.Module(symbol=sym, context=self.ctx, label_names = None)
        self.model.bind(data_shapes=[('data', (1, 3, 112, 112))])
        self.model.set_params(arg_params, aux_params)
        return self.model

    def get_feature(self, aligned):     # face aligned in <BGR> format with shape (112, 112, 3)
        aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        aligned = np.transpose(aligned, (2,0,1))
        input_blob = np.expand_dims(aligned, axis=0)
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        self.model.forward(db, is_train=False)
        embedding = self.model.get_outputs()[0].asnumpy()
        embedding = embedding.reshape((512,))
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    def get_embedding_from_directory(self, directory):
        print('reading %s' % directory)
        basename = os.path.basename(directory) + ".txt"
        basename = os.path.join(directory, basename)
        fname = glob(os.path.join(directory, "*.jpg"))
        fname.extend(glob(os.path.join(directory, "*.png")))
        if glob(basename):
            print("remove {}".format(basename))
            os.remove(basename)

        if fname:
            with open(basename, mode="w", encoding="utf-8") as f:
                for i, fn in tqdm(enumerate(fname)):
                    img = cv2.imread(fn)
                    embedding = self.get_feature(aligned=img)
                    embedding = [str(e) for e in embedding]
                    embedding = ",".join(embedding) + "\n"
                    f.write(embedding)
        else:
            print("There is no image in {}".format(directory))



if __name__ == "__main__":
    path = "models/insightface-r100-ii/model"
    mFR = MFaceRecognize(path)
    mFR.get_embedding_from_directory("D:/Me/M/m_database/HanhHuy")
