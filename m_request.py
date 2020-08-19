import cv2
import requests
import numpy as np

URL = "http://localhost:8022/predictions/m_server"

def m_get_embedding(aligned):
    retval, img_encoded = cv2.imencode(".png", aligned)
    data = img_encoded.tostring()
    files = {'data': data}
    response = requests.post(URL, files=files)  # , data=values
    if response.ok:
        r = response.json()
        embedding = r["embedding"][0]
        embedding = np.array(embedding, dtype=np.float32)
        return embedding
    else:
        return None

if __name__ == "__main__":
    from time import time
    _t = time()
    img = cv2.imread("example.png")
    embd = m_get_embedding(img)
    print("time: ", time() - _t)