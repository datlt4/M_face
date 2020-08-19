import os
import sys
import threading
import imutils
from tqdm import tqdm

from m_config import *
from m_face_detection import * # myFaceDetector
from m_face_recognition import * # myFaceRecognize, compare

OS = sys.platform
OPENCV_OBJECT_TRACKERS = cv2.TrackerMedianFlow_create

flagFaceDetection = 0

def IoU(boxA, boxB):  # (x, y, w, h) format
    ymin = max(boxA[0], boxB[0])
    xmin = max(boxA[1], boxB[1])
    ymax = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    xmax = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xmax - xmin + 1) * max(0, ymax - ymin + 1)
    boxAArea = (boxA[2] + 1) * (boxA[3] + 1)
    boxBArea = (boxB[2] + 1) * (boxB[3] + 1)
    iou = float(interArea) / float(boxAArea + boxBArea - interArea)
    return iou

def xyxy2xywh(*args):
    if len(args) == 1:
        x1, y1, x2, y2 = args[0]
    elif len(args) == 4:
        x1, y1, x2, y2 = args
    else:
        raise
    return x1, y1, x2 - x1, y2 - y1

def xywh2xyxy(*args):
    if len(args) == 1:
        x, y, w, h = args[0]
    elif len(args) == 4:
        x, y, w, h = args
    else:
        raise
    return x, y, x + w, y + h

def compare(people, e):
    dist = {}
    for k in people.keys():
        dist[k] = distance(e, people[k])

    id = "UNKNOWN"; value = RECOGNIZE_THRESHOLD
    for k in people.keys():
        _value = min(value, dist[k])
        if not _value == value:
            id = k; value = _value

    return id, value

class MVideoStreamer():
    def __init__(self, *args):
        self.capture = None
        self.status = None
        self.frame = None
        self.releaseFlag = True
        if not len(args) == 0:
            self.newStrSource(args[0])

    def newStrSource(self, src):
        del(self.capture)
        self.status = None
        self.frame = None
        self.releaseFlag = False
        if isinstance(src, int):
            if OS.startswith("win"):
                self.capture = cv2.VideoCapture(src + cv2.CAP_DSHOW)
            elif OS.startswith("linux"):
                self.capture = cv2.VideoCapture(src + cv2.CAP_V4L)
            elif OS.startswith("darwin"):
                self.capture = cv2.VideoCapture(src)
            else:
                sys.exit(1)
        elif isinstance(src, str):
            if src[-1] == "0":
                self.capture = cv2.VideoCapture(src[:-1])
            elif src[-1] == "1":
                self.capture = cv2.VideoCapture(src[:-1], cv2.CAP_FFMPEG)
            else:
                sys.exit(1)
        self.capture.set(3, 1280)
        self.capture.set(4, 960)

    def update(self):
        while True:
            if self.releaseFlag:
                self.capture.release()
                break
            else:
                if self.capture.isOpened():
                    (self.status, self.frame) = self.capture.read()

    def read(self):
        try:
            return self.status, self.frame
        except AttributeError as e:
            return None, None

    def release(self):
        self.releaseFlag = True

class Object:
    def __init__(self):
        self.state = False
        self.tracker = None
        self.faceHistory = []
        self.landmsHistory = []
        self.scoreHistory = []

    def createTracker(self, img, bbox):
        del(self.tracker)
        self.tracker = OPENCV_OBJECT_TRACKERS()
        self.tracker.init(img, tuple((int(i) for i in bbox)))
        self.state = True

    def updateTracker(self, resizedFrame, bbox, **kwargs):
        if bbox is None:
            exist, bbox = self.tracker.update(resizedFrame)
            if exist:
                self.bbox = tuple((int(i) for i in bbox))
            else:
                self.state = False
            return None
        else:
            score = kwargs["score"]
            landmark = kwargs["landmark"]
            orginFrame = kwargs["originFrame"]
            h_ratio = kwargs["h_ratio"]
            w_ratio = kwargs["w_ratio"]
            self.bbox = tuple((int(i) for i in bbox))
            x, y, w, h = [float(i) for i in bbox]
            x1, y1, x2, y2 = xywh2xyxy(x, y, w, h)
            x1 = int(x1 * w_ratio)
            x2 = int(x2 * w_ratio)
            y1 = int(y1 * h_ratio)
            y2 = int(y2 * h_ratio)
            landmark[0, :] = landmark[0, :] * w_ratio
            landmark[1, :] = landmark[1, :] * h_ratio
            self.createTracker(resizedFrame, (x, y, w, h))
            crop = orginFrame[y1:y2, x1:x2]
            self.faceHistory.append(crop)
            self.scoreHistory.append(score)
            self.landmsHistory.append(landmark)

    def deleteTracker(self):
        self.state = False
        del(self.tracker)
        self.tracker = None

    def returnBestFace(self):
        if len(self.faceHistory) == 0:
            return None
        else:
            scoreH = np.array(self.scoreHistory)
            bestface = self.faceHistory[np.argmax(scoreH)]
            landmarks = self.landmsHistory[np.argmax(scoreH)]
            bestface = align_face(bestface, landmarks)
            return bestface.astype(np.uint8)    # face

def main():
    mFD = MFaceDetector(MODEL_FACE_DETECTION_PATH)
    mFR = MFaceRecognize(MODEL_FACE_RECOGNIZE_PATH)

    img = np.random.randint(0, 256, size = (112, 112, 3), dtype=np.uint8)
    for _ in range(30):
        mFD.detect_faces(img)
        mFR.get_feature(img)
    del(img)

    # mVS = MVideoStreamer("D:/download/")
    mVS = MVideoStreamer(1)

    listObject = []
    people = {}

    dataset = glob(os.path.join(DATABASE, "*", "*.txt"))
    for txt_file in tqdm(dataset):
        name = os.path.basename(txt_file)[:-4]
        people[name] = []
        with open(txt_file, mode="r") as f:
            E = f.read().splitlines()
            embs = []
            for ee in E:
                embs.append([float(_e) for _e in ee.split(",")])
            e = np.array(embs, dtype=np.float32)
            people[name] = e

    thread = threading.Thread(target=mVS.update, daemon=True)
    thread.start()

    while True:
        status, frame = mVS.read()
        if status is None:
            continue
        elif not status:
            print("Reading frame from Stream was failed!")
            break
        else:
            frameHeight, frameWidth, _ = frame.shape
            frameDisplay = np.copy(frame)
            resizedFrame = imutils.resize(frame, HEIGHT_FRAME)
            resizedFrameHeight, resizedFrameWidth, _ = resizedFrame.shape
            h_ratio = float(frameHeight) / resizedFrameHeight
            w_ratio = float(frameWidth) / resizedFrameWidth

            global flagFaceDetection
            if flagFaceDetection == 0:
                bounding_boxes, landmarks, confidences = mFD.detect_faces(resizedFrame)
                flagFaceDetection = RESET_FLAG_FACE_DETECT
            else:
                bounding_boxes, landmarks, confidences = None, None, None
                flagFaceDetection -= 1
            
            for obj in listObject:
                obj.updateTracker(resizedFrame, None)

            if bounding_boxes is not None:
                overlap = [False] * len(bounding_boxes)
                for idx in range(len(listObject)):
                    ovlap = False
                    for j, bbox in enumerate(bounding_boxes):
                        _bbox = xyxy2xywh(bbox)
                        if IoU(_bbox, listObject[idx].bbox) > IOU_THRESHOLD:
                            ovlap = True
                            overlap[j] = True
                            _b = bounding_boxes[j]
                            landm = landmarks[j]
                            confd = confidences[j]
                            listObject[idx].updateTracker(resizedFrame, xyxy2xywh(_b), score=confd, landmark=landm, originFrame=frame, h_ratio=h_ratio, w_ratio=w_ratio)
                            if len(listObject[idx].scoreHistory) >= 20:
                                listObject[idx].state = False
                            break

                    if not ovlap:
                        listObject[idx].state = False

                for i in range(len(overlap)):
                    if not overlap[i]:
                        bbox = bounding_boxes[i]
                        landm = landmarks[i]
                        confd = confidences[i]
                        newObj = Object()
                        newObj.updateTracker(resizedFrame, xyxy2xywh(bbox), score=confd, landmark=landm, originFrame=frame, h_ratio=h_ratio, w_ratio=w_ratio)
                        listObject.append(newObj)
            else:
                if flagFaceDetection == RESET_FLAG_FACE_DETECT:
                    for idx in range(len(listObject)):
                        listObject[idx].state = False

            for idx in range(len(listObject) - 1, -1, -1):
                if not listObject[idx].state:
                    listObject[idx].deleteTracker()
                    crop = listObject[idx].returnBestFace()
                    emb = mFR.get_feature(crop)
                    id, dist = compare(people, emb)
                    print(id, dist)
                    cv2.imshow(id, crop)
                    del(listObject[idx])
                else:
                    x1, y1, x2, y2 = xywh2xyxy(listObject[idx].bbox)
                    cv2.rectangle(frameDisplay, (int(x1 * w_ratio), int(y1 * h_ratio)), (int(x2 * w_ratio), int(y2 * h_ratio)), (0, 0, 255), 2)

            if bounding_boxes is not None:
                for bbox in bounding_boxes:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frameDisplay, (int(x1 * w_ratio), int(y1 * h_ratio)), (int(x2 * w_ratio), int(y2 * h_ratio)), (0, 255, 0), 4)

            cv2.imshow("M", frameDisplay)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    mVS.release()
    cv2.destroyAllWindows()

# ===== main =====================================
if __name__ == "__main__":
    main()


