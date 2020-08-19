from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    import pkg_resources.py2_warn
except ImportError:
    pass

import os
import sys
import cv2
import imutils
import numpy as np
from tqdm import tqdm

from m_config import *
from m_face_detection import * # myFaceDetector
from m_face_recognition import * # myFaceRecognize, compare

from PyQt5.QtCore import Qt, QRect, QTimer, QThread, QThreadPool, QRunnable, QObject, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QHBoxLayout, QGridLayout, QMessageBox, QProgressBar, QDialog
from PyQt5.QtWidgets import QLabel, QFileDialog, QPushButton, QLineEdit, QFrame, QMenu, QInputDialog
from PyQt5.QtWidgets import QTabWidget, QApplication, QWidget, QAction, qApp, QGroupBox, QRadioButton
from PyQt5.QtGui import QFont, QPixmap, QIcon, QImage

strSrc = None
OS = sys.platform
OPENCV_OBJECT_TRACKERS = cv2.TrackerMedianFlow_create
mFD = MFaceDetector(MODEL_FACE_DETECTION_PATH)
# mFR = MFaceRecognize(MODEL_FACE_RECOGNIZE_PATH)

people = {}
dataset = glob(os.path.join(DATABASE, "*.txt"))
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

def recognizeToReturnID(cropped, aligned, people):
    # emb = mFR.get_feature(aligned)
    # emb = m_get_embedding_MMS(aligned=aligned)
    emb = m_get_embedding_Flask(aligned=aligned)
    _id, dist = compare(people, emb)
    return _id, dist, emb

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
        if isinstance(src, str):
            if src.isdigit():
                src = int(src)

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
            if src.startswith("rtsp://"):
                self.capture = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
            else:
                self.capture = cv2.VideoCapture(src)
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
        if len(self.faceHistory) < 3:
            return None, None
        else:
            scoreH = np.array(self.scoreHistory)
            bestFace = self.faceHistory[np.argmax(scoreH)]
            landmarks = self.landmsHistory[np.argmax(scoreH)]
            bestFaceAligned = align_face(bestFace, landmarks)
            return cv2.resize(bestFace, (112, 112)), bestFaceAligned.astype(np.uint8)     # face

    def returnListFace(self):
        if len(self.faceHistory) < 3:
            return None
        else:
            listFace = []
            for idx in range(len(self.scoreHistory)):
                faceAligned = align_face(self.faceHistory[idx], self.landmsHistory[idx])
                listFace.append(faceAligned)
        
            return listFace

class UpdateWorkerSignals(QObject):
    finished = pyqtSignal()

class UpdateWorker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super(UpdateWorker, self).__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = UpdateWorkerSignals()    
 
    @pyqtSlot()
    def run(self):
        result = self.fn(*self.args, **self.kwargs)
        try:
            self.signals.finished.emit()  # Done
        except RuntimeError as e:
            pass

class ResultWorkerSignals(QObject):
    finished = pyqtSignal(list)

class ResultWorker(QRunnable):
    def __init__(self, fn, **kwargs):
        super(ResultWorker, self).__init__()
        self.fn = fn
        self.cropped = kwargs["cropped"]
        self.aligned = kwargs["aligned"]
        self.people = kwargs["people"]
        self.signals = ResultWorkerSignals()    
 
    @pyqtSlot()
    def run(self):
        _id, dist, emb = self.fn(self.cropped, self.aligned, self.people)
        try:
            self.signals.finished.emit([_id, dist, emb, self.cropped])  # Done
        except RuntimeError as e:
            pass

class MainWindow(QMainWindow):
    def __init__(self, app):
        super(MainWindow, self).__init__()
        screen = app.primaryScreen()
        rect = screen.availableGeometry()
        self.title = TITLE
        self.l = 0
        if OS.startswith("win"):
            self.t = 30
        elif OS.startswith("linux"):
            self.t = 0
        elif OS.startswith("darwin"):
            self.t = 0
        else:
            sys.exit(1)
        self.w = rect.width()
        self.h = rect.height() - self.t
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.l, self.t, self.w, self.h)
        self.statusBar = self.statusBar()
        self._menuBar()
        self.mainLayout = MainLayout(self)
        self.closeEvent = self._closeEvent
        self.setCentralWidget(self.mainLayout)
        self.show()

    def _menuBar(self):
        menubar = self.menuBar()
        streamSrcMenu = menubar.addMenu('Stream')
        streamSrcMenu.addMenu(self._sourceMenu())
        return menubar

    def _sourceMenu(self):
        sourceMenu = QMenu('Source', self)
        sourceMenu.addAction(self._selectWebcam0Menu())
        sourceMenu.addAction(self._selectWebcam1Menu())
        sourceMenu.addAction(self._selectVideoMenu())
        sourceMenu.addAction(self._selectRtspMenu())
        return sourceMenu

    def _webcam0(self):
        self.statusBar.showMessage("Camera No.0 was selected")
        global strSrc
        strSrc = 0

    def _selectWebcam0Menu(self):
        selectWebcam0Menu = QAction("Select Webcam 0", self)
        selectWebcam0Menu.setShortcut('Ctrl+0')
        selectWebcam0Menu.setStatusTip('Open webcam 0')
        selectWebcam0Menu.triggered.connect(self._webcam0)
        return selectWebcam0Menu

    def _webcam1(self):
        self.statusBar.showMessage("Camera No.1 was selected")
        global strSrc
        strSrc = 1

    def _selectWebcam1Menu(self):
        selectWebcam1Menu = QAction("Select Webcam 1", self)
        selectWebcam1Menu.setShortcut('Ctrl+1')
        selectWebcam1Menu.setStatusTip('Open webcam 1')
        selectWebcam1Menu.triggered.connect(self._webcam1)
        return selectWebcam1Menu

    def _showDialog(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', '/home')
        if fname[0]:
            self.statusBar.showMessage("Selected video " + fname[0])
            global strSrc
            strSrc = fname[0]

    def _selectVideoMenu(self):
        selectVideoMenu = QAction('Open Video', self)
        selectVideoMenu.setShortcut('Ctrl+O')
        selectVideoMenu.setStatusTip('Open available video')
        selectVideoMenu.triggered.connect(self._showDialog)
        return selectVideoMenu

    def _showInputDialog(self):
        text, ok = QInputDialog.getText(self, 'Input Dialog', 'Enter RTSP link')
        if ok:
            global strSrc
            strSrc = text
            self.statusBar.showMessage("Ready to stream " + strSrc)

    def _selectRtspMenu(self):
        selectRtspMenu = QAction('Input RTSP link', self)
        selectRtspMenu.setShortcut('Ctrl+R')
        selectRtspMenu.setStatusTip('Stream RTSP to')
        selectRtspMenu.triggered.connect(self._showInputDialog)
        return selectRtspMenu

    def _closeEvent(self, event):
        # self.mainLayout.faceCompareUI.mVS.release()
        self.mainLayout.faceCompareUI.closeStream()
        self.mainLayout.faceRegisterUI.mVS.release()
        for i in range(10000000):
            pass

class MainLayout(QTabWidget):
    def __init__(self, parent):
        super(QTabWidget, self).__init__(parent)
        self.l = parent.x()
        self.t = parent.y()
        self.w = parent.width()
        self.h = parent.height()
        self.setGeometry(QRect(self.l, self.t, self.w, self.h * 0.95))
        self.initUI()

    def initUI(self):
        self.faceCompareUI = FaceCompareUI(self)
        self.addTab(self.faceCompareUI, "Recognition")
        self.faceRegisterUI = FaceRegisterUI(self)
        self.addTab(self.faceRegisterUI, "Register")

class FaceCompareUI(QGroupBox):
    def __init__(self, parent):
        super(QGroupBox, self).__init__(parent)
        self.l = parent.x()
        self.t = parent.y()
        self.h = parent.height()
        self.w = parent.width()
        self.timer = QTimer()
        self.flipFrame = True
        self.threadUpdateStream = QThreadPool()
        self.threadUpdateResult = QThreadPool()
        self.mVS = MVideoStreamer()
        self.flagFaceDetection = 0
        self.listObject = []
        self.timer.timeout.connect(self.timeout)
        if self.h < 800:
            self.rowCompareResultUI = 4
        else:
            self.rowCompareResultUI = 5
        self.initTabUI()

    def initTabUI(self):
        tabLayout = QHBoxLayout(self)
        tabLayout.addWidget(self.compareResultUI())
        tabLayout.addWidget(self.streamUI())
        tabLayout.addWidget(self.unknownResultUI())
        self.setLayout(tabLayout)

    def compareResultUI(self):
        column1 = QGroupBox()
        column1.setGeometry(self.l, self.t, self.w * 0.4, self.h)
        column1.setMaximumWidth(self.w * 0.15)
        # column1.setMaximumHeight(self.h)
        layout = QVBoxLayout(column1)
        self.listIDShow = [None] * self.rowCompareResultUI
        self.listFaceCropped = [None] * self.rowCompareResultUI
        self.listLabelImg = []
        self.listGroupFace = []
        self.flag = False
        for i in range(self.rowCompareResultUI):
            groupFace = QGroupBox(str(i + 1))
            self.listGroupFace.append(groupFace)
            groupFaceLayout = QHBoxLayout(groupFace)
            labelImg = QLabel(groupFace)
            self.listLabelImg.append(labelImg)
            groupFaceLayout.addWidget(labelImg)
            groupFaceLayout.setAlignment(Qt.AlignCenter)
            groupFace.setLayout(groupFaceLayout)
            layout.addWidget(groupFace)

        camBox = QGroupBox("Control camera")
        camBox.setMaximumHeight(self.h * 0.1)
        camBoxLayout = QHBoxLayout(camBox)
        self.flipButton = QRadioButton("Flip")
        self.flipButton.setChecked(True)
        self.flipButton.toggled.connect(self.onFlipClicked)
        self.resetButn = QPushButton("Clear")
        self.resetButn.setMinimumHeight(self.w * 0.03)
        self.resetButn.clicked.connect(self.resetButnHandle)
        self.camButn = QPushButton("Open Camera")
        self.camButn.setMinimumHeight(self.w * 0.03)
        self.camButn.clicked[bool].connect(self.OStream)
        self.camButn.setCheckable(True)
        camBoxLayout.addWidget(self.flipButton)
        camBoxLayout.addWidget(self.resetButn)
        camBoxLayout.addWidget(self.camButn)
        camBox.setLayout(camBoxLayout)

        layout.addWidget(camBox)
        column1.setLayout(layout)
        return column1
    
    def unknownResultUI(self):
        column1 = QGroupBox("UNKNOWN")
        column1.setGeometry(self.l, self.t, self.w * 0.2, self.h)
        column1.setMaximumWidth(self.w * 0.15)
        # column1.setMaximumHeight(self.h)
        layout = QVBoxLayout(column1)
        self.listUnknownIDShow = [None] * self.rowCompareResultUI
        self.listUnknownFaceCropped = [None] * self.rowCompareResultUI
        self.listUnknownLabelImg = []
        self.listUnknownGroupFace = []
        self.flag = False
        for i in range(self.rowCompareResultUI):
            groupFace = QGroupBox(str(i + 1))
            self.listUnknownGroupFace.append(groupFace)
            groupFaceLayout = QHBoxLayout(groupFace)
            labelImg = QLabel(groupFace)
            self.listUnknownLabelImg.append(labelImg)
            groupFaceLayout.addWidget(labelImg)
            groupFaceLayout.setAlignment(Qt.AlignCenter)
            groupFace.setLayout(groupFaceLayout)
            layout.addWidget(groupFace)

        column1.setLayout(layout)
        return column1    
    
    def streamUI(self):
        column2 = QGroupBox()
        column2.setGeometry(self.l, self.t, self.w * 0.6, self.h)
        layout = QHBoxLayout(column2)
        self.labelVideo = QLabel(column2)
        layout.addWidget(self.labelVideo)
        layout.setAlignment(Qt.AlignCenter)
        column2.setLayout(layout)
        return column2

    def onFlipClicked(self):
        radioButton = self.sender()
        if radioButton.isChecked():
            self.flipFrame = True
        else:
            self.flipFrame = False

    def resetButnHandle(self):
        del(self.listIDShow)
        self.listIDShow = [None] * self.rowCompareResultUI
        del(self.listFaceCropped)
        self.listFaceCropped = [None] * self.rowCompareResultUI
        for i in range(self.rowCompareResultUI):
            self.listGroupFace[i].setTitle(str(i + 1))
            self.listLabelImg[i].clear()
            self.listUnknownGroupFace[i].setTitle(str(i + 1))
            self.listUnknownLabelImg[i].clear()

    def OStream(self, pressed):
        if pressed:
            self.openStream()
        else:
            self.closeStream()

    def openStream(self):
        if strSrc is not None:
            if not self.mVS.releaseFlag:
                self.mVS.release()
            self.mVS.newStrSource(strSrc)
            worker = UpdateWorker(self.mVS.update)
            worker.signals.finished.connect(self.timer.stop)
            self.resetButnHandle()
            self.threadUpdateStream.start(worker)
            self.timer.start(5)
            self.camButn.setText("Close camera")
        else:
            self.camButn.toggle()
            msgBox = QMessageBox.information(
                self, "Notification", "There was no Streamming Source selected")

    def closeStream(self):
        self.mVS.release()
        self.camButn.setText("Open camera")
        self.flagFaceDetection = 0
        for i in range(len(self.listObject) - 1, -1, -1):
            self.listObject[i].deleteTracker()
            del(self.listObject[i])

    def core(self, resizedFrame, frame=None, h_ratio=1., w_ratio=1.):
        if frame is None:
            frame = np.copy(resizedFrame)

        frameDisplay = np.copy(frame)
        if self.flagFaceDetection == 0:
            bounding_boxes, landmarks, confidences = mFD.detect_faces(resizedFrame)
            self.flagFaceDetection = RESET_FLAG_FACE_DETECT
        else:
            bounding_boxes, landmarks, confidences = None, None, None
            self.flagFaceDetection -= 1

        for obj in self.listObject:
            obj.updateTracker(resizedFrame, None)

        if bounding_boxes is not None:
            overlap = [False] * len(bounding_boxes)
            for idx in range(len(self.listObject)):
                ovlap = False
                for j, bbox in enumerate(bounding_boxes):
                    _bbox = xyxy2xywh(bbox)
                    if IoU(_bbox, self.listObject[idx].bbox) > IOU_THRESHOLD:
                        ovlap = True
                        overlap[j] = True
                        _b = bounding_boxes[j]
                        landm = landmarks[j]
                        confd = confidences[j]
                        self.listObject[idx].updateTracker(resizedFrame, xyxy2xywh(_b), score=confd, landmark=landm, originFrame=frame, h_ratio=h_ratio, w_ratio=w_ratio)
                        break

                if not ovlap:
                    self.listObject[idx].state = False

            for i in range(len(overlap)):
                if not overlap[i]:
                    bbox = bounding_boxes[i]
                    landm = landmarks[i]
                    confd = confidences[i]
                    newObj = Object()
                    newObj.updateTracker(resizedFrame, xyxy2xywh(bbox), score=confd, landmark=landm, originFrame=frame, h_ratio=h_ratio, w_ratio=w_ratio)
                    self.listObject.append(newObj)
        else:
            if self.flagFaceDetection == RESET_FLAG_FACE_DETECT:
                for idx in range(len(self.listObject)):
                    self.listObject[idx].state = False

        for idx in range(len(self.listObject) - 1, -1, -1):
            if not self.listObject[idx].state:
                self.listObject[idx].deleteTracker()
                cropped, aligned = self.listObject[idx].returnBestFace()
                if cropped is not None:
                    frWorker = ResultWorker(recognizeToReturnID, cropped=cropped, aligned=aligned, people=people)
                    frWorker.signals.finished.connect(self.updateResult)
                    self.threadUpdateResult.start(frWorker)
                    # _id = recognizeToReturnID(cropped, aligned, people)
                del(self.listObject[idx])
            elif len(self.listObject[idx].scoreHistory) >= 20:
                cropped, aligned = self.listObject[idx].returnBestFace()
                frWorker = ResultWorker(recognizeToReturnID, cropped=cropped, aligned=aligned, people=people)
                frWorker.signals.finished.connect(self.updateResult)
                self.threadUpdateResult.start(frWorker)
                # _id = recognizeToReturnID(cropped, aligned, people)
                del(self.listObject[idx].scoreHistory)
                del(self.listObject[idx].faceHistory)
                del(self.listObject[idx].landmsHistory)
                self.listObject[idx].faceHistory = []
                self.listObject[idx].landmsHistory = []
                self.listObject[idx].scoreHistory = []
                x1, y1, x2, y2 = xywh2xyxy(self.listObject[idx].bbox)
                cv2.rectangle(frameDisplay, (int(x1 * w_ratio), int(y1 * h_ratio)), (int(x2 * w_ratio), int(y2 * h_ratio)), (0, 0, 255), 2)
            else:
                x1, y1, x2, y2 = xywh2xyxy(self.listObject[idx].bbox)
                cv2.rectangle(frameDisplay, (int(x1 * w_ratio), int(y1 * h_ratio)), (int(x2 * w_ratio), int(y2 * h_ratio)), (0, 0, 255), 2)

        return frameDisplay

    def updateResult(self, result):
        _id, dist, emb, cropped = result
        if _id == "UNKNOWN":
            _id = "NEW_{}".format(len(people))
            if not len(emb) == 0:
                people[_id] = emb[None]
                with open(os.path.join(DATABASE, "{}.txt".format(_id)), mode="w", encoding="utf-8") as f:
                    emb = [str(e) for e in emb]
                    emb = ",".join(emb) + "\n"
                    f.write(emb)
        elif _id.startswith("NEW_"):
            if dist > 0.67 * RECOGNIZE_THRESHOLD:
                people[_id] = np.vstack((people[_id], emb[None]))
                with open(os.path.join(DATABASE, "{}.txt".format(_id)), mode="a", encoding="utf-8") as f:
                    emb = [str(e) for e in emb]
                    emb = ",".join(emb) + "\n"
                    f.write(emb)
        else:
            pass


        if _id.startswith("NEW_"):
            if _id in self.listUnknownIDShow:
                idx = self.listUnknownIDShow.index(_id)
                del(self.listUnknownIDShow[idx])
                del(self.listUnknownFaceCropped[idx])
            else:
                del(self.listUnknownIDShow[-1])
                del(self.listUnknownFaceCropped[-1])

            self.listUnknownIDShow.insert(0, _id)
            self.listUnknownFaceCropped.insert(0, cropped)
            
            for idx in range(self.rowCompareResultUI):
                _id = self.listUnknownIDShow[idx]
                if _id is not None:
                    self.listUnknownGroupFace[idx].setTitle(_id)
                    face = cv2.cvtColor(self.listUnknownFaceCropped[idx], cv2.COLOR_BGR2RGB)
                    qImg = QImage(face.data, 112, 112, 336, QImage.Format_RGB888)
                    pix = QPixmap(qImg.scaled(112, 112))
                    self.listUnknownLabelImg[idx].setPixmap(pix)
                else:
                    self.listUnknownGroupFace[idx].setTitle(str(idx + 1))
                    self.listUnknownLabelImg[idx].clear()
        else:
            if _id in self.listIDShow:
                idx = self.listIDShow.index(_id)
                del(self.listIDShow[idx])
                del(self.listFaceCropped[idx])
            else:
                del(self.listIDShow[-1])
                del(self.listFaceCropped[-1])

            self.listIDShow.insert(0, _id)
            self.listFaceCropped.insert(0, cropped)
            
            for idx in range(self.rowCompareResultUI):
                _id = self.listIDShow[idx]
                if _id is not None:
                    self.listGroupFace[idx].setTitle(_id)
                    face = cv2.cvtColor(self.listFaceCropped[idx], cv2.COLOR_BGR2RGB)
                    qImg = QImage(face.data, 112, 112, 336, QImage.Format_RGB888)
                    pix = QPixmap(qImg.scaled(112, 112))
                    self.listLabelImg[idx].setPixmap(pix)
                else:
                    self.listGroupFace[idx].setTitle(str(idx + 1))
                    self.listLabelImg[idx].clear()

    def timeout(self):
        status, frame = self.mVS.read()
        if status is None:
            return
        elif not status:
            msgBox = QMessageBox.information(
                self, "Notification", "Reading frame from Stream was failed!")
            return
        else:
            if self.flipFrame:
                frame = cv2.flip(frame, 1)
            frameHeight, frameWidth, _ = frame.shape
            frameRatio = frameHeight / float(frameWidth)
            resizedFrame = imutils.resize(frame, HEIGHT_FRAME)
            resizedFrameHeight, resizedFrameWidth, _ = resizedFrame.shape
            h_ratio = float(frameHeight) / resizedFrameHeight
            w_ratio = float(frameWidth) / resizedFrameWidth

            frameDisplay = self.core(resizedFrame, frame, h_ratio, w_ratio)

            height, width, channel = frameDisplay.shape
            bytesPerLine = 3 * width
            frameDisplay = cv2.cvtColor(frameDisplay, cv2.COLOR_BGR2RGB)
            qImg = QImage(
                frameDisplay.data,
                width,
                height,
                bytesPerLine,
                QImage.Format_RGB888)
            hh = self.h * 0.7
            qImg = qImg.scaled(int(hh / frameRatio), int(hh))
            pix = QPixmap(qImg)
            self.labelVideo.setPixmap(pix)
            self.timer.start(1)

class FaceRegisterUI(QGroupBox):
    signal_facebook = pyqtSignal(list)

    def __init__(self, parent):
        super(QGroupBox, self).__init__(parent)
        self.l = parent.x()
        self.t = parent.y()
        self.h = parent.height()
        self.w = parent.width()
        self.timer = QTimer()
        self.flipFrame = True
        self.objRegister = None
        self.maxNoFaceR = 20
        self.flagFaceDetection = RESET_FLAG_FACE_DETECT * 3
        self.threadUpdateStream = QThreadPool()
        self.signal_facebook.connect(self.facebook)
        self.mVS = MVideoStreamer()
        self.timer.timeout.connect(self.timeout)
        self.initTabUI()

    def initTabUI(self):
        tabLayout = QHBoxLayout(self)
        videoBox = QGroupBox("Register")
        # column2.setGeometry(self.l, self.t, self.w * 0.6, self.h)
        videoBoxLayout = QHBoxLayout(videoBox)
        self.labelVideo = QLabel(self)
        videoBoxLayout.addWidget(self.labelVideo)
        videoBoxLayout.setAlignment(Qt.AlignCenter)
        videoBox.setLayout(videoBoxLayout)
        control = QGroupBox("Control")
        control.setMinimumWidth(self.w * 0.3)
        control.setMaximumWidth(self.w * 0.31)
        controlLayout = QVBoxLayout(control)
        
        faceFramesControl = QGroupBox()
        faceFramesControl.setMinimumHeight(self.h * 0.71)
        faceFramesControlLayout = QGridLayout(faceFramesControl)

        self.faceFrame = []
        for idx in range(self.maxNoFaceR):
            qLabel = QLabel(self)
            faceFramesControlLayout.addWidget(qLabel, idx // 4, idx % 4, 1, 1)
            self.faceFrame.append(qLabel)
        
        faceFramesControlLayout.setAlignment(Qt.AlignCenter)
        faceFramesControl.setLayout(faceFramesControlLayout)

        controlLayout.addWidget(faceFramesControl)

        controlLayout.addWidget(QLabel("ID:"))
        self.idLineEdit = QLineEdit("", self)
        self.flipButton = QRadioButton("Flip frame")
        self.flipButton.setChecked(True)
        self.flipButton.toggled.connect(self.onFlipClicked)
        self.camButn = QPushButton("Open camera")
        self.camButn.setCheckable(True)
        self.camButn.setMinimumHeight(self.h * 0.12)
        self.camButn.clicked[bool].connect(self.OStream)
        controlLayout.addWidget(self.idLineEdit)
        controlLayout.addWidget(self.flipButton)
        controlLayout.addWidget(self.camButn)
        control.setLayout(controlLayout)

        tabLayout.addWidget(videoBox)
        tabLayout.addWidget(control)
        self.setLayout(tabLayout)

    def onFlipClicked(self):
        radioButton = self.sender()
        if radioButton.isChecked():
            self.flipFrame = True
        else:
            self.flipFrame = False

    def resetButnHandle(self):
        del(self.objRegister)
        self.objRegister = None
        self.flagFaceDetection = 0
        self.noDetectFailed = 3
        for i in range(self.maxNoFaceR):
            self.faceFrame[i].clear()

    def OStream(self, pressed):
        if pressed:
            self.openStream()
        else:
            self.closeStream()

    def openStream(self):
        if strSrc is not None:
            if not self.idLineEdit.text() == "":
                if not self.mVS.releaseFlag:
                    self.mVS.release()
                self.mVS.newStrSource(strSrc)
                worker = UpdateWorker(self.mVS.update)
                worker.signals.finished.connect(self.timer.stop)
                self.threadUpdateStream.start(worker)
                self.resetButnHandle()
                self.timer.start(5)
                self.camButn.setText("Close camera")
            else:
                self.camButn.toggle()
                msgBox = QMessageBox.information(
                    self, "Notification", "Input username, please!")
        else:
            self.camButn.toggle()
            msgBox = QMessageBox.information(
                self, "Notification", "There was no Streamming Source selected")

    def closeStream(self):
        self.mVS.release()
        self.camButn.setText("Open camera")
        if self.objRegister is not None:
            if len(self.objRegister.faceHistory) > 0:
                self.register()
            self.objRegister.deleteTracker()

    def facebook(self, facebook):
        idx, face = facebook
        if idx >= self.maxNoFaceR:
            pass
        else:
            face = cv2.resize(face, (112, 112))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            qImg = QImage(face.data, 112, 112, 336, QImage.Format_RGB888)
            pix = QPixmap(qImg.scaled(112, 112))
            self.faceFrame[idx].setPixmap(pix)

    def register(self):
        listFace = self.objRegister.returnListFace()
        
        if listFace is None:
            msgBox = QMessageBox.information(
                self, "Notification", "There is not enough image for register (more than 3)")
        else:
            username = self.idLineEdit.text()
            with open(os.path.join(DATABASE, username + ".txt"), mode="w", encoding="utf-8") as f:
                embs = []
                for i in tqdm(range(len(listFace))):
                    # emb = mFR.get_feature(aligned=listFace[i])
                    # emb = m_get_embedding_MMS(aligned=listFace[i])
                    emb = m_get_embedding_Flask(aligned=listFace[i])
                    embs.append(emb)
                    emb = [str(e) for e in emb]
                    emb = ",".join(emb) + "\n"
                    f.write(emb)
                
            people[username] = np.array(embs, dtype=np.float32)
            _key = []
            for key in people.keys():
                if key.startswith("NEW_"):
                    for e in people[key]:
                        dist = distance(e, people[username])
                        if dist < RECOGNIZE_THRESHOLD:
                            _key.append(key)
                            os.remove(os.path.join(DATABASE, key + ".txt"))
                            break
            
            for k in _key:
                del(people[k])

            msgBox = QMessageBox.information(
                self, "Notification", "{} was registered successfully".format(username))

    def core(self, resizedFrame, frame=None, h_ratio=1., w_ratio=1.):
        if frame is None:
            frame = np.copy(resizedFrame)

        frameDisplay = np.copy(frame)

        if self.flagFaceDetection == 0:
            bounding_boxes, landmarks, confidences = mFD.detect_faces(resizedFrame)
            self.flagFaceDetection = RESET_FLAG_FACE_DETECT * 3
        else:
            bounding_boxes, landmarks, confidences = None, None, None
            self.flagFaceDetection -= 1

        if self.objRegister is not None:
            if self.objRegister.state:
                self.objRegister.updateTracker(resizedFrame, None)

        if bounding_boxes is not None:
            if self.objRegister is None:
                # import ipdb; ipdb.set_trace()
                centerFrame = np.array([resizedFrame.shape[1], resizedFrame.shape[0]]) / 2.0
                center = (bounding_boxes[:, [0, 1]] + bounding_boxes[:, [2, 3]]) / 2.0
                dist = []
                for c in center:
                    dist.append(distance(c, centerFrame))
                idx = dist.index(min(dist))
                landm = landmarks[idx]
                confd = confidences[idx]
                bbox = bounding_boxes[idx]
                self.objRegister = Object()
                self.objRegister.updateTracker(resizedFrame, xyxy2xywh(bbox), score=confd, landmark=landm, originFrame=frame, h_ratio=h_ratio, w_ratio=w_ratio)
                self.signal_facebook.emit([0, self.objRegister.faceHistory[-1]])
            else:
                if self.objRegister.state:
                    overlap = False
                    for j, bbox in enumerate(bounding_boxes):
                        _bbox = xyxy2xywh(bbox)
                        if IoU(_bbox, self.objRegister.bbox) > IOU_THRESHOLD:
                            overlap = True
                            _b = bounding_boxes[j]
                            landm = landmarks[j]
                            confd = confidences[j]
                            self.objRegister.updateTracker(resizedFrame, xyxy2xywh(_b), score=confd, landmark=landm, originFrame=frame, h_ratio=h_ratio, w_ratio=w_ratio)
                            self.signal_facebook.emit([len(self.objRegister.faceHistory) - 1, self.objRegister.faceHistory[-1]])
                            break

                    if not overlap:
                        self.objRegister.state = False

                    if len(self.objRegister.faceHistory) >= self.maxNoFaceR:
                        self.objRegister.state = False
        else:
            if self.flagFaceDetection == RESET_FLAG_FACE_DETECT * 3:
                self.noDetectFailed -= 1
                if self.noDetectFailed == 0:
                    self.objRegister.state = False
            else:
                self.noDetectFailed = 3

        if self.objRegister is not None:
            if not self.objRegister.state:
                self.objRegister.deleteTracker()
            else:
                x1, y1, x2, y2 = xywh2xyxy(self.objRegister.bbox)
                cv2.rectangle(frameDisplay, (int(x1 * w_ratio), int(y1 * h_ratio)), (int(x2 * w_ratio), int(y2 * h_ratio)), (0, 0, 255), 2)

        return frameDisplay

    def timeout(self):
        status, frame = self.mVS.read()
        if status is None:
            return
        elif not status:
            msgBox = QMessageBox.information(
                self, "Notification", "Reading frame from Stream was failed!")
            return
        else:
            if self.flipFrame:
                frame = cv2.flip(frame, 1)
            frameHeight, frameWidth, _ = frame.shape
            frameRatio = frameHeight / float(frameWidth)
            resizedFrame = imutils.resize(frame, HEIGHT_FRAME)
            resizedFrameHeight, resizedFrameWidth, _ = resizedFrame.shape
            h_ratio = float(frameHeight) / resizedFrameHeight
            w_ratio = float(frameWidth) / resizedFrameWidth

            frameDisplay = self.core(resizedFrame, frame, h_ratio, w_ratio)

            height, width, channel = frameDisplay.shape
            bytesPerLine = 3 * width
            frameDisplay = cv2.cvtColor(frameDisplay, cv2.COLOR_BGR2RGB)
            qImg = QImage(
                frameDisplay.data,
                width,
                height,
                bytesPerLine,
                QImage.Format_RGB888)
            if frameRatio < 0.66666666666667:
                hh = self.h * 0.7
            else:
                hh = self.h * 0.8
            qImg = qImg.scaled(int(hh / frameRatio), int(hh))
            pix = QPixmap(qImg)
            self.labelVideo.setPixmap(pix)
            self.timer.start(1)


def main():
    img = np.random.randint(0, 256, size = (112, 112, 3), dtype=np.uint8)
    for _ in tqdm(range(30)):
        mFD.detect_faces(img)
        # mFR.get_feature(img)
        # m_get_embedding_MMS(img)
        m_get_embedding_Flask(img)
    del(img)

    app = QApplication(sys.argv)
    window = MainWindow(app)
    window.show()
    sys.exit(app.exec_())

# ====================================================================
if __name__ == "__main__":
    main()
