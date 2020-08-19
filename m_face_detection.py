from __future__ import print_function

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import os
import cv2
from retinaface.data import cfg_mnet
from retinaface.models.retinaface import RetinaFace
from retinaface.layers.functions.prior_box import PriorBox
from retinaface.utils.box_utils import decode, decode_landm
from retinaface.utils.nms.py_cpu_nms import py_cpu_nms
from skimage import transform as trans
from uuid import uuid4
from glob import glob
from m_config import *

# reference facial points, a list of coordinates (x,y)
REFERENCE_FACIAL_POINTS = [
    [30.29459953, 51.69630051],
    [65.53179932, 51.50139999],
    [48.02519989, 71.73660278],
    [33.54930115, 92.3655014],
    [62.72990036, 92.20410156]
]

DEFAULT_CROP_SIZE = (112, 112)


def get_reference_facial_points(output_size=None,
                                inner_padding_factor=0.0,
                                outer_padding=(0, 0),
                                default_square=False):

    tmp_5pts = np.array(REFERENCE_FACIAL_POINTS)
    tmp_crop_size = np.array(DEFAULT_CROP_SIZE)

    if default_square:
        size_diff = max(tmp_crop_size) - tmp_crop_size
        tmp_5pts += size_diff / 2
        tmp_crop_size += size_diff

    if (output_size and
            output_size[0] == tmp_crop_size[0] and
            output_size[1] == tmp_crop_size[1]):
        return tmp_5pts

    if (inner_padding_factor == 0 and outer_padding == (0, 0)):
        if output_size is None:
            print('No paddings to do: return default reference points')
            return tmp_5pts
        else:
            raise FaceWarpException(
                'No paddings to do, output_size must be None or {}'.format(tmp_crop_size))

    if not (0 <= inner_padding_factor <= 1.0):
        raise FaceWarpException('Not (0 <= inner_padding_factor <= 1.0)')

    if ((inner_padding_factor > 0 or outer_padding[0] > 0 or outer_padding[1] > 0) and output_size is None):
        output_size = tmp_crop_size * (1 + inner_padding_factor * 2).astype(np.int32)
        output_size += np.array(outer_padding)
        print('deduced from paddings, output_size = ', output_size)

    if not (outer_padding[0] < output_size[0] and outer_padding[1] < output_size[1]):
        raise FaceWarpException('Not (outer_padding[0] < output_size[0]'
                                'and outer_padding[1] < output_size[1])')

    if inner_padding_factor > 0:
        size_diff = tmp_crop_size * inner_padding_factor * 2
        tmp_5pts += size_diff / 2
        tmp_crop_size += np.round(size_diff).astype(np.int32)

    size_bf_outer_pad = np.array(output_size) - np.array(outer_padding) * 2

    if size_bf_outer_pad[0] * tmp_crop_size[1] != size_bf_outer_pad[1] * tmp_crop_size[0]:
        raise FaceWarpException('Must have (output_size - outer_padding)'
                                '= some_scale * (crop_size * (1.0 + inner_padding_factor)')

    scale_factor = size_bf_outer_pad[0].astype(np.float32) / tmp_crop_size[0]
    tmp_5pts = tmp_5pts * scale_factor
    tmp_crop_size = size_bf_outer_pad
    reference_5point = tmp_5pts + np.array(outer_padding)
    tmp_crop_size = output_size

    return reference_5point

def get_affine_transform_matrix(src_pts, dst_pts):
    tfm = np.float32([[1, 0, 0], [0, 1, 0]])
    n_pts = src_pts.shape[0]
    ones = np.ones((n_pts, 1), src_pts.dtype)
    src_pts_ = np.hstack([src_pts, ones])
    dst_pts_ = np.hstack([dst_pts, ones])

    A, res, rank, s = np.linalg.lstsq(src_pts_, dst_pts_)

    if rank == 3:
        tfm = np.float32([
            [A[0, 0], A[1, 0], A[2, 0]],
            [A[0, 1], A[1, 1], A[2, 1]]
        ])
    elif rank == 2:
        tfm = np.float32([
            [A[0, 0], A[1, 0], 0],
            [A[0, 1], A[1, 1], 0]
        ])

    return tfm

def warp_and_crop_face(src_img,
                       facial_pts,
                       reference_pts=None,
                       crop_size=(96, 112),
                       align_type='smilarity'):
    if reference_pts is None:
        if crop_size[0] == 96 and crop_size[1] == 112:
            reference_pts = REFERENCE_FACIAL_POINTS
        else:
            default_square = False
            inner_padding_factor = 0
            outer_padding = (0, 0)
            output_size = crop_size

            reference_pts = get_reference_facial_points(output_size,
                                                        inner_padding_factor,
                                                        outer_padding,
                                                        default_square)

    ref_pts = np.float32(reference_pts)
    ref_pts_shp = ref_pts.shape
    if max(ref_pts_shp) < 3 or min(ref_pts_shp) != 2:
        raise FaceWarpException(
            'reference_pts.shape must be (K,2) or (2,K) and K>2')

    if ref_pts_shp[0] == 2:
        ref_pts = ref_pts.T

    src_pts = np.float32(facial_pts)
    src_pts_shp = src_pts.shape
    if max(src_pts_shp) < 3 or min(src_pts_shp) != 2:
        raise FaceWarpException(
            'facial_pts.shape must be (K,2) or (2,K) and K>2')

    if src_pts_shp[0] == 2:
        src_pts = src_pts.T

    if src_pts.shape != ref_pts.shape:
        raise FaceWarpException(
            'facial_pts and reference_pts must have the same shape')

    if align_type is 'cv2_affine':
        tfm = cv2.getAffineTransform(src_pts[0:3], ref_pts[0:3])
    elif align_type is 'affine':
        tfm = get_affine_transform_matrix(src_pts, ref_pts)
    else:
        tform = trans.SimilarityTransform()
        tform.estimate(src_pts, ref_pts)
        tfm = tform.params[0:2, :]

    face_img = cv2.warpAffine(src_img, tfm, (crop_size[0], crop_size[1]))

    return face_img  # BGR

def align_face(img, facial5points): # img in BGR format, facial5points in size (5, 2)
    facial5points = facial5points.T

    crop_size = (112, 112)
    default_square = True
    inner_padding_factor = 0.25
    outer_padding = (0, 0)
    output_size = (112, 112)

    # get the reference 5 landmarks position in the crop settings
    reference_5pts = get_reference_facial_points(
        output_size, inner_padding_factor, outer_padding, default_square)

    # dst_img = warp_and_crop_face(raw, facial5points)
    dst_img = warp_and_crop_face(img, facial5points, reference_pts=reference_5pts, crop_size=crop_size)
    return dst_img


class MFaceDetector:
    def __init__(self, model_dir):
        cudnn.benchmark = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # device = torch.device('cpu')
        self.model = self.load_model(model_dir).to(self.device)
        self.model.eval()

    def check_keys(self, model, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(model.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys
        unused_pretrained_keys = ckpt_keys - model_keys
        missing_keys = model_keys - ckpt_keys
        # print('Missing keys:{}'.format(len(missing_keys)))
        # print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
        # print('Used keys:{}'.format(len(used_pretrained_keys)))
        assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
        return True

    def remove_prefix(self, state_dict, prefix):
        ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
        # print('remove prefix \'{}\''.format(prefix))
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}

    def load_model(self, model_dir):
        # print('Loading pretrained model from {}'.format(model_dir))
        model = RetinaFace(cfg=cfg_mnet, phase='test')

        if torch.cuda.is_available():
            pretrained_dict = torch.load(model_dir, map_location=lambda storage, loc: storage.cuda(self.device))
        else:
            pretrained_dict = torch.load(model_dir, map_location=lambda storage, location: storage)
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = self.remove_prefix(pretrained_dict['state_dict'], 'module.')
        else:
            pretrained_dict = self.remove_prefix(pretrained_dict, 'module.')
        self.check_keys(model, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
        # print('Finished loading model!')
        return model

    def detect_faces(self, img_raw, confidence_threshold=0.9):  # img_raw in BRG format in size (h, w, 3)
        # default parameters
        top_k=5000
        keep_top_k=750
        nms_threshold=0.4
        resize=1

        img = np.float32(img_raw)
        im_height, im_width = img.shape[:2]
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)

        # tic = time.time()
        with torch.no_grad():
            loc, conf, landms = self.model(img)  # forward pass
            # print('net forward time: {:.4f}'.format(time.time() - tic))

        priorbox = PriorBox(cfg_mnet, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg_mnet['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg_mnet['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:keep_top_k, :]
        landms = landms[:keep_top_k, :]
        # print(landms.shape)
        landms = landms.reshape((-1, 5, 2))
        # print(landms.shape)
        landms = landms.transpose((0, 2, 1))
        # print(landms.shape)
        landms = landms.reshape(-1, 10, )
        # print(landms.shape)

        # bounding_boxes:   [[x1, y1, x2, y2]]
        # landmarks     :   [[x1, x2, x3, x4, x5, y1, y2, y3, y4, y5]]
        # confidences   :   [c]
        bounding_boxes= np.ones_like(dets[:, :4])
        for i, d in enumerate(dets[:, :4]):
            _x1, _y1, _x2, _y2 = d
            x1 = (1. + PADDING_FACE) * _x1 - PADDING_FACE * _x2
            x1 = 0 if x1 < 0 else x1
            x2 = (1. + PADDING_FACE) * _x2 - PADDING_FACE * _x1
            x2 = im_width if x2 > im_width else x2
            y1 = (1. + PADDING_FACE) * _y1 - PADDING_FACE * _y2
            y1 = 0 if y1 < 0 else y1
            y2 = (1. + PADDING_FACE) * _y2 - PADDING_FACE * _y1
            y2 = im_height if y2 > im_height else y2
            bounding_boxes[i] = np.array([x1, y1, x2, y2])
        # bounding_boxes = bounding_boxes.astype(np.uint32)
        tl = bounding_boxes[:, :2]
        tl = np.tile(tl.reshape((-1, 2, 1)), 5)
        # landmarks = landms.astype(np.uint32).reshape((-1, 2, 5)) - tl
        landmarks = landms.reshape((-1, 2, 5)) - tl
        confidences = dets[:, -1]
        if len(bounding_boxes) == 0:
            return None, None, None
        else:
            return bounding_boxes, landmarks, confidences

    def detect_faces_from_directory(self, directory):
        print('reading %s' % directory)
        fname = glob(os.path.join(directory, "*.jpg"))
        fname.extend(glob(os.path.join(directory, "*.png")))
        for i, fn in enumerate(fname):
            img = cv2.imread(fn)
            bounding_boxes, landmarks, confidences = mFD.detect_faces(img)
            if bounding_boxes is not None:
                for bbox, landm, conf in zip(bounding_boxes, landmarks, confidences):
                    x1, y1, x2, y2 = bbox.astype(np.uint32)
                    face_img = img[y1:y2, x1:x2]
                    points = landm.reshape((2, 5))
                    face_img = align_face(face_img, points)
                    cv2.imwrite(os.path.join(directory, uuid4().hex + ".jpg"), face_img)

            print("delete", fn)
            os.remove(fn)




if __name__ == "__main__":
    mFD = MFaceDetector('models/retinaface_mobinet/mobilenet_0_25_Final.pth')

    # mFD.detect_faces_from_directory(f"D:/Me/M/m_database/HanhHuy")

    img = np.ones((1000,1000,3), dtype=np.uint8)
    # img = cv2.imread("b.png", cv2.IMREAD_COLOR)
    # # img = np.ones((112, 112, 3), dtype=np.uint8)
    bounding_boxes, landmarks, confidences = mFD.detect_faces(img)
    import ipdb; ipdb.set_trace()

    # img_copy = np.copy(img)
    # for i in bounding_boxes:
    #     x1, y1, x2, y2 = i
    #     cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # cv2.imwrite("m.png", img_copy)
    
    # for bbox, landm, conf in zip(bounding_boxes, landmarks, confidences):
    #     x1, y1, x2, y2 = bbox
    #     face_img = img_copy[y1:y2, x1:x2]
    #     points = landm.reshape((2, 5))
    #     face_img = align_face(face_img, points)

    #     # for i, p in enumerate(points):
    #     #     x, y = p
    #     #     cv2.circle(face_img, (x, y), 3, (0, 0, 255), 2)

    #     cv2.imwrite("m.png", face_img)


