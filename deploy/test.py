import face_model
import argparse
import cv2
import sys
import numpy as np
import time
from pypinyin import pinyin, lazy_pinyin, Style

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='../models/model-r34-amf/model,0', help='path to load model.')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()


def cos_sim_mat(Matrix, B):
    """计算矩阵中每个行向量与另一个行向量的余弦相似度
    注册特征中只有一张图片的512维特征，会存在问题，需要reshape函数保证特征是(1x512)"""

    # 点乘
    num = np.dot(Matrix, B.T)

    # 范式
    denom = np.linalg.norm(Matrix.reshape(-1, 512), axis=1, keepdims=True) * np.linalg.norm(B)
    denom = denom.reshape(-1)

    # range [-1, 1]
    cos_val = num / denom

    # range [0, 1]
    sim = 0.5 + 0.5 * cos_val
    return sim


def person_recog(model, img, person_id_list, person_face_feature):
    """函数功能：先本地识别，识别不了，百度识别"""
    result = model.get_input_multi(img)
    if result is None:
        return None

    else:
        bboxs, aligneds = result
        for i in range(bboxs.shape[0]):
            bbox = bboxs[i, 0:4]
            height = bbox[3] - bbox[1]
            weight = bbox[2] - bbox[0]

            if weight*height < 60*60:
                continue
            print("person box (w, h)", weight, height)
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 1)
            img_f = aligneds[i]
            feat = model.get_feature(img_f)

            # print(person_face_feature.shape, feat.shape)
            # compare feat of (512,) to (size, 512) person features
            sim = cos_sim_mat(person_face_feature, feat)
            max_sim = max(sim)
            person_id_sim = "no"
            pinyin_str = "no"
            if max_sim > 0.74:
                person_id_sim = person_id_list[np.argmax(sim)]
                # cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                tmp = lazy_pinyin(person_id_sim)
                print(max_sim, person_id_sim, tmp)
                pinyin_str = ""
                for ii in range(len(tmp)):
                    pinyin_str += tmp[ii][0]

            cv2.putText(img, pinyin_str, (int(bbox[0]), int(bbox[1])+10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
        return img, person_id_list, person_face_feature


model = face_model.FaceModel(args)
img = cv2.imread('Tom_Hanks_54745.png')
img = model.get_input(img)
f1 = model.get_feature(img)
# gender, age = model.get_ga(img)
# print(gender)
# print(age)
# sys.exit(0)
img = cv2.imread('t1.jpg')
img = model.get_input(img)
f2 = model.get_feature(img)

face_features = list()
person_id_list = list()
face_features.append(f1)
person_id_list.append("Tom")
face_features.append(f2)
person_id_list.append("陈春雷")
face_features = np.array(face_features)

sim = cos_sim_mat(face_features, f2.T)
print(sim, "ccl")

cap = cv2.VideoCapture(0)

timeTick = time.time()
setDelay = 0.7

while cap.isOpened():

    start = time.time()
    if start >= timeTick + setDelay:
        ret, frame = cap.read()
        if ret:
            result = person_recog(model, frame, person_id_list, face_features)
            if result is None:
                pass
            else:
                frame, person_id_list, person_face_feature = result

            end = time.time()
            print("time cost:", end - start)
            cv2.imshow("rect and point", frame)
            k = cv2.waitKey(1)

            print("cv2.waitKey")
            if k == ord('q'):
                break
        timeTick = time.time()
    else:
        cap.grab()

cap.release()
cv2.destroyAllWindows()
