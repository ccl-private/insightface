import face_model
import face_model2
import argparse
import cv2
import sys
import numpy as np
import datetime


def age_model():
  parser = argparse.ArgumentParser(description='face model test')
  # general
  parser.add_argument('--image-size', default='112,112', help='')
  parser.add_argument('--image', default='Tom_Hanks_54745.png', help='')
  parser.add_argument('--model', default='model/model,0', help='path to load model.')
  parser.add_argument('--gpu', default=0, type=int, help='gpu id')
  parser.add_argument('--det', default=0, type=int,
                      help='mtcnn option, 1 means using R+O, 0 means detect from begining')
  args = parser.parse_args()

  return face_model.FaceModel(args)


def recog_model():
  parser = argparse.ArgumentParser(description='face model test')
  # general
  parser.add_argument('--image-size', default='112,112', help='')
  parser.add_argument('--model', default='../models/model-r34-amf/model,0', help='path to load model.')
  parser.add_argument('--ga-model', default='', help='path to load model.')
  parser.add_argument('--gpu', default=0, type=int, help='gpu id')
  parser.add_argument('--det', default=0, type=int,
                      help='mtcnn option, 1 means using R+O, 0 means detect from begining')
  parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
  parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
  args = parser.parse_args()
  return face_model2.FaceModel(args)


model = age_model()
model2 = recog_model()
img = cv2.imread('t1.jpg')
# img = cv2.imread(args.image)
img2 = model2.get_input(img)
img = model.get_input(img)
f1 = model2.get_feature(img2)
print(f1[0:10])
for _ in range(5):
  gender, age = model.get_ga(img)
time_now = datetime.datetime.now()
count = 200
for _ in range(count):
  gender, age = model.get_ga(img)
time_now2 = datetime.datetime.now()
diff = time_now2 - time_now
print('time cost', diff.total_seconds()/count)
print('gender is', gender)
print('age is', age)

