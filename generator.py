import numpy as np
import tensorflow as tf
import cv2
from datasets import dataclass as data

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

filename = 'source/output.avi'
key_frames = 50
max_pixel_offset = 2
max_time_offset = 2
image_size = 28
number_of_matches = 5

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize(img):
    assert img.size == image_size*image_size
    img.reshape([img.size])
    return tf.compat.as_bytes(img.tostring())

cap = cv2.VideoCapture(filename)
color_set = []
gray_set = []
custom_set = []
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        h_channel = np.copy(frame_hsv[:, :, 0])
        s_channel = np.copy(frame_hsv[:, :, 1])
        v_channel = np.copy(frame_hsv[:, :, 2])

        s_channel = np.divide(s_channel,255.)
        custom = np.multiply(h_channel, s_channel) + np.multiply(1 -s_channel, v_channel)
        custom = custom / 255

        color_set.append(np.asarray(frame))
        gray_set.append(np.asarray(gray))
        custom_set.append(np.asarray(custom))
        #cv2.imshow('frame', gray)

        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break
    else:
        break

cap.release()
cv2.destroyAllWindows()

custom_array = np.asarray(custom_set)
color_array = np.asarray(color_set)
gray_array = np.asarray(gray_set)
# Order of arguments:
# 0 - Frame
# 1 - Row
# 2 - Column
# 3 - Channel (optional)
set_info = custom_array.shape
color_dataset=[]
gray_dataset=[]
custom_dataset=[]

for i in range(key_frames):
    t_mean = np.random.randint(max_time_offset, set_info[0]-max_time_offset)
    x_mean = np.random.randint(max_pixel_offset, set_info[1]-image_size-max_pixel_offset)
    y_mean = np.random.randint(max_pixel_offset, set_info[2]-image_size-max_pixel_offset)
    custom_a = []
    gray_a = []
    color_a = []
    for j in range(number_of_matches):
        x = np.random.randint(x_mean-max_pixel_offset, x_mean+max_pixel_offset)
        y = np.random.randint(y_mean-max_pixel_offset, y_mean+max_pixel_offset)
        t = np.random.randint(t_mean-max_time_offset, t_mean+max_time_offset)
        custom_a.append(custom_array[t, x:x+image_size, y:y+image_size])
        color_a.append(color_array[t, x:x+image_size, y:y+image_size, :])
        gray_a.append(gray_array[t, x:x+image_size, y:y+image_size])
    custom_dataset.append(custom_a)
    gray_dataset.append(gray_a)
    color_dataset.append(gray_a)

set_obj = data.siamese()
set_obj.populate_data(color=color_dataset, gray=gray_dataset, custom=custom_dataset)
set_obj.save_set()
tah = set_obj.get_unmatched_pair('custom')
print("Fin")