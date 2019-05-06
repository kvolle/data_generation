import numpy as np
import tensorflow as tf
import cv2
import sys
import os
import scipy.io as sio

key_frames = 5
overlap_coeff = 0.2 # Set such that at least 64% of the pixels overlap for matches

number_of_matches = 5
percentage_matching = 50
truth = sio.loadmat('./source/Lip6IndoorDataSet/Lip6IndoorGroundTruth.mat')
truth_mat = truth['truth']
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
    #assert img.size == image_size*image_size # Removed this cause of shape issues
    img = img.reshape([img.size])
    return tf.compat.as_bytes(img.tostring())



srcdir = './source/Lip6IndoorDataSet/Images'
filename = srcdir + "/lip6kennedy_bigdoubleloop_%6d.ppm"
print(filename)

cap = cv2.VideoCapture(filename)
color_set = []
gray_set = []
custom_set = []
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        h_channel = np.copy(frame_hsv[:, :, 0])
        s_channel = np.copy(frame_hsv[:, :, 1])
        v_channel = np.copy(frame_hsv[:, :, 2])
        s_channel = np.divide(s_channel, 255.)
        custom = np.multiply(h_channel, s_channel) + np.multiply(1 - s_channel, v_channel)
        custom = np.round(custom).astype(dtype=int)
        color_set.append(np.asarray(frame))
        gray_set.append(np.asarray(gray))
        custom_set.append(np.asarray(custom))
        # break
    else:
        break

cap.release()

custom_array = np.asarray(custom_set, dtype=int)
color_array = np.asarray(color_set, dtype=int)
gray_array = np.asarray(gray_set, dtype=int)
# Order of arguments:
# 0 - Frame
# 1 - Row
# 2 - Column
# 3 - Channel (optional)
set_info = custom_array.shape
unmatched = []
matched = []
for i in range(set_info[0]):
    for j in range(i):
        if truth_mat[i,j] != 0:
            matched.append((i,j))
        else:
            unmatched.append((i,j))
match_len = len(matched)
unmatch_len = len(unmatched)

# open the TFRecords files
color_filename = './datasets/color.tfrecords'
color_writer = tf.python_io.TFRecordWriter(color_filename)
custom_filename = './datasets/custo_.tfrecords'
custom_writer = tf.python_io.TFRecordWriter(custom_filename)
gray_filename = './datasets/gray.tfrecords'
gray_writer = tf.python_io.TFRecordWriter(gray_filename)

for i in range(key_frames):
    if np.random.randint(0, 99) < percentage_matching:
        # Generate an example of matched pair
        (t_a, t_b) = matched[np.random.randint(0, match_len)]
        match = True
    else:
        # Generate an example of unmatched pair
        (t_a, t_b) = unmatched[np.random.randint(0, unmatch_len)]
        match = False
    custom_img_a = serialize(custom_array[t_a, :, :])
    color_img_a = serialize(color_array[t_a, :, :, 0:3])
    gray_img_a = serialize(gray_array[t_a, :, :])

    custom_img_b = serialize(custom_array[t_b, :, :])
    color_img_b = serialize(color_array[t_b, :, :, 0:3])
    gray_img_b = serialize(gray_array[t_b, :, :])
    cu = sys.getsizeof(custom_array[0, 1, 2])
    co = sys.getsizeof(color_array[0, 1, 2, 0])
    gr = sys.getsizeof(gray_array[0, 1, 2])

    custom_features = {
        'img_a': _bytes_feature(custom_img_a),
        'img_b': _bytes_feature(custom_img_b),
        'match': _int64_feature(match)
    }
    color_features = {
        'img_a': _bytes_feature(color_img_a),
        'img_b': _bytes_feature(color_img_b),
        'match': _int64_feature(match)
    }
    gray_features = {
        'img_a': _bytes_feature(gray_img_a),
        'img_b': _bytes_feature(gray_img_b),
        'match': _int64_feature(match)
    }

    # Create an example protocol buffer
    color_example = tf.train.Example(features=tf.train.Features(feature=color_features))
    custom_example = tf.train.Example(features=tf.train.Features(feature=custom_features))
    gray_example = tf.train.Example(features=tf.train.Features(feature=gray_features))

    # Serialize to string and write on the file
    color_writer.write(color_example.SerializeToString())
    custom_writer.write(custom_example.SerializeToString())
    gray_writer.write(gray_example.SerializeToString())

color_writer.close()
custom_writer.close()
gray_writer.close()
print("Fin")
