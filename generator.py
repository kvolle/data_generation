import numpy as np
import tensorflow as tf
import cv2
import sys

filename = 'source/nao_output.avi'
key_frames = 10000
max_pixel_offset = 2
max_time_offset = 2
image_size = 28
number_of_matches = 5
percentage_matching = 50

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
        #print("H: "+str(h_channel.min()) + " - " + str(h_channel.max()) + ": " + str(h_channel.mean()))
        #print("S: "+str(s_channel.min()) + " - " + str(s_channel.max()) + ": " + str(s_channel.mean()))
        #print("V: "+str(v_channel.min()) + " - " + str(v_channel.max()) + ": " + str(v_channel.mean()))
        s_channel = np.divide(s_channel,255.)
        custom = np.multiply(h_channel, s_channel) + np.multiply(1 -s_channel, v_channel)
        custom = np.round(custom).astype(dtype=int)
        #print("C: "+str(custom.min()) + " - " + str(custom.max()) + ": " + str(custom.mean()))

#        custom = custom / 255

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

custom_array = np.asarray(custom_set, dtype=int)
color_array = np.asarray(color_set, dtype=int)
gray_array = np.asarray(gray_set, dtype=int)
# Order of arguments:
# 0 - Frame
# 1 - Row
# 2 - Column
# 3 - Channel (optional)
set_info = custom_array.shape

# open the TFRecords files
color_filename = 'datasets/color.tfrecords'
color_writer = tf.python_io.TFRecordWriter(color_filename)
custom_filename = 'datasets/custom.tfrecords'
custom_writer = tf.python_io.TFRecordWriter(custom_filename)
gray_filename = 'datasets/gray.tfrecords'
gray_writer = tf.python_io.TFRecordWriter(gray_filename)

for i in range(key_frames):
    if np.random.randint(0,99) < percentage_matching:
        # Generate an example of matched pair
        t_mean = np.random.randint(max_time_offset, set_info[0] - max_time_offset)
        x_mean = np.random.randint(max_pixel_offset, set_info[1] - image_size - max_pixel_offset)
        y_mean = np.random.randint(max_pixel_offset, set_info[2] - image_size - max_pixel_offset)
        [x_a, x_b] = np.random.randint(x_mean - max_pixel_offset, x_mean + max_pixel_offset, 2)
        [y_a, y_b] = np.random.randint(y_mean - max_pixel_offset, y_mean + max_pixel_offset, 2)
        [t_a, t_b] = np.random.randint(t_mean - max_time_offset, t_mean + max_time_offset, 2)
        match = True
    else:
        # Generate an example of unmatched pair
        [t_a, t_b] = np.random.randint(max_time_offset, set_info[0] - max_time_offset, 2)
        [x_a, x_b] = np.random.randint(max_pixel_offset, set_info[1] - image_size - max_pixel_offset, 2)
        [y_a, y_b] = np.random.randint(max_pixel_offset, set_info[2] - image_size - max_pixel_offset, 2)
        match = False
    custom_img_a = serialize(custom_array[t_a, x_a:x_a + image_size, y_a:y_a + image_size])
    color_img_a = serialize(color_array[t_a, x_a:x_a + image_size, y_a:y_a + image_size, 0])
    gray_img_a = serialize(gray_array[t_a, x_a:x_a + image_size, y_a:y_a + image_size])

    custom_img_b = serialize(custom_array[t_b, x_b: x_b + image_size, y_b:y_b + image_size])
    color_img_b = serialize(color_array[t_b, x_b: x_b + image_size, y_b:y_b + image_size, 0])
    gray_img_b = serialize(gray_array[t_b, x_b: x_b + image_size, y_b:y_b + image_size])
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