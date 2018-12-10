# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import numpy as np
import tensorflow as tf
import csv 

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph


def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(
        file_reader, channels=3, name="png_reader")
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(
        tf.image.decode_gif(file_reader, name="gif_reader"))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
  else:
    image_reader = tf.image.decode_jpeg(
        file_reader, channels=3, name="jpeg_reader")
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result


def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label


if __name__ == "__main__":
  output_results = []
  #file_name = "tensorflow/examples/label_image/data/grace_hopper.jpg"
  model_file = '/home/william/m18_jorge/Desktop/THESIS/scripts/archae_aerial/results_training/output_graph.pb'
  label_file = '/home/william/m18_jorge/Desktop/THESIS/scripts/archae_aerial/results_training/output_labels.txt'
  input_height = 299
  input_width = 299
  input_mean = 0
  input_std = 255
  input_layer = 'Placeholder'
  output_layer = 'final_result'
  
  directory = '/home/william/m18_jorge/Desktop/THESIS/DATA/aerial_photos_plus/All_images/'
  lista = [f for f in os.listdir(directory)]
  graph = load_graph(model_file)
  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  input_operation = graph.get_operation_by_name(input_name)
  output_operation = graph.get_operation_by_name(output_name)
  base = '/home/william/m18_jorge/Desktop/THESIS/DATA/aerial_photos_plus/All_images/'
  for c, file_name in enumerate(lista[:]):
    print(file_name, c, '/', len(lista))
    file_nam = base + file_name
    t = read_tensor_from_image_file(
        file_nam,
        input_height=input_height,
        input_width=input_width,
        input_mean=input_mean,
        input_std=input_std)

    with tf.Session(graph=graph) as sess:
      results = sess.run(output_operation.outputs[0], {
          input_operation.outputs[0]: t
      })
    results = np.squeeze(results)

    #top_k = results.argsort()[-5:][::-1]
    top_k = results.argsort()[-1:][::-1]
    labels = load_labels(label_file)
    for i in top_k[:1]:
      output_results.append((labels[i], results[i]))
      print(labels[i], results[i])
    
  with open('Predictions.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for i, element in enumerate(output_results[:]):
      writer.writerow([element[0], element[1],lista[i]])
    print('Done!')
    
  #print(labels[0], results[0], labels[1], results[1])