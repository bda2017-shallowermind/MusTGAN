# Hello, Jason~

"""Script to transform audio files to tfrecord format"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import librosa
import tensorflow as tf
import os, sys


# should change this directory path
dataset_dir = "/home/anyj0527/dataset/wav/16khz/"
wavfiles_list = sorted([
    os.path.join(dataset_dir, fname) for fname in tf.gfile.ListDirectory(dataset_dir)
    if (lambda f: f.lower().endswith(".wav"))
])
# should change this directory path
sliced_dataset_dir = "./dataset/musicnet_4s_sliced/"
sliced_file_list = sorted([
    os.path.join(sliced_dataset_dir, fname) for fname 
        in tf.gfile.ListDirectory(sliced_dataset_dir)
    if (lambda f: f.lower().endswith(".wav"))
])


def slice_audio(file_list, slice_length):
  """Slice audio files into sliece_length chunks (maybe 64000 samples)"""
  for fname in file_list:
    y, sr = librosa.load(fname, sr=None)
    total_duration_sec = len(y)//slice_length
    for v in range(0, len(y), slice_length):
      sliced = y[v:v+slice_length]
      sliced_file_name = fname.split("/")[-1] + \
          str(v//slice_length) + "s-" + str(total_duration_sec) + "s.wav"
      if len(sliced) == slice_length:
        librosa.output.write_wav(sliced_dataset_dir + sliced_file_name, sliced, sr)


def tfrecord_write(sliced_file_list):
  """Make tfrecord consists of sliced audio files"""
  # should change this output tfrecord path
  writer = tf.python_io.TFRecordWriter("musicnet.tfrecord")
  for fname in sliced_file_list:
    y, sr = librosa.load(fname, sr=None)
    features = tf.train.Features(
        feature = { 'audio': tf.train.Feature(float_list=tf.train.FloatList(value=y)), } )
    example = tf.train.Example(features=features)
    serialized = example.SerializeToString()
    writer.write(serialized)
  writer.close()


if __name__ == "__main__":
  """Usage example
  slice_audio(wavfiles_list, 64000)
  tfrecord_write(sliced_file_list)
  """
  print("Hello, Jason!")


