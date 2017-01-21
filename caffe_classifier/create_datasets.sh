#!/usr/bin/env bash

unzip_dest="datasets/train/orig"
if [ ! -d "$unzip_dest/pklot_small" ]; then
  unzip datasets/train/orig/pklot_small.zip -d $unzip_dest
fi

python2.7 create_datasets.py
/opt/caffe/build/tools/compute_image_mean datasets/train/lmdb/train mean_image.binaryproto