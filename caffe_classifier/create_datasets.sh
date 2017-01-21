#!/usr/bin/env bash
python2.7 create_datasets.py
/opt/caffe/build/tools/compute_image_mean dataset/train_lmdb mean_image.binaryproto