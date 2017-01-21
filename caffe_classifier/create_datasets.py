import glob
import random
import numpy as np
import cv2
import shutil
from caffe.proto import caffe_pb2
import lmdb

config = {}
with open("config.txt") as f:
    for line in f:
       (key, val) = line.split("=")
       config[key] = val.replace("\n", "")

IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):
    # Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_CUBIC)
    return img


def make_datum(img, label):
    # image is numpy.ndarray format. BGR instead of RGB
    return caffe_pb2.Datum(
        channels=3,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        label=label,
        data=np.rollaxis(img, 2).tostring())


train_lmdb = 'dataset/train_lmdb'
test_lmdb = 'dataset/test_lmdb'

shutil.rmtree(train_lmdb, ignore_errors=True)
shutil.rmtree(test_lmdb, ignore_errors=True)

occupied_data = [img for img in glob.glob(config["occupied_dir"] +"/*jpg")]
empty_data = [img for img in glob.glob(config["empty_dir"] +"/*jpg")]

min_size = min(len(occupied_data), len(empty_data))
occupied_data = occupied_data[:min_size]
empty_data = empty_data[:min_size]
all_data = [("occupied", img) for img in occupied_data] + [("empty", img) for img in empty_data]

random.shuffle(all_data)
ration = 0.8
split_index = int(len(all_data)*0.8)
train_data = all_data[:split_index]
test_data = all_data[split_index:]

#move this to config!
mapping = {"occupied":0, "empty":1}


def create_db(lmdb_path, data, mapping):
    in_db = lmdb.open(lmdb_path, map_size=int(1e12))
    with in_db.begin(write=True) as in_txn:
        for in_idx, (label, img_path) in enumerate(data):
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
            label_idx = mapping[label]
            datum = make_datum(img, label_idx)
            in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())
            print '{:0>5d}'.format(in_idx) + ':' + img_path + ":" + label  + " : " +str(label_idx)
    in_db.close()


create_db(train_lmdb, train_data, mapping)
create_db(test_lmdb, test_data, mapping)