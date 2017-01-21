import json
import os

root_dir = os.path.dirname(os.path.abspath(__file__))

dataset = {
    "occupied": {
        "id": 0,
        "path": "datasets/train/orig/pklot_small/occupied"
    },
    "empty": {
        "id": 1,
        "path": "datasets/train/orig/pklot_small/empty"
    }

}

labels_mapping = dict((v["id"], k)  for k,v in dataset.items())

model = {
    "model_def_file": root_dir + "/" + "models/ordinary/caffenet_deploy_1.prototxt",
    "model_file": root_dir + "/" + "models/ordinary/caffe_model_1_iter_8000.caffemodel",
    "mean_file": root_dir + "/" + "mean_image.binaryproto"
}

val_dataset = {
    "occupied": root_dir + "/" + "datasets/val/occupied",
    "empty": root_dir + "/" + "datasets/val/empty"
}

format = lambda dict_data: json.dumps(dict_data, indent=4, sort_keys=True)
print "model: %s, dataset: %s" % (format(model), format(dataset))
