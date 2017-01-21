import json
import os

root_dir = os.path.dirname(os.path.abspath(__file__))

dataset = {
    "occupied": {
        "id": 0,
        "path": "/home/kontiki/Downloads/datasets/cars/pklot_small/Occupied"
    },
    "empty": {
        "id": 1,
        "path": "/home/kontiki/Downloads/datasets/cars/pklot_small/Empty"
    }

}

labels_mapping = dict((v["id"], k)  for k,v in dataset.items())

model = {
    "model_def_file": root_dir + "/" + "models/ordinary/caffenet_deploy_1.prototxt",
    "model_file": root_dir + "/" + "models/ordinary/caffe_model_1_iter_8000.caffemodel",
    "mean_file": root_dir + "/" + "mean_image.binaryproto"
}

val_dataset = {
    "occupied": root_dir + "/" + "test/test_data/occupied",
    "empty": root_dir + "/" + "test_data/empty"
}

format = lambda dict_data: json.dumps(dict_data, indent=4, sort_keys=True)
print "model: %s, dataset: %s" % (format(model), format(dataset))
