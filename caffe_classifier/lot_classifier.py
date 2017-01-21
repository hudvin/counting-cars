import caffe
import numpy as np

from caffe.proto import caffe_pb2


class LotClassifier:

    LOT_OCCUPIED = "OCCUPIED"
    LOT_EMPTY = "EMPTY"

    def __init__(self, model_def, model, labels_mapping, mean):
        caffe.set_mode_gpu()
        self.labels_mapping = labels_mapping
        # read mean image
        mean_blob = caffe_pb2.BlobProto()
        with open(mean) as mean_file:
            mean_blob.ParseFromString(mean_file.read())
        mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
            (mean_blob.channels, mean_blob.height, mean_blob.width))
        #create classifier
        self.net = caffe.Classifier(model_def, model, mean=mean_array,
                                    channel_swap=(2, 1, 0),
                                    raw_scale=255)

    def predict(self, lot_cv2_image):
        lot_cv2_image = self.prepare_cv2_image(lot_cv2_image)
        prediction = self.net.predict([lot_cv2_image])
        predicted_class_id = prediction[0].argmax()
        predicted_label = self.labels_mapping[predicted_class_id]
        return predicted_label

    def prepare_cv2_image(self, cv2_image):
        cv2_image = cv2_image / 255.
        cv2_image = cv2_image[:, :, (2, 1, 0)]
        return cv2_image
