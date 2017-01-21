import cv2
import glob

from lot_classifier import LotClassifier
from settings import model, dataset, labels_mapping, val_dataset


def test_classifier():
    classifier = LotClassifier(model["model_def_file"], model["model_file"], labels_mapping, model["mean_file"])

    wrong, total = 0, 0
    for group in val_dataset.items():
        for img_file in glob.glob(group[1] + "/*.jpg"):
            if classifier.predict(cv2.imread(img_file)) != group[0]:
                wrong += 1
            print "predicted:%s, real: %s, location:%s" % (
                classifier.predict(cv2.imread(img_file)), group[0], img_file)
            total += 1
    print "wrong: %s, total: %s" % (wrong, total)


if __name__ == '__main__':
    test_classifier()
