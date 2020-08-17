import csv
import os
import sys

import joblib
import numpy as np
from sklearn.svm import SVC

import logger
from imgnet_feature import ImgNetFeatureExtractor
# from src.object_detect.people_detect import PeopleDetect
# from src.face_detect.face_detect import FaceDetect
from constants import MODEL_DIR

VIDEO_EXTs = [".mkv"]
SHOW_VIDEO = True

CLASSIFIER = os.path.join(MODEL_DIR, 'classifier2.pkl')



def collect_features(train_data):
    logger.info('collect train data(features) from the raw images')

    feature_extractor = ImgNetFeatureExtractor()

    # --- check the raw images ----------------------------------------------------------
    labels = [dir_name for dir_name in os.listdir(train_data) if os.path.isdir(os.path.join(train_data, dir_name))]
    labels.sort()

    tails = []
    for i in range(len(labels)):
        line = np.zeros((len(labels)), dtype=np.uint8)
        line[i] = 1
        tails.append(line.tolist())
    """
    tails = [[1., 0., 0., 0.],
             [0., 1., 0., 0.],
             ...
             ]
    """

    # --- scanning the raw image dir ----------------------------------------------------
    features = []
    for dir_name in labels:
        sub_dir_path = os.path.join(train_data, dir_name)

        count = 0
        fns = [fn for fn in os.listdir(sub_dir_path) if os.path.splitext(fn)[1].upper() == ".JPG"]
        fns.sort()

        for fn in fns:
            path = os.path.join(sub_dir_path, fn)

            try:
                # Extract the feature vector per each image
                feature = feature_extractor.get_feature_from_image(path)
                sys.stdout.write("\r" + path)
                sys.stdout.flush()
            except Exception as e:
                print(e)
                continue
            line = feature.tolist()
            line.extend(tails[labels.index(dir_name)])
            features.append(line)
            count += 1

            # if count > 10:  # for only testing
            #     break

        logger.info(f" label: {dir_name}, counts #: {count}")

    # --- write the train_data.csv file on the same location --------------------------------------
    save_dir = train_data
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    feature_data_path = os.path.join(save_dir, "train_data.csv")
    with open(feature_data_path, 'w', newline='') as fp:  # for python 3x
        wr = csv.writer(fp, delimiter=',')
        wr.writerows(features)

    # write the train_label.txt on the same location
    feature_label_path = os.path.join(save_dir, "train_label.txt")
    with open(feature_label_path, 'w') as fp:
        for label in labels:
            fp.write(label + "\n")

    sys.stdout.write("create the train_data.csv successfully!\n")
    return save_dir


def load_feature_and_label(feature_data_path, feature_label_path):
    logger.info('load feature and labels')

    if not os.path.exists(feature_data_path):
        logger.warning(f" not exist train data {feature_data_path}")
        sys.exit(0)
    if not os.path.exists(feature_label_path):
        logger.warning(f" not exist train label {feature_label_path}")
        sys.exit(0)

    data = []
    labels = []
    label_names = []
    # --- loading labels ---------------------------------------------------------------------
    logger.info(' loading training labels ...')
    with open(feature_label_path, 'r') as fp:
        for line in fp:
            line = line.replace('\n', '')
            label_names.append(line)

    # --- loading data -----------------------------------------------------------------------
    logger.info(' loading training data ...')
    with open(feature_data_path) as fp:  # for python 2x
        csv_reader = csv.reader(fp, delimiter=',')
        for row in csv_reader:
            _feature = [float(row[i]) for i in range(0, len(row))]
            _label = _feature[-len(label_names):]
            data.append(np.asarray(_feature[:-len(label_names)]))

            label_idx = -1
            for i in range(len(label_names)):
                if _label[i] == 1.0:
                    label_idx = i
                    break
            if label_idx != -1:
                labels.append(label_names[label_idx])
            else:
                logger.error(' error on tails for label indicator')
                sys.exit(0)
    return data, labels, label_names


def train(train_data):
    logger.info('>>> train ')

    # --- load feature and label data ----------------------------------------------------------------
    data, labels, label_names = load_feature_and_label(
        feature_data_path=os.path.join(train_data, "train_data.csv"),
        feature_label_path=os.path.join(train_data, "train_label.txt")
    )

    # --- training -----------------------------------------------------------------------------------
    logger.info(' training... ')
    classifier = SVC(C=1.0, kernel='linear', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=True,
                     tol=0.001, cache_size=200, class_weight='balanced', verbose=False, max_iter=-1,
                     decision_function_shape='ovr', random_state=None)

    classifier.fit(data, labels)
    joblib.dump(classifier, CLASSIFIER)

    logger.info('>>> finished the training!')


def load_classifier_model():
    if not os.path.exists(CLASSIFIER):
        logger.warning(" not exist trained classifier {}\n".format(CLASSIFIER))
        sys.exit(0)

    try:
        # loading
        model = joblib.load(CLASSIFIER)
        return model
    except Exception as ex:
        print(ex)
        sys.exit(0)

def check_one_image_prec(feature):
    train_data = "../../data/all"
    feature = feature.reshape(1, -1)
    classifier = load_classifier_model()

    logger.info('>>> checking the precision... ')

    # --- load feature and label data ----------------------------------------------------------------
    data, labels, label_names = load_feature_and_label(
        feature_data_path=os.path.join(train_data, "train_data.csv"),
        feature_label_path=os.path.join(train_data, "train_label.txt"))
    probab = classifier.predict_proba(feature)

    max_ind = np.argmax(probab)

    predlbl = classifier.classes_[max_ind]
    # print(labels[predlbl])

def check_precision(train_data):
    # --- load trained classifier imgnet --------------------------------------------------------------
    classifier = load_classifier_model()

    logger.info('>>> checking the precision... ')

    # --- load feature and label data ----------------------------------------------------------------
    data, labels, label_names = load_feature_and_label(
        feature_data_path=os.path.join(train_data, "train_data.csv"),
        feature_label_path=os.path.join(train_data, "train_label.txt"))

    true_pos = 0
    false_neg = 0
    # --- check confuse matrix ------------------------------------------------------------------------
    for i in range(len(data)):
        feature = data[i]
        feature = feature.reshape(1, -1)

        # Get a prediction from the imgnet including probability:
        probab = classifier.predict_proba(feature)

        max_ind = np.argmax(probab)

        predlbl = classifier.classes_[max_ind]
        if predlbl == labels[i]:
            true_pos += 1
        else:
            false_neg += 1

    logger.info('>>> precision result')
    logger.info(f"\tpositive : (true) {true_pos}")
    logger.info(f"\tnegative : (false){false_neg}")
    total = len(data)
    precision = true_pos / total
    logger.info(f"\tprecision : {true_pos} / {total} : {round(precision * 100, 2)}%")


if __name__ == '__main__':

    _train_data = "../../data/all/"
    collect_features(train_data=_train_data)
    train(train_data=_train_data)
    check_precision(train_data=_train_data)

    # image_path = '../../data/all/Swedish Cavalry 1893 - Troopers/0de527b9-1d0d-453f-a194-2a3d170c32d4.jpg'
    # feature_extractor = ImgNetFeatureExtractor()
    # feature = feature_extractor.get_feature_from_image(image_path)
    # check_one_image_prec(feature)
    pass
