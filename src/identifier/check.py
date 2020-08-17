from src.identifier.imgnet_feature import ImgNetFeatureExtractor
import src.utils.logger as logger
import numpy as np
import os
import joblib
MODEL_DIR = '../../models/'
CLASSIFIER = os.path.join(MODEL_DIR, 'classifier.pkl')

def load_classifier_model():
    if not os.path.exists(CLASSIFIER):
        logger.warning(" not exist trained classifier {}\n".format(CLASSIFIER))
        return None

    try:
        # loading
        model = joblib.load(CLASSIFIER)
        return model
    except Exception as ex:
        print(ex)
        return None

def check_precision(train_path):
    # --- load trained classifier imgnet --------------------------------------------------------------
    classifier = load_classifier_model()

    logger.info('>>> checking the precision... ')

    # --- load feature and label data ----------------------------------------------------------------
    data, labels, label_names = load_feature_and_label(
        feature_data_path=os.path.join(train_path, "train_data.csv"),
        feature_label_path=os.path.join(train_path, "train_label.txt"))

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

    image_path = '../../data/all/Swedish Cavalry 1893 - Troopers/0de527b9-1d0d-453f-a194-2a3d170c32d4.jpg'
    feature_extractor = ImgNetFeatureExtractor()
    feature = feature_extractor.get_feature_from_image(image_path)
    model = load_classifier_model()
    probab = model.predict_proba(feature)

    max_ind = np.argmax(probab)