import os
import cv2
import numpy as np
import tensorflow as tf

from download import download_and_extract_model
from logger import logger
from constants import MODEL_DIR


IMGNET_DIR = os.path.join(MODEL_DIR, "imgnet")
os.makedirs(IMGNET_DIR, exist_ok=True)


class ImgNetFeatureExtractor:
    def __init__(self):
        self.model_dir = IMGNET_DIR

        self.__create_graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = False
        self.sess = tf.Session(config=config)
        self.softmax_tensor = self.sess.graph.get_tensor_by_name('pool_3:0')

        logger.info("load mxnet model.")
        test_img = np.ones((20, 20, 3), dtype=np.uint8)
        _tmp_data = cv2.imencode('.jpg', test_img)[1].tostring()
        prediction = self.sess.run(self.softmax_tensor, {'DecodeJpeg/contents:0': _tmp_data})
        prediction = np.squeeze(prediction)
        logger.info("\tlength of feature {}    {}.".format(len(prediction), "success" * (len(prediction) == 2048)))

    def __create_graph(self):
        # Creates a graph from saved GraphDef file and returns a saver.
        # Creates graph from saved graph_def.pb.
        if not os.path.exists(os.path.join(self.model_dir, 'classify_image_graph_def.pb')):
            data_url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
            download_and_extract_model(data_url=data_url, save_dir=self.model_dir)
        with tf.gfile.FastGFile(os.path.join(
                self.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

    def get_feature_from_image(self, img_path):
        """Runs extract the feature from the image.
            Args: img_path: Image file name.
        Returns:  predictions: 2048 * 1 feature vector
        """

        if not tf.gfile.Exists(img_path):
            tf.logging.fatal('File does not exist %s', img_path)
        image_data = tf.gfile.FastGFile(img_path, 'rb').read()

        prediction = self.sess.run(self.softmax_tensor, {'DecodeJpeg/contents:0': image_data})
        prediction = np.squeeze(prediction)
        return prediction

    def get_feature_from_cv_mat(self, cvimg):
        image_data = cv2.imencode('.jpg', cvimg)[1].tostring()
        prediction = self.sess.run(self.softmax_tensor, {'DecodeJpeg/contents:0': image_data})
        prediction = np.squeeze(prediction)
        return prediction
