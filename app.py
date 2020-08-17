from flask import Flask, request, render_template, redirect, send_from_directory, jsonify
import tensorflow as tf
import logging
import sys
import os
import joblib

# import csv
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
file_handler = logging.FileHandler('server.log')
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)

MODEL_DIR = './models'
IMGNET_DIR = os.path.join(MODEL_DIR, "imgnet")
LOG_DIR = './logs'
TRAIN_PATH = './data/prepare'



PROJECT_HOME = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = '{}/uploads/'.format(PROJECT_HOME)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

with tf.gfile.FastGFile(os.path.join(IMGNET_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

config = tf.ConfigProto()
config.gpu_options.allow_growth = False
tf_session = tf.Session(config=config)
tf_softmax_tensor = tf_session.graph.get_tensor_by_name('pool_3:0')
model1 = None
model2 = None
CLASSIFIER1 = os.path.join(MODEL_DIR, 'classifier.pkl')
CLASSIFIER2 = os.path.join(MODEL_DIR, 'classifier2.pkl')
if not os.path.exists(CLASSIFIER1):
    app.logger.warning(" not exist trained classifier {}\n".format(CLASSIFIER1))
    sys.exit(0)
try:
    # loading
    model1 = joblib.load(CLASSIFIER1)

except Exception as ex:
    print(ex)
    sys.exit(0)

try:
    # loading
    model2 = joblib.load(CLASSIFIER2)

except Exception as ex:
    print(ex)
    sys.exit(0)


def create_new_folder(local_dir):
    newpath = local_dir
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    return newpath

@app.route('/', methods=["GET", "POST"])
def hello_world():
    if request.method == 'POST':
        key1 = request.form['key1'] if 'key1' in request.form else None
        key2 = request.form['key2'] if 'key2' in request.form else None
        return jsonify({"key1":key1, "key2":key2})
    return render_template("upload_image.html")

@app.route("/upload-image", methods=["GET", "POST"])
def api_root():
    app.logger.info(PROJECT_HOME)
    if request.method == 'POST' and request.files['image']:
        app.logger.info(app.config['UPLOAD_FOLDER'])
        img = request.files['image'] if 'image' in request.files else None
        if img is None:
            return jsonify({'error': 'image not found'})
        req_label = request.form['label'] if 'label' in request.form else None
        req_model_number = request.form['model'] if 'model' in request.form else 1
        req_model_number = int(req_model_number)
        model = model1 if req_model_number == 1 else model2
        closest_count = request.form['closest_count'] if 'closest_count' in request.form else 5
        closest_count = int(closest_count)
        img_name = img.filename
        create_new_folder(app.config['UPLOAD_FOLDER'])
        saved_path = os.path.join(app.config['UPLOAD_FOLDER'], img_name)
        app.logger.info("saving {}".format(saved_path))
        img.save(saved_path)


        if not tf.gfile.Exists(saved_path):
            tf.logging.fatal('File does not exist %s', saved_path)
        image_data = tf.gfile.FastGFile(saved_path, 'rb').read()

        feature_prediction = tf_session.run(tf_softmax_tensor, {'DecodeJpeg/contents:0': image_data})
        feature_prediction = np.squeeze(feature_prediction)
        feature_prediction = feature_prediction.reshape(1, -1)

        probab = model.predict_proba(feature_prediction)
        max_ind = np.argmax(probab)
        max_label = model.classes_[max_ind]
        max_score = probab[0, max_ind]
        if not req_label:
            # get closest 5 labels
            copied_np = np.copy(probab)
            # closest_count = 5
            ids = np.argsort(-copied_np)[0][:closest_count]
            send_data = []
            for score_id in ids:
                score = probab[0][score_id]
                label = model.classes_[score_id]
                one_image_data = {"score": score, "label": label}
                send_data.append(one_image_data)

            return jsonify(send_data)
        # get score of req_label
        req_score_index_array = np.where(model.classes_ == req_label)[0]
        req_score_index = -1 if len(req_score_index_array) == 0 else req_score_index_array[0]
        req_score = probab[0, req_score_index] if req_score_index > -1 else 0
        send_data = {"max_label":max_label, "max_score":max_score, "label":req_label, "score":req_score}
        return jsonify(send_data)
    else:
        return render_template("upload_image.html")




if __name__ == "__main__":
    app.run(debug=True)