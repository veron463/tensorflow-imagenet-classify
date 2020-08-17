import os


# ----------------- [PATHS] ---------------------------------------------------------------
_cur_dir = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.join(_cur_dir, os.pardir, os.pardir)

LOG_DIR = os.path.join(ROOT_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

MODEL_DIR = os.path.join(ROOT_DIR, 'models')


# [pascal labels]
labels_path = os.path.join(MODEL_DIR, 'label_names', 'pascal.names')
# PASCAL_LABELS = open(labels_path).read().strip().split("\n")

P_PERSON = [15]
P_BICK_MOTOR = [2, 14]
P_VEHICLES = [6, 7]
P_TRAIN = [19]


# [coco labels]
# labels_path = os.path.join(MODEL_DIR, 'label_names', 'coco.names')
# COCO_LABELS = open(labels_path).read().strip().split("\n")


C_PERSON = [0]
C_VEHICLES = [2, 5, 7]
C_BICK_MOTOR = [1, 3]
C_TRAIN = [6]
C_TRAFFIC_SIGNS = [9, 11]


# -----------------  [DETECTORS] ---------------------------------------------------------
DET_SSD = "SSD-MOBILE"
DET_OPENVINO = "OPENVINO"


# ----------------- [KEYS] ---------------------------------------------------------------
# keys of detect object
KEY_RECT = "rect"
KEY_COLOR = "color"
KEY_LABEL = "label"
KEY_CONFIDENCE = "confidence"


DET_FACE_CNN = "CNN"
DET_FACE_HOG = "HOG"

# ----------------- [LABELS] ---------------------------------------------------------------
"back-view"
"konstantine"
"predrag"
"sussie"
"timothy"
