import cv2
from constants import KEY_RECT


def visualize(img, faces, peoples):
    show = img.copy()

    for face in faces:
        _x, _y, _w, _h = face[KEY_RECT]
        cv2.rectangle(show, (_x, _y), (_x + _w, _y + _h), (0, 0, 255), 2)

    for people in peoples:
        _x, _y, _w, _h = people[KEY_RECT]
        cv2.rectangle(show, (_x, _y), (_x + _w, _y + _h), (255, 0, 0), 2)

    return show
