import dlib
import cv2
import numpy as np
import time

Sensitivity_HIGH = "./shape_predictor_68_face_landmarks.dat"
Sensitivity_LOW = "./shape_predictor_5_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(
    Sensitivity_LOW
)
facerec = dlib.face_recognition_model_v1(
    "./dlib_face_recognition_resnet_model_v1.dat"
)

np_load_old = np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
descs = np.load("./descs_client.npy")[()]
# print(descs)


def encode_face(img):
    dets = detector(img, 1)

    if len(dets) == 0:
        return np.empty(0)

    for k, d in enumerate(dets):
        shape = sp(img, d)
        face_descriptor = facerec.compute_face_descriptor(img, shape)

        return np.array(face_descriptor)


cap = cv2.VideoCapture(10)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
prevTime = 0
if not cap.isOpened():
    exit()

_, img_bgr = cap.read()

count = 1

while True:
    ret, img_bgr = cap.read()
    if not ret:
        break
    curTime = time.time()

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    dets = detector(img_bgr, 1)

    for k, d in enumerate(dets):
        shape = sp(img_rgb, d)
        face_descriptor = facerec.compute_face_descriptor(img_rgb, shape)

        last_found = {"name": "unknown", "dist": 0.4, "color": (0, 0, 255)}

        for name, saved_desc in descs.items():
            dist = np.linalg.norm([face_descriptor] - saved_desc, axis=1)

            if dist < last_found["dist"]:
                last_found = {"name": name, "dist": dist,
                              "color": (255, 0, 0)}
        cv2.putText(
            img_bgr,
            last_found["name"],
            org=(d.left(), d.top()+50),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=last_found["color"],
            thickness=2,
        )

    sec = curTime - prevTime
    prevTime = curTime

    fps = 1/(sec)
    print(f"fps : {fps} ")
    img_bgr = cv2.resize(img_bgr,(640,480),cv2.INTER_AREA)
    cv2.imshow("img", img_bgr)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
