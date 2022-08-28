import dlib
import cv2
import numpy as np


detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(
    "./shape_predictor_5_face_landmarks.dat"
)
facerec = dlib.face_recognition_model_v1(
    "./dlib_face_recognition_resnet_model_v1.dat"
)


def find_faces(img):  
    dets = detector(img, 1)

    if len(dets) == 0:
        return np.empty(0), np.empty(0), np.empty(0)

    rects, shapes = [], []
    shapes_np = np.zeros((len(dets), 5, 2), dtype=np.int)

    for k, d in enumerate(dets):
        rect = ((d.left(), d.top()), (d.right(), d.bottom()))
        rects.append(rect)

        shape = sp(img, d)

        for i in range(0, 5):
            shapes_np[k][i] = (shape.part(i).x, shape.part(i).y)

        shapes.append(shape)

    return rects, shapes, shapes_np


def encode_faces(
    img, shapes
):  
    face_descriptors = []
    for shape in shapes:
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        face_descriptors.append(np.array(face_descriptor))

    return np.array(face_descriptors)


img_paths = {
    "client_1": "./img/picture_2.png",
    "client_2": "./img/picture_3.png",
    "client_3": "./img/picture_5.png",
    
}

descs = {"client_1": None,"client_2": None,"client_3": None}

for name, img_path in img_paths.items():
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    _, img_shapes, _ = find_faces(img_rgb)
    descs[name] = encode_faces(img_rgb, img_shapes)[0]

np.save("./descs_client.npy", descs)
# print(descs)
