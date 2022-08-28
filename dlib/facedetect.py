import cv2
import dlib
import sys
import numpy as np
import time

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "./shape_predictor_68_face_landmarks.dat"
)


cap = cv2.VideoCapture(10)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
prevTime = 0
count = 0
while True:
    ret, img = cap.read()
    if not ret:
        break 

    curTime = time.time()
    faces = detector(img)
    if faces:
        count+=1
        print("count : ", count)
        face = faces[0]  
        dlib_shape = predictor(img, face)
        shape_2d = np.array(
            [[p.x, p.y] for p in dlib_shape.parts()]
        )
        if count<35:
            img = cv2.putText(img,"wait..",(150,200),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255))
        elif count>=35:
            img = cv2.putText(img,"open!!",(150,200),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255))
        img = cv2.rectangle( 
            img,
            pt1=(face.left(), face.top()),
            pt2=(face.right(), face.bottom()),
            color=(255, 255, 255),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        # for s in shape_2d:  
        #     cv2.circle(
        #         img,
        #         center=tuple(s),
        #         radius=1,
        #         color=(255, 255, 255),
        #         thickness=2,
        #         lineType=cv2.LINE_AA,
        #     )
        if count == 40:
            print("open")
            break
    else:
        count = 0
    sec = curTime - prevTime
    prevTime = curTime

    fps = 1/(sec)

    # print(f"Time {sec} ")
    # print(fps)
    img = cv2.resize(img,(640,480),cv2.INTER_AREA)
    cv2.imshow("img", img)
    if cv2.waitKey(1) == ord("q"):
        break
