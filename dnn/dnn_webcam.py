import cv2
import time

# load model
model_path = '/home/linaro/Desktop/hackathon/dnn/models/opencv_face_detector_uint8.pb'
config_path = '/home/linaro/Desktop/hackathon/dnn/models/opencv_face_detector.pbtxt'
net = cv2.dnn.readNetFromTensorflow(model_path, config_path)

conf_threshold = 0.9

prevTime = 0
cap = cv2.VideoCapture(10)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 160)
print("width :%d, height : %d" % (cap.get(3), cap.get(4)))

while True:
    ret, img = cap.read()  

    if not ret:
        break

    curTime = time.time()
    h, w, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [
                                 104, 117, 123], False, False)
    net.setInput(blob)

    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)

            # draw rects
            cv2.rectangle(img, (x1, y1), (x2, y2),
                          (255, 255, 255), int(round(h/150)), cv2.LINE_AA)
            cv2.putText(img, '%.2f%%' % (confidence * 100.), (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # inference time

    sec = curTime - prevTime
    prevTime = curTime

    fps = 1/(sec)

    # print(f"Time {sec} ")
    print(fps)
    cv2.imshow("frame_color", img)  
    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()
