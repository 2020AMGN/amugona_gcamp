import cv2

cap = cv2.VideoCapture(10)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

cv2.namedWindow("test")

img_counter = 1

while True:
    ret, frame = cap.read()
    if not ret:
        print("error")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC
        print("close")
        break
    elif k % 256 == 32:
        # SPACE
        img_name = "picture_{0}.png".format(img_counter)
        cv2.imwrite(
            "./img/{0}".format(img_name),
            frame,
        )
        print("{0} save!".format(img_name))
        img_counter += 1

cap.release()

cv2.destroyAllWindows()
