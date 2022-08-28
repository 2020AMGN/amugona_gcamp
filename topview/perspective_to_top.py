import cv2
import numpy as np
img_path = 'C:/Users/ak803/Desktop/hackathon/data/picture/map_hackathon_5.png'
save = False #True면 저장

img = cv2.imread(img_path)

pts = np.array([[0, 510], [1036, 264], [1198, 719],
                [1280, 317]], dtype=np.float32)
ipm_pts = np.array([[100, 100], [1080, 100], [100, 620],
                    [1080, 620]], dtype=np.float32)

ipm_matrix = cv2.getPerspectiveTransform(pts, ipm_pts)
print(ipm_matrix)
a = np.array([567, 313, 1])
ipm_matrix = np.array(ipm_matrix)
at = np.matmul(ipm_matrix, a)
new_x = at[0]/at[2]
new_y = at[1]/at[2]

ipm = cv2.warpPerspective(img, ipm_matrix, img.shape[:2][::-1])

cv2.imshow('img', img)
cv2.imshow('ipm', ipm)
if save:
    cv2.imwrite('topview_hackathon.png', ipm)
cv2.waitKey()
