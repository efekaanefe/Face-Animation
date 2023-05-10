# import mediapipe
import cv2

img = cv2.imread("messi.jpg")

img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 

cv2.imshow("webcam", img)
cv2.waitKey()
cv2.destroyAllWindows()


