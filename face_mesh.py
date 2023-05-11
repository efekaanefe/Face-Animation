# import mediapipe
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("messi.jpg")

img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 

plt.imshow(img)
plt.show()
# if cv2.waitKey(1) & 0xFF == ord('q'):
#     cv2.destroyAllWindows()

# cv2.imshow("webcam", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# print("hello")

