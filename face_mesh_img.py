import cv2
import matplotlib.pyplot as plt
import dlib
import os
from landmark_processor import LandmarkProcessor

landmark_processor = LandmarkProcessor()

path = os.path.join('images' ,'messi2.jpg')

img = cv2.imread(path)

landmark_processor.process_img(img)

plt.imshow(img)
plt.show()

