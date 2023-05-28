import dlib 
import cv2
from landmark_processor import LandmarkProcessor
import time

cap = cv2.VideoCapture('videos\video00.mp4')
print(cap.isOpened())

while cap.isOpened():
    success, frame = cap.read()
    
    print(success)
    cv2.imshow('Frame',frame)
    
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
    
cap.release()
cv2.destroyAllWindows()
