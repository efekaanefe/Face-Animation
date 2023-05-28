import dlib 
import cv2
from landmark_processor import LandmarkProcessor
import time

cap = cv2.VideoCapture('videos\\video00.mp4')

while cap.isOpened():
    success, frame = cap.read()
    cv2.imshow('Face animation',frame)
    
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
    
cap.release()
cv2.destroyAllWindows()
