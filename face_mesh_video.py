import dlib 
import cv2
from landmark_processor import LandmarkProcessor
import time

landmark_processor = LandmarkProcessor()

cap = cv2.VideoCapture('videos\\video00.mp4')
last_time = time.time()
while cap.isOpened():
    success, frame = cap.read()
    cv2.imshow('Face animation',frame)
    
    landmark_processor.process_img(frame)
    
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
    # print("sec:",time.time()-last_time)  
    last_time = time.time()

cap.release()
cv2.destroyAllWindows()
