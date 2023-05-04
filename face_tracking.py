import cv2

video_capture = cv2.VideoCapture(0)

# Loading the required haar-cascade xml classifier file, classifies face from non face
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

print(face_cascade)
while True:
    # Capture frame-by-frame
    ret, img = video_capture.read()

    # Converting image to grayscale, detection model sees it as grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Applying the face detection method on the grayscale image
    faces_rect = face_cascade.detectMultiScale(gray_img, 1.1, 9)
    
    # Iterating through rectangles of detected faces, displaying the rectangle
    for (x, y, w, h) in faces_rect:
        cv2.rectangle(img, (x, y), (x+w, y+h), (70, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Video', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
