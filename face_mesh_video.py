import dlib 
import cv2

webcam = cv2.VideoCapture(0)

hog_face_detector = dlib.get_frontal_face_detector()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while webcam.isOpened():
    success, img = webcam.read()

    # applying face mesh model using mediapipe
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector(gray, 0)
        
    landmark_points = []

    for face in faces:

        face_landmarks = predictor(gray, face)
        
        for i in range(0,68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            landmark_points.append((x,y))
    for (x, y) in landmark_points:
        cv2.circle(img, (x, y), 1, (0,255,255),1)

    cv2.imshow("webcam", img)
    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

webcam.release()
cv2.destroyAllWindows()


