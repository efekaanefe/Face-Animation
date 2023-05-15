import dlib 
import cv2

green = (0,255,0)
blue = (0,0,255)
red = (255,0,0
        )
def draw_facial_part(img, landmarks, start, end, color = green, thic = 1):
    for i in range(start, end-1):
        cv2.line(img, landmarks[i], landmarks[i+1], color, thic)


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
        landmarks = predictor(gray, face)

    for i in range(0,68):
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        landmark_points.append((x,y))
    # for (x, y) in landmark_points:
    #     cv2.circle(img, (x, y), 1, (0,255,255),1)

    # right jawline
    draw_facial_part(img, landmark_points, 1, 9)
    # left jawline
    draw_facial_part(img, landmark_points, 8, 17)
    # right eyebrow
    draw_facial_part(img, landmark_points, 17, 22, blue)
    # left eyebrow
    draw_facial_part(img, landmark_points, 22, 27, blue)
    # nose vertical
    draw_facial_part(img, landmark_points, 27, 31, blue)
    # nose horizontal
    draw_facial_part(img, landmark_points, 31, 36, blue)
    # right eye
    draw_facial_part(img, landmark_points, 36, 42, red)
    cv2.line(img, landmark_points[36], landmark_points[41], red, 1)
    # left eye
    draw_facial_part(img, landmark_points, 42, 48, red)
    cv2.line(img, landmark_points[42], landmark_points[47], red, 1)
    # outer lip
    draw_facial_part(img, landmark_points, 48, 60, red)
    cv2.line(img, landmark_points[48], landmark_points[59], red, 1)
    # inner lip
    draw_facial_part(img, landmark_points, 60, 68, red)
    cv2.line(img, landmark_points[60], landmark_points[67], red, 1)



    cv2.imshow("webcam", img)
    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

webcam.release()
cv2.destroyAllWindows()


