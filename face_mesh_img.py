import cv2
import matplotlib.pyplot as plt
import dlib
import os


green = (0,255,0)
blue = (0,0,255)
red = (255,0,0
        )
def draw_facial_part(img, landmarks, start, end, color = green, thic = 5):
    for i in range(start, end-1):
        cv2.line(img, landmarks[i], landmarks[i+1], color, thic)


path = os.path.join('images' ,'messi2.jpg')

img = cv2.imread(path)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Detect faces in the grayscale image
faces = detector(gray, 0)
# Loop over each face
for face in faces:
    # Use the predictor to find facial landmarks
    landmarks = predictor(gray, face)

# Loop over the landmark points and extract their (x, y) coordinates
landmark_points = []
for i in range(68):
    x = landmarks.part(i).x
    y = landmarks.part(i).y
    landmark_points.append((x, y))


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
cv2.line(img, landmark_points[36], landmark_points[41], red, 5)
# left eye
draw_facial_part(img, landmark_points, 42, 48, red)
cv2.line(img, landmark_points[42], landmark_points[47], red, 5)
# outer lip
draw_facial_part(img, landmark_points, 48, 60, red)
cv2.line(img, landmark_points[48], landmark_points[59], red, 5)
# inner lip
draw_facial_part(img, landmark_points, 60, 68, red)
cv2.line(img, landmark_points[60], landmark_points[67], red, 5)


plt.imshow(img)
plt.show()

