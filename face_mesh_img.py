import cv2
import matplotlib.pyplot as plt
import dlib
import os
import numpy as np

green = (0,255,0)
blue = (0,0,255)
red = (255,0,0)

thin_thic = 5
thick_thic = 15

landmark_indices_dict = {  # (start, end, color, thic)
            "right_jawline": (1,9, green, thin_thic),
            "left_jawline": (8,17, green, thin_thic),
            "right_eyebrow": (17,22, blue, thick_thic),
            "left_eyebrow": (22,27, blue, thick_thic),
            "nose": (27,36, blue, thin_thic),
            "right_eye": (36,42, red, thin_thic),
            "left_eye": (42,48, red, thin_thic),
            "outer_lip": (48,60, red, thin_thic),
            "inner_lip": (60,68, blue, thin_thic)
        }

def draw_landmark_lines(img, landmarks, start, end, color = green, thic = 5):
    for i in range(start, end-1):
        cv2.line(img, landmarks[i], landmarks[i+1], color, thic)

def draw_landmark_polygons(img, landmark_points, start, end, color):
    positions = np.array(landmark_points[start:end])
    cv2.fillPoly(img, pts = [positions], color=color)

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

for key, value in landmark_indices_dict.items():
    start, end, color, thic = value
    # drawing lines
    draw_landmark_lines(img, landmark_points, start, end, color, thic)
    # drawing_polygons
    if key in ["right_eye", "left_eye", "inner_lip", "outer_lip"]:
        draw_landmark_polygons(img, landmark_points, start, end, color)

plt.imshow(img)
plt.show()

