# import mediapipe
import cv2
import matplotlib.pyplot as plt
import dlib

img = cv2.imread("messi.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# Detect faces in the grayscale image
faces = detector(gray, 0)
print(faces)
# Loop over each face
for face in faces:
    # Use the predictor to find facial landmarks
    landmarks = predictor(gray, face)
print(landmarks)

# Loop over the landmark points and extract their (x, y) coordinates
landmark_points = []
for i in range(68):
    x = landmarks.part(i).x
    y = landmarks.part(i).y
    landmark_points.append((x, y))
    print(x, y)

# plt.imshow(img)
# plt.show()

# if cv2.waitKey(1) & 0xFF == ord('q'):
#     cv2.destroyAllWindows()

# cv2.imshow("webcam", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# print("hello")

