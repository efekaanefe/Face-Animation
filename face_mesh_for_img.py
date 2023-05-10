import mediapipe as mp
import cv2

# mediapipe ain't working on my old laptop, gives -> Illegal instruction (core dumped)

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

img = cv2.imread("messi.jpg")

results = mp_face_mesh.FaceMesh(refine_landmarks = True)

# draw annotations on the image
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 
# if results.multi_face_landmarks:
#     for face_landmakrs in results.multi_face_landmarks:

#         # you can change the order to put more emphasis on the one at the last
#         mp_drawing.draw_landmarks(
#                 image = img, 
#                 landmark_list = face_landmakrs,
#                 connections = mp_face_mesh.FACEMESH_IRISES,
#                 landmark_drawing_spec =mp_drawing.DrawingSpec(color = (0,0,255)), # you can make it None 
#                 connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_iris_connections_style()
#                 )

#         mp_drawing.draw_landmarks(
#                 image = img, 
#                 landmark_list = face_landmakrs,
#                 connections = mp_face_mesh.FACEMESH_CONTOURS,
#                 landmark_drawing_spec = None ,
#                 connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_contours_style()
#                 )

#         mp_drawing.draw_landmarks(
#                 image = img, 
#                 landmark_list = face_landmakrs,
#                 connections = mp_face_mesh.FACEMESH_TESSELATION,
#                 landmark_drawing_spec = None,
#                 connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_tesselation_style()
#                 )


cv2.imshow("webcam", img)
cv2.waitKey()
# cv2.destroyAllWindows()


