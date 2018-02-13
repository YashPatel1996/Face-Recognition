import dlib
import cv2
from imutils import face_utils
import numpy as np

# You can download the required pre-trained face detection model here:
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
predictor_model = "shape_predictor_68_face_landmarks.dat"
# Create a HOG face detector using the built-in dlib class
face_detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor(predictor_model)

#win = dlib.image_window()

# Load the image
#image = cv2.imread(file_name)

# Run the HOG face detector on the image data


# Show the desktop window with the image


# Loop through each face we found in the image

cap = cv2.VideoCapture(0)
while (True):
    ret, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detected_faces = face_detector(gray, 1)
    print("Found {} faces in the image file".format(len(detected_faces)))
    #win.set_image(image)
    for i, face_rect in enumerate(detected_faces):
        # Detected faces are returned as an object with the coordinates
        # of the top, left, right and bottom edges
        print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(),
                                                                                 face_rect.right(), face_rect.bottom()))
        pose_landmarks = face_pose_predictor(image, face_rect)
        pose_landmarks = face_utils.shape_to_np(pose_landmarks)
    eyes_1_pts = pose_landmarks[36:42]
    eyes_2_pts = pose_landmarks[42:48]
    #nose_pts = pose_landmarks[27:36]
    nose_pts_left = np.array([pose_landmarks[27],pose_landmarks[31],pose_landmarks[30]])
    nose_pts_right = np.array([pose_landmarks[27],pose_landmarks[30],pose_landmarks[35]])
    nose_pts_lower = np.array([pose_landmarks[31],pose_landmarks[30],pose_landmarks[35]])
    lips_upper = np.array([pose_landmarks[48],pose_landmarks[49],pose_landmarks[50],pose_landmarks[51],pose_landmarks[52],pose_landmarks[53],pose_landmarks[54],pose_landmarks[60],pose_landmarks[61],pose_landmarks[62],pose_landmarks[63],pose_landmarks[64]])
    lips_lower = np.array([pose_landmarks[54],pose_landmarks[55],pose_landmarks[56],pose_landmarks[57],pose_landmarks[58],pose_landmarks[59],pose_landmarks[48],pose_landmarks[60],pose_landmarks[67],pose_landmarks[66],pose_landmarks[65],pose_landmarks[64]])
    inside_lips = pose_landmarks[60:68]
    eyebrow_left = pose_landmarks[17:22]
    eyebrow_right = pose_landmarks[22:27]
    eyebrow_left = eyebrow_left.reshape((-1, 1, 2))
    eyebrow_right = eyebrow_right.reshape((-1, 1, 2))
    eyes_1_pts = eyes_1_pts.reshape((-1, 1, 2))
    eyes_2_pts = eyes_2_pts.reshape((-1, 1, 2))
    nose_pts_right = nose_pts_right.reshape((-1, 1, 2))
    nose_pts_lower = nose_pts_lower.reshape((-1, 1, 2))
    nose_pts_left = nose_pts_left.reshape((-1, 1, 2))
    lips_upper = lips_upper.reshape((-1, 1, 2))
    lips_lower = lips_lower.reshape((-1, 1, 2))
    inside_lips = inside_lips.reshape((-1, 1, 2))

    cv2.fillPoly(image, [eyes_1_pts],(0, 255, 255))
    cv2.fillPoly(image, [eyes_2_pts], (0, 255, 255))
    cv2.fillPoly(image, [nose_pts_left],(0, 255, 255))
    cv2.fillPoly(image, [nose_pts_right],(0, 255, 0))
    cv2.fillPoly(image, [nose_pts_lower],(0, 0, 255))
    cv2.fillPoly(image, [lips_upper],(0, 255, 255))
    cv2.fillPoly(image, [lips_lower],(255, 255, 255))
    cv2.fillPoly(image, [inside_lips],(0, 0, 0))
    cv2.polylines(image,[eyebrow_left],isClosed=False,thickness=4,color=(255,255,255))
    cv2.polylines(image,[eyebrow_right],isClosed=False,thickness=4,color=(255,255,255))
    #print("this image: ",pose_landmarks[19])
    cv2.imshow("frame", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()