import numpy as np
import dlib

# Return coordinates for faces in image
def get_face_locations(face_detector, img, upsample=1):
    face_locations=[]
    for face in face_detector(img, upsample):
        loc = (face.top(), face.right(), face.bottom(), face.left())
        face_locations.append(tuple([max(loc[0], 0), min(loc[1], img.shape[1]), min(loc[2], img.shape[0]), max(loc[3], 0)]))
    return face_locations

# Return 128D Face Encoding
def get_face_encodings(face_encoder, face_landmarks, face_image, known_face_locations, num_jitters=1):
    face_locations=[]
    for face_location in known_face_locations:
        face_locations.append(dlib.rectangle(face_location[3], face_location[0], face_location[1], face_location[2]))

    raw_landmarks =[]
    for face_location in face_locations:
        raw_landmarks.append(face_landmarks(face_image, face_location))

    face_landmarks_set=[]
    for raw_landmark_set in raw_landmarks:
        face_landmarks_set.append(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters))

    return np.array(face_landmarks_set)


