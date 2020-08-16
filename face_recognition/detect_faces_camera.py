import cv2
import pickle
from PIL import Image, ImageDraw
import numpy as np
import dlib
from configparser import ConfigParser
from imutils import face_utils
from utils import faces
from utils import landmarks

# Predict Faces
def predict(face_detector, face_encoder, face_landmarks, img, threshold, model_path):

    with open(model_path, 'rb') as f:
        knn_clf = pickle.load(f)

    face_locations_img = faces.get_face_locations(face_detector,img)
    if len(face_locations_img) == 0:
        return []

    # Find encodings for image
    faces_encodings = faces.get_face_encodings(face_encoder, face_landmarks,img, known_face_locations=face_locations_img)

    # Find best matches for face in image above threshold
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    face_matches = [closest_distances[0][i][0] <= threshold for i in range(len(face_locations_img))]

    # Create tuple for predicted faces and locations
    predicted_faces=[]
    for label, location, rec in zip(knn_clf.predict(faces_encodings), face_locations_img, face_matches):
        if rec:
            predicted_faces.append(tuple([label, location]))
        else:
            predicted_faces.append(tuple(["unknown", location]))

    return predicted_faces

# Display predicted faces
def display_predictions(img, predictions):

    pil_image = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_image)

    for label, (top, right, bottom, left) in predictions:
        # Grow image to full
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # Draw a label with a name below the face
        label = label.encode("UTF-8")
        text_width, text_height = draw.textsize(label)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), label, fill=(255, 255, 255, 255))

    del draw
    return np.array(pil_image)

def main():

    # Read facial recognition configuration
    config = ConfigParser()
    config.read('config/recognition.cfg')

    face_encoding_model = config.get('models', 'face_encoding_model')
    face_landmark_model = config.get('models', 'face_landmark_model')
    knn_model = config.get('models', 'knn_model')
    threshold = float(config.get('predictions', 'threshold'))
    display_landmarks = int(config.get('display', 'landmarks'))

    # Enable face and landmark detector
    face_detector = dlib.get_frontal_face_detector()
    face_encoder = dlib.face_recognition_model_v1(face_encoding_model)
    face_landmarks = dlib.shape_predictor(face_landmark_model)

    camera = cv2.VideoCapture(int(0))
    if camera.isOpened():
        while True:
            ret_val, frame = camera.read()

            # Resize the image to half
            img = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

            # Display Facial landmarks
            if display_landmarks == 1:
                faces = face_detector(frame, 0)
                for face in faces:
                    shape = face_landmarks(frame, face)
                    shape = face_utils.shape_to_np(shape)
                    bgr = (0, 0, 0)
                    landmark = 0
                    for (sX, sY) in shape:
                        bgr = landmarks.get_landmark_color(bgr, landmark)
                        cv2.circle(frame, (sX, sY), 1, bgr, -1)
                        landmark += 1

            # Predict Face's
            predictions = predict(face_detector, face_encoder, face_landmarks, img, threshold, knn_model)

            frame = display_predictions(frame, predictions)

            # End camera stream
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

if __name__ == "__main__":
    main()