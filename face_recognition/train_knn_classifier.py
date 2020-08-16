import math
from sklearn import neighbors
import os
import os.path
import pickle
import re
from PIL import Image
import numpy as np
import dlib
from configparser import ConfigParser
from utils import faces

# Get image paths for each labeled person in training directory
def get_image_paths(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png)', f, flags=re.I)]

# Train the KNN Classifier
def train(face_detector, face_encoder, face_landmarks, knn_algorithm, weights, train_dir, n_neighbors):

    face_encoding = []
    labels = []

    # Loop through each person in the training set
    for label_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, label_dir)):
            continue

        # Loop through each training image for the current directory/person
        for img_path in get_image_paths(os.path.join(train_dir, label_dir)):
            im = Image.open(img_path)
            im = im.convert('RGB')
            image = np.array(im)
            face_bounding_boxes = faces.get_face_locations(face_detector, image)

            if len(face_bounding_boxes) != 1:
                continue
            else:
                # Get face encodings for current person
                face_encoding.append(faces.get_face_encodings(face_encoder, face_landmarks, image, known_face_locations=face_bounding_boxes)[0])
                labels.append(label_dir)

    # Automatically select number of neighbors for classifier
    if n_neighbors == 0:
        n_neighbors = int(round(math.sqrt(len(face_encoding))))

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algorithm, weights=weights)
    knn_clf.fit(face_encoding, labels)

    return knn_clf

def main():

    # Load facial recognition configurations
    config = ConfigParser()
    config.read('config/recognition.cfg')

    train_image_dir = config.get('training', 'train_image_dir')
    model_save_path = config.get('training', 'model_save_path')
    n_neighbors = int(config.get('training', 'n_neighbors'))
    knn_algorithm = config.get('training', 'knn_algorithm')
    weights = config.get('training', 'weights')

    face_encoding_model = config.get('models', 'face_encoding_model')
    face_landmark_model = config.get('models', 'face_landmark_model')

    face_detector = dlib.get_frontal_face_detector()
    face_encoder = dlib.face_recognition_model_v1(face_encoding_model)
    face_landmarks = dlib.shape_predictor(face_landmark_model)

    # Train the KNN Model
    knn_classifier = train(face_detector, face_encoder, face_landmarks, knn_algorithm, weights, train_image_dir, n_neighbors)

    # Save the trained KNN classifier
    with open(model_save_path, 'wb') as f:
        pickle.dump(knn_classifier, f)

if __name__ == "__main__":
    main()