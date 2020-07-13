import os
import dlib
from configparser import ConfigParser

def main():
    # create parser instance
    config = ConfigParser()

    # Read detection.cfg
    config.read('config/landmarks.cfg')

    landmarks_training_xml = config.get('training', 'landmarks_training_xml')
    landmarks_testing_xml = config.get('testing', 'landmarks_testing_xml')
    predictor_dat = config.get('model', 'predictor_dat')
    training_options = dlib.shape_predictor_training_options()
    training_options.oversampling_amount = int(config.get('training', 'oversampling_amount'))
    training_options.nu = float(config.get('training', 'nu'))
    training_options.tree_depth = int(config.get('training', 'tree_depth'))
    training_options.be_verbose = True if config.get('training', 'be_verbose') == "True" else False

    dlib.train_shape_predictor(landmarks_training_xml, predictor_dat, training_options)

    # average distance between predicted and actual face landmarks
    print("Training Accuracy Distance: {}".format(dlib.test_shape_predictor(landmarks_training_xml, predictor_dat)))
    print("Testing Accuracy Distance: {}".format(dlib.test_shape_predictor(landmarks_testing_xml, predictor_dat)))

if __name__ == "__main__":
    main()