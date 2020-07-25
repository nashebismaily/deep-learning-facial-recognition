import dlib
import cv2
from imutils import face_utils
from configparser import ConfigParser
from utils import landmarks

def main():
    config = ConfigParser()
    config.read('config/landmarks.cfg')
    predictor_dat = config.get('model', 'predictor_dat')
    display_landmarks = int(config.get('display', 'landmarks'))
    display_face_frame = int(config.get('display', 'face_frame'))
    display_overlay = int(config.get('display', 'overlay'))
    forehead_height = int(config.get('overlay', 'forehead_height'))
    overlay_image = config.get('overlay', 'hat')
    if display_overlay ==1:
        hat = cv2.imread(overlay_image, -1)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_dat)

    camera = cv2.VideoCapture(int(0))
    if camera.isOpened():
        while True:
            ret_val, frame  = camera.read()
            faces = detector(frame, 0)

            for face in faces:
                (x, y, w, h) = face_utils.rect_to_bb(face)

                if display_face_frame == 1:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                if display_overlay == 1:
                    landmarks.overlay_img_above_frame(frame, x, w, y-forehead_height, h, hat)

                if display_landmarks == 1:
                    shape = predictor(frame, face)
                    shape = face_utils.shape_to_np(shape)
                    bgr = (0, 0, 0)
                    landmark=0
                    for (sX, sY) in shape:
                        bgr = landmarks.get_landmark_color(bgr, landmark)
                        cv2.circle(frame, (sX, sY), 1, bgr, -1)
                        landmark+=1

            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()