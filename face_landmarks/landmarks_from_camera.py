from imutils import face_utils
import imutils
import dlib
import cv2
from configparser import ConfigParser

def main():

    rgb_list = [((0, 0, 255), (55, 0, 55), (34, 33, 87))]

    # create parser instance
    config = ConfigParser()

    # Read detection.cfg
    config.read('config/landmarks.cfg')
    predictor_dat = config.get('model', 'predictor_dat')

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_dat)


    #camera = cv2.VideoCapture(int(0))
    camera = cv2.VideoCapture(int(0))

    if camera.isOpened():
        while True:
            ret_val, frame  = camera.read()
            #frame = imutils.resize(frame, width=400)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # detect faces in the grayscale frame
            rects = detector(gray, 0)

            # loop over the face detections
            for rect in rects:

                (x, y, w, h) = face_utils.rect_to_bb(rect)
                #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                r=255
                g=0
                b=0
                point_counter=0
                for (sX, sY) in shape:
                    if point_counter == 17:
                        r = 255
                        g = 255
                        b = 0
                    elif point_counter == 22:
                        r = 0
                        g = 255
                        b = 0
                    elif point_counter ==  27:
                        r = 0
                        g = 255
                        b = 255
                    elif point_counter ==  31:
                        r = 170
                        g = 0
                        b = 255
                    elif point_counter == 36:
                        r = 255
                        g = 128
                        b = 0
                    elif point_counter == 42:
                        r = 0
                        g = 0
                        b = 255
                    elif point_counter == 48:
                        r = 255
                        g = 230
                        b = 255

                    cv2.circle(frame, (sX, sY), 1, (b, g, r), -1)
                    point_counter+=1


            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

    # do a bit of cleanup
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()