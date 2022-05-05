import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

header = 'poradie, topLeft_x, topLeft_y, topRight_x, topRight_y, bottomRight_x, bottomRight_y, bottomLeft_x, bottomLeft_y'

changeable = 'test.txt'


def power_aproximator(x_log):
    if x_log == 0.0:
        return -1
    return 92.055 * x_log ** (-0.97)


def power_diff_aproximator(x_log):
    if x_log == 0.0:
        return -1
    return 3.1724 * x_log ** (-0.379)


def num_generator():
    i = 0
    while True:
        yield i
        i += 1


GENERATOR = num_generator()

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)

BLACK = (0, 0, 0)

with open(changeable, 'a') as outFile:
    outFile.write(header + '\n')

r_width_px_old = 0.0
nejaky_int = 0
while True:

    _, frame = cap.read()

    img = frame.copy()

    arucoDict = cv.aruco.Dictionary_get(cv.aruco.DICT_7X7_50)
    arucoParams = cv.aruco.DetectorParameters_create()
    (corners, ids, rejected) = cv.aruco.detectMarkers(img, arucoDict,
                                                      parameters=arucoParams)

    if len(corners) > 0:
        # flatten the ArUco IDs list
        ids = ids.flatten()
        # loop over the detected ArUCo corners
        for (markerCorner, markerID) in zip(corners, ids):
            # extract the marker corners (which are always returned in
            # top-left, top-right, bottom-right, and bottom-left order)
            corner = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corner

            if cv.waitKey(1) & 0xFF == ord('k') and nejaky_int != 0:
                nejaky_int = 0
                print('Koniec aktualneho merania')

            if cv.waitKey(1) & 0xFF == ord('p') and nejaky_int == 0:
                cislo = GENERATOR.__next__()
                record = str(cislo) + ',' + str(topLeft[0]) + ',' + str(topLeft[1]) + ',' + str(
                    topRight[0]) + ',' + str(topRight[1]) + ',' + str(bottomRight[0]) + ',' + str(
                    bottomRight[1]) + ',' + str(bottomLeft[0]) + ',' + str(bottomLeft[1])
                nejaky_int += 1
                # record = ':'.join(v for key, v in dict_values.items())
                with open(changeable, 'a') as outFile:
                    outFile.write(record + '\n')
                print("Detegujem...")
                print("Poradie merania: ", cislo)

            # print(topLeft)
            list_tuple = np.array([topLeft, topRight, bottomRight, bottomLeft], np.int32)
            list_tuple = list_tuple.reshape(-1, 1, 2)
            cv.polylines(img, [list_tuple], True, RED)

            r_width_px = np.sqrt((int(topLeft[0] - topRight[0])) ** 2 + (int(topLeft[1] - topRight[1])) ** 2)
            result = power_diff_aproximator(abs(r_width_px - r_width_px_old))
            r_width_px_old = r_width_px
            if result != -1:
                string2 = str(result)
                cv.putText(img, "Dist: " + string2, (10, 30), cv.FONT_HERSHEY_COMPLEX, 1, GREEN, 2)

            # convert each of the (x, y)-coordinate pairs to integers
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

        # cv.rectangle(img, topLeft, bottomRight, RED, 2)
        # cv.polylines(img, [topLeft, topRight, bottomRight, bottomLeft], True, RED, 2)

    cv.imshow("Img", img)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()


