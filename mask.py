import cv2
import numpy as np
from detection.transform import show_thumb
from detection.transform import hough_lines_detection
from detection.transform import pipeline_GaussianBlur
from detection.Line import Line


l1 = 0
l2 = 110
l3 = 240
u1 = 255
u2 = 255
u3 = 255


def hough_lines_detection_configuration(img):
    lines = hough_lines_detection(img=img, rho=1, theta=np.pi / 180,
                                  threshold=1, min_line_len=80,
                                  max_line_gap=10)
    return lines


#filename = "./input/media.io_2019-01-25 07.33.58.mp4"
#filename = "./input/2019-03-23 12.12.02.mp4"
filenames = [
    "./input/2019-03-23 12.12.02.mp4",
    "./input/media.io_2019-01-25 07.33.58.mp4"]

filenameIndex = 0
cap = cv2.VideoCapture(filenames[filenameIndex])
while True:
    ret, frame = cap.read()
    if ret == False:
        filenameIndex = (filenameIndex + 1) % filenames.__len__()
        cap = cv2.VideoCapture(filenames[filenameIndex])
        ret, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([l1, l2, l3])
    upper_orange = np.array([u1, u2, u3])

    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    res_grey = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    res_grey = pipeline_GaussianBlur(res_grey, {
        "size": 9,
        "sigma": -0.1
    })

    lines = hough_lines_detection_configuration(res_grey)
    if(lines is not None and lines.any() is not None):
        detected_lines = [Line(l[0][0], l[0][1], l[0][2], l[0][3])
                          for l in lines]
        i = 0
        for line in detected_lines:
            i = i + 1
            color = i * 20 % 256
            line.draw(res, color=(255, 255, color), thickness=3)
            line.draw(frame, color=(255, 255, color), thickness=3)
        print("# lines = ", i)

    show_thumb("frame", frame, 1, 0)
    ##show_thumb("mask", mask, 0, 1)
    show_thumb("res", res, 0, 0)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

    summe = l1 + l2 + l3 + u1 + u2 + u3
    if k == 49:
        l1 = l1 + 5
    if k == 50:
        l2 = l2 + 5
    if k == 51:
        l3 = l3 + 5
    if k == 52:
        u1 = u1 - 5
    if k == 53:
        u2 = u2 - 5
    if k == 54:
        u3 = u3 - 5

    if k == 113:
        l1 = l1 - 5
    if k == 119:
        l2 = l2 - 5
    if k == 101:
        l3 = l3 - 5
    if k == 114:
        u1 = u1 + 5
    if k == 116:
        u2 = u2 + 5
    if k == 122:
        u3 = u3 + 5

    summe2 = l1 + l2 + l3 + u1 + u2 + u3

    if summe != summe2:
        print("(", l1, ",", l2, ",", l3, " - ", u1, ",", u2, ",", u3, ")")


cv2.destroyAllWindows()
cap.release()
