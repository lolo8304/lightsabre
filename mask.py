import cv2
import numpy as np 
from detection.transform import show_thumb
from detection.transform import hough_lines_detection
from detection.Line import Line


l1 = 0
l2 = 95
l3 = 230
u1 = 55
u2 = 255
u3 = 255


def hough_lines_detection_configuration(img):
    lines = hough_lines_detection(img=img, rho=1, theta=np.pi / 180,
        threshold=1, min_line_len=80, max_line_gap=10)
    return lines


filename = "./input/media.io_2019-01-25 07.33.58.mp4"
cap = cv2.VideoCapture(filename)
while True:
    ret, frame = cap.read()
    if ret == False:
        cap = cv2.VideoCapture(filename)
        ret, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([l1, l2, l3])
    upper_orange = np.array([u1, u2, u3])
    
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    res = cv2.bitwise_and(frame, frame, mask = mask)
    res_grey = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    lines = hough_lines_detection_configuration(res_grey)
    if(lines is not None and lines.any() is not None):
        detected_lines = [Line(l[0][0], l[0][1], l[0][2], l[0][3]) for l in lines]
        i = 0
        for line in detected_lines:
            i = i + 1
            line.draw(res, color=(0, 255, 0), thickness=100)

    ##show_thumb("frame", frame, 1, 0)
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
        print("(",l1,",",l2,",",l3," - ",u1,",",u2,",",u3,")")


cv2.destroyAllWindows()
cap.release()
    