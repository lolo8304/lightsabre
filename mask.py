import cv2
from skimage.measure import compare_ssim
import imutils
import numpy as np
from detection.transform import show_thumb
from detection.transform import hough_lines_detection
from detection.transform import pipeline_GaussianBlur
from detection.Line import Line

def detect_RED():
    return {
        "index": 0,
        "text" : "red",
        "l1" : 0,
        "l2" : 235,
        "l3" : 35,
        "u1" : 200,
        "u2" : 255,
        "u3" : 255
    }

def detect_BLACK():
    return {
        "index": 1,
        "text" : "black",
        "l1" : 0,
        "l2" : 0,
        "l3" : 255,
        "u1" : 95,
        "u2" : 255,
        "u3" : 255
    }

def detect_CAR_image(frame, car):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([ car["l1"], car["l2"], car["l3"] ])
    upper_orange = np.array([ car["u1"], car["u2"], car["u3"] ])

    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    res_grey = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    res_grey = pipeline_GaussianBlur(res_grey, {
        "size": 9,
        "sigma": -0.1
    })
    return { "mask": mask, "grey": res_grey, "color": res};


def drawLines(text, lines, frame, images, v_index):
    show_thumb("mask-"+text, images["mask"],    0, v_index)
    if(lines is not None and lines.any() is not None):
        detected_lines = [Line(l[0][0], l[0][1], l[0][2], l[0][3])
                          for l in lines]
        i = 0
        for line in detected_lines:
            i = i + 1
            color = i * 20 % 256
            #line.draw(images["color"], color=(255, 255, color), thickness=5)
            #line.draw(frame, color=(255, 255, color), thickness=5)
        # print("# lines = ", i)
    show_thumb("res-"+text, images["color"],      1, v_index)
    show_thumb("frame-"+text, frame,  2, v_index)

def detectAndDrawCar(frame, background_image, car):
    res_images = detect_CAR_image(frame, car);
    res_imagesBG = detect_CAR_image(background_image, car);
    detectDifference(res_images["mask"], res_imagesBG["mask"], False)
    #lines_B = hough_lines_detection_configuration(res_images["grey"])
    #drawLines(car["text"], lines_B, frame, res_images, car["index"])
    drawLines(car["text"], None, frame, res_images, car["index"])


def hough_lines_detection_configuration(img):
    lines = hough_lines_detection(img=img, rho=1, theta=np.pi / 180,
                                  threshold=1, min_line_len=80,
                                  max_line_gap=10)
    return lines

def sumLevelsCar(car):
    return car["l1"]+car["l2"]+car["l3"]+car["u1"]+car["u2"]+car["u3"];

def adaptLevelsCar(car, k):
    if k == 49:
        car["l1"] = car["l1"] + 5
    if k == 50:
        car["l2"] = car["l2"] + 5
    if k == 51:
        car["l3"] = car["l3"] + 5
    if k == 52:
        car["u1"] = car["u1"] - 5
    if k == 53:
        car["u2"] = car["u2"] - 5
    if k == 54:
        car["u3"] = car["u3"] - 5

    if k == 113:
        car["l1"] = car["l1"] - 5
    if k == 119:
        car["l2"] = car["l2"] - 5
    if k == 101:
        car["l3"] = car["l3"] - 5
    if k == 114:
        car["u1"] = car["u1"] + 5
    if k == 116:
        car["u2"] = car["u2"] + 5
    if k == 122:
        car["u3"] = car["u3"] + 5
    return car


## cool feature from py image search
## https://www.pyimagesearch.com/2017/06/19/image-difference-with-opencv-and-python/
def detectDifference(imageA, imageB, isColor):
    grayA = imageA
    grayB = imageB
    if (isColor):
        grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    # compute the Structural Similarity Index (SSIM) between the two
    # images, ensuring that the difference image is returned
    (score, diff) = compare_ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    print("SSIM: {}".format(score))
    # threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # loop over the contours
    for c in cnts:
        # compute the bounding box of the contour and then draw the
        # bounding box on both input images to represent where the two
        # images differ
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)
    #cv2.imshow("Original", imageA)
    #cv2.imshow("Modified", imageB)
    #cv2.imshow("Diff", diff)
    #cv2.imshow("Thresh", thresh)
    show_thumb("diff", diff, 0, 2)
    show_thumb("Thresh", thresh, 1, 2)




#filename = "./input/media.io_2019-01-25 07.33.58.mp4"
#filename = "./input/2019-03-23 12.12.02.mp4"
#filenames = [
#    "./input/2019-03-23 12.12.02.mp4",
#    "./input/media.io_2019-01-25 07.33.58.mp4"]
filenames = [
    "./input/2019-09-02 20.03.23.mov",
    "./input/2019-09-02 20.03.23-trim.mov"]

filenameIndex = 0
cap = cv2.VideoCapture(filenames[filenameIndex])
car_red = detect_RED()
car_black = detect_BLACK()

background_image = cv2.imread("./backgrounds/image.jpg")

while True:
    ret, frame = cap.read()
    if ret == False:
        filenameIndex = (filenameIndex + 1) % filenames.__len__()
        cap = cv2.VideoCapture(filenames[filenameIndex])
        ret, frame = cap.read()

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    if k == 48: ## char 0
        cv2.imwrite('./backgrounds/image.jpg', frame)
        background_image = cv2.imread("./backgrounds/image.jpg")

    red = 0
    if (red == 1):
        summe = sumLevelsCar(car_red)
        car_red = adaptLevelsCar(car_red, k)
        summe2 = sumLevelsCar(car_red)
        car = car_red
    if (red == 0):
        summe = sumLevelsCar(car_black)
        car_black = adaptLevelsCar(car_black, k)
        summe2 = sumLevelsCar(car_black)
        car = car_black

    if summe != summe2:
        print("(", car["l1"], ",", car["l2"], ",", car["l3"], " - ", car["u1"], ",", car["u2"], ",", car["u3"], ")")

    detectAndDrawCar(frame, background_image, car_black)
    detectAndDrawCar(frame, background_image, car_red)
    #detectDifference(frame, background_image, True)


cv2.destroyAllWindows()
cap.release()
