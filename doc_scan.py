# Scan Document

from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils

ap = argparse.ArgumentParser()
ap.add_argument('-i','--image',help="Path to image file.")
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
ratio = image.shape[0]/500.0
orig = image.copy()
height = imutils.resize(image, height=500)

# convert image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5,5),0)
edged = cv2.Canny(gray, 75,200)

# find contours
cnts = cv2.findContours(edged.copy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

for c in cnts:
    # approximate the contours
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    if len(approx) == 4:
        screenCnt = approx
        break

# show contour
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow('Outline',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
