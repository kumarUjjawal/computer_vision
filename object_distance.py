import numpy as np
import argparse
from scipy.spatial import distance as dist
import imutils
from imutils import perspective
from imutils import contours
import cv2

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0] * 0.5), (ptA[1] + ptB[1] * 0.5))

# pass argument
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to image")
ap.add_argument("-w", "--width", type=float, help="width of the object in inches")

args = vars(ap.parse_args())

# load the image, convert to grayscale, and blur
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (7,7), 0)

# find contours
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

# sort the contours from left to right and initialize the distance colors
# and reference object
(cnts,_) = contours.sort_contours(cnts)
colors = ((0,0,255), (240, 0, 159,), (0,165,255), (255,255,0),(255,0,255))
refObj = None
