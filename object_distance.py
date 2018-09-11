# Run: python distance_between.py --image images/example_01.png --width 0.955
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

# perform edge detection
edged = cv2.Canny(gray, 50,100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.errode(edged, None, iterations=1)

# find contours
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

# sort the contours from left to right and initialize the distance colors
# and reference object
(cnts,_) = contours.sort_contours(cnts)
colors = ((0,0,255), (240, 0, 159,), (0,165,255), (255,255,0),(255,0,255))
refObj = None

for c in cnts:
	if cv2.contourArea(c) < 100:
		continue

		# compute rotated bounding box of the contours
		box = cv2.minAreaRect(c)
		box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
		box = np.array(box, dtype='int')

		box = perspective.order_points(box)

		#compute the centre of the bounding box
		cX = np.average(box[:,0])
		cY = np.average(box[:,1])

		if refObj is None:
			(tl,tr,bl,br) = box
			(tlblX, tlblY) = midpoint(tl,bl)
			(trbrX, trbrY) = midpoint(tr,br)

			D = dist.eucledean((tlblX,tlblY),(trbrX,trbrY))
			refObj = (box, (cX,cY), D/args['width'])
			continue

			# draw the contours on the image
	orig = image.copy()
	cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
	cv2.drawContours(orig, [refObj[0].astype("int")], -1, (0, 255, 0), 2)

	# stack the reference coordinates and the object coordinates
	# to include the object center
	refCoords = np.vstack([refObj[0], refObj[1]])
	objCoords = np.vstack([box, (cX, cY)])

	# loop over the original points
	for ((xA, yA), (xB, yB), color) in zip(refCoords, objCoords, colors):
		# draw circles corresponding to the current points and
		# connect them with a line
		cv2.circle(orig, (int(xA), int(yA)), 5, color, -1)
		cv2.circle(orig, (int(xB), int(yB)), 5, color, -1)
		cv2.line(orig, (int(xA), int(yA)), (int(xB), int(yB)),
			color, 2)

		# compute the Euclidean distance between the coordinates,
		# and then convert the distance in pixels to distance in
		# units
		D = dist.euclidean((xA, yA), (xB, yB)) / refObj[2]
		(mX, mY) = midpoint((xA, yA), (xB, yB))
		cv2.putText(orig, "{:.1f}in".format(D), (int(mX), int(mY - 10)),
			cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

		# show the output image
		cv2.imshow("Image", orig)
		cv2.waitKey(0)
