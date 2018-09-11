# Detect Skin in images and videos

import cv2
import numpy as np
import imutils
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-v','--video',help='path to video file.')
args = vars(ap.parse_args())

# define the upper and lower boundaries of the HSV pixel intensities to be considered 'skin'
upper = np.array([0,133,77], dtype='uint8')
lower = np.array([255,173,127], dtype='uint8')

# if a video path was not supplied, grab the reference to the gray else load the video
if not args.get('video',False):
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(args['video'])

while True:
    (grabbed,frame) = camera.read()

    if args.get('video') and not grabbed:
        break

    # resize the frame and convert into HSV
    frame = imutils.resize(frame, width=400)
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skin_mask = cv2.inRange(converted, lower,upper)

    # apply erosion and dilation
    kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
    skin_mask = cv2.erode(skin_mask, kernal, iterations=2)
    skin_mask = cv2.dilate(skin_mask, kernal, iterations=2)

    # blur the skin_mask
    skin_mask = cv2.GaussianBlur(skin_mask, (3,3),0)
    skin = cv2.bitwise_and(frame,frame, mask=skin_mask)

    # show image
    cv2.imshow('images', np.hstack([frame,skin]))

    if cv2.waitKey(1) & 0xFF == ord('q'):
         break

camera.release()
cv2.destroyAllWindows()
