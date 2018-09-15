import numpy as np
import cv2

def color_transfer(source,target):
    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype('float32')
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype('float32')

    # compute color statistics
    (l_mean_src,l_std_src, a_mean_src, a_std_src, b_mean_src, b_std_src) = image_stat(source)
    (l_mean_trg, l_std_trg, a_mean_trg, a_std_trg, b_mean_trg, b_std_trg) = image_stat(target)

    (l,a,b) = cv2.split(target)
    l -= l_mean_trg
    a -= a_mean_trg
    b -= b_mean_trg

    l = (l_std_trg / l_std_src) * l
    a = (a_std_trg / a_std_src) * a
    b = (b_std_trg / b_std_src) * b

    l += l_mean_src
    a += a_mean_src
    b += b_mean_src

    l = np.clip(l, 0, 255)
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)

    transfer = cv2.merge([l, a, b])
    transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)

	# return the color transferred image
    return transfer

def image_stat(image):
    (l, a, b) = cv2.split(image)
    (lMean, lStd) = (l.mean(), l.std())
    (aMean, aStd) = (a.mean(), a.std())
    (bMean, bStd) = (b.mean(), b.std())

	# return the color statistics
    return (lMean, lStd, aMean, aStd, bMean, bStd)
