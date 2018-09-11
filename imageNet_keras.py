# Run: python imageNet_keras.py --image images/image_name.jpg --model vgg16

from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception # TensorFlow ONLY
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i','--image', required=True, help="path to image")
ap.add_argument('-model','--model', type=str, default='vgg16', help="name of pre-trained model")
args = vars(ap.parse_args())

# dictionary to map names to Keras classes
MODEL =   {
    'vgg16': VGG16
    'vgg19': VGG19
    'inception': InceptionV3
    'xception': Xception
    'resne': ResNet50
}

if args["model"] not in MODEL.keys():
    raise AssertionError("The --model name should be in 'MODEL' dictionary.")

input_shape = (224,224)
preprocess = imagenet_utils.preprocess_input

if args['model'] in ('inception','xception'):
    input_shape = (229,229)
    preprocess = preprocess_input

# load the network weights
print("[INFO] loading{}...".format(args['model']))
Network = MODEL[args['model']]
model = Network(weights='imagenet')

# load input image using Keras
print("[INFO] loading and pre-processing image...")
image = load_img(args["image"],target_size=input_shape)
image = img_to_array(image)

image = np.expand_dims(image,axis=0)

image = preprocess(image)

# clasify
print("[INFO] classifying image with '{}'...".format(args["model"]))
preds = model.predict(image)
P = imagenet_utils.decode_predictions(preds)

for (i, (imagenetID, label, prob)) in enumerate(P[0]):
	print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))


# display the image to our screen
orig = cv2.imread(args["image"])
(imagenetID, label, prob) = P[0][0]
cv2.putText(orig, "Label: {}, {:.2f}%".format(label, prob * 100),
	(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
cv2.imshow("Classification", orig)
cv2.waitKey(0)
