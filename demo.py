import os
import cv2
import numpy as np
from tensorflow import keras
from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions

from rich.progress import Progress

size = 4
webcam = cv2.VideoCapture(0)

vgg16 = keras.applications.VGG16(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)

# predict every n rames
n = 32
i = 0 # counter for frame prediction

# vgg size
width = 224
height = width

# rectangle
x1 = 300
y1 = x1
x2 = x1+width
y2 = y1+height

# text
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (x1,y1)
fontScale              = 1
fontColor              = (0,0,0)
thickness              = 3
lineType               = 2
lastLabel = ''

while True:
	i+=1

	(rval, im) = webcam.read()
	im = cv2.flip(im,1,1)

	cv2.rectangle(im, (300, 300), (624, 624), (224, 0, 0), 2)
	cropped = im[y1:y2, x1:x2]

	# Do a prediction every n frames
	if (i % n == 0):
		x = np.expand_dims(cropped, axis=0)
		x = preprocess_input(x)
	
		# Predict
		preds = vgg16.predict(x)
		preds = decode_predictions(preds, top=5)[0]
		os.system('clear')
		with Progress() as progress:
			bars = []
			for i, pred in enumerate(preds):
				category = pred[1]
				prob = pred[2]
				# set bars for console
				task = progress.add_task(category, total=1)
				progress.update(task,advance=prob)
				# look at the top pred for setting text
				if i == 0:
					lastLabel = f"{category} {prob:.0%}"

	# draw the last predicted label
	cv2.putText(im, lastLabel, 
	    bottomLeftCornerOfText, 
	    font, 
	    fontScale,
	    fontColor,
	    thickness,
	    lineType)

	cv2.imshow('LIVE', im)
	key = cv2.waitKey(10)
	if key == 27:
		break
webcam.release()

