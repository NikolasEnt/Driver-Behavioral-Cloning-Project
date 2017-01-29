import cv2
import argparse
import base64
import json
import time
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf

from img_tools import *

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

IMG_PROC_X = 128
IMG_PROC_Y = 64
STEER_GAIN = 1.0
i = 0 #Number of frame
img_prev = None
steering_ang_prev = 0

@sio.on('telemetry')
def telemetry(sid, data):
	global i
	global img_prev
	global steering_ang_prev
	# The current steering angle of the car
	steering_angle = data["steering_angle"]
	# The current throttle of the car
	throttle = data["throttle"]
	# The current speed of the car
	speed = data["speed"]
	# The current i		mage from the center camera of the car
	imgString = data["image"]
	image = Image.open(BytesIO(base64.b64decode(imgString)))
	image_array = np.asarray(image)
	if i%3==0: #Use only every 5th frame for predictions
		b,g,r = cv2.split(image_array)
		image_array = cv2.merge((r,g,b))
		image_array = pred_img(image_array, IMG_PROC_X, IMG_PROC_Y)
		if i<1:
			img_conc=np.concatenate((image_array, image_array), axis=2)
		else:
			img_conc=np.concatenate((image_array, img_prev), axis=2)
		transformed_image_array = img_conc[None, :, :, :]
		# This model currently assumes that the features of the model are just the images. Feel free to change this.
		steering_angle = STEER_GAIN*float(model.predict(transformed_image_array, batch_size=1))
		steering_ang_prev = steering_angle
		img_prev = image_array
	# The driving model currently just outputs a constant throttle. Feel free to edit this.
	throttle = 1 #Full speed ahead!
	print(steering_ang_prev, throttle)
	send_control(steering_ang_prev, throttle)
	i = i+1

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)

def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Remote Driving')
	parser.add_argument('model', type=str,
	help='Path to model definition json. Model weights should be on the same path.')
	args = parser.parse_args()
	with open(args.model, 'r') as jfile:
		model = model_from_json(json.load(jfile))
	model.compile("adam", "mse")
	weights_file = args.model.replace('json', 'h5')
	model.load_weights(weights_file)
	# wrap Flask application with engineio's middleware
	app = socketio.Middleware(sio, app)
	# deploy as an eventlet WSGI server
	eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

