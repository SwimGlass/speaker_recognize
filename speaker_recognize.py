#!/usr/bin/env python
#!/usr/local/bin/python
#!/usr/bin/env PYTHONIOENCODING="utf-8" python
import os
import librosa
import tflearn
import numpy as np
import speech_data as data
import shutil
import tensorflow as tf
print("You are using tensorflow version "+ tf.__version__) #+" tflearn version "+ tflearn.version)
if tf.__version__ >= '0.12' and os.name == 'nt':
	print("sorry, tflearn is not ported to tensorflow 0.12 on windows yet!(?)")
	quit() # why? works on Mac?

learning_rate = 0.0001 
training_iters = 300000  # steps 
width = 512
height = 512
batch_size = 64 
LABELED_DIR = "data/png_files"
train_data_path = "data/spoken_numbers/"

def speaker(filename):  # vom Dateinamen
	return filename.split("_")[1]
def get_speakers(path):
	files = os.listdir(path)
	def nobad(name):
		return "_" in name and not "." in name.split("_")[1]
	speakers=list(set(map(speaker,filter(nobad,files))))
	print(len(speakers)," speakers: ",speakers)
	return speakers

#speakers = get_speakers(train_data_path)
speakers = os.listdir(LABELED_DIR)
number_classes=len(speakers)
print("speakers",speakers)

#if not os.path.exists("data/png_files"):
#    	os.makedirs("data/png_files")
#for speaker in speakers:
#	direct_name = "data/png_files/"+speaker
#	if not os.path.exists(direct_name):
#    		os.makedirs(direct_name)
# load data
print('Loading data')
X, Y = tflearn.data_utils.image_preloader(LABELED_DIR, image_shape=(width, height), mode='folder', normalize=True, grayscale=True, categorical_labels=True, files_extension=None, filter_channel=False)
X_shaped = np.squeeze(X)
trainX, trainY = X_shaped, Y

# Network building
print('Building network')
net = tflearn.input_data(shape=[None, width, height])
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, number_classes, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy')
model = tflearn.DNN(net, tensorboard_verbose=3)
print('Training network')
model.fit(trainX, trainY, validation_set=0.15, n_epoch=100, show_metric=True, batch_size=batch_size)
model.save("tflearn.lstm.model")
