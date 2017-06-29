#!/usr/bin/env python
#!/usr/local/bin/python
#!/usr/bin/env PYTHONIOENCODING="utf-8" python
import os
import librosa
import tflearn
import numpy as np
import speech_data as data

# Simple speaker recognition demo, with 99% accuracy in under a minute ( on digits sample )

# | Adam | epoch: 030 | loss: 0.05330 - acc: 0.9966 -- iter: 0000/1000
# 'predicted speaker for 9_Vicki_260 : result = ', 'Vicki'
import tensorflow as tf
print("You are using tensorflow version "+ tf.__version__) #+" tflearn version "+ tflearn.version)
if tf.__version__ >= '0.12' and os.name == 'nt':
	print("sorry, tflearn is not ported to tensorflow 0.12 on windows yet!(?)")
	quit() # why? works on Mac?

learning_rate = 0.0001 
training_iters = 300000  # steps 
batch_size = 64 
width = 20  # mfcc features
height = 80  # (max) length of utterance
classes = 10  # digits  
test_path = "data/data_thuyg20_sre/ubm/"
vald_path = "data/test_data/"

speakers = data.new_get_speakers(test_path)
number_classes=len(speakers)
print("speakers",speakers)

batch=data.my_wave_batch_generator(batch_size=64, source=data.Source.DIGIT_WAVES, target=data.Target.speaker)
#vald_batch=data.my_wave_batch_generator(batch_size=64, source=data.Source.DIGIT_WAVES, target=data.Target.speaker,path=vald_path)
X,Y=next(batch)
#batch=data.mfcc_batch_generator(batch_size=64)
#batch=data.my_mfcc_batch_generator(batch_size=64, source=data.Source.DIGIT_WAVES, target=data.Target.speaker)
#vald_batch=data.my_mfcc_batch_generator(batch_size=64, source=data.Source.DIGIT_WAVES, target=data.Target.speaker,path=vald_path)


# Classification
tflearn.init_graph(num_cores=8, gpu_memory_fraction=0.5)

net = tflearn.input_data(shape=[None, 8192]) #Two wave chunks
#net = tflearn.input_data(shape=[None,width,height])
net = tflearn.fully_connected(net, 64)
net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net, number_classes, activation='softmax')
net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy')
model = tflearn.DNN(net)
model.fit(X, Y, n_epoch=100, show_metric=True, snapshot_step=100)
########################################### MFCC net ######################################
#net = tflearn.input_data(shape=[None,width,height])
#net = tflearn.lstm(net, 128*4, dropout=0.5)
#net = tflearn.lstm(net, 64, dropout=0.5)
#net = tflearn.fully_connected(net, number_classes, activation='softmax')
#net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy')
#model = tflearn.DNN(net, tensorboard_verbose=0)
# Training

#trainX, trainY = next(batch)
#testX, testY = next(vald_batch)
#model.fit(trainX, trainY, n_epoch=100, validation_set=(testX, testY), show_metric=True, batch_size=batch_size)
#model.fit(trainX, trainY, n_epoch=20000, show_metric=True, batch_size=batch_size)


###########################################################################################
#model.fit(X, Y, n_epoch=70, show_metric=True, snapshot_step=100)

# demo_file = "8_Vicki_260.wav"
#model.save("tflearn.lstm.model")
##demo_path = "data/test_data/"
##demo_file = "8_SwimGlass_140.wav"
#demo=data.load_wav_file(data.path + demo_file)
##wave, sr = librosa.load(demo_path+demo_file, mono=True)
##mfcc = librosa.feature.mfcc(wave, sr)
##mfcc=np.pad(mfcc,((0,0),(0,80-len(mfcc[0]))), mode='constant', constant_values=0)
##batch_features = []
##batch_features.append(np.array(mfcc))
##result=model.predict(batch_features)
##result=data.one_hot_to_item(result,speakers)
##print("predicted speaker for %s : result = %s "%(demo_file,result)) # ~ 97% correct
