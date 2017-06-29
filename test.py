import os 
path = "/home/swimglass/tensorflow-speech-recognition/data/data_thuyg20_sre/ubm/"
files = os.listdir(path) 
def nobad(name):                                                                                                                                          
    return name.split("_")[0]
def speaker(filename):
    return filename.split("_")[0] 
speakers = list(set(map(speaker,files)))
print speakers
