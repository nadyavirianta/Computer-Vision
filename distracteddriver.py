#import library
import numpy as np
import cv2
import seaborn as sns
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import os
import pandas as pd

import keras
from keras import backend as K
import time
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, BatchNormalization
from keras.layers import MaxPool2D
from keras.preprocessing.image import ImageDataGenerator

#%%
# input path 
base_path = "D:/nadya/Skripsi/state-farm-distracted-driver-detection/imgs/train/"
model_path = "D:/nadya/Gemastik/"
history_path = "D:/nadya/Gemastik/"

#%%
#fungsi evaluasi
def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1(y_true, y_pred):
    precision_m = precision(y_true, y_pred)
    recall_m = recall(y_true, y_pred)
    return 2*((precision_m*recall_m)/(precision_m+recall_m+K.epsilon()))


class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
#%%
#fungsi untuk plot history
def plot_history(history,modelName):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(history_path+"{}_acc.png".format(modelName))
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig(history_path+"{}_loss.png".format(modelName))
    plt.show()
    
    plt.plot(history.history['precision'])
    plt.plot(history.history['val_precision'])
    plt.title('Model Precision')
    plt.ylabel('Precision')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(history_path+"{}_precision.png".format(modelName))
    plt.show()
      
    plt.plot(history.history['recall'])
    plt.plot(history.history['val_recall'])
    plt.title('Model Recall')
    plt.ylabel('Recall')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(history_path+"{}_recall.png".format(modelName))
    plt.show()
    
    plt.plot(history.history['f1'])
    plt.plot(history.history['val_f1'])
    plt.title('Model F-measure')
    plt.ylabel('F-measure')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(history_path+"{}_f1.png".format(modelName))
    plt.show()
    
    plt.plot(times)
    plt.title('Training Time')
    plt.ylabel('Time')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    plt.savefig(history_path+"{}_training.png".format(modelName))
    plt.show()
#%%
#load data train dan validation
train_datagen = ImageDataGenerator(rescale = 1./255, validation_split=0.2)
training_set = train_datagen.flow_from_directory(base_path, target_size = (64,64), class_mode ='categorical',subset='training')
val_set = train_datagen.flow_from_directory(base_path,target_size = (64, 64),class_mode = 'categorical',subset='validation')

#%%
#membuat arsitektur CNN
def model(modelName, inputShape, trainData, valData, epoch, lr, batchSize, numKernelConv1, sizeKernelConv1, numKernelConv2, sizeKernelConv2, sizeDense1, dropout):
    classifier = Sequential()
    classifier.add(Conv2D(numKernelConv1, sizeKernelConv1, input_shape=inputShape))
    classifier.add(Activation('relu'))
    classifier.add(BatchNormalization())
    classifier.add(MaxPool2D(pool_size=(2,2)))
    classifier.add(Dropout(dropout))
    classifier.add(Conv2D(numKernelConv2, sizeKernelConv2))
    classifier.add(Activation('relu'))
    classifier.add(BatchNormalization())
    classifier.add(MaxPool2D(pool_size=(2,2)))
    classifier.add(Dropout(dropout))                    
    classifier.add(Flatten())
    classifier.add(Dense(sizeDense1))
    classifier.add(BatchNormalization())
    classifier.add(Activation('relu'))
    classifier.add(Dropout(dropout))
    
    classifier.add(Dense(10))
    classifier.add(Activation('softmax'))
    opt = keras.optimizers.Adam(lr=lr)
    
    classifier.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy',precision, recall,f1])
    
    nb_samples = int(len(training_set.filenames)/batchSize)
    
    checkpoint = ModelCheckpoint(model_path+"{}.h5".format(modelName), verbose=1, monitor= "val_acc",save_best_only=True)
    time_callback = TimeHistory()
    callbacks_list = [checkpoint, time_callback]
    history = classifier.fit_generator(trainData, steps_per_epoch =nb_samples, epochs = epoch, validation_data = valData,callbacks = callbacks_list)
    times = time_callback.times
    classifier_json = classifier.to_json()
    with open(model_path+"{}.json".format(modelName), "w") as json_file:
        json_file.write(classifier_json)
        
    classifier.save_weights(model_path+"{}.h5".format(modelName))
    print("Saved classifier to disk")
    
    return history, times

#parameter 
inputShape = (64,64,3)
trainData = training_set
valData = val_set
epoch = 50
lr = 0.002
batchSize = 64
numKernelConv1 = 64
sizeKernelConv1 =(3,3)
numKernelConv2 = 64
sizeKernelConv2 = (3,3)
sizeDense1 = 128
dropout = 0.1

modelName = "modelGemastikRGB64_epoch{}_lr{}_batch{}_numConv1{}_sizeConv1{}_numConv2{}_sizeDense1{}_dropout{}_dropoutonConv".format(epoch, lr, batchSize, numKernelConv1, sizeKernelConv1, numKernelConv2, sizeDense1, dropout)
history, times = model(modelName, inputShape, trainData, valData, epoch, lr, batchSize, numKernelConv1, sizeKernelConv1, numKernelConv2, sizeKernelConv2, sizeDense1, dropout)

hist_df = pd.DataFrame(history.history) 
hist_df["times"] = times
# or save to csv: 
hist_csv_file = history_path+'history_{}.csv'.format(modelName)
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)


plot_history(history,modelName)

#%%
#load model
from keras.models import model_from_json
opt = keras.optimizers.Adam(lr=lr)
json_file = open(model_path+"{}.json".format(modelName), 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights 
loaded_model.load_weights(model_path+"{}.h5".format(modelName))
print("Loaded model from disk")
 
# evaluate loaded model pada data validasi/test
loaded_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy',precision, recall,f1])

hist_txt_file = history_path+'hasilVal_{}.txt'.format(modelName)
hist = loaded_model.evaluate_generator(valData)

with open(hist_txt_file, 'w') as f:
    for item in hist:
        f.write("%s\n" % item)