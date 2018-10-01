import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
from Core import DataHelper

# I use CSV table for loading the train data
training_samples_path = ".\\Data\\metadata.csv"
model_path = ".\\TrainedNetwork\\dnn_model"

# resampling of wav files
sample_resample = 16000

# time in seconds, to split samples into
# around 10 ms has best accuracy
sample_split_time = 0.01

# for determining input_shape of NN
sample_size = int(sample_resample * sample_split_time)

# DNN configuration,
# I could get from 20mB to 200kB with same success rate (around 95%)

layer1_count = 250  # all combinations of different motors and noises, maybe
layer2_count = 50

# 'Learning speed'
epoch_count = 120
batch_size = 50
test_percent = 0.9


# Loading audio samples and converting to labels and numpy arrays
print("Loading samples...")
train_samples = DataHelper.load_training_samples(training_samples_path, sample_resample, sample_split_time)
tr_data, labels = DataHelper.samples_to_data_label_pair(train_samples)

# encoding labels to vectors
lb = LabelEncoder()
data_labels = np.array(labels)
data_labels = np_utils.to_categorical(lb.fit_transform(data_labels))
dl_count = len(set(labels))

# create some basic NN model with 2 layers
model = Sequential()



# I've read sigmoid is not ideal for DNN, so I used relu
# Dropout works best at 0.5

model.add(Dense(layer1_count, input_shape=(sample_size,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(layer2_count))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(dl_count))
model.add(Activation('softmax'))

# split train and test data
X_train, X_test, y_train, y_test = train_test_split(tr_data, data_labels, test_size=test_percent, random_state=0)

# I used loss function and optimizer which are used for sound processing
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
model.fit(X_train, y_train, batch_size=batch_size, epochs=epoch_count, validation_data=(X_test, y_test))

# Serialize model and weights
print("Saving model...")
DataHelper.save_model(model_path, model, labels)

print("Done!")