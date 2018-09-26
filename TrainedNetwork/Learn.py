import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics
import librosa
from sklearn.preprocessing import LabelEncoder
import Helpers
import sys


data_resample = 16000
data_time = 0.5

train_data = Helpers.get_training_data('ManualClassify.csv', resample=data_resample, sample_time=data_time)
train_count = len(train_data)

layer1_count = 1000
layer2_count = 200

layer1_drop = 0.5
layer2_drop = 0.5

sample_size = int(data_resample * data_time)


batch_size = int(train_count / 25 + 1)
epoch_count = 8





print("Learning on {} samples".format(train_count))

X = []
y = []

for one in train_data:
    q, sample_rate = one['data'],one['rate']
    # we extract mfcc feature from data
    #mfccs = np.mean(librosa.feature.mfcc(y=q, sr=sample_rate, n_mfcc=data_time).T, axis=0)

    #feature = mfccs
    label = one['label']
    X.append(q)
    y.append(label)

X = np.array(X)
y = np.array(y)

lb = LabelEncoder()
y = np_utils.to_categorical(lb.fit_transform(y))


xlen = X.shape[1]


num_labels = y.shape[1]
# build model
model = Sequential()

model.add(Dense(layer1_count, input_shape=(sample_size,)))
model.add(Activation('relu'))
model.add(Dropout(layer1_drop))

model.add(Dense(layer2_count))
model.add(Activation('relu'))
model.add(Dropout(layer2_drop))

model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

model.fit(X, y, batch_size=batch_size, epochs=epoch_count, validation_data=(X, y))

print("saving model...")
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.weights")

