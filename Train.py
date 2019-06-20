import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from Core import Processing

# I use CSV table for loading the train data
metadata_paths = ".\\Data\\metadata.csv"
model_path = ".\\TrainedNetwork\\dnn_model"

# resampling of wav files
sample_resample = 16000

# time to split samples into (seconds)
# around 10 ms had the best accuracy
sample_split_time = 0.01

# determine the input_shape of NN
sample_size = int(sample_resample * sample_split_time)

# DNN configuration

layer1_count = 250
layer2_count = 50

epoch_count = 10
batch_size = 50
test_percent = 0.5


# Loading audio samples and converting them into labels and numpy array combination
print("Loading samples...")
train_samples = Processing.load_training_samples(metadata_paths, sample_resample, sample_split_time)
tr_data, labels = Processing.samples_to_label_pairs(train_samples)

# encoding labels to vectors
lb = LabelEncoder()
data_labels = np.array(labels)
data_labels = np_utils.to_categorical(lb.fit_transform(data_labels))
dl_count = len(set(labels))

# create classifier model with 2 layers
model = Sequential()

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

# I used loss function and optimizer which are commonly used in sound processing
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
model.fit(X_train, y_train, batch_size=batch_size, epochs=epoch_count, validation_data=(X_test, y_test))

# Serialize model and weights
print("Saving model...")
Processing.save_model(model_path, model, labels)

print("Done!")