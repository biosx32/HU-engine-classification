import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import DataPrep


# I use CSV table for loading train data
sample_table_path = "train_data_labels.csv"

# resampling and splitting of wav files
sample_resample = 16000
sample_split_time = 0.05
sample_size = int(sample_resample * sample_split_time)

# nn configuration
layer1_count = 500  # all combinations of different motors and noises
layer2_count = 80
epoch_count = 10
batch_size = 100

# Loading audio samples and converting to labels and numpy arrays
train_samples = DataPrep.get_training_samples(sample_table_path, sample_resample, sample_split_time)
tr_data, labels = DataPrep.samples_to_data_label_pair(train_samples)

# encoding labels to vectors
lb = LabelEncoder()
data_labels = np.array(labels)
data_labels = np_utils.to_categorical(lb.fit_transform(data_labels))
dl_count = len(set(labels))

# create some basic NN model with 2 layers
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
X_train, X_test, y_train, y_test = train_test_split(tr_data, data_labels, test_size = 0.3, random_state = 0)

# I used loss functions and optimizers which are used for sound processing
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
model.fit(X_train, y_train, batch_size=batch_size, epochs=epoch_count, validation_data=(X_test, y_test))

# I expect accuracy at least > 70%, but probably more if trained on more data

# Serialize model and weights
print("Saving model...")
model.save_weights("model.dnn.weights")
model_json = model.to_json()

with open("model.dnn.json", "w") as json_file:
    json_file.write(model_json)

DataPrep.save_nn_labels(labels)

print("Save successful...")