import numpy as np
from keras.models import model_from_json
import Core.DataHelper as DataPrep
import json


def load_dnn_model(filename):
	# just load pre-trained model, weights and labels
	with open(filename+'.dnn', 'r') as json_file:
		loaded_model_json = json_file.read()

	model = model_from_json(loaded_model_json)
	model.load_weights(filename+".dnn.weights")
	with open(filename+'.dnn.labels', "rt") as readfile:
		text = readfile.read()
		labels = json.loads(text)
	return model, labels


def predict_label(model, labels, audio):
	# Get size of input data for NN
	input_shape = model.layers[0].input_shape[1]

	# Split into parts to feed NN
	parts = DataPrep.split_by_data(audio, input_shape)

	# self explanatory
	if not parts:
		print("Not big enough sample to predict!!!")
		return "SAMPLE_NOT_BIG_ENOUGH"

	# convert sample to numpy-like array
	data = DataPrep.samples_to_data(parts)

	# predict ...
	predictions = model.predict(data)

	# average all parts of audio to get most probable result
	avg = np.average(predictions, axis=0)

	# get most probable label
	max_index = avg.argmax()
	predicted_label = labels[str(max_index)]

	return predicted_label


def generate_labels_for_sample(mod_lab, large_sample, label_interval=5.0):
	ac_labels = []
	model, labels = mod_lab

	# Split audio into chunks
	samples = DataPrep.split_sample_by_time(large_sample, label_interval)

	# Generate labels for every 'x' seconds interval
	for i, sample in enumerate(samples):
		# Predict label for one chunk
		label = predict_label(model, labels, sample)

		# I decided to skip last sample, if it's not long enough to fill interval
		if not label:
			continue

		start = float(i * label_interval)
		end = (i * label_interval + label_interval)
		entry = DataPrep.create_ac_entry(start, end, label)

		# merge labels with same text
		if ac_labels and ac_labels[-1]['label'] == label:
			ac_labels[-1]['end'] = end
		else:
			ac_labels.append(entry)

	return ac_labels



