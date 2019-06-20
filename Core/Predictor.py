import numpy as np
from keras.models import model_from_json
import Core.Processing as DataPrep
import json


def load_dnn_model(filename):
	# load pre-trained model, weights and labels
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

	# Split the audio into smaller chunks
	parts = DataPrep.split_by_data(audio, input_shape)

	if not parts:
		print("SAMPLE TOO SMALL")
		return "SAMPLE_TOO_SMALL"

	# convert sample to numpy-like array
	data = DataPrep.samples_to_numpy(parts)

	predictions = model.predict(data)

	# average probabilities of all sample chunks
	avg = np.average(predictions, axis=0)

	# find the most probable label
	max_index = avg.argmax()
	predicted_label = labels[str(max_index)]

	return predicted_label


def generate_labels_for_sample(mod_lab, large_sample, label_interval=5.0):
	audio_labels = []
	model, labels = mod_lab

	# Split audio into chunks
	samples = DataPrep.split_sample_by_time(large_sample, label_interval)

	# Generate labels for every 'x' seconds
	for i, sample in enumerate(samples):
		# Predict label for one chunk
		label = predict_label(model, labels, sample)

		if not label:
			continue

		start = float(i * label_interval)
		end = (i * label_interval + label_interval)
		entry = {'start': start, 'end': end,'label': label}


		# merge labels with same name
		if audio_labels and audio_labels[-1]['label'] == label:
			audio_labels[-1]['end'] = end
		else:
			audio_labels.append(entry)

	return audio_labels



