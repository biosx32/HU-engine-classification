import numpy as np
from keras.models import model_from_json
import DataPrep
import json


def load_dnn_model(filename):
	with open(filename+'.dnn.json', 'r') as json_file:
		loaded_model_json = json_file.read()

	model = model_from_json(loaded_model_json)
	model.load_weights(filename+".dnn.weights")
	with open(filename+'.dnn.labels', "rt") as readfile:
		text = readfile.read()
		labels = json.loads(text)
	return model, labels


def PredictLabel(model, labels, audio):
	# split audio by size that was trained on model

	input_shape = model.layers[0].input_shape[1]


	parts = DataPrep.split_by_data(audio, input_shape)

	if not parts:
		print("Not big enough sample to predict!!!")
		return "SAMPLE_NOT_BIG_ENOUGH"

	data = DataPrep.samples_to_data(parts)

	predictions = model.predict(data)
	avg = np.average(predictions, axis=0)
	max_index = avg.argmax()

	predicted_label = labels[str(max_index)]

	# print("Real: {}".format(audio['label']))
	# print("Predicted: {}".format(predicted_label))

	return predicted_label


def generate_labels_for_sample(mod_lab, large_sample, label_interval=5.0):
	ac_labels = []
	model, labels = mod_lab
	samples = DataPrep.split_sample_by_time(test_sample, label_interval)

	for i, sample in enumerate(samples):
		label = PredictLabel(model, labels, sample)
		# when sample not aligned to 'interval' seconds, we skip
		# because It's just prototype
		if not label:
			continue
		start = float(i * label_interval)
		end = (i * label_interval + label_interval)
		entry = DataPrep.ac_entry(start, end, label)

		if ac_labels and ac_labels[-1]['label'] == label:
			ac_labels[-1]['end'] = end
		else:
			ac_labels.append(entry)
	return ac_labels


classify_audio_path = 'TrainedNetwork\\engines.wav'
report_path = classify_audio_path+'_report.txt'
model_path = 'TrainedNetwork\\model'

mlb = load_dnn_model(model_path)
audio, rate = DataPrep.load_audio_file(classify_audio_path, 16000)
test_sample = DataPrep.create_sample('unknown', rate, audio)

result = generate_labels_for_sample(mlb, test_sample, label_interval=0.1)
DataPrep.save_ac_output(report_path, result)
print(result)




