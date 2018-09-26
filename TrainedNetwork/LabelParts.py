import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
import Helpers
import librosa
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model: Sequential = model_from_json(loaded_model_json)
model.load_weights("model.weights")
x=None


labels = ["Damaged", "Healthy", "Noise"]

def _biggest_float_index(source):
	biggest_val = -9999999
	biggest_idx = -1
	for i in range(len(source)):
		if source[i] > biggest_val:
			biggest_idx = i
			biggest_val = source[i]
	return biggest_idx


def Predict(audio):

	parts = Helpers.split_by_samples(audio, sample_count=8000, align_strict=True)
	if not parts:
		return None
	real_label = audio['label']
	count = len(parts)
	sum = np.zeros([1, len(labels)])
	for sample in parts:
		arr = np.array([sample['data']])
		a = model.predict(arr)[0]
		sum= sum + a

	avg = (sum/count)[0].tolist()

	#for i, label in enumerate(labels):
	#	print("{} - chance {}".format(label, avg[i]))

	mindex = _biggest_float_index(avg)
	predict = labels[mindex]
	#print("Predicted: {}".format(predict))
	#print("Real: {}".format(real_label))

	return predict

scan_interval = 1

y, rate = librosa.load('engines.wav', sr=16000)
sample_audio = Helpers.create_sample('unknown', 16000, y)
samples = Helpers._split_sample(sample_audio, scan_interval)

ac_labels = []
for i, sample in enumerate(samples):
	label = Predict(sample)
	if not label:
		continue
	start = float(i * scan_interval)
	end = (i * scan_interval + scan_interval)
	entry = Helpers.ac_entry(start,end,label)

	if ac_labels and ac_labels[-1]['label'] == label:
		ac_labels[-1]['end'] = end
	else:
		ac_labels.append(entry)

print("finish")
print(ac_labels)
Helpers.save_ac_output('ac_labels.txt', ac_labels)





