"""
Helper module for data generation
"""

import csv
import librosa
import librosa.display
import numpy as np
import json
import datetime
import time
import os
import tensorflow as tf
import keras


def save_model(model_path, model, labels):
	try:
		os.mkdir("TrainedNetwork")
	except IOError: pass

	try:
		model.save_weights(model_path + ".dnn.weights")
		model_json = model.to_json()

		with open(model_path + ".dnn", "w") as json_file:
			json_file.write(model_json)

		save_nn_labels(model_path + ".dnn.labels", labels)

		print("Save successful...")

	except IOError:
		print("Could not save model files")


def samples_to_data(samples):
	# convert samples to numpy-like array
	sample_size = len(samples[0]['data'])
	Datas = np.zeros((len(samples), sample_size))

	for i, _sam in enumerate(samples):
		label, data, sample_rate = _sam['label'], _sam['data'], _sam['rate']
		Datas[i] = data

	return Datas


def save_nn_labels(path, labels):
	# just saves NN labels
	labels = list(sorted(set(labels)))
	nums = range(len(labels))
	labels_dict = dict(zip(nums, labels))
	with open(path, 'wt') as outfile:
		raw = json.dumps(labels_dict)
		outfile.write(raw)


def samples_to_data_label_pair(samples):
	# returns labels and train data
	data = samples_to_data(samples)
	labels = [x['label'] for x in samples]
	return data, labels


def create_ac_entry(start, end, label):
	# not necessary but just for consistency
	return { 'start':start, 'end':end,'label': label }

def ace_to_str(entry):
	# convert ac entry to audacity line
	a=float(entry['start'])
	b=float(entry['end'])
	c=entry['label']
	return "{:5.6f}\t{:5.6f}\t{}\n".format(a,b,c)


def save_ac_output(filename, label_list):
	# save audacity label list
	with open(filename, "wt") as outfile:
		for entry in label_list:
			label = ace_to_str(entry)
			outfile.write(label)



class Sample:
	# Wrapper class of samples for better output
	def __init__(self, dt):
		for key in dt:
			self.__dict__[key] = dt[key]

	def __getitem__(self, item):
		return self.__dict__[item]

	def __str__(self):
		res = "SAMPLE(Label: {}, Rate: {}, Length: {}, Time: {}))".format(
			self.label, self.rate, len(self.data), len(self.data)/self.rate
		)

		return res

	def __repr__(self):
		return self.__str__()


def create_sample(label, rate, data):
	# crate dict with label, rate and data...
	# Return data to improve efficiency
	data = {
			'label': label,
			'rate': rate,
			'data': data
		}

	return Sample(data)


def get_time_in_sec(string):
	# convert time from sources to correct format (seconds)
	if string.count(':') <1:
		return string

	try:
		tsk = time.strptime(string, '%H:%M:%S')
	except ValueError: pass
	try:
		tsk = time.strptime(string, '%M:%S')
	except ValueError: return string

	sec = datetime.timedelta(hours=tsk.tm_hour, minutes=tsk.tm_min, seconds=tsk.tm_sec).total_seconds()
	return sec


def load_sources_file(filename):
	# just creates source list
	source_list = []
	relative_dir = os.path.dirname(filename) + "\\"

	with open(filename, 'rt') as csvfile:
		csv_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
		for i, row in enumerate(csv_reader):

			if i == 0 or len(row) != 4:
				# skip headers and wrong entries
				continue

			start = get_time_in_sec(row[1])
			end = get_time_in_sec(row[2])

			_source = {
				'filename': relative_dir+row[0],
				'start': start,
				'end': end,
				'label': row[3]
			}

			source_list.append(_source)

	return source_list




def load_audio_file(filename, resample):
	# ... you didn't see this, ok?
	return librosa.load(filename, sr=resample)


def load_audio_cache(source_list, resample):
	# load all necessary audio files only once
	_AudioCache = {}

	for source in source_list:
		filename = source['filename']

		if filename not in _AudioCache:
			# probably can be done faster by using low-level audio converter
			_AudioCache[filename] = load_audio_file(filename, resample)
		else:
			print("Using cached entry for file {}".format(filename))

	return _AudioCache


def load_audio_samples(audio_cache, sample_sources):
	# Create samples from audio_cache
	_AudioSamples = []

	for audio_source in sample_sources:
		filename = audio_source['filename']

		if filename not in audio_cache.keys():
			print("Not cached audio file: {}. Skipping...".format(filename))
			continue

		audio_data, audio_rate = audio_cache[filename]

		time_start = float(audio_source['start'])
		time_end = float(audio_source['end'])
		label = audio_source['label']

		start_sample = librosa.time_to_samples(time_start, sr=audio_rate)
		end_sample = librosa.time_to_samples(time_end, sr=audio_rate)

		# todo trim noise
		try:
			cutout_data = audio_data[start_sample:end_sample]
		except IndexError:
			# didn't happen, python doesn't care about ranges?
			print("Error in getting part of audio file: Out of Range error")
			continue

		sample = create_sample(label, audio_rate, cutout_data)
		_AudioSamples.append(sample)

	return _AudioSamples


def _sample_to_shifted_group(sample):
	# Generate pitch-shifted samples from one sample
	# It's pretty much slow and I'm not sure if it helps, but it may
	shifted_samples = []

	shift_start = -0.8
	shift_end = 0.2
	shift_step = 0.3

	shift_count = int((shift_end - shift_start) / shift_step)
	print("Generating  {:2.3f} shifted samples for {} ".format(shift_count, sample))

	audio_data = sample['data']
	audio_rate = sample['rate']
	audio_label = sample['label']

	shift_i = shift_start
	while shift_i <= shift_end:
		gen_data = librosa.effects.pitch_shift(audio_data, sr=audio_rate, n_steps=shift_i)
		gen_sample = create_sample(audio_label, audio_rate, gen_data)

		shifted_samples.append(gen_sample)
		shift_i += shift_step

	return shifted_samples


def split_by_data(raw_sample, data_count):
	# split samples by (samples?) xD ...
	# I mean data blocks
	sub_samples = []

	sample_data = raw_sample['data']
	audio_rate = raw_sample['rate']
	label = raw_sample['label']
	data_count = data_count
	step_counter = 0

	while step_counter < len(sample_data):
		end_step = step_counter + data_count

		# last sample is skipped, for having all samples the same size

		if end_step > len(sample_data):
			return sub_samples

		sub_data = sample_data[step_counter:end_step]
		subsample = create_sample(label, audio_rate, sub_data)
		sub_samples.append(subsample)

		step_counter += data_count

	return sub_samples


def split_sample_by_time(sample, time=1.0):
	# exactly like it sounds
	audio_rate = sample['rate']
	step_size = librosa.time_to_samples(time, sr=audio_rate)
	return split_by_data(sample, step_size)


def g_split_sample_by_time(sample_list, split_by_s=10):
	# same as above but for sample list
	split_samples = []

	for sample in sample_list:
		sub_samples = split_sample_by_time(sample, split_by_s)
		split_samples.extend(sub_samples)

	return split_samples


def generate_shifted_samples(sample_list):
	# Generate pitch-shifted samples from sample_list
	shifted_samples = []

	for sample in sample_list:
		shifted_group = _sample_to_shifted_group(sample)
		shifted_samples.extend(shifted_group)

	return shifted_samples


def load_training_samples(fromfile, resample, sample_time):
	# Get training sources from CSV file
	SampleSources = load_sources_file(fromfile)

	# Load all required audio files
	audio_cache = load_audio_cache(SampleSources, resample)

	# Generate samples from AudioCache
	print("-" * 20, "Audio Samples", "-" * 20)
	loaded_samples = load_audio_samples(audio_cache, SampleSources)
	print(*loaded_samples, sep='\n')

	# Experimental hack to create more training data by shifting audio pitch
	print("-" * 20, "Shifted Samples", "-" * 20)
	shifted_samples = generate_shifted_samples(loaded_samples)

	# Split samples by 'x' seconds to get "more" samples
	print("-" * 20, "Split Samples", "-" * 20)
	split_samples = g_split_sample_by_time(shifted_samples, split_by_s=sample_time)

	print("Generated {} samples in total".format(len(split_samples)))

	return split_samples


