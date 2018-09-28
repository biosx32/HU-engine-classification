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



def samples_to_data(samples):
	sample_size = len(samples[0]['data'])
	Datas = np.zeros((len(samples), sample_size))

	for i, _sam in enumerate(samples):
		label, data, sample_rate = _sam['label'], _sam['data'], _sam['rate']
		Datas[i] = data

	return Datas


def save_nn_labels(labels):
	labels = list(sorted(set(labels)))
	nums = range(len(labels))
	labels_dict = dict(zip(nums, labels))
	with open('model.dnn.labels', 'wt') as outfile:
		raw = json.dumps(labels_dict)
		outfile.write(raw)


def samples_to_data_label_pair(samples):
	data = samples_to_data(samples)
	labels = [x['label'] for x in samples]
	return data, labels


def ac_entry(start,end,label):
	# audacity label
	return {
		'start':start,
		'end':end,
		'label': label
	}

def ac_line(entry):
	# audacity formated line
	a=float(entry['start'])
	b=float(entry['end'])
	c=entry['label']
	return "{:5.6f}\t{:5.6f}\t{}\n".format(a,b,c)


def save_ac_output(filename, label_list):
	#save audacity label list
	with open(filename, "wt") as outfile:
		for entry in label_list:
			label = ac_line(entry)
			outfile.write(label)



class Sample:
	# Wrapper class for better output
	def __init__(self, dt):
		for key in dt:
			self.__dict__[key] = dt[key]

	def __getitem__(self, item):
		return self.__dict__[item]

	def __str__(self):
		res = "S:: Label: {}, Rate: {}, Length: {}, Time: {})".format(
			self.label, self.rate, len(self.data), len(self.data)/self.rate
		)

		return res

	def __repr__(self):
		return self.__str__()


def create_sample(label, rate, data):
	# Return data to improve efficiency
	data = {
			'label': label,
			'rate': rate,
			'data': data
		}

	return Sample(data)


def get_time_in_sec(string):
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
	source_list = []

	with open(filename, 'rt') as csvfile:
		csv_reader = csv.reader(csvfile, delimiter=',', quotechar='"')

		# read every csv line and append to
		for i, row in enumerate(csv_reader):

			if i == 0 or len(row) != 4:
				# skip headers and wrong entries
				continue

			start = get_time_in_sec(row[1])
			end = get_time_in_sec(row[2])

			_source = {
				'filename': row[0],
				'start': start,
				'end': end,
				'label': row[3]
			}

			source_list.append(_source)

	return source_list




def load_audio_file(filename, resample):
	return librosa.load(filename, sr=resample)


def load_audio_cache(source_list, resample):
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
	# get raw audio samples for every label
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
			print("Error in getting part of audio file: Out of Range error")
			continue

		sample = create_sample(label, audio_rate, cutout_data)
		_AudioSamples.append(sample)

	return _AudioSamples


def _sample_to_shifted_group(raw_sample):
	# Generate shifted audio
	shifted_samples = []

	shift_start = -0.8
	shift_end = 0.2
	shift_step = 0.3

	shift_count = int((shift_end - shift_start) / shift_step)
	print("Generating  {:2.3f} shifted samples for {} ".format(shift_count, raw_sample))

	audio_data = raw_sample['data']
	audio_rate = raw_sample['rate']
	audio_label = raw_sample['label']

	shift_i = shift_start
	while shift_i <= shift_end:
		gen_data = librosa.effects.pitch_shift(audio_data, sr=audio_rate, n_steps=shift_i)
		gen_sample = create_sample(audio_label, audio_rate, gen_data)

		shifted_samples.append(gen_sample)
		shift_i += shift_step

	return shifted_samples



def split_by_data(raw_sample, data_count):
	_SplitSamples = []

	rsample_data = raw_sample['data']
	audio_rate = raw_sample['rate']
	label = raw_sample['label']
	data_count = data_count
	step_counter = 0

	while step_counter < len(rsample_data):
		end_step = step_counter + data_count

		# last sample is skipped, for having all samples the same size

		if end_step > len(rsample_data):
			return _SplitSamples

		subsample_data = rsample_data[step_counter:end_step]
		subsample = create_sample(label, audio_rate, subsample_data)
		_SplitSamples.append(subsample)

		step_counter += data_count

	return _SplitSamples

def split_sample_by_time(raw_sample, time=1.0):
	_SplitSamples = []

	rsample_data = raw_sample['data']
	audio_rate = raw_sample['rate']
	label = raw_sample['label']
	step_size = librosa.time_to_samples(time, sr=audio_rate)
	return split_by_data(raw_sample, step_size)


def gsplit_sample_by_time(sample_list, split_by_s=10):
	split_samples = []

	for sample in sample_list:
		sub_samples = split_sample_by_time(sample, split_by_s)
		split_samples.extend(sub_samples)

	return split_samples


def generate_shifted_samples(raw_samples):
	shifted_samples = []

	for splitted in raw_samples:
		shifted_group = _sample_to_shifted_group(splitted)
		shifted_samples.extend(shifted_group)

	return shifted_samples


def get_training_samples(fromfile, resample, sample_time):
	# Get training sources from CSV file
	SampleSources = load_sources_file(fromfile)

	# Load all required audio files
	AudioCache = load_audio_cache(SampleSources, resample)

	# Generate samples from AudioCache
	print("-" * 20, "Raw Samples", "-" * 20)
	RawSamples = load_audio_samples(AudioCache, SampleSources)
	print(*RawSamples, sep='\n')

	# Experimental hack to create more training data by shifting audio pitch
	print("-" * 20, "Shifted Samples", "-" * 20)
	ShiftedSamples = generate_shifted_samples(RawSamples)

	# Free some resources to save memory
	AudioCache = None

	# Split samples by 'x' seconds to get "more" samples
	print("-" * 20, "Split Samples", "-" * 20)
	SplittedSamples = gsplit_sample_by_time(ShiftedSamples, split_by_s=sample_time)

	print("Generated {} samples in total".format(len(SplittedSamples)))

	return SplittedSamples


