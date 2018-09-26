import csv
import librosa
import sys
import IPython.display
import librosa.display
import sounddevice as sd
from time import sleep


def _dbg_print(*args, **kwargs):
	if 1:
		print("dbg: ", *args, **kwargs)


class Sample:
	# Wrapper class for better output
	def __init__(self, dt):
		for key in dt:
			self.__dict__[key] = dt[key]

	def __getitem__(self, item):
		return self.__dict__[item]

	def __str__(self):
		res = "Sample(Label: {}, Rate: {}, Data len: {}, Time: {})".format(
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


def load_sources_file(filename):
	source_list = []

	with open(filename, 'rt') as csvfile:
		csv_reader = csv.reader(csvfile, delimiter=',', quotechar='"')

		# read every csv line and append to
		for i, row in enumerate(csv_reader):

			if i == 0 or len(row) != 4:
				# skip headers and wrong entries
				continue

			_source = {
				'filename': row[0],
				'start': row[1],
				'end': row[2],
				'label': row[3]
			}

			source_list.append(_source)

	return source_list


def load_audio_cache(source_list, resample=8000):
	_AudioCache = {}

	for source in source_list:
		filename = source['filename']

		if filename not in _AudioCache:
			# probably can be done faster by using low-level audio converter
			_AudioCache[filename] = librosa.load('engines.wav', sr=resample)
		else:
			_dbg_print("Using cached entry for file {}".format(filename))

	return _AudioCache


def load_audio_samples(audio_cache, sample_sources):
	# get raw audio samples for every label
	_AudioSamples = []

	for audio_source in sample_sources:
		filename = audio_source['filename']

		if filename not in audio_cache.keys():
			_dbg_print("Not cached audio file: {}".format(filename))
			continue

		audio_data, audio_rate = audio_cache[filename]

		time_start = float(audio_source['start'])
		time_end = float(audio_source['end'])
		label = audio_source['label']

		# _dbg_print("Audio Sample range: ",time_start, time_end)

		start_sample = librosa.time_to_samples(time_start, sr=audio_rate)
		end_sample = librosa.time_to_samples(time_end, sr=audio_rate)

		# todo Safter way to get audio part?
		try:
			cutout_data = audio_data[start_sample:end_sample]
		except IndexError:
			_dbg_print("Error in getting part of audio file: Out of Range error")
			continue

		sample = create_sample(label, audio_rate, cutout_data)
		_AudioSamples.append(sample)

		# _dbg_print("Playing sample: ", sample)
		# sd.play(cutout_data, audio_rate*4)
		# sd.wait()

	return _AudioSamples


def _sample_to_shifted_group(raw_sample):
	# Generate shifted audio
	shifted_samples = []

	shift_start = -0.8
	shift_end = 0.2
	shift_step = 0.2

	shift_count = (shift_end - shift_start) / shift_step
	print("Generating  {} shiftsamples for {} ".format(shift_count, raw_sample))

	audio_data = raw_sample['data']
	audio_rate = raw_sample['rate']
	audio_label = raw_sample['label']

	shift_i = shift_start
	while shift_i <= shift_end:
		gen_data = librosa.effects.pitch_shift(audio_data, sr=audio_rate, n_steps=shift_i)
		gen_sample = create_sample(audio_label, audio_rate, gen_data)

		shifted_samples.append(gen_sample)
		shift_i += shift_step

		# print("playing shifted data..")
		# sd.play(gen_data, audio_rate)
		# sd.wait()

	return shifted_samples


def _split_sample(raw_sample, seconds=1):
	_SplitSamples = []

	rsample_data = raw_sample['data']
	audio_rate = raw_sample['rate']
	label = raw_sample['label']
	step_size = librosa.time_to_samples(seconds, sr=audio_rate)
	step_counter = 0

	sub_count = int(0.999 +len(rsample_data)/step_size)

	# iterate by X samples
	_dbg_print("Generating {} SplitSamples from {}".format(sub_count, label))

	while step_counter < len(rsample_data):
		end_step = step_counter + step_size

		# last sample will be probably shorter
		if end_step > len(rsample_data):
			end_step = len(rsample_data)

		subsample_data = rsample_data[step_counter:end_step]
		subsample = create_sample(label, audio_rate, subsample_data)
		_SplitSamples.append(subsample)

		step_counter += step_size

		# print("playing subsample...")
		# sd.play(subsample_data, audio_rate)
		# sd.wait()

	# print("Playing all _SplitSamples...")
	# for i in _SplitSamples:
	#  	 sd.play(i['data'], i['rate'])
	# 	 sd.wait()

	return _SplitSamples


def split_samples_by_time(sample_list, split_by=10):
	split_samples = []

	for sample in sample_list:
		sub_samples = _split_sample(sample, split_by)
		split_samples.extend(sub_samples)

	return split_samples


def generate_shifted_samples(raw_samples):
	shifted_samples = []

	for splitted in raw_samples:
		shifted_group = _sample_to_shifted_group(splitted)
		shifted_samples.extend(shifted_group)

	return shifted_samples

def get_training_data(fromfile):
	# Get training sources from CSV file
	SampleSources = load_sources_file(fromfile)

	# Load all required audio files
	# todo compare resampling success
	AudioCache = load_audio_cache(SampleSources, resample=8000)

	# Generate samples from AudioCache
	print("-" * 20, "Raw Samples", "-" * 20)
	RawSamples = load_audio_samples(AudioCache, SampleSources)
	print(*RawSamples, sep='\n')

	# Experimental trick to create more training data by shifting audio pitch
	# todo check accuracy
	# ShiftedSamples = get_shifted_samples(RawSamples)
	# print(*ShiftedSamples,sep='\n')
	print("-" * 20, "Shifted Samples", "-" * 20)

	# Free some resources to save memory
	# AudioCache = None

	# Split samples by 5 seconds to get "more" samples
	print("-" * 20, "Split Samples", "-" * 20)
	SplittedSamples = split_samples_by_time(RawSamples, split_by=5)

	print(*SplittedSamples, sep='\n')

	return SplittedSamples


