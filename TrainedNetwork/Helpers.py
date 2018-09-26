import csv
import librosa
import sys
import IPython.display
import librosa.display
import sounddevice as sd
from time import sleep

def _dbg_print(*args, **kwargs):
	if 1:
		print("dbg: ",*args,**kwargs)


class Sample:
	def __init__(self, dt):
		for key in dt:
			self.__dict__[key] = dt[key]

	def __getitem__(self, item):
		return self.__dict__[item]


	def __str__(self):
		res = "Sample(Label: {}, Rate: {}, Datalen: {}, Dur: {})".format(
			self.label, self.rate, len(self.data), len(self.data)/self.rate
		)

		return res

	def __repr__(self):
		return self.__str__()

def create_sample(label, rate, data):
	data =  {
			'label': label,
			'rate': rate,
			'data': data
		}

	#use Sample object or just return data
	return Sample(data)



def get_training_sources(filename):
	# open csv and create list of sample sources
	SampleSources = []

	with open(filename, 'rt') as csvfile:

		csv_reader = csv.reader(csvfile, delimiter=',', quotechar='"')

		# read every csv line and append to
		for i, row in enumerate(csv_reader):

			if i == 0 or len(row) != 4:
				# skip headers and wrong entries
				continue


			source = {
				'filename': row[0],
				'start': row[1],
				'end': row[2],
				'label': row[3]
			}

			SampleSources.append(source)

	return SampleSources


def load_audio_cache(sample_sources):
	# create a dictionary of filename: audio_data
	_AudioCache = {}

	for sample_src in sample_sources:
		filename = sample_src['filename']

		if filename not in _AudioCache:
			# downsample wav file to 8khz for same training data
			# probably can be faster by using C++ audio converter
			_AudioCache[filename] = librosa.load('engines.wav', sr=None)
			#todo resample to 8kHz
		else:
			_dbg_print("Using cached entry for file {}".format(filename))

	return _AudioCache


def get_raw_samples(audio_cache, sample_sources):
	# get raw audio samples for every label
	raw_samples = []
	for sample_src in sample_sources:
		filename = sample_src['filename']

		if filename not in audio_cache.keys():
			_dbg_print("Not cached audio: {}".format(filename))
			continue

		audio_data, audio_samplerate = audio_cache[filename]
		time_start = float(sample_src['start'])
		time_end = float(sample_src['end'])

		#_dbg_print("sample range: ",time_start, time_end)

		start_sample = librosa.time_to_samples(time_start, sr=audio_samplerate)
		end_sample = librosa.time_to_samples(time_end, sr=audio_samplerate)

		cutout_data = audio_data[start_sample:end_sample]
		label = sample_src['label']
		sample = create_sample(label, audio_samplerate, cutout_data)
		raw_samples.append(sample)

		#_dbg_print("Playing sample: ", sample)
		#sd.play(cutout_data, audio_samplerate*4)
		#sd.wait()
	return raw_samples

def get_shiftsample_group(raw_sample):
	#only one down-pitch, because it sound

	shift_start = -0.8
	shift_end = 0.2
	shift_step = 0.2
	shift_count = (shift_end - shift_start) / shift_step
	print("Generating  {} shiftsamples for {} ".format(shift_count, raw_sample))

	audio_data = raw_sample['data']
	audio_rate = raw_sample['rate']
	audio_label = raw_sample['label']

	shifted_samples = []

	shift_i = shift_start

	while shift_i <= shift_end:
		gen_data = librosa.effects.pitch_shift(audio_data, sr=audio_rate, n_steps=shift_i)
		gen_sample = create_sample(audio_label, audio_rate, gen_data)

		shifted_samples.append(gen_sample)

		#print("playing shifted data..")
		#sd.play(gen_data, audio_rate)
		#sd.wait()

		shift_i += shift_step

	return shifted_samples


def split_sample_by(raw_sample, seconds=1):
	subsamples = []

	rsample_data = raw_sample['data']
	audio_rate = raw_sample['rate']
	label = raw_sample['label']
	step_size = librosa.time_to_samples(seconds, sr=audio_rate)
	step_counter = 0

	sub_count = int(0.999 +len(rsample_data)/step_size)

	# iterate by X samples
	print("Generating {} subsamples from {}".format(sub_count, label))

	while step_counter < len(rsample_data):
		end_step = step_counter + step_size

		# last sample will be probably shorter
		if end_step > len(rsample_data):
			end_step = len(rsample_data)

		subsample_data = rsample_data[step_counter:end_step]

		subsample = create_sample(label, audio_rate, subsample_data)
		subsamples.append(subsample)

		duration = len(subsample_data)/audio_rate

		rstr = r"Created subsample from {} - S:{} E:{}, R:{}, D:{}"
		#print(rstr.format(label, step_counter, end_step, audio_rate, duration))

		step_counter += step_size

		#print("playing subsample...")
		#sd.play(subsample_data, audio_rate)
		#sd.wait()

	#print("Playing all subsamples...")
	#for i in subsamples:
	#	sd.play(i['data'], i['rate'])
	#	sd.wait()

	return subsamples



def get_splitted_samples(raw_sample_list, split_by=10):
	splitted_samples = []
	for rsample in raw_sample_list:
		subsamples = split_sample_by(rsample, split_by)
		splitted_samples.extend(subsamples)

	return splitted_samples


def get_shifted_samples(raw_samples):
	shifted_samples = []
	for splitted in raw_samples:
		shifted_group = get_shiftsample_group(splitted)
		shifted_samples.extend(shifted_samples)
	return shifted_samples



SampleSources = get_training_sources('ManualClassify.csv')
AudioCache = load_audio_cache(SampleSources)
RawSamples = get_raw_samples(AudioCache, SampleSources)
SplittedSamples = get_splitted_samples(RawSamples, split_by=30)
ShiftedSamples = get_shifted_samples(SplittedSamples)

print("-"*20,"Raw Samples", "-"*20)
print(*RawSamples,sep='\n')

print("-"*20,"Splitted Samples", "-"*20)
print(*SplittedSamples,sep='\n')

print("-"*20,"Shifted Samples", "-"*20)
print(*ShiftedSamples,sep='\n')

for s in SplittedSamples:
	sd.play(s['data'], s['rate'])
	sd.wait()
print(RawSamples)
print(SplittedSamples)




