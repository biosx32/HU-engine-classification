import csv
import librosa
import sys
import IPython.display
import librosa.display
import sounddevice as sd

def _dbg_print(*args, **kwargs):
	if 1:
		print("dbg: ",*args,**kwargs)


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
	samples = []
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

		#cutout_data = librosa.effects.remix(audio_data, (start_sample,end_sample))
		cutout_data = audio_data[start_sample:end_sample]

		sample = {
			'label': sample_src['label'],
			'rate': audio_samplerate,
			'data': cutout_data
		}
		print(sample)

		_dbg_print("Playing sample: ", sample)
		sd.play(cutout_data, audio_samplerate*4)
		sd.wait()


def GenerateTrainingData():
	#split every sample to one second and generate 10 pitches for every sample
	pass



SampleSources = get_training_sources('ManualClassify.csv')
AudioCache = load_audio_cache(SampleSources)
Samples = get_raw_samples(AudioCache, SampleSources)




