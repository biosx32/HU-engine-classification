import sys
def ProcessArgs():
	path = sys.argv[1]
	valid = ValidateAudioFile(path)
	if not valid:
		print(MSG_INCORRECT_AUDIO)
		sys.exit(-1)

	AudioFile = GetAudioFile(path)


def WavToText(audio_file):
	pass



if __name__ == "__main__":
	pass
