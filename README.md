Requirements: keras, numpy, sklearn, librosa

* Train network
	- put wave files into data folder
	- write out filename, time (start, stop) and label in metadata CSV file, example is in 'metedata.csv'
	- run 'python3 Train.py'

* Classify wav file
	- put .wav file in Data folder or main folder
	- run 'python3 Classify.py [-out path] [-model path] [-interval xy] wav-file-path'
	- if '-out' not provided, it will generate file like '[audio-file].report.txt'

	
# notice: windows-only
