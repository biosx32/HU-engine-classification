# Supported platforms: windows

Requirements: keras, numpy, sklearn, librosa

* Train network
	- put audio files into Data folder
	- fill out 'metadata.csv' 
		- in the format of '[filename],[time-start],[time-end],[description-label], one file per line
	- run 'python3 Train.py'
	- this will generate dnn model in 'trainedNetwork' folder

* Classify wav file
	- run 'python3 Classify.py [audio-path]'
	- optional arguments are:
		'-out'  => "Output path - default is [audio-filename].report.txt")
		'-model'  =>  "Change DNN model save path"
		'-interval' => "experimental, change audio sampling interval"
