from Core.Predictor import load_dnn_model, generate_labels_for_sample
from Core import Processing
import argparse

model_path='.\\TrainedNetwork\\dnn_model'

parser = argparse.ArgumentParser(description='Healthy/unhealthy motor classifier')
parser.add_argument('audio-path', type=str, help="Absolute or relative path")
parser.add_argument('-out', type=str, default=None, help="Output path - default is [audio-path].report.txt")
parser.add_argument('-model', type=str, default=None, help="DNN model path")
parser.add_argument('-interval', type=float, default=2, help="Interval for averaging audio parts")

args = vars(parser.parse_args())

audio_path = args['audio-path']
model_path = args['model'] or model_path
output_path = args['out'] or audio_path + ".report.txt"
interval = float(args['interval'])


print("--- Healthy/unhealthy motor classifier ---")
print("Loading model...")
dnn_model = load_dnn_model(model_path)

print("Loading audio sample...")
audio, rate = Processing.load_audio_file(audio_path, 16000)
audio_sample = Processing.create_sample('verify_sample', rate, audio)


print("Saving labels at: ", output_path)
timestamps = generate_labels_for_sample(dnn_model, audio_sample, label_interval=interval)
Processing.save_ac_output(output_path, timestamps)

print("Done.")

