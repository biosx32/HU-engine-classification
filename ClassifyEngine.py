import pandas as pd
import librosa
import matplotlib
import argparse

parser = argparse.ArgumentParser(
	description='Healthy/unhealthy motor classifier')
parser.add_argument('-a', '--audio-file', type=str)
parser.add_argument('-o', '--output', type=str, default=None)
args = parser.parse_args()

parser.print_help()


print(args)