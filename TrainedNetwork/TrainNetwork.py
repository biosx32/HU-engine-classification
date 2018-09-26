import pandas as pd
import librosa
import matplotlib
import argparse

parser = argparse.ArgumentParser(
	description='Train motor')
parser.add_argument('-f', '--file', type=str)
parser.add_argument('-o', '--output', type=str, default='NN_trained.nn')
parser.add_argument('-r', '--rate', type=int, default=44100)
args = parser.parse_args()

