import wfdb
import matplotlib.pyplot as plt 
import numpy as np 
from scipy.signal import resample

# point to the downloaded folder
fp = open('./data/RECORDS', 'r')
recordings = fp.readlines()
fp.close()
recordings = [f.strip() for f in recordings]

DIR = './data/'

# define the save directory
SAVE = '/media/PPGECG/Data4/'

WL = 2.0
WS = 0.05
INXEX = 0

subject_to_recording = {}

# load data
for file in recordings:
	record = wfdb.rdrecord('{}{}'.format(DIR, file))
	
	signal = record.p_signal
	signal = signal.T  
	fs = record.fs

	for index in range(0, signal.shape[-1], int(WS*fs)):
		if index + int(WL*fs) < signal.shape[-1]:
			segment = signal[:2, index:index + int(WL*fs)]
			
			ecg, ppg = segment[0, ], segment[1, ]
			ecg, ppg = resample(ecg, int(WL*1000)), resample(ppg, int(WL*1000))
			segment = np.concatenate([np.expand_dims(ecg, axis=0), np.expand_dims(ppg, axis=0)], axis=0)

			# filename
			filename = '{}s_{}.npy'.format(SAVE, INXEX)
			np.save(filename, segment)
			print(filename, segment.shape)

			# record
			if file in subject_to_recording:
				subject_to_recording[file].append(filename)
			else:
				subject_to_recording[file] = [filename]

			INXEX += 1

# subject to recording is used for subject-independent evaluations. 
import pickle
with open('{}subject_to_recording.pickle'.format(SAVE), 'wb') as handle:
    pickle.dump(subject_to_recording, handle, protocol=pickle.HIGHEST_PROTOCOL)

