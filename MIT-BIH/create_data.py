import glob 
import numpy as np 
import wfdb 
import matplotlib.pyplot as plt
import librosa

plt.style.use('ggplot')

DIR = './physionet.org/files/mitdb/1.0.0/'

fp = open('{}{}'.format(DIR, 'RECORDS'), 'r')
data = fp.readlines()
data = [f.strip() for f in data]
fp.close()

toremove = ['106', '114', '203', '213', '221', '222', '228', '102', '104', '107', '217']

T = 1.0
INDEX = 0
SAVE = './Data/D2/'

y = []

for file in data:

	if file in toremove:
		continue

	record = wfdb.rdrecord('{}{}'.format(DIR, file))

	waveform = record.__dict__['p_signal'].T
	fs = record.__dict__['fs']

	annotation = wfdb.rdann('{}{}'.format(DIR, file), 'atr')
	for ann, sym in zip(annotation.__dict__['sample'], annotation.__dict__['symbol']):
		
		for shift in range(-20, 20, 5):
			segment = waveform[:, ann+shift-int(fs*T/2): ann + shift + int(fs*T/2)]

			if segment.shape[-1] != int(T*fs):
				continue

			if sym in ['N', 'L', 'R']:
				filename = '{}s_{}_{}.npy'.format(SAVE, sym, INDEX)

				np.save(filename, segment)
				print(filename, segment.shape)
				INDEX += 1

				y.append(sym)

print(INDEX)
print(np.unique(np.array(y)))
