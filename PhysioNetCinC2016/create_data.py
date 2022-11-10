import numpy as np 
from scipy.io import wavfile

'''
WL = 4.0
WS = 0.1

a = 117091
b = 19510
c = 14101
d = 6159
e = 409373
f = 33249

'''

DB = 'e'
DIR = './physionet.org/files/challenge-2016/1.0.0/training-{}/'.format(DB)
SR = 2000

WL = 1.0
label_to_ws = {0:0.4, 1:0.03}

SAVE = './Data/D6/'.format(DB)

data = []
y = []

fp = open('{}RECORDS-normal'.format(DIR))
data += fp.readlines()
y += [0]*len(data)

fp.close()
fp = open('{}RECORDS-abnormal'.format(DIR))
data += fp.readlines()
y += [1]*(len(data) - len(y))

fp.close()

R_INDEX = 1

Y = []

for y_, file in zip(y, data):
    samplerate, data = wavfile.read('{}{}.wav'.format(DIR, file.strip()))

    for index in range(0, data.shape[-1], int(label_to_ws[y_]*SR)):
        if index + int(WL*samplerate) < data.shape[-1]:
            sample = data[index:index + int(WL*SR)]
            
            filename = '{}s_{}_{}.npy'.format(SAVE, y_, R_INDEX)
            np.save(filename, sample)
            R_INDEX += 1

            print('saved:', filename, sample.shape)

            Y.append(y_)

Y = np.array(Y)

print(np.where(Y == 0)[0].shape, np.where(Y == 1)[0].shape)
print(R_INDEX)
