import torch.utils.data
import numpy as np
import cv2
import librosa
import librosa.display
import csv
from scipy import signal


class Audio(torch.utils.data.Dataset):
    def __init__(self, split='train', interval=9):
        self.base_path = '/home/yzhang8/data/datasets/'
        file_path = self.base_path + 'countix_'+split+'_audio.csv'
        class_dict = {'battle rope training':0, 'bouncing ball (not juggling)':1, 'bouncing on trampoline':2, 'clapping':3, 'gymnastics tumbling':4, 'juggling soccer ball':5, 'jumping jacks':6,
                      'mountain climber (exercise)':7, 'planing wood':8, 'playing ping pong':9, 'playing tennis':10, 'running on treadmill': 11, 'sawing wood':12, 'skipping rope':13,
                      'slicing onion':14, 'swimming':15, 'tapping pen':16, 'using a wrench':17, 'using a sledge hammer':18}
        self.split = split
        name_list = []
        count_list = []

        with open(file_path) as f:
            f_csv = csv.reader(f)
            for i, row in enumerate(f_csv):
                if float(row[-2]) == 1:
                    if row[-1] in class_dict.keys() or row[-1].startswith("swimming"):
                        name_list.append(row[0])
                        count_list.append(float(row[5]))


        self.name_list = name_list
        self.count_list = count_list
        self.interval = interval

    def __getitem__(self, index):
        filename = self.base_path + "countix_"+self.split+"_audio/"+self.name_list[index] +".wav"

        y, sr = librosa.load(filename)
        frequencies, times, spectrogram = signal.spectrogram(y, sr, nperseg=512, noverlap=353)
        spectrogram = np.log(spectrogram + 1e-7)

        mean = np.mean(spectrogram)
        std = np.std(spectrogram)
        spectrogram = np.divide(spectrogram - mean, std + 1e-9)
        if self.split == 'train':
            noise = np.random.uniform(-0.05,0.05, spectrogram.shape)
            spectrogram = spectrogram + noise
        if self.split=='train' and np.random.uniform(0,1,(1,))[0]>0.5:
            spectrogram = spectrogram[:,::-1]

        count = float(self.count_list[index])
        if self.split=='train' and spectrogram.shape[1]<250 and np.random.uniform(0,1,(1,))[0]>0.7:
            length1 = spectrogram.shape[1]
            length1 = int(np.random.uniform(0.5,1, (1,))[0]*length1)
            count2 = length1 / float(spectrogram.shape[1]) * count
            start1 = np.random.choice(spectrogram.shape[1]-length1, (1,))[0]
            spectrogram2 = spectrogram[:, start1:(start1+length1)]
            if np.random.uniform(0,1,(1,))[0]>0.5:
                length2 = int(np.random.uniform(0.8,1.2, (1,))[0]*spectrogram.shape[1])
                spectrogram = cv2.resize(spectrogram, (length2, 257))
            if np.random.uniform(0,1,(1,))[0]>0.5:
                spectrogram = np.concatenate((spectrogram, spectrogram2), axis=1)
            else:
                spectrogram = np.concatenate((spectrogram2, spectrogram), axis=1)
            count = count+count2


        if spectrogram.shape[1] < 500:
            tmp1 = np.zeros((257, 500-spectrogram.shape[1]))
            spectrogram = np.concatenate((spectrogram, tmp1), axis=1)
        else:
            if count>8:
                length1 = int(np.random.uniform(0.8,1.2, (1,))[0]*500)
                length1 = min(length1, spectrogram.shape[1])
                if length1 == spectrogram.shape[1]:
                    start1=0
                else:
                    start1 = np.random.choice(spectrogram.shape[1] - length1, (1,))[0]
                count = length1 / float(spectrogram.shape[1]) * count
                spectrogram = spectrogram[:, start1:(start1 + length1)]
                spectrogram = cv2.resize(spectrogram, (500,257))
            else:

                spectrogram = cv2.resize(spectrogram, (500,257))


        if self.split=='train':
            start1 = np.random.choice(256-self.interval, (1,))[0]
            spectrogram[start1:(start1+self.interval),:] = 0

        return spectrogram.astype(np.float32), count


    def __len__(self):
        return len(self.name_list)