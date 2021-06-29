import torch.utils.data
import numpy as np
import cv2
import librosa
import librosa.display
import csv
from scipy import signal

def read_video(video_filename, width=112, height=112,):
  """Read video from file."""
  cap = cv2.VideoCapture(video_filename)
  fps = cap.get(cv2.CAP_PROP_FPS)
  frames = []
  if cap.isOpened():
    while True:
      success, frame_bgr = cap.read()
      if not success:
        break
      frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
      frame_rgb = cv2.resize(frame_rgb, (width, height))
      frames.append(frame_rgb)
  frames = np.asarray(frames)
  return frames, fps


class Audio(torch.utils.data.Dataset):
    def __init__(self, split='train', interval=9, lambda1=0.29):
        self.base_path = '/home/yzhang8/data/datasets/'
        file_path = self.base_path + 'countix_'+split+'_audio.csv'
        class_dict = {'battle rope training':0, 'bouncing ball (not juggling)':1, 'bouncing on trampoline':2, 'clapping':3, 'gymnastics tumbling':4, 'juggling soccer ball':5, 'jumping jacks':6,
                      'mountain climber (exercise)':7, 'planing wood':8, 'playing ping pong':9, 'playing tennis':10, 'running on treadmill': 11, 'sawing wood':12, 'skipping rope':13,
                      'slicing onion':14, 'swimming':15, 'tapping pen':16, 'using a wrench':17, 'using a sledge hammer':18}
        self.split = split
        name_list = []
        count_list = []
        start_list = []
        end_list = []
        start_crop_list = []
        with open(file_path) as f:
            f_csv = csv.reader(f)
            for i, row in enumerate(f_csv):
                if float(row[-2]) == 1:
                    if row[-1] in class_dict.keys() or row[-1].startswith("swimming"):
                        name_list.append(row[0])
                        count_list.append(float(row[5]))
                        start_list.append(float(row[1]))
                        end_list.append(float(row[2]))
                        start_crop_list.append(float(row[3]))


        self.name_list = name_list
        self.count_list = count_list
        self.interval = interval
        self.start_list = start_list
        self.end_list = end_list
        self.start_crop_list = start_crop_list
        self.lambda1 = lambda1


    def __getitem__(self, index):
        video1 = '/home/yzhang8/data/datasets/countix_'+self.split+"_segments/"+self.name_list[index]+".mp4"
        video1,fps = read_video(video1)
        video1 = video1/255.0
        video1 = (video1 - np.array([0.485,0.456,0.406]).reshape((1,1,1,3)))/np.array([0.229,0.224,0.225]).reshape((1,1,1,3))

        start1 = self.start_list[index]- self.start_crop_list[index]
        end1 = self.end_list[index] - self.start_list[index] + start1
        video1 = video1[int(start1):int(end1)]
        avg_period = (end1 - start1) / self.count_list[index]
        sample_rate = int(np.floor((avg_period + 2) / 32.0) + 1)

        pred1 = np.load("results_train/"+self.name_list[index]+".npy")
        gt_pred1 = pred1[sample_rate-1]
        err1 = np.abs(pred1 - gt_pred1)/gt_pred1

        neg_sr_list = []
        for sr in range(1,8):
            if err1[sr-1] > self.lambda1 or sr < sample_rate:
                neg_sr_list.append(sr)
        if len(neg_sr_list)==0:
            for sr in range(1,8):
                if abs(sr-sample_rate)>2:
                    neg_sr_list.append(sr)

        if video1.shape[0] < sample_rate * 64:
            tmp1 = np.zeros((sample_rate * 64 - video1.shape[0], 112, 112, 3))
            video2 = np.concatenate((video1, tmp1), axis=0)
        else:
            video2 = video1.copy()
        start_idx = np.random.choice(video2.shape[0] - sample_rate * 63, (1,))[0]
        video = video2[start_idx + sample_rate * np.arange(64), :, :, :]
        start_ratio1 = start_idx/float(video1.shape[0])
        end_ratio1 = min(1,(start_idx+sample_rate*64.0)/float(video1.shape[0]))
        sample_rate_n = np.random.choice(neg_sr_list, (1,))[0]

        if video1.shape[0]-start_idx < sample_rate_n * 64:
            tmp1 = np.zeros((sample_rate_n * 64 - video1.shape[0]+start_idx, 112, 112, 3))
            video_neg = np.concatenate((video1, tmp1), axis=0)
        else:
            video_neg = video1.copy()
        #start_idx = np.random.choice(video_neg.shape[0] - sample_rate_n * 63, (1,))[0]
        video_neg = video_neg[start_idx + sample_rate_n * np.arange(64), :, :, :]
        start_ratio2 = start_idx/float(video1.shape[0])
        end_ratio2 = min(1,(start_idx+sample_rate_n*64.0)/float(video1.shape[0]))

        video = np.transpose(video, (3,0,1,2))
        video_neg = np.transpose(video_neg, (3,0,1,2))
        video = np.stack((video, video_neg))

        #############################################################################################
        filename = self.base_path + "countix_"+self.split+"_audio/"+self.name_list[index] +".wav"

        y, sr = librosa.load(filename)
        frequencies, times, spectrogram = signal.spectrogram(y, sr, nperseg=512, noverlap=353)
        spectrogram = np.log(spectrogram + 1e-7)

        mean = np.mean(spectrogram)
        std = np.std(spectrogram)
        spectrogram_ori = np.divide(spectrogram - mean, std + 1e-9)

        start1 = int(np.round(start_ratio1*spectrogram_ori.shape[1]))
        end1 = int(np.round(min(spectrogram_ori.shape[1], end_ratio1*spectrogram_ori.shape[1])))
        spectrogram = spectrogram_ori[:, start1:end1]
        if self.split == 'train':
            noise = np.random.uniform(-0.05,0.05, spectrogram.shape)
            spectrogram = spectrogram + noise
        if self.split=='train' and np.random.uniform(0,1,(1,))[0]>0.5:
            spectrogram = spectrogram[:,::-1]
        if spectrogram.shape[1] < 500:
            tmp1 = np.zeros((257, 500-spectrogram.shape[1]))
            spectrogram = np.concatenate((spectrogram, tmp1), axis=1)
        else:
            spectrogram = cv2.resize(spectrogram, (500,257))
        if self.split=='train':
            start1 = np.random.choice(256-self.interval, (1,))[0]
            spectrogram[start1:(start1+self.interval),:] = 0

        start1 = int(np.round(start_ratio2*spectrogram_ori.shape[1]))
        end1 = int(np.round(min(spectrogram_ori.shape[1], end_ratio2*spectrogram_ori.shape[1])))
        spectrogram2 = spectrogram_ori[:, start1:end1]

        if self.split == 'train':
            noise = np.random.uniform(-0.05,0.05, spectrogram2.shape)
            spectrogram2 = spectrogram2 + noise
        if self.split=='train' and np.random.uniform(0,1,(1,))[0]>0.5:
            spectrogram2 = spectrogram2[:,::-1]

        if spectrogram2.shape[1] < 500:
            tmp1 = np.zeros((257, 500-spectrogram2.shape[1]))
            spectrogram2 = np.concatenate((spectrogram2, tmp1), axis=1)
        else:
            spectrogram2 = cv2.resize(spectrogram2, (500,257))
        if self.split=='train':
            start1 = np.random.choice(256-self.interval, (1,))[0]
            spectrogram2[start1:(start1+self.interval),:] = 0

        spectrogram = np.stack((spectrogram, spectrogram2))

        return spectrogram.astype(np.float32), video.astype(np.float32)


    def __len__(self):
        return len(self.name_list)