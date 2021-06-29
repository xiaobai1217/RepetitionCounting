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
    def __init__(self, split='train', interval=9, audio_threshold = 0.42, visual_threshold = 0.365, magnitude=0.17, threshold=0.2):
        self.base_path = '/home/yzhang8/data/datasets/'
        file_path = self.base_path + 'countix_'+split+'_audio.csv'
        class_dict = {'battle rope training':0, 'bouncing ball (not juggling)':1, 'bouncing on trampoline':2, 'clapping':3, 'gymnastics tumbling':4, 'juggling soccer ball':5, 'jumping jacks':6,
                      'mountain climber (exercise)':7, 'planing wood':8, 'playing ping pong':9, 'playing tennis':10, 'running on treadmill': 11, 'sawing wood':12, 'skipping rope':13,
                      'slicing onion':14, 'swimming':15, 'tapping pen':16, 'using a wrench':17, 'using a sledge hammer':18}
        self.split = split
        name_list = []
        count_list = []
        start_list = []
        start_crop_list = []
        end_list = []

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
        self.audio_records = np.load("train4confrecords_audio.npy")
        self.visual_records = np.load("train4confrecords.npy")
        self.audio_threshold = audio_threshold
        self.visual_threshold = visual_threshold
        visual_id_list = []
        for ii in range(self.visual_records.shape[0]):
            if self.visual_records[ii,-1] < self.visual_threshold:
                visual_id_list.append([self.visual_records[ii,0], self.visual_records[ii,1]])
        self.visual_id_list = np.array(visual_id_list).astype(np.int8)

        audio_id_list = []
        for ii in range(self.audio_records.shape[0]):
            if self.audio_records[ii,-1] < self.audio_threshold:
                audio_id_list.append(self.audio_records[ii,0])
        self.audio_id_list = np.array(audio_id_list).astype(np.int8)
        #print("d")
        self.magnitude = magnitude
        self.threshold = threshold


    def __getitem__(self, index):
        video1 = '/home/yzhang8/data/datasets/countix_'+self.split+"_segments/"+self.name_list[index]+".mp4"
        video1,fps = read_video(video1)
        video1 = video1/255.0
        video1 = (video1 - np.array([0.485,0.456,0.406]).reshape((1,1,1,3)))/np.array([0.229,0.224,0.225]).reshape((1,1,1,3))

        start1 = self.start_list[index]- self.start_crop_list[index]
        end1 = self.end_list[index] - self.start_list[index] + start1
        video1 = video1[int(start1):int(end1)]
        avg_period = (end1 - start1) / self.count_list[index]
        #sample_rate = int(np.floor((avg_period + 2) / 32.0) + 1)
        sample_rate = np.random.choice(5, (1,))[0]+1

        if video1.shape[0] < sample_rate*64:
            tmp1 = np.zeros((sample_rate*64-video1.shape[0], 112,112,3))
            video1 = np.concatenate((video1, tmp1), axis=0)
        start_idx = np.random.choice(video1.shape[0]-sample_rate*63,(1,))[0]
        video = video1[start_idx + sample_rate * np.arange(64), :, :, :]

        video = np.transpose(video, (3,0,1,2))

        ############################################################################################
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
                spectrogram = spectrogram[:, start1:(start1 + length1)]
                spectrogram = cv2.resize(spectrogram, (500,257))
            else:

                spectrogram = cv2.resize(spectrogram, (500,257))

        if self.split=='train':
            start1 = np.random.choice(256-self.interval, (1,))[0]
            spectrogram[start1:(start1+self.interval),:] = 0

        ############################################################################################

        audio_preds = []
        for ii in self.audio_id_list:
            audio_pred = np.load("audio_train_results%02d/"%ii+self.name_list[index]+".npy").astype(np.float32)
            audio_preds.append(audio_pred[0])

        if self.split=='train':
            visual_preds = []
            for ii in range(self.visual_id_list.shape[0]):
                visual_pred = np.load("results4trainconf/results%02d_%02d/"%(self.visual_id_list[ii,0], self.visual_id_list[ii,1])+self.name_list[index]+".npy").astype(np.float32)
                visual_preds.append(np.sum(visual_pred))
        else:
            visual_preds = np.sum(np.load("results_val/"+self.name_list[index]+".npy")).astype(np.float32)
        audio_pred = np.mean(audio_preds)
        visual_pred = np.mean(visual_preds)

        audio_err = abs(audio_pred - count) / count
        visual_err = abs(visual_pred - visual_pred) / count
        if self.split == 'train':
            if audio_err > visual_err and (audio_err - visual_err) > self.threshold:
                err_direction = np.sign(audio_pred - count)
                noise = err_direction * count*self.magnitude
                audio_pred = audio_pred + noise
            elif visual_err > audio_err and (visual_err - audio_err) > self.threshold:
                err_direction = np.sign(visual_pred - count)
                noise = self.magnitude*err_direction * count
                visual_pred = visual_pred + noise

        return spectrogram.astype(np.float32), video.astype(np.float32), float(audio_pred), float(visual_pred), count


    def __len__(self):
        return len(self.name_list)