import pathlib
import torch.utils.data
import os
import numpy as np
import cv2
import collections
import skimage.draw
import math
import csv

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

class Countix(torch.utils.data.Dataset):
    def __init__(self,split='train'
                 ):
        """length = None means take all possible"""
        name_list = []
        count_list = []
        start_list = []
        start_crop_list = []
        end_list = []
        class_list = []
        end_crop_list = []
        class_dict = {'battle rope training':0, 'bouncing ball (not juggling)':1, 'bouncing on trampoline':2, 'clapping':3, 'gymnastics tumbling':4, 'juggling soccer ball':5, 'jumping jacks':6,
                      'mountain climber (exercise)':7, 'planing wood':8, 'playing ping pong':9, 'playing tennis':10, 'running on treadmill': 11, 'sawing wood':12, 'skipping rope':13,
                      'slicing onion':14, 'swimming':15, 'tapping pen':16, 'using a wrench':17, 'using a sledge hammer':18,'bench pressing':19, 'bouncing on bouncy castle':20, 'crawling baby':21,
                      'doing aerobics':22, 'exercising arm': 23, 'front raises':24, 'hammer throw':25, 'headbanging':26, 'hula hooping':27, 'lunge':28, 'pirouetting':29, 'playing ukulele':30, 'pull ups':31,
                      'pumping fist':32, 'push up':33, 'rope pushdown':34, 'shaking head':35, 'shoot dance':36, 'situp':37, 'skiing slalom':38, 'spinning poi':39,'squat':40, 'swinging on something':41, 'else': 42
                      }

        #name_list[name_id], start_list[name_id], end_list[name_id], start_crop_list[name_id], end_crop_list[name_id],count_list[name_id]chiseling wood
        neg_list = []
        with open("countix_"+split+"_examples_clean.csv") as f:
            f_csv = csv.reader(f)
            for i, row in enumerate(f_csv):
                name_list.append(row[0])
                if row[-1].startswith("swimming"):
                    class_list.append(15)
                try:
                    class_list.append(class_dict[row[-2]])
                except:
                    class_list.append(class_dict['else'])
                start_list.append(float(row[1]))
                end_list.append(float(row[2]))
                start_crop_list.append(float(row[3]))
                end_crop_list.append(float(row[4]))
                count_list.append(float(row[5]))

        self.name_list = name_list
        self.count_list = count_list
        self.start_list = start_list
        self.end_list = end_list
        self.start_crop_list = start_crop_list
        self.end_crop_list = end_crop_list
        self.split = split
        self.class_list = class_list

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
        #candidata_sample_rates = [sample_rate,]

        #while avg_period / (sample_rate+1)>8 and (sample_rate+1) <= 5:
        #    candidata_sample_rates.append(sample_rate+1)
        #    sample_rate += 1
        #sample_rate = np.random.choice(candidata_sample_rates, (1,))[0]
        if video1.shape[0] < sample_rate*64:
            tmp1 = np.zeros((sample_rate*64-video1.shape[0], 112,112,3))
            video1 = np.concatenate((video1, tmp1), axis=0)
        start_idx = np.random.choice(video1.shape[0]-sample_rate*63,(1,))[0]
        video = video1[start_idx + sample_rate * np.arange(64), :, :, :]

        count = min(sample_rate*64.0/video1.shape[0],1) * self.count_list[index]

        video = np.transpose(video, (3,0,1,2))

        class1 = self.class_list[index]

        return video.astype(np.float32), count, class1

    def __len__(self):
        return len(self.name_list)


def _defaultdict_of_lists():
    return collections.defaultdict(list)
