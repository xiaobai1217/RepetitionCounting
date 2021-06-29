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

        #name_list[name_id], start_list[name_id], end_list[name_id], start_crop_list[name_id], end_crop_list[name_id],count_list[name_id]chiseling wood
        sample_rates = []
        neg_list = []
        with open("countix_"+split+"_sr.csv") as f:
            f_csv = csv.reader(f)
            for i, row in enumerate(f_csv):
                name_list.append(row[0])
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
            if err1[sr-1] > 0.29 or sr < sample_rate:
                neg_sr_list.append(sr)

        if video1.shape[0] < sample_rate * 64:
            tmp1 = np.zeros((sample_rate * 64 - video1.shape[0], 112, 112, 3))
            video2 = np.concatenate((video1, tmp1), axis=0)
        else:
            video2 = video1.copy()
        start_idx = np.random.choice(video2.shape[0] - sample_rate * 63, (1,))[0]
        video = video2[start_idx + sample_rate * np.arange(64), :, :, :]

        sample_rate_n = np.random.choice(neg_sr_list, (1,))[0]
        if video1.shape[0]-start_idx < sample_rate_n * 64:
            tmp1 = np.zeros((sample_rate_n * 64 - video1.shape[0]+start_idx, 112, 112, 3))
            video_neg = np.concatenate((video1, tmp1), axis=0)
        else:
            video_neg = video1.copy()
        #start_idx = np.random.choice(video_neg.shape[0] - sample_rate_n * 63, (1,))[0]
        video_neg = video_neg[start_idx + sample_rate_n * np.arange(64), :, :, :]



        video = np.transpose(video, (3,0,1,2))
        video_neg = np.transpose(video_neg, (3,0,1,2))
        video = np.stack((video, video_neg))

        return video.astype(np.float32)

    def __len__(self):
        return len(self.name_list)


def _defaultdict_of_lists():
    return collections.defaultdict(list)
