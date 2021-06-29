import torch
import torchvision
import torch.nn.functional as F
from dataloader2 import Countix
import numpy as np
from sync_batchnorm import convert_model
import tqdm
import models
import cv2
import csv
import os

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


name_list = []
count_list = []
start_list = []
start_crop_list = []
end_list = []
class_list = []
end_crop_list = []


# name_list[name_id], start_list[name_id], end_list[name_id], start_crop_list[name_id], end_crop_list[name_id],count_list[name_id]chiseling wood
with open("countix_train_examples_clean.csv") as f:
    f_csv = csv.reader(f)
    for i, row in enumerate(f_csv):
        name_list.append(row[0])
        start_list.append(float(row[1]))
        end_list.append(float(row[2]))
        start_crop_list.append(float(row[3]))
        end_crop_list.append(float(row[4]))
        count_list.append(float(row[5]))

with open("countix_train_sr.csv", "a") as f:
    for name_id, name1 in enumerate(name_list):
        pred1 = np.load("results_train/"+name1+".npy")
        start1 = start_list[name_id] - start_crop_list[name_id]
        end1 = end_list[name_id] - start_list[name_id] + start1
        avg_period = (end1 - start1) / count_list[name_id]
        count_gt = count_list[name_id]
        sr_gt = int(np.floor((avg_period + 2) / 32.0) + 1)

        gt_pred1 = pred1[sr_gt-1]

        err1 = np.abs(pred1 - gt_pred1)/gt_pred1
        err2 = np.abs(pred1 - count_gt)/count_gt
        neg_sr_list = []

        for sr in range(1,8):
            if err1[sr-1] > 0.3:
                neg_sr_list.append(sr)

        #print(name_id, pred1, sr_gt, neg_sr_list)
        if len(neg_sr_list)>0:
            f.write( "{},{},{},{},{},{}\n".format(name_list[name_id], start_list[name_id], end_list[name_id], start_crop_list[name_id], end_crop_list[name_id],count_list[name_id]))
            f.flush()
