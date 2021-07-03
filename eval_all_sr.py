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


np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.enabled = False  # 0.811

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.video.r2plus1d_18(pretrained=True)
model.fc = torch.nn.Linear(512,34*43)
model.fc2 = torch.nn.Linear(512,43)
model = convert_model(model)


if device.type == "cuda":
    model = torch.nn.DataParallel(model)
    #model_sr = torch.nn.DataParallel(model_sr)

model = model.cuda()
#model_sr = model_sr.cuda()

checkpoint = torch.load("best.pt")
model.load_state_dict(checkpoint['state_dict'])
#checkpoint = torch.load("best_sr.pt")
#model_sr.load_state_dict(checkpoint['state_dict'])

name_list = []
count_list = []
start_list = []
start_crop_list = []
end_list = []
class_list = []
end_crop_list = []


# name_list[name_id], start_list[name_id], end_list[name_id], start_crop_list[name_id], end_crop_list[name_id],count_list[name_id]chiseling wood
with open("countix_val_examples_clean.csv") as f:
    f_csv = csv.reader(f)
    for i, row in enumerate(f_csv):
        name_list.append(row[0])
        start_list.append(float(row[1]))
        end_list.append(float(row[2]))
        start_crop_list.append(float(row[3]))
        end_crop_list.append(float(row[4]))
        count_list.append(float(row[5]))

tensor = torch.Tensor(np.arange(2,36)).type(torch.FloatTensor).cuda().unsqueeze(0)

save_dir = 'results_train/'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
model.eval()
for name_id, name1 in enumerate(name_list):
    video1 = "/home/yzhang8/data/datasets/countix_val_segments/" + name1 + ".mp4"
    video1, fps = read_video(video1)
    video1 = video1 / 255.0
    video1 = (video1 - np.array([0.485, 0.456, 0.406]).reshape((1, 1, 1, 3))) / np.array(
        [0.229, 0.224, 0.225]).reshape((1, 1, 1, 3))
    start1 = start_list[name_id] - start_crop_list[name_id]
    end1 = end_list[name_id] - start_list[name_id] + start1
    avg_period = (end1 - start1) / count_list[name_id]
    results_list = []
    for sample_rate in range(1,8):
        video1 = "/home/yzhang8/data/datasets/countix_val_segments/" + name1 + ".mp4"
        video1, fps = read_video(video1)
        video1 = video1 / 255.0
        video1 = (video1 - np.array([0.485, 0.456, 0.406]).reshape((1, 1, 1, 3))) / np.array(
            [0.229, 0.224, 0.225]).reshape((1, 1, 1, 3))
        start1 = start_list[name_id] - start_crop_list[name_id]
        end1 = end_list[name_id] - start_list[name_id] + start1
        video1 = video1[int(start1):int(end1)]

        #sample_rate = int(np.floor((avg_period + 2) / 32.0) + 1)
        #sample_rate = int(np.load("sample_rates/"+name1+".npy")[0])
        video_list2 = None
        if video1.shape[0] < sample_rate*64:
            tmp1 = np.zeros((sample_rate*64-video1.shape[0], 112,112,3))
            video_list = np.concatenate((video1, tmp1), axis=0)
            video_list = np.expand_dims(video_list, axis=0)
        else:
            start_idx = 0
            video_list = []
            video_list2 = []
            while start_idx<video1.shape[0]:
                if video1.shape[0] < start_idx + sample_rate*64:
                    new_sr = sample_rate
                    # if avg_period / (sample_rate+2)>8:
                    #     new_sr = sample_rate+2
                    #     video_list = video_list[:-1]
                    #     start_idx = max(0, start_idx - sample_rate * 64)
                    # else:
                    #     new_sr = sample_rate

                    tmp1 = np.zeros((new_sr * 64+start_idx - video1.shape[0], 112, 112, 3))
                    video11 = np.concatenate((video1, tmp1), axis=0)
                    video = video11[start_idx + new_sr * np.arange(64), :, :, :]
                    video_list.append(video)

                    new_sr = sample_rate+2
                    video_list2 = video_list2[:-1]
                    start_idx = max(0, start_idx - sample_rate * 64)

                    tmp1 = np.zeros((new_sr * 64+start_idx - video1.shape[0], 112, 112, 3))
                    video11 = np.concatenate((video1, tmp1), axis=0)
                    video = video11[start_idx + new_sr * np.arange(64), :, :, :]
                    video_list2.append(video)
                    break


                else:
                    video = video1[start_idx + sample_rate * np.arange(64), :, :, :]
                    start_idx += sample_rate * 64
                    video_list.append(video)
                    video_list2.append(video)

                #     tmp1 = np.zeros((sample_rate * 64+start_idx - video1.shape[0], 112, 112, 3))
                #     video1 = np.concatenate((video1, tmp1), axis=0)

                #start_idx += sample_rate*64
                #video_list.append(video)
            video_list = np.stack(video_list)
            video_list2 = np.stack(video_list2)
        #video = (video - np.array([0.485,0.456,0.406]).reshape((1,1,1,3)))/np.array([0.229,0.224,0.225]).reshape((1,1,1,3))

        video_list = np.transpose(video_list, (0,4,1,2,3)).astype(np.float32)
        if video_list2 is not None:
            video_list2 = np.transpose(video_list2, (0, 4, 1, 2, 3)).astype(np.float32)
            video_list2 = torch.Tensor(video_list2).type(torch.FloatTensor).cuda()

        video_list = torch.Tensor(video_list).type(torch.FloatTensor).cuda()

        with torch.no_grad():
            outputs1, class_out = model(video_list)
            outputs = []
            for ii in range(video_list.size()[0]):
                outputs_tmp = []
                for jj in range(43):
                    outputs2 = F.softmax(outputs1[ii:(ii + 1), jj * 34:(jj + 1) * 34], dim=1)
                    outputs2 = torch.sum(outputs2 * tensor, dim=1, keepdim=True)
                    outputs_tmp.append(outputs2)
                outputs_tmp = torch.cat(outputs_tmp, dim=1)
                outputs.append(outputs_tmp)
            outputs = torch.cat(outputs)
            outputs = torch.sum(torch.softmax(class_out, dim=1) * outputs, dim=1)

        if video_list2 is not None:
            with torch.no_grad():
                outputs1, class_out = model(video_list2)
                outputs22 = []
                for ii in range(video_list2.size()[0]):
                    outputs_tmp = []
                    for jj in range(43):
                        outputs2 = F.softmax(outputs1[ii:(ii + 1), jj * 34:(jj + 1) * 34], dim=1)
                        outputs2 = torch.sum(outputs2 * tensor, dim=1, keepdim=True)
                        outputs_tmp.append(outputs2)
                    outputs_tmp = torch.cat(outputs_tmp, dim=1)
                    outputs22.append(outputs_tmp)
                outputs22 = torch.cat(outputs22)
                outputs22 = torch.sum(torch.softmax(class_out, dim=1) * outputs22, dim=1)
                outputs22 = outputs22.detach().cpu().numpy()

        outputs = outputs.detach().cpu().numpy()
        if video_list2 is not None:
            if outputs22.shape[0] == 1:
                outputs = (np.sum(outputs)+np.sum(outputs22))/2.0
            else:
                last_outputs = outputs[-2:]
                last_outputs22 = outputs22[-1]
                outputs = np.sum(outputs[:-2])+(last_outputs22+np.sum(last_outputs))/2.0
        else:
            outputs = np.sum(outputs)
        results_list.append(outputs)
    print(name_id, results_list, count_list[name_id],avg_period)


    np.save(save_dir+name1+".npy", np.array(results_list))
