import torch
import torch.nn.functional as F
import numpy as np
import cv2
import random

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

def get_audio_count(spectrogram_ori, a_model, a_tensor, threshold=16, audio_branch_num=43):
    spectrogram2 = None
    if spectrogram_ori.shape[1] < 500:
        tmp1 = np.zeros((257, 500 - spectrogram_ori.shape[1]))
        spectrogram = np.concatenate((spectrogram_ori, tmp1), axis=1)
    else:
        start1 = 0
        spec_list = []
        while start1 < spectrogram_ori.shape[1]:
            aa = spectrogram_ori[:,int(start1):int(start1+500)]
            if aa.shape[1]< 500:
                tmp1 = np.zeros((257, 500 - aa.shape[1]))
                aa = np.concatenate((aa, tmp1), axis=1)
            spec_list.append(aa)
            start1 += 500
        spec_list = np.stack(spec_list)
        spectrogram = spec_list.copy()
        # else:
        spectrogram2 = cv2.resize(spectrogram_ori, (500,257))
    if len(spectrogram.shape) == 2:
        spectrogram = np.expand_dims(spectrogram, axis=0)
        spectrogram = np.expand_dims(spectrogram, axis=0)
    else:
        spectrogram = np.expand_dims(spectrogram, axis=1)
    spectrogram = torch.Tensor(spectrogram).type(torch.FloatTensor)

    tmp, class_out,_ = a_model(spectrogram.cuda())
    outputs11 = []
    for ii in range(spectrogram.size()[0]):
        outputs_tmp = []
        for jj in range(audio_branch_num):
            outputs2 = F.softmax(tmp[ii:(ii + 1), jj * 18:(jj + 1) * 18], dim=1)
            outputs2 = torch.sum(outputs2 * a_tensor, dim=1, keepdim=True)
            outputs_tmp.append(outputs2)
        outputs_tmp = torch.cat(outputs_tmp, dim=1)
        outputs11.append(outputs_tmp)
    outputs11 = torch.cat(outputs11)
    outputs11 = torch.sum(torch.softmax(class_out, dim=1) * outputs11, dim=1)
    outputs11 = torch.sum(outputs11)


    if outputs11.item() < threshold and spectrogram_ori.shape[1]>500:
        spectrogram2 = np.expand_dims(spectrogram2, axis=0)
        spectrogram2 = np.expand_dims(spectrogram2, axis=0)
        spectrogram2 = torch.Tensor(spectrogram2).type(torch.FloatTensor)
        tmp2, class_out2,_ = a_model(spectrogram2.cuda())
        outputs22 = []
        for ii in range(spectrogram.size()[0]):
            outputs_tmp = []
            for jj in range(audio_branch_num):
                outputs2 = F.softmax(tmp2[ii:(ii + 1), jj * 18:(jj + 1) * 18], dim=1)
                outputs2 = torch.sum(outputs2 * a_tensor, dim=1, keepdim=True)
                outputs_tmp.append(outputs2)
            outputs_tmp = torch.cat(outputs_tmp, dim=1)
            outputs22.append(outputs_tmp)
        outputs22 = torch.cat(outputs22)
        outputs22 = torch.sum(torch.softmax(class_out2, dim=1) * outputs22, dim=1)
        outputs11 = torch.sum(outputs22)
    outputs11 = outputs11.item()
    return outputs11

def get_visual_count(video1, sample_rate, model,  tensor, visual_branch_num=41):
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
                tmp1 = np.zeros((new_sr * 64+start_idx - video1.shape[0], 112, 112, 3))
                video11 = np.concatenate((video1, tmp1), axis=0)
                video = video11[start_idx + new_sr * np.arange(64), :, :, :]
                video_list.append(video)

                new_sr = sample_rate+2
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
        video_list = np.stack(video_list)
    if video_list2 is not None and len(video_list2)>0:
        video_list2 = np.stack(video_list2)
    else:
        video_list2=None

    video_list = np.transpose(video_list, (0,4,1,2,3)).astype(np.float32)
    if video_list2 is not None and len(video_list2) > 0:
        video_list2 = np.transpose(video_list2, (0, 4, 1, 2, 3)).astype(np.float32)
        video_list2 = torch.Tensor(video_list2).type(torch.FloatTensor).cuda()

    video_list = torch.Tensor(video_list).type(torch.FloatTensor).cuda()

    with torch.no_grad():
        outputs1, class_out,_,_ = model(video_list)
        outputs = []
        for ii in range(video_list.size()[0]):
            outputs_tmp = []
            for jj in range(visual_branch_num):
                outputs2 = F.softmax(outputs1[ii:(ii + 1), jj * 34:(jj + 1) * 34], dim=1)
                outputs2 = torch.sum(outputs2 * tensor, dim=1, keepdim=True)
                outputs_tmp.append(outputs2)
            outputs_tmp = torch.cat(outputs_tmp, dim=1)
            outputs.append(outputs_tmp)
        outputs = torch.cat(outputs)
        outputs = torch.sum(torch.softmax(class_out, dim=1) * outputs, dim=1)

    if video_list2 is not None:
        with torch.no_grad():
            outputs1, class_out,_ ,_ = model(video_list2)
            outputs22 = []
            for ii in range(video_list2.size()[0]):
                outputs_tmp = []
                for jj in range(visual_branch_num):
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
        if outputs.shape[0] <=2:
            outputs = (np.sum(outputs)+np.sum(outputs22))/2.0
        else:
            last_outputs = outputs[-2:]
            last_outputs22 = outputs22[-1]
            outputs = np.sum(outputs[:-2])+(last_outputs22+np.sum(last_outputs))/2.0
    else:
        outputs = np.sum(outputs)

    return outputs


def get_sr(spectrogram, video1, model, a_model, a_sr_model, v_sr_model):
    video_list = []
    for sample_rate in [1,2,3,4,5]:
        if video1.shape[0] < sample_rate*64:
            tmp1 = np.zeros((sample_rate*64-video1.shape[0], 112,112,3))
            video2 = np.concatenate((video1, tmp1), axis=0)
            video2 = video2[sample_rate * np.arange(64), :, :, :]
        else:
            video2 = video1[sample_rate * np.arange(64), :, :, :]
        video_list.append(video2)
    video_list = np.stack(video_list)
    video_list = np.transpose(video_list, (0,4,1,2,3)).astype(np.float32)

    spectrogram_list = []
    for sample_rate in [1,2,3,4,5]:
        end_ratio1 = (sample_rate * 64.0) / float(video1.shape[0])
        if end_ratio1 > 1:
            tmp1 = np.zeros((257, int(np.round((end_ratio1-1)* spectrogram.shape[1]))))
            spectrogram1 = np.concatenate((spectrogram, tmp1), axis=1)
        else:
            spectrogram1 = spectrogram[:, :int(np.round(end_ratio1*spectrogram.shape[1]))]
        #print(spectrogram1.shape)
        if spectrogram1.shape[1] < 500:
            tmp1 = np.zeros((257, 500 - spectrogram1.shape[1]))
            spectrogram1 = np.concatenate((spectrogram1, tmp1), axis=1)
        else:
            spectrogram1 = cv2.resize(spectrogram1, (500, 257))
        spectrogram_list.append(spectrogram1)
    spectrogram_list = np.stack(spectrogram_list)
    spectrogram_list = torch.Tensor(spectrogram_list).type(torch.FloatTensor).cuda()
    spectrogram_list = spectrogram_list.unsqueeze(1)

    video_list = torch.Tensor(video_list).type(torch.FloatTensor).cuda()
    with torch.no_grad():
        _, _, audio_feat = a_model(spectrogram_list)
        _, class_out, feat,_ = model(video_list)

        audio_feat = a_sr_model(audio_feat.detach())
        outputs1 = v_sr_model(feat.detach(), audio_feat.flatten(1))
        class_out = torch.softmax(class_out.detach(), dim=1)
        scores = torch.sum(outputs1 * class_out, dim=1)


    scores = scores.detach().view(-1).cpu().numpy()
    sample_rate = np.argmax(scores)+1
    return sample_rate

def get_conf(spectrogram_ori, audio_pred, sample_rate, video1, audio_model, visual_model, conf_model ):
    spectrogram_list = []
    if audio_pred > 11 and spectrogram_ori.shape[1] > 500:
        start_idx = 0
        while start_idx + 500 < spectrogram_ori.shape[1] or len(spectrogram_list) == 0:
            spectrogram = spectrogram_ori[:, start_idx:(start_idx + 500)]
            spectrogram_list.append(spectrogram)
            start_idx += 500
        spectrogram_list = np.stack(spectrogram_list)


    elif spectrogram_ori.shape[1] > 500:
        spectrogram = cv2.resize(spectrogram_ori, (500, 257))
        spectrogram_list.append(spectrogram)
        spectrogram_list = np.stack(spectrogram_list)

    elif spectrogram_ori.shape[1] <= 500:
        tmp1 = np.zeros((257, 500 - spectrogram_ori.shape[1]))
        spectrogram = np.concatenate((spectrogram_ori, tmp1), axis=1)
        spectrogram_list.append(spectrogram)
        spectrogram_list = np.stack(spectrogram_list)

    video_list = []
    start_idx = 0
    while start_idx + sample_rate*64 < video1.shape[0] or len(video_list)==0:
        if video1.shape[0] < sample_rate*64:
            tmp1 = np.zeros((sample_rate*64-video1.shape[0], 112,112,3))
            video2 = np.concatenate((video1, tmp1), axis=0)
            video2 = video2[start_idx+sample_rate * np.arange(64), :, :, :]
        else:
            video2 = video1[start_idx+sample_rate * np.arange(64), :, :, :]
        start_idx += sample_rate*64

        video = np.transpose(video2, (3, 0, 1, 2))
        video_list.append(video)
    video_list = np.stack(video_list)

    if spectrogram_list.shape[0] < video_list.shape[0]:
        idxes = np.random.choice(spectrogram_list.shape[0], (video_list.shape[0]-spectrogram_list.shape[0],))
        tmp = spectrogram_list[idxes,:,:]
        spectrogram_list = np.concatenate((spectrogram_list, tmp), axis=0)
    elif spectrogram_list.shape[0] > video_list.shape[0]:
        idxes = np.random.choice(video_list.shape[0], (spectrogram_list.shape[0]-video_list.shape[0],))
        tmp = video_list[idxes,:,:]
        video_list = np.concatenate((video_list, tmp), axis=0)

    idxes = np.arange(video_list.shape[0])
    random.shuffle(idxes)
    video_list = video_list[idxes]
    spectrogram_list = spectrogram_list[idxes]

    spectrogram = torch.Tensor(spectrogram_list).type(torch.FloatTensor).cuda().unsqueeze(1)
    video_list = torch.Tensor(video_list).type(torch.FloatTensor).cuda()

    with torch.no_grad():
        _, _, audio_feat = audio_model(spectrogram)
        _, _, _,visual_feat = visual_model(video_list)
        outputs1 = conf_model(audio_feat.detach(), visual_feat.detach())
        conf = outputs1[:,0:1]
        conf = torch.sigmoid(conf)
        conf = torch.mean(conf)
    conf = conf.detach().item()
    return conf