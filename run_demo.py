import torch
import numpy as np
from sync_batchnorm import convert_model
import models
import VGGSound
from VGGSound.test import args_dict
from VGGSound.model import AVENet
import torch.nn as nn
import librosa
from scipy import signal
from util import get_sr, get_audio_count, get_conf, get_visual_count, read_video


np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.enabled = False  # 0.811

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#-----------------------------------------------load models trained on Countix-AV-------------------------------------------
model = models.video.r2plus1d_18(pretrained=True)
model.fc = torch.nn.Linear(512,34*41)
model.fc2 = torch.nn.Linear(512,41)
model = convert_model(model)

args = args_dict()
audio_model= AVENet(args)
audio_model.audnet.fc = torch.nn.Linear(512, 18*43)
audio_model.audnet.fc2 = torch.nn.Linear(512,43)

checkpoint = torch.load("audio_checkpoint.pt")
audio_model.load_state_dict(checkpoint['state_dict'])

audio_model = audio_model.cuda()
audio_model.eval()
tensor = torch.Tensor(np.arange(2,20).astype(np.float32)).cuda()

a_sr_model = VGGSound.models.resnet.SubNet()
a_sr_model = a_sr_model.cuda()
v_sr_model = models.video.resnet.SubNet()
v_sr_model.fc = nn.Sequential(torch.nn.Linear(1024,41))
v_sr_model = convert_model(v_sr_model)

conf_model = VGGSound.models.resnet.ConfSubNet(lambda1=14)
conf_model.fc = nn.Sequential(nn.Dropout(0.27),torch.nn.Linear(1024, 2),)
checkpoint = torch.load("conf_checkpoint.pt")
conf_model.load_state_dict(checkpoint['v_state_dict'])
conf_model = conf_model.cuda()
conf_model.eval()

if device.type == "cuda":
    model = torch.nn.DataParallel(model)
    v_sr_model = torch.nn.DataParallel(v_sr_model)

model = model.cuda()
checkpoint = torch.load("visual_checkpoint.pt")
model.load_state_dict(checkpoint['state_dict'])
model.eval()

checkpoint = torch.load("sr_checkpoint.pt")
v_sr_model.load_state_dict(checkpoint['sr_state_dict'])
v_sr_model.eval()
a_sr_model.load_state_dict(checkpoint['a_sr_state_dict'])
a_sr_model.eval()

tensor = torch.Tensor(np.arange(2,36)).type(torch.FloatTensor).cuda().unsqueeze(0)
audio_tensor = torch.Tensor(np.arange(2,20).astype(np.float32)).cuda()

video1 = "eEG7ZOQG6LM.00.mp4"
video1,fps = read_video(video1)
video1 = video1/255.0
video1 = (video1 - np.array([0.485,0.456,0.406]).reshape((1,1,1,3)))/np.array([0.229,0.224,0.225]).reshape((1,1,1,3))
# when we generated the video segment, we preserved some background context, so we need to do this line to remove the background
video1 = video1[10:265]
#--------------------------------------------------------------------------------------------
filename = "eEG7ZOQG6LM.00.wav"
y, sr = librosa.load(filename)

frequencies, times, spectrogram = signal.spectrogram(y, sr, nperseg=512, noverlap=353)
spectrogram = np.log(spectrogram + 1e-7)
mean = np.mean(spectrogram)
std = np.std(spectrogram)
spectrogram_ori = np.divide(spectrogram - mean, std + 1e-9)
#--------------------------------------------------------------------------------------------
sample_rate = get_sr(spectrogram_ori, video1, model, audio_model, a_sr_model, v_sr_model)
outputs11 = get_audio_count(spectrogram_ori, audio_model,audio_tensor)
outputs = get_visual_count(video1, sample_rate, model)
conf = get_conf(spectrogram_ori, outputs11, sample_rate, video1, audio_model, model, conf_model)
outputs = conf*outputs + (1-conf)*outputs11
outputs = np.round(outputs)

groundtruth=8
mae_err = abs(outputs-groundtruth) / groundtruth
if abs(outputs-groundtruth) <= 1:
    OBO = 1
else:
    OBO = 0

print(mae_err, OBO)


