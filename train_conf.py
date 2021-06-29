import torch
import tqdm
import numpy as np
import os
import torch.nn.functional as F
from data_loader_conf import Audio
import torch.nn as nn

from VGGSound.model import AVENet
from VGGSound.test import args_dict
from VGGSound.models.resnet import SubNet
import VGGSound
import argparse
import models
from sync_batchnorm import convert_model

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('-dropout', type=float, help='input a int')



args = parser.parse_args()
dropout_rate = args.dropout #

save_id = int(np.round(dropout_rate*100))
print(save_id)

audio_threshold = 0.40
visual_threshold = 0.36


np.random.seed(0)
torch.manual_seed(0)

torch.backends.cudnn.deterministic = True #0.821
torch.backends.cudnn.benchmark = False
batch_size = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = Audio(split="train", audio_threshold=audio_threshold, visual_threshold=visual_threshold, threshold=0.2)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=5, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=False)
val_dataset = Audio(split="val", audio_threshold=audio_threshold, visual_threshold=visual_threshold)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=5, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=False)

args = args_dict()
model= AVENet(args)
model.audnet.fc = torch.nn.Linear(512, 18*43)
model.audnet.fc2 = torch.nn.Linear(512,43)
checkpoint = torch.load("best_a.pt")
model.load_state_dict(checkpoint['state_dict'])
model = model.cuda()

a_sub_model = SubNet(lambda1=14)
a_sub_model.load_state_dict(checkpoint['state_dict'], strict=False)
a_sub_model.fc = nn.Sequential(nn.Dropout(0.27),
                               torch.nn.Linear(1024, 2),)
a_sub_model = a_sub_model.cuda()

visual_model = models.video.r2plus1d_18(pretrained=True)
visual_model.fc = torch.nn.Linear(512,34*41)
visual_model.fc2 = torch.nn.Linear(512,41)

visual_model = convert_model(visual_model)
if device.type == "cuda":
    visual_model = torch.nn.DataParallel(visual_model)
visual_model = visual_model.cuda()
checkpoint = torch.load("best41.pt")
visual_model.load_state_dict(checkpoint['state_dict'])

criterion = nn.MSELoss()
tensor = torch.Tensor(np.arange(2,20).astype(np.float32)).cuda()
#optim = torch.optim.SGD(model.parameters(),lr=1e-4, momentum=0.9, weight_decay=1e-4)
optim = torch.optim.SGD(list(a_sub_model.parameters()), lr=1e-3, momentum=0.9, weight_decay=1e-4)
dataloaders = {'train':train_dataloader, 'val':val_dataloader}
save_path = 'audio_output/'
if not os.path.exists(save_path):
    os.mkdir(save_path)
model.eval()
BestLoss = float("inf")
with open(save_path+ "log.csv", "a") as f:
    for epoch in range(15):
        print("Epoch: %02d"%epoch)
        for split in ['train', 'val']:
            total_loss = 0
            count = 0
            total_loss2 = 0
            correct = 0
            a_sub_model.train(split=='train')
            #v_sub_model.train(split=='train')

            with tqdm.tqdm(total=len(dataloaders[split])) as pbar:
                for (i, (spectrogram, video, audio_pred, visual_pred, outcome)) in enumerate(dataloaders[split]):
                    audio_pred = audio_pred.type(torch.FloatTensor)
                    visual_pred = visual_pred.type(torch.FloatTensor)
                    spectrogram = spectrogram.unsqueeze(1)
                    spectrogram = spectrogram.cuda()
                    audio_pred = audio_pred.cuda().unsqueeze(1)
                    visual_pred = visual_pred.cuda().unsqueeze(1)
                    outcome = outcome.type(torch.FloatTensor).cuda()
                    with torch.no_grad():
                        outputs1, class_out, audio_feat = model(spectrogram)
                        _,_,visual_feat = visual_model(video)
                        #feat = torch.cat((audio_feat, visual_feat), dim=1)
                    outputs = a_sub_model(audio_feat.detach(),visual_feat.detach())
                    conf = outputs[:,0:1]
                    # sigma = outputs[:,1:2]
                    # sigma = torch.sigmoid(sigma)*2
                    #outputs = v_sub_model(visual_feat.detach(), audio_feat.detach())
                    conf = torch.sigmoid(conf)
                    outputs = conf * visual_pred + audio_pred*(1-conf)
                    #outputs = outputs * torch.cat((audio_pred, visual_pred), dim=1)
                    #outputs = torch.sum(outputs, dim=1)
                    loss = criterion(outputs.view(-1), outcome)
                    loss3 = torch.mean(torch.abs(outputs.view(-1) - outcome) / outcome)
                    total_loss += loss3.item()*spectrogram.size()[0]
                    count += spectrogram.size()[0]

                    if split == 'train':
                        optim.zero_grad()
                        loss3.backward()
                        optim.step()

                    pbar.set_postfix_str("loss1: {:.4f}".format(total_loss/float(count)))
                    pbar.update()

            if split=='val' and total_loss/float(count) < BestLoss:
                BestLoss = total_loss/float(count)
                save = {
                    'epoch': epoch,
                    'v_state_dict': a_sub_model.state_dict(),
                    'best_loss': BestLoss,
                }

                torch.save(save, save_path+ "best_conf%02d.pt"%save_id)

            f.write("{},{},{}\n".format(epoch,split,total_loss/float(count)))
            f.flush()

    f.write("Best loss: {}\n".format(BestLoss,))
    f.flush()
