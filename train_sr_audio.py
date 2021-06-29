import torch
import tqdm
import numpy as np
import os
import torch.nn.functional as F
from data_loader_sr_audio import Audio
import torch.nn as nn
import models
from sync_batchnorm import convert_model
from VGGSound.model import AVENet
from VGGSound.test import args_dict
import argparse
import VGGSound

parser = argparse.ArgumentParser()
parser.add_argument('-lambda1', type=float, help='input a int')
args = parser.parse_args()
lambda1 = args.lambda1
id1 = int(np.round(lambda1*100))

np.random.seed(0)
torch.manual_seed(0)

torch.backends.cudnn.deterministic = True #0.821
torch.backends.cudnn.benchmark = False
batch_size = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = Audio(split="train",lambda1=lambda1)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=5, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=False)
val_dataset = Audio(split="val",lambda1=lambda1)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=5, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=False)

args = args_dict()
model= AVENet(args)
checkpoint = torch.load("/home/yzhang8/RepNet/vggsound_avgpool.pth.tar")
model.audnet.fc = torch.nn.Linear(512, 18*43)
model.audnet.fc2 = torch.nn.Linear(512,43)
checkpoint = torch.load("audio_output/best_a.pt")
model.load_state_dict(checkpoint['state_dict'])
model = model.cuda()
model.eval()

a_sub_model = VGGSound.models.resnet.SubNet()
a_sub_model.load_state_dict(torch.load("/home/yzhang8/RepNet/vggsound_avgpool.pth.tar"), strict=False)
a_sub_model = a_sub_model.cuda()


v_model = models.video.r2plus1d_18(pretrained=True)
v_model.fc = torch.nn.Linear(512,34*41)#*43)
v_model.fc2 = torch.nn.Linear(512,41)
v_model = convert_model(v_model)

sub_model = models.video.resnet.SubNet()
sub_model.fc = torch.nn.Linear(512, 1*41)
sub_model = convert_model(sub_model)

if device.type == "cuda":
    v_model = torch.nn.DataParallel(v_model)
    sub_model = torch.nn.DataParallel(sub_model)

v_model = v_model.cuda()
checkpoint = torch.load("best_v.pt")
v_model.load_state_dict(checkpoint['state_dict'])
v_model.eval()
checkpoint = torch.load("best_sr.pt")
sub_model.load_state_dict(checkpoint['state_dict'])
sub_model.module.fc = nn.Sequential(torch.nn.Linear(1024,41))
sub_model = sub_model.cuda()

criterion = torch.nn.MarginRankingLoss(margin=1.0)#torch.nn.BCELoss()
tensor = torch.Tensor(np.arange(2,28).astype(np.float32)).cuda()
optim = torch.optim.SGD(list(sub_model.module.fc.parameters())+list(a_sub_model.parameters()), lr=1e-3, momentum=0.9, weight_decay=1e-4)
#optim = torch.optim.SGD([{'params':list(model.audnet.fc.parameters())+list(model.audnet.fc2.parameters())+list(v_model.module.fc.parameters())+list(v_model.module.fc2.parameters())},{'params':list(v_model.module.fc3.parameters()),'lr':1e-3}], lr=1e-4, momentum=0.9, weight_decay=1e-4)
dataloaders = {'train':train_dataloader, 'val':val_dataloader}
save_path = 'audio_output/'
if not os.path.exists(save_path):
    os.mkdir(save_path)
BestLoss = float("inf")

with open(save_path+ "log.csv", "a") as f:
    for epoch in range(15):
        print("Epoch: %02d"%epoch)
        for split in ['train', 'val']:
            total_loss = 0
            count = 0
            total_loss2 = 0
            correct = 0
            sub_model.train(split=='train')
            a_sub_model.train(split=='train')
            with tqdm.tqdm(total=len(dataloaders[split])) as pbar:
                for (i, (spectrogram, video)) in enumerate(dataloaders[split]):
                    b1, b2, c, f1, h, w = video.size()
                    video = video.view(b1 * b2, c, f1, h, w)
                    video = video.cuda()
                    b1,b2,h,w=spectrogram.size()
                    spectrogram = spectrogram.view(b1*b2, h,w)
                    spectrogram = spectrogram.unsqueeze(1)
                    spectrogram = spectrogram.cuda()

                    with torch.no_grad():
                        _, _, audio_feat = model(spectrogram)
                        _, class_out, feat = v_model(video)

                    audio_feat = a_sub_model(audio_feat.detach())
                    outputs1 = sub_model(feat.detach(), audio_feat.flatten(1))
                    class_out = torch.softmax(class_out.detach(), dim=1)
                    outputs1 = torch.sum(outputs1 * class_out, dim=1)

                    outputs1 = outputs1.view(b1, b2)
                    outputs1_p = outputs1[:, 0]
                    outputs1_n = outputs1[:, 1]
                    tensor1 = np.ones((b1,))
                    tensor1 = torch.Tensor(tensor1).type(torch.LongTensor).cuda()
                    loss = criterion(outputs1_p, outputs1_n, tensor1)

                    if split == 'train':
                        optim.zero_grad()
                        loss.backward()
                        optim.step()
                    total_loss += loss.item()*b1
                    count += b1
                    pbar.set_postfix_str("loss1: {:.4f}".format(total_loss/float(count)))
                    pbar.update()

            if split=='val' and total_loss/float(count) < BestLoss:
                BestLoss = total_loss/float(count)
                save = {
                    'epoch': epoch,
                    'sr_state_dict': sub_model.state_dict(),
                    'a_sr_state_dict':a_sub_model.state_dict(),
                    'best_loss': BestLoss,
                }

                torch.save(save, save_path+ "best2_%03d.pt"%id1)

            f.write("{},{},{}\n".format(epoch,split,total_loss/float(count)))
            f.flush()

    f.write("Best loss: {}\n".format(BestLoss,))
    f.flush()
