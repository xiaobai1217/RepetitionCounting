import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-interval', type=int, help='input a int')
args = parser.parse_args()
interval1 = args.interval
import torch
import tqdm
import numpy as np
import os
import torch.nn.functional as F
from data_loader2 import Audio
import torch.nn as nn


from VGGSound.model import AVENet
from VGGSound.test import args_dict


np.random.seed(0)
torch.manual_seed(0)

torch.backends.cudnn.deterministic = True #0.821
torch.backends.cudnn.benchmark = False
batch_size = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = Audio(split="train", interval=interval1)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=5, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=False)
val_dataset = Audio(split="val")
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=5, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=False)

args = args_dict()
model= AVENet(args)
checkpoint = torch.load("/home/yzhang8/RepNet/vggsound_avgpool.pth.tar")
model.load_state_dict(checkpoint['model_state_dict'])
model.audnet.fc = torch.nn.Linear(512, 18*43)
model.audnet.fc2 = torch.nn.Linear(512,43)

model = model.cuda()
criterion = nn.MSELoss()
criterion2 = nn.CrossEntropyLoss()
tensor = torch.Tensor(np.arange(2,20).astype(np.float32)).cuda()
#optim = torch.optim.SGD(model.parameters(),lr=1e-4, momentum=0.9, weight_decay=1e-4)
optim = torch.optim.SGD(list(model.audnet.layer3.parameters())+list(model.audnet.layer4.parameters())+list(model.audnet.fc.parameters())+list(model.audnet.fc2.parameters()), lr=1e-4, momentum=0.9, weight_decay=1e-4)
dataloaders = {'train':train_dataloader, 'val':val_dataloader}
save_path = 'audio_output/'
if not os.path.exists(save_path):
    os.mkdir(save_path)
BestLoss = float("inf")
with open(save_path+ "log.csv", "a") as f:
    for epoch in range(30):
        print("Epoch: %02d"%epoch)
        for split in ['train', 'val']:
            total_loss = 0
            count = 0
            total_loss2 = 0
            correct = 0
            model.train(split=='train')
            with tqdm.tqdm(total=len(dataloaders[split])) as pbar:
                for (i, (X, outcome)) in enumerate(dataloaders[split]):

                    X = X.unsqueeze(1)
                    X = X.cuda()
                    outcome = outcome.type(torch.FloatTensor).cuda()
                    outputs1, class_out = model(X)
                    # outputs = F.softmax(outputs, dim=1)
                    # outputs = torch.sum(outputs * tensor, dim=1)
                    outputs = []
                    for ii in range(len(outcome)):
                        outputs_tmp = []
                        for jj in range(43):
                            outputs2 = F.softmax(outputs1[ii:(ii+1), jj*18:(jj+1)*18], dim=1)
                            outputs2 = torch.sum(outputs2 * tensor, dim=1,keepdim=True)
                            outputs_tmp.append(outputs2)
                        outputs_tmp = torch.cat(outputs_tmp, dim=1)
                        outputs.append(outputs_tmp)
                    outputs = torch.cat(outputs)
                    outputs = torch.sum(torch.softmax(class_out, dim=1) * outputs, dim=1)
                    loss = criterion(outputs.view(-1),outcome)

                    loss2_label = torch.Tensor(1 - np.eye(X.size()[0])).type(torch.FloatTensor).cuda()
                    class_out = torch.softmax(class_out, dim=1)
                    class_out2 = class_out.permute(1, 0).contiguous()
                    loss2 = torch.sum(torch.matmul(class_out, class_out2) * loss2_label)
                    loss3 = torch.mean(torch.abs(outputs - outcome) / outcome)

                    #loss2 = criterion2(class_out, labels_1)
                    total_loss += loss.item()*X.size()[0]
                    count += X.size()[0]
                    #total_loss2 += loss2.item()*X.size()[0]
                    #_, predicted = torch.max(class_out.data, 1)
                    #correct += (predicted == labels_1).sum().item()

                    if split == 'train':
                        optim.zero_grad()
                        (loss+loss2*10+loss3*10).backward()
                        optim.step()

                    pbar.set_postfix_str("loss1: {:.4f}".format(total_loss/float(count)))
                    pbar.update()

            if split=='val' and total_loss/float(count) < BestLoss:
                BestLoss = total_loss/float(count)
                save = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_loss': BestLoss,
                }

                torch.save(save, save_path+ "best%02d.pt"%interval1)

            f.write("{},{},{}\n".format(epoch,split,total_loss/float(count)))
            f.flush()

    f.write("Best loss: {}\n".format(BestLoss,))
    f.flush()