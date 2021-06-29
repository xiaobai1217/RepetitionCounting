import torch
import torchvision
import torch.nn.functional as F
from dataloader import Countix
import numpy as np
from sync_batchnorm import convert_model
import tqdm
import models

# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('-number', type=int, help='input a int')
# args = parser.parse_args()
number = 41#args.number

np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.enabled = False  # 0.811
batch_size = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = Countix(split="train")
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=5, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=True)
val_dataset = Countix(split="val")
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=5, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=True)

model = models.video.r2plus1d_18(pretrained=True)
model.fc = torch.nn.Linear(512,34*number)
model.fc2 = torch.nn.Linear(512,number)

model = convert_model(model)
if device.type == "cuda":
    model = torch.nn.DataParallel(model)
model = model.cuda()
dataloaders = {'train': train_dataloader, 'val': val_dataloader}
criterion = torch.nn.MSELoss()
criterion2 = torch.nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)  # Standard
tensor = torch.Tensor(np.arange(2,36)).type(torch.FloatTensor).cuda().unsqueeze(0)
BestLoss = float("inf")
loss2_label = torch.Tensor(1-np.eye(batch_size)).type(torch.FloatTensor).cuda()
with open("log.csv", "a") as f:
    for epoch in range(8):
        print("Epoch: %02d"%epoch)
        for phase in ['train', 'val']:
            with torch.set_grad_enabled(phase == 'train'):
                dataloader = dataloaders[phase]
                total_loss = 0
                judge_loss = 0
                count = 0
                with tqdm.tqdm(total=len(dataloader)) as pbar:
                    for (i, (video, labels, classes)) in enumerate(dataloader):

                        video = video.cuda()
                        labels = labels.type(torch.FloatTensor).cuda()
                        classes = classes.type(torch.LongTensor).cuda()

                        outputs1, class_out,_ = model(video)

                        outputs = []
                        for ii in range(len(labels)):
                            outputs_tmp = []
                            for jj in range(number):
                                outputs2 = F.softmax(outputs1[ii:(ii + 1), jj * 34:(jj + 1) * 34], dim=1)
                                outputs2 = torch.sum(outputs2 * tensor, dim=1, keepdim=True)
                                outputs_tmp.append(outputs2)
                            outputs_tmp = torch.cat(outputs_tmp, dim=1)
                            outputs.append(outputs_tmp)
                        outputs = torch.cat(outputs)
                        outputs = torch.sum(torch.softmax(class_out, dim=1) * outputs, dim=1)

                        class_out = torch.softmax(class_out, dim=1)
                        class_out2 = class_out.permute(1,0).contiguous()
                        loss2 = torch.sum(torch.matmul(class_out, class_out2)*loss2_label)

                        loss = criterion(outputs, labels)
                        #loss2 = criterion2(class_out, classes)
                        loss3 = torch.mean(torch.abs(outputs - labels)/labels)

                        if phase == 'train':
                            optim.zero_grad()
                            (loss+loss2*10+loss3*10).backward()
                            optim.step()

                        count += labels.size()[0]
                        total_loss += loss.item()*labels.size()[0]
                        epoch_loss = total_loss / float(count)
                        pbar.set_postfix_str("{:.2f} ({:.2f})".format(epoch_loss, loss.item()))
                        pbar.update()

            f.write("{},{},{}\n".format(epoch, phase, epoch_loss))
            f.flush()
        if BestLoss>epoch_loss:
            BestLoss = epoch_loss
            save = {
                'state_dict': model.state_dict(),
            }
            torch.save(save, "best%02d.pt"%number)

