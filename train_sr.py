import torch
import torchvision
import torch.nn.functional as F
from dataloader_sr import Countix
import numpy as np
from sync_batchnorm import convert_model
import tqdm
import models
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-margin', type=float, help='input a int')
args = parser.parse_args()
margin = args.margin#2.9 best
id1 = int(np.round(margin * 10))
print('train sr',id1)

np.random.seed(0)
torch.manual_seed(0)
# torch.backends.cudnn.enabled = False  # 0.811
torch.backends.cudnn.deterministic = True #0.821
torch.backends.cudnn.benchmark = False
batch_size = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = Countix(split="train")
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=5, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=True)
val_dataset = Countix(split="val")
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=5, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=True)

model = models.video.r2plus1d_18(pretrained=True)
model.fc = torch.nn.Linear(512,34*41)#*43)
model.fc2 = torch.nn.Linear(512,41)
model = convert_model(model)

sub_model = models.video.resnet.SubNet()
checkpoint = torch.load('r2plus1d_18-91a641e6.pth')
sub_model.load_state_dict(checkpoint, strict=False)
sub_model.fc = torch.nn.Linear(512, 1*41)
#sub_model.fc2 = torch.nn.Linear(512, branches)
sub_model = convert_model(sub_model)
if device.type == "cuda":
    model = torch.nn.DataParallel(model)
    sub_model = torch.nn.DataParallel(sub_model)

model = model.cuda()
checkpoint = torch.load("best41.pt")
model.load_state_dict(checkpoint['state_dict'])
sub_model = sub_model.cuda()
model.eval()
dataloaders = {'train': train_dataloader, 'val': val_dataloader}
criterion = torch.nn.MarginRankingLoss(margin=margin)#torch.nn.BCELoss()
criterion2 = torch.nn.CrossEntropyLoss()
optim = torch.optim.SGD(sub_model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)  # Standard
tensor = torch.Tensor(np.arange(2)).type(torch.FloatTensor).cuda().unsqueeze(0)
BestLoss = float("inf")
judge_label = np.concatenate((np.ones((batch_size,)), np.zeros((batch_size))))
judge_label = torch.Tensor(judge_label).type(torch.FloatTensor).cuda()
tensor1 = np.ones((batch_size,))
tensor1 = torch.Tensor(tensor1).type(torch.LongTensor).cuda()
with open("log.csv", "a") as f:
    for epoch in range(5):
        print("Epoch: %02d"%epoch)
        for phase in ['train', 'val']:
            with torch.set_grad_enabled(phase == 'train'):
                dataloader = dataloaders[phase]
                total_loss = 0
                acc=0
                judge_loss = 0
                count = 0
                sub_model.train(phase=='train')
                with tqdm.tqdm(total=len(dataloader)) as pbar:
                    for (i, video) in enumerate(dataloader):

                        b1, b2, c, f1, h, w = video.size()
                        video = video.view(b1*b2, c, f1 , h, w)
                        video = video.cuda()

                        with torch.no_grad():
                            _, class_out, feat = model(video)

                        outputs1 = sub_model(feat.detach())
                        class_out = torch.softmax(class_out.detach(), dim=1)
                        outputs1 = torch.sum(outputs1 * class_out, dim=1)

                        outputs1 = outputs1.view(b1, b2)
                        outputs1_p = outputs1[:,0]
                        outputs1_n = outputs1[:,1]

                        loss = criterion(outputs1_p, outputs1_n, tensor1)

                        # class_out = class_out.view(b1,b2, -1)
                        # class_out = class_out[:,0,:]
                        # class_out2 = class_out.permute(1,0).contiguous()
                        # loss2_label = torch.Tensor(1 - np.eye(b1 )).type(torch.FloatTensor).cuda()
                        # loss2 = torch.sum(torch.matmul(class_out, class_out2)*loss2_label)

                        if phase == 'train':
                            optim.zero_grad()
                            #(loss+loss2).backward()
                            loss.backward()
                            optim.step()

                        count += b1
                        # acc+=acc1
                        total_loss += loss.item()*batch_size
                        epoch_loss = total_loss / float(count)
                        pbar.set_postfix_str("{:.4f} ({:.4f}) ".format(epoch_loss, loss.item()))
                        pbar.update()

                        if i == 600 or i == 1200:
                            dataloader1 = val_dataloader
                            total_loss1 = 0
                            count1 = 0
                            sub_model.eval()
                            with torch.set_grad_enabled(False):
                                for (t, video) in enumerate(dataloader1):
                                    b1, b2, c, f1, h, w = video.size()
                                    video = video.view(b1 * b2, c, f1, h, w)
                                    video = video.cuda()

                                    with torch.no_grad():
                                        _, class_out, feat = model(video)

                                        outputs1 = sub_model(feat.detach())
                                        class_out = torch.softmax(class_out.detach(), dim=1)
                                        outputs1 = torch.sum(outputs1 * class_out, dim=1)

                                        outputs1 = outputs1.view(b1, b2)
                                        outputs1_p = outputs1[:, 0]
                                        outputs1_n = outputs1[:, 1]

                                        loss = criterion(outputs1_p, outputs1_n, tensor1)


                                    count1 += b1
                                    # acc+=acc1
                                    total_loss1 += loss.item() * batch_size
                                    epoch_loss1 = total_loss1 / float(count1)
                                print(epoch_loss1)
                                f.write("{},{},{}\n".format(epoch, phase, epoch_loss1))
                                f.flush()
                            if BestLoss > epoch_loss1:
                                BestLoss = epoch_loss1
                                save = {
                                    'state_dict': sub_model.state_dict(),
                                }
                                torch.save(save, "best_sr%02d.pt"%id1)
                            sub_model.train()


            f.write("{},{},{}\n".format(epoch, phase, epoch_loss))
            f.flush()
        if BestLoss>epoch_loss:
            BestLoss = epoch_loss
            save = {
                'state_dict': sub_model.state_dict(),
            }
            torch.save(save, "best_sr%02d.pt"%id1)
