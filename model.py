ROOT = "C:\\Program Files\\jupyter-workspace\\wizardChess\\data\\"

"""
==========Example Usage==========
seg = wizSegmenter(img_size=256)
seg.define()
seg.train(batch_size=6, epochs=50, lr=6e-3, weight_decay=1e-4)
seg.curves()
seg.sample()
"""

import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import io
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from IPython.display import clear_output

class wizDataset(torch.utils.data.Dataset):
    def __init__(self, root=ROOT, img_size=1072):
        self.IMG_SIZE = img_size
        self.transform = A.Compose(
            [
                A.Affine(p=0.5, scale=(0.95, 1.05)),
                A.Resize(img_size, img_size),
                A.VerticalFlip(p=0.15),
                A.HorizontalFlip(p=0.15),
                A.Affine(p=0.2, shear=(-5, 5)),
                A.RGBShift(p=0.2, r_shift_limit=10, g_shift_limit=10, b_shift_limit=10),
                A.RandomBrightnessContrast(p=0.2, brightness_limit=0.1, contrast_limit=0.1),
                
            ]
        )

        imgs = []
        labels = []
        for i in os.listdir(root+"images"):
            imgs += [root+"images\\"+i] #changed for colab
            labels += [root+"labels\\"+i.split(".")[0]+".png"]
        self.data = list(zip(imgs, labels))
        random.shuffle(self.data)

        class_counts = [0, 0]
        for datapoint in self.data[:len(self.data)//2]:
            label = io.imread(datapoint[1], as_gray=True)
            class_counts[0] += np.sum(label==0)
            class_counts[1] += np.sum(label>0)

        self.class_weights = torch.tensor([1-class_count/sum(class_counts) for class_count in class_counts],
                                          dtype=torch.float).to("cuda")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = io.imread(self.data[idx][0])
        #image = cv2.resize(image, (self.IMG_SIZE, self.IMG_SIZE))#.reshape(3, self.IMG_SIZE, self.IMG_SIZE)/255.0
        mask = io.imread(self.data[idx][1], cv2.IMREAD_UNCHANGED)#, as_gray=True)
        #mask = cv2.resize(mask, (self.IMG_SIZE, self.IMG_SIZE))#.reshape(1, self.IMG_SIZE, self.IMG_SIZE)
        
        if self.transform is not None:
            transformed = self.transform(image = image, mask = mask)
            image = transformed["image"]
            mask = transformed["mask"]

        mask[mask > 0] = 1
        image = image.reshape(3, self.IMG_SIZE, self.IMG_SIZE)/255.0
        #mask = mask.reshape(1, self.IMG_SIZE, self.IMG_SIZE)
        mask = mask.reshape(self.IMG_SIZE, self.IMG_SIZE)

        return torch.tensor(image, dtype = torch.float32).to("cuda"), torch.tensor(mask, dtype = torch.int64).to("cuda")

class train_val_test_split(torch.utils.data.Dataset):
    def __init__(self, dataset, split = (.85, .1, .05), mode="train"):
        self.dataset = dataset
        self.split = split
        self.mode = 0 if mode == "train" else 1 if mode == "val" else 2 if mode == "test" else -1
        self.interval = (round(sum(split[0:self.mode])*len(dataset)), round(sum(split[0:self.mode+1])*len(dataset)))

    def __len__(self):
        return round(len(self.dataset) * self.split[self.mode])

    def __getitem__(self, idx):
        # if idx+self.interval[0] >= self.interval[1]:
        #     raise StopIteration
        return self.dataset[idx + self.interval[0]]


class attentionBlock(nn.Module):
    def __init__(self, skip_channels, upsample_channels, hidden_channels):
        super(attentionBlock, self).__init__()
        self.Wskip = nn.Conv2d(in_channels=upsample_channels, out_channels=hidden_channels, kernel_size=1)
        self.Wupsample = nn.Conv2d(in_channels=skip_channels, out_channels=hidden_channels, kernel_size=1)
        self.Walpha = nn.Conv2d(in_channels=hidden_channels, out_channels=1, kernel_size=1)

    def forward(self, skip, upsample):
        skip1 = self.Wskip(skip)
        upsample1 = F.interpolate(self.Wupsample(upsample), skip.shape[2:], mode='bilinear', align_corners=False)
        concat_logits = F.relu(skip1 + upsample1)
        alpha = torch.sigmoid(self.Walpha(concat_logits))
        return alpha * skip


class doubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(doubleConv, self).__init__()
        self.batch_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=1, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), stride=1,
                      padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        x = self.batch_conv(x)
        return x


class downBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(downBlock, self).__init__()
        self.pool_conv = nn.Sequential(nn.MaxPool2d(2), doubleConv(in_channels, out_channels))

    def forward(self, x):
        x = self.pool_conv(x)
        return x


class upBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(upBlock, self).__init__()
        self.tconv1 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3),
                                         stride=2, padding=1, output_padding=1)
        self.atten1 = attentionBlock(skip_channels=out_channels, upsample_channels=out_channels,
                                     hidden_channels=out_channels // 2)
        self.conv = doubleConv(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x, skip):
        x = self.tconv1(x)
        x = F.interpolate(x, skip.shape[2:], mode='bilinear', align_corners=False)
        skip = self.atten1(skip, x)
        x = self.conv(torch.cat([skip, x], dim=1))
        return x


class attentionSegmenter(nn.Module):  # U-Net model
    def __init__(self):  # needs regularization (dropout or batch normalization)
        super(attentionSegmenter, self).__init__()

        # downsampling section
        self.inConv = doubleConv(in_channels=3, out_channels=64)

        self.down1 = downBlock(in_channels=64, out_channels=128)
        self.down2 = downBlock(in_channels=128, out_channels=256)
        self.down3 = downBlock(in_channels=256, out_channels=512)
        self.down4 = downBlock(in_channels=512, out_channels=1024)

        # upsampling section
        self.up1 = upBlock(in_channels=1024, out_channels=512)
        self.up2 = upBlock(in_channels=512, out_channels=256)
        self.up3 = upBlock(in_channels=256, out_channels=128)
        self.up4 = upBlock(in_channels=128, out_channels=64)

        self.outConv = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=(1, 1), stride=1, padding='same')

    def forward(self, x):
        # downsampling
        x1 = self.inConv(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # upsampling
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x = self.outConv(x)
        return x
        
class wizSegmenter:
    def __init__(self, root="C:\\Program Files\\jupyter-workspace\\wizardChess\\data\\",img_size=1072):
        self.history = {"train_loss":[], "train_acc":[], "validation_loss":[], "validation_acc":[], "precision":[], "recall":[], "lr":[]}
        self.dataset = wizDataset(root = root, img_size = img_size)

        #self.train_data, self.val_data = train_test_split(self.dataset, test_size = 0.1, train_size = 0.9)
        self.train_data = train_val_test_split(self.dataset, split=(.9, .1, 0), mode = "train")
        self.val_data = train_val_test_split(self.dataset, split=(.9, .1, 0), mode = "val")
        # self.test_data = train_val_test_split(dataset, split=(.85,.1,.05), mode = "test")
        print(len(self.train_data), "training", len(self.val_data), "validation")
    
    def define(self):
        self.net = attentionSegmenter().to("cuda")

    def get_lr(self, optimizer):
      for param_group in optimizer.param_groups:
        return param_group['lr']
        
    def train(self, batch_size, epochs, lr, weight_decay):
        dataloader = torch.utils.data.DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        valdataloader = torch.utils.data.DataLoader(self.val_data, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(params=self.net.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=epochs,
                                            steps_per_epoch=len(dataloader))
        loss_criterion = nn.CrossEntropyLoss(weight=self.dataset.class_weights)
        
        for epoch in range(epochs):
            self.net.train()
            loss_sum = 0
            
            # (accuracy_numerator, accuracy_denominator)
            metrics = (0, 0)
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs} training"):
                self.net.zero_grad()
                outputs = self.net(batch[0])
                loss = loss_criterion(outputs, batch[1])
                loss_sum += loss.item()
                
                # calculate accuracy
                batch_metrics = (torch.sum(torch.argmax(outputs, dim=1) == batch[1]).item(), torch.numel(batch[1]))
                metrics = [a + b for a, b in zip(metrics, batch_metrics)]

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                self.history["lr"] += [self.get_lr(optimizer)]
                scheduler.step()
          
            if (epoch+1) % 5 == 0:
                clear_output(wait=True)

            print(f"loss: {loss_sum / len(dataloader)}\t accuracy:{metrics[0]/metrics[1]}\t lr:" + str(float('%.2g' % self.history["lr"][-1])))
            self.history["train_loss"] += [loss_sum / len(dataloader)]
            self.history["train_acc"] += [metrics[0]/metrics[1]]

            self.net.eval()
            loss_sum = 0

            # (accuracy_numerator, accuracy_denominator, recall_numerator, recall_denominator, precision_numerator, precision_denominator)
            metrics = (0, 0, 0, 0, 0, 0)

            with torch.no_grad():
                for batch in tqdm(valdataloader, desc= "validating"):
                    outputs = self.net(batch[0])
                    loss_sum += loss_criterion(outputs, batch[1]).item()
                    #calculate accuracy, precision, & recall
                    batch_metrics = self.evaluate(outputs, batch[1])
                    metrics = [a + b for a, b in zip(metrics, batch_metrics)]
            print(f"val loss: {loss_sum / len(valdataloader)}\t val accuracy:{metrics[0]/metrics[1]}\t " + 
                  f"val recall:{metrics[2]/metrics[3]}\t val precision:{metrics[4]/metrics[5]}")
            self.history["validation_loss"] += [loss_sum/len(valdataloader)]
            self.history["validation_acc"] += [metrics[0]/metrics[1]]
            self.history["recall"] += [metrics[2]/metrics[3]]
            self.history["precision"] += [metrics[4]/metrics[5]]

            self.sample()

    def evaluate(self, output, target):
        true_pos= 0
        true_neg = 0
        false_pos = 0
        false_neg = 0
        
        pred = torch.argmax(F.softmax(output, dim=1), dim=1)
        # pred = pred.reshape(-1)
        # expected = target.reshape(-1)
        
        true_pos = torch.sum(pred * target).item()
        true_neg = torch.sum((pred==0) * (target==0)).item()
        false_pos = torch.sum(pred * (target==0)).item()
        false_neg = torch.sum((pred==0) * target).item()

        return (true_pos + true_neg,
                true_pos + true_neg + false_pos + false_neg, 
                true_pos, 
                true_pos + false_neg, 
                true_pos, 
                true_pos + false_pos)
        


    def curves(self, separate=False):
        if separate:
            plt.figure(1)
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title("Train Loss")
            plt.plot(self.history["train_loss"])
            plt.figure(2)
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title("Validation Loss")
            plt.plot(self.history["validation_loss"], color="orange")
            
            plt.figure(3)
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.title("Train Accuracy")
            plt.plot(self.history["train_acc"])
            plt.figure(4)
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.title("Validation Accuracy")
            plt.plot(self.history["validation_acc"], color="orange")

            plt.figure(5)
            plt.xlabel("Batches")
            plt.ylabel("Learning Rate")
            plt.title("lr over time")
            plt.plot(self.history["lr"])
            
            plt.figure(6)
            plt.xlabel("Epochs")
            plt.title("Precision")
            plt.plot(self.history["precision"], color="green")
            
            plt.figure(7)
            plt.xlabel("Epochs")
            plt.title("Recall")
            plt.plot(self.history["recall"], color="red")
        else:
            plt.figure(1)
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title("Train Loss (blue) & Val Loss (orange)")
            plt.plot(self.history["train_loss"]) #blue is training loss
            plt.plot(self.history["validation_loss"]) #orange is validation loss
            
            plt.figure(2)
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.title("Train Accuracy (blue) & Val Accuracy (orange)")
            plt.plot(self.history["train_acc"]) #blue is training loss
            plt.plot(self.history["validation_acc"]) #orange is validation loss

            plt.figure(3)
            plt.xlabel("Batches")
            plt.ylabel("Learning Rate")
            plt.title("lr over time")
            plt.plot(self.history["lr"])
            
            plt.figure(4)
            plt.xlabel("Epochs")
            plt.title("Precision (green) & Recall (red)")
            plt.plot(self.history["precision"], color="green")
            plt.plot(self.history["recall"], color="red")
        plt.show()
    
    def sample(self):
        selection = random.randint(1,len(self.val_data)-1)
        selection = self.val_data[selection]
        output = self.net(selection[0].unsqueeze(0)).squeeze(0)
        output = nn.Softmax(dim=0)(output)[1]
        plt.imshow(selection[0].reshape(self.dataset.IMG_SIZE, self.dataset.IMG_SIZE, 3).cpu().numpy())
        plt.show()
        # plt.imshow(selection[1].cpu().numpy(), cmap='seismic', vmax=1, vmin=0)
        # plt.show()
        plt.imshow(output.detach().cpu().numpy(), cmap='seismic', vmax=1, vmin=0)
        plt.show()
