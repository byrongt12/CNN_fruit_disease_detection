import csv

import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

from torchvision import models
import torch


class ResNet18(nn.Module):
    def __init__(self):
        self.name = 'ResNet'
        super(ResNet18, self).__init__()
        resnet = models.resnet18(pretrained=True)
        layers = list(resnet.children())[:8]

        self.features1 = nn.Sequential(*layers[:6])
        self.features2 = nn.Sequential(*layers[6:])
        self.classifier = nn.Sequential(nn.BatchNorm1d(512), nn.Softmax(-1))
        self.bb = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 4))

    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = F.relu(x)
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        x = x.view(x.shape[0], -1)
        return self.classifier(x), self.bb(x)

    def getName(self):
        return self.name


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 3, dropout: float = 0.5) -> None:
        super().__init__()
        self.name = 'AlexNet'

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        '''self.classifier = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(256 * 6 * 6, 512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout),
                    nn.Linear(512, 512),
                    nn.ReLU(inplace=True),
                    nn.Linear(512, 3)
        )

        self.bb = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(256 * 6 * 6, 512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout),
                    nn.Linear(512, 512),
                    nn.ReLU(inplace=True),
                    nn.Linear(512, 4)
        )'''

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(nn.BatchNorm1d(256), nn.Softmax(-1))
        self.bb = nn.Sequential(nn.BatchNorm1d(256), nn.Linear(256, 4))

    def forward(self, x):

        x = self.features(x)
        x = self.avgpool(x)
        x = F.relu(x)
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        x = x.view(x.shape[0], -1)

        return self.classifier(x), self.bb(x)

    def getName(self):
        return self.name


class VGG11(nn.Module):
    def __init__(self, in_channels=3, num_classes=3):
        super(VGG11, self).__init__()
        self.name = 'VGG11'
        self.in_channels = in_channels
        self.num_classes = num_classes
        # convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        '''# fully connected linear layers
        self.linear_layers_classify = nn.Sequential(
            nn.Linear(in_features=512*7*7, out_features=4096),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(in_features=4096, out_features=self.num_classes)
        )
        self.linear_layers_bb = nn.Sequential(
            nn.Linear(in_features=512 * 7 * 7, out_features=4096),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(in_features=4096, out_features=4)
        )'''
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(nn.BatchNorm1d(512), nn.Softmax(-1))
        self.bb = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 4))


    def forward(self, x):
        x = self.conv_layers(x)
        x = self.avgpool(x)
        x = F.relu(x)
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        x = x.view(x.shape[0], -1)

        return self.classifier(x), self.bb(x)

    def getName(self):
        return self.name


def update_optimizer(optimizer, lr):
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = lr


def train_epocs(model, optimizer, train_dl, val_dl, epochs=10, C=1000):
    train_loss_arr = []
    train_acc_arr = []
    val_loss_arr = []
    val_acc_arr = []
    torch.cuda.empty_cache()
    idx = 0
    for i in range(epochs):
        model.train()
        total = 0
        sum_loss = 0
        correct = 0
        for x, y_class, y_bb in train_dl:
            batch = y_class.shape[0]
            x = x.cuda().float()
            y_class = y_class.cuda()
            y_bb = y_bb.cuda().float()
            out_class, out_bb = model(x)
            loss_class = F.cross_entropy(out_class, y_class, reduction="sum")
            loss_bb = F.l1_loss(out_bb, y_bb, reduction="none").sum(1)
            loss_bb = loss_bb.sum()
            loss = loss_class + loss_bb / C
            _, pred = torch.max(out_class, 1)
            correct += pred.eq(y_class).sum().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            idx += 1
            total += batch
            sum_loss += loss.item()

        train_loss = sum_loss / total
        train_acc = correct / total
        val_loss, val_acc = val_metrics(model, val_dl, C)
        train_loss_arr.append(train_loss)
        train_acc_arr.append(train_acc)
        if val_loss < 15:
            val_loss_arr.append(val_loss)
        val_acc_arr.append(val_acc)
        print("train_loss %.3f train_acc %.3f val_loss %.3f val_acc %.3f" % (train_loss, train_acc, val_loss, val_acc))

        if i == epochs - 1:
            '''f = open('../results/custom/' + model.getName() + '.csv', 'a', newline='')
            writer = csv.writer(f, delimiter=',')
            data_to_write = [train_loss, train_acc, val_loss, val_acc]
            writer.writerow(data_to_write)

            plt.plot(train_loss_arr, label='Training loss')
            plt.plot(val_loss_arr, label='validation loss')
            plt.title('Training and Validation loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()

            plt.plot(train_acc_arr, label='Train acc')
            plt.plot(val_acc_arr, label='Validation acc')
            plt.title('Training and Validation accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.show()'''

    return train_loss_arr, val_acc_arr


def val_metrics(model, valid_dl, C=1000):
    model.eval()
    total = 0
    sum_loss = 0
    correct = 0
    for x, y_class, y_bb in valid_dl:
        batch = y_class.shape[0]
        x = x.cuda().float()
        y_class = y_class.cuda()
        y_bb = y_bb.cuda().float()
        out_class, out_bb = model(x)
        loss_class = F.cross_entropy(out_class, y_class, reduction="sum")
        loss_bb = F.l1_loss(out_bb, y_bb, reduction="none").sum(1)
        loss_bb = loss_bb.sum()
        loss = loss_class + loss_bb / C
        _, pred = torch.max(out_class, 1)
        correct += pred.eq(y_class).sum().item()
        sum_loss += loss.item()
        total += batch
    return sum_loss / total, correct / total
