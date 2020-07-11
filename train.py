from data_iapr import MyDataset
from model import ImageNet
from model import TextNet
from model import save_model

import torch
from torch import nn
import torch.optim as optim
import torch.utils.data as Data
from torch.autograd import Function
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

import ot
import random
import numpy as np
from test import test_main

view_name = ['image', 'text']


class WassersteinLoss(Function):
    def __init__(self, m, reg):
        super(WassersteinLoss, self).__init__()
        self.m = m / np.max(m)
        self.log = {}
        self.reg = reg

    def forward(self, x, y):
        self.save_for_backward(x, y)
        loss = torch.zeros(1).cuda()
        for i in range(x.shape[0]):
            a = x[i].cpu().numpy()
            a[a <= 0] = 1e-9
            a = a / np.sum(a)

            b = y[i].cpu()
            b = nn.Softmax()(b.view(1, -1)).data.numpy().reshape((-1,))
            b[b <= 0] = 1e-9
            b = b / np.sum(b)

            dis, log_i = ot.sinkhorn2(a, b, self.m, self.reg, log=True)
            self.log[i] = torch.FloatTensor(log_i['u'])
            loss += dis[0]

        return loss

    def backward(self, grad_output):
        x, y, = self.saved_tensors

        grad_x = []
        L = y.shape[1]
        e = torch.ones(1, x.shape[1]).cuda()
        for i in range(x.shape[0]):
            u = self.log[i].cuda()
            u = torch.log(u.view(1, -1)) / self.reg - torch.log(torch.sum(u, 0))[0] * e / (self.reg * L)
            grad_x.append(u)

        grad_x = torch.cat(grad_x)
        return grad_x, None


def cal_distance_matrix(k):
    mm = np.zeros(k.shape)
    for i in range(mm.shape[0]):
        for j in range(mm.shape[1]):
            mm[i][j] = k[i][i] + k[j][j] - 2*k[i][j]
    return mm


def pre_train(hp, models):
    print("----------start pre-training models----------")
    train_data = MyDataset()
    train_loader = Data.DataLoader(dataset=train_data, batch_size=hp['pre_size'], shuffle=True, num_workers=12)

    models[1].cuda()
    models[1].train()

    optimizer = optim.Adam([{'params': models[1].parameters()}], lr=hp['pre_lr'])
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch in range(hp['pre_epoch']):
        scheduler.step()
        running_loss = 0.0
        models[1].train()
        for step, (x, y) in enumerate(train_loader):
            b_x = Variable(x[1]).cuda()
            b_y = Variable(y).cuda()

            # forward
            h = models[1](b_x)

            # loss
            loss_func = nn.MSELoss()
            loss = loss_func(h, b_y)
            running_loss += loss.data[0] * x[1].size(0)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # epoch loss
        epoch_loss = running_loss / len(train_data)
        print('epoch {}/{} | Loss: {:.9f}'.format(epoch, hp['pre_epoch'], epoch_loss))
    print("----------end pre-training models----------")
    return models


def train_main(hp):
    models = [ImageNet(hp['label']), TextNet(hp['data_name'], hp['label'])]
    models = pre_train(hp, models)

    print("----------start training models----------")
    view_num = len(models)  # num of view
    l = hp['label']  # num of label

    # 初始化K0,M矩阵
    if 'k_0' in hp.keys():
        k_0 = hp['k_0']
    else:
        k_0 = torch.nn.Softmax()(torch.eye(l))
        k_0 = k_0.data.numpy()
    k_0_inv = np.linalg.inv(k_0)
    m = cal_distance_matrix(k_0)

    trade = hp['trade_off']  # 平衡系数

    lr = hp['lr']
    for i in range(view_num):
        models[i].cuda()

    train_data = MyDataset()
    train_loader = Data.DataLoader(dataset=train_data, batch_size=hp['batch_size'], shuffle=True, num_workers=12)

    for epoch in range(hp['epoch']):
        # first stage
        par = []
        for i in range(view_num):
            models[i].train()
            par.append({'params': models[i].parameters()})

        optimizer = optim.Adam(par, lr=lr)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

        for epoch_1 in range(hp['epoch_1']):
            scheduler.step()
            total_loss = 0
            for step, (x, y) in enumerate(train_loader):
                print(step)
                # get data
                for i in range(view_num):
                    b_x = Variable(x[i]).cuda()
                    b_y = Variable(y).cuda()

                    # forward
                    h = models[i](b_x)

                    # loss
                    loss_func = WassersteinLoss(m, hp['reg'])
                    loss = loss_func(h, b_y)
                    total_loss += loss.data[0] * x[i].size(0)

                    # backward
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            print("epoch " + str(epoch) + " | epoch_1 " + str(epoch_1) + " | loss: %.9f" % (total_loss / len(train_data)))

        # seconde stage
        for i in range(view_num):
            models[i].eval()
        T = np.zeros((l, l))

        # calculate T

        for i in range(len(train_data)):
            # get data
            x, b_y = train_data[i]
            b_y = nn.Softmax()(b_y.view(1, -1)).data.numpy().reshape((-1,))
            b_y[b_y <= 0] = 1e-9
            b_y = b_y / np.sum(b_y)
            for j in range(view_num):
                if j == 0:
                    b_x = Variable(x[j].view(1, 4, 3, 224, 224), volatile=True).cuda()
                else:
                    b_x = Variable(x[j].view(1, 4, -1), volatile=True).cuda()

                # forward
                h = models[j](b_x).cpu().data.numpy()
                h[h <= 0] = 1e-9
                h = h / np.sum(h)
                Gs = ot.sinkhorn(h.reshape(-1), b_y.reshape(-1), m / np.max(m), hp['reg'])
                T += Gs
            break
        # T /= (bag_num * view_num)

        # calculate K
        G = np.zeros((l, l))
        for i in range(l):
            for j in range(l):
                if i == j:
                    for k in range(l):
                        if k != i:
                            G[i][j] -= (T[i][k] + T[k][i])
                else:
                    G[i][j] = 2 * T[i][j]
        # K = np.linalg.inv(k_0_inv - G / trade)
        K = k_0 + G / trade
        K = (K + K.T) / 2
        u, v = np.linalg.eig(K)
        u[u < 0] = 0
        K = np.dot(v, np.dot(np.diag(u), v.T))

        # calculate M
        m = cal_distance_matrix(K)

    save_model(hp['data_name'], "1", models)
    print("----------end training models----------")
    test_main(hp, models, save=True)
    np.save('./parameter/' + hp['data_name'] + '/relation_1.npy', K)
