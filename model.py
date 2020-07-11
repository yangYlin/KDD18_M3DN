import os
import torch
from torch import nn
from torchvision import models

linear_model = {
    'iapr': [6514, 1024, 1024, 512, 512]
}


class ImageNet(nn.Module):
    def __init__(self, label):
        super(ImageNet, self).__init__()
        self.feature = models.resnet18(pretrained=True)
        self.feature.fc = nn.Linear(512, label)
        self.maxpool = nn.MaxPool2d(kernel_size=(4, 1))
        self.softmax = nn.Softmax()
        self._initialize_weights()

    def forward(self, x):
        N = x.size(0)
        x = self.feature(x.view(N * 4, 3, 224, 224))
        y = self.maxpool(x.view(N, 4, -1))
        z = self.softmax(y.view(N, -1))
        return z

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.1)
                m.bias.data.zero_()


class TextNet(nn.Module):
    def __init__(self, data_name, label):
        super(TextNet, self).__init__()
        self.feature = make_layers(linear_model[data_name])
        self.fc = nn.Linear(linear_model[data_name][-1], label)
        self.maxpool = nn.MaxPool2d(kernel_size=(4, 1))
        self.softmax = nn.Softmax()
        self._initialize_weights()

    def forward(self, x):
        N = x.size(0)
        x = self.feature(x.view(N * 4, -1))
        x = self.fc(x)
        y = self.maxpool(x.view(N, 4, -1))
        z = self.softmax(y.view(N, -1))
        return z

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.1)
                m.bias.data.zero_()


def make_layers(cfg):
    layers = []
    n = len(cfg)
    input_dim = cfg[0]
    for i in range(1, n):
        output_dim = cfg[i]
        if i < n - 1:
            layers += [nn.Linear(input_dim, output_dim), nn.BatchNorm2d(output_dim), nn.Dropout(), nn.ReLU(inplace=True)]
        else:
            layers += [nn.Linear(input_dim, output_dim), nn.ReLU(inplace=True)]
        input_dim = output_dim
    return nn.Sequential(*layers)


def load_model(label, data_name, pre_train, model_id=-1):
    print("----------start loading models----------")
    my_models = [ImageNet(label), TextNet(data_name, label)]
    if pre_train is True:
        for i in range(len(my_models)):
            path = './parameter/' + data_name + '/model/model_' + model_id + '_' + str(i) + '.pkl'
            if os.path.exists(path):
                my_models[i].load_state_dict(torch.load(path))
            else:
                print(path)
                print("No such model parameter !!")

    print("----------end loading models----------")
    print("model information: ")
    for i in range(len(my_models)):
        print(my_models[i])
    return my_models


def save_model(data_name, model_id, models):
    for i in range(len(models)):
        path = './parameter/' + data_name + '/model/model_' + model_id + '_' + str(i) + '.pkl'
        torch.save(models[i].state_dict(), open(path, 'wb'))
