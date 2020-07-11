import os
import csv
import torch
import numpy as np
from PIL import Image
import torch.utils.data as Data
import torchvision.transforms as transforms


error_id = ['4072']


def deal_single_text(x, cut=3):
    d = x.shape[0]
    s = 0
    bag = []
    for i in range(cut):
        dim = int(d / (cut - i))
        text = np.zeros((x.shape[0]))
        text[s:(s + dim)] = x[s:(s+dim)]
        bag.append(text.reshape(1, -1))
        d -= dim
        s += dim
    bag = np.concatenate(bag)
    bag = torch.FloatTensor(bag)
    return bag


class MyDataset(Data.Dataset):  # 需要继承data.Dataset
    def __init__(self, img_blcok=2, text_block=4, train=True, transform=transforms.ToTensor()):
        # TODO
        # 1. Initialize file path or list of file names.
        self.root = './dataset/'
        self.train = train
        self.transform = transform
        self.img_blcok = img_blcok
        self.text_block = text_block
        self.label = {}
        with open("./dataset/label.csv", "r") as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                self.label[row[0]] = torch.zeros(255)
                for i in range(1, len(row)):
                    self.label[row[0]][int(row[i])] = 1
        if self.train is True:
            self.data_file = []
            with open('./dataset/train_id.txt', 'r') as f:
                for row in f:
                    x = row.strip().split()
                    if x[0] in error_id:
                        continue
                    self.data_file.append(x[0])
        else:
            self.data_file = []
            with open('./dataset/test_id.txt', 'r') as f:
                for row in f:
                    x = row.strip().strip('\n').split()
                    if x[0] in error_id:
                        continue
                    self.data_file.append(x[0])

    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        idx = self.data_file[index]
        target = self.label[idx]

        file_path = self.root + '/picture/' + idx + '.jpg'
        if os.path.exists(file_path):
            try:
                img = Image.open(file_path).convert('RGB').resize((224 * self.img_blcok, 224 * self.img_blcok))
                img = self.transform(img)
            except:
                print("图片无法打开 ", file_path)
        if img is None:
            print(file_path)
            print("未找到图片: ", idx, self.label[idx])
        imgs = []
        for i in range(self.img_blcok):
            for j in range(self.img_blcok):
                imgs.append(img[0:3, (i * 224):((i+1) * 224), (j * 224):((j+1) * 224)])
        imgs = torch.stack(imgs, 0)

        file_path = self.root + '/text/' + idx + '.npy'
        text = np.load(file_path)
        texts = deal_single_text(text, cut=self.text_block)
        return [imgs, texts], target

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.data_file)
