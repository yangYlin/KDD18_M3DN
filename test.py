from measure.measure_average_precision import average_precision
from measure.measure_hamming_loss import hamming_loss
from measure.measure_ranking_loss import ranking_loss
from measure.measure_example_auc import example_auc
from measure.measure_example_f1 import example_f1
from measure.measure_one_error import one_error
from measure.measure_macro_auc import macro_auc
from measure.measure_micro_auc import micro_auc
from measure.measure_macro_f1 import macro_f1
from measure.measure_micro_f1 import micro_f1
from measure.measure_coverage import coverage

from torch.autograd import Variable
import torch.utils.data as Data
from data_iapr import MyDataset
from model import load_model
import torch
import numpy as np


evaluation = ['average_precision', 'coverage', 'ranking_loss', 'macro_auc', 'micro_auc', 'example_auc',
              'hamming_loss', 'one_error', 'macro_f1', 'micro_f1', 'example_f1']  # 评价指标

view_name = ['Image', 'Text']


def test_single_view(view_id, x, y, hp):
    result = {}
    if 0 in hp['eval']:
        result[evaluation[0]] = average_precision(x, y)
    if 1 in hp['eval']:
        result[evaluation[1]] = coverage(x, y)
    if 2 in hp['eval']:
        result[evaluation[2]] = ranking_loss(x, y)
    if 3 in hp['eval']:
        result[evaluation[3]] = macro_auc(x, y)
    if 4 in hp['eval']:
        result[evaluation[4]] = micro_auc(x, y)
    if 5 in hp['eval']:
        result[evaluation[5]] = example_auc(x, y)
    if 6 in hp['eval']:
        if 'thread' in hp.keys():
            result[evaluation[6]] = hamming_loss(x, y, thread=hp['thread'])
        else:
            print('No thread for prediction(default = 0.5)!!')
            result[evaluation[6]] = hamming_loss(x, y)
    if 7 in hp['eval']:
        result[evaluation[7]] = one_error(x, y)
    if 8 in hp['eval']:
        if 'thread' in hp.keys():
            result[evaluation[8]] = macro_f1(x, y, thread=hp['thread'])
        else:
            print('No thread for prediction(default = 0.5)!!')
            result[evaluation[8]] = macro_f1(x, y)
    if 9 in hp['eval']:
        if 'thread' in hp.keys():
            result[evaluation[9]] = micro_f1(x, y, thread=hp['thread'])
        else:
            print('No thread for prediction(default = 0.5)!!')
            result[evaluation[9]] = micro_f1(x, y)
    if 10 in hp['eval']:
        if 'thread' in hp.keys():
            result[evaluation[0]] = example_f1(x, y, thread=hp['thread'])
        else:
            print('No thread for prediction(default = 0.5)!!')
            result[evaluation[0]] = example_f1(x, y)
    return result


def test_main(hp, models=None, save=False):
    print("----------start testing models----------")
    if models is None:
        models = load_model(hp['label'], pre_train=True, model_id="1")
    view_num = len(models)
    for i in range(view_num):
        models[i].cuda()
        models[i].eval()

    # calculate output
    test_data = MyDataset(train=False)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=32, num_workers=12)
    h = [[] for i in range(view_num)]
    label = [[] for i in range(view_num)]

    for step, (x, y) in enumerate(test_loader):
        for i in range(view_num):
            b_x = Variable(x[i], volatile=True).cuda()

            # forward
            hx = models[i](b_x)
            h[i].append(hx.cpu().data)
            label[i].append(y)

    for i in range(view_num):
        h[i] = torch.cat(h[i], dim=0)
        h[i] = h[i].numpy()
        label[i] = torch.cat(label[i], dim=0)
        label[i] = label[i].numpy()

    result = {}

    # test single view
    for i in range(view_num):
        # test
        result[view_name[i]] = test_single_view(i, h[i], label[i], hp)

        # show test result
        print("test result : ", view_name[i])
        for key in result[view_name[i]].keys():
            print(key, result[view_name[i]][key], '\n')

    # test all view
    h_average = []
    h_max = []
    for i in range(h[0].shape[0]):
        z = []
        for j in range(view_num):
            z.append(h[j][i].reshape(1, -1))
        z = np.concatenate(z)
        h_average.append(np.mean(z, axis=0).reshape(1, -1))
        h_max.append(np.max(z, axis=0).reshape(1, -1))
    h_average = np.concatenate(h_average, axis=0)
    h_max = np.concatenate(h_max, axis=0)

    result['avg'] = test_single_view(view_num, h_average, label[0], hp)
    print("test result : average all")
    for key in result['avg'].keys():
        print(key, result['avg'][key], '\n')

    result['max'] = test_single_view(view_num, h_max, label[0], hp)
    print("test result : max all")
    for key in result['max'].keys():
        print(key, result['max'][key], '\n')

    if save is True:
        for key in result.keys():
            path = './parameter/' + hp['data_name'] + '/result/result_' + key + '_1_.txt'
            with open(path, 'w') as f:
                for k in result[key].keys():
                    f.write(k + ' ' + str(result[key][k]) + '\n')
    print("----------end testing models----------")
