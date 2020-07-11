import os
from train import train_main
from test import test_main
from configuration import load_parameter
from configuration import save_parameter


def main(hyper_parameter):
    if hyper_parameter['train'] is True:
        is_train = True
    else:
        is_train = False
    if hyper_parameter['test'] is True:
        is_test = True
    else:
        is_test = False
    if is_train is False and is_test is False:
        return

    # 获取超参
    hp = load_parameter(hyper_parameter['pre_parameter'], parameter_id=hyper_parameter['parameter_id'])

    # 训练模型
    if is_train is True:
        train_main(hp)

    # 测试模型
    if is_test is True:
        test_main(hp, save=False)

    # 保存超参
    save_parameter(hp['data_name'], "1", hp)


import model
if __name__ == '__main__':
    hyper_parameter = {}
    hyper_parameter['pre_parameter'] = False  # 是否使用保存的超参
    hyper_parameter['parameter_id'] = "1"  # 使用保存超参的编号

    hyper_parameter['pre_train'] = False  # 是否使用保存的模型
    hyper_parameter['model_id'] = "1"  # 使用保存模型的编号

    hyper_parameter['train'] = True  # 是否训练模型
    hyper_parameter['test'] = False  # 是否测试模型

    os.environ["CUDA_VISIBLE_DEVICES"] = "11"
    main(hyper_parameter)