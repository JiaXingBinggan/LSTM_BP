import pandas as pd
import numpy as np
import datetime
import os
import argparse
import random
import models.Model as Model
import models.create_data as Data

import torch
import torch.nn as nn
import torch.utils.data

import logging
import sys

from config import Config

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_model(model_name, feature_nums, hidden_dims, bi_lstm, neuron_nums, dropout_rate):
    '''
        获取模型
        :param model_name: 模型名
        :param feature_nums: 特征数量，其等于look_back
        :param hidden_dims:  LSTM隐藏层数量
        :param bi_lstm: 是否使用双向LSTM
        :param neuron_nums: MLP的神经网络结构
        :return:
    '''
    return Model.RNN(feature_nums, hidden_dims, bi_lstm)


def get_dataset(data_path, dataset_name, feature_name, look_back):
    '''
        读取数据
        :param data_path: 数据根目录
        :param dataset_name: 数据名称
        :param feature_name: 指标名称
        :param look_back: 为几行数据作为为特征维度数量
        :return:
    '''
    # AEP,AP,ATSI,DAY,DEOK,DOM,DUQ,EKPC,MIDATL,NI
    data_path = data_path + dataset_name
    datas = pd.read_csv(data_path)[[feature_name]].values # 取出对应feature的数据

    # 归一化
    max_value = np.max(datas)
    min_value = np.min(datas)
    scalar = max_value - min_value
    datas = list(map(lambda x: x / scalar, datas))

    data_x = [] # 特征
    data_y = [] # 标签
    for i in range(len(datas) - look_back):
        data_x.append(datas[i:i + look_back])
        data_y.append(datas[i + look_back])

    return np.asarray(data_x).reshape(-1, 1, look_back), np.asarray(data_y).reshape(-1, 1, 1), scalar


def train(model, optimizer, data_loader, loss, device):
    '''
        训练函数
        :param model: pytorch模型类
        :param optimizer: 优化器
        :param data_loader: pytorch数据加载器
        :param loss: 损失
        :param device: 采用cpu还是gpu
        :return:
    '''
    model.train()  # 转换为训练模式
    total_loss = 0
    log_intervals = 0
    for features, labels in data_loader:
        features, labels = features.float().to(device), labels.to(device)
        y = model(features)

        train_loss = loss(y, labels.float())

        model.zero_grad()
        train_loss.backward()
        optimizer.step()
        total_loss += train_loss.item()  # 取张量tensor里的标量值，如果直接返回train_loss很可能会造成GPU out of memory

        log_intervals += 1

    return total_loss / log_intervals


def sub(model, features, device):
    '''
        评估函数
        :param model: pytorch模型类
        :param data_loader: pytorch数据加载器
        :param loss: 损失
        :param device: 采用cpu还是gpu
        :return:
    '''
    model.eval()

    y = model(features)

    return y


def main(args, logger):
    '''
        主函数
        :param args: 超参定义器
        :param logger: 日志句柄
        :return:
    '''
    device = torch.device(args.device)  # 指定运行设备

    data_x, data_y, scalar = get_dataset(args.data_path, args.dataset_name, args.feature_name, args.look_back)

    train_dataset = Data.generate_dataset(data_x, data_y)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, 
                                                    num_workers=args.num_workers)

    feature_nums = args.look_back
    model = get_model(args.model_name, feature_nums, args.hidden_dims, args.bi_lstm, 
                      args.neuron_nums, args.dropout_rate).to(device)
    logger.info(model)

    loss = nn.SmoothL1Loss()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_time = datetime.datetime.now()
    train_epoch_loss = []
    for epoch_i in range(args.epochs):
        torch.cuda.empty_cache()  # 清理无用的cuda中间变量缓存

        train_average_loss = train(model, optimizer, train_data_loader, loss, device) # 训练
        train_epoch_loss.append(train_average_loss)

        train_end_time = datetime.datetime.now()

        if epoch_i % args.print_interval == 0: # 打印结果
            logger.info('feature {}, model {}, epoch {}, train_{}_loss {}, '
                        '[{}s]'.format(args.feature_name, args.model_name, epoch_i,
                                         args.loss_type, train_average_loss, (train_end_time - start_time).seconds))

    torch.save(model.state_dict(), os.path.join(args.save_param_dir,
                                                args.model_name + '_best_' + args.loss_type + '.pth')) # 存储参数

    predict_data = torch.Tensor(data_x[-1]).unsqueeze(1).to(device)
    predict_feature = sub(model, predict_data, device).item() * scalar
    print(predict_feature)
    return predict_feature


# 用于预训练传统预测点击率模型
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default=Config.data_path)
    parser.add_argument('--dataset_name', default='data.csv', help='dataset')
    parser.add_argument('--model_name', default=Config.model_name, help='RNN')
    parser.add_argument('--neuron_nums', type=list, default=Config.neuron_nums)
    parser.add_argument('--dropout_rate', type=float, default=Config.dropout_rate)
    parser.add_argument('--num_workers', default=Config.num_workers, help='4, 8, 16, 32')
    parser.add_argument('--hidden_dims', default=Config.hidden_dims)
    parser.add_argument('--bi_lstm', default=Config.bi_lstm, help='1, 2')
    parser.add_argument('--look_back', default=Config.look_back, help='以几行数据为特征维度数量')
    parser.add_argument('--feature_name', default='X1(GDP)', help='X1(GDP),X2(全市人均可以配收入),'
                                                           'X3（社会消费品零售总额）,X4（一产值）,'
                                                           'X5（二产值）,X6（三产值）,'
                                                           'Y（货运量）')
    parser.add_argument('--epochs', type=int, default=Config.epochs)
    parser.add_argument('--lr', type=float, default=Config.lr)
    parser.add_argument('--weight_decay', type=float, default=Config.weight_decay)
    parser.add_argument('--batch_size', type=int, default=Config.batch_size)
    parser.add_argument('--print_interval', type=int, default=Config.print_interval)
    parser.add_argument('--device', default=Config.device)

    parser.add_argument('--loss_type', type=str, default='smoothl1loss', help='smoothl1loss')
    parser.add_argument('--save_log_dir', default=Config.save_log_dir)
    parser.add_argument('--save_res_dir', default=Config.save_res_dir)
    parser.add_argument('--save_param_dir', default=Config.save_param_dir)

    args = parser.parse_args()

    # 设置随机数种子
    setup_seed(Config.seed)

    if not os.path.exists(args.save_log_dir):
        os.mkdir(args.save_log_dir)

    if not os.path.exists(args.save_res_dir):
        os.mkdir(args.save_res_dir)

    if not os.path.exists(args.save_param_dir):
        os.mkdir(args.save_param_dir)

    logging.basicConfig(level=logging.DEBUG,
                        filename=args.save_log_dir + args.model_name + '_output.log',
                        datefmt='%Y/%m/%d %H:%M:%S',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    logger.info('===> start training!  ')
    current_feature_output_dicts = dict.fromkeys(tuple('X1(GDP),X2(全市人均可以配收入),'
                                                    'X3（社会消费品零售总额）,'
                                                    'X4（一产值）,X5（二产值）,'
                                                    'X6（三产值）,'.split(',')))

    features = []
    for feature_name in current_feature_output_dicts.keys():
        logger.info('===> now excuate the feature {}  '.format(feature_name))
        args.feature_name = feature_name
        features.append(main(args, logger))

    print(features)



