import argparse
import torch
import datetime
import json
import yaml
import os
import logging
import numpy as np
import pickle
from traj_dataset import MinMaxScaler, get_the_dataloader
from utils import *
from models import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


class MinMaxScaler():
    def fit(self, data):
        if type(data) != torch.Tensor:
            data = torch.from_numpy(data)
        self.data_max = torch.max(data, dim=0).values
        self.data_min = torch.min(data, dim=0).values

    def transform(self, data):
        if type(data) != torch.Tensor:
            data = torch.from_numpy(data)
        scaled_data = (data - self.data_min) / (self.data_max - self.data_min)
        return scaled_data

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, scaled_data):
        ori_data = scaled_data.cuda() * (self.data_max.cuda() - self.data_min.cuda()) + self.data_min.cuda()
        return ori_data


# class MinMaxScaler():
#     def fit(self, data):
#         data = data.astype(np.float64)
#         assert len(data.shape) <= 2 and len(data.shape) > 0
#         if type(data) == np.adarray:
#             self.data_max = np.max(data, axis = 0)
#             self.data_min = np.min(data, axis = 0)
#         elif type(data) == torch.Tensor:
#             self.data_max = torch.max(data, dim=0)
#             self.data_min = torch.min(data, dim=0)

#     def transform(self, data):
#         data = data.astype(np.float64)
#         scaled_data = (data - self.data_min) / (self.data_max - self.data_min)
#         return scaled_data

#     def fit_transform(self, data):
#         data = data.astype(np.float64)
#         self.fit(data)
#         return self.transform(data)

#     def inverse_transform(self, scaled_data):
#         try:
#             ori_data = scaled_data * (self.data_max - self.data_min) + self.data_min
#         except:
#             ori_data = scaled_data * (torch.tensor(self.data_max) - torch.tensor(self.data_min)) + torch.tensor(self.data_min)
#         return ori_data

scaler = pickle.load(open('../data/scaler.pkl', 'rb'))
print("check一下scaler信息", scaler.data_max, scaler.data_min)


def main(args):
    SEED = args.seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    X = np.load('../data/all_data_X.npy', allow_pickle=True)
    y = np.load('../data/all_data_y_origin.npy', allow_pickle=True)
    print("查看是否采样", args.sample)
    # print(X[0])
    # print(y[0])
    # # scaler = pickle.load(open('../data/scaler.pkl', 'rb'))
    time_delta = np.load('../data/all_time_delta.npy', allow_pickle=True)
    pre_time_feature = np.load('../data/time_start_end.npy', allow_pickle=True)
    lac_id = np.load('../data/lac_id.npy', allow_pickle=True)
    shape0 = pre_time_feature.shape[0]
    shape1 = pre_time_feature.shape[1]
    pre_time_feature = pre_time_feature.reshape(-1, 2)
    # start_feature = pd.to_datetime(pre_time_feature[:,0],unit='s').hour.values.reshape(shape0, shape1, -1).astype(np.int64)
    # end_feature = pd.to_datetime(pre_time_feature[:,1],unit='s').hour.values.reshape(shape0, shape1, -1).astype(np.int64)
    pre_start_feature = pd.to_datetime(pre_time_feature[:, 0], unit='s')
    pre_end_feature = pd.to_datetime(pre_time_feature[:, 1], unit='s')

    start_hour_feature = pre_start_feature.hour.values.reshape(shape0, shape1, 1, -1).astype(np.int64)
    start_week_feature = pre_start_feature.weekday.values.reshape(shape0, shape1, 1, -1).astype(np.float64)
    end_hour_feature = pre_end_feature.hour.values.reshape(shape0, shape1, 1, -1).astype(np.int64)
    end_week_feature = pre_end_feature.weekday.values.reshape(shape0, shape1, 1, -1).astype(np.float64)

    ####明天修改一下time_util全部，今天的目的是先跑上
    start_hour = np.concatenate((start_hour_feature, end_hour_feature), axis=2)
    end_hour = np.concatenate((start_week_feature, end_week_feature), axis=2)
    time_feature = np.concatenate((start_hour, end_hour), axis=-1)
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.sample == True:
        print("采样的算法")
        X = X[::10]
        y = y[::10]
        time_delta = time_delta[::10]
        time_feature = time_feature[::10]
        foldername = (
            "./save/{}_{}_{}_{}_sample".format(args.model_type, args.layers, args.batch_size, current_time)
        )
    else:
        foldername = (
            "./save/{}_{}_{}_{}".format(args.model_type, args.layers, args.batch_size, current_time)
        )

    if args.use_time == True:
        foldername = foldername + '_time_use{}'.format(args.dimension)
    if args.use_id == True:
        foldername = foldername + '_id_use'.format(args.channels)
    foldername = foldername + '_dim{}/'.format(args.channels)
    path = 'config/' + args.config
    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    config['seed'] = SEED

    #     print(json.dumps(config, indent=4))

    print('model folder:', foldername)
    os.makedirs(foldername, exist_ok=True)
    with open(foldername + "config.json", "w") as f:
        json.dump(config, f, indent=4)

    train_loader, val_loader, test_loader = get_the_dataloader(X, y, time_delta, time_feature, lac_id, args.sample,
                                                               args.batch_size, target_dim=args.input_dim,
                                                               device=args.device,
                                                               num_workers=args.num_workers
                                                               )
    del X, y, time_delta, time_feature
    if args.model_type == 'TCN':
        model = TCN(config['diffusion'], args, inputdim=2, seq_len=11, device=args.device).to(args.device)
    elif args.model_type == 'BiLSTM':
        model = BiLSTM(config['diffusion'], args, inputdim=2, seq_len=11, device=args.device).to(args.device)
    elif args.model_type == 'GRU':
        model = GRU(config['diffusion'], args, inputdim=2, seq_len=11, device=args.device).to(args.device)
    elif args.model_type == 'fc':
        model = FC(config['diffusion'], args, inputdim=2, seq_len=11, device=args.device).to(args.device)
    elif args.model_type == 'Att':
        model = Att(config['diffusion'], args, inputdim=2, seq_len=11, device=args.device).to(args.device)
    else:
        raise Exception('You should input the right model!')
    # model = nn.DataParallel(model, device_ids = [0,1])
    if args.modelfolder == '':
        train(
            model, args,
            config['train'],
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            scaler=scaler,
            foldername=foldername
        )
    else:
        model.load_state_dict(torch.load('./save/{}'.format(args.foldername) + "/model.pth", map_location=args.device))

    logging.basicConfig(filename=foldername + '/test_model.log', level=logging.DEBUG)
    logging.info('model_name = {}'.format(args.modelfolder))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model_Selection')
    parser.add_argument('--config', type=str, default='base.yaml')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu', help='Device for Attack')
    parser.add_argument('--num_workers', type=int, default=4, help='Device for Attack')
    parser.add_argument('--modelfolder', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_type', type=str, default='TCN')
    parser.add_argument('--channels', type=int, default=128)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--Bi', type=bool, default=False)
    parser.add_argument('--learning_rate', type=float, default=4e-4)
    parser.add_argument('--nheads', type=int, default=8)
    parser.add_argument('--timeemb', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--side_channel', type=int, default=64)
    parser.add_argument('--dimension', type=int, default=64)
    parser.add_argument('--input_dim', type=int, default=2)
    parser.add_argument('--sample', type=bool, default=False)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--use_time', type=bool, default=False)
    parser.add_argument('--time_type', type=str, default='Semantic')
    parser.add_argument('--use_id', type=bool, default=False)
    parser.add_argument('--id_emb', type=int, default=32)

    args = parser.parse_args()
    print(args)
    main(args)