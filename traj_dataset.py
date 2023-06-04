import pickle
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
import torch
from tqdm import tqdm
# from sklearn.preprocessing import MinMaxScaler
import os
# from time_util import *
import sklearn
# from utils import *
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

def generate_raw_data(file_name,i):
    with open('data/{}'.format(file_name),'r') as f:
        all_data = json.load(f)
    """
    data中的字段:
        -laccell:基站ID
        -latitude:基站纬度
        -longtitude：基站精度
        -procedureStartTime: 进入时间
        -procedureEndTime:离开时间
    目前按照每个用户的序列长度为11进行金酸
        感觉停留时间也可以作为特征：
        - deltatime:停留时间
    """
    X = []
    Y = []
    dis = [[] for i in range(11)]
    for data in tqdm(all_data):
        data_traj = data['P']
        data_raw_label = data['Y']
        data_after_label = [np.float64(data_raw_label['x']), np.float64(data_raw_label['y'])]
        data_id = data['f']['imsi']
        data_slice = []
        for i,traj_point in enumerate(data_traj):
            laccell = traj_point['laccell']
            start_time = pd.to_datetime(traj_point['procedureStartTime'], format='%Y-%m-%d %H:%M:%S')
            end_time = pd.to_datetime(traj_point['procedureEndTime'], format='%Y-%m-%d %H:%M:%S')
            start_time = pd.to_datetime(start_time, unit='s')
            end_time = pd.to_datetime(end_time, unit='s')
            latitude = np.float64(traj_point['latitude'])
            longtitude = np.float64(traj_point['longitude'])
            # dis[i].append(geodesic([latitude, longtitude], data_after_label))
            delta_time = (end_time - start_time).value/ pd.Timedelta(24,unit='H').value
            #print("查看delta_time:",delta_time.value)
            # print("查看当前delta_time:", delta_time)
            data_slice.append([np.float64(latitude), np.float64(longtitude), data_id, delta_time, laccell, start_time, end_time])
        # print(np.mean(dis))
        X.append(data_slice)
        Y.append(data_after_label)
    # for j in range(11):
    #     print("最后的结果是:",j, np.mean(dis[i]))
    X = np.array(X)
    Y = np.array(Y)
    np.save('data/all_data_X_origin_{}.npy'.format(i),X)
    np.save('data/all_data_y_origin_{}.npy'.format(i),Y)
    # print("查看构造的数据的形状:",X.shape, Y.shape)  ###generate_raw_data
    del X,Y,dis,all_data


class MinMaxScaler():
    def fit(self,data):
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

def generate_training_info(X, y, dimention):
    time_delta = X[:,:,3]
    time_start_end = X[:,:,5:7]
    # print(time_start_end)
    get_shape = time_start_end.shape
    time_start_end_0 = pd.Series(pd.to_datetime(time_start_end[:,:,0].reshape(-1,)))
    time_start_end_0 = time_start_end_0.apply(lambda x: (x - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s'))
    time_start_end_1 = pd.Series(pd.to_datetime(time_start_end[:,:,1].reshape(-1,)))
    time_start_end_1 = time_start_end_1.apply(lambda x: (x - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s'))
    tryit = SemanticEncoder(24, 'cpu')
    time_start_end_0 = tryit(time_start_end_0.to_numpy()).detach().numpy()
    time_start_end_1 = tryit(time_start_end_1.to_numpy()).detach().numpy()
    time_start_end_1 = time_start_end_1.reshape(get_shape[0], get_shape[1], -1)
    time_start_end_0 = time_start_end_0.reshape(get_shape[0], get_shape[1], -1)
    time_feature = np.concatenate((time_start_end_0, time_start_end_1), axis = -1).astype(np.float64)
    time_delta = time_delta.astype(np.float64)
    print("查看time_feature的形状:",time_feature.shape)

    np.save('../data/all_time_delta.npy',time_delta)
    np.save('../data/time_start_end.npy',time_feature)

def generate_scaler():
    X = np.load('../data/all_data_X_origin.npy',allow_pickle=True)
    X = X[:,:,:2].astype(np.float64)
    np.save('../data/all_data_X_pre.npy',X)
    print("略微查看坐标:",X[0])
    shape = X.shape
    y = np.load('../data/all_data_y_origin.npy', allow_pickle=True)
    print("查看一下OTT",y[0])
    train_index = np.load('../data/train_index.npy', allow_pickle=True)
    print(train_index, len(train_index))
    train_data = X[train_index].reshape(-1,2)
    train_data = train_data.astype('float64')
    scaler = MinMaxScaler()
    scaler.fit(train_data)
    X = scaler.transform(X.reshape(-1,2)).reshape(shape)
    y = scaler.transform(y)
    print(X[0])
    np.save('../data/all_data_X.npy',X)
    np.save('../data/all_data_y.npy',y)
    with open('../data/scaler.pkl','wb') as f:
        pickle.dump(scaler,f)

# scaler = pickle.load(open('data/scaler.pkl', 'rb'))
# time_delta = np.load('data/all_time_delta.npy', allow_pickle=True)
# time_feature = np.load('data/time_start_end.npy', allow_pickle=True)
# print(time_feature)



class traj_data(Dataset):
    def __init__(self,  X, y, time_delta, time_feature, data_length=11, device = 'cpu',target_dim=2, mode = 'train', load_from_exist = False):
        self.seq_length = data_length
        self.target_dim = target_dim
        self.device = device
        #X, y= generate_raw_data()
        #
        # X = X[0::10]
        # y = y_origin[0::10]
        # time_feature = np.load('../data/time_start_end.npy', allow_pickle=True)
        # print("查看current_Data:", train_len)
        # self.scaler = scaler
        if mode == 'train':
            train_index = np.load('../data/train_index.npy',allow_pickle=True)
            current_data = X[train_index]
            current_label = y[train_index]
            current_time_delta = time_delta[train_index]
            current_time_feature = time_feature[train_index]
        elif mode == 'val':
            val_index = np.load('../data/val_index.npy',allow_pickle=True)
            current_data = X[val_index]
            current_label = y[val_index]
            current_time_delta = time_delta[val_index]
            current_time_feature = time_feature[val_index]
        elif mode == 'test':
            test_index = np.load('../data/test_index.npy',allow_pickle=True)
            current_data = X[test_index]
            current_label = y[test_index]
            current_time_delta = time_delta[test_index]
            current_time_feature = time_feature[test_index]
            # print("查看test的内容:",current_data)
        else:
            raise Exception('You should get the right mode')
        ###

        self.observed_data = current_data
        self.ground_truth_data = current_label
        self.time_delta = current_time_delta.astype(np.float64)
        self.time_start_end = current_time_feature.astype(np.float64)
        # self.origin_ground_truth = origin.astype(np.float64)
        # self.extra_features = current_data_extra


    def __getitem__(self, index):
        observed_data = self.observed_data[index]
        ground_truth = self.ground_truth_data[index]
        time_delta = self.time_delta[index]
        time_start_end = self.time_start_end[index]
        # extra_features = self.extra_features[index]
        sequence_data = (
            observed_data, ground_truth, time_delta, time_start_end
        )
        return sequence_data

    def __len__(self):
        return len(self.observed_data)

def get_the_dataloader(X,y,time_delta,time_feature, batch_size, target_dim = 2, device='cpu', num_workers = 0):
    train_dataset = traj_data(X,y,time_delta,time_feature, mode='train',target_dim=target_dim, device=device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
    val_dataset = traj_data(X,y,time_delta,time_feature,mode='val', target_dim=target_dim, device=device)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
    test_dataset = traj_data(X,y,time_delta,time_feature,mode='test', target_dim=target_dim, device=device)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
    del train_dataset, val_dataset, test_dataset
    # scaler = train_dataset.scaler
    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    # pass
    generate_raw_data('ottWithSignaling.json','0305')
    # generate_raw_data('ottWithSignaling_0306_1000.json','0306')
    # generate_raw_data('ottWithSignaling_0307_1000.json','0307')
    # generate_raw_data('ottWithSignaling_0308_1000.json','0308')
    # generate_raw_data('ottWithSignaling_0309_1000.json','0309')
    # generate_raw_data('ottWithSignaling_0310_1000.json','0310')
    # X_ori = np.load('../data/all_data_X_origin.npy', allow_pickle=True)
    # y = np.load('../data/all_data_y_origin.npy', allow_pickle=True)
    print("完成读取")
    # generate_training_info(X_ori,y)
    # import random
    # random.seed(1)
    # data_index = list([int(i) for i in range(len(X_ori))])
    # random.shuffle(data_index)
    # # print(index)
    # new_index = np.array(data_index)
    # # print(new_index)
    # train_len = int(0.8 * len(new_index))
    # test_len = int(0.1 * len(new_index))
    # valid_len = len(new_index) - train_len - test_len
    # train_index = new_index[:train_len]
    # test_index = new_index[-test_len:]
    # val_index = new_index[train_len: train_len + valid_len]
    # # # print(val_index)
    # # print("进入baocun")
    # np.save('../data/train_index.npy', train_index)
    # np.save('../data/test_index.npy', test_index)
    # np.save('../data/val_index.npy', val_index)
    # generate_scaler()
    # generate_training_info(X_ori,y,24)
    print("完成")