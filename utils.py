import logging
import numpy as np
import torch
# from geopy.distance import geodesic
from math import sin, asin, cos, radians, fabs, sqrt, atan2, sqrt
from torch.optim import Adam
import pickle
from tqdm import tqdm
import torch.nn as nn

EARTH_RADIUS = 6371393  # 地球平均半径，6371km


def geodesic(pos1, pos2):
    """用haversine公式计算球面两点间的距离。"""
    # 经纬度转换成弧度
    # lat0 = radians(pos1[0])
    # lat1 = radians(pos2[0])
    # lng0 = radians(pos1[1])
    # lng1 = radians(pos2[1])
    ####
    lat0 = torch.deg2rad(pos1[:, 0])
    lat1 = torch.deg2rad(pos2[:, 0])
    lng0 = torch.deg2rad(pos1[:, 1])
    lng1 = torch.deg2rad(pos2[:, 1])
    dLat = lat1 - lat0
    dLon = lng1 - lng0
    a = torch.sin(dLat / 2) * torch.sin(dLat / 2) + torch.sin(dLon / 2) * torch.sin(dLon / 2) * torch.cos(
        lat0) * torch.cos(lat1)
    distance = 2 * EARTH_RADIUS * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    return distance


def pre_get_metrics(x, y):
    assert len(x) == len(y)
    dis = []
    counter = len(x)
    count = 0
    for i in range(len(x)):
        tmp_dis = geodesic(x[i], y[i])
        dis.append(tmp_dis)
        if tmp_dis <= 150:
            count += 1
    dis = torch.tensor(dis, requires_grad=True)
    rmse = torch.sqrt(torch.mean(torch.pow(dis, 2)))
    mae = torch.mean(dis)
    rate_150 = count / counter * 100
    print("get_the_result:{},{},{}".format(rmse, mae, rate_150))
    return rmse, mae, rate_150


def get_metrics(x, y):
    assert len(x) == len(y)
    dis = []
    counter = len(x)
    count = 0
    dis = geodesic(x, y)
    count = len(dis[dis <= 150.0])
    dis = torch.tensor(dis, requires_grad=True)
    rmse = torch.sqrt(torch.mean(torch.pow(dis, 2)))
    mae = torch.mean(dis)
    rate_150 = count / counter * 100
    print("get_the_result:{},{},{}".format(rmse, mae, rate_150))
    return rmse, mae, rate_150


def train(model, args, config, train_loader, val_loader, test_loader, scaler=None, foldername=""):
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-6)
    is_lr_decay = config['is_lr_decay']
    if foldername != "":
        output_path = foldername + '/model.pth'
        logging.basicConfig(filename=foldername + '/train_model.log', level=logging.DEBUG)
    if is_lr_decay:
        p1 = int(0.75 * config["epochs"])
        p2 = int(0.9 * config["epochs"])
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[p1, p2], gamma=args.gamma
        )

    best_valid_loss = 1e10
    valid_epoch_interval = 1
    loss_func = nn.MSELoss()
    for epoch in range(args.epoch):
        model.train()
        avg_loss_valid = 0
        all_eval_points = []
        all_prediction = []
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch, data in enumerate(it, start=1):
                (_, ground_truth, _, _, _) = data
                ground_truth = ground_truth.cuda()
                optimizer.zero_grad()
                prediction = model(data)
                prediction = scaler.inverse_transform(prediction).cuda()
                # ground_truth = scaler.inverse_transform(ground_truth)
                # print("查看相关",prediction.shape, ground_truth)
                # loss_func = nn.MSELoss()
                # print(prediction.shape)
                loss = loss_func(prediction, ground_truth)
                loss.backward()
                avg_loss_valid += loss.item()
                all_prediction.append(prediction)
                all_eval_points.append(ground_truth)
                it.set_postfix(
                    ordered_dict={
                        'avg_epoch_loss': avg_loss_valid / batch,
                        'epoch': epoch, },
                    refresh=False,
                )
                optimizer.step()
            all_prediction = torch.cat(all_prediction, dim=0)
            all_eval_points = torch.cat(all_eval_points, dim=0)
            get_metrics(all_prediction, all_eval_points)
            logging.info('valid_avg_epoch_loss' + str(avg_loss_valid / batch) + ", epoch:" + str(epoch))
            if is_lr_decay:
                lr_scheduler.step()

        if val_loader is not None and (epoch + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            all_eval_points = []
            all_prediction = []
            with tqdm(val_loader, mininterval=5.0, maxinterval=50.0) as it:
                for batch, data in enumerate(it, start=1):
                    with torch.no_grad():
                        optimizer.zero_grad()
                        _, ground_truth, _, _, _ = data
                        ground_truth = ground_truth.cuda()
                        prediction = model(data)
                        prediction = scaler.inverse_transform(prediction).cuda()
                        # ground_truth = scaler.inverse_transform(ground_truth)
                        # todo：loss_func = nn.MSELoss()
                        loss = loss_func(prediction, ground_truth)
                        avg_loss_valid += loss.item()
                        all_prediction.append(prediction)
                        all_eval_points.append(ground_truth)
                        it.set_postfix(
                            ordered_dict={
                                'valid_avg_epoch_loss': avg_loss_valid / batch,
                                'epoch': epoch, },
                            refresh=False,
                        )
                logging.info('valid_avg_epoch_loss' + str(avg_loss_valid / batch) + ", epoch:" + str(epoch))
            all_prediction = torch.cat(all_prediction, dim=0)
            all_eval_points = torch.cat(all_eval_points, dim=0)
            rmse, mae, rating = get_metrics(all_prediction, all_eval_points)
            if best_valid_loss > mae:
                best_valid_loss = mae
                print(
                    '\n best loss is updated to ',
                    avg_loss_valid / batch,
                    'at', epoch
                )
                evaluate(model, test_loader, scaler, foldername)
                if foldername != '':
                    torch.save(model.state_dict(), output_path)

    evaluate(model, test_loader, scaler, foldername)


def evaluate(model, test_loader, scaler, foldername=""):
    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        evalpoints_total = 0

        all_observed_point = []
        all_evalpoint = []

        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                output = model(test_batch)
                predict = scaler.inverse_transform(output).cuda()
                (_, groundTruth, _, _, _) = test_batch
                groundTruth = groundTruth.cuda()
                # groundTruth = scaler.inverse_transform(groundTruth)
                all_observed_point.append(groundTruth)
                all_evalpoint.append(predict)
                evalpoints_total += 1
            with open(
                    foldername + "/generated_outputs_nsample.pk", "wb"
            ) as f:
                all_evalpoint = torch.cat(all_evalpoint, dim=0)
                all_observed_point = torch.cat(all_observed_point, dim=0)
                rmse_total, mae_total, rating = get_metrics(all_evalpoint, all_observed_point)
                pickle.dump(
                    [
                        all_evalpoint,
                        all_observed_point,
                        scaler,
                    ],
                    f,
                )

            with open(
                    foldername + "/result_nsample.pk", "wb"
            ) as f:
                pickle.dump(
                    [
                        rmse_total, mae_total, rating
                    ],
                    f,
                )
                print("RMSE:", rmse_total)
                print("MAE:", mae_total)
                logging.info("RMSE={}".format(rmse_total))
                logging.info("MAE={}".format(mae_total))
                print("rate_150:", rating)
                logging.info("rate_150={}".format(rating))