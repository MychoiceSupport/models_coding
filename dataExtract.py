import json
import os

import numpy as np
import pandas as pd
import math

from tqdm import tqdm
from geopy import distance

distance_threshold = 1000
continue_distance_threshold = 3000


def numpy_encoder(obj):
    if isinstance(obj, np.int64):
        return int(obj)
    raise TypeError("Object of type %s is not JSON serializable" % type(obj).__name__)


def getDistance2(lat1, lon1, lat2, lon2):
    if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
        return 10000
    try:
        d = distance.distance((lat1, lon1), (lat2, lon2)).m
    except ValueError:
        d = 10000
    except TypeError:
        d = 10000
    return d


def old_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    dis = R * c * 1000
    return dis


def isContain(time, sTime, eTime):
    return sTime <= time <= eTime


def dataExtract(signaling, ott):
    ottGroups = ott.groupby('UID')
    signalGroups = signaling.groupby('UID')

    ottGroupKeys = list(ottGroups.groups.keys())
    signalingGroupKeys = list(signalGroups.groups.keys())

    data4Later = []
    ottLen = len(ottGroupKeys)
    sigLen = len(signalingGroupKeys)

    x = 0
    y = 0
    with tqdm(total=min(ottLen, sigLen)) as pbar:
        while x < ottLen and y < sigLen:
            pbar.update(1)
            ott_group_data = ottGroups.get_group(ottGroupKeys[x])
            signal_group_data = signalGroups.get_group(signalingGroupKeys[y])

            ott_group_data.sort_values(by=['time'], ascending=True, inplace=True)
            signal_group_data.sort_values(by=['procedureStartTime', 'procedureEndTime'], ascending=True, inplace=True)

            ott_UID = ott_group_data.iloc[0]['UID']
            sig_UID = signal_group_data.iloc[0]['UID']
            if ott_UID < sig_UID:
                x += 1
                continue
            elif ott_UID > sig_UID:
                y += 1
                continue
            else:
                x += 1
                y += 1
                signaling4uid = signal_group_data

                ott4uid = ott_group_data

                cur = 5
                length = len(signaling4uid)
                for ottRow in ott4uid.itertuples():

                    while cur < length - 5:
                        sigRow = signaling4uid.iloc[cur]

                        if sigRow['procedureStartTime'] <= ottRow.time <= sigRow['procedureEndTime']:
                            drop = False
                            if getDistance2(ottRow.latitude, ottRow.longitude, sigRow['latitude'],
                                            sigRow['longitude']) > distance_threshold:
                                break
                            curData = dict()
                            curData['P'] = []
                            for i in range(cur - 5, cur + 6):
                                curSigRow = signaling4uid.iloc[i]
                                curData['P'].append({'CID': curSigRow['CID'],
                                                     'procedureStartTime': curSigRow['procedureStartTime'].strftime(
                                                         '%Y-%m-%d %H:%M:%S'),
                                                     'procedureEndTime': curSigRow['procedureEndTime'].strftime(
                                                         '%Y-%m-%d %H:%M:%S'),
                                                     'latitude': curSigRow['latitude'],
                                                     'longitude': curSigRow['longitude']})
                                if i > cur - 5:
                                    lastSigRow = signaling4uid.iloc[i - 1]
                                    if getDistance2(lastSigRow['latitude'], lastSigRow['longitude'], curSigRow['latitude'],
                                                curSigRow['longitude']) > continue_distance_threshold:
                                        drop = True
                            curData['Y'] = {'x': ottRow.latitude, 'y': ottRow.longitude,
                                            't': ottRow.time.strftime('%Y-%m-%d %H:%M:%S')}
                            curData['f'] = {'UID': ott_UID}
                            if not drop:
                                data4Later.append(curData)

                            break
                        elif sigRow['procedureStartTime'] > ottRow.time:

                            break
                        elif sigRow['procedureEndTime'] < ottRow.time:

                            cur += 1
    return data4Later


def readDataFromCsv(folder_path):
    # 获取文件夹中所有CSV文件的路径
    csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]

    # 创建一个空的DataFrame对象，用于存储所有数据
    all_data = pd.DataFrame()

    # 逐个读取CSV文件，并将它们连接到all_data中
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        all_data = pd.concat([all_data, df], ignore_index=True)

    # 返回连接后的DataFrame对象
    return all_data


def readDataFromParquet(folder_path):
    df = pd.read_parquet(folder_path)
    return df


if __name__ == '__main__':
    for day in ["04", "05", "09", "10"]:
        ott = readDataFromParquet('../data/ott_202303{}'.format(day))
        ott['time'] = pd.to_datetime(ott['time'], unit='s') + pd.Timedelta(hours=8)
        ott.sort_values(by=['UID', 'time'], ascending=True, inplace=True)
        ott['latitude'] = ott['latitude'] - 0.0055
        ott['longitude'] = ott['longitude'] - 0.0055

        signaling = readDataFromCsv('../data/signal03{}'.format(day))
        # process4Time
        signaling['procedureStartTime'] = pd.to_datetime(signaling['procedureStartTime'], unit='ms') + pd.Timedelta(hours=8)
        signaling['procedureEndTime'] = pd.to_datetime(signaling['procedureEndTime'], unit='ms') + pd.Timedelta(hours=8)
        signaling.sort_values(by=['UID', 'procedureStartTime', 'procedureEndTime'], ascending=True, inplace=True)
        signaling['latitude'] = signaling['latitude'] - 0.000055
        signaling['longitude'] = signaling['longitude'] - 0.000055

        ret = dataExtract(signaling=signaling, ott=ott)

        with open('../data/ottWithSignaling_03{}_1000.json'.format(day), 'w') as file:
            json.dump(ret, file, default=numpy_encoder)
