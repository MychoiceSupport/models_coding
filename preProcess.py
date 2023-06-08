import os
import pandas as pd
from tqdm import tqdm


def readData(folder_path):
    df = pd.read_parquet(folder_path)
    return df


if __name__ == '__main__':
    for day in ["04", "05", "09", "10"]:
        ott = readData('../data/ott_202303{}'.format(day))
        UID = ott[['UID']]
        UID.drop_duplicates(inplace=True)

        directory = '../data/signal_202303{}'.format(day)
        files = os.listdir(directory)  # 获取目录中的所有文件
        signal_files = [file for file in files if file.endswith('.parquet')]

        i = 0
        for signal_file in tqdm(signal_files):
            # 逐批次读取Parquet数据
            signaling = readData(directory + '/' + signal_file)

            #获取UID出现过的数据
            merged_df = pd.merge(signaling, UID, on='UID')

            merged_df.to_csv('../data/signal03{}/signal_{}.csv'.format(day, i), index=False)

            i += 1


