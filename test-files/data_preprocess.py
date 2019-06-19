import numpy as np
from numpy import genfromtxt
import pandas as pd

def preprocessData(source_dir, target_dir, seq_length=20):
    # Read csv file into a dataframe object:
    df = pd.read_csv('..' + source_dir, sep = ',', header = None)
    df = df.transpose()
    df.columns = ['frame_num', 'ped_id', 'y', 'x']
    # Convert first two columns from float to int
    df['frame_num'] = df.frame_num.astype(int)
    df['ped_id'] = df.ped_id.astype(int)
    # Sort dataframe based on ped_id and then by frame_num to match datasets used in the Pytorch implementation
    df = df.sort_values(by = ['ped_id', 'frame_num'])
    # Subtract 1 from frame_num column to match datasets used in the Pytorch implementation
    df = df - [1, 0, 0, 0]

    # Keep sequences of length 20
    gdf = df.groupby(['ped_id'])
    for ped_id, group in gdf:
        group_count = group['ped_id'].count()
        if group_count < seq_length:
            df = df.drop(group.index)
        elif group_count > seq_length:
            excess_count = group_count - seq_length
            df = df.drop((group.index[-excess_count:]))

    # Write resulting dataframe into target direction
    np.savetxt('..' + target_dir, df.values, fmt='%d %d %f %f')


data_dirs = ['/data/raw/eth/hotel/', '/data/raw/eth/univ/', '/data/raw/ucy/univ/', '/data/raw/ucy/zara/zara01/', '/data/raw/ucy/zara/zara02/']
target_names = ['hotel.txt', 'eth.txt', 'ucy.txt', 'zara01.txt', 'zara02.txt']
# Call preprocess on all 5 datasets
for idx, dir in enumerate(data_dirs):
    preprocessData(dir + 'pixel_pos.csv', dir + target_names[idx])
