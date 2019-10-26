import numpy as np
from numpy import genfromtxt
import pandas as pd
import struct
from os import listdir
from os.path import isfile, join


# Globals:
court_width = 360
court_height = 400
bytes_per_int = 4
num_of_ints = 16

def readPossessionFile(file_name, num_of_ints, pos = 0):
    res_tuples =[]
    f = open(file_name, 'rb')
    # Change the current file position to the last byte
    f.seek(0, 2)
    # Set byte count as size of file
    file_size = f.tell()
    # Return current file position to the first byte
    f.seek(0, 0)
    while file_size - f.tell() > 0:
        bytes = f.read(num_of_ints * bytes_per_int)
        tuple = struct.unpack("{}i".format(num_of_ints), bytes)
        res_tuples.append(list(tuple))
        pos += len(tuple)
        f.seek(pos * bytes_per_int, 0)
    f.close()
    return res_tuples


def loadIntegers(fn, num, pos = 0):
    with open(fn, "rb") as f:
        f.seek(pos * bytes_per_int, 0)
        bytes = f.read(num * bytes_per_int)
        tup = struct.unpack("{}i".format(num), bytes)
    return np.array(tup)


path = 'data/raw_basketball/train/'
files_list = [f for f in listdir(path) if isfile(join(path, f))]

# Set according to teh dataset description
max_num_frames = 70
max_num_players = 10
num_players_plus_ball = 11

target_dir = 'data/basketball/total_train/basketball_total_train_'

'''
# Get the minimum length of possessions:
import re
min_length = max_num_frames
max_length = max_num_frames
test_files = [f for f in listdir('../data/raw_basketball/eval/') if isfile(join(path, f))]
train_files = files_list
for i, file in enumerate(train_files):
    m = re.search('len-(.+?).bin', file)
    if m:
        seq_length = int(m.group(1))
        if seq_length < min_length:
            min_length = seq_length
        if seq_length > max_length:
            max_length = seq_length
print(min_length, max_length)
'''


for i, file in enumerate(files_list):
    # Read current file
    res = readPossessionFile(join(path, file), num_of_ints)
    df = pd.DataFrame(res, columns=['ball', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'l1', 'l2', 'l3', 'l4', 'goal'])
    # Delete unnecessary columns
    df = df.drop(['l1', 'l2', 'l3', 'l4', 'goal'], axis=1)
    # Initialize the final data frame object
    players_df = pd.DataFrame(columns=['frame_num', 'player_id', 'y', 'x'])
    # Set frame_num, player_id, x, y values
    for j, player in enumerate(df.columns):
        num_of_frames = df[player].count()
        frame_num_col = i * max_num_frames + np.array(range(num_of_frames))
        player_id_col = i * num_players_plus_ball + np.full((num_of_frames), j)
        pos_y = np.floor(df[player].values / court_width)
        pos_x = df[player].values % court_width
        current_df = pd.DataFrame({'frame_num': frame_num_col, 'player_id': player_id_col, 'y': pos_y, 'x': pos_x}, columns=['frame_num', 'player_id', 'y', 'x'])
        players_df = pd.concat([players_df, current_df])
    # Keep sequences of length 50
    players_df.index = range(players_df.shape[0])
    seq_length = 50
    gdf = players_df.groupby(['player_id'])
    for player_id, group in gdf:
        group_count = group['player_id'].count()
        if group_count < seq_length:
            players_df = players_df.drop(group.index)
        elif group_count > seq_length:
            excess_count = group_count - seq_length
            players_df = players_df.drop((group.index[-excess_count:]))

    np.savetxt(target_dir+str(i)+'.txt', players_df.values, fmt='%d %d %f %f')


'''
res = read_possession_file('../data/raw_basketball/train/poss-0_len-70.bin', num_of_ints)
df = pd.DataFrame(res, columns=['ball', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'l1', 'l2', 'l3', 'l4', 'goal'])
# Delete unnecessary columns
df = df.drop(['ball', 'l1', 'l2', 'l3', 'l4', 'goal'], axis=1)
players_df = pd.DataFrame(columns = ['frame_num', 'player_id', 'y', 'x'])
for i, player in enumerate(df.columns):
    num_of_frames = df[player].count()
    frame_num_col = np.array(range(num_of_frames))
    player_id_col = np.full((num_of_frames), i)
    pos_y = np.floor(df[player].values / court_width)
    pos_x = df[player].values % court_width
    players_df = pd.concat([players_df, pd.DataFrame({'frame_num': frame_num_col, 'player_id': player_id_col, 'y': pos_y, 'x': pos_x}, columns=['frame_num', 'player_id', 'y', 'x'])])
'''