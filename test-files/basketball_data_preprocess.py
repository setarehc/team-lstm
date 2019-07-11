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

def read_possession_file(file_name, num_of_ints, pos = 0):
    res_tuples =[]
    df = pd.DataFrame(columns=['ball', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'l1', 'l2', 'l3', 'l4', 'goal'])
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

def load_ints(fn, num, pos = 0):
    with open(fn, "rb") as f:
        f.seek(pos * bytes_per_int, 0)
        bytes = f.read(num * bytes_per_int)
        tup = struct.unpack("{}i".format(num), bytes)
    return np.array(tup)

path = '../data/raw_basketball/train/'
files_list = [f for f in listdir(path) if isfile(join(path, f))]

# Initialize the final data frame object
players_df = pd.DataFrame(columns = ['frame_num', 'player_id', 'y', 'x'])

# Set according to teh dataset description
max_num_frames = 70
max_num_players = 10
print(len(files_list))
for i, file in enumerate(files_list):
    print(i)
    res = read_possession_file(join(path, file), num_of_ints)
    df = pd.DataFrame(res, columns=['ball', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'l1', 'l2', 'l3', 'l4', 'goal'])
    # Delete unnecessary columns
    df = df.drop(['ball', 'l1', 'l2', 'l3', 'l4', 'goal'], axis=1)
    for j, player in enumerate(df.columns):
        num_of_frames = df[player].count()
        frame_num_col = i * max_num_frames + np.array(range(num_of_frames))
        player_id_col = i * max_num_players + np.full((num_of_frames), j)
        pos_y = np.floor(df[player].values / court_width)
        pos_x = df[player].values % court_width
        players_df = pd.concat([players_df, pd.DataFrame({'frame_num': frame_num_col, 'player_id': player_id_col, 'y': pos_y, 'x': pos_x}, columns=['frame_num', 'player_id', 'y', 'x'])])


target_dir =
np.savetxt('..' + target_dir, players_df.values, fmt='%d %d %f %f')
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