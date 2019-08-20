import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pandas as pd
import numpy as np
import itertools
import math

dataset_dimensions = {'hotel.txt': [720, 576], 'eth.txt': [720, 576],
                                   'ucy.txt': [720, 576], 'zara01.txt': [720, 576],
                                   'zara02.txt': [720, 576], 'biwi': [720, 576],
                                   'crowds': [720, 576], 'stanford': [595, 326],
                                   'mot': [768, 576], 'overfit': [768, 576],
                                   'basketball': [400, 360], 'dataloader': [768, 578]}

def convert_to_tensor(seq_data, persons_list):
    """
    Function that converts seq_data into a tensor of size (20, max_num_persons, 2) and a look-up table
    max_num_persons = maximum number of persons present in all of the frames of the sequence
    For people not present in a frame, x and y values are set to 0
    :return:
    tensor_seq_data: tensor sequence data of size (20, max_num_persons, 2)
    lookup_table: a table from person_id to the index in which it appears in the tensor_seq_data
    """
    # Get unique persons ids in the whole sequence
    unique_ids = pd.unique(list(itertools.chain.from_iterable(persons_list))).astype(int)

    # Create a lookup table which maps person_id to tensor_seq_data index
    lookup_table = dict(zip(unique_ids, range(0, len(unique_ids))))

    # Create the tensor seq_data
    tensor_seq_data = np.zeros(shape=(len(seq_data), len(lookup_table), 2))

    for ind, frame in enumerate(seq_data):
        corr_index = [lookup_table[x] for x in frame[:, 0]]
        tensor_seq_data[ind, corr_index, :] = frame[:, 1:3]

    return_arr = torch.from_numpy(np.array(tensor_seq_data)).float()

    return return_arr, lookup_table


class TrajectoryDataset(Dataset):

    def __init__(self, filename, seq_length=20, keep_every=1):

        self.filename = filename
        self.seq_length = seq_length  # Original dataset sequence length (ped_data = 20 and basketball_data = 50)
        self.keep_every = keep_every  # Keeps every keep_every entries of the input dataset (to recreate Kevin Murphy's work, needs be set to 5)

        print('Now processing: ', filename)

        self.frame_to_data, self.frame_list, self.frame_to_num_persons, self.frame_to_persons, self.orig_data, self.idx_to_person, self.person_to_frames = self.frame_preprocess()

    def frame_preprocess(self):
        '''
        Function that will pre-process the input data file
        :param
        filename: file path
        '''

        column_names = ['frame_num', 'person_id', 'y', 'x']

        # Copy of the original data
        orig_data = []

        df = pd.read_csv(self.filename, dtype={'frame_num': 'int', 'person_id': 'int'}, delimiter=' ', header=None,
                         names=column_names)
        # Sort dataframe based on ped_id and then by frame_num
        df = df.sort_values(by=['person_id', 'frame_num'])

        # Keep only the sequences of length self.seq_length
        gdf = df.groupby(['person_id'])
        for ped_id, group in gdf:
            group_count = group['person_id'].count()
            if group_count < self.seq_length:
                df = df.drop(group.index)
            elif group_count > self.seq_length:
                excess_count = group_count - self.seq_length
                df = df.drop((group.index[-excess_count:]))

        # Keep every self.keep_every entries of the input dataset
        del_list = [idx for idx in list(df.index.data) if idx % self.keep_every != 0]
        df = df.drop(df.index[del_list])

        target_ids = np.array(df.drop_duplicates(subset={'person_id'}, keep='first', inplace=False)['person_id'])

        # Dictionary from sequence/person index to person id
        idx_to_person = dict(zip(range(len(target_ids)), target_ids))

        data = np.array(df)

        # Dictionary from person id to frame list
        person_to_frames = {}
        for person in target_ids:
            person_to_frames[person] = data[data[:, 1] == person][:, 0]

        orig_data.append(data)

        frame_list = data[:, 0].tolist()
        # Remove duplicates from the frame_list
        frame_list = list(dict.fromkeys(frame_list))

        # Dictionary from frame number to a numpy array of size (num_persons, 3)
        # Array contains person_id, x, y for all persons in the frame
        frame_to_data = {}

        # Dictionary from frame number to list of person ids present in the frame
        frame_to_persons = {}

        # Dictionary from frame number to number of persons present in the frame
        frame_to_num_persons = {}

        for idx, frame in enumerate(frame_list):
            persons_data = data[data[:, 0] == frame][:, 1:]
            persons_list = persons_data[:, 0].tolist()

            # Extract person_id, x and y positions all persons present in the current frame
            persons_with_pos = persons_data[:, [0, 2, 1]]

            frame_to_data[frame] = persons_with_pos
            frame_to_persons[frame] = persons_list
            frame_to_num_persons[frame] = len(persons_list)

        return frame_to_data, frame_list, frame_to_num_persons, frame_to_persons, orig_data, idx_to_person, person_to_frames

    @property
    def target_ids(self):
        return list(self.person_to_frames.keys())

    def __getitem__(self, idx):
        """
        Function that returns sequence #idx in the dataset + dataset folder name
        :param
        idx: index of sequence to be returned
        """
        # Go from sequence/person index to person_id
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        target_id = self.idx_to_person[idx]
        frame_list = self.person_to_frames[target_id]

        seq_data = [self.frame_to_data[frame] for frame in frame_list]
        seq_persons_list = [self.frame_to_persons[frame] for frame in frame_list]
        seq_num_persons_list = [self.frame_to_num_persons[frame] for frame in frame_list]

        folder_name = self.filename

        return seq_data, seq_num_persons_list, seq_persons_list, folder_name

    def __len__(self):
        # Returns sequence length
        return len(self.person_to_frames)


# Test Block
if __name__ == '__main__':
    from os import listdir
    from os.path import isfile, join
    path = 'data/original/train/'
    files_list = [f for f in listdir(path) if isfile(join(path, f))]
    batch_size = 5
    all_datasets = ConcatDataset([TrajectoryDataset(join(path, file)) for file in files_list])

    train_loader = DataLoader(all_datasets, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False, collate_fn=lambda x: x)

    print(len(train_loader))

    for batch_idx, batch in enumerate(train_loader):
        if len(batch) < 5:
            print(batch_idx)


    d = TrajectoryDataset('data/original/train/hotel.txt', keep_every=2)
    batch_size = 5
    #assert batch_size == 1
    dataloader = DataLoader(d, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False, collate_fn=lambda x: x)

    print(len(dataloader))

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx == 24:
            for i in range(batch_size):
                tuple = batch[i]

