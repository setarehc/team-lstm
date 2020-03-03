import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pandas as pd
import numpy as np
import itertools

dataset_dimensions = {'hotel.txt': [720, 576], 'eth.txt': [720, 576],
                                   'ucy.txt': [720, 576], 'zara01.txt': [720, 576],
                                   'zara02.txt': [720, 576], 'biwi': [720, 576],
                                   'crowds': [720, 576], 'stanford': [595, 326],
                                   'mot': [768, 576], 'overfit': [768, 576],
                                   'basketball': [400, 360], 'dataloader': [768, 578]}

def convertToTensor(seq_data, persons_list):
    """
    Function that converts seq_data into a tensor of size (seq_len, max_num_persons, 2) and a look-up table
    max_num_persons = maximum number of persons present in all of the frames of the sequence
    For people not present in a frame, x and y values are set to 0
    :return:
    tensor_seq_data: tensor sequence data of size (seq_len, max_num_persons, 2)
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

def tensorizeData(all_xy_posns, all_ids):
    """
    Fucntion that converts data of a batch into two tensors; positions (seq_len, batch_size, max_num_persons, 2) and mask (seq_len, batch_size, max_num_persons)
    max_num_persons = maximum number of persons in all of the frames of all sequences in the batch
    For people not present in a frame, x and y values are set to 0
    :input:
    all_xy_posns: a list of length batch_size where each element is a seq_data (a list of length seq_length) where each element is an array of positions of persons 
    present in that particular frame with size (num_persons, 3)
    all-ids: a list of ids of persons present in all sequences in a batch
    :return:
    tensor_xy_posns: padded positions of all persons across frames of a batch(seq_len, batch_size, max_num_persons, 2)
    mask: a (seq_len, batch_size, max_num_persons) binary tensor; mask[f, b, i] = 1 means persons with index=i is present in frame f of batch b
    """
    # Get unique persons ids in the whole sequence
    unique_ids = pd.unique(list(itertools.chain.from_iterable(all_ids))).astype(int)

    # Create a lookup table which maps person_id to tensor_seq_data index
    lookup_dict = dict(zip(unique_ids, range(0, len(unique_ids))))

    # Create the tensor seq_data
    seq_length = len(all_xy_posns[0])
    batch_size = len(all_xy_posns)
    max_num_persons = len(lookup_dict)
    dim = 2
    tensor_xy_posns = np.zeros(shape=(seq_length, batch_size, max_num_persons, 2))
    mask = np.zeros(shape=(seq_length, batch_size, max_num_persons))

    # Set values of all_xy_posns and mask
    for batch_idx, batch in enumerate(all_xy_posns):
        for frame_idx, frame in enumerate(batch):
            corr_index = [lookup_dict[x] for x in frame[:, 0]]
            tensor_xy_posns[frame_idx, batch_idx, corr_index, :] = frame[:, 1:3]
            mask[frame_idx, batch_idx, corr_index] = 1
    #import pdb; pdb.set_trace()
    final_tensor_xy_posns = torch.from_numpy(np.array(tensor_xy_posns)).float()
    final_mask = torch.from_numpy(np.array(mask)).float()

    return final_tensor_xy_posns, final_mask, lookup_dict


class TrajectoryDataset(Dataset):

    def __init__(self, folder_path, seq_length=20, keep_every=1, persons_to_keep=None):

        self.folder_path = folder_path
        self.seq_length = seq_length  # Original dataset sequence length (ped_data = 20 and basketball_data = 50)
        self.keep_every = keep_every  # Keeps every keep_every entries of the input dataset (to recreate Kevin Murphy's work, needs be set to 5)
        self.persons_to_keep = persons_to_keep  # Keeps only persons i where persons_to_keep[i]=1 (to recreate Kevin Murphy's work, needs be set to [1,1,1,1,1,1,0,0,0,0,0])

        print('Now processing: ', folder_path)

        self.frame_to_data, self.frame_list, self.frame_to_num_persons, self.frame_to_persons, self.orig_data, self.idx_to_person, self.person_to_frames = self.frame_preprocess()

    def normalize_data(self, df):
        normalized_df = df
        normalized_df['y'] = df['y'] - df['y'].mean()
        normalized_df['y'] = normalized_df['y'].div(df['y'].std())
        normalized_df['x'] = df['x'] - df['x'].mean()
        normalized_df['x'] = normalized_df['x'].div(df['x'].std())
        return normalized_df

    def frame_preprocess(self):
        '''
        Function that will pre-process the input data file
        '''

        column_names = ['frame_num', 'person_id', 'y', 'x']

        # Copy of the original data
        orig_data = []

        df = pd.read_csv(self.folder_path, dtype={'frame_num': 'int', 'person_id': 'int'}, delimiter=' ', header=None,
                         names=column_names)
        # Sort dataframe based on ped_id and then by frame_num
        df = df.sort_values(by=['person_id', 'frame_num'])

        # *Kevin Murphy's*
        # Keep only the self.persons_to_keep persons in the dataset (for total_train and total_test in basketball dataset)
        if self.persons_to_keep is not None:
            gdf = df.groupby(['person_id'])
            for person_id, group in gdf:
                if self.persons_to_keep[person_id % 11] == 0: # TODO have a variable instead of 11
                    df = df.drop(group.index)

        # Normalize x and y values for basketball dataset
        if self.folder_path.split('/')[-3] == 'basketball':
            df['y'] = df['y'].div(dataset_dimensions['basketball'][0])
            df['x'] = df['x'].div(dataset_dimensions['basketball'][1])

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
        if self.keep_every > 1:
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

        folder_path = self.folder_path

        return seq_data, seq_num_persons_list, seq_persons_list, folder_path

    def __len__(self):
        # Returns sequence length
        return len(self.person_to_frames)


# Test Block
if __name__ == '__main__':
    '''
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
    '''
    dset = TrajectoryDataset('data/original/train/hotel.txt')
    batch_size = 5
    #assert batch_size == 1
    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False, collate_fn=lambda x: x)

    print(len(dataloader))
    print(len(dataloader.dataset))

    for idx, batch in enumerate(dataloader):
        if idx != len(batch):
            x_seq, numPedsList_seq, PedsList_seq, folder_path = batch[0]

