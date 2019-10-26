from sacred import Ingredient
import os

common_ingredient = Ingredient('common')
dataset_ingredient = Ingredient('dataset')

@common_ingredient.config
def commonCfg():
    input_size = 2
    output_size = 5

    # RNN size parameter (dimension of the output/hidden state)
    rnn_size = 128  # size of RNN hidden state

    # Dimension of the embeddings parameter
    embedding_size = 64  # Embedding dimension for the spatial coordinates

    # Size of neighborhood to be considered parameter
    neighborhood_size = 32  # Neighborhood size to be considered for social grid

    # Size of the social grid parameter
    grid_size = 4   # Grid size of the social grid

    # Cuda parameter
    use_cuda = True  # Use GPU or not

    # GRU parameter
    gru = False  # True : GRU cell, False: LSTM cell

    # drive option
    drive = False   # Use Google drive or not

    # Model will be saved in this directory
    save_dir = 'model'

    validate = False

@common_ingredient.named_config
def debug():
    os.makedirs('/tmp/team_lstm_out', exist_ok=True)
    save_dir='/tmp/team_lstm_out'


@dataset_ingredient.config
def datasetCfg():
    dataset = 'basketball'; assert dataset != None
    train_dataset_path = 'data/basketball/train'; assert train_dataset_path != None
    test_dataset_path =  'data/basketball/test'; assert test_dataset_path != None
    # Length of sequence to be considered parameter
    seq_length = 10; assert seq_length != None  # RNN sequence length
    obs_length = 6; assert obs_length != None  # observation length
    pred_length = seq_length - obs_length; assert pred_length != None  # prediction length
    keep_every = 5; assert keep_every != None  # Keeps every keep_every entries of the input dataset (to recreate Kevin Murphy's work, needs be set to 5)
    orig_seq_len = 50; assert orig_seq_len != None  # Original dataset sequence length (ped_data = 20 and basketball_data = 50)
    persons_to_keep = None  # Indicates players to keep in the dataset

@dataset_ingredient.named_config
def basketball_total():
    dataset = 'basketball_total'
    train_dataset_path = 'data/basketball/total_train'
    test_dataset_path = 'data/basketball/total_test'
    seq_length = 10
    obs_length = 6
    keep_every = 5
    orig_seq_len = 50
    persons_to_keep = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  # Keep the ball and players 1-5

@dataset_ingredient.named_config
def basketball():
    dataset = 'basketball'
    train_dataset_path = 'data/basketball/train'
    test_dataset_path = 'data/basketball/test'
    seq_length = 10
    obs_length = 6
    keep_every = 5
    orig_seq_len = 50


@dataset_ingredient.named_config
def basketball_small():
    dataset = 'basketball_small'
    train_dataset_path = 'data/basketball/small_train'
    test_dataset_path = 'data/basketball/small_test'
    seq_length = 50
    obs_length = 20
    keep_every = 1
    orig_seq_len = 50


@dataset_ingredient.named_config
def original():
    dataset = 'original'
    train_dataset_path = 'data/original/train'
    test_dataset_path = 'data/original/test'
    seq_length = 20
    obs_length = 8
    keep_every = 1
    orig_seq_len = 20


class DotDict(dict):
    """
    https://stackoverflow.com/questions/13520421/recursive-dotdict/13520518
    a dictionary that supports dot notation 
    as well as dictionary access notation 
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            if isinstance(value, list):
                value = [x for x in value]
            self[key] = value