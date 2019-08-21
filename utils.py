from sacred import Ingredient

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


@dataset_ingredient.config
def datasetCfg():
    dataset = None; assert dataset != None
    dataset_path = None; assert dataset_path != None
    # Length of sequence to be considered parameter
    seq_length = None; assert seq_length != None  # RNN sequence length
    obs_length = None; assert obs_length != None  # observation length
    pred_length = seq_length - obs_length; assert pred_length != None  # prediction length
    keep_every = None; assert keep_every != None  # Keeps every keep_every entries of the input dataset (to recreate Kevin Murphy's work, needs be set to 5)
    orig_seq_len = None; assert orig_seq_len != None  # Original dataset sequence length (ped_data = 20 and basketball_data = 50)


@dataset_ingredient.named_config
def basketball():
    dataset = 'basketball'
    dataset_path = 'data/basketball/train'
    seq_length = 10
    obs_length = 6
    keep_every = 5
    orig_seq_len = 50


@dataset_ingredient.named_config
def basketball_small():
    dataset = 'basketball_small'
    dataset_path = 'data/basketball/small_train'
    seq_length = 10
    obs_length = 6
    keep_every = 5
    orig_seq_len = 50


@dataset_ingredient.named_config
def original():
    dataset = 'original'
    dataset_path = 'data/original/train'
    seq_length = 20
    obs_length = 8
    keep_every = 1
    orig_seq_len = 20
