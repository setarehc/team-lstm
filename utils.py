from sacred import Ingredient

data_ingredient = Ingredient('common')

@data_ingredient.config
def cfg():
    input_size = 2
    output_size = 5

    # RNN size parameter (dimension of the output/hidden state)
    rnn_size = 128  # size of RNN hidden state

    # Length of sequence to be considered parameter
    seq_length = 20  # RNN sequence length
    obs_length = 8  # observation length
    pred_length = 12  # prediction length

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
