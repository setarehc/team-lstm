import torch
import torch.nn as nn
import numpy as np

from torch.autograd import Variable

class BaseModel(nn.Module):
    def __init__(self, args):
        super(BaseModel, self).__init__()

        self.args = args
        self.use_cuda = args.use_cuda
        
        # Store required sizes
        self.rnn_size = args.rnn_size
        self.embedding_size = args.embedding_size
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.maxNumPeds = args.maxNumPeds
        self.seq_length = args.seq_length

        # Linear layer to embed the input position
        self.input_embedding_layer = nn.Linear(self.input_size, self.embedding_size)