import torch
import torch.nn as nn
import numpy as np

from torch.autograd import Variable

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()