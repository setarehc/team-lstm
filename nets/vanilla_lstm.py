import torch
import torch.nn as nn
import numpy as np

from torch.autograd import Variable

from .base import BaseModel

import trajectory_dataset as td
import helper

class VanillaModel(BaseModel):

    def __init__(self, args, infer=False):
        '''
        Initializer function
        params:
        args: Training arguments
        infer: Training or test time (true if test time)
        '''
        super(VanillaModel, self).__init__(args)

        self.infer = infer

        if infer:
            # Test time
            self.seq_length = 1
        else:
            # Training time
            self.seq_length = args.seq_length

        # Store required sizes
        self.rnn_size = args.rnn_size
        self.embedding_size = args.embedding_size
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.maxNumPeds=args.maxNumPeds
        self.seq_length=args.seq_length

        # The LSTM cell
        self.cell = nn.LSTMCell(self.embedding_size, self.rnn_size)

        # Linear layer to map the hidden state of LSTM to output
        self.output_layer = nn.Linear(self.rnn_size, self.output_size)

        # ReLU and dropout unit
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)
            
    def forward(self, batch_item, hidden_states=None, cell_states=None):
        '''
        Forward pass for the model
        params:
        input_data: Input positions
        hidden_states: Hidden states of the peds
        cell_states: Cell states of the peds
        PedsList: id of peds in each frame for this sequence

        returns:
        outputs_return: Outputs corresponding to bivariate Gaussian distributions
        hidden_states
        cell_states
        '''
        ##self, input_data, hidden_states, cell_states ,PedsList, num_pedlist, look_up
        input_data, persons_list, _, lookup, _, _ = batch_item

        num_persons = len(lookup)

        if hidden_states is None: 
            hidden_states = torch.zeros(num_persons, self.rnn_size)
        if cell_states is None: 
            cell_states = torch.zeros(num_persons, self.rnn_size)
        outputs = torch.zeros(self.seq_length, num_persons, self.output_size)
        
        if self.use_cuda:
            hidden_states = hidden_states.cuda()
            cell_states = cell_states.cuda()      
            outputs = outputs.cuda()

        # For each frame in the sequence
        for framenum,frame in enumerate(input_data):
            # Peds present in the current frame
            nodeIDs = [int(nodeID) for nodeID in persons_list[framenum]]

            if len(nodeIDs) == 0:
                # If no peds, then go to the next frame
                continue

            # List of nodes
            list_of_nodes = [lookup[x] for x in nodeIDs]

            corr_index = Variable((torch.LongTensor(list_of_nodes)))
            if self.use_cuda:            
                corr_index = corr_index.cuda()

            # Select the corresponding input positions
            nodes_current = frame[list_of_nodes,:]

            # Get the corresponding hidden and cell states
            hidden_states_current = torch.index_select(hidden_states, 0, corr_index)
            cell_states_current = torch.index_select(cell_states, 0, corr_index)


            # Embed inputs
            input_embedded = self.dropout(self.relu(self.input_embedding_layer(nodes_current)))

            # One-step of the LSTM
            h_nodes, c_nodes = self.cell(input_embedded, (hidden_states_current, cell_states_current))
            
            # Compute the output
            outputs[framenum, corr_index.data] = self.output_layer(h_nodes)

            # Update hidden and cell states
            hidden_states[corr_index.data] = h_nodes
            cell_states[corr_index.data] = c_nodes

        return outputs, hidden_states, cell_states

    def computeLoss(self, outputs, batch_item):
        input_data, persons_list, _, lookup, _, _ = batch_item
        return helper.Gaussian2DLikelihood(outputs, input_data, persons_list, lookup)
    
    def toCuda(self, batch_item):
        #input_data, persons_list, _, lookup, _, _ = batch_item
        if self.use_cuda:
            batch_item[0] = batch_item[0].cuda()
        return batch_item
    
    @staticmethod
    def collateFn(items, args):
        batch=[]
        for x_seq, num_persons_list_seq, persons_list_seq, folder_path in items:
            # Get processing file name and then get dimensions of file
            folder_name = helper.getFolderName(folder_path, args.dataset)
            dataset_dim = td.dataset_dimensions[folder_name]
            # Dense vector (tensor) creation
            x_seq, lookup_seq = td.convertToTensor(x_seq, persons_list_seq)
            # Vectorize trajectories in sequence
            x_seq, init_values_dict = helper.vectorizeSeq(x_seq, persons_list_seq, lookup_seq)

            batch.append([x_seq, persons_list_seq, num_persons_list_seq, lookup_seq, dataset_dim, init_values_dict])
        return batch