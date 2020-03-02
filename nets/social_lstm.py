import torch
import torch.nn as nn
import numpy as np

from torch.autograd import Variable

from .base import BaseModel

import trajectory_dataset as td
import helper
import grid

import itertools
import pandas as pd

class SocialModel(BaseModel):

    def __init__(self, args):
        '''
        Initializer function
        params:
        args: Training arguments
        '''
        super(SocialModel, self).__init__(args)
        
        # Store required sizes
        self.grid_size = args.grid_size


        # The LSTM cell
        self.cell = nn.LSTMCell(2*self.embedding_size, self.rnn_size)


        # Linear layer to embed the social tensor
        self.tensor_embedding_layer = nn.Linear(self.grid_size*self.grid_size*self.rnn_size, self.embedding_size)

        # Linear layer to map the hidden state of LSTM to output
        self.output_layer = nn.Linear(self.rnn_size, self.output_size)

        # ReLU and dropout unit
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)

    def getSocialTensor(self, grid, hidden_states):
        '''
        Computes the social tensor for a given grid mask and hidden states of all peds
        params:
        grid : Grid masks
        hidden_states : Hidden states of all peds
        '''
        # Number of peds
        numNodes = grid.size()[0]

        # Construct the variable
        social_tensor = Variable(torch.zeros(numNodes, self.grid_size*self.grid_size, self.rnn_size))
        if self.use_cuda:
            social_tensor = social_tensor.cuda()
        
        # For each ped
        for node in range(numNodes):
            # Compute the social tensor
            social_tensor[node] = torch.mm(torch.t(grid[node]), hidden_states)

        # Reshape the social tensor
        social_tensor = social_tensor.view(numNodes, self.grid_size*self.grid_size*self.rnn_size)
        return social_tensor
    
    def getAllPersons(self, batch):
        all_persons_list = []
        for batch_item in batch:
            all_persons_list.append(batch_item[2])
        
        return_list = list(itertools.chain.from_iterable(all_persons_list))
        return list(itertools.chain.from_iterable(return_list))
            
    def forward(self, batch, hidden_states=None, cell_states=None):
        '''
        Forward pass for the model
        params:
        batch: A single batch item of the dataset
        hidden_states: Hidden states of persons
        cell_states: Cell states of persons

        returns:
        outputs_return: Outputs corresponding to bivariate Gaussian distributions
        hidden_states
        cell_states
        '''
        
        batch_data = batch[0]
        mask = batch[1]

        seq_length = mask.size(0)
        batch_size = mask.size(1)
        max_num_persons = mask.size(2)

        all_persons_batch = pd.unique(self.getAllPersons(batch_data))
        lookup_batch = {i:idx for i,idx in zip(all_persons_batch, range(len(all_persons_batch)))}

        if hidden_states is None: 
            hidden_states = torch.zeros(batch_size, max_num_persons, self.rnn_size)
        if cell_states is None: 
            cell_states = torch.zeros(batch_size, max_num_persons, self.rnn_size)
        
        outputs = torch.zeros(seq_length, batch_size, max_num_persons, self.output_size)
        
        if self.use_cuda:
            hidden_states = hidden_states.cuda()
            cell_states = cell_states.cuda()
            outputs = outputs.cuda()

        # Loop through the batch
        for batch_idx, batch_item in enumerate(batch_data):
            # Extract sequence data
            input_data, grids, persons_list, lookup = batch_item
            num_persons = len(lookup)

            # For each frame in the sequence
            for framenum, frame in enumerate(input_data):
                # Peds present in the current frame
                node_ids = [int(node_id) for node_id in persons_list[framenum]]

                if len(node_ids) == 0:
                    # If no peds, then go to the next frame
                    continue

                # List of current node indices across the whole sequence
                node_indices_seq = [lookup[x] for x in node_ids]

                # List of corresponding indices across the whole batch
                node_indices_batch = [lookup_batch[i] for i in node_ids]
                corr_index = Variable((torch.LongTensor(node_indices_batch))) # TODO: remove variable 
                if self.use_cuda:            
                    corr_index = corr_index.cuda()

                # Select the corresponding input positions
                nodes_current = frame[node_indices_seq, :]
                # Get the corresponding grid masks
                grid_current = grids[framenum]

                # Get the corresponding hidden and cell states
                hidden_states_current = torch.index_select(hidden_states, 1, corr_index)[batch_idx]
                cell_states_current = torch.index_select(cell_states, 1, corr_index)[batch_idx]

                # Compute the social tensor by performing social pooling
                social_tensor = self.getSocialTensor(grid_current, hidden_states_current)
                # Embed the social tensor
                tensor_embedded = self.dropout(self.relu(self.tensor_embedding_layer(social_tensor)))

                # Embed inputs
                input_embedded = self.dropout(self.relu(self.input_embedding_layer(nodes_current)))
                
                # Concat input
                concat_embedded = torch.cat((input_embedded, tensor_embedded), 1)

                # One-step of the LSTM
                h_nodes, c_nodes = self.cell(concat_embedded, (hidden_states_current, cell_states_current))

                # Compute the output
                outputs[framenum, batch_idx, corr_index.data] = self.output_layer(h_nodes)

                # Update hidden and cell states
                hidden_states[batch_idx, corr_index.data] = h_nodes
                cell_states[batch_idx, corr_index.data] = c_nodes

        return outputs, hidden_states, cell_states
    
    def computeLoss(self, outputs, batch_item):
        input_data, _, persons_list, _, lookup, _, _ = batch_item
        return helper.Gaussian2DLikelihood(outputs, input_data, persons_list, lookup)
    
    def computeLossBatch(self, outputs, batch):
        batch_data = batch[0]
        all_persons_batch = pd.unique(self.getAllPersons(batch_data))
        lookup_batch = {i:idx for i,idx in zip(all_persons_batch, range(len(all_persons_batch)))}
        loss = 0
        for batch_idx, batch_item in enumerate(batch_data):
            input_data, _, persons_list, lookup = batch_item
            node_ids = pd.unique(list(itertools.chain.from_iterable(persons_list)))
            corr_indices = torch.LongTensor([lookup_batch[i] for i in node_ids]).to(outputs.device)
            corr_outputs = torch.index_select(outputs[:, batch_idx, :, :], 1, corr_indices)
            #import pdb; pdb.set_trace()
            loss += helper.Gaussian2DLikelihood(corr_outputs, input_data, persons_list, lookup)/len(persons_list)
        return loss/len(batch_data) #TODO: Check if correct and check if consistent with computeLossBatch of graph model

    def toCuda(self, batch_item):
        if self.use_cuda:
            batch_item[0] = batch_item[0].cuda()        
            for i in range(len(batch_item[1])):
                batch_item[1][i] = batch_item[1][i].cuda()
        return batch_item
    
    def toCudaBatch(self, batch):
        #batch_data, mask = batch
        if self.use_cuda:
            batch[1] = batch[1].cuda()
            for batch_item in batch[0]:
                batch_item[0] = batch_item[0].cuda()        
                for i in range(len(batch_item[1])):
                    batch_item[1][i] = batch_item[1][i].cuda()
        return batch

    @staticmethod
    def collateFn(items, args):
        batch=[]

        all_seq_data = []
        all_persons = []
        batch_data = []
        for x_seq, _, persons_list_seq, folder_path in items:
            # Dense vector (tensor) creation
            all_seq_data.append(x_seq)
            all_persons.append(list(itertools.chain.from_iterable(persons_list_seq)))

            x_seq, lookup_seq = td.convertToTensor(x_seq, persons_list_seq)
            # Get processing file name and then get dimensions of file
            folder_name = helper.getFolderName(folder_path, args.dataset)
            dataset_dim = td.dataset_dimensions[folder_name]
            # Grid mask calculation and storage depending on grid parameter
            grid_seq = grid.getSequenceGridMask(x_seq, dataset_dim, persons_list_seq, args.neighborhood_size,
                                           args.grid_size, args.use_cuda)
            # Vectorize trajectories in sequence
            #x_seq, init_values_dict = helper.vectorizeSeq(x_seq, persons_list_seq, lookup_seq)
            batch_data.append([x_seq, grid_seq, persons_list_seq, lookup_seq])
        _, mask = td.tensorizeData(all_seq_data, all_persons)
        batch = [batch_data, mask]

        return batch