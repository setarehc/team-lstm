import torch
import torch.nn as nn
import numpy as np

from torch.autograd import Variable

from .base import BaseModel

import trajectory_dataset as td
import helper
import grid

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

    def getSocialTensor(self, grid_batch, hidden_states, corr_indices, framenum):
        '''
        Computes the social tensor for a given grid mask and hidden states of all peds
        params:
        grid : Grid masks
        hidden_states : Hidden states of all peds
        '''
        num_nodes = len(corr_indices[0])
        social_tensor = hidden_states.new_zeros(num_nodes, self.grid_size*self.grid_size, self.rnn_size)
        

        # NOTE: This part has so many assumpotions!!
        # - items in corr_indices are assumed to be in the same order as ped_list (persons_list)
        # - corr index has all items from a particular batch next to eachother
        hidden_states_batched = {}
        for batch_idx in corr_indices[0]:
            if batch_idx not in hidden_states_batched:
                hidden_states_batched[batch_idx] = hidden_states[[i for i in range(len(hidden_states)) if corr_indices[0][i] == batch_idx]]
        last_batch = -1
        batch_change_idx = 0
        for idx, (batch_idx, ped_idx) in enumerate(zip(*corr_indices)):
            if batch_idx != last_batch:
                batch_change_idx = idx
            # Compute the social tensor
            social_tensor[idx] = torch.mm(torch.t(grid_batch[batch_idx][framenum][idx - batch_change_idx]), hidden_states_batched[batch_idx])

        # Reshape the social tensor
        social_tensor = social_tensor.view(num_nodes, self.grid_size*self.grid_size*self.rnn_size)
        return social_tensor
            
    def forward(self, batch, hidden_states=None, cell_states=None):
        # TODO: Add social tensor calculation outside of model (in collate function)
        '''
        Forward pass for the model
        params:
        input_data: Input positions
        grids: Grid masks
        hidden_states: Hidden states of the peds
        cell_states: Cell states of the peds
        persons_list: id of peds in each frame for this sequence

        returns:
        outputs_return: Outputs corresponding to bivariate Gaussian distributions
        hidden_states
        cell_states
        '''
        input_data_batch, grids_batch, persons_list_batch, lookup_batch = batch

        seq_length = input_data_batch.size(0)
        batch_size = input_data_batch.size(1)
        num_persons = input_data_batch.size(2)

        # Initialize hidden states, cell states and outputs
        if hidden_states is None:
            hidden_states = torch.zeros(batch_size, num_persons, self.rnn_size)
        if cell_states is None:
            cell_states = torch.zeros(batch_size, num_persons, self.rnn_size)

        outputs = torch.zeros(seq_length, batch_size, num_persons, self.output_size)

        if self.use_cuda:
            hidden_states = hidden_states.cuda()
            cell_states = cell_states.cuda()
            outputs = outputs.cuda()

        for framenum in range(seq_length):
            frame = input_data_batch[framenum]
            _corr_indices = [(b, lookup_batch[b][x]) for b in range(batch_size) for x in persons_list_batch[b][framenum]]
            corr_indices = list(zip(*_corr_indices))
            frame_current = frame[corr_indices[0], corr_indices[1], :]
            hidden_states_current = hidden_states[corr_indices[0], corr_indices[1]]
            cell_states_current = cell_states[corr_indices[0], corr_indices[1]]
            
            # Compute the social tensor
            #social_tensor = frame_current.new_zeros(frame_current.size(0), self.grid_size*self.grid_size*self.rnn_size)#torch.zeros(batch_size, num_persons, self.grid_size*self.grid_size*self.rnn_size).cuda() # TODO: Temporary!
            social_tensor = self.getSocialTensor(grids_batch, hidden_states_current, corr_indices, framenum)

            # Embed the social tensor
            #tensor_embedded = self.dropout(self.relu(self.tensor_embedding_layer(social_tensor.view(social_tensor.size(0) * social_tensor.size(1), -1 ))))
            tensor_embedded = self.dropout(self.relu(self.tensor_embedding_layer(social_tensor)))

            # Embed inputs
            #input_embedded = self.dropout(self.relu(self.input_embedding_layer(frame.view(frame.size(0) * frame.size(1), -1))))
            input_embedded = self.dropout(self.relu(self.input_embedding_layer(frame_current)))
            # Concat input
            concat_embedded = torch.cat((input_embedded, tensor_embedded), 1)

            # One-step of the LSTM
            h_nodes, c_nodes = self.cell(concat_embedded, (hidden_states_current, cell_states_current))
            output_nodes = self.output_layer(h_nodes)

            # # # Compute the output
            outputs[framenum, corr_indices[0], corr_indices[1]] = output_nodes

            # Update hidden and cell states
            hidden_states[corr_indices[0], corr_indices[1]] = h_nodes
            cell_states[corr_indices[0], corr_indices[1]] = c_nodes
        return outputs, hidden_states, cell_states

    def loss(self, outputs, batch):
        input_data, _, persons_list, lookup = batch
        return helper.Gaussian2DLikelihood(outputs, input_data, persons_list, lookup)

    def toCuda(self, batch):
        if self.use_cuda:
            batch[0] = batch[0].cuda()
            for b in range(len(batch[1])):
                for i in range(len(batch[1][b])):
                    batch[1][b][i] = batch[1][b][i].cuda()
        return batch

        ### OLD CODE!! ###
        #input_data, grids, persons_list, _, lookup = batch_item
        if self.use_cuda:        
            for i in range(len(batch_item[1])):
                batch_item[1][i] = batch_item[1][i].cuda()
            batch_item[0] = batch_item[0].cuda()
        return batch_item

    @staticmethod
    def collateFn(items, args):
        batch=[]
        #input_data, grids, persons_list, _, lookup = batch_item
        def MB(persons_list_batch):
            import pandas as pd
            import itertools
            mb = 0
            for persons_list in persons_list_batch:
                unique_ids = pd.unique(list(itertools.chain.from_iterable(persons_list))).astype(int)
                mb = max(mb, len(unique_ids))
            return mb

        def padPedsList(peds_list_seq_batch, max_size):
            ret = torch.zeros( (len(peds_list_seq_batch), len(peds_list_seq_batch[0]), max_size), dtype=torch.long) #TODO: type = long?
            for i in range(len(peds_list_seq_batch)):
                peds_list_seq = peds_list_seq_batch[i]
                for j in range(len(peds_list_seq)):
                    frame_peds = torch.tensor(peds_list_seq[j])
                    ret[i][j][0:len(frame_peds)] = frame_peds
            return ret
                    
        batch_size = len(items)

        x_seq_batch = [x[0] for x in items]
        num_peds_list_batch = [x[1] for x in items]
        peds_list_seq_batch = [x[2] for x in items]
        folder_path_batch = [x[3] for x in items]

        seq_length = len(peds_list_seq_batch[0])
        dim = 2
        mb = MB(peds_list_seq_batch)

        seq_tensor = torch.zeros((seq_length, batch_size, mb, dim))
        lookup_batch = []
        grid_seq_batch = []
        for i in range(batch_size):
            x_seq, lookup = td.convertToTensor(x_seq_batch[i], peds_list_seq_batch[i], mb)
            seq_tensor[:, i, :, :] = x_seq
            lookup_batch.append(lookup)
            
            folder_name = helper.getFolderName(folder_path_batch[i], args.dataset)
            dataset_dim = td.dataset_dimensions[folder_name]
            grid_seq = grid.getSequenceGridMask(x_seq, dataset_dim, peds_list_seq_batch[i], args.neighborhood_size,
                                            args.grid_size, args.use_cuda)
            grid_seq_batch.append(grid_seq)
        
        #peds_list_tensor = padPedsList(peds_list_seq_batch, mb).transpose(0, 1)
        #num_peds_tensor = torch.tensor(num_peds_list_batch, dtype=torch.long).transpose(0, 1)

        batch = [seq_tensor, grid_seq_batch, peds_list_seq_batch, lookup_batch]

        return batch


        ##### OLD CODE!! #####

        for x_seq, num_peds_list_seq, peds_list_seq, folder_path in items:
            # Dense vector (tensor) creation
            x_seq, lookup_seq = td.convertToTensor(x_seq, peds_list_seq)
            # Get processing file name and then get dimensions of file
            folder_name = helper.getFolderName(folder_path, args.dataset)
            dataset_dim = td.dataset_dimensions[folder_name]
            # Grid mask calculation and storage depending on grid parameter
            grid_seq = grid.getSequenceGridMask(x_seq, dataset_dim, peds_list_seq, args.neighborhood_size,
                                           args.grid_size, args.use_cuda)
            # Vectorize trajectories in sequence
            x_seq, _ = helper.vectorizeSeq(x_seq, peds_list_seq, lookup_seq)

            batch.append([x_seq, grid_seq, peds_list_seq, num_peds_list_seq, lookup_seq])

        return batch

