import torch
import torch.nn as nn
import numpy as np

from torch.autograd import Variable

from .base import BaseModel

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
            
    def forward(self, input_data, grids, hidden_states, cell_states ,PedsList, num_pedlist, look_up):
        # TODO: Add social tensor calculation outside of model (in collate function)

        '''
        Forward pass for the model
        params:
        input_data: Input positions
        grids: Grid masks
        hidden_states: Hidden states of the peds
        cell_states: Cell states of the peds
        PedsList: id of peds in each frame for this sequence

        returns:
        outputs_return: Outputs corresponding to bivariate Gaussian distributions
        hidden_states
        cell_states
        '''

        numNodes = len(look_up)
        outputs = Variable(torch.zeros(self.seq_length * numNodes, self.output_size))
        if self.use_cuda:            
            outputs = outputs.cuda()

        # For each frame in the sequence
        for framenum,frame in enumerate(input_data):

            # Peds present in the current frame
            nodeIDs = [int(nodeID) for nodeID in PedsList[framenum]]

            if len(nodeIDs) == 0:
                # If no peds, then go to the next frame
                continue


            # List of nodes
            list_of_nodes = [look_up[x] for x in nodeIDs]

            corr_index = Variable((torch.LongTensor(list_of_nodes))) 
            if self.use_cuda:            
                corr_index = corr_index.cuda()

            # Select the corresponding input positions
            nodes_current = frame[list_of_nodes,:]
            # Get the corresponding grid masks
            grid_current = grids[framenum]

            # Get the corresponding hidden and cell states
            hidden_states_current = torch.index_select(hidden_states, 0, corr_index)
            cell_states_current = torch.index_select(cell_states, 0, corr_index)

            # Compute the social tensor
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
            outputs[framenum*numNodes + corr_index.data] = self.output_layer(h_nodes)

            # Update hidden and cell states
            hidden_states[corr_index.data] = h_nodes
            cell_states[corr_index.data] = c_nodes

        # Reshape outputs
        outputs_return = Variable(torch.zeros(self.seq_length, numNodes, self.output_size))
        if self.use_cuda:
            outputs_return = outputs_return.cuda()
        for framenum in range(self.seq_length):
            for node in range(numNodes):
                outputs_return[framenum, node, :] = outputs[framenum*numNodes + node, :]

        return outputs_return, hidden_states, cell_states
