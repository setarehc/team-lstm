import torch
import torch.nn as nn
import numpy as np

from torch.autograd import Variable

from .base import BaseModel

import trajectory_dataset as td
import helper

import itertools

class GraphModel(BaseModel):

    def __init__(self, args):
        '''
        Initializer function
        params:
        args: Training arguments
        '''
        super(GraphModel, self).__init__(args)
        
        # Store required sizes
        self.grid_size = args.grid_size


        # The LSTM cell
        self.cell = nn.LSTMCell(2*self.embedding_size, self.rnn_size)

        # g in RN model
        self.g_module = nn.Sequential(nn.Linear(2*self.rnn_size, 256),
                                      nn.ReLU(),
                                      nn.Linear(256, 256),
                                      nn.ReLU(),
                                      nn.Linear(256, 256),
                                      nn.ReLU(),
                                      nn.Linear(256, self.rnn_size))

        # f in RN model
        self.f_module = nn.Sequential(nn.Linear(self.rnn_size, 256),
                                                    nn.ReLU(),
                                                    nn.Linear(256, 256),
                                                    nn.Dropout(args.dropout),
                                                    nn.ReLU(),
                                                    nn.Linear(256, self.embedding_size))

        # Linear layer to map the hidden state of LSTM to output
        self.output_layer = nn.Linear(self.rnn_size, self.output_size)

        # ReLU and dropout unit
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)

    def getGraphTensorDumb(self, hidden_states_current):
        numNodes = len(hidden_states_current)
        graph_tensor = torch.zeros(numNodes, self.rnn_size).to(hidden_states_current.device)
        for i in range(numNodes):
            for j in range(numNodes):
                graph_tensor[i, :] += self.g_module(torch.cat((hidden_states_current[i], hidden_states_current[j]), 0))
        return graph_tensor

    def getGraphTensor(self, hidden_states_current):
        numNodes = len(hidden_states_current)
        X = hidden_states_current.unsqueeze(1).repeat(1, numNodes, 1).reshape(numNodes*numNodes, -1)
        Y = hidden_states_current.repeat(numNodes, 1)
        ret = self.g_module(torch.cat((X, Y), 1))
        return torch.sum(ret.reshape(numNodes, numNodes, -1), 1)
        # graph_tensor = torch.zeros(numNodes, self.rnn_size).to(hidden_states_current.device)
        # for i in range(numNodes):
        #     X = hidden_states_current[i].repeat(numNodes, 1)
        #     graph_tensor[i, :] = torch.sum(self.g_module(torch.cat((X, hidden_states_current), 1)), 0)
        # return graph_tensor

    def forward(self, batch, hidden_states=None, cell_states=None):
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

        input_data, PedsList, num_pedlist, look_up = batch

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

            # Get the corresponding hidden and cell states
            hidden_states_current = torch.index_select(hidden_states, 0, corr_index)
            cell_states_current = torch.index_select(cell_states, 0, corr_index)

            # Compute the social tensor
            graph_tensor = self.getGraphTensor(hidden_states_current)
            # Embed the social tensor
            tensor_embedded = self.f_module(graph_tensor)

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


    @staticmethod
    def collateFn(items, args):
        batch=[]
        for x_seq, num_peds_list_seq, peds_list_seq, folder_path in items:
            # Get unique persons ids in the whole sequence
            unique_ids = set(pid for peds_list in peds_list_seq for pid in peds_list)
            max_num_persons = len(unique_ids)
            # Dense vector (tensor) creation
            x_seq, lookup_seq = td.convertToTensor(x_seq, peds_list_seq, max_num_persons)
            # Vectorize trajectories in sequence
            x_seq, _ = helper.vectorizeSeq(x_seq, peds_list_seq, lookup_seq)

            batch.append([x_seq, peds_list_seq, num_peds_list_seq, lookup_seq])

        return batch