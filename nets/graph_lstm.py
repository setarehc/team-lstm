import torch
import torch.nn as nn
import numpy as np

from torch.autograd import Variable

from .base import BaseModel

import trajectory_dataset as td
import helper

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
        self.graph_type = args.graph_type

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

    def getGraphTensor(self, hidden_states_current, adj_matrix):
        num_nodes = len(hidden_states_current)
        X = hidden_states_current.unsqueeze(1).expand(hidden_states_current.size()[:1] + torch.Size([num_nodes]) + hidden_states_current.size()[1:]).reshape(num_nodes*num_nodes, -1)
        Y = hidden_states_current.unsqueeze(0).expand(torch.Size([num_nodes]) + hidden_states_current.size()).reshape(num_nodes*num_nodes, -1)
        G = self.g_module(torch.cat((X, Y), 1)).reshape(num_nodes, num_nodes, -1)
        G[adj_matrix == 0] = 0
        res = torch.sum(G, dim=0)
        return res

    def getAdjMatrix(self, ped_ids):
        num_nodes = len(ped_ids)
        if self.graph_type == 'FC': # fully connected graph
            return torch.ones(num_nodes, num_nodes)
        elif self.graph_type == 'EYE': # no one is connected to anyone
            return torch.eye(num_nodes)
        # The following structures only make sense for the basketball dataset
        # Assumption: persons_to_keep = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        elif self.graph_type == 'TM': # connects offensive players (doesn't include the ball)
            adjm = torch.ones(num_nodes-1, num_nodes-1)
            adjm = torch.cat((torch.zeros(1, num_nodes-1), adjm))
            adjm = torch.cat((torch.zeros(num_nodes, 1), adjm), 1)
            return adjm
        # Assumption: persons_to_keep = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        elif self.graph_type == 'TMS': # connects offensive players - connects defensive players
            num_team_players = int((num_nodes-1) / 2)
            tmp1 = torch.cat((torch.ones(num_team_players, num_team_players), torch.zeros(num_team_players, num_team_players)), 1)
            tmp2 = torch.cat((torch.zeros(num_team_players, num_team_players), torch.ones(num_team_players, num_team_players)), 1)
            tmp3 = torch.cat((tmp1, tmp2))
            adjm = torch.cat((torch.zeros(1, num_nodes-1), tmp3))
            adjm = torch.cat((torch.zeros(num_nodes, 1), adjm), 1)
            return adjm
        elif self.graph_type == 'TMSB': # connects offensive players and the ball - connects defensive players and the ball
            num_team_players = int((num_nodes-1) / 2)
            tmp11 = torch.ones(num_team_players+1, num_team_players+1)
            tmp12 = torch.cat((torch.ones(1, num_team_players), torch.zeros(num_team_players, num_team_players)))
            tmp21 = torch.cat((torch.ones(num_team_players, 1), torch.zeros(num_team_players, num_team_players)), 1)
            tmp22 = torch.ones(num_team_players, num_team_players)
            tmp1 = torch.cat((tmp11, tmp12), 1)
            tmp2 = torch.cat((tmp21, tmp22), 1)
            adjm = torch.cat((tmp1, tmp2))
            return adjm
 

    def forward(self, batch_item, hidden_states=None, cell_states=None):
        '''
        Forward pass for the model
        params:
        batch_item: A single batch of dataset
        hidden_states: Hidden states of persons
        cell_states: Cell states of persons

        returns:
        outputs_return: Outputs corresponding to bivariate Gaussian distributions
        hidden_states
        cell_states
        '''
        
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
            nodeIDs = [int(node_id) for node_id in persons_list[framenum]]

            adj_matrix = self.getAdjMatrix(nodeIDs).to(hidden_states.device)

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

            # Compute the social tensor
            graph_tensor = self.getGraphTensor(hidden_states_current, adj_matrix)
            # Embed the social tensor
            tensor_embedded = self.f_module(graph_tensor)

            # Embed inputs
            input_embedded = self.dropout(self.relu(self.input_embedding_layer(nodes_current)))
            
            # Concat input
            concat_embedded = torch.cat((input_embedded, tensor_embedded), 1)

            # One-step of the LSTM
            h_nodes, c_nodes = self.cell(concat_embedded, (hidden_states_current, cell_states_current))


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
