import torch
import torch.nn as nn
import numpy as np

from torch.autograd import Variable

from .base import BaseModel

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

    '''
    def getIndicesLists(self, adj_matrix):
        
        #Returns indices lists that are used to pair up hidden states of connected persons
        
        # Initialize lists
        l1 = []
        l2 = []
        for i in range(len(adj_matrix)):
            for j in range(len(adj_matrix)):
                if adj_matrix[i][j] == 1:
                    l1.append(i)
                    l2.append(j)       
        return l1, l2
    '''
    
    def getGraphTensor(self, hidden_states_current, adj_matrix):
        numNodes = len(hidden_states_current)
        # If fully connected graph:
        if  torch.all(adj_matrix == 1):
            X = hidden_states_current.unsqueeze(1).repeat(1, numNodes, 1).reshape(numNodes*numNodes, -1)
            Y = hidden_states_current.repeat(numNodes, 1)
            ret = self.g_module(torch.cat((X, Y), 1))
            return torch.sum(ret.reshape(numNodes, numNodes, -1), 1)
        
        else:
            graph_tensor = torch.zeros(numNodes, self.rnn_size).to(hidden_states_current.device)
            for idx in range(len(adj_matrix)):
                l1 = [idx for item in adj_matrix[idx] if item==1]
                l2 = [i for i, val in enumerate(adj_matrix[idx]) if val==1]
                if len(l1) == 0:
                    if len(l2) != 0:
                        raise Exception('l1 and l2 are not representing the same adjacency list.')
                    # TODO: Check if makes sense to leave it zero or to feed zero to the g_module and set graph_tensor to what that returns
                    # X = torch.zeros(self.rnn_size)
                    # Y = torch.zeros(self.rnn_size)
                    # graph_tensor[idx, :] = self.g_module(torch.cat((X, Y), 1))
                else:
                    X = torch.cat([hidden_states_current[[x]] for x in l1])
                    Y = torch.cat([hidden_states_current[[y]] for y in l2])
                    graph_tensor[idx, :] = torch.sum(self.g_module(torch.cat((X, Y), 1)), 0)
            return graph_tensor
        '''      
            #import pdb; pdb.set_trace()
            graph_tensor = torch.zeros(numNodes, self.rnn_size).to(hidden_states_current.device)
            l1, l2 = self.getIndicesLists(adj_matrix)
            X = torch.cat([hidden_states_current[[x]] for x in l1])
            Y = torch.cat([hidden_states_current[[y]] for y in l2])
            ret = self.g_module(torch.cat((X, Y), 1))
            for idx, item in enumerate(l1):
                graph_tensor[item] += ret[idx]
            for item, (first, last) in l1:
                graph_tensor[item] = torch.sum(ret[first:last])
            #return graph_tensor
            opt1 = graph_tensor'''

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
 

    def forward(self, input_data, grids, hidden_states, cell_states, PedsList, num_pedlist, look_up):
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
        outputs = torch.zeros(self.seq_length, numNodes, self.output_size)
        if self.use_cuda:            
            outputs = outputs.cuda()

        # For each frame in the sequence
        for framenum,frame in enumerate(input_data):

            # Peds present in the current frame
            nodeIDs = [int(nodeID) for nodeID in PedsList[framenum]]

            adj_matrix = self.getAdjMatrix(nodeIDs)

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
