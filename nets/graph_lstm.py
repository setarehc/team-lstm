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

    def getGraphTensor(self, hidden_states, adj_matrix, mask):
        batch_size, num_nodes, rnn_size = hidden_states.shape
        X = hidden_states.unsqueeze(2).expand(hidden_states.size()[:2] + torch.Size([num_nodes]) + hidden_states.size()[2:]).reshape(batch_size, num_nodes*num_nodes, -1)
        Y = hidden_states.unsqueeze(1).expand(hidden_states.size()[:1] + torch.Size([num_nodes]) + hidden_states.size()[1:]).reshape(batch_size, num_nodes*num_nodes, -1)
        G = self.g_module(torch.cat((X, Y), -1).reshape(-1, 2*rnn_size)).reshape(batch_size, num_nodes, num_nodes, -1)
        adj_matrix_expanded = adj_matrix.unsqueeze(0).expand(torch.Size([batch_size]) + adj_matrix.size())
        G[adj_matrix_expanded == 0] = 0
        G[mask == 0] = 0
        G[mask.unsqueeze(1).expand(torch.Size([batch_size, num_nodes, num_nodes])) == 0] = 0
        res = torch.sum(G, dim=2)
        return res

    def getAdjMatrix(self, num_nodes):
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
 

    def forward(self, batch, hidden_states=None, cell_states=None):
        '''
        Forward pass for the model
        params:
        batch: A single batch of dataset
        hidden_states: Hidden states of persons
        cell_states: Cell states of persons
        returns:
        outputs_return: Outputs corresponding to bivariate Gaussian distributions
        hidden_states
        cell_states
        '''
        #input_data, persons_list, _, lookup, _, _ = batch_item
        all_xy_posns, mask = batch

        #num_persons = len(lookup)
        seq_length = all_xy_posns.size(0)
        batch_size = all_xy_posns.size(1)
        max_num_persons = all_xy_posns.size(2)

        if hidden_states is None: 
            hidden_states = torch.zeros(batch_size, max_num_persons, self.rnn_size)
        if cell_states is None: 
            cell_states = torch.zeros(batch_size, max_num_persons, self.rnn_size)
        
        outputs = torch.zeros(seq_length, batch_size, max_num_persons, self.output_size)

        if self.use_cuda:      
            hidden_states = hidden_states.cuda()
            cell_states = cell_states.cuda()      
            outputs = outputs.cuda()

        # For each frame in the sequence
        for framenum in range(seq_length):
            
            frame_data = all_xy_posns[framenum]
            mask_data = mask[framenum]
            
            adj_matrix = self.getAdjMatrix(max_num_persons).to(hidden_states.device)
            # Compute the social tensor
            graph_tensor = self.getGraphTensor(hidden_states, adj_matrix, mask_data)
            # Embed the social tensor
            tensor_embedded = self.f_module(graph_tensor)

            # Embed inputs
            input_embedded = self.dropout(self.relu(self.input_embedding_layer(frame_data)))
            
            # Concat input
            concat_embedded = torch.cat((input_embedded, tensor_embedded), 2)

            # One-step of the LSTM
            h_nodes, c_nodes = self.cell(concat_embedded.reshape(batch_size * max_num_persons, -1),
                                        (hidden_states.reshape(batch_size * max_num_persons, -1), 
                                        cell_states.reshape(batch_size * max_num_persons, -1)))
            
            # Update hidden and cell states
            h_nodes = h_nodes.reshape(batch_size, max_num_persons, -1)
            c_nodes = c_nodes.reshape(batch_size, max_num_persons, -1)

            h_nodes[mask_data==0] = hidden_states[mask_data==0]
            hidden_states = h_nodes

            c_nodes = c_nodes - cell_states
            cell_states = cell_states + c_nodes * mask_data.unsqueeze(2).expand(mask_data.size() + torch.Size([self.rnn_size]))

            # Compute the output
            outputs[framenum] = self.output_layer(hidden_states)

        return outputs, hidden_states, cell_states

    def computeLoss(self, outputs, batch_item):
        input_data, persons_list, _, lookup, _, _ = batch_item
        return helper.Gaussian2DLikelihood(outputs, input_data, persons_list, lookup)
    
    def computeLossBatch(self, outputs, batch):
        all_xy_posns, mask = batch
        return helper.Gaussian2DLikelihoodBatch(outputs, all_xy_posns, mask)

    def toCuda(self, batch_item):
        #input_data, persons_list, num_persons_list, lookup, dataset_dim, init_values_dict = batch_item
        if self.use_cuda:
            batch_item[0] = batch_item[0].cuda()
        return batch_item
    
    def toCudaBatch(self, batch):
        #data, mask = batch
        if self.use_cuda:
            batch[0] = batch[0].cuda()
            batch[1] = batch[1].cuda()
        return batch

    def sample(self, batch, args, saved_args):
        '''
        The sample function
        params:
        batch: A single batch item
        net: The model
        args: Model arguments
        saved_args: Configuration arguments used for training the model
        '''
        #import pdb; pdb.set_trace()
        batch = self.toCudaBatch(batch)

        # Unpack batch
        xy_posns = batch[0]
        mask = batch[1]

        batch_size = xy_posns.size(1)
        num_persons = xy_posns.size(2)
        dim = 2

        with torch.no_grad():
            # Initialize hidden and cell states
            hidden_states = torch.zeros(batch_size, num_persons, args.rnn_size)
            cell_states = torch.zeros(batch_size, num_persons, args.rnn_size)
            
            # Initialize the return data structure
            sampled_xy_posns = torch.zeros(args.obs_length + args.pred_length, batch_size, num_persons, dim)
            
            if args.use_cuda:
                hidden_states = hidden_states.cuda()
                cell_states = cell_states.cuda()
                sampled_xy_posns = sampled_xy_posns.cuda()

            # For the observed part of the trajectory
            for tstep in range(args.obs_length - 1):
                # Forward prop
                outputs, hidden_states, cell_states = self.forward([xy_posns[tstep].unsqueeze(0), mask[tstep].unsqueeze(0)],
                                                        hidden_states,
                                                        cell_states)

                # Extract the mean, std and corr of the bivariate Gaussian
                mux, muy, sx, sy, corr = helper.getCoefBatch(outputs)
                # Sample from the bivariate Gaussian
                next_x, next_y = helper.sampleGaussian2dBatch(mux.data, muy.data, sx.data, sy.data, corr.data, mask[tstep].unsqueeze(0))
                # Save predicted positions into return data structure
                sampled_xy_posns[tstep + 1, :, :, 0] = next_x
                sampled_xy_posns[tstep + 1, :, :, 1] = next_y

            # For the predicted part of the trajectory
            for tstep in range(args.obs_length - 1, args.pred_length + args.obs_length - 1):
                # Do a forward prop
                if tstep == args.obs_length - 1:
                    #net_input = x_seq[tstep].view(1, num_persons, 2)
                    net_input = xy_posns[tstep].unsqueeze(0)
                else:
                    #net_input = sampled_x_seq[tstep].view(1, num_persons, 2)
                    net_input = sampled_xy_posns[tstep].unsqueeze(0)

                outputs, hidden_states, cell_states = self.forward([net_input, mask[tstep].unsqueeze(0)],
                                                        hidden_states,
                                                        cell_states)
                    
                # Extract the mean, std and corr of the bivariate Gaussian
                mux, muy, sx, sy, corr = helper.getCoefBatch(outputs)
                # Sample from the bivariate Gaussian
                next_x, next_y = helper.sampleGaussian2dBatch(mux.data, muy.data, sx.data, sy.data, corr.data, mask[tstep].unsqueeze(0))

                # Save predicted positions into return data structure
                sampled_xy_posns[tstep + 1, :, :, 0] = next_x
                sampled_xy_posns[tstep + 1, :, :, 1] = next_y

            # Revert the points back to the original space
            #sampled_x_seq = revertSeq(sampled_x_seq.data.cpu(), peds_list, lookup, init_values_dict)
            return sampled_xy_posns

    @staticmethod
    def collateFn(items, args):
        batch=[]
        
        all_seq_data = []
        all_persons = []
        for x_seq, num_persons_list_seq, persons_list_seq, folder_path in items:
            all_seq_data.append(x_seq)
            all_persons.append(list(itertools.chain.from_iterable(persons_list_seq)))
        
        data, mask, _ = td.tensorizeData(all_seq_data, all_persons)
        batch = [data, mask]
        return batch
        
        ###Old Implementation###
        # batch=[]
        # for x_seq, num_persons_list_seq, persons_list_seq, folder_path in items:
        #     # Get processing file name and then get dimensions of file
        #     folder_name = helper.getFolderName(folder_path, args.dataset)
        #     dataset_dim = td.dataset_dimensions[folder_name]
        #     # Dense vector (tensor) creation
        #     import pdb; pdb.set_trace()
        #     x_seq, lookup_seq = td.convertToTensor(x_seq, persons_list_seq)
        #     # TODO remove vectorizeSeq from below
        #     # Vectorize trajectories in sequence
        #     x_seq, init_values_dict = helper.vectorizeSeq(x_seq, persons_list_seq, lookup_seq)

        #     batch.append([x_seq, persons_list_seq, num_persons_list_seq, lookup_seq, dataset_dim, init_values_dict])
        # return batch
