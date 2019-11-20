import numpy as np
import torch
from torch.autograd import Variable

import os
import shutil
from os import walk
import math

from nets import SocialModel
from nets import GraphModel
from olstm_model import OLSTMModel
from vlstm_model import VLSTMModel

from os import listdir
from os.path import isfile, join
from trajectory_dataset import *



#one time set dictionary for a exist key
class WriteOnceDict(dict):
    def __setitem__(self, key, value):
        if not key in self:
            super(WriteOnceDict, self).__setitem__(key, value)

#(1 = social lstm, 2 = obstacle lstm, 3 = vanilla lstm)
def getMethodName(index):
    # return method name given index
    return {
        1 : 'SOCIALLSTM',
        2 : 'OBSTACLELSTM',
        3 : 'VANILLALSTM'
    }.get(index, 'SOCIALLSTM')

def getModel(args, arguments, infer = False):
    # return a model given index and arguments
    index = args.method
    model_type = args.model
    if index == 1:
        if model_type == 'social':
            return SocialModel(arguments)
        elif model_type == 'graph':
            return GraphModel(arguments)
        else:
            raise ValueError(f'Unexpected value for args.model ({args.model})')
    elif index == 2:
        return OLSTMModel(arguments, infer)
    elif index == 3:
        return VLSTMModel(arguments, infer)
    else:
        return SocialModel(arguments)

def getCoef(outputs):
    '''
    Extracts the mean, standard deviation and correlation
    params:
    outputs : Output of the SRNN model
    '''
    mux, muy, sx, sy, corr = outputs[:, :, 0], outputs[:, :, 1], outputs[:, :, 2], outputs[:, :, 3], outputs[:, :, 4]

    sx = torch.exp(sx)
    sy = torch.exp(sy)
    corr = torch.tanh(corr)
    return mux, muy, sx, sy, corr


def sampleGaussian2d(mux, muy, sx, sy, corr, nodesPresent, look_up):
    '''
    Parameters
    ==========

    mux, muy, sx, sy, corr : a tensor of shape 1 x numNodes
    Contains x-means, y-means, x-stds, y-stds and correlation

    nodesPresent : a list of nodeIDs present in the frame
    look_up : lookup table for determining which ped is in which array index

    Returns
    =======

    next_x, next_y : a tensor of shape numNodes
    Contains sampled values from the 2D gaussian
    '''
    o_mux, o_muy, o_sx, o_sy, o_corr = mux[0, :], muy[0, :], sx[0, :], sy[0, :], corr[0, :]

    numNodes = mux.size()[1]
    next_x = torch.zeros(numNodes)
    next_y = torch.zeros(numNodes)
    converted_node_present = [look_up[node] for node in nodesPresent]
    for node in range(numNodes):
        if node not in converted_node_present:
            continue
        mean = [o_mux[node], o_muy[node]]
        cov = [[o_sx[node]*o_sx[node], o_corr[node]*o_sx[node]*o_sy[node]],
                [o_corr[node]*o_sx[node]*o_sy[node], o_sy[node]*o_sy[node]]]

        mean = np.array(mean, dtype='float')
        cov = np.array(cov, dtype='float')
        next_values = np.random.multivariate_normal(mean, cov, 1)
        next_x[node] = next_values[0][0]
        next_y[node] = next_values[0][1]

    return next_x, next_y

def getMeanError(ret_nodes, nodes, assumedNodesPresent, trueNodesPresent, using_cuda, look_up):
    '''
    Parameters
    ==========

    ret_nodes : A tensor of shape pred_length x numNodes x 2
    Contains the predicted positions for the nodes

    nodes : A tensor of shape pred_length x numNodes x 2
    Contains the true positions for the nodes

    nodesPresent lists: A list of lists, of size pred_length
    Each list contains the nodeIDs of the nodes present at that time-step

    look_up : lookup table for determining which ped is in which array index

    Returns
    =======

    Error : Mean euclidean distance between predicted trajectory and the true trajectory
    '''
    pred_length = ret_nodes.size()[0]
    error = torch.zeros(pred_length)
    if using_cuda:
        error = error.cuda()

    for frame_idx in range(pred_length):
        num_peds_in_frame = 0

        for ped_id in assumedNodesPresent[frame_idx]:
            ped_id = int(ped_id)

            if ped_id not in trueNodesPresent[frame_idx]:
                continue

            ped_idx = look_up[ped_id]

            pred_pos = ret_nodes[frame_idx, ped_idx, :]
            true_pos = nodes[frame_idx, ped_idx, :]

            error[frame_idx] += torch.norm(pred_pos - true_pos, p=2)
            num_peds_in_frame += 1

        if num_peds_in_frame > 1:
            error[frame_idx] = error[frame_idx] / num_peds_in_frame

    return torch.mean(error)


def getFinalError(ret_nodes, nodes, assumedNodesPresent, trueNodesPresent, look_up):
    '''
    Parameters
    ==========

    ret_nodes : A tensor of shape pred_length x numNodes x 2
    Contains the predicted positions for the nodes

    nodes : A tensor of shape pred_length x numNodes x 2
    Contains the true positions for the nodes

    nodesPresent lists: A list of lists, of size pred_length
    Each list contains the nodeIDs of the nodes present at that time-step

    look_up : lookup table for determining which ped is in which array index


    Returns
    =======

    Error : Mean final euclidean distance between predicted trajectory and the true trajectory
    '''
    pred_length = ret_nodes.size()[0]
    error = 0
    num_peds_in_frame = 0

    # Last time-step
    tstep = pred_length - 1
    for ped_id in assumedNodesPresent[tstep]:
        ped_id = int(ped_id)


        if ped_id not in trueNodesPresent[tstep]:
            continue

        ped_idx = look_up[ped_id]


        pred_pos = ret_nodes[tstep, ped_idx, :]
        true_pos = nodes[tstep, ped_idx, :]

        error += torch.norm(pred_pos - true_pos, p=2)
        num_peds_in_frame += 1

    if num_peds_in_frame > 1:
        error = error / num_peds_in_frame

    return error

def Gaussian2DLikelihoodInference(outputs, targets, nodesPresent, pred_length, look_up):
    '''
    Computes the likelihood of predicted locations under a bivariate Gaussian distribution at test time

    Parameters:

    outputs: Torch variable containing tensor of shape seq_length x numNodes x 1 x output_size
    targets: Torch variable containing tensor of shape seq_length x numNodes x 1 x input_size
    nodesPresent : A list of lists, of size seq_length. Each list contains the nodeIDs that are present in the frame
    '''
    seq_length = outputs.size()[0]
    obs_length = seq_length - pred_length

    # Extract mean, std devs and correlation
    mux, muy, sx, sy, corr = getCoef(outputs)

    # Compute factors
    normx = targets[:, :, 0] - mux
    normy = targets[:, :, 1] - muy
    sxsy = sx * sy

    z = (normx/sx)**2 + (normy/sy)**2 - 2*((corr*normx*normy)/sxsy)
    negRho = 1 - corr**2

    # Numerator
    result = torch.exp(-z/(2*negRho))
    # Normalization factor
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

    # Final PDF calculation
    result = result / denom

    # Numerical stability
    epsilon = 1e-20

    result = -torch.log(torch.clamp(result, min=epsilon))
    #print(result)

    loss = 0
    counter = 0

    for framenum in range(obs_length, seq_length):
        nodeIDs = nodesPresent[framenum]
        nodeIDs = [int(nodeID) for nodeID in nodeIDs]

        for nodeID in nodeIDs:

            nodeID = look_up[nodeID]
            loss = loss + result[framenum, nodeID]
            counter = counter + 1

    if counter != 0:
        return loss / counter
    else:
        return loss


def Gaussian2DLikelihood(outputs, targets, nodesPresent, look_up):
    '''
    params:
    outputs : predicted locations
    targets : true locations
    assumedNodesPresent : Nodes assumed to be present in each frame in the sequence
    nodesPresent : True nodes present in each frame in the sequence
    look_up : lookup table for determining which ped is in which array index

    '''
    seq_length = outputs.size()[0]
    # Extract mean, std devs and correlation
    mux, muy, sx, sy, corr = getCoef(outputs)

    # Compute factors
    normx = targets[:, :, 0] - mux
    normy = targets[:, :, 1] - muy
    sxsy = sx * sy

    z = (normx/sx)**2 + (normy/sy)**2 - 2*((corr*normx*normy)/sxsy)
    negRho = 1 - corr**2

    # Numerator
    result = torch.exp(-z/(2*negRho))
    # Normalization factor
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

    # Final PDF calculation
    result = result / denom

    # Numerical stability
    epsilon = 1e-20

    #if torch.any(result > 1):
    #    import pdb; pdb.set_trace()

    result = -torch.log(torch.clamp(result, min=epsilon))

    loss = 0
    counter = 0

    for framenum in range(seq_length):

        nodeIDs = nodesPresent[framenum]
        nodeIDs = [int(nodeID) for nodeID in nodeIDs]

        for nodeID in nodeIDs:
            nodeID = look_up[nodeID]
            loss = loss + result[framenum, nodeID]
            counter = counter + 1

    if counter != 0:
        return loss / counter
    else:
        return loss

##################### Data related methods ######################

def removeFileExtention(file_name):
    # remove file extension (.txt) given filename
    return file_name.split('.')[0]

def addFileExtention(file_name, extention):
    # add file extension (.txt) given filename

    return file_name + '.' + extention

def clearFolder(path):
    # remove all files in the folder
    if os.path.exists(path):
        shutil.rmtree(path)
        print("Folder succesfully removed: ", path)
    else:
        print("No such path: ",path)

def deleteFile(path, file_name_list):
    # delete given file list
    for file in file_name_list:
        file_path = os.path.join(path, file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print("File succesfully deleted: ", file_path)
            else:    ## Show an error ##
                print("Error: %s file not found" % file_path)
        except OSError as e:  ## if failed, report it back to the user ##
            print ("Error: %s - %s." % (e.filename,e.strerror))

def getAllFileNames(path):
    # return all file names given directory
    files = []
    for (dirpath, dirnames, filenames) in walk(path):
        files.extend(filenames)
        break
    return files

def createDirectories(base_folder_path, folder_list):
    # create folders using a folder list and path
    for folder_name in folder_list:
        directory = os.path.join(base_folder_path, folder_name)
        if not os.path.exists(directory):
            os.makedirs(directory)


def uniqueList(l):
  # get unique elements from list
  x = []
  for a in l:
    if a not in x:
      x.append(a)
  return x


def getAngle(p1, p2):
    # return angle between two points
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return ((ang1 - ang2) % (2 * np.pi))


def vectorizeSeq(x_seq, PedsList_seq, lookup_seq):
    # substract first frame value to all frames for a ped. Therefore, convert absolute pos. to relative pos.
    first_values_dict = WriteOnceDict()
    vectorized_x_seq = x_seq.clone()
    for ind, frame in enumerate(x_seq):
        for ped in PedsList_seq[ind]:
            first_values_dict[ped] = frame[lookup_seq[ped], 0:2]
            vectorized_x_seq[ind, lookup_seq[ped], 0:2]  = frame[lookup_seq[ped], 0:2] - first_values_dict[ped][0:2]

    return vectorized_x_seq, first_values_dict


def translate(x_seq, PedsList_seq, lookup_seq, value):
    # translate al trajectories given x and y values
    vectorized_x_seq = x_seq.clone()
    for ind, frame in enumerate(x_seq):
        for ped in PedsList_seq[ind]:
            vectorized_x_seq[ind, lookup_seq[ped], 0:2]  = frame[lookup_seq[ped], 0:2] - value[0:2]

    return vectorized_x_seq


def revertSeq(x_seq, PedsList_seq, lookup_seq, first_values_dict):
    # convert velocity array to absolute position array
    absolute_x_seq = x_seq.clone()
    for ind, frame in enumerate(x_seq):
        for ped in PedsList_seq[ind]:
            absolute_x_seq[ind, lookup_seq[ped], 0:2] = frame[lookup_seq[ped], 0:2] + first_values_dict[ped][0:2]

    return absolute_x_seq


def rotate(origin, point, angle):
        """
        Rotate a point counterclockwise by a given angle around a given origin.

        The angle should be given in radians.
        """
        ox, oy = origin
        px, py = point

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        #return torch.cat([qx, qy])
        return [qx, qy]

def timeLrScheduler(optimizer, epoch, lr_decay=0.5, lr_decay_epoch=10):
    """Decay learning rate by a factor of lr_decay every lr_decay_epoch epochs"""
    if epoch % lr_decay_epoch:
        return optimizer

    print("Optimizer learning rate has been decreased.")

    for param_group in optimizer.param_groups:
        param_group['lr'] *= (1. / (1. + lr_decay * epoch))
    return optimizer


def sample_validation_data(x_seq, Pedlist, grid, args, net, look_up, num_pedlist, dataloader):
    '''
    The validation sample function
    params:
    x_seq: Input positions
    Pedlist: Peds present in each frame
    args: arguments
    net: The model
    num_pedlist : number of peds in each frame
    look_up : lookup table for determining which ped is in which array index
    '''
    # Number of peds in the sequence
    numx_seq = len(look_up)

    total_loss = 0

    # Construct variables for hidden and cell states
    with torch.no_grad():
        hidden_states = Variable(torch.zeros(numx_seq, net.args.rnn_size))
        if args.use_cuda:
            hidden_states = hidden_states.cuda()
        if not args.gru:
            cell_states = Variable(torch.zeros(numx_seq, net.args.rnn_size))
            if args.use_cuda:
                cell_states = cell_states.cuda()
        else:
            cell_states = None


        ret_x_seq = Variable(torch.zeros(args.seq_length, numx_seq, 2))

        # Initialize the return data structure
        if args.use_cuda:
            ret_x_seq = ret_x_seq.cuda()

        ret_x_seq[0] = x_seq[0]

        # For the observed part of the trajectory
        for tstep in range(args.seq_length -1):
            loss = 0
            # Do a forward prop
            out_, hidden_states, cell_states = net(x_seq[tstep].view(1, numx_seq, 2), [grid[tstep]], hidden_states, cell_states, [Pedlist[tstep]], [num_pedlist[tstep]], dataloader, look_up)
            # loss_obs = Gaussian2DLikelihood(out_obs, x_seq[tstep+1].view(1, numx_seq, 2), [Pedlist[tstep+1]])

            # Extract the mean, std and corr of the bivariate Gaussian
            mux, muy, sx, sy, corr = getCoef(out_)
            # Sample from the bivariate Gaussian
            next_x, next_y = sampleGaussian2d(mux.data, muy.data, sx.data, sy.data, corr.data, Pedlist[tstep], look_up)
            ret_x_seq[tstep + 1, :, 0] = next_x
            ret_x_seq[tstep + 1, :, 1] = next_y
            loss = Gaussian2DLikelihood(out_[0].view(1, out_.size()[1], out_.size()[2]), x_seq[tstep].view(1, numx_seq, 2), [Pedlist[tstep]], look_up)
            total_loss += loss


    return ret_x_seq, total_loss / args.seq_length


def sampleValidationDataVanilla(x_seq, Pedlist, args, net, look_up, num_pedlist, dataloader):
    '''
    The validation sample function for vanilla method
    params:
    x_seq: Input positions
    Pedlist: Peds present in each frame
    args: arguments
    net: The model
    num_pedlist : number of peds in each frame
    look_up : lookup table for determining which ped is in which array index

    '''
    # Number of peds in the sequence
    numx_seq = len(look_up)

    total_loss = 0

    # Construct variables for hidden and cell states
    hidden_states = Variable(torch.zeros(numx_seq, net.args.rnn_size), volatile=True)
    if args.use_cuda:
        hidden_states = hidden_states.cuda()
    if not args.gru:
        cell_states = Variable(torch.zeros(numx_seq, net.args.rnn_size), volatile=True)
        if args.use_cuda:
            cell_states = cell_states.cuda()
    else:
        cell_states = None


    ret_x_seq = Variable(torch.zeros(args.seq_length, numx_seq, 2), volatile=True)

    # Initialize the return data structure
    if args.use_cuda:
        ret_x_seq = ret_x_seq.cuda()

    ret_x_seq[0] = x_seq[0]

    # For the observed part of the trajectory
    for tstep in range(args.seq_length -1):
        loss = 0
        # Do a forward prop
        out_, hidden_states, cell_states = net(x_seq[tstep].view(1, numx_seq, 2), hidden_states, cell_states, [Pedlist[tstep]], [num_pedlist[tstep]], dataloader, look_up)
        # loss_obs = Gaussian2DLikelihood(out_obs, x_seq[tstep+1].view(1, numx_seq, 2), [Pedlist[tstep+1]])

        # Extract the mean, std and corr of the bivariate Gaussian
        mux, muy, sx, sy, corr = getCoef(out_)
        # Sample from the bivariate Gaussian
        next_x, next_y = sampleGaussian2d(mux.data, muy.data, sx.data, sy.data, corr.data, Pedlist[tstep], look_up)
        ret_x_seq[tstep + 1, :, 0] = next_x
        ret_x_seq[tstep + 1, :, 1] = next_y
        loss = Gaussian2DLikelihood(out_[0].view(1, out_.size()[1], out_.size()[2]), x_seq[tstep].view(1, numx_seq, 2), [Pedlist[tstep]], look_up)
        total_loss += loss


    return ret_x_seq, total_loss / args.seq_length


def rotateTrajWithTargetPed(x_seq, angle, PedsList_seq, lookup_seq):
    # rotate sequence given angle
    origin = (0, 0)
    vectorized_x_seq = x_seq.clone()
    for ind, frame in enumerate(x_seq):
        for ped in PedsList_seq[ind]:
            point = frame[lookup_seq[ped], 0:2]
            rotated_point = rotate(origin, point, angle)
            vectorized_x_seq[ind, lookup_seq[ped], 0] = rotated_point[0]
            vectorized_x_seq[ind, lookup_seq[ped], 1] = rotated_point[1]
    return vectorized_x_seq


# *Kevin Murphy*
def getNormalizedL2Distance(ret_nodes, nodes, assumedNodesPresent, trueNodesPresent, use_cuda, look_up):
    '''
    Parameters
    ==========

    ret_nodes : A tensor of shape pred_length x numNodes x 2
    Contains the predicted positions for the nodes

    nodes : A tensor of shape pred_length x numNodes x 2
    Contains the true positions for the nodes

    nodesPresent lists: A list of lists, of size pred_length
    Each list contains the nodeIDs of the nodes present at that time-step

    look_up : lookup table for determining which ped is in which array index

    Returns
    =======

    Error : normalized euclidean distance between predicted trajectory and the true trajectory
    '''
    pred_length = ret_nodes.size()[0]
    error = torch.zeros(pred_length)

    if use_cuda:
        error = error.cuda()

    for frame_idx in range(pred_length):
        num_peds_in_frame = 0

        for ped_id in assumedNodesPresent[frame_idx]:
            ped_id = int(ped_id)

            if ped_id not in trueNodesPresent[frame_idx]:
                continue

            ped_idx = look_up[ped_id]

            pred_pos = ret_nodes[frame_idx, ped_idx, :]
            true_pos = nodes[frame_idx, ped_idx, :]

            # Implementation based on resource 1 in guide.md:
            norm_pred_pos = pred_pos#(pred_pos / get_size(pred_pos)) if get_size(pred_pos) != 0 else pred_pos
            norm_true_pos = true_pos#(true_pos / get_size(true_pos)) if get_size(true_pos) != 0 else true_pos
            normalized_l2_dist = torch.norm(norm_pred_pos - norm_true_pos, p=2)

            '''
            # Implementation based on resource 2 in guide.md:
            num = get_squared_norm((get_mean_shift(pred_pos)-get_mean_shift(true_pos)))
            denom = get_squared_norm(get_mean_shift(pred_pos)) + get_squared_norm(get_mean_shift(true_pos))
            normalized_l2_dist = 0.5 * num / denom
            '''

            error[frame_idx] += normalized_l2_dist
            num_peds_in_frame += 1

        if num_peds_in_frame > 1:
            error[frame_idx] = error[frame_idx] / num_peds_in_frame

    return error

def getMeanShift(tensor):
    '''
    Returns mean shifted version of input tensor
    :return: tensor - mean(tensor)
    '''
    return tensor - torch.mean(tensor)


def getSquaredNorm(tensor, p=2):
    '''
    Returns squared norm of input tensor
    '''
    return torch.norm(tensor, p=p) ** 2

def getSize(tensor):
    '''
    Returns size of input tensor
    :return: size(tensor)
    '''
    return torch.norm(tensor)

def loadData(dataset_path, seq_length, keep_every, valid_percentage, batch_size, max_val_size, persons_to_keep, filename=None):
    '''
    Dataset that creates and returns train/validation dataloaders of all datasets in dataset path
    :param dataset_path: path of datasets
    :param seq_length: original dataset sequence length (ped_data = 20 and basketball_data = 50)
    :param keep_every: # keeps every keep_every entries of the input dataset (to recreate Kevin Murphy's work, needs be set to 5)
    :param valid_percentage: percentage of validation data
    :param batch_size: dataset batch_size
    :param max_val_size: maximum size of validation (=1000)
    :param persons_to_keep: binary list indicating persons to consider in dataset (for Kevin Murphy's setting = [1,1,1,1,1,1,0,0,0,0,0])
    :return: train_loader and valid_loader
    '''
    if filename is not None and os.path.exists(filename):
        print(f'Dataset filename is given and the object exists. Loading from the dataset object file {filename}')
        all_datasets = torch.load(filename)
    else:
        print('Dataset filename is not given or the object does not exist. Loading from the raw files')
        # Determine the train files path
        files_list = [f for f in listdir(dataset_path) if isfile(join(dataset_path, f))]
        # Concat datasets associated to the files in train path
        all_datasets = ConcatDataset([TrajectoryDataset(join(dataset_path, file), seq_length, keep_every, persons_to_keep) for file in files_list])
        if filename is not None and not os.path.exists(filename):
            print(f'Saving the dataset object to file {filename}')
            torch.save(all_datasets, filename)

    valid_size = int(len(all_datasets) * valid_percentage / 100)
    if valid_size > max_val_size:
        valid_size = max_val_size
    train_size = len(all_datasets) - valid_size
    train_dataset, valid_dataset = torch.utils.data.random_split(all_datasets, [train_size, valid_size])
    # Create the data loader objects
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False,
                              collate_fn=lambda x: x)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False,
                              collate_fn=lambda x: x)

    # Debug: overfit to a single sequence
    #valid_loader = train_loader

    return train_loader, valid_loader


def getFolderName(folder_path, dataset):
    if dataset in ['basketball', 'basketball_small', 'basketball_total']:
        return folder_path.split('/')[-3]
    return folder_path.split('/')[-1]