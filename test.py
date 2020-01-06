import pickle
import time
import subprocess
from helper import *
from grid import getSequenceGridMask, getGridMask
from utils import DotDict
import sacred
import utils
from sacred.observers import MongoObserver
from trajectory_dataset import *
from os import listdir
from os.path import isfile, join
import json

ex = sacred.Experiment('test', ingredients=[utils.common_ingredient, utils.dataset_ingredient, utils.model_ingredient])
#ex.observers.append(MongoObserver.create(url='localhost:27017', db_name='MY_DB'))


@ex.config
def cfg():
    # Size of batch
    batch_size = 1

    # Model to be loaded
    epoch = 14  # 'Epoch of model to be loaded'

    # Number of iteration -> we are trying many times to get lowest test error derived from observed part and prediction of observed
    # part.Currently it is useless because we are using direct copy of observed part and no use of prediction.Test error will be 0.
    iteration = 1  # 'Number of iteration to create test file (smallest test error will be selected)'

    dataset_filename = None  # If given, loads dataset from this path instead of processing the files.
    if dataset_filename is not None:
        os.makedirs(os.path.dirname(dataset_filename), exist_ok=True)

    saved_model_dir = None # If given, loads trained model of this path.

    results_dir = None # If given, saves results in this path.

@ex.named_config
def debug(common):
    os.makedirs('/tmp/team_lstm_out', exist_ok=True)
    common['save_dir']='/tmp/team_lstm_out'


def init(seed, _config, _run):
    # Next five lines are to call args.seq_length instead of args.common.seq_length
    config = {k: v for k, v in _config.items()}
    common_config = config['common']
    config.pop('common')
    for k, v in common_config.items():
        assert k not in config
        config[k] = v

    dataset_config = config['dataset']
    config.pop('dataset')
    for k, v in dataset_config.items():
        assert k not in config
        config[k] = v

    args = DotDict(config)
    # utils.seedAll(seed) # TODO: implement seedAll
    return args


def testHelper(net, test_loader, sample_args, saved_args, result_directory):

    num_batches = math.floor(len(test_loader.dataset) / sample_args.batch_size)

    # Variable to maintain total error
    total_error = 0
    final_error = 0
    # *Kevin Murphy*
    norm_l2_dists = torch.zeros(sample_args.obs_length)
    if sample_args.use_cuda:
        norm_l2_dists = norm_l2_dists.cuda()

    for batch_idx, batch in enumerate(test_loader):

        for batch_item in batch:
            start = time.time()
            # Unpack batch
            if saved_args.model == 'social':
                x_seq, _, peds_list, _, lookup, _, _ = batch_item
            else:
                x_seq, peds_list, _, lookup, _, _ = batch_item
            
            # Will be used for error calculation
            orig_x_seq = x_seq.clone()

            # *CUDA*
            if sample_args.use_cuda:
                x_seq = x_seq.cuda()
                orig_x_seq = orig_x_seq.cuda()
            
            # The sample function
            sampled_x_seq = sample(batch_item, net, sample_args, saved_args)

            if sample_args.use_cuda:
                sampled_x_seq = sampled_x_seq.cuda()

            # *ORIGINAL TEST*
            total_error += getMeanError(sampled_x_seq[sample_args.obs_length:].data,
                                        orig_x_seq[sample_args.obs_length:].data,
                                        peds_list[sample_args.obs_length:],
                                        peds_list[sample_args.obs_length:],
                                        sample_args.use_cuda, lookup)

            final_error += getFinalError(sampled_x_seq[sample_args.obs_length:].data,
                                        orig_x_seq[sample_args.obs_length:].data,
                                        peds_list[sample_args.obs_length:],
                                        peds_list[sample_args.obs_length:], lookup)

            # *Kevin Murphy*
            norm_l2_dists += getNormalizedL2Distance(sampled_x_seq[:sample_args.obs_length].data,
                                                        orig_x_seq[:sample_args.obs_length].data,
                                                        peds_list[:sample_args.obs_length],
                                                        peds_list[:sample_args.obs_length],
                                                    sample_args.use_cuda, lookup)
            
            # Save predicted results
            #import pdb; pdb.set_trace()
            res = {'pred_pos': sampled_x_seq[:sample_args.obs_length].tolist(),
                   'gt_pos': orig_x_seq[:sample_args.obs_length].tolist(),
                   'persons': peds_list[:sample_args.obs_length],
                   'lookup': {int(k): v for k,v in lookup.items()}}
            os.makedirs(os.path.join(result_directory, str(sample_args.epoch)), exist_ok=True)
            with open(os.path.join(result_directory, str(sample_args.epoch), 'config.json'), "w") as fp:
                json.dump(res, fp)
            
            end = time.time()

        print(' Processed trajectory number : ', batch_idx + 1, 'out of', num_batches,
              'trajectories in time', end - start)

    return total_error, final_error, norm_l2_dists


def test(sample_args, _run):

    # For drive run; running on lab-machine
    if sample_args.results_dir is not None:
        f_prefix = sample_args.results_dir
    else:
        f_prefix = '.'

    # Run sh file for folder creation
    if not os.path.isdir("log/"):
      #print("Directory creation script is running...")
      subprocess.call([f_prefix+'/make_directories.sh'])

    # Save directory
    if sample_args.saved_model_dir is not None:
        save_directory = sample_args.saved_model_dir
    else:
        #save_directory = os.path.join(f_prefix, save_dir, method_name, model_name)
        save_directory = 'models/202'

    # Define the path for the config file for saved args
    with open(os.path.join(save_directory, 'config.json'), 'rb') as f:
        saved_args = DotDict(json.load(f))

    # Set model names to load
    model_type = saved_args.model
    method_name = getMethodName(model_type)
    model_name = "LSTM"
    save_tar_name = method_name+"_lstm_model_"

    # Extract experiment number
    exp_num = save_directory.split('/')[-1]

    # Set results/errors directories for plotting in the future
    result_directory = os.path.join(f_prefix, 'results/', exp_num)#, str(sample_args.epoch))
    error_directory = os.path.join(f_prefix, 'errors/', exp_num)#, str(sample_args.epoch))

    seq_len = sample_args.seq_length

    #import pdb; pdb.set_trace()

    # Determine the test files path
    # Debug: manually debug the code
    #path = 'data/basketball/small_test'
    path = sample_args.test_dataset_path

    # Load data
    datasets = buildDatasets(dataset_path=path,
                             seq_length=sample_args.orig_seq_len,
                             keep_every=sample_args.keep_every,
                             persons_to_keep=sample_args.persons_to_keep, 
                             filename=sample_args.dataset_filename)
    
    test_loader, _ = loadData(all_datasets=datasets,
                              valid_percentage=0,
                              batch_size=sample_args.batch_size,
                              max_val_size=0,
                              args=saved_args)

    num_batches = math.floor(len(test_loader.dataset) / sample_args.batch_size)
    smallest_err = 100000
    smallest_err_iter_num = -1

    submission_store = []  # store submission data points (txt)
    result_store = []  # store points for plotting


    for iteration in range(sample_args.iteration):
        # Initialize net
        net = getModel(saved_args, True)

        if sample_args.use_cuda:
            net = net.cuda()

        # Get the checkpoint path
        checkpoint_path = os.path.join(save_directory, save_tar_name+str(sample_args.epoch)+'.tar')
        if os.path.isfile(checkpoint_path):
            print('Loading checkpoint')
            checkpoint = torch.load(checkpoint_path)
            model_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['state_dict'])
            print('Loaded checkpoint at epoch', model_epoch)
        else:
            raise ValueError('Incorrect checkpoint: file does not exist')

    total_error, final_error, norm_l2_dists = testHelper(net, test_loader, sample_args, saved_args, result_directory)

    # Save errors
    res = {'total_error': total_error.item(),
           'final_error': final_error.item(),
           'norm_l2_dists': norm_l2_dists.tolist()}
    os.makedirs(os.path.join(error_directory, str(sample_args.epoch)), exist_ok=True)
    with open(os.path.join(error_directory, str(sample_args.epoch), 'config.json'), "w") as fp:
        json.dump(res, fp)

    if total_error < smallest_err:
        # print("**********************************************************")
        # print('Best iteration has been changed. Previous best iteration: ', smallest_err_iter_num + 1, 'Error: ', smallest_err / num_batches)
        # print('New best iteration : ', iteration + 1, 'Error: ',total_error / num_batches)
        smallest_err_iter_num = iteration
        smallest_err = total_error

    print('Iteration:', iteration + 1, ' Total testing mean error of the predicted part is ',
          total_error / num_batches)
    print('Iteration:', iteration + 1, 'Total testing final error of the predicted part is ',
          final_error / num_batches)
    # *Kevin Murphy*
    norm_l2_dists = norm_l2_dists / num_batches
    for i in range(sample_args.obs_length):
        print('Normalized l2 distances for step %d is %f' % (i, norm_l2_dists.data[i]))
    print('Smallest error iteration:', smallest_err_iter_num + 1)


def sample(batch_item, net, args, saved_args):
    '''
    The sample function
    params:
    batch_item: A single batch item
    net: The model
    args: Model arguments
    saved_args: Configuration arguments used for training the model
    '''
    #import pdb; pdb.set_trace()
    # Unpack batch
    if saved_args.model == 'social':
        x_seq, grids, peds_list, num_peds_list, lookup, dataset_dim, init_values_dict = batch_item
    else:
        x_seq, peds_list, num_peds_list, lookup, dataset_dim, init_values_dict = batch_item

    if args.use_cuda:
        x_seq = x_seq.cuda()

    # Number of peds in the sequence
    num_persons = len(lookup)

    with torch.no_grad():
        # Initialize hidden and cell states
        hidden_states = torch.zeros(num_persons, args.rnn_size)
        cell_states = torch.zeros(num_persons, args.rnn_size)
        # Initialize the return data structure
        sampled_x_seq = torch.zeros(args.obs_length + args.pred_length, num_persons, 2)
        if args.use_cuda:
            hidden_states = hidden_states.cuda()
            cell_states = cell_states.cuda()
            sampled_x_seq = sampled_x_seq.cuda()

        # For the observed part of the trajectory
        for tstep in range(args.obs_length - 1):
            # Do a forward prop
            if saved_args.model == 'social':
                out_obs, hidden_states, cell_states = net([x_seq[tstep].view(1, num_persons, 2),
                                                          [grids[tstep]], [peds_list[tstep]], 
                                                          [num_peds_list[tstep]], lookup,
                                                          dataset_dim, init_values_dict],
                                                          hidden_states, cell_states)
            else:
                out_obs, hidden_states, cell_states = net([x_seq[tstep].view(1, num_persons, 2),
                                                          [peds_list[tstep]], [num_peds_list[tstep]],
                                                          lookup, dataset_dim, init_values_dict],
                                                          hidden_states, cell_states)

            # Extract the mean, std and corr of the bivariate Gaussian
            mux, muy, sx, sy, corr = getCoef(out_obs)
            # Sample from the bivariate Gaussian
            next_x, next_y = sampleGaussian2d(mux.data, muy.data, sx.data, sy.data, corr.data, peds_list[tstep], lookup)
            # Save predicted positions into return data structure
            sampled_x_seq[tstep + 1, :, 0] = next_x
            sampled_x_seq[tstep + 1, :, 1] = next_y

        # Set last seen grid
        if saved_args.model == 'social':
            prev_grid = grids[args.obs_length - 1].clone()

        # For the predicted part of the trajectory
        for tstep in range(args.obs_length - 1, args.pred_length + args.obs_length - 1):
            # Do a forward prop
            if tstep == args.obs_length - 1:
                net_input = x_seq[tstep].view(1, num_persons, 2)
            else:
                net_input = sampled_x_seq[tstep].view(1, num_persons, 2)
            
            if saved_args.model == 'social':
                outputs, hidden_states, cell_states = net([net_input,
                                                          [prev_grid], [peds_list[tstep]], [num_peds_list[tstep]],
                                                          lookup, dataset_dim, init_values_dict],
                                                          hidden_states, cell_states) 
            else:
                outputs, hidden_states, cell_states = net([net_input, 
                                                          [peds_list[tstep]], [num_peds_list[tstep]],
                                                          lookup, dataset_dim, init_values_dict],
                                                          hidden_states, cell_states)
                

            # Extract the mean, std and corr of the bivariate Gaussian
            mux, muy, sx, sy, corr = getCoef(outputs)
            # Sample from the bivariate Gaussian
            next_x, next_y = sampleGaussian2d(mux.data, muy.data, sx.data, sy.data, corr.data, peds_list[tstep],
                                              lookup)

            # Save predicted positions into return data structure
            sampled_x_seq[tstep + 1, :, 0] = next_x
            sampled_x_seq[tstep + 1, :, 1] = next_y

            # Compute grid masks based on the predicted positions
            if saved_args.model == 'social':
                cor_pred_positions = getCorPredPositions(sampled_x_seq[tstep+1], peds_list[tstep+1], lookup, args)
                prev_grid = getGridMask(cor_pred_positions.data.cpu(), dataset_dim, len(peds_list[tstep+1]), saved_args.neighborhood_size, saved_args.grid_size)
                prev_grid = torch.from_numpy(prev_grid).float()
                if args.use_cuda:
                    prev_grid = prev_grid.cuda()

        # Revert the points back to the original space
        #sampled_x_seq = revertSeq(sampled_x_seq.data.cpu(), peds_list, lookup, init_values_dict)
        return sampled_x_seq

def getCorPredPositions(pred_positions, peds_list, lookup, args):
    peds_list = [int(ped) for ped in peds_list]
    peds_indices = torch.LongTensor([lookup[ped] for ped in peds_list])
    if args.use_cuda:
        peds_indices = peds_indices.cuda()
    return torch.index_select(pred_positions, 0, peds_indices)

def submissionPreprocess(dataloader, sampled_x_seq, pred_length, obs_length, target_id):
    seq_lenght = pred_length + obs_length

    # begin and end index of obs. frames in this seq.
    begin_obs = (dataloader.frame_pointer - seq_lenght)
    end_obs = (dataloader.frame_pointer - pred_length)

    # get original data for frame number and ped ids
    observed_data = dataloader.orig_data[dataloader.dataset_pointer][begin_obs:end_obs, :]
    frame_number_predicted = dataloader.get_frame_sequence(pred_length)
    ret_x_seq_c = sampled_x_seq.copy()
    ret_x_seq_c[:, [0, 1]] = ret_x_seq_c[:, [1, 0]]  # x, y -> y, x
    repeated_id = np.repeat(target_id, pred_length)  # add id
    id_integrated_prediction = np.append(repeated_id[:, None], ret_x_seq_c, axis=1)
    frame_integrated_prediction = np.append(frame_number_predicted[:, None], id_integrated_prediction,
                                            axis=1)  # add frame number
    result = np.append(observed_data, frame_integrated_prediction, axis=0)

    return result


@ex.automain
def experiment(_seed, _config, _run):
    args = init(_seed, _config, _run)
    test(args, _run)