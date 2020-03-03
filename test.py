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

        batch = net.toCudaBatch(batch)

        # Unpack batch
        xy_posns = batch[0]
        mask = batch[1]
        
        # Predict next xy positions
        sampled_xy_posns = net.sample(batch, sample_args, saved_args)

        if sample_args.use_cuda:
                sampled_xy_posns = sampled_xy_posns.cuda()
        
        for batch_idx in range(mask.size(1)):
            start = time.time() #TODO: check whether start and end are correctly placed or not

            sampled_seq = sampled_xy_posns[:, batch_idx, :, :]
            orig_seq = xy_posns[:, batch_idx, :, :]
            persons_seq = getPedsList(mask[:, batch_idx, :])
            lookup_seq = {i: j for (i,j) in zip(range(xy_posns.size(2)),range(xy_posns.size(2)))}

            total_error += getMeanError(sampled_seq[sample_args.obs_length:].data,
                                        orig_seq[sample_args.obs_length:].data,
                                        persons_seq[sample_args.obs_length:],
                                        persons_seq[sample_args.obs_length:],
                                        sample_args.use_cuda, lookup_seq)

            final_error += getFinalError(sampled_seq[sample_args.obs_length:].data,
                                        orig_seq[sample_args.obs_length:].data,
                                        persons_seq[sample_args.obs_length:],
                                        persons_seq[sample_args.obs_length:], lookup_seq)

            # *Kevin Murphy*
            norm_l2_dists += getNormalizedL2Distance(sampled_seq[:sample_args.obs_length].data,
                                                        orig_seq[:sample_args.obs_length].data,
                                                        persons_seq[:sample_args.obs_length],
                                                        persons_seq[:sample_args.obs_length],
                                                    sample_args.use_cuda, lookup_seq)
            
            # Save predicted results
            #import pdb; pdb.set_trace()
            if result_directory is not None:
                res = {'pred_pos': sampled_seq[:sample_args.obs_length].tolist(),
                    'gt_pos': orig_seq[:sample_args.obs_length].tolist(),
                    'persons': persons_seq[:sample_args.obs_length],
                    'lookup': {int(k): v for k,v in lookup_seq.items()}}
                os.makedirs(os.path.join(result_directory, str(sample_args.epoch)), exist_ok=True)
                with open(os.path.join(result_directory, str(sample_args.epoch), 'config.json'), "w") as fp:
                    json.dump(res, fp)
            
            end = time.time()
        
        print(' Processed trajectory number : ', batch_idx + 1, 'out of', num_batches,
              'trajectories in time', end - start)

    return total_error, final_error, norm_l2_dists #TODO: needs devision by batch_size


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
        save_directory = 'models/243'

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

        #import pdb; pdb.set_trace()
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
    res = {'total_error': total_error.item()/num_batches,
           'final_error': final_error.item()/num_batches,
           'norm_l2_dists': [i/num_batches for i in norm_l2_dists.tolist()]}
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