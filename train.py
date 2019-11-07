import time
import pickle
import subprocess
from grid import getSequenceGridMask
from helper import *
from utils import DotDict
import sacred
import utils
from sacred.observers import MongoObserver
from trajectory_dataset import *
from test import testHelper
import copy
import db
import json

ex = sacred.Experiment('train', ingredients=[utils.common_ingredient, utils.dataset_ingredient])
db.init(ex)
#ex.observers.append(MongoObserver.create(url='localhost:27017', db_name='MY_DB'))
ex.captured_out_filter = lambda text: 'Output capturing turned off.'

@ex.config
def cfg():
    # Size of batch
    batch_size = 5

    # Number of epochs
    num_epochs = 30

    # Frequency at which the model should be saved parameter
    save_every = 400  # save frequency

    # TODO: (resolve) Clipping gradients for now. No idea whether we should
    # Gradient value at which it should be clipped
    grad_clip = 10.  # clip gradients at this value

    # Learning rate parameter
    learning_rate = 0.003  # learning rate

    # Decay rate for the learning rate parameter
    decay_rate = 0.95  # decay rate for rmsprop

    # Dropout not implemented.
    # Dropout probability parameter
    dropout = 0.5  # dropout probability

    # Maximum number of pedestrians to be considered
    maxNumPeds = 27  # Maximum Number of Pedestrians

    # Lambda regularization parameter (L2)
    lambda_param = 0.0005  # L2 regularization parameter

    # store grids in epoch 0 and use further.2 times faster -> Intensive memory use around 12 GB
    grid = True  # Whether store grids and use further epoch

    # Percentage of validation data out of all the data
    valid_percentage = 10
    max_val_size = 1000   # If 10% of size of all the data > 1000, consider only a 1000

    # Method selection
    method = 1  # 'Method of lstm will be used (1 = social lstm, 2 = obstacle lstm, 3 = vanilla lstm)'

    dataset_filename = None  # If given, will load the dataset from this path instead of processing the files.
    if dataset_filename is not None:
        os.makedirs(os.path.dirname(dataset_filename), exist_ok=True)


def init(seed, _config, _run):
    # Next five lines are to call args.use_cuda instead of args.common.use_cuda
    config = {k:v for k,v in _config.items()}
    common_config = config['common']
    config.pop('common')
    for k, v in common_config.items():
        assert k not in config
        config[k] = v

    # Next five lines are to call args.seq_length instead of args.dataset.seq_length
    dataset_config = config['dataset']
    config.pop('dataset')
    for k, v in dataset_config.items():
        assert k not in config
        config[k] = v

    args = DotDict(config)
    # utils.seedAll(seed) # TODO: implement seedAll
    return args


def train(args, _run):
    origin = (0, 0)
    reference_point = (0, 1)

    # Set directory to save the trained model
    inner_dir = args.save_prefix
    if inner_dir is None:
        inner_dir = 'tmp' if _run._id is None else str(_run._id)
    save_directory = os.path.join(args.save_dir, inner_dir)
    if os.path.isdir(save_directory):
        shutil.rmtree(save_directory)

    train_loader, valid_loader = loadData(args.train_dataset_path, args.orig_seq_len, args.keep_every, args.valid_percentage, args.batch_size, args.max_val_size, args.persons_to_keep, filename=args.dataset_filename)

    model_name = "LSTM"
    method_name = "SOCIALLSTM"
    save_tar_name = method_name + "_lstm_model_"
    if args.gru:
        model_name = "GRU"
        save_tar_name = method_name + "_gru_model_"

    # Save the arguments int the config file
    os.makedirs(save_directory, exist_ok=True) #TODO: fix this!
    with open(os.path.join(save_directory, 'config.json'), 'w') as f:
        json.dump(args, f)

    # Path to store the checkpoint file (trained model)
    def checkpoint_path(x):
        return os.path.join(save_directory, save_tar_name + str(x) + '.tar')

    # model creation
    if args.model == 'social':
        net = SocialModel(args)
    elif args.model == 'graph':
        net = GraphModel(args)
    else:
        raise ValueError(f'Unexpected value for args.model ({args.model})')
    if args.use_cuda:
        net = net.cuda()

    optimizer = torch.optim.Adagrad(net.parameters(), weight_decay=args.lambda_param)

    num_batch = 0

    # Training
    for epoch in range(args.num_epochs):
        print('****************Training epoch beginning******************')
        loss_epoch = 0
        num_seen_sequences = 0

        # For each batch
        for batch_idx, batch in enumerate(train_loader):
            start = time.time()

            loss_batch = 0

            # Check if last batch is shorter that batch_size
            # batch_size = len(batch) if (len(batch) < args.batch_size) else args.batch_size
            if len(batch) < args.batch_size:
                continue

            # For each sequence
            for sequence in range(args.batch_size):
                # Get the data corresponding to the current sequence
                x_seq, num_peds_list_seq, peds_list_seq, folder_path = batch[sequence]

                # Dense vector (tensor) creation
                x_seq, lookup_seq = convertToTensor(x_seq, peds_list_seq)

                # Get processing file name and then get dimensions of file
                folder_name = getFolderName(folder_path, args.dataset)
                dataset_data = dataset_dimensions[folder_name]

                # Grid mask calculation and storage depending on grid parameter
                grid_seq = getSequenceGridMask(x_seq, dataset_data, peds_list_seq, args.neighborhood_size,
                                               args.grid_size, args.use_cuda)

                # Replace relative positions with true positions in x_seq
                x_seq, _ = vectorizeSeq(x_seq, peds_list_seq, lookup_seq)

                if args.use_cuda:
                    x_seq = x_seq.cuda()

                # Number of peds in this sequence
                numNodes = len(lookup_seq)

                hidden_states = Variable(torch.zeros(numNodes, args.rnn_size))
                if args.use_cuda:
                    hidden_states = hidden_states.cuda()

                cell_states = Variable(torch.zeros(numNodes, args.rnn_size))
                if args.use_cuda:
                    cell_states = cell_states.cuda()

                # Zero out gradients
                net.zero_grad()
                optimizer.zero_grad()

                # Forward prop
                outputs, _, _ = net(x_seq, grid_seq, hidden_states, cell_states, peds_list_seq, num_peds_list_seq,
                                    lookup_seq)

                # Increment number of seen sequences
                num_seen_sequences += 1

                # Debug


                # Compute loss
                loss = Gaussian2DLikelihood(outputs, x_seq, peds_list_seq, lookup_seq)
                loss_batch += loss.item()

                # Free the memory
                # *basketball*
                del x_seq
                del hidden_states
                del cell_states
                torch.cuda.empty_cache()

                # Compute gradients
                loss.backward()

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)

                # Update parameters
                optimizer.step()

            end = time.time()
            loss_epoch += loss_batch
            num_batch += 1

            num_batches = math.floor(len(train_loader.dataset) / args.batch_size)

            print('{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}'.format(
                epoch * num_batches + batch_idx,
                args.num_epochs * num_batches,
                epoch,
                loss_batch, end - start))

            '''
            if args.validate:
                # Validate
                if batch_idx % 5000 == 0:
                    if len(valid_loader) > 0:
                        #TEST
                        t_dataset, _ = torch.utils.data.random_split(all_datasets, [1000, len(all_datasets)-1000])
                        # Create the data loader objects
                        t_loader = DataLoader(t_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                                  pin_memory=False,
                                                  collate_fn=lambda x: x)
                        t_loss = validLoss(net, t_loader, args)
                        _run.log_scalar(metric_name='t.loss', value=t_loss, step=epoch + batch_idx / num_batches)
                        ttt_loss = loss_epoch / num_seen_sequences
                        _run.log_scalar(metric_name='ttt.loss', value=ttt_loss, step=epoch + batch_idx / num_batches)
                        valid_loss = validLoss(net, valid_loader, args)
                        total_error, final_error, norm_l2_dists = testHelper(net, valid_loader, args, args)
                        total_error = total_error.item() if isinstance(total_error, torch.Tensor) else total_error
                        final_error = final_error.item() if isinstance(final_error, torch.Tensor) else final_error
                        _run.log_scalar(metric_name='valid.loss', value=valid_loss, step=epoch+batch_idx/num_batches)
                        _run.log_scalar(metric_name='valid.total_error', value=total_error, step=epoch+batch_idx/num_batches)
                        _run.log_scalar(metric_name='valid.final_error', value=final_error, step=epoch+batch_idx/num_batches)
                        for i, l in enumerate(norm_l2_dists):
                            error = norm_l2_dists[i].item() if isinstance(norm_l2_dists[i], torch.Tensor) else norm_l2_dists[i]
                            _run.log_scalar(metric_name=f'valid.norm_l2_dist_{i}', value=error, step=epoch+batch_idx/num_batches)
            '''

        loss_epoch /= num_seen_sequences

        # Log loss values
        #log_file_curve.write("Training epoch: " + str(epoch) + " loss: " + str(loss_epoch) + '\n')

        # Sacred metrics plot
        _run.log_scalar(metric_name='train.loss', value=loss_epoch, step=epoch)
        
        if args.validate:
            # Validate
            if len(valid_loader) > 0:
                mux, muy, sx, sy, corr = getCoef(outputs)
                #import pdb; pdb.set_trace()
                _run.log_scalar(metric_name='valid.mux', value=torch.mean(mux).item(), step=epoch)
                _run.log_scalar(metric_name='valid.muy', value=torch.mean(muy).item(), step=epoch)
                _run.log_scalar(metric_name='valid.sx', value=torch.mean(sx).item(), step=epoch)
                _run.log_scalar(metric_name='valid.sy', value=torch.mean(sy).item(), step=epoch)
                valid_loss = validLoss(net, valid_loader, args)
                total_error, final_error, norm_l2_dists = testHelper(net, valid_loader, args, args)
                total_error = total_error.item() if isinstance(total_error, torch.Tensor) else total_error
                final_error = final_error.item() if isinstance(final_error, torch.Tensor) else final_error
                _run.log_scalar(metric_name='valid.loss', value=valid_loss, step=epoch)
                _run.log_scalar(metric_name='valid.total_error', value=total_error, step=epoch)
                _run.log_scalar(metric_name='valid.final_error', value=final_error, step=epoch)
                for i, l in enumerate(norm_l2_dists):
                    error = norm_l2_dists[i].item() if isinstance(norm_l2_dists[i], torch.Tensor) else norm_l2_dists[i]
                    _run.log_scalar(metric_name=f'valid.norm_l2_dist_{i}', value=error, step=epoch)


        # Save the model after each epoch
        print('Saving model')
        torch.save({
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path(epoch))

    # Close logging files
    # log_file.close()
    # log_file_curve.close()

def validLoss(net, valid_loader, args):
    '''
    Calculates log-likelihood loss on validation dataset
    :return: average log-likelihood loss
    '''
    with torch.no_grad():
        num_seen_sequences = 0
        total_loss = 0

        for batch_idx, batch in enumerate(valid_loader):

            loss_batch = 0

            # Check if last batch is shorter that batch_size
            # batch_size = len(batch) if (len(batch) < args.batch_size) else args.batch_size
            if len(batch) < args.batch_size:
                continue

            # For each sequence
            for sequence in range(args.batch_size):
                # Get the data corresponding to the current sequence
                x_seq, num_peds_list_seq, peds_list_seq, folder_path = batch[sequence]

                # Dense vector (tensor) creation
                x_seq, lookup_seq = convertToTensor(x_seq, peds_list_seq)

                # Get processing file name and then get dimensions of file
                folder_name = getFolderName(folder_path, args.dataset)
                dataset_data = dataset_dimensions[folder_name]

                # Grid mask calculation and storage depending on grid parameter
                grid_seq = getSequenceGridMask(x_seq, dataset_data, peds_list_seq, args.neighborhood_size,
                                               args.grid_size, args.use_cuda)

                # Vectorize trajectories in sequence
                x_seq, _ = vectorizeSeq(x_seq, peds_list_seq, lookup_seq)

                if args.use_cuda:
                    x_seq = x_seq.cuda()

                # Number of peds in this sequence per frame
                numNodes = len(lookup_seq)

                hidden_states = Variable(torch.zeros(numNodes, args.rnn_size))
                if args.use_cuda:
                    hidden_states = hidden_states.cuda()

                cell_states = Variable(torch.zeros(numNodes, args.rnn_size))
                if args.use_cuda:
                    cell_states = cell_states.cuda()

                # Forward prop
                outputs, _, _ = net(x_seq, grid_seq, hidden_states, cell_states, peds_list_seq, num_peds_list_seq,
                                    lookup_seq)

                # Increment number of seen sequences
                num_seen_sequences += 1

                # Compute loss
                loss = Gaussian2DLikelihood(outputs, x_seq, peds_list_seq, lookup_seq)
                loss_batch += loss.item()

            total_loss += loss_batch

        return total_loss / num_seen_sequences

@ex.automain
def experiment(_seed, _config, _run):
    args = init(_seed, _config, _run)
    train(args, _run)
