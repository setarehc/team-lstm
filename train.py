import time
import pickle
import subprocess
from grid import getSequenceGridMask
from helper import *
from types import SimpleNamespace
import sacred
import utils
from sacred.observers import MongoObserver
from trajectory_dataset import *
from os import listdir
from os.path import isfile, join

ex = sacred.Experiment('train', ingredients=[utils.data_ingredient])
ex.observers.append(MongoObserver.create(url='localhost:27017', db_name='MY_DB'))


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

    # number of validation will be used
    num_validation = 2  # Total number of validation dataset for validate accuracy

    # frequency of validation
    freq_validation = 1  # Frequency number(epoch) of validation using validation data

    # frequency of optimazer learning decay
    freq_optimizer = 8  # Frequency number(epoch) of learning decay for optimizer

    # store grids in epoch 0 and use further.2 times faster -> Intensive memory use around 12 GB
    grid = True  # Whether store grids and use further epoch


def init(seed, config, _run):
    # Next five lines are to call args.seq_length instead of args.common.seq_length
    common_config = config['common']
    config.pop('common')
    for k, v in common_config.items():
        assert k not in config
        config[k] = v

    args = SimpleNamespace(**config)
    # utils.seedAll(seed) # TODO: implement seedAll
    _run.info['args'] = args.__dict__
    return args


def train(args, _run):
    origin = (0, 0)
    reference_point = (0, 1)

    prefix = ''
    f_prefix = '.'
    if args.drive is True:
        prefix = 'drive/semester_project/social_lstm_final/'
        f_prefix = 'drive/semester_project/social_lstm_final'

    if not os.path.isdir("log/"):
        print("Directory creation script is running...")
        subprocess.call([f_prefix + '/make_directories.sh'])

    args.freq_validation = np.clip(args.freq_validation, 0, args.num_epochs)
    validation_epoch_list = list(range(args.freq_validation, args.num_epochs + 1, args.freq_validation))
    validation_epoch_list[-1] -= 1

    # Determine the train files path
    path = 'data/original/train'
    files_list = [f for f in listdir(path) if isfile(join(path, f))]
    # Concat datasets associated to the files in train path
    all_datasets = ConcatDataset([TrajectoryDataset(join(path, file)) for file in files_list])
    # Create the data loader object
    train_loader = DataLoader(all_datasets, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False,
                              collate_fn=lambda x: x)

    model_name = "LSTM"
    method_name = "SOCIALLSTM"
    save_tar_name = method_name + "_lstm_model_"
    if args.gru:
        model_name = "GRU"
        save_tar_name = method_name + "_gru_model_"

    # Log directory
    log_directory = os.path.join(prefix, 'log/')
    plot_directory = os.path.join(prefix, 'plot/', method_name, model_name)
    plot_train_file_directory = 'validation'

    # Logging files
    log_file_curve = open(os.path.join(log_directory, method_name, model_name, 'log_curve.txt'), 'w+')
    log_file = open(os.path.join(log_directory, method_name, model_name, 'val.txt'), 'w+')

    # model directory
    save_directory = os.path.join(prefix, 'model/')

    # Save the arguments int the config file
    with open(os.path.join(save_directory, method_name, model_name, 'config.pkl'), 'wb') as f:
        pickle.dump(args, f)

    # Path to store the checkpoint file
    def checkpoint_path(x):
        return os.path.join(save_directory, method_name, model_name, save_tar_name + str(x) + '.tar')

    # model creation
    net = SocialModel(args)
    if args.use_cuda:
        net = net.cuda()

    optimizer = torch.optim.Adagrad(net.parameters(), weight_decay=args.lambda_param)

    grids = []
    num_batch = 0

    num_of_datasets = len(train_loader.dataset.datasets)
    [grids.append([]) for dataset in range(num_of_datasets)]

    # Training
    for epoch in range(args.num_epochs):
        print('****************Training epoch beginning******************')
        loss_epoch = 0

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
                x_seq, numPedsList_seq, PedsList_seq, folder_path = batch[sequence]

                # Dense vector (tensor) creation
                x_seq, lookup_seq = convert_to_tensor(x_seq, PedsList_seq)

                # Get processing file name and then get dimensions of file
                folder_name = folder_path.split('/')[-1]
                dataset_data = dataset_dimensions[folder_name]

                # Grid mask calculation and storage depending on grid parameter
                grid_seq = getSequenceGridMask(x_seq, dataset_data, PedsList_seq, args.neighborhood_size,
                                               args.grid_size, args.use_cuda)

                # Vectorize trajectories in sequence
                x_seq, _ = vectorize_seq(x_seq, PedsList_seq, lookup_seq)

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

                # Zero out gradients
                net.zero_grad()
                optimizer.zero_grad()

                # Forward prop
                outputs, _, _ = net(x_seq, grid_seq, hidden_states, cell_states, PedsList_seq, numPedsList_seq,
                                    train_loader, lookup_seq)

                # Compute loss
                loss = Gaussian2DLikelihood(outputs, x_seq, PedsList_seq, lookup_seq)
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
            loss_batch = loss_batch / args.batch_size
            loss_epoch += loss_batch
            num_batch += 1

            num_batches = math.floor(train_loader.dataset.cummulative_sizes[-1] / args.batch_size)

            print('{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}'.format(
                epoch * num_batches + batch_idx,
                args.num_epochs * num_batches,
                epoch,
                loss_batch, end - start))

        loss_epoch /= num_batches
        # Log loss values
        log_file_curve.write("Training epoch: " + str(epoch) + " loss: " + str(loss_epoch) + '\n')

        # Sacred metrics plot
        _run.log_scalar(metric_name='train.loss', value=loss_epoch, step=epoch)

        # Save the model after each epoch
        print('Saving model')
        torch.save({
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path(epoch))

    # Close logging files
    log_file.close()
    log_file_curve.close()


@ex.automain
def experiment(_seed, _config, _run):
    args = init(_seed, _config, _run)
    train(args, _run)
