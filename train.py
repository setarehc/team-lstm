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
import glob

ex = sacred.Experiment('train', ingredients=[utils.common_ingredient, utils.dataset_ingredient, utils.model_ingredient])
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

    # Dropout probability parameter
    dropout = 0.5  # dropout probability

    # Maximum number of pedestrians to be considered
    maxNumPeds = 27  # Maximum Number of Pedestrians

    # Lambda regularization parameter (L2)
    lambda_param = 0.0005  # L2 regularization parameter

    # Store grids in epoch 0 and use further.2 times faster -> Intensive memory use around 12 GB
    grid = True  # Whether store grids and use further epoch

    # Percentage of validation data out of all the data
    valid_percentage = 10
    max_val_size = 1000   # If 10% of size of all the data > 1000, consider only a 1000

    dataset_filename = None  # If given, will load the dataset from this path instead of processing the files.
    if dataset_filename is not None:
        os.makedirs(os.path.dirname(dataset_filename), exist_ok=True)

    # If we need to continue training a pre-trained model, load it from saved_model_dir
    saved_model_dir = None 

@ex.named_config
def debug(common):
    os.makedirs('/tmp/team_lstm_out', exist_ok=True)
    common['save_dir']='/tmp/team_lstm_out'


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

    # Next five lines are to call args.graph_type instead of args.model.graph_type
    model_config = config['model']
    config.pop('model')
    for k, v in model_config.items():
        assert k not in config
        config[k] = v

    args = DotDict(config)
    # utils.seedAll(seed) # TODO: implement seedAll
    return args

class MyDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

def train(args, _run):

    # Set directory to save the trained model
    if args.saved_model_dir is not None:
            save_directory = args.saved_model_dir
    else:
        inner_dir = args.save_prefix
        if inner_dir is None:
            inner_dir = 'tmp' if _run._id is None else str(_run._id)
        save_directory = os.path.join(args.save_dir, inner_dir)
        if os.path.isdir(save_directory):
            shutil.rmtree(save_directory)

    # Load data
    datasets = buildDatasets(dataset_path=args.train_dataset_path,
                             seq_length=args.orig_seq_len,
                             keep_every=args.keep_every,
                             persons_to_keep=args.persons_to_keep,
                             filename=args.dataset_filename)

    train_loader, valid_loader = loadData(all_datasets=datasets,
                                          valid_percentage=args.valid_percentage,
                                          batch_size=args.batch_size,
                                          max_val_size=args.max_val_size,
                                          args=args)
    
    model_type = args.model
    method_name = getMethodName(model_type)
    model_name = "LSTM"
    save_tar_name = method_name+"_lstm_model_"

    # Save the arguments int the config file
    os.makedirs(save_directory, exist_ok=True) #TODO: fix this!
    with open(os.path.join(save_directory, 'config.json'), 'w') as f:
        json.dump(args, f)

    # Path to store the checkpoint file (trained model)
    def checkpoint_path(x):
        return os.path.join(save_directory, save_tar_name + str(x) + '.tar')
    
    # If we should continue training a pre-trained model
    if args.saved_model_dir is not None:
        # Load the config file of model:
        with open(os.path.join(save_directory, 'config.json'), 'rb') as f:
            saved_args = DotDict(json.load(f))
        # Build empty net with the above config:
        net = getModel(saved_args, True)
        #net = MyDataParallel(net)
        if args.use_cuda:
            net = net.cuda()
        # Get the last trained epoch to continue training:
        list_of_saved_models = glob.glob(os.path.join(args.saved_model_dir, '*.tar'))
        latest_model = get_latets_file(list_of_saved_models)
        checkpoint = torch.load(latest_model)
        init_epoch = checkpoint['epoch'] + 1
        net.load_state_dict(checkpoint['state_dict'])
        # Build empty optimizer:
        optimizer = torch.optim.Adagrad(net.parameters(), weight_decay=args.lambda_param)
        # Load the state dictionary of optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        # model creation
        if args.model == 'social':
            net = SocialModel(args)
        elif args.model == 'graph':
            net = GraphModel(args)
        elif args.model == 'vanilla':
            net = VanillaModel(args)
        else:
            raise ValueError(f'Unexpected value for args.model ({args.model})')
        #net = MyDataParallel(net)
        if args.use_cuda:
            net = net.cuda()
        init_epoch = 0
        optimizer = torch.optim.Adagrad(net.parameters(), weight_decay=args.lambda_param)

    # Training
    for epoch in range(init_epoch, args.num_epochs+init_epoch):
        print('****************Training epoch beginning******************')
        loss_epoch = 0
        num_seen_sequences = 0

        # For each batch
        for batch_idx, batch in enumerate(train_loader):
            start = time.time()

            loss_batch = 0
            #import pdb; pdb.set_trace()
            # Check if last batch is shorter that batch_size
            curr_batch_length = 0
            if args.model == 'graph':
                curr_batch_length = batch[0].size(1)
            else:
                curr_batch_length = len(batch)

            if curr_batch_length < args.batch_size:
                continue
            
            batch = net.toCudaBatch(batch)

            optimizer.zero_grad()

            # Forward prop
            outputs, _, _ = net(batch)
            
            num_seen_sequences += curr_batch_length #TODO: Check if correct

            # Compute loss
            loss = net.computeLossBatch(outputs, batch)
            loss_batch += loss #TODO: Check if needs devision by batch_size or not

            # Compute gradients
            loss_batch.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)

            # Update parameters
            optimizer.step()

            end = time.time()
            loss_epoch += loss_batch.item()

            num_batches = math.floor(len(train_loader.dataset) / args.batch_size)

            print('{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}'.format(
                epoch * num_batches + batch_idx,
                args.num_epochs * num_batches,
                epoch,
                loss_batch.item(), end - start))

        loss_epoch /= num_batches

        # Sacred metrics plot
        _run.log_scalar(metric_name='train.loss', value=loss_epoch, step=epoch)
        
        if args.validate:
            # Validate
            if len(valid_loader) > 0:
                mux, muy, sx, sy, corr = getCoef(outputs)
                _run.log_scalar(metric_name='valid.mux', value=torch.mean(mux).item(), step=epoch)
                _run.log_scalar(metric_name='valid.muy', value=torch.mean(muy).item(), step=epoch)
                _run.log_scalar(metric_name='valid.sx', value=torch.mean(sx).item(), step=epoch)
                _run.log_scalar(metric_name='valid.sy', value=torch.mean(sy).item(), step=epoch)
                #import pdb; pdb.set_trace()
                valid_loss = validLoss(net, valid_loader, args)
                total_error, final_error, norm_l2_dists = testHelper(net, valid_loader, args, args, None)
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
        num_batches = 0
        total_loss = 0
        
        for batch_idx, batch in enumerate(valid_loader):
            
            if args.model == 'graph':
                curr_batch_length = batch[0].size(1)
            else:
                curr_batch_length = len(batch)
            # Check if last batch is shorter that batch_size
            if curr_batch_length < args.batch_size:
                continue
            
            batch = net.toCudaBatch(batch)

            # Forward prop
            outputs, _, _ = net(batch)

            # Increment number of seen sequences
            num_batches += 1 #TODO: Check if correct

            # Compute loss
            loss = net.computeLossBatch(outputs, batch)
            total_loss += loss.item() #TODO: Check if needs devision by batch_size or not

        return total_loss / num_batches

@ex.automain
def experiment(_seed, _config, _run):
    args = init(_seed, _config, _run)
    train(args, _run)
