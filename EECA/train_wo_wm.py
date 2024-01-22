"""Training models without watermark."""

import argparse
import traceback

from babel.numbers import format_decimal

from torch.backends import cudnn

import models
import torchvision

from helpers.utils import *
from helpers.loaders import *
from helpers.image_folder_custom_class import *

from trainer import train_wo_wms
from transformers import get_scheduler

# possible models to use
model_names = sorted(name for name in models.__dict__ if name.islower() and callable(models.__dict__[name]))
# print('models : ', model_names)

# set up argument parser
parser = argparse.ArgumentParser(description='Train models without watermarks.')

# model and dataset
parser.add_argument('--dataset', default='cifar10', help='the dataset to train on [cifar10]')
parser.add_argument('--num_classes', default=10, type=int, help='number of classes for classification')
parser.add_argument('--arch', metavar='ARCH', default='cnn_cifar10', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: cnn_cifar10)')

# hyperparameters
parser.add_argument('--runname', default='cifar10_custom_cnn', help='the exp name')
parser.add_argument('--epochs_wo_wm', default=2, type=int, help='number of epochs trained without watermarks')
parser.add_argument('--batch_size', default=64, type=int, help='the batch size')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--lradj', default=0.1, type=int, help='multiple the lr by lradj every 20 epochs')
parser.add_argument('--optim', default='SGD', help='optimizer (default SGD)')
parser.add_argument('--sched', default='MultiStepLR', help='scheduler (default MultiStepLR)')
parser.add_argument('--patience', default=20, help='early stopping patience (default 20)')

# cuda
parser.add_argument('--cuda', default='cuda:0', help='set cuda (e.g. cuda:0)')

# for testing with a smaller subset
parser.add_argument('--test_quot', default=None, type=int,
                    help='the quotient of data subset (for testing reasons; default: None)')


args = parser.parse_args()

try:
    device = torch.device(args.cuda) if torch.cuda.is_available() else 'cpu'

    cwd = os.getcwd()

    # create log and config file
    log_dir = os.path.join(cwd, 'log', str(args.dataset), str(args.arch), 'clean')
    os.makedirs(log_dir, exist_ok=True)
    configfile = os.path.join(log_dir, time.strftime("%Y%m%d-%H%M%S_") + 'conf_' + str(args.runname) + '.txt')
    logfile = os.path.join(log_dir, time.strftime("%Y%m%d-%H%M%S_") + 'log_' + str(args.runname) + '.txt')
    set_up_logger(logfile)

    # create save_dir, results_dir and loss_plots_dir
    save_dir = os.path.join(cwd, 'checkpoint', 'clean')
    os.makedirs(save_dir, exist_ok=True)

    # set up paths for dataset
    train_db_path = os.path.join(cwd, 'data')
    test_db_path = os.path.join(cwd, 'data')

    # save configuration parameters
    with open(configfile, 'w') as f:
        for arg in vars(args):
            f.write('{}: {}\n'.format(arg, getattr(args, arg)))

    # load train, valid and test set
    # valid_size = 0.1  # https://arxiv.org/abs/1512.03385 uses 0.1 for resnet
    transform_train, transform_test = get_data_transforms(args.dataset)
    train_set, test_set, valid_set = get_dataset(args.dataset, train_db_path, test_db_path, transform_train, transform_test,
                                                 testquot=args.test_quot)
    train_loader, test_loader, valid_loader = get_dataloader(train_set, test_set, args.batch_size, valid_set, shuffle=True)

    # set up loss
    criterion = nn.CrossEntropyLoss()

except Exception as e:
    msg = 'An error occurred during setup: ' + str(e)
    logging.error(msg)

try:
    runname = args.runname

    # create new model
    logging.info('Building model. new Model: ' + args.arch)
    net = models.__dict__[args.arch](num_classes=args.num_classes)
    net.to(device)

    # set up optimizer and scheduler
    if args.sched == 'WarmUp':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        t_total = len(train_loader) * args.epochs_wo_wm
        warmup_steps = int(t_total * 0.2)
        scheduler = get_scheduler(
                'linear',
                optimizer, 
                num_warmup_steps=warmup_steps,
                num_training_steps=t_total
            )
    else:
        optimizer, scheduler = set_up_optim_sched(args, net)
    
    logging.info('Training model.')

    real_acc, val_loss, epoch = train_wo_wms(args.epochs_wo_wm, net, criterion, optimizer, scheduler,
                                            args.patience, train_loader, test_loader, valid_loader,
                                            device, save_dir, args.runname)


    del net
    del optimizer
    del scheduler

except Exception as e:
    msg = 'An error occurred during training in ' + args.runname + ': ' + str(e)
    logging.error(msg)

    traceback.print_tb(e.__traceback__)