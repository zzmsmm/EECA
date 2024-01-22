import argparse
import traceback

from babel.numbers import format_decimal

from torch.backends import cudnn

import numpy as np
import models
import torch
import torchvision
import torchvision.transforms as transforms
import copy
from itertools import cycle

from helpers.utils import *
from helpers.loaders import *
from helpers.image_folder_custom_class import *
from trainer import test
from attacks.pruning import prune

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
parser.add_argument('--K', default=10, type=int)
parser.add_argument('--wm_batch_size', default=50, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--coding', default='Full-Entropy')
parser.add_argument('--wm_method', default='noise')
parser.add_argument('--runname', default='')
parser.add_argument('--attack_method', default='', help='fine-tune, pruning')
parser.add_argument('--pruning_rate', default=0.8, type=float)
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--lradj', default=0.1, type=int, help='multiple the lr by lradj every 20 epochs')
parser.add_argument('--optim', default='SGD', help='optimizer (default SGD)')
parser.add_argument('--sched', default='MultiStepLR_cifar10', help='scheduler (default MultiStepLR)')
parser.add_argument('--epochs', default=1, type=int, help='number of epochs attacked')
parser.add_argument('--cuda', default='cuda:0', help='set cuda (e.g. cuda:0)')

args = parser.parse_args()

try:
    device = torch.device(args.cuda) if torch.cuda.is_available() else 'cpu'

    cwd = os.getcwd()

    criterion = nn.CrossEntropyLoss()

    train_db_path = os.path.join(cwd, 'data')
    test_db_path = os.path.join(cwd, 'data')
    transform_train, transform_test = get_data_transforms(args.dataset)
    if args.wm_method == 'noise':
        #train_set, test_set, valid_set = get_dataset(args.dataset, train_db_path, test_db_path, transform_train, transform_test, testquot=0.2)
        train_set, test_set, valid_set = get_dataset(args.dataset, train_db_path, test_db_path, transform_train, transform_test, testquot=0.4)   
    else:
        train_set, test_set, valid_set = get_dataset(args.dataset, train_db_path, test_db_path, transform_train, transform_test, testquot=0.4)       
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=False, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=True)

    for i in range(args.K):
        wm_db_path = os.path.join(cwd, 'data', 'trigger_set')
        if args.wm_method == 'ood':
            transform = get_wm_transform('mnist')
        else:
            transform = get_wm_transform(args.dataset)
        trigger_set = get_trg_set(os.path.join(wm_db_path, args.dataset, args.arch, args.wm_method, f'C{args.num_classes}_K{args.K}', args.coding, f'{args.runname}_{i+1}'), 
                                'labels.txt', 400, transform)
        wm_loader = torch.utils.data.DataLoader(trigger_set, batch_size=args.wm_batch_size, shuffle=False)
        print('Size of Testing Set: %d, size of Trigger Set: %d' % (len(test_set), len(trigger_set)))

        net = models.__dict__[args.arch](num_classes=args.num_classes)
        net.load_state_dict(torch.load(os.path.join('checkpoint', str(args.dataset), str(args.arch), 'watermark', str(args.wm_method), f'C{args.num_classes}_K{args.K}', args.coding, 
                                        f'{args.runname}_{i+1}.ckpt'), map_location=device))
        print(f'loading model {args.runname}_{i+1}.ckpt...')
        if args.wm_method == 'exponential_weighting':
            net.enable_ew(2.0)
        net.to(device)

        test_acc = test(net, criterion, test_loader, device)
        wm_acc = test(net, criterion, wm_loader, device)

        print("Test acc: %.3f%%" % test_acc)
        print("WM acc: %.3f%%" % wm_acc)

        if args.attack_method == 'pruning':
            print("pruning attack start......")
            prune(net, args.arch, args.pruning_rate)
            
            test_acc = test(net, criterion, test_loader, device)
            wm_acc = test(net, criterion, wm_loader, device)

            print("Test acc: %.3f%%" % test_acc)
            print("WM acc: %.3f%%" % wm_acc)
            
            print("saving model...")
            save_dir = os.path.join('checkpoint', str(args.dataset), str(args.arch), 'pruning', 
                                    str(args.wm_method), f'C{args.num_classes}_K{args.K}', args.coding)
            os.makedirs(save_dir, exist_ok=True)
            torch.save(net.state_dict(), os.path.join(save_dir, f'{args.runname}_{i+1}_pruning.ckpt'))
            
        else:
            print('fine-tune attack start......')
            optimizer, scheduler = set_up_optim_sched(args, net)
            for epoch in range(args.epochs):
                #print('\nEpoch: %d' % epoch)
                net.train()

                train_losses = []
                train_loss = 0
                correct, total = 0, 0

                for batch_idx, (inputs, targets) in enumerate(train_loader):
                    # print('\nBatch: %d' % batch_idx)

                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    outputs = net(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward(retain_graph=True)
                    optimizer.step()
                    train_losses.append(loss.item())

                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += predicted.eq(targets.data).cpu().sum()
                    train_acc = 100. * correct / total
                    '''
                    progress_bar(batch_idx, len(train_loader), 'Loss: %.3f lr: %.4f| Acc: %.3f%% (%d/%d)' %
                     (np.average(train_losses), optimizer.param_groups[0]['lr'], train_acc, correct, total))
                    '''
                train_loss = np.average(train_losses)
                print(('Epoch %d: Train loss: %.3f | Train Acc: %.3f%% (%d/%d)'
                  % (epoch, train_loss, train_acc, correct, total)))
                
                test_acc = test(net, criterion, test_loader, device)
                wm_acc = test(net, criterion, wm_loader, device)

                print("Test acc: %.3f%%" % test_acc)
                print("WM acc: %.3f%%" % wm_acc)
            #'''
            print("saving model...\n")
            save_dir = os.path.join('checkpoint', str(args.dataset), str(args.arch), 'fine-tune', 
                                    str(args.wm_method), f'C{args.num_classes}_K{args.K}', args.coding)
            os.makedirs(save_dir, exist_ok=True)
            torch.save(net.state_dict(), os.path.join(save_dir, f'{args.runname}_{i+1}_fine-tune.ckpt'))
            #'''
except Exception as e:
    msg = 'An error occurred during setup: ' + str(e)
    logging.error(msg)