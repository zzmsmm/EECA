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

import math
import gc

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
parser.add_argument('--theta', default=0.786, type=float)
parser.add_argument('--alpha', default=0, type=int)
parser.add_argument('--wm_batch_size', default=50, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--coding', default='Full-Entropy')
parser.add_argument('--wm_method', default='noise')
parser.add_argument('--runname', default='')
parser.add_argument('--cuda', default='cuda:0', help='set cuda (e.g. cuda:0)')

args = parser.parse_args()

alphas = [0.1, 1/30, 1e-7] # 1/3K
theta = -0.1 * math.log2(0.1) - 0.9 * math.log2(0.9) + 0.1 * math.log2(9) # H(1/C) + 1/C * log_2^(C-1) = -1/C * log_2^(1/C) - (1-1/C) * log_2^(1-1/C) + 1/C * log_2^(C-1)
alpha = alphas[args.alpha]
theta = args.theta
print('alpha: %.3f' % alpha)
print('theta: %.3f' % theta)
print(f'{args.runname}\t-----------------------')

try:
    device = torch.device(args.cuda) if torch.cuda.is_available() else 'cpu'
    cwd = os.getcwd()
    train_db_path = os.path.join(cwd, 'data')
    test_db_path = os.path.join(cwd, 'data')
    transform_train, transform_test = get_data_transforms(args.dataset)
    train_set, test_set, valid_set = get_dataset(args.dataset, train_db_path, test_db_path, transform_train, transform_test)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=False, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=True)

    wm_db_path = os.path.join(cwd, 'data', 'trigger_set')
    if args.wm_method == 'ood':
        transform = get_wm_transform('mnist')
    else:
        transform = get_wm_transform(args.dataset)
    trigger_set = get_trg_set(os.path.join(wm_db_path, args.dataset, args.arch, args.wm_method, f'C{args.num_classes}_K{args.K}', args.coding, f'{args.runname}_1'), 
                              'labels.txt', 400, transform)
    wm_loader = torch.utils.data.DataLoader(trigger_set, batch_size=args.wm_batch_size, shuffle=False)
    print('Size of Testing Set: %d, size of Trigger Set: %d' % (len(test_set), len(trigger_set)))

    net_0 = models.__dict__[args.arch](num_classes=args.num_classes)
    net_0.load_state_dict(torch.load(os.path.join('checkpoint', 'clean', f'{args.dataset}_{args.arch}_clean.ckpt'), map_location=device))
    net_0.to(device)
    net_0.eval()
    
    '''
    with torch.no_grad():
        for i in range(args.K):
            net = models.__dict__[args.arch](num_classes=args.num_classes)
            net.load_state_dict(torch.load(os.path.join('checkpoint', str(args.dataset), str(args.arch), 'watermark', str(args.wm_method), f'C{args.num_classes}_K{args.K}', args.coding, 
                                            f'{args.runname}_{i+1}.ckpt'), map_location=device))
            if args.wm_method == 'exponential_weighting':
                net.enable_ew(2.0)
            net.to(device)
            net.eval()

            correct = 0
            wrong = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)

                outputs_0 = outputs = net_0(inputs)
                _, predicted_0 = torch.max(outputs_0.data, 1)

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)

                total += targets.size(0)
                wrong += (targets.size(0) - predicted_0.eq(predicted.data).cpu().sum())
                correct += targets.eq(predicted.data).cpu().sum()

            print('model %d fA Test Set Acc: %.3f%% (%d/%d)'
                    % (i+1, 100. * correct / total, correct, total))
            print('model %d fA Test Set compared to f0 Error Probability: %.3f%% (%d/%d)'
                    % (i+1, 100. * wrong / total, wrong, total))
            net = None
            gc.collect()
    
    '''

    wrong = 0
    total = 0
    minus = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(wm_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            predicteds = []
            for i in range(args.K):
                net = models.__dict__[args.arch](num_classes=args.num_classes)
                net.load_state_dict(torch.load(os.path.join('checkpoint', str(args.dataset), str(args.arch), 'watermark', str(args.wm_method), f'C{args.num_classes}_K{args.K}', args.coding, 
                                                    f'{args.runname}_{i+1}.ckpt'), map_location=device))
                if args.wm_method == 'exponential_weighting':
                    net.enable_ew(2.0)
                net.to(device)
                net.eval()

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                predicteds.append(predicted)

                net = None
                gc.collect()

            final_predicted = torch.zeros(args.wm_batch_size,)
            for i in range(args.wm_batch_size):
                v = np.zeros(args.num_classes)
                for j in range(args.K):
                    v[predicteds[j][i].item()] += 1
                if np.all(v > 0):
                    minus += 1
                else:
                    v0 = v
                    v = (v + alpha) / float(args.K + args.num_classes * alpha)
                    h = np.sum(-v * np.log2(v))
                    if h <= theta:
                        final_predicted[i] = torch.tensor(np.argmax(v))
                    else:
                        final_predicted[i] = torch.tensor(np.argmin(v))
                        if v0[np.argmin(v)] == 0:
                            wrong += 1
            final_predicted = final_predicted.to(device)
            total += targets.size(0)

    print('fA Trigger Marking Assumption Failure Probability: %.3f%% (%d/%d)'
            % (100. * wrong / (total - minus), wrong, (total - minus)))


    correct = 0
    wrong = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs_0 = outputs = net_0(inputs)
            _, predicted_0 = torch.max(outputs_0.data, 1)

            predicteds = []
            for i in range(args.K):
                net = models.__dict__[args.arch](num_classes=args.num_classes)
                net.load_state_dict(torch.load(os.path.join('checkpoint', str(args.dataset), str(args.arch), 'watermark', str(args.wm_method), f'C{args.num_classes}_K{args.K}', args.coding, 
                                                    f'{args.runname}_{i+1}.ckpt'), map_location=device))
                if args.wm_method == 'exponential_weighting':
                    net.enable_ew(2.0)
                net.to(device)
                net.eval()

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                predicteds.append(predicted)

                net = None
                gc.collect()
            final_predicted = torch.zeros(args.batch_size,)
            for i in range(args.batch_size):
                v = np.zeros(args.num_classes)
                for j in range(args.K):
                    v[predicteds[j][i].item()] += 1 
                v0 = v
                v = (v + alpha) / float(args.K + args.num_classes * alpha)
                h = np.sum(-v * np.log2(v))
                if h <= theta:
                    final_predicted[i] = torch.tensor(np.argmax(v))
                else:
                    final_predicted[i] = torch.tensor(np.argmin(v))
            final_predicted = final_predicted.to(device)
            total += targets.size(0)
            wrong += (targets.size(0) - predicted_0.eq(final_predicted.data).cpu().sum())
            correct += targets.eq(final_predicted.data).cpu().sum()
            # print(total)

    print('fA Test Set Acc: %.3f%% (%d/%d)'
            % (100. * correct / total, correct, total))
    print('fA Test Set compared to f0 Error Probability: %.3f%% (%d/%d)'
            % (100. * wrong / total, wrong, total))
    # '''
except Exception as e:
    msg = 'An error occurred during setup: ' + str(e)
    logging.error(msg)