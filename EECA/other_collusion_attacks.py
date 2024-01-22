# Weight Average
# Majority Vote
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
parser.add_argument('--wm_batch_size', default=50, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--coding', default='Full-Entropy')
parser.add_argument('--wm_method', default='noise')
parser.add_argument('--runname', default='')
parser.add_argument('--collusion_method', default='weight_average', help='weight_average, majority_vote')
parser.add_argument('--attack_method', default=None, help='fine-tune, pruning, no')
parser.add_argument('--cuda', default='cuda:0', help='set cuda (e.g. cuda:0)')

args = parser.parse_args()

print(f'{args.collusion_method}, {args.runname}\t-----------------------')

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

    if args.attack_method != None:
        print(f"{args.attack_method}\t------------------")
    if args.collusion_method == 'weight_average':
        print('Start to average the weights...')
        net_avg = models.__dict__[args.arch](num_classes=args.num_classes)
        avg_weights = net_avg.state_dict()
        for i in range(args.K):
            net = models.__dict__[args.arch](num_classes=args.num_classes)
            if args.attack_method != None:
                net.load_state_dict(torch.load(os.path.join('checkpoint', str(args.dataset), str(args.arch), str(args.attack_method), str(args.wm_method), f'C{args.num_classes}_K{args.K}', args.coding, 
                                            f'{args.runname}_{i+1}_{args.attack_method}.ckpt'), map_location=device))
            else:
                net.load_state_dict(torch.load(os.path.join('checkpoint', str(args.dataset), str(args.arch), 'watermark', str(args.wm_method), f'C{args.num_classes}_K{args.K}', args.coding, 
                                            f'{args.runname}_{i+1}.ckpt'), map_location=device))
            for name_param in avg_weights:
                if i == 0:
                    avg_weights[name_param] = net.state_dict()[name_param]
                else:
                    avg_weights[name_param] += net.state_dict()[name_param]
            net = None
            gc.collect()
        for name_param in avg_weights:
            avg_weights[name_param] = avg_weights[name_param] / args.K
        net_avg.load_state_dict(avg_weights)
        if args.wm_method == 'exponential_weighting':
            net_avg.enable_ew(2.0)
        net_avg.to(device)
        net_avg.eval()
    
    wrong = 0
    total = 0
    minus = 0

    wm_targets = []
    for i in range(args.K):
        path = os.path.join(wm_db_path, args.dataset, args.arch, args.wm_method, f'C{args.num_classes}_K{args.K}', args.coding, f'{args.runname}_{i+1}')
        wm_target = np.loadtxt(os.path.join(path, 'labels.txt'))
        wm_targets.append(wm_target)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(wm_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            if args.collusion_method == 'majority_vote':    
                predicteds = []
                for i in range(args.K):
                    net = models.__dict__[args.arch](num_classes=args.num_classes)
                    if args.attack_method != None:
                        net.load_state_dict(torch.load(os.path.join('checkpoint', str(args.dataset), str(args.arch), str(args.attack_method), str(args.wm_method), f'C{args.num_classes}_K{args.K}', args.coding, 
                                                f'{args.runname}_{i+1}_{args.attack_method}.ckpt'), map_location=device))
                    else:
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
                    v0 = np.zeros(args.num_classes)
                    for j in range(args.K):
                        v0[predicteds[j][i].item()] += 1
                        v[int(wm_targets[j][batch_idx * args.wm_batch_size + i])] += 1
                    final_predicted[i] = torch.tensor(np.argmax(v0))
                    if np.all(v > 0):
                        minus += 1
                    elif v[int(final_predicted[i].item())] == 0:
                        wrong += 1
            else:
                outputs = net_avg(inputs)
                _, final_predicted = torch.max(outputs.data, 1)
                for i in range(args.wm_batch_size):
                    v = np.zeros(args.num_classes)
                    for j in range(args.K):
                        # v[predicteds[j][i].item()] += 1
                        v[int(wm_targets[j][batch_idx * args.wm_batch_size + i])] += 1
                    if np.all(v > 0):
                        minus += 1
                    elif v[final_predicted[i].item()] == 0:
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

            if args.collusion_method == 'majority_vote':
                predicteds = []
                for i in range(args.K):
                    net = models.__dict__[args.arch](num_classes=args.num_classes)
                    if args.attack_method != None:
                        net.load_state_dict(torch.load(os.path.join('checkpoint', str(args.dataset), str(args.arch), str(args.attack_method), str(args.wm_method), f'C{args.num_classes}_K{args.K}', args.coding, 
                                                    f'{args.runname}_{i+1}_{args.attack_method}.ckpt'), map_location=device))
                    else:
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
                    final_predicted[i] = torch.tensor(np.argmax(v))
            else:
                outputs = net_avg(inputs)
                _, final_predicted = torch.max(outputs.data, 1)
            final_predicted = final_predicted.to(device)
            total += targets.size(0)
            wrong += (targets.size(0) - predicted_0.eq(final_predicted.data).cpu().sum())
            correct += targets.eq(final_predicted.data).cpu().sum()

    print('fA Test Set Acc: %.3f%% (%d/%d)'
            % (100. * correct / total, correct, total))
    print('fA Test Set compared to f0 Error Probability: %.3f%% (%d/%d)'
            % (100. * wrong / total, wrong, total))
    
except Exception as e:
    msg = 'An error occurred during setup: ' + str(e)
    logging.error(msg)