from watermarks.base import WmMethod

import os
import logging
import random
import numpy as np

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from helpers.utils import image_char, save_triggerset, get_size, find_tolerance, get_trg_set
from helpers.loaders import get_data_transforms, get_wm_transform, get_dataset
from helpers.transforms import EmbedText
from helpers.tinyimagenetloader import TrainTinyImageNetDataset

from trainer import test, train, train_on_augmented


class Ood(WmMethod):
    def __init__(self, args):
        super().__init__(args)

        self.path = os.path.join(os.getcwd(), 'data', 'trigger_set')
        os.makedirs(self.path, exist_ok=True)  # path where to save trigger set if has to be generated

    def gen_watermarks(self, device):
        cwd = os.getcwd()
        train_db_path = os.path.join(cwd, 'data')
        test_db_path = os.path.join(cwd, 'data')

        train_transform, test_transform = get_data_transforms('mnist')

        wm_set, test_set, _ = get_dataset('mnist', train_db_path, test_db_path, train_transform, test_transform,
                                          valid_size=None, testquot=self.test_quot, size_train=None, size_test=None)

        if self.coding != 'Full-Entropy':
            load_path = os.path.join(os.getcwd(), 'Coding_Sheet', f'C{self.num_classes}_K{self.K}_{self.coding}.txt')
            p_list = np.loadtxt(load_path)
        for i in random.sample(range(len(wm_set)), len(wm_set)):  # iterate randomly
            img, lbl = wm_set[i]
            img = img.to(device)
            self.trigger_set.append(img)
            if len(self.trigger_set) == self.size:
                break
        sample_list = list(range(0, self.num_classes))
        for i in range(0, self.K):
            trigger_set = []
            for idx, img in enumerate(self.trigger_set):
                if self.coding == 'Full-Entropy':
                    lbl = random.choices(sample_list, k=1)
                else:
                    lbl = random.choices(sample_list, weights=p_list[idx], k=1)
                trigger_set.append((img, lbl[0]))
            if self.save_wm:
                save_triggerset(trigger_set, os.path.join(self.path, self.dataset, self.arch, 'ood', f'C{self.num_classes}_K{self.K}', self.coding), f'{self.runname}_{i+1}')
                print(f'{i+1} watermarks generation done')

    def embed(self, net, criterion, optimizer, scheduler, train_set, test_set, train_loader, test_loader, valid_loader,
              device, save_dir):

        transform = get_wm_transform('mnist')
        self.trigger_set = get_trg_set(os.path.join(self.path, self.dataset, self.arch, 'ood', f'C{self.num_classes}_K{self.K}', self.coding, self.runname), 'labels.txt', self.size,
                                                    transform)

        self.loader()

        if self.embed_type == 'pretrained':
            # load model
            logging.info("Load model: " + self.loadmodel + ".ckpt")
            net.load_state_dict(torch.load(os.path.join('checkpoint', 'clean', self.loadmodel + '.ckpt')))

        real_acc, wm_acc, val_loss, epoch = train_on_augmented(self.epochs_w_wm, device, net, optimizer, criterion,
                                                               scheduler, self.patience, train_loader, test_loader,
                                                               valid_loader, self.wm_loader, save_dir, self.save_model)

        logging.info("Done embedding.")

        return real_acc, wm_acc, val_loss, epoch
        