#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 21:28:44 2020

@author: jiahao
"""


import os
import pprint
from collections import OrderedDict, defaultdict

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from batch_engine import valid_trainer, batch_trainer
from dataset.AttrDataset import AttrDataset, get_transform
from loss.CE_loss import CEL_Sigmoid
from models.base_block import FeatClassifier, BaseClassifier
from models.resnet import resnet50
from models.resnet_hr import resnet50_hr
from models.senet import se_resnet101, se_resnet50
from models.densenet import densenet121
from models.resnext import resnext101_32x4d
from models.dpn import dpn68, dpn92
from tools.function import get_model_log_path, get_pedestrian_metrics
from tools.utils import time_str, save_ckpt, ReDirectSTD, set_seed
from datetime import datetime
import sys

import torchvision.models as models

# resnet18 = models.resnet18(pretrained=True)

from torch.utils.tensorboard import SummaryWriter

set_seed(605)
log_dir = 'runs/' + datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir)

import argparse


def argument_parser():
    parser = argparse.ArgumentParser(description="attribute recognition",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("dataset", type=str, default="PETA")
    parser.add_argument("--model", type=str, default="resnet50_hr")
    parser.add_argument("--debug", action='store_false')

    parser.add_argument("--batchsize", type=int, default=2)
    parser.add_argument("--train_epoch", type=int, default=1)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--lr_ft", type=float, default=0.01, help='learning rate of feature extractor')
    parser.add_argument("--lr_new", type=float, default=0.1, help='learning rate of classifier_base')
    parser.add_argument('--classifier', type=str, default='base', help='classifier name')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument("--train_split", type=str, default="trainval", choices=['train', 'trainval'])
    parser.add_argument("--valid_split", type=str, default="test", choices=['test', 'valid'])
    parser.add_argument('--device', default=-1, type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument("--redirector", action='store_false')
    parser.add_argument('--use_bn', action='store_false')

    return parser

def main(args):
    visenv_name = args.dataset
    exp_dir = os.path.join('exp_result', args.dataset)
    model_dir, log_dir = get_model_log_path(exp_dir, visenv_name)
    stdout_file = os.path.join(log_dir, f'stdout_{time_str()}.txt')
    save_model_path = os.path.join(model_dir, 'ckpt_max.pth')

    if args.redirector:
        print('redirector stdout')
        ReDirectSTD(stdout_file, 'stdout', False)

    pprint.pprint(OrderedDict(args.__dict__))

    print('-' * 60)
    print(f'use GPU{args.device} for training')
    print(f'train set: {args.dataset} {args.train_split}, test set: {args.valid_split}')

    train_tsfm, valid_tsfm = get_transform(args)
    print(train_tsfm)

    train_set = AttrDataset(args=args, split=args.train_split, transform=train_tsfm)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    valid_set = AttrDataset(args=args, split=args.valid_split, transform=valid_tsfm)

    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    print(f'{args.train_split} set: {len(train_loader.dataset)}, '
          f'{args.valid_split} set: {len(valid_loader.dataset)}, '
          f'attr_num : {train_set.attr_num}')

    labels = train_set.label
    # sample_weight = labels.mean(0)
    sample_weight = labels[labels!=2].reshape((labels.shape[0], labels.shape[1])).mean(0)
    # sample_weight = np.where(labels!=2,labels,np.nan).mean(0)

    backbone = getattr(sys.modules[__name__], args.model)()
    
    if "dpn68" in args.model:
        net_parameter = 832
    elif "dpn" in args.model:
        net_parameter = 2688
    elif "densenet" in args.model:
        net_parameter = 1024
    else:
        net_parameter = 2048
    
    classifier = BaseClassifier(netpara=net_parameter, nattr=train_set.attr_num)
    model = FeatClassifier(backbone, classifier)

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    criterion = CEL_Sigmoid(sample_weight)

    if torch.cuda.is_available():
        param_groups = [{'params': model.module.finetune_params(), 'lr': args.lr_ft},
                        {'params': model.module.fresh_params(), 'lr': args.lr_new}]
    else:
        param_groups = [{'params': model.finetune_params(), 'lr': args.lr_ft},
                        {'params': model.fresh_params(), 'lr': args.lr_new}]
    optimizer = torch.optim.SGD(param_groups, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
    lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=4)

    best_metric, epoch = trainer(epoch=args.train_epoch,
                                 model=model,
                                 train_loader=train_loader,
                                 valid_loader=valid_loader,
                                 criterion=criterion,
                                 optimizer=optimizer,
                                 lr_scheduler=lr_scheduler,
                                 path=save_model_path)

    print(f'{visenv_name},  best_metrc : {best_metric} in epoch{epoch}')


def trainer(epoch, model, train_loader, valid_loader, criterion, optimizer, lr_scheduler,
            path):
    maximum = float(-np.inf)
    best_epoch = 0

    result_list = defaultdict()

    for i in range(epoch):

        train_loss, train_gt, train_probs = batch_trainer(
            epoch=i,
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
        )

        valid_loss, valid_gt, valid_probs = valid_trainer(
            model=model,
            valid_loader=valid_loader,
            criterion=criterion,
        )

        lr_scheduler.step(metrics=valid_loss, epoch=i)

        train_result = get_pedestrian_metrics(train_gt, train_probs)
        valid_result = get_pedestrian_metrics(valid_gt, valid_probs)

        # tensorboard added
        # writer.add_scalar(tag, function, iteration)
        writer_step = i

        writer.add_scalars('Loss', {'Train':train_loss, 'Valid':valid_loss}, writer_step)
        writer.add_scalars('Accuracy', {'Train':train_result.instance_acc, 'Valid':valid_result.instance_acc}, writer_step)
        writer.add_scalars('Precision', {'Train':train_result.instance_prec, 'Valid':valid_result.instance_prec}, writer_step)
        writer.add_scalars('Recall', {'Train':train_result.instance_recall, 'Valid':valid_result.instance_recall}, writer_step)
        writer.add_scalars('F1', {'Train':train_result.instance_f1, 'Valid':valid_result.instance_f1}, writer_step)
        writer.add_scalars('Mean Accuracy', {'Train':train_result.ma, 'Valid':valid_result.ma}, writer_step)
        writer.add_scalars('Pos Recall', {'Train':np.mean(train_result.label_pos_recall), 'Valid':np.mean(valid_result.label_pos_recall)}, writer_step)
        writer.add_scalars('Neg Recall', {'Train':np.mean(train_result.label_neg_recall), 'Valid':np.mean(valid_result.label_neg_recall)}, writer_step)

        print(f'Evaluation on test set, \n',
              'ma: {:.4f},  pos_recall: {:.4f} , neg_recall: {:.4f} \n'.format(
                  valid_result.ma, np.mean(valid_result.label_pos_recall), np.mean(valid_result.label_neg_recall)),
              'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
                  valid_result.instance_acc, valid_result.instance_prec, valid_result.instance_recall,
                  valid_result.instance_f1))

        print(f'{time_str()}')
        print('-' * 60)

        cur_metric = valid_result.ma

        if cur_metric > maximum:
            maximum = cur_metric
            best_epoch = i
            save_ckpt(model, path, i, maximum)

        result_list[i] = [train_result, valid_result]

    writer.close()

    torch.save(result_list, os.path.join(os.path.dirname(path), 'metric_log.pkl'))

    return maximum, best_epoch


parser = argument_parser()
args = parser.parse_args(["PETA"])
main(args)

    # os.path.abspath()

"""
载入的时候要：
from tools.function import LogVisual
sys.modules['LogVisual'] = LogVisual
log = torch.load('./save/2018-10-29_21:17:34trlog')
"""
