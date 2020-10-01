import os
import pprint
from collections import OrderedDict, defaultdict
import json

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from batch_engine import valid_trainer, batch_trainer
from config import argument_parser
from dataset.AttrDataset import AttrDataset, get_transform, AttrDataset_new, parse_transformation_dict
from loss.CE_loss import CEL_Sigmoid

from models.base_block import FeatClassifier, BaseClassifier
from models.resnet import resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d
from models.senet import se_resnet101, se_resnet50
from models.dpn import dpn68, dpn68b, dpn92, dpn98, dpn131, dpn107
from models.densenet import densenet121, densenet169, densenet201, densenet161

from tools.function import get_model_log_path, get_pedestrian_metrics
from tools.utils import time_str, save_ckpt, ReDirectSTD, set_seed

from torch.utils.tensorboard import SummaryWriter

from datetime import datetime
import getpass
import inspect
import sys

set_seed(605)

def main(args):
    visenv_name = args.dataset
    exp_dir = os.path.join('exp_result', args.dataset)
    model_dir, log_dir = get_model_log_path(exp_dir, visenv_name)

    # Added for logging purposes
    user = getpass.getuser()
    fixed_time_str = time_str()
    stdout_file = os.path.join(log_dir, "_".join(['stdout', user, f'{fixed_time_str}.txt']) )
    save_model_path = os.path.join(model_dir,  "_".join(['ckpt_max', user, f'{fixed_time_str}.pth']) )
    trackitems_dir = os.path.join(log_dir, "_".join(['trackitems', user, f'{fixed_time_str}.txt']) )

    if args.redirector:
        print('redirector stdout')
        ReDirectSTD(stdout_file, 'stdout', False)

    pprint.pprint(OrderedDict(args.__dict__))

    print('-' * 60)
    print(f'use GPU{args.device} for training')
    print(f'train set: {args.dataset} {args.train_split}, test set: {args.valid_split}')

    train_tsfm, valid_tsfm = get_transform(args)
    print(train_tsfm)

    #train_set = AttrDataset(args=args, split=args.train_split, transform=train_tsfm)
    train_set = AttrDataset_new(args=args, split=args.train_split, transformation_dict=args.train_transform)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    #valid_set = AttrDataset(args=args, split=args.valid_split, transform=valid_tsfm)
    valid_set = AttrDataset_new(args=args, split=args.valid_split, transformation_dict=args.valid_transform)

    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    print(f'{args.train_split} set: {len(train_loader.dataset)}, '
          f'{args.valid_split} set: {len(valid_loader.dataset)}, '
          f'attr_num : {train_set.attr_num}')

    labels = train_set.label
    # sample_weight = labels.mean(0)
    sample_weight = np.nanmean(np.where(labels!=2,labels,np.nan), axis=0)

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

    # Added for logging purposes
    with open(trackitems_dir, "a") as f:
        code, line_no = inspect.getsourcelines(get_transform)
        for line in code:
            f.write(str(line))
        f.write(str("\n\n"))
        
        f.write(str(args.__dict__))
        f.write(str("\n\n"))
        
        f.write(str(lr_scheduler.__dict__))
        f.write(str("\n\n"))
        
        model_str = str(model).lower()
        have_dropout = 'dropout' in model_str
        f.write('dropout: %s' %(have_dropout))
        f.write(str("\n\n"))

        have_leaky_relu = 'leaky_relu' in model_str
        f.write('leaky_relu: %s' %(have_leaky_relu))
        f.write(str("\n\n"))


    best_metric, epoch = trainer(epoch=args.train_epoch,
                                 model=model,
                                 train_loader=train_loader,
                                 valid_loader=valid_loader,
                                 criterion=criterion,
                                 optimizer=optimizer,
                                 lr_scheduler=lr_scheduler,
                                 path=save_model_path,
                                 measure="f1")

    print(f'{visenv_name},  best_metrc : {best_metric} in epoch{epoch}')

    # Added for logging purposes
    with open(trackitems_dir, "a") as f:
        f.write(f'{visenv_name},  best_metrc : {best_metric} in epoch{epoch}')
        f.write("\n\n")

def trainer(epoch, model, train_loader, valid_loader, criterion, optimizer, lr_scheduler,
            path, measure):
    maximum = float(-np.inf)
    best_epoch = 0

    result_list = defaultdict()
    
    df_metrics = pd.DataFrame(columns=['epoch', 'train_loss', 'train_instance_acc', 'train_instance_prec', 'train_instance_recall', 'train_instance_f1', 'train_ma', 'train_pos_recall', 'train_neg_recall',
                               'valid_loss', 'valid_instance_acc', 'valid_instance_prec', 'valid_instance_recall', 'valid_instance_f1', 'valid_ma', 'valid_pos_recall', 'valid_neg_recall'])


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
        
         # create metrics dataframe to save as csv

        new_metrics = { 
            'epoch':i,
            'train_loss':train_loss,
            'train_instance_acc':train_result.instance_acc,
            'train_instance_prec':train_result.instance_prec,
            'train_instance_recall':train_result.instance_recall,
            'train_instance_f1':train_result.instance_f1,
            'train_ma':train_result.ma,
            'train_pos_recall':np.mean(train_result.label_pos_recall),
            'train_neg_recall':np.mean(train_result.label_neg_recall),
            'valid_loss':valid_loss,
            'valid_instance_acc':valid_result.instance_acc,
            'valid_instance_prec':valid_result.instance_prec,
            'valid_instance_recall':valid_result.instance_recall,
            'valid_instance_f1':valid_result.instance_f1,
            'valid_ma':valid_result.ma,
            'valid_pos_recall':np.mean(valid_result.label_pos_recall),
            'valid_neg_recall':np.mean(valid_result.label_neg_recall)
            }
        #append row to the dataframe
        df_metrics = df_metrics.append(new_metrics, ignore_index=True)
        df_metrics.to_csv(csv_file_name, index=False)

        # We only allow "accuracy" or "f1"
        assert((measure.lower()=="accuracy") or (measure.lower()=="f1"))
        if measure == 'accuracy':
            cur_metric = valid_result.ma
        elif measure == 'f1':
            cur_metric = valid_result.instance_f1

        if cur_metric > maximum:
            maximum = cur_metric
            best_epoch = i
            save_ckpt(model, path, i, maximum)

        result_list[i] = [train_result, valid_result]

    writer.close()

    torch.save(result_list, os.path.join(os.path.dirname(path), 'metric_log.pkl'))

    return maximum, best_epoch


if __name__ == '__main__':
    
    parser = argument_parser()
    args = parser.parse_args()
    
    log_dir = 'runs/' + args.dataset+"_"+args.model+"_"+datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_dir = 'csv_folder/'
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    csv_file_name = os.path.join(csv_dir, args.dataset+"_"+args.model+"_"+datetime.now().strftime("%Y%m%d-%H%M%S") +'.csv')
    writer = SummaryWriter(log_dir)

    main(args)

    # os.path.abspath()

"""
载入的时候要：
from tools.function import LogVisual
sys.modules['LogVisual'] = LogVisual
log = torch.load('./save/2018-10-29_21:17:34trlog')
"""
