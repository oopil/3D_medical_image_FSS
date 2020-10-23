import argparse
import os

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
from dataloaders_medical.prostate import *

import few_shot_segmentor as fs
import utils.evaluator as eu
from settings import Settings
from solver import Solver
from utils.data_utils import get_imdb_dataset
from utils.log_utils import LogWriter
from utils.shot_batch_sampler import OneShotBatchSampler, get_lab_list

torch.set_default_tensor_type('torch.FloatTensor')

def load_data(data_params):
    print("Loading dataset")
    train_data, test_data = get_imdb_dataset(data_params)
    print("Train size: %i" % len(train_data))
    print("Test size: %i" % len(test_data))
    return train_data, test_data


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def train(train_params, common_params, data_params, net_params):
    _config= {
        "size":256,
        "n_shot": 1,
        "mode": 'test', # 'train' or 'test',
        "target": 1,
        "s_idx":0,
        "add_target":False,
        "record":False,
        "dataset": 'BCV',  # 'VOC' or 'COCO',
        "board": "try",
        "n_steps": 50000,  # 30000,
        "n_iter": train_params['iterations'],#5000,
        "label_sets": 0,
        "batch_size": 5,
        "lr_milestones": [10000, 20000, 50000],
        # "lr_milestones": [10000, 20000, 30000],
        "align_loss_scaler": 1,
        "ignore_label": 255,
        "n_work": 1,
        "task": {
            'n_ways': 1,
            'n_shots': 1,
            'n_queries': 1,
        },

        "data_src" : "/user/home2/soopil/Datasets/MICCAI2015challenge/Abdomen/RawData/Training_2d_2",
    }

    make_data = meta_data
    tr_dataset, val_dataset, ts_dataset = make_data(_config)
    trainloader = DataLoader(
        dataset=tr_dataset,
        batch_size=_config['batch_size'],
        shuffle=True,
        num_workers=_config['n_work'],
        pin_memory=False, #True load data while training gpu
        drop_last=True
    )
    validationloader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=_config['n_work'],
        pin_memory=False,#True
        drop_last=False
    )

    model_prefix = 'BCV_try'#'sne_position_all_type_channel_'
    folds = ['fold1', 'fold2', 'fold3', 'fold4']
    fold="fold1"
    final_model_path = os.path.join(
        common_params['save_model_dir'], model_prefix + fold + '.pth.tar')

    few_shot_model = fs.FewShotSegmentorDoubleSDnet(net_params)

    solver = Solver(few_shot_model,
                    device=common_params['device'],
                    num_class=net_params['num_class'],
                    optim_args={"lr": train_params['learning_rate'],
                                "weight_decay": train_params['optim_weight_decay'],
                                "momentum": train_params['momentum']},
                    model_name=common_params['model_name'],
                    exp_name=train_params['exp_name'],
                    labels=data_params['labels'],
                    log_nth=train_params['log_nth'],
                    num_epochs=train_params['num_epochs'],
                    lr_scheduler_step_size=train_params['lr_scheduler_step_size'],
                    lr_scheduler_gamma=train_params['lr_scheduler_gamma'],
                    use_last_checkpoint=train_params['use_last_checkpoint'],
                    log_dir=common_params['log_dir'],
                    exp_dir=common_params['exp_dir'])

    solver.train(trainloader, validationloader)
    solver.save_best_model(final_model_path)
    print("final model saved @ " + str(final_model_path))


def evaluate(eval_params, net_params, data_params, common_params, train_params):
    eval_model_path = eval_params['eval_model_path']
    num_classes = net_params['num_class']
    labels = data_params['labels']
    data_dir = eval_params['data_dir']
    query_txt_file = eval_params['query_txt_file']
    support_txt_file = eval_params['support_txt_file']
    remap_config = eval_params['remap_config']
    device = common_params['device']
    log_dir = common_params['log_dir']
    exp_dir = common_params['exp_dir']
    exp_name = train_params['exp_name']
    save_predictions_dir = eval_params['save_predictions_dir']

    orientation = eval_params['orientation']

    logWriter = LogWriter(num_classes, log_dir, exp_name, labels=labels)

    folds = ['fold1', 'fold2', 'fold3', 'fold4']

    for fold in folds:
        prediction_path = os.path.join(exp_dir, exp_name)
        prediction_path = prediction_path + "_" + fold
        prediction_path = os.path.join(prediction_path, save_predictions_dir)

        eval_model_path = os.path.join(
            'saved_models', eval_model_path + '_' + fold + '.pth.tar')
        query_labels = get_lab_list('val', fold)
        num_classes = len(fold)

        avg_dice_score = eu.evaluate_dice_score(eval_model_path,
                                                num_classes,
                                                query_labels,
                                                data_dir,
                                                query_txt_file,
                                                support_txt_file,
                                                remap_config,
                                                orientation,
                                                prediction_path,
                                                device,
                                                logWriter, fold=fold)

        logWriter.log(avg_dice_score)
    logWriter.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', required=True,
                        help='run mode, valid values are train and eval')
    parser.add_argument('--device', '-d', required=False,
                        help='device to run on')
    args = parser.parse_args()

    settings = Settings()
    common_params, data_params, net_params, train_params, eval_params = settings['COMMON'], settings['DATA'], settings[
        'NETWORK'], settings['TRAINING'], settings['EVAL']

    if args.device is not None:
        common_params['device'] = args.device

    if args.mode == 'train':
        train(train_params, common_params, data_params, net_params)
    elif args.mode == 'eval':
        evaluate(eval_params, net_params, data_params,
                 common_params, train_params)
    else:
        raise ValueError(
            'Invalid value for mode. only support values are train and eval')
