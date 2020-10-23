"""Training Script"""
import os
import shutil
import numpy as np
import pdb
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from config import ex
from util.utils import set_seed, CLASS_LABELS, date
from dataloaders_medical.prostate import *
# from models.fewshot import FewShotSeg
from settings import Settings
import few_shot_segmentor as fs
from torch.optim import lr_scheduler
from nn_common_modules import losses

def overlay_color(img, mask, label, scale=50):
    """
    :param img: [1, 256, 256]
    :param mask: [1, 256, 256]
    :param label: [1, 256, 256]
    :return:
    """
    # pdb.set_trace()
    scale = np.mean(img.cpu().numpy())
    mask = mask[0]
    label = label[0]
    zeros = torch.zeros_like(mask)
    zeros = [zeros for _ in range(3)]
    zeros[0] = mask
    mask = torch.stack(zeros,dim=0)
    zeros[1] = label
    label = torch.stack(zeros,dim=0)
    img_3ch = torch.cat([img,img,img],dim=0)
    masked = img_3ch+mask.float()*scale+label.float()*scale
    return [masked]

@ex.automain
def main(_run, _config, _log):
    settings = Settings()
    common_params, data_params, net_params, train_params, eval_params = settings['COMMON'], settings['DATA'], settings[
        'NETWORK'], settings['TRAINING'], settings['EVAL']

    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

    set_seed(_config['seed'])
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)

    _log.info('###### Load data ######')
    data_name = _config['dataset']
    if data_name == 'BCV':
        make_data = meta_data
    else:
        print(f"data name : {data_name}")
        raise ValueError('Wrong config for dataset!')

    tr_dataset, val_dataset, ts_dataset = make_data(_config)
    trainloader = DataLoader(
        dataset=tr_dataset,
        batch_size=_config['batch_size'],
        shuffle=True,
        num_workers=_config['n_work'],
        pin_memory=False, #True load data while training gpu
        drop_last=True
    )
    _log.info('###### Create model ######')
    model = fs.FewShotSegmentorDoubleSDnet(net_params).cuda()
    model.train()

    _log.info('###### Set optimizer ######')
    optim = torch.optim.Adam
    optim_args = {"lr": train_params['learning_rate'],
                  "weight_decay": train_params['optim_weight_decay'],}
                  # "momentum": train_params['momentum']}
    optim_c = optim(list(model.conditioner.parameters()), **optim_args)
    optim_s = optim(list(model.segmentor.parameters()), **optim_args)
    scheduler_s = lr_scheduler.StepLR(optim_s, step_size=100, gamma=0.1)
    scheduler_c = lr_scheduler.StepLR(optim_c, step_size=100, gamma=0.1)
    criterion = losses.DiceLoss()

    if _config['record']:  ## tensorboard visualization
        _log.info('###### define tensorboard writer #####')
        _log.info(f'##### board/train_{_config["board"]}_{date()}')
        writer = SummaryWriter(f'board/train_{_config["board"]}_{date()}')

    iter_print = _config["iter_print"]
    iter_n_train = len(trainloader)
    _log.info('###### Training ######')
    for i_epoch in range(_config['n_steps']):
        epoch_loss = 0
        for i_iter, sample_batched in enumerate(trainloader):
            # Prepare input
            s_x = sample_batched['s_x'].cuda()  # [B, Support, slice_num=1, 1, 256, 256]
            X = s_x.squeeze(2)  # [B, Support, 1, 256, 256]
            s_y = sample_batched['s_y'].cuda()  # [B, Support, slice_num, 1, 256, 256]
            Y = s_y.squeeze(2)  # [B, Support, 1, 256, 256]
            Y = Y.squeeze(2)  # [B, Support, 256, 256]
            q_x = sample_batched['q_x'].cuda()  # [B, slice_num, 1, 256, 256]
            query_input = q_x.squeeze(1)  # [B, 1, 256, 256]
            q_y = sample_batched['q_y'].cuda()  # [B, slice_num, 1, 256, 256]
            y2 = q_y.squeeze(1)  # [B, 1, 256, 256]
            y2 = y2.squeeze(1)  # [B, 256, 256]
            y2 = y2.type(torch.LongTensor).cuda()

            entire_weights = []
            for shot_id in range(_config["n_shot"]):
                input1 = X[:, shot_id, ...] # use 1 shot at first
                y1 = Y[:, shot_id, ...] # use 1 shot at first
                condition_input = torch.cat((input1, y1.unsqueeze(1)), dim=1)
                weights = model.conditioner(condition_input) # 2, 10, [B, channel=1, w, h]
                entire_weights.append(weights)

            # pdb.set_trace()
            avg_weights=[[],[None, None, None, None]]
            for i in range(9):
                weight_cat = torch.cat([weights[0][i] for weights in entire_weights],dim=1)
                avg_weight = torch.mean(weight_cat,dim=1,keepdim=True)
                avg_weights[0].append(avg_weight)

            avg_weights[0].append(None)

            output = model.segmentor(query_input, avg_weights)
            loss = criterion(F.softmax(output, dim=1), y2)
            optim_s.zero_grad()
            optim_c.zero_grad()
            loss.backward()
            optim_s.step()
            optim_c.step()

            epoch_loss += loss
            if iter_print:
                print(f"train, iter:{i_iter}/{iter_n_train}, iter_loss:{loss}", end='\r')

        scheduler_c.step()
        scheduler_s.step()
        print(f'step {i_epoch+1}: loss: {epoch_loss}                               ')

        if _config['record']:
            batch_i = 0
            frames = []
            query_pred = output.argmax(dim=1)
            query_pred = query_pred.unsqueeze(1)
            frames += overlay_color(q_x[batch_i,0], query_pred[batch_i].float(), q_y[batch_i,0])
            # frames += overlay_color(s_xi[batch_i], blank, s_yi[batch_i], scale=_config['scale'])
            visual = make_grid(frames, normalize=True, nrow=2)
            writer.add_image("train/visual", visual, i_epoch)

        save_fname = f'{_run.observers[0].dir}/snapshots/last.pth'
        torch.save(model.state_dict(),save_fname)
