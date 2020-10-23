"""Training Script"""
import os
import shutil
import numpy as np
import pdb
import random

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
from torchvision.transforms import Compose
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from config import ex
from util.utils import set_seed, CLASS_LABELS, date
from dataloaders_medical.prostate import *
from models.fewshot import FewShotSeg

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


    _log.info('###### Create model ######')
    model = FewShotSeg(pretrained_path=_config['path']['init_path'], cfg=_config['model'])
    model = nn.DataParallel(model.cuda(), device_ids=[_config['gpu_id'],])
    model.train()


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
    _log.info('###### Set optimizer ######')
    optimizer = torch.optim.SGD(model.parameters(), **_config['optim'])
    scheduler = MultiStepLR(optimizer, milestones=_config['lr_milestones'], gamma=0.1)
    criterion = nn.CrossEntropyLoss(ignore_index=_config['ignore_label'])

    if _config['record']:  ## tensorboard visualization
        _log.info('###### define tensorboard writer #####')
        _log.info(f'##### board/train_{_config["board"]}_{date()}')
        writer = SummaryWriter(f'board/train_{_config["board"]}_{date()}')

    log_loss = {'loss': 0, 'align_loss': 0}
    _log.info('###### Training ######')
    for i_iter, sample_batched in enumerate(trainloader):
        # Prepare input
        s_x_orig = sample_batched['s_x'].cuda()  # [B, Support, slice_num=1, 1, 256, 256]
        s_x = s_x_orig.squeeze(2) # [B, Support, 1, 256, 256]
        s_y_fg_orig = sample_batched['s_y'].cuda()  # [B, Support, slice_num, 1, 256, 256]
        s_y_fg = s_y_fg_orig.squeeze(2) # [B, Support, 1, 256, 256]
        s_y_fg = s_y_fg.squeeze(2) # [B, Support, 256, 256]
        s_y_bg = torch.ones_like(s_y_fg) - s_y_fg
        q_x_orig = sample_batched['q_x'].cuda()  # [B, slice_num, 1, 256, 256]
        q_x = q_x_orig.squeeze(1) # [B, 1, 256, 256]
        q_y_orig = sample_batched['q_y'].cuda()  # [B, slice_num, 1, 256, 256]
        q_y = q_y_orig.squeeze(1) # [B, 1, 256, 256]
        q_y = q_y.squeeze(1).long() # [B, 256, 256]
        s_xs = [[s_x[:,shot, ...] for shot in range(_config["n_shot"])]]
        s_y_fgs = [[s_y_fg[:,shot, ...] for shot in range(_config["n_shot"])]]
        s_y_bgs = [[s_y_bg[:,shot, ...] for shot in range(_config["n_shot"])]]
        q_xs = [q_x]
        """
        Args:
            supp_imgs: support images
                way x shot x [B x 1 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 1 x H x W], list of tensors
            qry_pred: [B, 2, H, W]
        """

        # Forward and Backward
        optimizer.zero_grad()
        query_pred, align_loss = model(s_xs, s_y_fgs, s_y_bgs, q_xs) #[B, 2, w, h]
        query_loss = criterion(query_pred, q_y)
        loss = query_loss + align_loss * _config['align_loss_scaler']
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Log loss
        query_loss = query_loss.detach().data.cpu().numpy()
        align_loss = align_loss.detach().data.cpu().numpy() if align_loss != 0 else 0
        _run.log_scalar('loss', query_loss)
        _run.log_scalar('align_loss', align_loss)
        log_loss['loss'] += query_loss
        log_loss['align_loss'] += align_loss

        # print loss and take snapshots
        if (i_iter + 1) % _config['print_interval'] == 0:
            loss = log_loss['loss'] / (i_iter + 1)
            align_loss = log_loss['align_loss'] / (i_iter + 1)
            print(f'step {i_iter+1}: loss: {loss}, align_loss: {align_loss}')

            if _config['record']:
                batch_i = 0
                frames = []
                query_pred = query_pred.argmax(dim=1)
                query_pred = query_pred.unsqueeze(1)
                frames += overlay_color(q_x_orig[batch_i,0], query_pred[batch_i].float(), q_y_orig[batch_i,0])
                visual = make_grid(frames, normalize=True, nrow=2)
                writer.add_image("train/visual", visual, i_iter)


            print(f"train - iter:{i_iter} \t => model saved", end='\n')
            save_fname = f'{_run.observers[0].dir}/snapshots/last.pth'
            torch.save(model.state_dict(),save_fname)
