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
from torchvision.transforms import Compose
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from nn_common_modules import losses

if __name__ == '__main__':
    from util.utils import set_seed, CLASS_LABELS, date
    from config import ex
    from tensorboardX import SummaryWriter
    from dataloaders_medical.common import *
    from dataloaders_medical.prostate import *
    from model import MedicalFSS
else:
    from .util.utils import set_seed, CLASS_LABELS, date
    from .config import ex
    from tensorboardX import SummaryWriter
    from .dataloaders_medical.common import *
    from .dataloaders_medical.prostate import *
    from .model import MedicalFSS

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

@ex.capture
def get_info(_run):
    print(_run._id)
    print(_run.experiment_info["name"])

@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

    print(f"experiment : {_run.experiment_info['name']} , ex_ID : {_run._id}")
    set_seed(_config['seed'])
    cudnn.enabled = True
    cudnn.benchmark = True
    device = torch.device(f"cuda:{_config['gpu_id']}")
    # torch.cuda.set_device(device=_config['gpu_id'])
    # torch.set_num_threads(1)
    model = MedicalFSS(_config,device).to(device)

    _log.info('###### Load data ######')
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
        # batch_size=_config['batch_size'],
        shuffle=True,
        num_workers=_config['n_work'],
        pin_memory=False,#True
        drop_last=False
    )

    # all_samples = test_loader_Spleen(split=1) # for iterative validation

    _log.info('###### Set optimizer ######')
    print(_config['optim'])
    # optimizer = torch.optim.SGD(model.parameters(), **_config['optim'])
    optimizer = torch.optim.Adam(list(model.parameters()),
                                 _config['optim']['lr'])
    # scheduler = MultiStepLR(optimizer, milestones=_config['lr_milestones'], gamma=0.1)
    scheduler = MultiStepLR(optimizer, milestones=_config['lr_milestones'], gamma=0.1)
    criterion_ce = nn.CrossEntropyLoss()
    # criterion = nn.BCELoss()
    criterion = losses.DiceLoss()

    if _config['record']:  ## tensorboard visualization
        _log.info('###### define tensorboard writer #####')
        _log.info(f'##### board/train_{_config["board"]}_{date()}')
        writer = SummaryWriter(f'board/train_{_config["board"]}_{date()}')

    i_iter = 0
    log_loss = {'loss': 0, 'align_loss': 0}
    min_val_loss = 100000.0
    min_iter = 0
    min_epoch = 0
    iter_n_train, iter_n_val = len(trainloader), len(validationloader)

    _log.info('###### Training ######')
    q_slice_n = _config['q_slice']
    blank = torch.zeros([1, 256, 256]).to(device)
    iter_print = _config['iter_print']

    for i_epoch in range(_config['n_steps']):
        loss_epoch = 0

        ## training stage
        for i_iter, sample_train in enumerate(trainloader):
            preds = []
            loss_per_video = 0.0
            optimizer.zero_grad()
            s_x = sample_train['s_x'].to(device)  # [B, Support, slice_num, 1, 256, 256]
            s_y = sample_train['s_y'].to(device)  # [B, Support, slice_num, 1, 256, 256]
            q_x = sample_train['q_x'].to(device) #[B, slice_num, 1, 256, 256]
            q_y = sample_train['q_y'].type(torch.LongTensor).to(device) #[B, slice_num, 1, 256, 256]
            preds = model(s_x,s_y,q_x)

            for frame_id in range(q_slice_n):
                q_yi = q_y[:, frame_id, :, :, :] # [B, 1, 256, 256]
                q_yi2 = q_yi.squeeze(1)  # [B, 256, 256]
                yhati = preds[frame_id]
                # pdb.set_trace()
                loss = criterion(F.softmax(yhati, dim=1), q_yi2)+criterion_ce(F.softmax(yhati, dim=1), q_yi2)
                # loss = criterion(F.softmax(yhati, dim=1), q_yi2)
                # loss = criterion(yhati, q_yi)
                loss_per_video += loss
                preds.append(yhati)

            loss_per_video.backward()
            optimizer.step()
            loss_epoch += loss_per_video
            if iter_print:
                print(f"train, iter:{i_iter}/{iter_n_train}, iter_loss:{loss_per_video}", end='\r')

            if _config['record'] and i_iter == 0:
                batch_i = 0
                frames = []
                for frame_id in range(0, q_slice_n):
                    # query_pred = output.argmax(dim=1)

                    frames += overlay_color(q_x[batch_i, frame_id], preds[frame_id][batch_i].round(), q_y[batch_i, frame_id], scale=_config['scale'])
                for frame_id in range(0, q_slice_n):
                    frames += overlay_color(s_x[batch_i,  0, frame_id], blank, s_y[batch_i, 0, frame_id], scale=_config['scale'])

                visual = make_grid(frames, normalize=True, nrow=5)
                writer.add_image("train/visual", visual, i_epoch)

        with torch.no_grad(): ## validation stage
            loss_valid = 0
            preds = []

            for i_iter, sample_valid in enumerate(validationloader):
                loss_per_video = 0.0
                optimizer.zero_grad()
                s_x = sample_valid['s_x'].to(device)  # [B, slice_num, 1, 256, 256]
                s_y = sample_valid['s_y'].to(device)  # [B, slice_num, 1, 256, 256]
                q_x = sample_valid['q_x'].to(device)  # [B, slice_num, 1, 256, 256]
                q_y = sample_valid['q_y'].type(torch.LongTensor).to(device)  # [B, slice_num, 1, 256, 256]

                preds = model(s_x, s_y, q_x)

                for frame_id in range(q_slice_n):
                    q_yi = q_y[:, frame_id, :, :, :]  # [B, 1, 256, 256]
                    q_yi2 = q_yi.squeeze(1)  # [B, 256, 256]
                    yhati = preds[frame_id]
                    loss = criterion(F.softmax(yhati, dim=1), q_yi2) + criterion_ce(F.softmax(yhati, dim=1), q_yi2)
                    # loss = criterion(F.softmax(yhati, dim=1), q_yi2)

                    # loss = criterion(yhati, q_yi)
                    loss_per_video += loss
                    preds.append(yhati)

                loss_valid += loss_per_video

                if iter_print:
                    print(f"valid, iter:{i_iter}/{iter_n_val}, iter_loss:{loss_per_video}", end='\r')

                if _config['record'] and i_iter == 0:
                    batch_i = 0
                    frames = []
                    for frame_id in range(0, q_slice_n):
                        frames += overlay_color(q_x[batch_i, frame_id], preds[frame_id][batch_i].round(), q_y[batch_i, frame_id], scale=_config['scale'])

                    for frame_id in range(0, q_slice_n):
                        frames += overlay_color(s_x[batch_i, 0, frame_id], blank, s_y[batch_i, 0, frame_id], scale=_config['scale'])

                    visual = make_grid(frames, normalize=True, nrow=5)
                    writer.add_image("valid/visual", visual, i_epoch)

        if min_val_loss > loss_valid:
            min_epoch = i_epoch
            min_val_loss = loss_valid
            print(f"train - epoch:{i_epoch}/{_config['n_steps']}, epoch_loss:{loss_epoch} valid_loss:{loss_valid} \t => model saved", end='\n')
            save_fname = f'{_run.observers[0].dir}/snapshots/lowest.pth'
        else:
            print(f"train - epoch:{i_epoch}/{_config['n_steps']}, epoch_loss:{loss_epoch} valid_loss:{loss_valid} - min epoch:{min_epoch}", end='\n')
            save_fname = f'{_run.observers[0].dir}/snapshots/last.pth'

        _run.log_scalar("training.loss", float(loss_epoch), i_epoch)
        _run.log_scalar("validation.loss", float(loss_valid), i_epoch)
        _run.log_scalar("min_epoch", min_epoch, i_epoch)
        if _config['record']:
            writer.add_scalar('loss/train_loss', loss_epoch, i_epoch)
            writer.add_scalar('loss/valid_loss', loss_valid, i_epoch)

        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, save_fname
        )

    writer.close()
