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
if __name__ == '__main__':
    from util.utils import set_seed, CLASS_LABELS, date
    from config import ex
    from tensorboardX import SummaryWriter
    from dataloaders_medical.common import *
    from dataloaders_medical.prostate import *
    from models.encoder import Encoder
    from models.decoder import Decoder
else:
    from .util.utils import set_seed, CLASS_LABELS, date
    from .config import ex
    from tensorboardX import SummaryWriter
    from .dataloaders_medical.common import *
    from .dataloaders_medical.prostate import *
    from .models.encoder import Encoder
    from .models.decoder import Decoder


def overlay(img, mask, idx=0, scale=50):
    """
    :param img: [1, 256, 256]
    :param mask: [1, 256, 256]
    :return:
    """
    # pdb.set_trace()
    mask = mask[0]
    zeros = torch.zeros_like(mask)
    zeros = [zeros for _ in range(3)]
    zeros[idx] = mask
    mask = torch.stack(zeros,dim=0)
    img_3ch = torch.cat([img,img,img],dim=0)
    masked = img_3ch+mask.float()*scale
    return [masked]

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
    device = torch.device(f"cuda:{_config['gpu_id']}")
    # torch.cuda.set_device(device=_config['gpu_id'])
    # torch.set_num_threads(1)

    resize_dim = _config['input_size']
    encoded_h = int(resize_dim[0] / 2**_config['n_pool'])
    encoded_w = int(resize_dim[1] / 2**_config['n_pool'])
    encoder = Encoder(_config['path']['init_path'], device)#.to(device)
    decoder = Decoder(input_res=(encoded_h, encoded_w), output_res=resize_dim).to(device)

    _log.info('###### Load data ######')
    data_name = _config['dataset']
    if data_name == 'prostate':
        make_data = meta_data
    else:
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
    optimizer = torch.optim.Adam(list(encoder.parameters()) +
                                 list(decoder.parameters()),
                                 _config['optim']['lr'])
    # scheduler = MultiStepLR(optimizer, milestones=_config['lr_milestones'], gamma=0.1)
    scheduler = MultiStepLR(optimizer, milestones=_config['lr_milestones'], gamma=0.1)
    pos_weight = torch.tensor([0.3 , 1], dtype=torch.float).to(device)
    # criterion = nn.CrossEntropyLoss(weight=pos_weight)
    criterion = nn.BCELoss()
    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # criterion = nn.MSELoss()

    if _config['record']:  ## tensorboard visualization
        _log.info('###### define tensorboard writer #####')
        _log.info(f'##### board/train_{_config["board"]}_{date()}')
        writer = SummaryWriter(f'board/train_{_config["board"]}_{date()}')

    blank = torch.zeros([1, 256, 256]).to(device)

    i_iter = 0
    log_loss = {'loss': 0, 'align_loss': 0}
    min_val_loss = 100000.0
    min_iter = 0
    min_epoch = 0
    iter_n_train, iter_n_val = len(trainloader), len(validationloader)
    _log.info('###### Training ######')

    for i_epoch in range(_config['n_steps']):
        loss_epoch = 0

        for i_iter, sample_train in enumerate(trainloader):
            # optimizer.zero_grad()
            x = sample_train['x'].to(device) #[B, 1, 256, 256]
            y = sample_train['y'].to(device) #[B, 1, 256, 256]
            x_encode, ft_list = encoder(x)
            yhat = decoder(x_encode, ft_list)
            loss = criterion(yhat, y)
            loss.backward()
            optimizer.step()
            loss_epoch += loss
            print(f"train, iter:{i_iter}/{iter_n_train}, iter_loss:{loss}", end='\r')

        if _config['record']:
            # pdb.set_trace()
            batch_i = 0
            frames = overlay_color(x[batch_i], yhat[batch_i].round(), y[batch_i], scale=_config['scale'])
            visual = make_grid(frames, normalize=True, nrow=5)
            writer.add_image("train/visual", visual, i_epoch)

        with torch.no_grad():
            loss_valid = 0

            for i_iter, sample_valid in enumerate(validationloader):
                x = sample_valid['x'].to(device)  # [B, 1, 256, 256]
                y = sample_valid['y'].to(device)  # [B, 1, 256, 256]
                x_encode, ft_list = encoder(x)
                yhat = decoder(x_encode, ft_list)
                loss = criterion(yhat, y)
                loss_valid += loss
                print(f"valid, iter:{i_iter}/{iter_n_val}, iter_loss:{loss}", end='\r')

            if _config['record']:
                batch_i = 0
                frames = overlay_color(x[batch_i], yhat[batch_i].round(), y[batch_i], scale=_config['scale'])
                visual = make_grid(frames, normalize=True, nrow=5)
                writer.add_image("valid/visual", visual, i_epoch)

        if min_val_loss > loss_valid:
            min_epoch = i_epoch
            min_val_loss = loss_valid
            print(f"train - epoch:{i_epoch}, epoch_loss:{loss_epoch} valid_loss:{loss_valid} \t => model saved", end='\n')
            save_fname = f'{_run.observers[0].dir}/snapshots/lowest.pth'
        else:
            print(f"train - epoch:{i_epoch}, epoch_loss:{loss_epoch} valid_loss:{loss_valid} - min epoch:{min_epoch}", end='\n')
            save_fname = f'{_run.observers[0].dir}/snapshots/last.pth'

        _run.log_scalar("training.loss", float(loss_epoch), i_epoch)
        _run.log_scalar("validation.loss", float(loss_valid), i_epoch)
        _run.log_scalar("min_epoch", min_epoch, i_epoch)
        if _config['record']:
            writer.add_scalar('loss/train_loss', loss_epoch, i_epoch)
            writer.add_scalar('loss/valid_loss', loss_valid, i_epoch)

        torch.save({
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                # 'loss': epoch_avg_loss,
            }, save_fname
        )
    writer.close()