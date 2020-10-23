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
    from models.encoder import SupportEncoder, QueryEncoder
    from models.decoder import Decoder
else:
    from .util.utils import set_seed, CLASS_LABELS, date
    from .config import ex
    from tensorboardX import SummaryWriter
    from .dataloaders_medical.common import *
    from .dataloaders_medical.prostate import *
    from .models.encoder import SupportEncoder, QueryEncoder
    from .models.decoder import Decoder

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

    resize_dim = _config['input_size']
    encoded_h = int(resize_dim[0] / 2**_config['n_pool'])
    encoded_w = int(resize_dim[1] / 2**_config['n_pool'])

    s_encoder = SupportEncoder(_config['path']['init_path'], device)#.to(device)
    q_encoder = QueryEncoder(_config['path']['init_path'], device)#.to(device)
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

    _log.info('###### Set optimizer ######')
    print(_config['optim'])
    optimizer = torch.optim.Adam(#list(initializer.parameters()) +
                                 list(s_encoder.parameters()) +
                                 list(q_encoder.parameters()) +
                                 list(decoder.parameters()),
                                 _config['optim']['lr'])
    scheduler = MultiStepLR(optimizer, milestones=_config['lr_milestones'], gamma=0.1)
    pos_weight = torch.tensor([0.3 , 1], dtype=torch.float).to(device)
    criterion = nn.BCELoss()

    if _config['record']:  ## tensorboard visualization
        _log.info('###### define tensorboard writer #####')
        _log.info(f'##### board/train_{_config["board"]}_{date()}')
        writer = SummaryWriter(f'board/train_{_config["board"]}_{date()}')

    iter_n_train = len(trainloader)
    _log.info('###### Training ######')
    for i_epoch in range(_config['n_steps']):
        loss_epoch = 0
        blank = torch.zeros([1, 256, 256]).to(device)

        for i_iter, sample_train in enumerate(trainloader):
            ## training stage
            optimizer.zero_grad()
            s_x = sample_train['s_x'].to(device)  # [B, Support, slice_num, 1, 256, 256]
            s_y = sample_train['s_y'].to(device)  # [B, Support, slice_num, 1, 256, 256]
            q_x = sample_train['q_x'].to(device) #[B, slice_num, 1, 256, 256]
            q_y = sample_train['q_y'].to(device) #[B, slice_num, 1, 256, 256]
            # loss_per_video = 0.0
            s_xi = s_x[:, :, 0, :, :, :]  # [B, Support, 1, 256, 256]
            s_yi = s_y[:, :, 0, :, :, :]

            # for s_idx in range(_config["n_shot"]):
            s_x_merge = s_xi.view(s_xi.size(0) * s_xi.size(1), 1, 256, 256)
            s_y_merge = s_yi.view(s_yi.size(0) * s_yi.size(1), 1, 256, 256)
            s_xi_encode_merge, _ = s_encoder(s_x_merge, s_y_merge)  # [B*S, 512, w, h]

            s_xi_encode = s_xi_encode_merge.view(s_yi.size(0), s_yi.size(1), 512, encoded_w, encoded_h)
            s_xi_encode_avg = torch.mean(s_xi_encode, dim=1)
            # s_xi_encode, _ = s_encoder(s_xi, s_yi)  # [B, 512, w, h]
            q_xi = q_x[:, 0, :, :, :]
            q_yi = q_y[:, 0, :, :, :]
            q_xi_encode, q_ft_list = q_encoder(q_xi)
            sq_xi = torch.cat((s_xi_encode_avg, q_xi_encode), dim=1)
            yhati = decoder(sq_xi, q_ft_list)  # [B, 1, 256, 256]
            loss = criterion(yhati, q_yi)
            # loss_per_video += loss
            # loss_per_video.backward()
            loss.backward()
            optimizer.step()
            loss_epoch += loss
            print(f"train, iter:{i_iter}/{iter_n_train}, iter_loss:{loss}", end='\r')

            if _config['record'] and i_iter == 0:
                batch_i = 0
                frames = []
                frames += overlay_color(q_xi[batch_i], yhati[batch_i].round(), q_yi[batch_i], scale=_config['scale'])
                visual = make_grid(frames, normalize=True, nrow=2)
                writer.add_image("train/visual", visual, i_epoch)

                if _config['record'] and i_iter == 0:
                    batch_i = 0
                    frames = []
                    frames += overlay_color(q_xi[batch_i], yhati[batch_i].round(), q_yi[batch_i],
                                            scale=_config['scale'])
                    # frames += overlay_color(s_xi[batch_i], blank, s_yi[batch_i], scale=_config['scale'])
                    visual = make_grid(frames, normalize=True, nrow=5)
                    writer.add_image("valid/visual", visual, i_epoch)

        print(f"train - epoch:{i_epoch}/{_config['n_steps']}, epoch_loss:{loss_epoch}", end='\n')
        save_fname = f'{_run.observers[0].dir}/snapshots/last.pth'

        _run.log_scalar("training.loss", float(loss_epoch), i_epoch)
        if _config['record']:
            writer.add_scalar('loss/train_loss', loss_epoch, i_epoch)

        torch.save({
                's_encoder': s_encoder.state_dict(),
                'q_encoder': q_encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, save_fname
        )

    writer.close()