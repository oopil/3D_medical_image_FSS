"""Evaluation Script"""
import os
import shutil
import pdb
import tqdm
import numpy as np
import torch
import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torchvision.transforms import Compose
from torchvision.utils import make_grid
from math import isnan

from models.encoder import SupportEncoder, QueryEncoder
from models.decoder import Decoder

# from util.metric import Metric
from util.utils import set_seed, CLASS_LABELS, get_bbox, date
from config import ex
# from tensorboardX import SummaryWriter
from dataloaders_medical.prostate import *
import matplotlib.pyplot as plt

def try_mkdirs(path):
    try:
        os.makedirs(path)
        return True
    except:
        return False

@ex.automain
def main(_run, _config, _log):
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
    device = torch.device(f"cuda:{_config['gpu_id']}")

    _log.info('###### Create model ######')
    resize_dim = _config['input_size']
    encoded_h = int(resize_dim[0] / 2**_config['n_pool'])
    encoded_w = int(resize_dim[1] / 2**_config['n_pool'])

    s_encoder = SupportEncoder(_config['path']['init_path'], device)#.to(device)
    q_encoder = QueryEncoder(_config['path']['init_path'], device)#.to(device)
    decoder = Decoder(input_res=(encoded_h, encoded_w), output_res=resize_dim).to(device)

    checkpoint = torch.load(_config['snapshot'], map_location='cpu')
    s_encoder.load_state_dict(checkpoint['s_encoder'])
    q_encoder.load_state_dict(checkpoint['q_encoder'])
    decoder.load_state_dict(checkpoint['decoder'])

    # initializer.eval()
    # encoder.eval()
    # convlstmcell.eval()
    # decoder.eval()

    _log.info('###### Load data ######')
    data_name = _config['dataset']
    make_data = meta_data
    max_label = 1

    tr_dataset, val_dataset, ts_dataset = make_data(_config)
    testloader = DataLoader(
        dataset=ts_dataset,
        batch_size=1,
        shuffle=False,
        # num_workers=_config['n_work'],
        pin_memory=False, # True
        drop_last=False
    )

    _log.info('###### Testing begins ######')
    # metric = Metric(max_label=max_label, n_runs=_config['n_runs'])
    img_cnt = 0
    # length = len(all_samples)
    length = len(testloader)
    img_lists = []
    pred_lists = []
    label_lists = []

    saves = {}
    for subj_idx in range(len(ts_dataset.get_cnts())):
        saves[subj_idx] = []

    with torch.no_grad():
        loss_valid = 0
        batch_i = 0 # use only 1 batch size for testing

        for i, sample_test in enumerate(testloader): # even for upward, down for downward
            subj_idx, idx = ts_dataset.get_test_subj_idx(i)
            img_list = []
            pred_list = []
            label_list = []
            preds = []

            s_x = sample_test['s_x'].to(device)  # [B, slice_num, 1, 256, 256]
            s_y = sample_test['s_y'].to(device)  # [B, slice_num, 1, 256, 256]
            q_x = sample_test['q_x'].to(device)  # [B, slice_num, 1, 256, 256]
            q_y = sample_test['q_y'].to(device)  # [B, slice_num, 1, 256, 256]
            s_fname = sample_test['s_fname']
            q_fname = sample_test['q_fname']

            s_xi = s_x[:, 0, :, :, :] #[B, 1, 256, 256]
            s_yi = s_y[:, 0, :, :, :]
            s_xi_encode, _ = s_encoder(s_xi, s_yi) #[B, 512, w, h]
            q_xi = q_x[:, 0, :, :, :]
            q_yi = q_y[:, 0, :, :, :]
            q_xi_encode, q_ft_list = q_encoder(q_xi)
            sq_xi = torch.cat((s_xi_encode, q_xi_encode),dim=1)
            yhati = decoder(sq_xi, q_ft_list)  # [B, 1, 256, 256]

            preds.append(yhati.round())
            img_list.append(q_xi[batch_i].cpu().numpy())
            pred_list.append(yhati[batch_i].round().cpu().numpy())
            label_list.append(q_yi[batch_i].cpu().numpy())

            saves[subj_idx].append([subj_idx, idx, img_list, pred_list, label_list])
            print(f"test, iter:{i}/{length} - {subj_idx}/{idx} \t\t", end='\r')
            img_lists.append(img_list)
            pred_lists.append(pred_list)
            label_lists.append(label_list)

            q_fname_split = q_fname[0][0].split("/")
            q_fname_split[-6] = "Training_2d_2_pred"
            try_mkdirs("/".join(q_fname_split[:-1]))
            o_q_fname = "/".join(q_fname_split)
            np.save(o_q_fname,yhati.round().cpu().numpy())
            # print(q_fname[0][0])
            # print(o_q_fname)

    try_mkdirs("figure")
    print("start computing dice similarities ... total ", len(saves))
    for subj_idx in range(len(saves)):
        save_subj = saves[subj_idx]
        dices = []

        for slice_idx in range(len(save_subj)):
            subj_idx, idx, img_list, pred_list, label_list = save_subj[slice_idx]

            for j in range(len(img_list)):
                dice = np.sum([label_list[j] * pred_list[j]]) * 2.0 / (np.sum(pred_list[j]) + np.sum(label_list[j]))
                dices.append(dice)

        plt.clf()
        plt.bar([k for k in range(len(dices))],dices)
        plt.savefig(f"figure/bar_{_config['target']}_{subj_idx}.png")