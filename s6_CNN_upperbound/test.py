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
from models.encoder import Encoder
from models.decoder import Decoder
# from util.metric import Metric
from util.utils import set_seed, CLASS_LABELS, get_bbox, date
from config import ex
from tensorboardX import SummaryWriter
from dataloaders_medical.prostate import *

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

    encoder = Encoder(_config['path']['init_path'], device)#.to(device)
    decoder = Decoder(input_res=(encoded_h, encoded_w), output_res=resize_dim).to(device)

    checkpoint = torch.load(_config['snapshot'], map_location='cpu')
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])

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
        pin_memory=True,
        drop_last=False
    )
    # all_samples = test_loader_Spleen()
    # all_samples = test_loader_Prostate()

    if _config['record']:
        _log.info('###### define tensorboard writer #####')
        board_name=f'board/test_{_config["board"]}_{date()}'
        _log.info(board_name)
        writer = SummaryWriter(board_name)

    _log.info('###### Testing begins ######')
    # metric = Metric(max_label=max_label, n_runs=_config['n_runs'])
    length = len(testloader)
    slice_cnt = ts_dataset.get_cnts()
    subj_old_idx = -1
    img_list = []
    pred_list = []
    label_list = []

    with torch.no_grad():
        batch_i = 0 # use only 1 batch size for testing
        for i_iter, sample_valid in enumerate(testloader):
            x = sample_valid['x'].to(device)  # [B, 1, 256, 256]
            y = sample_valid['y'].to(device)  # [B, 1, 256, 256]
            x_encode, ft_list = encoder(x)
            yhat = decoder(x_encode, ft_list)
            print(f"test, iter:{i_iter}/{length}", end='\r')
            img_list.append(x[batch_i].cpu().numpy())
            pred_list.append(yhat[batch_i].round().cpu().numpy())
            label_list.append(y[batch_i].cpu().numpy())

    print()
    dice_similarities = []
    img_lists = []
    pred_lists = []
    label_lists = []
    start_idx = 0

    print(f"{len(slice_cnt)} slice count : {slice_cnt}")
    for i,slice in enumerate(slice_cnt):
        imgs = img_list[start_idx: start_idx+slice]
        labels = label_list[start_idx: start_idx+slice]
        preds = pred_list[start_idx: start_idx+slice]
        start_idx += slice

        if _config['record']:
            frames = []
            for j in range(slice):
                batch_i = 0
                frames += overlay_color(torch.tensor(imgs[j]), torch.tensor(preds[j]), torch.tensor(labels[j]), scale=_config['scale'])

            visual = make_grid(frames, normalize=True, nrow=5)
            writer.add_image(f"test/{i}_query_image", visual, i)

        img_arr = np.concatenate(imgs, axis=0)
        pred_arr = np.concatenate(preds, axis=0)
        label_arr = np.concatenate(labels, axis=0)
        dice = np.sum([label_arr * pred_arr]) * 2.0 / (np.sum(pred_arr) + np.sum(label_arr))
        dice_similarities.append(dice)
        _run.log_scalar("test.dice_score", float(dice), i)
        print(f"board, iter:{i}/{len(slice_cnt)}", end='\r')

    print()
    print(f"test result \n n : {len(dice_similarities)}, mean dice score : {np.mean(dice_similarities)} \n dice similarities : {dice_similarities}")
    return