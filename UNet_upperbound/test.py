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
# from util.metric import Metric
from util.utils import set_seed, CLASS_LABELS, get_bbox, date
from config import ex
from tensorboardX import SummaryWriter
from dataloaders_medical.prostate import *
from model import MedicalFSS

from nn_common_modules import losses
import torch.nn.functional as F

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

    _log.info('###### Load data ######')
    data_name = _config['dataset']
    make_data = meta_data
    q_slice_n = _config['q_slice']
    iter_print = _config['iter_print']

    if _config['record']:
        _log.info('###### define tensorboard writer #####')
        board_name = f'board/test_{_config["board"]}_{date()}'
        writer = SummaryWriter(board_name)

    if _config["n_update"]:
        _log.info('###### fine tuning with support data of target organ #####')
        _config["n_shot"] = _config["n_shot"]-1
        _log.info('###### Create model ######')
        model = MedicalFSS(_config, device).to(device)
        checkpoint = torch.load(_config['snapshot'], map_location='cpu')
        print("checkpoint keys : ", checkpoint.keys())
        # initializer.load_state_dict(checkpoint['initializer'])
        model.load_state_dict(checkpoint['model'])

        tr_dataset, val_dataset, ts_dataset = make_data(_config, is_finetuning=True)
        trainloader = DataLoader(
            dataset=tr_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=False,
            drop_last=False
        )

        optimizer = torch.optim.Adam(list(model.parameters()),_config['optim']['lr'])
        # criterion_ce = nn.CrossEntropyLoss()
        # criterion = losses.DiceLoss()
        criterion = nn.BCELoss()

        for i_iter, sample_train in enumerate(trainloader):
            preds = []
            loss_per_video = 0.0
            optimizer.zero_grad()
            s_x = sample_train['s_x'].to(device)  # [B, Support, slice_num, 1, 256, 256]
            s_y = sample_train['s_y'].to(device)  # [B, Support, slice_num, 1, 256, 256]
            preds = model(s_x)

            for frame_id in range(q_slice_n):
                s_yi = s_y[:, 0, frame_id, 0, :, :] # [B, 1, 256, 256]
                yhati = preds[frame_id]
                # loss = criterion(F.softmax(yhati, dim=1), q_yi2)
                # loss = criterion(F.softmax(yhati, dim=1), q_yi2)+criterion_ce(F.softmax(yhati, dim=1), q_yi2)
                loss = criterion(yhati, s_yi)

                loss_per_video += loss
                preds.append(yhati)

            loss_per_video.backward()
            optimizer.step()

            if iter_print:
                print(f"train, iter:{i_iter}/{_config['n_update']}, iter_loss:{loss_per_video}", end='\r')

        _config["n_shot"] = _config["n_shot"]+1
    else:
        _log.info('###### Create model ######')
        model = MedicalFSS(_config, device).to(device)
        checkpoint = torch.load(_config['snapshot'], map_location='cpu')
        print("checkpoint keys : ", checkpoint.keys())
        # initializer.load_state_dict(checkpoint['initializer'])
        model.load_state_dict(checkpoint['model'])

    model.n_shot = _config["n_shot"]
    tr_dataset, val_dataset, ts_dataset = make_data(_config)
    testloader = DataLoader(
        dataset=ts_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=False,
        drop_last=False
    )

    _log.info('###### Testing begins ######')
    # metric = Metric(max_label=max_label, n_runs=_config['n_runs'])
    img_cnt = 0
    # length = len(all_samples)
    length = len(testloader)
    blank = torch.zeros([1, 256, 256]).to(device)
    reversed_idx = list(reversed(range(q_slice_n)))
    ch = 256 # number of channels of embedding


    img_lists = []
    pred_lists = []
    label_lists = []

    saves = {}
    n_test = len(ts_dataset.q_cnts)
    for subj_idx in range(n_test):
        saves[subj_idx] = []

    with torch.no_grad():
        batch_idx = 0 # use only 1 batch size for testing

        for i, sample_test in enumerate(testloader): # even for upward, down for downward
            subj_idx, idx = ts_dataset.get_test_subj_idx(i)
            img_list, pred_list, label_list, preds = [],[],[],[]
            s_x = sample_test['s_x'].to(device)  # [B, Support, slice_num, 1, 256, 256]
            s_y = sample_test['s_y'].to(device)  # [B, Support, slice_num, 1, 256, 256]
            preds = model(s_x)

            for frame_id in range(q_slice_n):
                s_xi = s_x[:, 0, frame_id, :, :, :] # only 1 shot in upperbound model
                s_yi = s_y[:, 0, frame_id, :, :, :] # [B, 1, 256, 256]
                yhati = preds[frame_id]
                # pdb.set_trace()

                preds.append(yhati.round())
                img_list.append(s_xi[batch_idx].cpu().numpy())
                pred_list.append(yhati.round().cpu().numpy())
                label_list.append(s_yi[batch_idx].cpu().numpy())

            saves[subj_idx].append([subj_idx, idx, img_list, pred_list, label_list])

            if iter_print:
                print(f"test, iter:{i}/{length} - {subj_idx}/{idx} \t\t", end='\r')

            img_lists.append(img_list)
            pred_lists.append(pred_list)
            label_lists.append(label_list)

            # if _config['record']:
            #     frames = []
            #     for frame_id in range(0, q_x.size(1)):
            #         frames += overlay_color(q_x[batch_idx, frame_id], preds[frame_id-1][batch_idx].round(), q_y[batch_idx, frame_id], scale=_config['scale'])
            #     visual = make_grid(frames, normalize=True, nrow=5)
            #     writer.add_image(f"test/{subj_idx}/{idx}_query_image", visual, i)

    center_idx = (q_slice_n//2)+1 -1 # 5->2 index
    dice_similarities = []
    for subj_idx in range(n_test):
        imgs, preds, labels = [], [], []
        save_subj = saves[subj_idx]

        for i in range(len(save_subj)):
            subj_idx, idx, img_list, pred_list, label_list = save_subj[i]
            # if idx==(q_slice_n//2):
            if idx==0:
                for j in range((q_slice_n//2)+1):# 5//2 + 1 = 3
                    imgs.append(img_list[idx+j])
                    preds.append(pred_list[idx+j])
                    labels.append(label_list[idx+j])

            elif idx==(len(save_subj)-1):
                # pdb.set_trace()
                for j in range((q_slice_n//2)+1):# 5//2 + 1 = 3
                    imgs.append(img_list[center_idx+j])
                    preds.append(pred_list[center_idx+j])
                    labels.append(label_list[center_idx+j])

            else:
                imgs.append(img_list[center_idx])
                preds.append(pred_list[center_idx])
                labels.append(label_list[center_idx])

        # pdb.set_trace()
        img_arr = np.concatenate(imgs, axis=0)
        pred_arr = np.concatenate(preds, axis=0)
        label_arr = np.concatenate(labels, axis=0)
        dice = np.sum([label_arr * pred_arr]) * 2.0 / (np.sum(pred_arr) + np.sum(label_arr))
        dice_similarities.append(dice)
        print(f"{len(imgs)} slice -> computing dice scores. {subj_idx}/{n_test}. {ts_dataset.q_cnts[subj_idx] }/{len(save_subj)} => {len(imgs)}", end='\r')

        if _config['record']:
            frames = []
            for frame_id in range(0, len(imgs)):
                frames += overlay_color(torch.tensor(imgs[frame_id]), torch.tensor(preds[frame_id]), torch.tensor(labels[frame_id]), scale=_config['scale'])
            print(len(frames))
            visual = make_grid(frames, normalize=True, nrow=5)
            writer.add_image(f"test/{subj_idx}", visual, i)
            writer.add_scalar(f'dice_score/{i}', dice)

    print(f"test result \n n : {len(dice_similarities)}, mean dice score : \
    {np.mean(dice_similarities)} \n dice similarities : {dice_similarities}")

    if _config['record']:
        writer.add_scalar(f'dice_score/mean', np.mean(dice_similarities))

