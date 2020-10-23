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
from torchvision.utils import make_grid

from util.utils import set_seed, CLASS_LABELS, get_bbox, date
from config import ex

from tensorboardX import SummaryWriter
from dataloaders_medical.prostate import *
from settings import Settings
import few_shot_segmentor as fs

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

    settings = Settings()
    common_params, data_params, net_params, train_params, eval_params = settings['COMMON'], settings['DATA'], settings[
        'NETWORK'], settings['TRAINING'], settings['EVAL']

    _log.info('###### Create model ######')
    model = fs.FewShotSegmentorDoubleSDnet(net_params).cuda()
    # model = nn.DataParallel(model.cuda(), device_ids=[_config['gpu_id'],])
    if not _config['notrain']:
        model.load_state_dict(torch.load(_config['snapshot'], map_location='cpu'))
    model.eval()


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


    if _config['record']:
        _log.info('###### define tensorboard writer #####')
        board_name = f'board/test_{_config["board"]}_{date()}'
        writer = SummaryWriter(board_name)

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

            s_x = sample_test['s_x'].cuda()  # [B, Support, slice_num=1, 1, 256, 256]
            X = s_x.squeeze(2)  # [B, Support, 1, 256, 256]
            s_y = sample_test['s_y'].cuda()  # [B, Support, slice_num, 1, 256, 256]
            Y = s_y.squeeze(2)  # [B, Support, 1, 256, 256]
            Y = Y.squeeze(2)  # [B, Support, 256, 256]
            q_x = sample_test['q_x'].cuda()  # [B, slice_num, 1, 256, 256]
            query_input = q_x.squeeze(1)  # [B, 1, 256, 256]
            q_y = sample_test['q_y'].cuda()  # [B, slice_num, 1, 256, 256]
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
            for k in range(9):
                weight_cat = torch.cat([weights[0][k] for weights in entire_weights],dim=1)
                avg_weight = torch.mean(weight_cat,dim=1,keepdim=True)
                avg_weights[0].append(avg_weight)

            avg_weights[0].append(None)

            output = model.segmentor(query_input, avg_weights)

            q_yhat = output.argmax(dim=1)
            q_yhat = q_yhat.unsqueeze(1)

            preds.append(q_yhat)
            img_list.append(q_x[batch_i,0].cpu().numpy())
            pred_list.append(q_yhat[batch_i].cpu().numpy())
            label_list.append(q_y[batch_i,0].cpu().numpy())

            saves[subj_idx].append([subj_idx, idx, img_list, pred_list, label_list])
            print(f"test, iter:{i}/{length} - {subj_idx}/{idx} \t\t", end='\r')
            img_lists.append(img_list)
            pred_lists.append(pred_list)
            label_lists.append(label_list)

    print("start computing dice similarities ... total ", len(saves))
    dice_similarities = []
    for subj_idx in range(len(saves)):
        imgs, preds, labels = [], [], []
        save_subj = saves[subj_idx]
        for i in range(len(save_subj)):
            # print(len(save_subj), len(save_subj)-q_slice_n+1, q_slice_n, i)
            subj_idx, idx, img_list, pred_list, label_list = save_subj[i]
            # print(subj_idx, idx, is_reverse, len(img_list))
            # print(i, is_reverse, is_reverse_next, is_flip)

            for j in range(len(img_list)):
                imgs.append(img_list[j])
                preds.append(pred_list[j])
                labels.append(label_list[j])

        # pdb.set_trace()
        img_arr = np.concatenate(imgs, axis=0)
        pred_arr = np.concatenate(preds, axis=0)
        label_arr = np.concatenate(labels, axis=0)
        # pdb.set_trace()
        # print(ts_dataset.slice_cnts[subj_idx] , len(imgs))
        # pdb.set_trace()
        dice = np.sum([label_arr * pred_arr]) * 2.0 / (np.sum(pred_arr) + np.sum(label_arr))
        dice_similarities.append(dice)
        print(f"computing dice scores {subj_idx}/{10}", end='\n')

        if _config['record']:
            frames = []
            for frame_id in range(0, len(save_subj)):
                frames += overlay_color(torch.tensor(imgs[frame_id]), torch.tensor(preds[frame_id]).float(), torch.tensor(labels[frame_id]))
            visual = make_grid(frames, normalize=True, nrow=5)
            writer.add_image(f"test/{subj_idx}", visual, i)
            writer.add_scalar(f'dice_score/{i}', dice)

    print(f"test result \n n : {len(dice_similarities)}, mean dice score : \
    {np.mean(dice_similarities)} \n dice similarities : {dice_similarities}")

    if _config['record']:
        writer.add_scalar(f'dice_score/mean', np.mean(dice_similarities))
