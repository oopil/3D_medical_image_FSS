import sys
import glob
import json
import re
from glob import glob
from util.utils import *
from dataloaders_medical.decathlon import *
from dataloaders_medical.dataset_decathlon import *
from dataloaders_medical.dataset_CT_ORG import *
import numpy as np

class MetaSliceData_train():
    def __init__(self, datasets, iter_n = 100):
        super().__init__()
        self.datasets = datasets
        self.dataset_n = len(datasets)
        self.iter_n =iter_n

    def __len__(self):
        return self.iter_n

    def __getitem__(self, idx):
        dataset = random.sample(self.datasets, 1)[0]
        return dataset.__getitem__(idx)

def metadata():
    info = {
    "src_dir" : "/user/home2/soopil/Datasets/MICCAI2015challenge/Abdomen/RawData/Training",
    "trg_dir" : "/user/home2/soopil/Datasets/MICCAI2015challenge/Abdomen/RawData/Training_2d", # 144 setting
    "trg_dir2" : "/user/home2/soopil/Datasets/MICCAI2015challenge/Abdomen/RawData/Training_2d_2", # 144 setting
    "trg_dir3" : "/user/home2/soopil/Datasets/MICCAI2015challenge/Abdomen/RawData/Training_2d_denoise",

    # "trg_dir" : "/home/soopil/Desktop/Dataset/MICCAI2015challenge/Abdomen/RawData/Training_2d", # desktop setting
    "Tasks" : [i for i in range(1,14)],
    # "Tasks" : [i for i in range(1,17+1)],
    # training : [1,2,3,5,6,7,8,9,14,15]
    # testing : [1,3,6,14]
    "Organs" : ["background",
              "spleen", #1
              "right kidney", #2
              "left kidney", #3
              "gallbladder", #4
              "esophagus", #5
              "liver", #6
              "stomach", #7
              "aorta", #8
              "inferior vana cava", #9
              "portal vein & splenic vein", #10
              "pancreas", #11
              "right adrenal gland", #12
              "left adrenal gland", #13
              "bladder", #14
              "uturus", #15
              "rectum", #16
              "small bowel", #17
              ],
    }
    return info

def meta_data(_config):
    def path_collect(idx, option='train'):
        # img_paths = glob(f"{meta['trg_dir2']}/{idx}/{option}/img/*")
        # label_paths = glob(f"{meta['trg_dir2']}/{idx}/{option}/label/*")
        img_paths = glob(f"{_config['data_src']}/{idx}/{option}/img/*")
        label_paths = glob(f"{_config['data_src']}/{idx}/{option}/label/*")
        return img_paths, label_paths

    def spliter(idx):
        tr_imgs, tr_labels = path_collect(idx, 'train')
        val_imgs, val_labels = path_collect(idx, 'valid')
        ts_imgs, ts_labels = path_collect(idx, 'test')
        return tr_imgs, tr_labels, val_imgs, val_labels, ts_imgs, ts_labels

    target_task = _config['target']
    meta = metadata()
    print(meta['trg_dir'])
    # tasks = meta['Tasks']
    tasks = [1,2,3,5,6,7,8,9,14,15]
    # tasks_remove = [4, 10, 12, 13] # 7 11
    # tasks_remove = [4, 5, 8, 9, 10, 11, 12, 13]
    # tasks_remove = [4, 5, 8, 9, 10, 11, 12, 13, 16, 17]
    ## we sholdn't use both left and right kidneys
    # for task in tasks_remove:
    #     tasks.remove(task)

    kidneys = [2,3]
    if target_task in kidneys:
        kidneys.remove(target_task)
        other_task = kidneys[0]
        try:
            tasks.remove(other_task)
        except:
            pass

    print(f"tasks : {tasks}")
    datasets = {}
    for task in tasks:
        tr_imgs, tr_labels, val_imgs, val_labels, ts_imgs, ts_labels = spliter(task)
        datasets[task] = [TrainLoader(tr_imgs, tr_labels, _config), TestLoader(val_imgs, val_labels, _config), TestLoader(ts_imgs, ts_labels, _config)]

    tr_imgs, tr_labels, val_imgs, val_labels, ts_imgs, ts_labels = spliter(target_task)
    if _config['add_target']:
        n_add_target = _config['add_target']
        datasets[target_task] = [TrainLoader(tr_imgs[:n_add_target], tr_labels[:n_add_target], _config), TestLoader(val_imgs, val_labels, _config), TestLoader(ts_imgs, ts_labels, _config)]
        val_dataset = datasets[target_task][1]
        ts_dataset = datasets[target_task][2]
        tr_datasets = [dataset[0] for dataset in datasets.values()]
    else:
        val_dataset = datasets[target_task][1]
        ts_dataset = datasets[target_task][2]
        datasets.pop(target_task) #dictionary pop(key)
        tr_datasets = [dataset[0] for dataset in datasets.values()]

    print(f"training tasks : {datasets.keys()}")
    print(f"target tasks : {target_task}")
    ## set the support volume for testing
    if _config["internal_test"]:
        pass
    else:
        # _, _, ts_dataset = external_trainset(_config,target_task)
        tr_imgs, tr_labels, ts_dataset = external_testset(_config, target_task)

    val_dataset.set_support_volume(tr_imgs[_config['s_idx']:_config['s_idx'] + _config['n_shot']],
                                   tr_labels[_config['s_idx']:_config['s_idx'] + _config['n_shot']])
    ts_dataset.set_support_volume(tr_imgs[_config['s_idx']:_config['s_idx'] + _config['n_shot']],
                                  tr_labels[_config['s_idx']:_config['s_idx'] + _config['n_shot']])
    meta_tr_dataset = MetaSliceData_train(tr_datasets, iter_n=_config['n_iter'])

    return meta_tr_dataset, val_dataset, ts_dataset


def external_testset(_config, target_task):
    def decathlon_spliter(idx):
        def path_collect(idx, option='train'):
            tasks = ["Task01_BrainTumour",
                     "Task02_Heart",
                     "Task03_Liver",
                     "Task04_Hippocampus",
                     "Task05_Prostate",
                     "Task06_Lung",
                     "Task07_Pancreas",
                     "Task08_HepaticVessel",
                     "Task09_Spleen",
                     "Task10_Colon",
                     "Task11_Davis"
                     ]

            src_path = '/user/home2/soopil/Datasets/Decathlon_2d'
            img_paths = glob(f"{src_path}/{tasks[idx - 1]}/{option}/img/*")
            label_paths = glob(f"{src_path}/{tasks[idx - 1]}/{option}/label/*")
            return img_paths, label_paths

        tr_imgs, tr_labels = path_collect(idx, 'train')
        ts_imgs, ts_labels = path_collect(idx, 'test')
        return tr_imgs, tr_labels, ts_imgs, ts_labels

    def CT_ORG_spliter(idx):
        def path_collect(idx, option='train'):
            Organs = ["background",
                      "Liver",  # 1
                      "Bladder",  # 2
                      "Lung",  # 3
                      "Kidney",  # 4
                      "Bone",  # 5
                      "Brain",  # 6
                      ],

            src_path = "/user/home2/soopil/Datasets/CT_ORG/Training_2d_align"
            img_paths = glob(f"{src_path}/{idx}/{option}/img/*")
            label_paths = glob(f"{src_path}/{idx}/{option}/label/*")
            return img_paths, label_paths

        tr_imgs, tr_labels = path_collect(idx, 'train')
        ts_imgs, ts_labels = path_collect(idx, 'test')
        return tr_imgs, tr_labels, ts_imgs, ts_labels

    external = _config["external_test"]
    print(f"external testset : {external}")
    if external == "decathlon":
        if target_task == 1:  # spleen
            target_idx_decath = 9
            tr_imgs, tr_labels, ts_imgs, ts_labels = decathlon_spliter(target_idx_decath)
            ts_dataset = Spleen_test(ts_imgs, ts_labels, _config)

        elif target_task == 6:
            target_idx_decath = 3
            tr_imgs, tr_labels, ts_imgs, ts_labels = decathlon_spliter(target_idx_decath)
            ts_dataset = Liver_test(ts_imgs, ts_labels, _config)

        else:
            print("There isn't according organ in Decathlon dataset.")
            assert False

        print(f"target index in external dataset : {target_idx_decath}")

    elif external == "CT_ORG":
        if target_task == 3:  # kidney
            target_idx_ctorg = 4
            tr_imgs, tr_labels, ts_imgs, ts_labels = CT_ORG_spliter(target_idx_ctorg)
            ts_dataset = TestLoader_CTORG(ts_imgs, ts_labels, _config)

        elif target_task == 6:  # liver
            target_idx_ctorg = 1
            tr_imgs, tr_labels, ts_imgs, ts_labels = CT_ORG_spliter(target_idx_ctorg)
            ts_dataset = TestLoader_CTORG(ts_imgs, ts_labels, _config)

        elif target_task == 14:  # bladder
            target_idx_ctorg = 2
            tr_imgs, tr_labels, ts_imgs, ts_labels = CT_ORG_spliter(target_idx_ctorg)
            ts_dataset = TestLoader_CTORG(ts_imgs, ts_labels, _config)

        else:
            print("There isn't according organ in CT_ORG dataset.")
            assert False

        print(f"target index in external dataset : {target_idx_ctorg}")

    else:
        print("configuration of external dataset is wrong")
        assert False

    return tr_imgs, tr_labels, ts_dataset


if __name__=="__main__":
    pass