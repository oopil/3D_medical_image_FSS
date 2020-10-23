import os
import re
import sys
import json
import math
import random
import numpy as np
sys.path.append("/home/soopil/Desktop/github/python_utils")
# sys.path.append("../dataloaders_medical")
from dataloaders_medical.common import *
# from common import *
import cv2
from cv2 import resize

def totensor(arr):
    tensor = torch.from_numpy(arr).float()
    return tensor

def random_augment(s_imgs, s_labels, q_imgs, q_labels):
    ## do random rotation and flip
    k = random.sample([i for i in range(0, 4)], 1)[0]
    s_imgs = np.rot90(s_imgs, k, (3, 4)).copy()
    s_labels = np.rot90(s_labels, k, (3, 4)).copy()
    q_imgs = np.rot90(q_imgs, k, (2, 3)).copy()
    q_labels = np.rot90(q_labels, k, (2, 3)).copy()

    if random.random() < 0.5:
        s_imgs = np.flip(s_imgs, 3).copy()
        s_labels = np.flip(s_labels, 3).copy()
        q_imgs = np.flip(q_imgs, 2).copy()
        q_labels = np.flip(q_labels, 2).copy()

    if random.random() < 0.5:
        s_imgs = np.flip(s_imgs, 4).copy()
        s_labels = np.flip(s_labels, 4).copy()
        q_imgs = np.flip(q_imgs, 3).copy()
        q_labels = np.flip(q_labels, 3).copy()

    return s_imgs, s_labels, q_imgs, q_labels

class Base_dataset_ctorg():
    def __init__(self, img_paths, label_paths, config):
        """
        dataset constructor for training
        """
        super().__init__()
        self.mode = config['mode']
        self.length = config['n_iter']
        self.size = config['size']
        self.img_paths = img_paths
        self.label_paths = label_paths

        self.is_train = True
        if str(self.__class__).split(".")[-1][:4]=="Test":
            self.is_train = False

        ## load file names in advance
        self.img_lists = []
        self.slice_cnts = []
        for img_path in self.img_paths:
            fnames = os.listdir(img_path)
            self.slice_cnts.append(len(fnames))
            fnames = [int(e.split(".")[0]) for e in fnames]
            fnames.sort()
            fnames = [f"{e}.npy" for e in fnames]
            self.img_lists.append(fnames)

    def get_sample(self, img_path, label_path):
        seed = random.randrange(0,1000)
        img_path, label_path = img_path, label_path
        img = self.img_load(img_path, seed)
        img = resize(img, dsize=(self.size, self.size), interpolation=cv2.INTER_AREA)
        img = np.expand_dims(img, axis=0)
        label = np.load(label_path)
        label = resize(label, dsize=(self.size, self.size), interpolation=cv2.INTER_NEAREST)
        label = np.expand_dims(label, axis=0)

        # align with other dataset
        img = np.flip(img, 1).copy()
        label = np.flip(label, 1).copy()

        sample = {
            "x":totensor(img),
            "y":totensor(label), #.long()
        }
        return sample

    def getitem_train(self):
        subj_idx = random.randrange(0, len(self.img_paths))
        subj_img_path = self.img_paths[subj_idx]
        subj_label_path = self.label_paths[subj_idx]
        fnames = self.img_lists[subj_idx]
        idx = random.randrange(0, len(fnames))
        fname_selected = fnames[idx]
        img_paths_selected = f"{subj_img_path}/{fname_selected}"
        label_paths_selected = f"{subj_label_path}/{fname_selected}"
        return self.get_sample(img_paths_selected, label_paths_selected)

    def getitme_test(self, idx):
        subj_idx, idx = self.get_subj_idx(idx)
        subj_img_path = self.img_paths[subj_idx]
        subj_label_path = self.label_paths[subj_idx]
        fnames = self.img_lists[subj_idx]
        fname_selected = fnames[idx]
        img_paths_selected = f"{subj_img_path}/{fname_selected}"
        label_paths_selected = f"{subj_label_path}/{fname_selected}"
        return self.get_sample(img_paths_selected, label_paths_selected)

    def get_len_train(self):
        return self.length

    def get_len_test(self):
        self.length = 0
        self.slice_cnts = []
        for label_path in self.label_paths:
            label_names = os.listdir(label_path)
            cnt_file = len(label_names)
            self.length += cnt_file
            self.slice_cnts.append(cnt_file)
        return self.length

    def get_subj_idx(self,idx):
        subj_idx = 0
        # print(f"{idx}/{self.length}")
        for cnt in self.slice_cnts:
            if idx < cnt:
                break
            else:
                idx -= cnt # idx is slice_idx here
                subj_idx += 1
        # print(self.slice_cnts)
        # print(subj_idx, idx)
        return subj_idx, idx

    def get_cnts(self):
        ## only for test loader
        return self.slice_cnts

    def img_load(self, img_path, seed=0):
        img_arr = np.load(img_path)+0.25
        return img_arr

class BaseLoader_CTORG(Base_dataset_ctorg):
    modal_i = [0] # there is only one modality
    label_i = 1.0 # there is only one label for each image

class TrainLoader_CTORG(BaseLoader_CTORG):
    def __len__(self):
        return self.get_len_train()

    def __getitem__(self, idx):
        return self.getitem_train()

class TestLoader_CTORG(BaseLoader_CTORG):
    def __len__(self):
        return self.get_len_test()

    def __getitem__(self, idx):
        return self.getitme_test(idx)

if __name__ == "__main__":
    pass
    # main()