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

def prostate_img_process(img_arr, HE=False):
    if HE:
        img_arr = equalize_hist(img_arr) * 255.0
    else:
        img_arr = normalize(img_arr, type=0)
    return img_arr

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

class Base_dataset():
    def __init__(self, img_paths, label_paths, config):
        """
        dataset constructor for training
        """
        super().__init__()
        self.mode = config['mode']
        self.length = len(img_paths)
        print(f"# of data : {self.length}")
        self.valid_img_n = len(img_paths)
        self.size = config['size']
        self.img_paths = img_paths
        self.label_paths = label_paths
        self.q_slice = config["q_slice"]
        self.n_shot = config["n_shot"]
        self.s_idx = config["s_idx"]

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

        ## remove ids if its slice number is less than max_slice number
        if self.is_train:
            remove_ids = []
            for i, img_list in enumerate(self.img_lists):
                if len(img_list) < self.q_slice:
                    remove_ids.append(i)
            print(self.is_train, self.__class__, self.valid_img_n," # of remove ids : ", len(remove_ids))

            for id in reversed(remove_ids):
                self.img_paths.pop(id)
                self.label_paths.pop(id)
                self.img_lists.pop(id)
                self.valid_img_n -= 1

        ## count the test counts for validation
        else:
            self.q_cnts = []
            for img_list in self.img_lists:
                self.q_cnts.append(len(img_list)-self.q_slice+1)

            self.length = sum(self.q_cnts)

    def get_sample(self, s_img_paths_all, s_label_paths_all, q_img_paths, q_label_paths):
        seed = random.randrange(0,1000)
        # s_length = len(s_img_paths)

        s_imgs_all, s_labels_all = [],[]
        for s_idx, s_img_paths in enumerate(s_img_paths_all):
            s_label_paths = s_label_paths_all[s_idx]
            imgs, labels = [],[]

            for i in range(len(s_img_paths)):
                img_path, label_path = s_img_paths[i], s_label_paths[i]
                img = self.img_load(img_path, seed)
                img = resize(img, dsize=(self.size, self.size), interpolation=cv2.INTER_AREA)
                img = np.expand_dims(img, axis=0)
                imgs.append(img)
                label = np.load(label_path)
                label = resize(label, dsize=(self.size, self.size), interpolation=cv2.INTER_NEAREST)
                label = np.expand_dims(label, axis=0)
                labels.append(label)

            s_imgs = np.stack(imgs,axis=0)
            s_labels = np.stack(labels,axis=0)
            s_imgs_all.append(s_imgs)
            s_labels_all.append(s_labels)

        s_imgs = np.stack(s_imgs_all,axis=0)
        s_labels = np.stack(s_labels_all,axis=0)

        q_length = len(q_img_paths)
        imgs, labels = [],[]
        for i in range(len(q_img_paths)):
            img_path, label_path = q_img_paths[i], q_label_paths[i]
            img = self.img_load(img_path, seed)
            img = resize(img, dsize=(self.size, self.size), interpolation=cv2.INTER_AREA)
            img = np.expand_dims(img, axis=0)
            imgs.append(img)
            label = np.load(label_path)
            label = resize(label, dsize=(self.size, self.size), interpolation=cv2.INTER_NEAREST)
            label = np.expand_dims(label, axis=0)
            labels.append(label)

        q_imgs = np.stack(imgs,axis=0)
        q_labels = np.stack(labels,axis=0)

        # print(imgs.shape) [slice_num,1,256,256]?
        if self.is_train: ## random augmentation : flip, rotation
            s_imgs, s_labels, q_imgs, q_labels  = random_augment(s_imgs, s_labels, q_imgs, q_labels)

        sample = {
            "s_x":totensor(s_imgs),
            "s_y":totensor(s_labels), #.long()
            "q_x":totensor(q_imgs),
            "q_y":totensor(q_labels), #.long()
            # "s_length":s_length,
            # "q_length":q_length,
            "s_fname":s_img_paths_all,
            "q_fname":q_img_paths,
        }
        return sample

    def handle_idx(self, q_idxs, q_n, s_n):
        """
        choose slices for support indices
        :return: supp_idxs
        """
        s_idxs = []
        for q_idx in q_idxs:
            q_ratio = (q_idx)/(q_n-1)
            s_idx = round((s_n-1)*q_ratio)
            s_idxs.append(s_idx)

        return s_idxs

    def random_flip_z(self, q, s):
        if random.random() < 0.5:
            q.reverse()
            s.reverse()
        return q,s

    def getitem_train(self):
        ## choose support and target
        idx_space = [i for i in range(self.valid_img_n)]
        # print(idx_space, self.n_shot)
        subj_idxs = random.sample(idx_space, self.n_shot+1)
        s_subj_idxs = subj_idxs[:self.n_shot]
        q_subj_idx = subj_idxs[self.n_shot]
        q_subj_img_path = self.img_paths[q_subj_idx]
        q_subj_label_path = self.label_paths[q_subj_idx]
        q_fnames = self.img_lists[q_subj_idx]
        q_idx_start = random.randrange(0,len(q_fnames)-self.q_slice)
        q_idxs = [n for n in range(q_idx_start, q_idx_start+self.q_slice)]

        is_flip = False
        if random.random() < 0.5:
            is_flip = True
            q_fnames.reverse() # ??? when to reverse

        s_img_paths_all, s_label_paths_all = [],[]
        for s_subj_idx in s_subj_idxs:
            s_subj_img_path = self.img_paths[s_subj_idx]
            s_subj_label_path = self.label_paths[s_subj_idx]
            s_fnames = self.img_lists[s_subj_idx]

            ## flip augmentation
            if is_flip:
                s_fnames.reverse()

            ## choose support and query slice
            s_idxs = self.handle_idx(q_idxs, len(q_fnames), len(s_fnames))
            # s_idx = self.handle_idx(len(s_fnames), q_idx, len(q_fnames))
            s_fnames_selected = [s_fnames[idx] for idx in s_idxs]
            ## define path, load data, and return
            s_img_paths_selected = [f"{s_subj_img_path}/{fname}" for fname in s_fnames_selected]
            s_label_paths_selected = [f"{s_subj_label_path}/{fname}" for fname in s_fnames_selected]
            s_img_paths_all.append(s_img_paths_selected)
            s_label_paths_all.append(s_label_paths_selected)

        q_fnames_selected = [q_fnames[idx] for idx in q_idxs]
        q_img_paths_selected = [f"{q_subj_img_path}/{fname}" for fname in q_fnames_selected]
        q_label_paths_selected = [f"{q_subj_label_path}/{fname}" for fname in q_fnames_selected]
        return self.get_sample(s_img_paths_all, s_label_paths_all, q_img_paths_selected, q_label_paths_selected)

    def getitme_test(self, idx):
        q_subj_idx, q_idx_start = self.get_test_subj_idx(idx)
        q_subj_img_path = self.img_paths[q_subj_idx]
        q_subj_label_path = self.label_paths[q_subj_idx]
        q_fnames = self.img_lists[q_subj_idx]
        q_idxs = [n for n in range(q_idx_start, q_idx_start+self.q_slice)]

        s_img_paths_all, s_label_paths_all = [],[]
        for s_idx in range(self.n_shot):
            s_subj_img_path = self.s_img_paths[s_idx]
            s_subj_label_path = self.s_label_paths[s_idx]
            s_fnames = self.s_fnames_list[s_idx]
            ## choose support and query slice
            s_idxs = self.handle_idx(q_idxs, len(q_fnames), len(s_fnames))
            s_fnames_selected = [s_fnames[idx] for idx in s_idxs]
            ## define path, load data, and return
            s_img_paths_selected = [f"{s_subj_img_path}/{fname}" for fname in s_fnames_selected]
            s_label_paths_selected = [f"{s_subj_label_path}/{fname}" for fname in s_fnames_selected]
            s_img_paths_all.append(s_img_paths_selected)
            s_label_paths_all.append(s_label_paths_selected)

        q_fnames_selected = [q_fnames[idx] for idx in q_idxs]
        q_img_paths_selected = [f"{q_subj_img_path}/{fname}" for fname in q_fnames_selected]
        q_label_paths_selected = [f"{q_subj_label_path}/{fname}" for fname in q_fnames_selected]
        return self.get_sample(s_img_paths_all, s_label_paths_all, q_img_paths_selected, q_label_paths_selected)

    def get_len_train(self):
        return self.length

    def get_len_test(self):
        return self.length

    def get_val_subj_idx(self, idx):
        for subj_idx,cnt in enumerate(self.q_cnts):
            if idx < cnt:
                return subj_idx, idx*self.q_slice
            else:
                idx -= cnt

        print("get_val_subj_idx function is not working.")
        assert False

    def get_test_subj_idx(self, idx):
        # for subj_idx,cnt in enumerate(self.slice_cnts):
        for subj_idx,cnt in enumerate(self.q_cnts):
            if idx < cnt:
                return subj_idx, idx
            else:
                idx -= cnt

        print("get_test_subj_idx function is not working.")
        assert False

    def get_cnts(self):
        ## only for test loader
        return self.q_cnts
        # return self.slice_cnts

    def img_load(self, img_path, seed=0):
        img_arr = np.load(img_path)
        return img_arr

class BaseLoader(Base_dataset):
    modal_i = [0] # there is only one modality
    label_i = 1.0 # there is only one label for each image

class TrainLoader(BaseLoader):
    def __len__(self):
        return self.get_len_train()

    def __getitem__(self, idx):
        return self.getitem_train()

class TestLoader(BaseLoader):
    def __len__(self):
        return self.get_len_test()

    def __getitem__(self, idx):
        return self.getitme_test(idx)

    def set_support_volume(self, s_img_paths, s_label_paths):
        ## set support img path and label path for validation and testing
        self.s_img_paths = []
        self.s_label_paths = []
        self.s_fnames_list = []

        for i in range(len(s_img_paths)):
            s_fnames = os.listdir(s_img_paths[i])
            s_fnames = [int(e.split(".")[0]) for e in s_fnames]
            s_fnames.sort()
            print(f'support img {i} path : {s_img_paths[i]} length : {len(s_fnames)}')
            self.s_img_paths.append(s_img_paths[i])
            self.s_label_paths.append(s_label_paths[i])
            self.s_fnames_list.append([f"{e}.npy" for e in s_fnames])

if __name__ == "__main__":
    pass
    # main()