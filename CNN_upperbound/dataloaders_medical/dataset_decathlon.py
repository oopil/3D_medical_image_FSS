import os
import re
import sys
import json
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
    # tensor = F.interpolate(tensor, size=size,mode=interp)
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
        img = np.flip(img, 2).copy()
        label = np.flip(label, 2).copy()

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
        img_arr = np.load(img_path)
        return img_arr
class Spleen_Base(Base_dataset):
    modal_i = 0
    label_i = 1.0

class Spleen_train(Spleen_Base):
    def __len__(self):
        return self.get_len_train()

    def __getitem__(self, idx):
        return self.getitem_train()

class Spleen_test(Spleen_Base):
    def __len__(self):
        return self.get_len_test()

    def __getitem__(self, idx):
        return self.getitme_test(idx)

class Liver_Base(Base_dataset):
    modal_i = 0 # only 1 modality
    label_i = 1.0 # use both 1 : cancer / 2 : liver

class Liver_train(Liver_Base):
    def __len__(self):
        return self.get_len_train()

    def __getitem__(self, idx):
        return self.getitem_train()

class Liver_test(Liver_Base):
    def __len__(self):
        return self.get_len_test()

    def __getitem__(self, idx):
        return self.getitme_test(idx)

class Tumor_Base(Base_dataset):
    modal_i = [0, 1, 2, 3] # 4 modalities
    label_i = 3.0 # 1 : edema / 2 : non enhancing tumor / 3 : enhancing tumour

    def img_load(self, img_path, seed=0):
        modal_idx = seed%len(self.modal_i)
        img_arr = np.load(img_path)
        return img_arr[modal_idx] # synchronize with query img and other support img

class Tumor_train(Tumor_Base):
    def __len__(self):
        return self.get_len_train()

    def __getitem__(self, idx):
        return self.getitem_train()

class Tumor_test(Tumor_Base):
    def __len__(self):
        return self.get_len_test()

    def __getitem__(self, idx):
        return self.getitme_test(idx)

class Prostate_Base(Base_dataset):
    modality_n = 2
    modal_i = 0
    label_i = 2.0
    def img_load(self, img_path, seed=0):
        img_arr = np.load(img_path)
        return img_arr[self.modal_i]

class Prostate_train(Prostate_Base):
    def __len__(self):
        return self.get_len_train()

    def __getitem__(self, idx):
        return self.getitem_train()

class Prostate_test(Prostate_Base):
    def __len__(self):
        return self.get_len_test()

    def __getitem__(self, idx):
        return self.getitme_test(idx)

class Hippo_Base(Base_dataset):
    modal_i = 0
    label_i = 1.0 # use both 1.0 and 2.0

class Hippo_train(Hippo_Base):
    def __len__(self):
        return self.get_len_train()

    def __getitem__(self, idx):
        return self.getitem_train()

class Hippo_test(Hippo_Base):
    def __len__(self):
        return self.get_len_test()

    def __getitem__(self, idx):
        return self.getitme_test(idx)

class Lung_Base(Base_dataset):
    modal_i = 0 # only 1 modality
    label_i = 1.0 # use both 1 : cancer

class Lung_train(Lung_Base):
    def __len__(self):
        return self.get_len_train()

    def __getitem__(self, idx):
        return self.getitem_train()

class Lung_test(Lung_Base):
    def __len__(self):
        return self.get_len_test()

    def __getitem__(self, idx):
        return self.getitme_test(idx)

class HepaticVessel_Base(Base_dataset):
    modality_n = 1
    # modal_i = 0
    label_i = 1.0 # 1 for vessel, 2 for tumour
    # use only vessel

class HepaticVessel_train(HepaticVessel_Base):
    def __len__(self):
        return self.get_len_train()

    def __getitem__(self, idx):
        return self.getitem_train()

class HepaticVessel_test(HepaticVessel_Base):
    def __len__(self):
        return self.get_len_test()

    def __getitem__(self, idx):
        return self.getitme_test(idx)

class Heart_Base(Base_dataset):
    modality_n = 1
    # modal_i = 0
    label_i = 1.0 # 1 for left atrium

class Heart_train(Heart_Base):
    def __len__(self):
        return self.get_len_train()

    def __getitem__(self, idx):
        return self.getitem_train()

class Heart_test(Heart_Base):
    def __len__(self):
        return self.get_len_test()

    def __getitem__(self, idx):
        return self.getitme_test(idx)

class Pancreas_Base(Base_dataset):
    modality_n = 1 # only 1 modality
    # modal_i = 0
    label_i = 1.0 # 1 for pancreas, 2 for cancer
    # use all of them

class Pancreas_train(Pancreas_Base):
    def __len__(self):
        return self.get_len_train()

    def __getitem__(self, idx):
        return self.getitem_train()

class Pancreas_test(Pancreas_Base):
    def __len__(self):
        return self.get_len_test()

    def __getitem__(self, idx):
        return self.getitme_test(idx)

class Colon_Base(Base_dataset):
    modality_n = 1 # only 1 modality
    # modal_i = 0
    label_i = 1.0 # 1 for colon cancer primaries
    # use 1.0

class Colon_train(Colon_Base):
    def __len__(self):
        return self.get_len_train()

    def __getitem__(self, idx):
        return self.getitem_train()

class Colon_test(Colon_Base):
    def __len__(self):
        return self.get_len_test()

    def __getitem__(self, idx):
        return self.getitme_test(idx)

if __name__ == "__main__":
    pass
    # main()