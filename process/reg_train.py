import os
import sys
from glob import glob
import json
import numpy as np
from PIL import Image
import random
import shutil
import cv2
from cv2 import resize
import json
from sklearn.model_selection import train_test_split
import SimpleITK as sitk

def metadata():
    info = {
    "src_dir" : "/media/NAS/nas_187/soopil/data/MICCAI2015challenge/Abdomen/RawData/Training_2d_2_reg",
    # "src_dir" : "/user/home2/soopil/Datasets/MICCAI2015challenge/Abdomen/RawData/Training_denoise",
    "trg_dir" : "/media/NAS/nas_187/soopil/data/MICCAI2015challenge/Abdomen/RawData/Training_2d_2_reg_train",
    }
    return info

def read_sitk(path):
    itk_img = sitk.ReadImage(path)
    # print(path, itk_img.GetSpacing())
    arr = sitk.GetArrayFromImage(itk_img)
    arr = np.array(arr, dtype=np.float32)
    return arr

def try_mkdirs(path):
    try:
        os.makedirs(path)
        return True
    except:
        return False

def handle_idx(q_idx, q_n, s_n):
    """
    choose slices for support indices
    :return: supp_idxs
    """
    q_ratio = (q_idx)/(q_n-1)
    s_idx = round((s_n-1)*q_ratio)
    return s_idx

def meta_data():
    meta = metadata()
    print("start data preprocessing ...")
    print("source directory : ",meta['src_dir'])

    ## preprocess
    n_sample_per_organ = 50
    src_dir = meta['src_dir']
    organs = [1,3,6,14]
    organ = organs[0]
    subjs = glob(f"{src_dir}/{organ}/train/img/*")
    idx_space = [i for i in range(len(subjs))]
    cnt = 0
    for organ in organs:
        for idx in range(n_sample_per_organ):
            mov_subj_idx, fix_subj_idx = random.sample(idx_space, 2)
            # print(mov_subj_idx, fix_subj_idx)
            mov_subj_dir = subjs[mov_subj_idx]
            fix_subj_dir = subjs[fix_subj_idx]
            mov_files = os.listdir(mov_subj_dir)
            mov_files.sort()
            fix_files = os.listdir(fix_subj_dir)
            fix_files.sort()
            mov_file = random.sample(mov_files, 1)[0]
            mov_file_idx = mov_files.index(mov_file)
            fix_file_idx = handle_idx(mov_file_idx, len(mov_files), len(fix_files))
            fix_file = fix_files[fix_file_idx]

            # print(mov_files)
            # print(fix_files)
            # print(mov_file, len(mov_files), len(fix_files), mov_file_idx, fix_file_idx)

            mov_file_path = f"{mov_subj_dir}/{mov_file}"
            fix_file_path = f"{fix_subj_dir}/{fix_file}"
            # print(mov_file_path)
            # print(fix_file_path)

            mov_split = mov_file_path.split("/")
            fix_split = fix_file_path.split("/")
            mov_split[-6] = "Training_2d_2_reg_train"
            trg_dir = "/".join(mov_split[:-5])
            # print(trg_dir)
            mov_out = f"{trg_dir}/{cnt}/mov.npy"
            fix_out = f"{trg_dir}/{cnt}/fix.npy"
            try_mkdirs(f"{trg_dir}/{cnt}")
            shutil.copyfile(mov_file_path, mov_out)
            shutil.copyfile(fix_file_path, fix_out)
            cnt += 1
            print(f"organ : {organ},  {idx}/{n_sample_per_organ}", end='\r')
        print()

if __name__ == "__main__":
    meta_data()
