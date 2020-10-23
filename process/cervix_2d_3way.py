import os
import sys
from glob import glob
import json
import numpy as np
sys.path.append("/home/soopil/Desktop/github/python_utils")
sys.path.append("../dataloaders_medical")
from PIL import Image
import SimpleITK as sitk
import numpy as np
import random
import shutil
import cv2
from cv2 import resize
import json
import pdb
from sklearn.model_selection import train_test_split

def metadata():
    info = {
    "src_dir" : "/user/home2/soopil/Datasets/MICCAI2015challenge/Cervix/RawData/Training",
    # "src_dir" : "/user/home2/soopil/Datasets/MICCAI2015challenge/Abdomen/RawData/Training_denoise",
    # "trg_dir" : "/user/home2/soopil/Datasets/MICCAI2015challenge/Cervix/RawData/Training_2d_3way",
    "trg_dir" : "/user/home2/soopil/Datasets/MICCAI2015challenge/Cervix/RawData/Training_2d_3way_re",
    }
    return info

def read_sitk(path):
    itk_img = sitk.ReadImage(path)
    # print(path, itk_img.GetSpacing())
    arr = sitk.GetArrayFromImage(itk_img)
    arr = np.array(arr, dtype=np.float32)
    return arr

def normalize(arr, type=0):
    # print(np.mean(arr*255.0), np.std(arr*255.0), np.amin(arr*255.0), np.amax(arr*255.0))
    if type == 0: # min and max
        mini = np.amin(arr)
        arr -= mini
        maxi = np.amax(arr)
        arr_norm = arr/maxi
    elif type == 1: # stddev and mean
        mean = np.mean(arr)
        stddev = np.std(arr)
        arr_norm = (arr-mean)/stddev
    elif type == 2: # CT configuration
        arr_norm = (arr+1024)/4096.0
        arr_norm = np.clip(arr_norm, 0, 1)
        print("normalize", np.amax(arr), np.amin(arr) ,"=>", np.amax(arr_norm), np.amin(arr_norm))
    elif type == 3: # CT configuration + mean, stddev alignment
        arr_norm = (arr+1024)/4096.0
        arr_norm = np.clip(arr_norm, 0, 1)
        arr_norm = normalize(arr_norm, type=1)
        print("normalize", np.amax(arr), np.amin(arr) ,"=>", np.amax(arr_norm), np.amin(arr_norm))
    return arr_norm

def check_OOI(shape, center, hsize):
    [x, y] = shape
    [cx, cy] = center
    [nx, ny] = center
    if x < cx + hsize:
        nx = x - hsize
    elif cx - hsize < 0:
        nx = hsize

    if y < cy + hsize:
        ny = y - hsize
    elif cy - hsize < 0:
        ny = hsize
    return [nx, ny]

def try_mkdirs(path):
    try:
        os.makedirs(path)
        return True
    except:
        return False

def save(dir_path, img_slice, label_slice, label, option, subj, pos, input_size):
    try_mkdirs(f"{dir_path}/{label+13}/{option}/img_orig/{subj}")
    try_mkdirs(f"{dir_path}/{label+13}/{option}/label_orig/{subj}")
    opath = f"{dir_path}/{label+13}/{option}/img_orig/{subj}/{pos}.npy"
    np.save(opath, img_slice)
    opath = f"{dir_path}/{label+13}/{option}/label_orig/{subj}/{pos}.npy"
    np.save(opath, label_slice)
    # print(img_slice.shape)
    img_slice = resize(img_slice, dsize=input_size, interpolation=cv2.INTER_AREA)
    # print(img_slice.shape)
    label_slice = resize(label_slice, dsize=input_size, interpolation=cv2.INTER_NEAREST)
    print(img_slice.shape, label_slice.shape, end="\r")
    try_mkdirs(f"{dir_path}/{label+13}/{option}/img/{subj}")
    try_mkdirs(f"{dir_path}/{label+13}/{option}/label/{subj}")
    opath = f"{dir_path}/{label+13}/{option}/img/{subj}/{pos}.npy"
    np.save(opath, img_slice)
    opath = f"{dir_path}/{label+13}/{option}/label/{subj}/{pos}.npy"
    np.save(opath, label_slice)


def meta_data():
    meta = metadata()
    print("start data preprocessing ...")
    print("source directory : ",meta['src_dir'])

    ## check label info
    subjs = os.listdir(f"{meta['src_dir']}/label")
    subjs = [e[5:] for e in subjs]
    # label_paths = glob(f"{meta['src_dir']}/label/*")
    # for i, label_path in enumerate(label_paths):
    #     label_arr = read_sitk(label_path)
    #     print(i, label_path, label_arr.shape, len(np.unique(label_arr)), np.unique(label_arr))

    ## split subjects into train, valid, test
    # print(subjs)
    # subj_n = len(subjs)
    # subjs_train_whole, subjs_test = train_test_split(subjs, test_size = 0.33, random_state = 0)
    # subjs_train, subjs_valid = train_test_split(subjs_train_whole, test_size = 0.25, random_state = 0)
    #
    # subj_split = {
    #     "train":subjs_train,
    #     "valid":subjs_valid,
    #     "test":subjs_test,
    # }
    # print(len(subjs), "=> ", len(subjs_train), len(subjs_valid), len(subjs_test))
    # with open("MICCAI2015_subj_split.json", "w") as json_file:
    #     json.dump(subj_split, json_file)

    with open("cervix_subj_split.json", "r") as json_file:
        subj_split = json.load(json_file)
    print(subj_split)
    subjs_train = subj_split["train"]
    subjs_valid = subj_split["valid"]
    subjs_test = subj_split["test"]

    ## preprocess
    def process(meta, subjs, option):
        ## original setting
        hsizes = [256, # 0 background
                  128 - 32, # 1 bladder
                  128 - 32, # 2 uterus
                  128 - 32, # 3 rectum
                  128 - 32, # 4 small bowel
                  ]
        out_size = (256,256)

        # margin = 20 # marginal pixels
        margins = [0, # 0 background
                   40, # 1 bladder
                   40, # 2 uterus
                   40, # 3 rectum
                   40, # 4 small bowel
                  ]
        margin_z = 5 # marginal pixels in z axis
        out_size = (256,256)

        for i, subj_fname in enumerate(subjs):
            subj = subj_fname.split(".")[0]
            img_path = f"{meta['src_dir']}/img/{subj_fname}-Image.nii.gz"
            label_path = f"{meta['src_dir']}/label/{subj_fname}-Mask.nii.gz"
            assert os.path.exists(img_path)
            assert os.path.exists(label_path)
            img_arr = read_sitk(img_path)
            img_arr = normalize(img_arr, type=2) #1
            label_arr = read_sitk(label_path)
            labels_unique = np.unique(label_arr).astype(dtype=np.int)
            labels_unique = np.delete(labels_unique, np.argwhere(labels_unique == 0))

            for label in labels_unique:
                hsize = hsizes[label]
                a_label_arr = (label_arr==label)*1.0
                whole_pos = np.array(np.nonzero(a_label_arr))
                [med_x, med_y, med_z] = np.mean(whole_pos, axis=1)
                med_x, med_y, med_z = int(med_x), int(med_y), int(med_z)
                img_arr_organ = img_arr[:, med_y - hsize:med_y + hsize, med_z - hsize:med_z + hsize]
                label_arr_organ = label_arr[:, med_y - hsize:med_y + hsize, med_z - hsize:med_z + hsize]

                ## [z,x,y] order
                minis = np.min(whole_pos, axis=1)
                maxis = np.max(whole_pos, axis=1)

                ## z - axial
                trg_dir = f"{meta['trg_dir']}_z"
                is_pass = False
                for pos in range(minis[0]-margin_z, maxis[0]+margin_z):
                    try:
                        img_slice_z = img_arr[pos, minis[1]-margin:maxis[1]+margin, minis[2]-margin: maxis[2]+margin]
                        label_slice_z = a_label_arr[pos, minis[1]-margin:maxis[1]+margin, minis[2]-margin:maxis[2]+margin]
                        save(trg_dir, img_slice_z, label_slice_z, label, option, subj, pos, (256,256))
                    except:
                        is_pass = True
                        pass

                if is_pass:
                    print("pass.")

                ## x - coronal?
                trg_dir = f"{meta['trg_dir']}_x"
                for pos in range(minis[1]-margin, maxis[1]+margin):
                    img_slice_x = img_arr[minis[0]-margin_z:maxis[0]+margin_z, pos, minis[2]-margin: maxis[2]+margin]
                    label_slice_x = a_label_arr[minis[0]-margin_z:maxis[0]+margin_z, pos, minis[2]-margin: maxis[2]+margin]
                    save(trg_dir, img_slice_x, label_slice_x, label, option, subj, pos, (256,128))

                ## y - sagittal?
                trg_dir = f"{meta['trg_dir']}_y"
                for pos in range(minis[2]-margin, maxis[2]+margin):
                    img_slice_y = img_arr[minis[0]-margin_z:maxis[0]+margin_z, minis[1]-margin: maxis[1]+margin, pos]
                    label_slice_y = a_label_arr[minis[0]-margin_z:maxis[0]+margin_z, minis[1]-margin: maxis[1]+margin, pos]
                    save(trg_dir, img_slice_y, label_slice_y, label, option, subj, pos, (256,128))

                # print(img_slice_z.shape, img_slice_x.shape, img_slice_y.shape)
                print(f"{i}/{len(subjs)} {subj} {label} {img_slice_z.shape} {img_slice_x.shape} {img_slice_y.shape}", end='\n')

        print("processing is done.")

    process(meta, subjs_train, option="train")
    process(meta, subjs_valid, option="valid")
    process(meta, subjs_test, option="test")

if __name__ == "__main__":
    meta_data()