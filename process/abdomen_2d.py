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
from sklearn.model_selection import train_test_split

def metadata():
    info = {
    "src_dir" : "/user/home2/soopil/Datasets/MICCAI2015challenge/Abdomen/RawData/Training",
    # "src_dir" : "/user/home2/soopil/Datasets/MICCAI2015challenge/Abdomen/RawData/Training_denoise",
    "trg_dir" : "/user/home2/soopil/Datasets/MICCAI2015challenge/Abdomen/RawData/Training_2d",
    "trg_dir2" : "/user/home2/soopil/Datasets/MICCAI2015challenge/Abdomen/RawData/Training_2d_2",
    "trg_dir3" : "/user/home2/soopil/Datasets/MICCAI2015challenge/Abdomen/RawData/Training_2d_denoise",
    "trg_dir4" : "/user/home2/soopil/Datasets/MICCAI2015challenge/Abdomen/RawData/Training_2d_nocrop",
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
    print(subjs)
    subj_n = len(subjs)
    subjs_train_whole, subjs_test = train_test_split(subjs, test_size = 0.33, random_state = 0)
    subjs_train, subjs_valid = train_test_split(subjs_train_whole, test_size = 0.25, random_state = 0)

    subj_split = {
        "train":subjs_train,
        "valid":subjs_valid,
        "test":subjs_test,
    }
    print(len(subjs), "=> ", len(subjs_train), len(subjs_valid), len(subjs_test))
    with open("MICCAI2015_subj_split.json", "w") as json_file:
        json.dump(subj_split, json_file)

    ## preprocess
    def process(meta, subjs, option):
        # hsize = 256
        hsizes = [256, # 0 background
                  128 - 32, # 1 spleen
                  128 - 32, # 2 right kidney
                  128 - 32, # 3 left kidney
                  128 - 64, # 4 gallbladder
                  128 - 64, # 5 esophagus
                  256 - 64, # 6 liver
                  128 + 32, # 7 stomach
                  128 - 32, # 8 aorta ?
                  128 - 32, # 9 inferior vana cava
                  128 - 0, # 10  por-tal vein & splenic vein,
                  128 + 32, # 11 pancreas
                  128 - 32, # 12 right adrenal gland
                  128 - 32, # 13 left adrenal gland
                  ]
        out_size = (256,256)

        for i, subj_fname in enumerate(subjs):
            subj = subj_fname.split(".")[0]
            img_path = f"{meta['src_dir']}/img/img{subj_fname}"
            label_path = f"{meta['src_dir']}/label/label{subj_fname}"
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
                cnt = np.sum(a_label_arr, axis=(1, 2))
                nonzero_pos = list(np.nonzero(cnt)[0])
                whole_pos = np.array(np.nonzero(a_label_arr))
                x, y, z = img_arr.shape
                [med_x, med_y, med_z] = np.mean(whole_pos, axis=1)
                med_x, med_y, med_z = int(med_x), int(med_y), int(med_z)
                is_crop = False

                if np.amax([y, z]) > hsize*2:
                    is_crop = True

                [med_y, med_z] = check_OOI([y,z], [med_y, med_z], hsize)

                for pos in nonzero_pos:

                    if is_crop:
                        img_slice = img_arr[pos, med_y-hsize:med_y+hsize, med_z-hsize:med_z+hsize]
                        label_slice = a_label_arr[pos, med_y-hsize:med_y+hsize, med_z-hsize:med_z+hsize]

                    else:
                        img_slice = img_arr[pos]
                        label_slice = a_label_arr[pos]

                    img_slice = resize(img_slice, dsize=out_size, interpolation=cv2.INTER_AREA)
                    label_slice = resize(label_slice, dsize=out_size, interpolation=cv2.INTER_NEAREST)
                    try_mkdirs(f"{meta['trg_dir4']}/{label}/{option}/img/{subj}")
                    try_mkdirs(f"{meta['trg_dir4']}/{label}/{option}/label/{subj}")
                    opath = f"{meta['trg_dir4']}/{label}/{option}/img/{subj}/{pos}.npy"
                    np.save(opath, img_slice)
                    opath = f"{meta['trg_dir4']}/{label}/{option}/label/{subj}/{pos}.npy"
                    np.save(opath, label_slice)

                print(f"label {label}/{len(labels_unique)} {is_crop}", end="\r")

            print(f"{i}/{len(subjs)} {subj} {img_arr.shape} {img_slice.shape} {is_crop} {[med_y, med_z]}", end='\n')

        print("processing is done.")

    process(meta, subjs_train, option="train")
    process(meta, subjs_valid, option="valid")
    process(meta, subjs_test, option="test")

if __name__ == "__main__":
    meta_data()