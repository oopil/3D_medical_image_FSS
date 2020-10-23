import os
import sys
from glob import glob
import json
import numpy as np
sys.path.append("/home/soopil/Desktop/github/python_utils")
sys.path.append("../dataloaders_medical")
import SimpleITK as sitk
import numpy as np
from skimage.segmentation import slic

def read_sitk(path):
    itk_img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(itk_img)
    arr = np.array(arr, dtype=np.float64)
    return arr, itk_img

def try_mkdirs(path):
    try:
        os.makedirs(path)
        return True
    except:
        return False

def save_sitk(arr, itk_ref, opath):
    sitk_oimg = sitk.GetImageFromArray(arr)
    sitk_oimg.CopyInformation(itk_ref)
    sitk.WriteImage(sitk_oimg, opath)

def main():
    # src_path = '/home/soopil/Desktop/Dataset/MICCAI2015challenge/Abdomen/RawData/Training/label'
    src_path = '/home/soopil/Desktop/Dataset/MICCAI2015challenge/Cervix/RawData/Training/label'
    # src_path = '/home/soopil/Desktop/Dataset/CT_ORG/Training/label'
    src_parts = src_path.split('/')
    trg_path = '/'.join(src_parts[:-2])
    trg_path = os.path.join(trg_path, 'separate_label')
    try_mkdirs(trg_path)
    print(f"src_path : {src_path}")
    print(f"trg_path : {trg_path}")
    fnames = os.listdir(src_path)
    fnames.sort()

    for fname in reversed(fnames):
        print(fname)
        fpath = os.path.join(src_path, fname)
        fn = fname.split(".")[0]
        arr_orig, itk = read_sitk(fpath) ## dtype = np.int64
        labels = list(np.unique(arr_orig))
        labels.remove(0)
        for label in labels:
            fname = fn + f"_{str(int(label))}"+".nii.gz"
            opath = os.path.join(trg_path, fname)
            arr = (arr_orig==label)*(label+13)*1.0
            print(label, type(label), np.unique(arr))
            save_sitk(arr, itk, opath)


if __name__ == "__main__":
    main()