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
	return arr_norm

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
    src_path = '/home/soopil/Desktop/Dataset/MICCAI2015challenge/Abdomen/RawData/Training/label'
    # src_path = '/home/soopil/Desktop/Dataset/CT_ORG/Training/label'
    src_parts = src_path.split('/')
    trg_path = '/'.join(src_parts[:-2])
    trg_path = os.path.join(trg_path, 'kidney_only')
    try_mkdirs(trg_path)
    print(f"src_path : {src_path}")
    print(f"trg_path : {trg_path}")
    fnames = os.listdir(src_path)
    fnames.sort()

    for fname in fnames:
        print(fname)
        fpath = os.path.join(src_path, fname)
        opath = os.path.join(trg_path, fname)
        arr, itk = read_sitk(fpath) ## dtype = np.int64
        arr = (arr == 3) *1.0
        save_sitk(arr, itk, opath)

if __name__ == "__main__":
    main()