"""
Dataset classes for common uses
"""
import random
import SimpleITK as sitk
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision.transforms.functional as tr_F
from skimage.exposure import equalize_hist
import pdb

def crop_resize(slice):
    x_size, y_size = np.shape(slice)
    slice = slice[40:x_size - 20, 50:y_size - 50]
    slice = resize(slice, (240, 240))
    return slice

def fill_empty_space(arr):
    arr[arr==0] = np.mean(arr)
    return arr

def prostate_sample(img_arr, label_arr, isize):
    img = Image.fromarray(img_arr.astype(np.uint8))
    label = Image.fromarray(label_arr.astype(np.uint8))
    sample = {
        'image':img,
        'label':label,
        'inst':label,
        'scribble':label,
    }
    # pdb.set_trace()
    sample = resize(sample, (isize,isize))
    sample = to_tensor_normalize(sample)
    return sample

def prostate_mask(sample, isize):
    # pdb.set_trace()
    label = sample['label']
    fg_mask = torch.where(label == 1, torch.ones_like(label), torch.zeros_like(label))
    bg_mask = torch.ones_like(label) - fg_mask
    fg_mask = fg_mask.expand((1, isize, isize))
    bg_mask = bg_mask.expand((1, isize, isize))
    return {'fg_mask': fg_mask,
            'bg_mask': bg_mask,
            }

def get_support_sample(ipath, lpath, modal_index, mask_n, is_HE, shift=0):
    arr = read_npy(ipath, modal_index, is_HE)
    # pdb.set_trace() ## for debugging
    # arr = fill_empty_space(arr)
    arr_mask = read_sitk(lpath)

    ## for 2-way(binary) segmentation
    arr_mask = (arr_mask>0)*1.0
    # arr_mask = (arr_mask == mask_n) * 1.0
    cnt = np.sum(arr_mask, axis=(1, 2))
    maxarg = np.argmax(cnt)
    slice = arr[maxarg+shift, :, :]
    slice = crop_resize(slice)
    # slice = normalize(slice)
    slice = convert3ch(slice)
    save_img(slice, "tmp_img.png")
    slice_mask = arr_mask[maxarg+shift, :, :]*255.0
    slice_mask = crop_resize(slice_mask)
    # slice_mask = convert3ch(slice_mask)
    save_img(slice_mask, "tmp_label.png")
    sample = read_sample("tmp_img.png", "tmp_label.png")
    # sample = transforms(sample)
    sample = to_tensor_normalize(sample)
    return sample

    ## for 5 way segmentation
    # arr_mask = (arr_mask == mask_n)*1.0
    # if mask_n == 2:
    #     arr_mask = (arr_mask > 0)*1.0
    # elif mask_n == 1:
    #     arr_mask = (arr_mask == 1)*1.0
    # elif mask_n == 4:
    #     arr_mask  = (arr_mask == 4)*1.0 + (arr_mask == 1)*1.0
    # else:
    #     raise("invalid mask_n")


def getMask(sample, class_id=1, class_ids=[0, 1]):
    label = sample['label']
    empty = sample['empty']
    fg_mask = torch.where(label == class_id, torch.ones_like(label), torch.zeros_like(label))
    brain_bg_mask = empty
    # bg_mask = torch.ones_like(label) - fg_mask - empty
    bg_mask = torch.ones_like(label) - fg_mask
    # brain_fg_mask = torch.ones_like(label) - empty
    brain_fg_mask = torch.ones_like(label) - empty - fg_mask
    fg_mask = fg_mask.expand((1, 240, 240))
    bg_mask = bg_mask.expand((1, 240, 240))
    brain_fg_mask = brain_fg_mask.expand((1, 240, 240))
    brain_bg_mask = brain_bg_mask.expand((1, 240, 240))
    return {'fg_mask': fg_mask,
            'bg_mask': bg_mask,
            'brain_fg_mask': brain_fg_mask,
            'brain_bg_mask': brain_bg_mask,}

def read_npy(path, modal_index, is_HE):
    arr = np.load(path)[modal_index]
    if is_HE:
        arr = equalize_hist(arr)
    arr = normalize(arr, type=0)
    return arr

def convert3ch(slice, axis=2):
    slice = np.expand_dims(slice, axis=axis)
    slice = np.concatenate([slice, slice, slice], axis=axis)
    return slice

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
	return arr_norm*255.0

def map_distribution(arr, tg_mean=0, tg_std=1, tg_min=0, tg_max=255):
    arr_nonzero = arr[np.nonzero(arr)]
    ## input arr range : (0,255)
    mean, std, mini, maxi = np.mean(arr_nonzero), np.std(arr_nonzero), np.amin(arr_nonzero), np.amax(arr_nonzero)
    Z = (arr - mean) / std ## map arr into standard var Z
    new_arr = Z*tg_std + tg_mean ## map Z into the target distribution
    print(np.mean(new_arr), np.std(new_arr), np.amin(new_arr), np.amax(new_arr))
    new_arr = np.clip(new_arr, tg_min, tg_max)
    return new_arr

def to_tensor_normalize(sample):
    img, label = sample['image'], sample['label']
    inst, scribble = sample['inst'], sample['scribble']

    ## map distribution
    # pdb.set_trace()
    # arr = np.array(img)
    # arr = map_distribution(arr, tg_mean=0.456, tg_std=0.224, tg_min=-10, tg_max=10)
    # img = Image.fromarray(arr.astype(dtype=np.uint8))

    img = tr_F.to_tensor(img)
    empty = (img[0] == 0.0) * 1.0
    img = tr_F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    label = torch.Tensor(np.array(label)).long()
    img = img.expand((1, 3, 240, 240))
    label = label/255.0
    sample['empty'] = empty.long()
    sample['image'] = img
    sample['label'] = label
    sample['inst'] = inst
    sample['scribble'] = scribble
    return sample

def read_sample(img_path, label_path):
    sample = {}
    sample['image'] = Image.open(img_path)
    sample['label'] = Image.open(label_path)
    sample['inst'] = Image.open(label_path)
    sample['scribble'] = Image.open(label_path)
    # Save the original image (without normalization)
    sample['original'] = Image.open(img_path)
    return sample

def read_sitk(path):
    itk_img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(itk_img)
    arr = np.array(arr, dtype=np.float32)
    return arr

def save_sitk(arr, itk_ref, opath):
  sitk_oimg = sitk.GetImageFromArray(arr)
  sitk_oimg.CopyInformation(itk_ref)
  sitk.WriteImage(sitk_oimg, opath)

def save_img(arr, path):
    im = Image.fromarray(arr.astype(np.uint8))
    im.save(path)

def load_img(path):
    arr = read_PIL(path)
    return to_tensor_normalize(arr)

def load_seg(path):
    arr = read_PIL(path)
    arr = np.expand_dims(arr, axis=0)
    arr = tr_F.to_tensor(arr)
    return arr

def read_PIL(path):
    im = Image.open(path)
    arr = np.array(im, dtype=np.float32)
    # arr = np.swapaxes(arr, 0, 2)
    return arr

def resize(sample, size):
    img, label = sample['image'], sample['label']
    img = tr_F.resize(img, size)
    label = tr_F.resize(label, size, interpolation=Image.NEAREST)
    sample['image'] = img
    sample['label'] = label
    return sample
