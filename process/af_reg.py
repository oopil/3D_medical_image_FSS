import sys
import pdb
import os
import SimpleITK as sitk
import numpy as np
from skimage.io import imsave
from glob import glob
import psutil

p = psutil.Process(os.getpid())
# p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS) # for Windows
p.nice(19) # for Linux

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    return path


cwd = os.path.dirname(os.path.abspath(__file__))
main_name = 'Affine_#1'
data_path = cwd + '/data'

result_img_path = make_dir(cwd + '/result/' + main_name+'-imgs')
result_Tx_path = make_dir(cwd + '/result/' + main_name+'-Tx')


def command_iteration(method) :
    print("{0:3} = {1:10.5f}".format(method.GetOptimizerIteration(),
                                     method.GetMetricValue()))

#for Non-Rigid Body Registration
def B_SPLINE(img1, img2, loss='MSE'):

    fixed = sitk.GetImageFromArray(img2)

    moving = sitk.GetImageFromArray(img1)
    transformDomainMeshSize=[8]*moving.GetDimension()
    tx = sitk.BSplineTransformInitializer(fixed,
                                        transformDomainMeshSize,
                                        order=3)


    R = sitk.ImageRegistrationMethod()
    if loss == 'MSE': R.SetMetricAsMeanSquares()
    elif loss == 'NCC': R.SetMetricAsCorrelation()
    elif loss == 'MI': R.SetMetricAsMattesMutualInformation()
    R.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5,
                       numberOfIterations=500,
                       maximumNumberOfCorrections=5,
                       maximumNumberOfFunctionEvaluations=1000,
                       costFunctionConvergenceFactor=1e+7,
                    #    lowerBound=0.1, upperBound=100
    )


    R.SetInitialTransform(tx, True)
    R.SetInterpolator(sitk.sitkLinear)
    outTx = R.Execute(fixed, moving)

    print("-------")
    print("Optimizer stop condition: {0}".format(R.GetOptimizerStopConditionDescription()))
    print(" Iteration: {0}".format(R.GetOptimizerIteration()))
    print(" Metric value: {0}".format(R.GetMetricValue()))

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(outTx)
    
    return resampler, outTx

#for Rigid Body Registration
def AFFINE(img1, img2, loss='MSE'):

    fixed = sitk.GetImageFromArray(img2)
    moving = sitk.GetImageFromArray(img1)
    
    R = sitk.ImageRegistrationMethod()

    if loss == 'MSE': R.SetMetricAsMeanSquares()
    elif loss == 'NCC': R.SetMetricAsCorrelation()
    elif loss == 'MI': R.SetMetricAsMattesMutualInformation()

    R.SetOptimizerAsRegularStepGradientDescent(learningRate=1e-2,
                                           minStep=1e-4,
                                           numberOfIterations=1500,
                                           gradientMagnitudeTolerance=1e-8)
    R.SetInitialTransform(sitk.AffineTransform(fixed.GetDimension()))
    R.SetInterpolator(sitk.sitkLinear)
    outTx = R.Execute(fixed, moving)

    print("-------")
    print("Optimizer stop condition: {0}".format(R.GetOptimizerStopConditionDescription()))
    print(" Iteration: {0}".format(R.GetOptimizerIteration()))
    print(" Metric value: {0}".format(R.GetMetricValue()))

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(outTx)
    
    return resampler, outTx

def run_resampler(resampler, img):
    return np.array(sitk.GetArrayFromImage(resampler.Execute(sitk.GetImageFromArray(img.astype(float)))))

def rescale_0to1(img):
    img = img.astype(float)
    if np.max(img) > 2.0:
        img = img / 255.
    return img

def float2int(x):
    return (x*255).astype(np.uint8)

def read_data(p = data_path):
    src_dir = "/media/NAS/nas_187/soopil/data/MICCAI2015challenge/Abdomen/RawData/Training_2d_2_reg_train_v2"
    subjs = glob(f"{src_dir}/*")
    # subjs.sort()
    mov_set = [f"{e}/mov.npy" for e in subjs]
    fix_set = [f"{e}/fix.npy" for e in subjs]
    name_set = [e.split("/")[-1] for e in subjs]
    return name_set, mov_set, fix_set

if __name__ == '__main__':
    print('+--='*30)
    data_path = ""
    name_set, mov_set, fix_set = read_data(data_path)
    result_img_path = "/media/NAS/nas_187/soopil/data/MICCAI2015challenge/Abdomen/RawData/Training_2d_2_reg_train_nonrigid"
    result_Tx_path = "/media/NAS/nas_187/soopil/data/MICCAI2015challenge/Abdomen/RawData/Training_2d_2_reg_train_nonrigid_Tx"
    make_dir(result_img_path)
    make_dir(result_Tx_path)

    for i in range(len(name_set)):

        moving = mov_set[i]
        fixed = fix_set[i]
        moving = np.load(moving)
        fixed = np.load(fixed)

        moving = rescale_0to1(moving)
        fixed = rescale_0to1(fixed)

        #for Rigid Body Registration
        # resampler, outTx = AFFINE(moving.astype(float), fixed.astype(float), loss = 'MSE')# loss = MSE / MI / NCC

        #for Non-Rigid Body Registration
        resampler, outTx = B_SPLINE(moving.astype(float), fixed.astype(float), loss = 'MSE')# loss = MSE / MI / NCC

        warp = run_resampler(resampler, moving)
        warp = rescale_0to1(warp)
        mov_fix = np.clip(np.stack([moving, fixed, moving], axis=2), 0, 1)
        warp_fix = np.clip(np.stack([warp, fixed, warp], axis=2), 0, 1)

        imsave(result_img_path + '/' + name_set[i] + '_mov.png', float2int(moving))
        imsave(result_img_path + '/' + name_set[i] + '_fix.png', float2int(fixed))
        imsave(result_img_path + '/' + name_set[i] + '_mov_fix.png', float2int(mov_fix))
        imsave(result_img_path + '/' + name_set[i] + '_warp_fix.png', float2int(warp_fix))
        imsave(result_img_path + '/' + name_set[i] + '_warp.png', float2int(warp))
        sitk.WriteTransform(outTx, result_Tx_path + '/' + name_set[i]+'.txt')
        print(name_set[i])