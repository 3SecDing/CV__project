import os
import math
import argparse
import numpy as np
from PIL import Image
from skimage import io
from scipy.signal import convolve2d


def compute_psnr(img1, img2):
    if isinstance(img1,str):
        img1=io.imread(img1)
    if isinstance(img2,str):
        img2=io.imread(img2)
    mse = np.mean( (img1/255. - img2/255.) ** 2 )
    if mse < 1.0e-10:
       return 1000000000000
    PIXEL_MAX = 1
    psnr = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return mse, psnr


def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)

def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):
    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")

    M, N = im1.shape
    C1 = (k1*L)**2
    C2 = (k2*L)**2
    window = matlab_style_gauss2D(shape=(win_size,win_size), sigma=1.5)
    window = window/np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1*im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2*im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1*im2, window, 'valid') - mu1_mu2

    ssim_map = ((2*mu1_mu2+C1) * (2*sigmal2+C2)) / ((mu1_sq+mu2_sq+C1) * (sigma1_sq+sigma2_sq+C2))

    return np.mean(np.mean(ssim_map))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="See In The Dark!")
    parser.add_argument("--imgs_dir", '-i', default="", help="path to results images dir", type=str, )
    args = parser.parse_args()

    imgs_dir = args.imgs_dir
    images = os.listdir(imgs_dir)
    total_mean_mse = 0
    total_mean_psnr = 0
    total_mean_ssim = 0
    total_num = 0

    for image in images:
        if not 'out' in image:
            continue
        try:
            pred = np.asarray(Image.open(os.path.join(imgs_dir, image)))
            gt_image_name = image.replace('out', 'gt')
            gt = np.asarray(Image.open(os.path.join(imgs_dir, gt_image_name)))
        # pred = np.asarray(Image.open('sony_light_resiual_bilinear_reduce_model_test_results/10035_00_100_gt.png'))
        # gt = np.asarray(Image.open('sony_light_resiual_bilinear_reduce_model_test_results/10035_00_100_out.png'))
        # # if not:
        # img1 = np.asarray(Image.open('./img1.png').convert('L'))
        # img2 = np.asarray(Image.open('./img2.png').convert('L'))
            print("image pred and gt:", image, gt_image_name)
            mean_mse = 0
            mean_psnr = 0
            mean_ssim = 0
            for i in range(3):
                mse, psnr = compute_psnr(pred[:, :, i], gt[:, :, i])
                ssim = compute_ssim(pred[:, :, i], gt[:, :, i])
                mean_mse += mse
                mean_psnr += psnr
                mean_ssim += ssim

            print('mse = %.6f, psnr = %.6f, ssim = %.6f' % (mean_mse / 3, mean_psnr / 3, mean_ssim / 3))
            total_mean_psnr += mean_psnr
            total_mean_mse += mean_mse
            total_mean_ssim += mean_ssim
            total_num += 1
        except Exception as e:
            print("error and continue:", e)
            continue

    print(f"total mean mse={total_mean_mse / total_num / 3:.6f}\n"
          f"total mean psnr={total_mean_psnr / total_num / 3:.6f}\n"
          f"total mean ssim={total_mean_ssim / total_num / 3:.6f}\n")


