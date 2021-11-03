'''
calculate the PSNR and SSIM.
same as MATLAB's resu
'''
import os
import math
import numpy as np
import cv2
import glob
import imageio


def main():
    # Configurations

    # GT - Ground-truth;
    # Gen: Generated / Restored / Recovered images
    model = ['GSRV7']
    dataset = ['B100','Set14','Set5','Urban100']
    gt_basedir = '/home/yx/dataset/benchmark/'
    sr_basedir = '/home/yx/Documents/gyq/experiment0/'
    scale = 4
    crop_border = scale
    test_Y = True  # True: test Y channel only; False: test RGB channels
    # suffix = '_x2_SR'  # suffix for Gen images
    suffix = '_x'+str(scale)+'_SR'  # suffix for Gen images
    #suffix = 'x'+str(scale)
    all_mem = []
    for m in range(len(model)):
        get_aver = [[],[],[],[]]
        get_aver[0].append(model[m]+' PSNR:')
        get_aver[2].append(model[m]+' SSIM:')
        for n in range(len(dataset)):
            print('Begin test model:',model[m] ,'\t|Test Set:',dataset[n])
            folder_GT = gt_basedir + dataset[n] + '/HR'
            folder_Gen = sr_basedir + model[m] + '/results-'+ dataset[n]

            PSNR_all = []
            SSIM_all = []
            img_list = sorted(glob.glob(folder_GT + '/*'))

            if test_Y:
                print('Testing Y channel.')
            else:
                print('Testing RGB channels.')

            for i, img_path in enumerate(img_list):
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                im_GT = cv2.imread(img_path) / 255.
                im_Gen = cv2.imread(os.path.join(folder_Gen, base_name + suffix + '.png')) / 255.

                if test_Y and im_GT.shape[2] == 3:  # evaluate on Y channel in YCbCr color space
                    im_GT_in = bgr2ycbcr(im_GT)
                    im_Gen_in = bgr2ycbcr(im_Gen)
                else:
                    im_GT_in = im_GT
                    im_Gen_in = im_Gen
                wi = im_GT_in.shape[0]
                hi = im_GT_in.shape[1]

                # crop borders
                if im_GT_in.ndim == 3:
                    cropped_GT = im_GT_in[crop_border:wi-crop_border, crop_border:hi-crop_border, :]
                    cropped_Gen = im_Gen_in[crop_border:wi-crop_border, crop_border:hi-crop_border, :]
                    # cropped_GT = im_GT_in[crop_border:-crop_border, crop_border:-crop_border, :]
                    # cropped_Gen = im_Gen_in[crop_border:-crop_border, crop_border:-crop_border, :]
                elif im_GT_in.ndim == 2:
                    cropped_GT = im_GT_in[crop_border:wi-crop_border, crop_border:hi-crop_border]
                    cropped_Gen = im_Gen_in[crop_border:wi-crop_border, crop_border:hi-crop_border]
                else:
                    raise ValueError('Wrong image dimension: {}. Should be 2 or 3.'.format(im_GT_in.ndim))

                # imageio.imsave('gt/'+base_name+'.png',cropped_GT*255)
                # imageio.imsave('sr/'+base_name+'.png',cropped_Gen*255)
                # calculate PSNR and SSIM
                PSNR = calculate_psnr(cropped_GT * 255, cropped_Gen * 255)

                SSIM = calculate_ssim(cropped_GT * 255, cropped_Gen * 255)
                print('{:3d} - {:25}. \tPSNR: {:.6f} dB, \tSSIM: {:.6f}'.format(
                    i + 1, base_name, PSNR, SSIM))
                PSNR_all.append(PSNR)
                SSIM_all.append(SSIM)
            print('Average: PSNR: {:.6f} dB, SSIM: {:.6f}'.format(
                sum(PSNR_all) / len(PSNR_all),
                sum(SSIM_all) / len(SSIM_all)))
            
            get_aver[1].append(sum(PSNR_all) / len(PSNR_all))
            get_aver[3].append(sum(SSIM_all) / len(SSIM_all))
        print(model[m],'：\n',get_aver)
        all_mem.append(get_aver)
    
    print('all：\n',all_mem)
        


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def bgr2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


if __name__ == '__main__':
    main()
