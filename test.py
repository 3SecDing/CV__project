import os
import time
import glob
import rawpy
import torch
import random
import argparse

import numpy as np
from PIL import Image
from model_light import LightModel
from model_normal import NormalModel
from model_light_resiual import LightResiualModel
from model_light_resiual_reduce import LightResiualReduce
from config import _C as cfg

def pack_raw_images(raw_image):
    # pack Bayer image to 4 channels
    # im = raw_image.raw_image_visible.astype(np.float32)
    im = np.maximum(raw_image - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    packed_raw_image = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)

    return packed_raw_image


def test(cfg):
    device = torch.device('cpu')
    # device = torch.device(f'cuda:{cfg.GPU_ID}')
    model_type = cfg.MODEL.BACKBONE
    upsample_type = cfg.MODEL.UPSAMPLE_TYPE
    gt_images_dir = cfg.DATASET.GT_IMAGES_DIR
    images_dir = cfg.DATASET.TRAIN_IMAGES_DIR
    test_results_dir = cfg.TEST.RESULTS_DIR
    test_checkpoint = cfg.TEST.CHECKPOINT

    if not os.path.exists(test_results_dir):
        os.makedirs(test_results_dir)

    if model_type == 'normal':
        model = NormalModel()
    elif model_type == 'light':

        model = LightModel()
    elif model_type == 'light_resiual':
        if upsample_type == 'deconv':
            model = LightResiualModel()
        elif upsample_type == 'bilinear':
            model = LightResiualModel(upsample_type)
    elif model_type == 'light_resiual_reduce':
        if upsample_type == 'deconv':
            model = LightResiualReduce()
        elif upsample_type == 'bilinear':
            model = LightResiualReduce(upsample_type)
    else:
        print("Not supported model type!!!")
        return
    model_weights = torch.load(test_checkpoint, map_location='cpu')
    model.load_state_dict(model_weights)
    model.to(device)
    print("model:", model)

    test_image_names = glob.glob(gt_images_dir + '1*.ARW')
    test_image_ids = [int(image_name.split('/')[-1].split('_')[0]) for image_name in test_image_names]

    for test_image_id in test_image_ids:
        test_images = glob.glob(images_dir + f'{test_image_id:05d}_00*.ARW')

        for idx in range(len(test_images)):
            test_image_path = test_images[idx]
            test_image_name = test_image_path.split('/')[-1]
            print("image name:", test_image_name)

            gt_images = glob.glob(gt_images_dir + f'{test_image_id:05d}_00*.ARW')
            gt_image_path = gt_images[0]
            gt_image_name = gt_image_path.split('/')[-1]

            test_exposure = float(test_image_name[9:-5])
            gt_exposure = float(gt_image_name[9:-5])

            ratio = min(gt_exposure / test_exposure, 300)

            raw_image = rawpy.imread(test_image_path)
            test_image = raw_image.raw_image_visible.astype(np.float32)
            test_image_input_full = np.expand_dims(pack_raw_images(test_image), axis=0) * ratio
            test_image_input_full = np.minimum(test_image_input_full, 1.0)
            # h, w = test_image_input_full.shape[1], test_image_input_full.shape[2]
            # test_image_input_full = test_image_input_full[:, :512, 0:512, :]
            print('test image shape:', test_image_input_full.shape)

            test_image = raw_image.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            test_image_scale_full = np.expand_dims(np.float32(test_image / 65535.0), axis=0)

            gt_raw_image = rawpy.imread(gt_image_path)
            gt_image = gt_raw_image.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            gt_image_full = np.expand_dims(np.float32(gt_image / 65535.0), axis=0)

            in_img = torch.from_numpy(test_image_input_full).permute(0, 3, 1, 2).to(device)
            # total_time = 0
            # for i in range(10000):
            #     test_start = time.time()
            #     out_img = model(in_img)
            #     test_end = time.time()
            #     total_time += test_end - test_start
            test_start = time.time()
            out_img = model(in_img)
            test_end = time.time()

            output = out_img.permute(0, 2, 3, 1).cpu().data.numpy()
            output = np.minimum(np.maximum(output, 0), 1)

            output = output[0, :, :, :]
            gt_full = gt_image_full[0, :, :, :]
            scale_full = test_image_scale_full[0, :, :, :]
            origin_full = scale_full
            # scale the low-light image to the same mean of the groundtruth
            scale_full = scale_full * np.mean(gt_full) / np.mean(scale_full)


            Image.fromarray((origin_full * 255).astype('uint8')).save(
                test_results_dir + '%5d_00_%d_ori.png' % (test_image_id, ratio))
            Image.fromarray((output * 255).astype('uint8')).save(
                test_results_dir + '%5d_00_%d_out.png' % (test_image_id, ratio))# f'{test_image_id:05d}_00_{ratio}_out.png')
            Image.fromarray((scale_full * 255).astype('uint8')).save(
                test_results_dir + '%5d_00_%d_scale.png' % (test_image_id, ratio))
            Image.fromarray((gt_full * 255).astype('uint8')).save(test_results_dir + '%5d_00_%d_gt.png' % (test_image_id, ratio))

            print("test time per image:", test_end - test_start)#, ", ave_time:", total_time / 10000)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="See In The Dark!")
    parser.add_argument("--config", '-c', default="configs/sony_normal.yaml",
        metavar="FILE", help="path to config file",type=str,)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    test(cfg)
