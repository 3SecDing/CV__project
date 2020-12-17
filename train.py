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


# get train and test image ids
def get_image_ids(images_dir):
    train_image_names = glob.glob(images_dir + '0*.ARW')
    train_image_ids = [int(image_name.split('/')[-1].split('_')[0]) for image_name in train_image_names]

    test_image_names = glob.glob(images_dir + '1*.ARW')
    test_image_ids = [int(image_name.split('/')[-1].split('_')[0]) for image_name in test_image_names]

    return train_image_ids, test_image_ids

def pack_raw_images(raw_image):
    # pack Bayer image to 4 channels
    im = raw_image.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    packed_raw_image = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)

    return packed_raw_image

def loss_compute(pred_img, gt_img):
    return torch.abs(pred_img - gt_img).mean()

def adjust_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * 0.1

def save_results_and_checkpoint(results_dir, model_dir, pred_img, gt_patch, model, train_idx, ratio, epoch):
    pred = pred_img.permute(0, 2, 3, 1).cpu().data.numpy()
    pred = np.minimum(np.maximum(pred, 0), 1)

    out_img = np.concatenate((gt_patch[0, :, :, :], pred[0, :, :, :]), axis=1)
    Image.fromarray((out_img * 255).astype('uint8')).save(results_dir + f'{train_idx:05}_00_train_{ratio}.jpg')

    torch.save(model.state_dict(), os.path.join(model_dir, f'checkpoint_{epoch:04d}.pth'))
    print(f"model save as checkpoint_{epoch:04d}.pth!!!")

# def images_buffer(images_dir, image_ids):
#
#     for image_id in image_ids:
#         images = glob.blob(os.path.join(images_dir, f'{image_id}_00*.ARW'))
#
#         images = glob.blob(os.path.join(images_dir, f'{image_id}_00*.ARW'))



def train(cfg):
    device = torch.device(f'cuda:{cfg.GPU_ID}')
    print("device:", device)
    lr = cfg.TRAIN.LEARNING_RATE
    model_type = cfg.MODEL.BACKBONE
    upsample_type = cfg.MODEL.UPSAMPLE_TYPE
    epoch_max = cfg.TRAIN.EPOCH_MAX
    gt_images_dir = cfg.DATASET.GT_IMAGES_DIR
    train_images_dir = cfg.DATASET.TRAIN_IMAGES_DIR
    patch_size = cfg.TRAIN.PATCH_SIZE
    checkpoint_save_interval = cfg.TRAIN.SAVE_INTERVAL
    train_results_dir = cfg.TRAIN.RESULTS_DIR
    checkpoint_dir = cfg.TRAIN.CHECKPOINT_DIR
    lr_step = cfg.TRAIN.LR_STEP
    resume_model_weights = cfg.TRAIN.RESUME
    start_epoch = 0

    if not os.path.exists(train_results_dir):
        os.makedirs(train_results_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)


    if model_type == 'normal':
        if upsample_type == 'deconv':
            model = NormalModel()
        elif upsample_type == 'bilinear':
            model = NormalModel(upsample_type)
    elif model_type == 'light':
        if upsample_type == 'deconv':
            model = LightModel()
        elif upsample_type == 'bilinear':
            model = LightModel(upsample_type)
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
    if resume_model_weights != '':
        start_epoch = int(resume_model_weights.split('.')[0].split('_')[-1])
        model_weights = torch.load(resume_model_weights, map_location='cpu')
        model.load_state_dict(model_weights)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_gt_image_ids, test_gt_image_ids = get_image_ids(gt_images_dir)

    print("train length and test length:", len(train_gt_image_ids), len(test_gt_image_ids))

    gt_images = [None] * 6000
    train_images = {}
    train_images['300'] = [None] * len(train_gt_image_ids)
    train_images['250'] = [None] * len(train_gt_image_ids)
    train_images['100'] = [None] * len(train_gt_image_ids)

    g_loss = np.zeros((5000, 1))
    total_train_time = 0
    total_train_iter = 0


    for epoch in range(start_epoch, epoch_max + 1):
        iteration = 0
        for step in lr_step:
            if epoch == step:
                adjust_learning_rate(optimizer)

        epoch_start = time.time()
        for idx in np.random.permutation(len(train_gt_image_ids)):
            train_idx = train_gt_image_ids[idx]

            data_process_start = time.time()
            gt_image_names = glob.glob(os.path.join(gt_images_dir, f'{train_idx:05d}_00*.ARW'))
            gt_image_path = gt_image_names[0]
            gt_image_name = gt_image_path.split('/')[-1]

            train_image_names = glob.glob(os.path.join(train_images_dir, f'{train_idx:05d}_00*.ARW'))
            train_image_path = random.choice(train_image_names)
            train_image_name = train_image_path.split('/')[-1]

            train_exposure = float(train_image_name[9:-5])
            gt_exposure = float(gt_image_name[9:-5])
            ratio = min(gt_exposure / train_exposure, 300)



            if train_images[str(ratio)[0:3]][idx] is None:
                data_preprocess_start = time.time()
                train_raw_image = rawpy.imread(train_image_path)
                train_images[str(ratio)[0:3]][idx] = np.expand_dims(pack_raw_images(train_raw_image), axis=0) * ratio

                gt_raw_image = rawpy.imread(gt_image_path)
                im = gt_raw_image.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
                gt_images[idx] = np.expand_dims(np.float32(im / 65535.0), axis=0)
                print(f"data preprocess time: {time.time() - data_preprocess_start:.3f}")

            h = train_images[str(ratio)[0:3]][idx].shape[1]
            w = train_images[str(ratio)[0:3]][idx].shape[2]

            y = np.random.randint(0, h - patch_size)
            x = np.random.randint(0, w - patch_size)
            # print("h, w, x, y:", h, w, x, y)
            train_patch = train_images[str(ratio)[0:3]][idx][:, y:y + patch_size, x:x + patch_size, :]
            gt_patch = gt_images[idx][:, y * 2:y * 2 + patch_size * 2, x * 2:x * 2 + patch_size * 2, :]

            if np.random.randint(2, size=1)[0] == 1:  # random flip
                train_patch = np.flip(train_patch, axis=1)
                gt_patch = np.flip(gt_patch, axis=1)
            if np.random.randint(2, size=1)[0] == 1:
                train_patch = np.flip(train_patch, axis=2)
                gt_patch = np.flip(gt_patch, axis=2)
            if np.random.randint(2, size=1)[0] == 1:  # random transpose
                train_patch = np.transpose(train_patch, (0, 2, 1, 3))
                gt_patch = np.transpose(gt_patch, (0, 2, 1, 3))

            train_patch = np.minimum(train_patch, 1.0)
            gt_patch = np.maximum(gt_patch, 0.0)

            train_img = torch.from_numpy(train_patch).permute(0, 3, 1, 2).to(device)
            gt_img = torch.from_numpy(gt_patch).permute(0, 3, 1, 2).to(device)
            data_process_end = time.time()

            model.zero_grad()

            train_time_start = time.time()
            pred_img = model(train_img)

            loss = loss_compute(pred_img, gt_img)
            loss.backward()

            optimizer.step()
            train_time_end = time.time()

            g_loss[idx] = loss.data.cpu()

            mean_loss = np.mean(g_loss[np.where(g_loss)])
            iteration += 1
            total_train_iter += 1
            total_train_time += train_time_end - train_time_start

            print(f"epoch: {epoch}, iteration: {iteration}, loss:{mean_loss:.3}, "
                  f"iter time:{time.time() - data_process_start:.3}, "
                  f"data process time:{data_process_end - data_process_start:.3}, "
                  f"train iter time: {train_time_end - train_time_start:.3}")

            if epoch % checkpoint_save_interval == 0:

                epoch_result_dir = train_results_dir + f'{epoch:04}/'

                if not os.path.isdir(epoch_result_dir):
                    os.makedirs(epoch_result_dir)

                save_results_and_checkpoint(epoch_result_dir, checkpoint_dir, pred_img,
                                            gt_patch, model, train_idx, ratio, epoch)

        print(f"epoch time: {time.time() - epoch_start}, "
              f"mean train time iteration: {total_train_time / total_train_iter:.3f}")

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
    train(cfg)