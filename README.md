# pytorch learning to see in the dark
Learning to See in the Dark Implement by PyTorch


### Original tensorflow version
Chen Chen, Qifeng Chen, Jia Xu, and Vladlen Koltun, "Learning to See in the Dark", in CVPR, 2018. <br/>
[Tensorflow code](https://github.com/cchen156/Learning-to-See-in-the-Dark) <br/>
[Paper](http://cchen156.web.engr.illinois.edu/paper/18CVPR_SID.pdf)


## Requirements
- 64 GB RAM 
- GPU memory above 3GB, here use GTX 1080ti, GPU memory 11GB
- PyTorch >= 0.4.0 (1.x are also OK)
- RawPy >= 0.10 

The program have been tested on Ubuntu 18.04.

## Download Dataset
Download the dataset following the instruction in the [original code](https://github.com/cchen156/Learning-to-See-in-the-Dark) and unzip it under the directory `dataset`.

```
pytorch_learning_to_see_in_the_dark
  ├── dataset
  │   ├── image-here  (not used, just a instructor)
  │   ├── Sony        (dataset Sony)
  │   │   ├── long   （gt images)
  |   │   ├── short   (train images)
  .   .   .
```
## Config
we use `yaml` type file to configurate the train and test process
```
pytorch_learning_to_see_in_the_dark
  ├── configs
  │   ├── sony_normal.yaml  (origin paper network architecture)
  │   ├── sony_light_resiual.yaml       (our designed network architecture)
  .   .   .
```

## Training
train origin network arch, change the yaml file to try different network arch:<br>
arch origin: `python train.py --config configs/sony_normal.yaml` <br>
arch 1: `python train.py --config configs/sony_light_resiual.yaml` <br>
arch 2: `python train.py --config configs/sony_light_resiual_bilinear.yaml` <br>
arch 3: `python train.py --config configs/sony_light_resiual_reduce_bilinear.yaml` <br>
arch 3 and epoch max=1000: `python train.py --config configs/sony_light_resiual_add.yaml` <br>
- The trained model is only for `.ARW` photos taken by Sony cameras, it only expriment on Sony dataset.
- It will save model and generate training result images every 100 epochs. 
- The trained models checkpoints will be saved in config item `TRAIN.RESULTS_DIR`.
- the result images will be saved in `TRAIN.RESULTS_DIR`.
- You can change any configuration of the model in `.yaml` file, such as `MODEL.BACKBONE, DATASET, LEARNING_RATE, EPOCH_MAX` etc. 
and you can change any kind of dirs of save results. 

## Testing
arch origin: `python test.py --config configs/sony_normal.yaml TEST.CHECKPOINT sony_normal_model_checkpoint/checkpoint_4000.pth` <br>
arch 1: `python test.py --config configs/sony_light_resiual.yaml TEST.CHECKPOINT sony_light_resiual_model_checkpoint/checkpoint_4000.pth` <br>
arch 2: `python test.py --config configs/sony_light_resiual_bilinear.yaml TEST.CHECKPOINT sony_light_resiual_lininear_model_checkpoint/checkpoint_4000.pth` <br>
arch 3: `python test.py --config configs/sony_light_resiual_reduce_bilinear.yaml TEST.CHECKPOINT sony_light_resiual_bilinear_reduce_model_checkpoint/checkpoint_4000.pth` <br>
arch 3 and epoch max=1000: `python test.py --config configs/sony_light_resiual_add.yaml TEST.CHECKPOINT sony_light_resiual_add_model_checkpoint/checkpoint_4000.pth` <br>


- This testing script is only for checking the performance of the trained model.
- This `.yaml` file is same with training `.yaml` file, and test any checkpoint you want by the config item `TEST.CHECKPOINT`， it accept the checkpoint path. 
- Here test the whole image on CPU, because of 11GB memory GPU not enough for the whole image as input.
- The result will be saved in `TEST.RESULTS_DIR` with `gt` as ground truth images, `scale` as scaled images, `ori` as input images, and `out` as output images.

## Compute PSNR and SSIM
for example, arch origin test results: `python computer_PSNR_SSIM.py --imgs_dir sony_normal_model_test_results`


### License
MIT License.


