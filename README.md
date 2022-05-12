# DRANet (CVPR 2021)
### Pytorch implementation of paper:
### [Seunghun Lee, Sunghyun Cho, Sunghoon Im, "DRANet: Disentangling Representation and Adaptation Networks for Unsupervised Cross-Domain Adaptation", CVPR (2021)](https://arxiv.org/abs/2103.13447)
## Requirements
```
Pytorch 1.8.0
CUDA 11.1
python 3.8.10
numpy 1.21.0
scipy 1.7.1
tensorboardX
prettytable
```
## Data Preparation
Download [MNIST-M](https://github.com/fungtion/DANN), [Cityscapes](https://www.cityscapes-dataset.com/), [GTA5](https://download.visinf.tu-darmstadt.de/data/from_games/)
## Folder Structure of Datasets
```
├── data
      ├── MNIST
      ├── USPS
      ├── mnist_m
            ├── mnist_m_train
                      ├── *.png
            ├── mnist_m_test
                      ├── *.png
            ├── mnist_m_train_labels.txt
            ├── mnist_m_test_labels.txt
      ├── Cityscapes
            ├── GT
                   ├── train
                   ├── val
                   ├── test
            ├── Images
                   ├── train
                   ├── val
                   ├── test
      ├── GTA5
            ├── GT
                   ├── 01_labels
                   ├── 02_labels
                   ├── ...
            ├── Images
                   ├── 01_images
                   ├── 02_images
                   ├── ...
      
├── data_list
      ├── Cityscapes
              ├── train_imgs.txt
              ├── val_imgs.txt
              ├── train_labels.txt
              ├── val_labels.txt
      ├── GTA5
              ├── train_imgs.txt
              ├── train_labels.txt

```
## Train
You must input the task(clf or seg), datasets(M, MM, U, G, C), and experiment name.
```
python train.py -T [task] -D [datasets] --ex [experiment_name]
example) python train.py -T clf -D M MM --ex M2MM
```
## Test
Input the same experiment_name that you trained and specific iteration.
```
python test.py -T [task] -D [datasets] --ex [experiment_name (that you trained)] --load_step [specific iteration]
example) python test.py -T clf -D M MM --ex M2MM --load_step 100000
```
## Tensorboard
You can see all the results of each experiment on tensorboard.
```
CUDA_VISIBLE_DEVICES=-1 tensorboard --logdir tensorboard --bind_all
```
