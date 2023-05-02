# Image Reconstruction Attacks in Distributed Deep Learning Systems using Autoencoders
Welcome to the GitHub repository for "Image Reconstruction Attacks in Distributed Deep Learning Systems using Autoencoders". 
## Abstract
Distributed Deep Learning Systems leverage IoT devices and Edge Computing to effectively process data and implement large Deep Learning models at scale. In this work, I present an 
Image Reconstruction Attack on Distributed Deep Learning Systems for Computer Vision tasks using Autoencoders. In particular, the proposed *black-box* attack targets models that use 
skip connections and complex model architectures, which makes reconstruction from intermediate results a non-trivial task. I am able to achieve **48.6dB** PSNR (best) and **0.24** 
MS-SSIM for my reconstructions. I discuss the advantages of incorporating upsampling in a neural network, and how it can improve the reconstruction quality and effectiveness of the attack. 
As a preventive measure, I posit that using a subset of feature maps during compute-offloading can be an effective mitigation technique at the expense of accuracy and changes in standard model architecture.

## Installation
Go ahead and clone the repo using the following command:
```
git clone https://github.com/piyushnags/reconstruction
```

Make sure to install the dependencies listed in the requirements file:
```
pip install -r requirements.txt
```

## Methods
An asymmetric autoencoder architecture is used to simulate the blackbox reconstruction attack. The design of the decoder is independent of the architecture of the vicitim model (encoder). There are twop proposed decoder architectures: decoder with *transposed convolutions* and decoder with *bicubic upsampling*.

## Quickstart
To train the autoencoder using transposed convolutions, run the following command:
```
python main.py --use_pretrained --decoder_depth medium --train --aug --lr 1e-3 --optim adam\
--num_epochs 25 --batch_size 512 --num_batches 99 --noise_var 0.02 --noise_mean 0.02 --device cuda\
--test_noise_var 0.02 --test_noise_mean 0.02\
--data_dir /path/to/celeba/dataset/
```

To train the autoencoder using bicubic upsampling, run the following command:
```
python main.py --use_pretrained --train --aug --lr 2.5e-4 --optim adam --interpolation upsample\
--num_epochs 25 --batch_size 128 --num_batches 198 --noise_var 0.02 --noise_mean 0.02 --device cuda\
--test_noise --test_noise_var 0.02 --test_noise_mean 0.02 --gamma 0.93\
--data_dir /path/to/celeba/dataset/
```

You can directly evaluate the performance of the model without training it if you have the .pth or .ckpt file using:
```
python main.py --use_pretrained --decoder_depth medium --eval_pth\
--model_path /path/to/model.pth/file/\
--aug --test_noise --test_noise_var 0.02 --test_noise_mean 0.02\
--device cuda --batch_size 128 --num_batches 198\
--data_dir /path/to/celeba/dataset/
```

You can generate the visualizations including training/validation curves, feature map and layer visualizations, hooks, sample results, etc. using:
```
python main.py --use_pretrained --interpolation upsample --visualize\
--model_path /path/to/model.pth/file/ --device cuda\
--test_noise --test_noise_var 0.02 --test_noise_mean 0.02\
--data_dir /path/to/celeba/dataset/
```

## Pre-trained Weights and Links to Dataset
- Pre-trained weights for Autoencoder using Transposed Convolutions: https://drive.google.com/file/d/1JTko5pZgUCWIQxNdQ5IfBnp5Rls719v9/view?usp=sharing
- Pre-trained weights for Autoencoder using Bicubic Upsampling: https://drive.google.com/file/d/1-LGC7rNGPteyLk82hsh8-6dHI60QOnkr/view?usp=sharing
- Download CelebA dataset from: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

## TODO
- [ ] Add support for Docker containers 
