'''
Main Util functions
'''

# Built-in Imports
import os, time, io
import argparse
import zipfile
from typing import Any, Tuple, List

# Math and Visualization Imports
import numpy as np
import matplotlib.pyplot as plt
import cv2

# PyTorch Imports
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T


unnormalize = T.Compose([
    T.Normalize(
        mean = [ 0., 0., 0. ],
        std = [ 1/0.229, 1/0.224, 1/0.225 ]),
    T.Normalize(
        mean = [ -0.485, -0.456, -0.406 ],
        std = [ 1., 1., 1. ]),
])


class ZipDataset(Dataset):
    def __init__(self, root_path, cache_into_memory=False):
        if cache_into_memory:
            f = open(root_path, 'rb')
            self.zip_content = f.read()
            f.close()
            self.zip_file = zipfile.ZipFile(io.BytesIO(self.zip_content), 'r')
        else:
            self.zip_file = zipfile.ZipFile(root_path, 'r')
            
        self.name_list = list(filter(lambda x: x[-4:] == '.jpg', self.zip_file.namelist()))
        self.to_tensor = T.ToTensor()

    def __getitem__(self, key):
        buf = self.zip_file.read(name=self.name_list[key])
        img = cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_COLOR)
        img = self.to_tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return img

    def __len__(self):
        return len(self.name_list)


class AddNoise(object):
    def __init__(self, var=0.1, mean=0.):
        self.std = var**0.5
        self.mean = mean
  

    def __call__(self, x: Tensor) -> Tensor:
        x += (torch.randn(x.size())*self.std + self.mean)
        return torch.clamp(x, 0, 1)


class AutoDataset(Dataset):
    def __init__(self, zip_dataset, transforms=None):
        super(AutoDataset, self).__init__()
        self.dset = zip_dataset
        self.transforms = transforms
        self.preprocess = T.Compose([
            T.Resize((320,320)),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ])


    def __getitem__(self, idx):
        img = self.dset[idx]
        img_ = self.preprocess(img)
        img = self.preprocess(img)

        if self.transforms is not None:
            img_ = self.transforms(img)
        
        return img_, img


    def __len__(self):
        return len(self.dset)
    


def parse():
    parser = argparse.ArgumentParser()
    
    # Training parameters
    parser.add_argument('--train', action='store_true', help='Run training script')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate for training')
    parser.add_argument('--optim', type=str, default='adam', help='optimizer used during training')
    parser.add_argument('--scheduler', type=str, default='step', help='Scheduler for adaptive learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay to prevent weight explosion')
    parser.add_argument('--step_size', type=int, default=3, help='step size for step lr scheduler')
    parser.add_argument('--gamma', type=float, default=0.975, help='Decay for step LR scheduler')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of Epochs to train')
    parser.add_argument('--log_interval', type=int, default=5, help='Frequency of logging checkpoints')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training and validation')
    parser.add_argument('--num_batches', type=int, default=330, help='Total training batches for training and validation split as 90/10')
    
    # Model config
    parser.add_argument('--use_pretrained', action='store_true', help='Uses pretrained mobilenet backbone')
    parser.add_argument('--decoder_depth', type=str, default='light', help='Depth/complexity of the decoder. Light does not work well at all')
    parser.add_argument('--sparse', action='store_true', help='Flag to enable sparse loss i.e., Sparse Autoencoder')
    
    # Resume Training
    parser.add_argument('--resume', action='store_true', help='flag to resume training from checkpoint')
    parser.add_argument('--model_path', type=str, default='', help='Path to ckpt or pth file')

    # Evaluate Existing Model
    parser.add_argument('--eval_pth', action='store_true', help='Evaluate model from a .pth file')
    parser.add_argument('--eval_ckpt', action='store_true', help='Evaluate model from checkpoint')

    # General parameters
    parser.add_argument('--save_dir', type=str, default='results')
    parser.add_argument('--data_dir', type=str, default='data/', help='Root dir of data')
    parser.add_argument('--device', type=str, default='cpu', help='Device to train on')
    parser.add_argument('--visualize', action='store_true', help='flag to visualize some results')
    parser.add_argument('--test_noise', action='store_true', help='flag to enable additive noise when visuaizing results')
    parser.add_argument('--noise_var', type=float, default=0.05, help='Variance of Additive Gaussian Noise used during augmentation')
    parser.add_argument('--noise_mean', type=float, default=0.05, help='Mean of Additive Gaussian Noise used during augmentation')
    parser.add_argument('--test_noise_var', type=float, default=0.05, help='Variance of Additive Gaussian Noise when testing')
    parser.add_argument('--test_noise_mean', type=float, default=0.05, help='Mean of Additive Gaussian Noise when testing')
    parser.add_argument('--sparse_reg', type=float, default=1e-3, help='regularization for l1 sparsity')

    args = parser.parse_args()
    return args


def get_dataset(root: str) -> ZipDataset:
    if not os.path.exists(root):
        raise ValueError('Data root dir provided does not exist!')
    zip_dataset = ZipDataset(root, cache_into_memory=True)
    return zip_dataset


def get_loaders(args: Any) -> Tuple[DataLoader, DataLoader]:
    zip_dataset = get_dataset(args.data_dir)

    g_var = args.noise_var
    g_mean = args.noise_mean
    augment = T.Compose([
        AddNoise(g_var, g_mean)
    ])
    dataset = AutoDataset(zip_dataset, augment)

    val_batches = (args.num_batches // 11) * args.batch_size
    train_batches = (args.num_batches - (args.num_batches // 11) ) * args.batch_size    
    train_data, val_data, _ = torch.utils.data.random_split(
        dataset, [train_batches, val_batches, len(dataset) - args.batch_size*args.num_batches]
    )
    train_loader = DataLoader(train_data, args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, args.batch_size, num_workers=2)

    return train_loader, val_loader


def plot_losses(args: Any, train_losses: List, val_losses: List):
    plt.figure( figsize=(12,12) )
    n = len(train_losses) + 1
    plt.plot( list(range(1, n)), train_losses )
    plt.plot( list(range(1, n)), val_losses )
    plt.title('Loss vs Epcohs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Training', 'Validation'])

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    plt.savefig( os.path.join(args.save_dir, 'loss.png'), dpi='figure' )
    plt.show()


def visualize_samples(args: Any, model: nn.Module):
    zip_dataset = get_dataset(args.data_dir)
    augment = None
    if args.test_noise:
        augment = T.Compose([
            AddNoise(args.test_noise_var, args.test_noise_mean)
        ])
    dataset = AutoDataset(zip_dataset, augment)
    
    save_dir = args.save_dir
    if args.device == 'cuda':
        device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
    else:
        device = torch.device('cpu')
    
    fig = plt.figure( figsize=(15,15) )
    for i in range(15):
        ax = fig.add_subplot(5, 3, i+1)
        img = dataset[i][0]
        ax.imshow(img.permute(1,2,0))
        ax.axis('off')
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    plt.savefig( os.path.join(save_dir, 'chosen_samples.png'), dpi='figure' )
    
    model.eval()
    fig = plt.figure( figsize=(15,15) )
    
    with torch.no_grad():
        for i in range(15):
            ax = fig.add_subplot(5, 3, i+1)
            img = dataset[i][0].to(device)
            img_ = model(img.unsqueeze(0))
            img_ = img_.detach().cpu().squeeze()
            img_ = unnormalize(img_)
            ax.imshow(img_.permute(1,2,0))
            ax.axis('off')
    
    plt.savefig( os.path.join(save_dir, 'sample_results.png'), dpi='figure' )
    plt.show()

    test_img = dataset[0][0].to(device)
    with torch.no_grad():
        model.inspect_result(test_img.unsqueeze(0))
