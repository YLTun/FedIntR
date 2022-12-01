import os
import copy
import collections
from glob import glob
from tqdm import tqdm

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

import utils


def get_config(dataset_name):
    """Dataset settings."""
    data_config = {
       'cifar_10':  {'img_size': 32, 'channels': 3, 'batch_size': 512, 'train_transform': get_transform('cifar_10', True), 'test_transform': get_transform('cifar_10')},
       'cifar_100':  {'img_size': 32, 'channels': 3, 'batch_size': 512, 'train_transform': get_transform('cifar_100', True), 'test_transform': get_transform('cifar_100')},
       'svhn':  {'img_size': 32, 'channels': 3, 'batch_size': 512, 'train_transform': get_transform('svhn', True), 'test_transform': get_transform('svhn')},
       'fashion_mnist': {'img_size': 32, 'channels': 3, 'batch_size': 512, 'train_transform': get_transform('fashion_mnist', True), 'test_transform': get_transform('fashion_mnist')},
       'mnist': {'img_size': 32, 'channels': 3, 'batch_size': 512, 'train_transform': get_transform('mnist', True), 'test_transform': get_transform('mnist')},
       'tiny_imagenet_200': {'img_size': 32, 'channels': 3, 'batch_size': 512, 'train_transform': get_transform('tiny_imagenet_200', train_set=True, img_size=32), 'test_transform': get_transform('tiny_imagenet_200', img_size=32)},
    }

    train_config = {
        'lr': 0.01,                         # [0.01, 0.005]
        'weight_decay': 0.00001, 
        'momentum' :0.9,
        'rounds': 100,
        'local_epochs': 10,                 # [1, 5, 10, 20, 40]
        'save_interval': 25,
        'optim': 'sgd',                     # ['sgd', 'adam']
    }

    return data_config[dataset_name], train_config

def get_config_moon():
    moon_config = {
        'temperature': 0.5,
    }
    return moon_config

def get_config_fedir():
    fedir_config = {
        'temperature': 0.5,
    }
    return fedir_config


def get_transform(dataset_name, train_set=False, img_size=32):
    """Given dataset name, we get the corresponding transform."""

    tf = list()
    if train_set and dataset_name != 'mnist' and dataset_name != 'svhn' and dataset_name != 'chars74k_fnt_num':
        tf.append(transforms.RandomHorizontalFlip())

    tf.extend([
            transforms.ToTensor(), 
            transforms.Resize(img_size),
            transforms.Normalize([0.5], [0.5]),
    ])
    return transforms.Compose(tf)


# This can handle empty samples in a folder.
class ImgDataset(Dataset):
    def __init__(self, parent_dir, label_idx_dict=None, transform=None):
        self.img_list = []
        self.label_list = []
        self.label_idx_dict = label_idx_dict
        
        sub_dirs = [f.name for f in os.scandir(parent_dir) if f.is_dir()]
        sub_dirs.sort()
        if self.label_idx_dict is None:
            self.label_idx_dict = {label:idx for idx, label in enumerate(sub_dirs)}

        self.classes = self.label_idx_dict.keys()  # To show what classes are in the dataset.
            
        for sub_dir in sub_dirs:
            full_path = os.path.join(parent_dir, sub_dir)
            file_extensions = ['*.JPG', '*.JPEG', '*.jpg', '*.png', '*.PNG']
            
            # img_paths = glob(os.path.join(full_path, '*.JPG')) + glob(os.path.join(full_path, '*.jpg'))
            
            img_paths = []
            for extension in file_extensions:
                img_paths.extend(glob(os.path.join(full_path, extension)))
            img_paths.sort()

            labels = [self.label_idx_dict[sub_dir]] * len(img_paths)
            self.img_list += img_paths
            self.label_list += labels
            
        self.transform = transform
        
    def __len__(self):
        return len(self.label_list)
    
    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        image = default_loader(img_path)
        label = self.label_list[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


@torch.no_grad()
def evaluate_model(model, data_loader, tqdm_desc=None,):
    device = next(model.parameters()).device

    loss_metric = utils.MeanMetric()
    acc_metric = utils.MeanMetric()

    loss_ce = torch.nn.CrossEntropyLoss()

    with utils.eval_mode(model):
        for (x, y) in tqdm(data_loader, desc=tqdm_desc):
            x = x.to(device)
            y = y.to(device)
            y_pred, _ = model(x)
            loss = loss_ce(y_pred, y)
            
            loss_metric.update_state(loss.item())
            acc_metric.update_state(utils.compute_accuracy(y, y_pred))

    return loss_metric.result(), acc_metric.result()

