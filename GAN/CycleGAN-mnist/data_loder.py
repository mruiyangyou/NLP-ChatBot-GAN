import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from Config import DefautConfig
config = DefautConfig()

def get_loader(config):
    transform_svhn = transforms.Compose([
        transforms.Scale(config.img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_mnist = transforms.Compose([
        transforms.Scale(config.img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])


    svhn = datasets.SVHN(config.svhn_path, transform = transform_svhn)
    mnist = datasets.MNIST(config.mnist_path, transform = transform_mnist)

    svhn_loader = DataLoader(dataset = svhn,
                             batch_size=config.batch_size,
                             shuffle=True,
                             num_workers=config.num_workers)

    mnist_loader = DataLoader(dataset=mnist,
                             batch_size=config.batch_size,
                             shuffle=True,
                             num_workers=config.num_workers)

    return svhn_loader, mnist_loader
