import torch
import torchvision.transforms as transforms
from data.poison import PoisonDataset

# https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/trainer.py
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std= [0.229, 0.224, 0.225])

train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         transforms.RandomCrop(32, padding=4),
         transforms.ToTensor(),
         normalize])

test_transform = transforms.Compose(
    [transforms.ToTensor(),
     normalize])

def cifar10_loader(path, batch_size=128, train=True, oracle=False, augment=True, 
        poison=True, dataset=None):

    if dataset is None:
        transform = train_transform if train and augment else test_transform

        if path == "clean": poison = False
        if oracle and train: poison = False
        path = path if poison else None

        dataset = PoisonDataset(root='datasets', train=train, 
            transform=transform, download=True, poison_params=path)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=128, shuffle=train and augment,
        num_workers=2, pin_memory=True)
    return dataset, dataloader

