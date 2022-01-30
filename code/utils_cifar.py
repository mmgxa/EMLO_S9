import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np



def _get_transform():
    return transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
    

def get_train_data_loader():
    transform = _get_transform()
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform)
    return torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

    
def get_test_data_loader():
    transform = _get_transform()
    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=transform)
    return torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)
    
# function to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

classes = tuple(torchvision.datasets.CIFAR100(root='./data', train=True, download=True).class_to_idx)
