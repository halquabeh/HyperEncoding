import os
import pickle

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST, MNIST, SVHN

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def calculate_norms(train_dataset, val_dataset):
    total_sum, total_sum1 = 0, 0
    total_count, total_count1 = 0, 0

    for example, _ in train_dataset:
        total_sum += example
        total_count += 1

    for example1, _ in val_dataset:
        total_sum1 += example1
        total_count1 += 1

    norm = total_sum / total_count
    norm1 = total_sum1 / total_count1

    return norm, norm1


def save_norms(norm, norm1, norm_file, norm1_file):
    with open(norm_file, 'wb') as f:
        pickle.dump(norm, f)
    with open(norm1_file, 'wb') as f:
        pickle.dump(norm1, f)


def load_norms(norm_file, norm1_file):
    with open(norm_file, 'rb') as f:
        norm = pickle.load(f)
    with open(norm1_file, 'rb') as f:
        norm1 = pickle.load(f)
    return norm, norm1


def norms_exist(norm_file, norm1_file):
    return os.path.exists(norm_file) and os.path.exists(norm1_file)


def get_norms(train_dataset, val_dataset, norm_file='norm.pkl', norm1_file='norm1.pkl'):
    if norms_exist(norm_file, norm1_file):
        norm, norm1 = load_norms(norm_file, norm1_file)
        print("Loaded norms from memory.")
    else:
        norm, norm1 = calculate_norms(train_dataset, val_dataset)
        save_norms(norm, norm1, norm_file, norm1_file)
        print("Calculated and saved norms.")
    return norm, norm1


def mnist(normalized=False):
    transform = transforms.Compose([transforms.ToTensor()])
    norm = ((0.1307,), (0.3081,))
    train_dataset = MNIST(root=path, train=True, download=True, transform=transform)
    val_dataset = MNIST(root=path, train=False, download=True, transform=transform)

    if normalized:
        norm, _ = get_norms(train_dataset, val_dataset, norm_file='norms/norm_mnist.pkl', norm1_file='norms/norm1_mnist.pkl')
        zeros_in_norm = torch.count_nonzero(norm == 0)
        print(f"Number of zeros in norm: {zeros_in_norm.item()}")
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(norm, (1,))])
        train_dataset = MNIST(root=path, train=True, download=True, transform=transform)
        val_dataset = MNIST(root=path, train=False, download=True, transform=transform)

    return train_dataset, val_dataset, norm


def fashion_mnist(normalized=False):
    transform_train = transforms.Compose([transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.ToTensor()])
    train_dataset = FashionMNIST(root=path, train=True, download=True, transform=transform_train)
    val_dataset = FashionMNIST(root=path, train=False, download=True, transform=transform_test)
    norm = ((0.1307,), (0.3081,))
    if normalized:
        norm, _ = get_norms(train_dataset, val_dataset, norm_file='norm_fashion_mnist.pkl', norm1_file='norm1_fashion_mnist.pkl')
        norm = norm.mean()
        transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize(norm, (1,))])
        transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(norm, (1,))])
        train_dataset = FashionMNIST(root=path, train=True, download=True, transform=transform_train)
        val_dataset = FashionMNIST(root=path, train=False, download=True, transform=transform_test)
    return train_dataset, val_dataset, norm


def cifar10(normalized=False):
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.ToTensor()])
    norm = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_dataset = CIFAR10(root=path, train=True, download=True, transform=transform_train)
    val_dataset = CIFAR10(root=path, train=False, download=True, transform=transform_test)

    if normalized:
        norm, _ = get_norms(train_dataset, val_dataset, norm_file='norms/norm_cifar10.pkl', norm1_file='norms/norm1_cifar10.pkl')
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(norm, (1,))])
        transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(norm, (1,))])
        train_dataset = CIFAR10(root=path, train=True, download=True, transform=transform_train)
        val_dataset = CIFAR10(root=path, train=False, download=True, transform=transform_test)

    return train_dataset, val_dataset, norm


def cifar100(normalized=False):
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.ToTensor()])
    norm = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_dataset = CIFAR100(root=path, train=True, download=True, transform=transform_train)
    val_dataset = CIFAR100(root=path, train=False, download=True, transform=transform_test)

    if normalized:
        norm, _ = get_norms(train_dataset, val_dataset, norm_file='norms/norm_cifar100.pkl', norm1_file='norms/norm1_cifar100.pkl')
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(norm, (1,))])
        transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(norm, (1,))])
        train_dataset = CIFAR100(root=path, train=True, download=True, transform=transform_train)
        val_dataset = CIFAR100(root=path, train=False, download=True, transform=transform_test)

    return train_dataset, val_dataset, norm


def svhn(normalized=False):
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.ToTensor()])
    norm = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_dataset = SVHN(root=path, split='train', download=True, transform=transform_train)
    val_dataset = SVHN(root=path, split='test', download=True, transform=transform_test)

    if normalized:
        norm, _ = get_norms(train_dataset, val_dataset, norm_file='norms/norm_SVHN.pkl', norm1_file='norms/norm1_SVHN.pkl')
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(norm, (1,))])
        transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(norm, (1,))])
        train_dataset = SVHN(root=path, split='train', download=True, transform=transform_train)
        val_dataset = SVHN(root=path, split='test', download=True, transform=transform_test)

    return train_dataset, val_dataset, norm


def _imagenet_folder_dataset(root_dir, norm_file_prefix, normalized=False):
    transform_train = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor()])
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    dataset_root = os.path.join(path, root_dir)
    train_dataset = torchvision.datasets.ImageFolder(os.path.join(dataset_root, 'train.X'), transform_train)
    val_dataset = torchvision.datasets.ImageFolder(os.path.join(dataset_root, 'val.X'), transform_test)

    if normalized:
        norm, _ = get_norms(
            train_dataset,
            val_dataset,
            norm_file=f'norms/norm_{norm_file_prefix}.pkl',
            norm1_file=f'norms/norm1_{norm_file_prefix}.pkl',
        )
        transform_train = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(norm, (1,))])
        transform_test = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(norm, (1,))])
        train_dataset = torchvision.datasets.ImageFolder(os.path.join(dataset_root, 'train.X'), transform_train)
        val_dataset = torchvision.datasets.ImageFolder(os.path.join(dataset_root, 'val.X'), transform_test)

    return train_dataset, val_dataset, norm


def imagenet100(normalized=False):
    return _imagenet_folder_dataset('Image100', 'imageNet100', normalized=normalized)


def imagenet1000(normalized=False):
    return _imagenet_folder_dataset('Image1000', 'imageNet1000', normalized=normalized)


if __name__ == "__main__":
    train_dataset, val_dataset, _ = svhn()

    pixel_sum = 0
    pixel_sum_sq = 0
    num_pixels = 0

    for data, _ in train_dataset:
        pixel_sum += torch.sum(data)
        pixel_sum_sq += torch.sum(data ** 2)
        num_pixels += data.numel()
    print(data.shape)
    mean = pixel_sum / num_pixels
    std = (pixel_sum_sq / num_pixels - mean ** 2) ** 0.5

    print(f"Mean of all pixels in training dataset: {mean}")
    print(f"Standard deviation of all pixels in training dataset: {std}")
    k1 = 0
    counter = 0
    for data, _ in train_dataset:
        x = data.reshape(-1)
        counter += 1
        k1 += (2 * x * (1 - x)).sum()

    print(f'total={k1}, counter={counter}, avg:{k1/counter}')

    train_dataset, val_dataset, _ = svhn(normalized=True)

    k2 = 0
    counter = 0
    for data, _ in train_dataset:
        x = data.reshape(-1)
        counter += 1
        k2 += (2 * torch.abs(x) * (1 - torch.abs(x))).sum()

    print(f'total={k2}, counter={counter}, avg:{k2/counter}')
    print(k1 / k2)
