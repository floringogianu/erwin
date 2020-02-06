from torchvision import transforms as T
from torchvision import datasets as D


def get_dsets(root="./data/cifar-data"):
    """ Returns configured train and test datasets.
    """
    normalize = T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_transform = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize,
        ]
    )
    test_transform = T.Compose([T.ToTensor(), normalize])

    train_dset = D.CIFAR10(
        root=root, train=True, download=True, transform=train_transform
    )

    test_dset = D.CIFAR10(root=root, train=False, transform=test_transform)

    return train_dset, test_dset
