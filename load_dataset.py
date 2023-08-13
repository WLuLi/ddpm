import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms

IMG_SIZE = 32  # CIFAR-10 images are originally 32x32

def load_transformed_dataset():
    data_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),  # Optional, since CIFAR-10 is already 32x32
        transforms.RandomHorizontalFlip(),  # randomly flip
        transforms.ToTensor(),  # transform it into a torch tensor -> [0, 1]
        transforms.Lambda(lambda t: (t*2) - 1)  # [-1, 1]
    ]
    data_transform = transforms.Compose(data_transforms)

    train = torchvision.datasets.CIFAR10(root=".", download=True, transform=data_transform, train=True)
    test = torchvision.datasets.CIFAR10(root=".", download=True, transform=data_transform, train=False)

    return torch.utils.data.ConcatDataset([train, test])

def show_tensor_image(image):
    """
    the reverse of the load function
    """
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t+1)/2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),
        transforms.Lambda(lambda t: t *255),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage()
    ])

    if len(image.shape) == 4:
        image = image[0, :, :, :]
    plt.imshow(reverse_transforms(image))

# # Example usage
# dataset = load_transformed_dataset()
# loader = DataLoader(dataset, batch_size=32, shuffle=True)
# images, labels = next(iter(loader))
# show_tensor_image(images[0])
