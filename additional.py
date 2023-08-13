# show images
# data
import matplotlib.pyplot as plt
import torch
import torchvision
import pylab

# dataset
def show_images(dataset, num_samples=20, cols=4):
    """Plots some images from a dataset"""
    # plt.figure -> create a figure
    plt.figure(figsize=(15, 15))
    # enumerate -> returns an enumerate object. It contains the index and value of all the items in the object as tuples
    for i, img in enumerate(dataset):
        if i == num_samples:
            break
        # plt.subplot -> add a subplot to the current figure
        plt.subplot(num_samples // cols + 1, cols, i + 1)
        # plt.imshow -> display data as an image
        plt.imshow(img[0])
