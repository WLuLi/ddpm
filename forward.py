import torch
from torch.utils.data import DataLoader
# torchvision is a package that provides access to popular datasets, model architectures, and image transformations for computer vision
import torchvision
from torchvision import transforms
# matplotlib is a plotting library to visualize images for python and numpy
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as functional
import pylab

# const define
IMG_SIZE = 64
BATCH_SIZE = 4
T = 100

def linear_beta_schedule(timesteps, start=0.0001,end=0.02):
    # linspace -> returns evenly spaced numbers over a specified interval
    return torch.linspace(start, end, timesteps);       

def get_index_from_list(vals, tensor, x_shape):
    """
    Returns the index in the list vals
    """
    # shape -> returns the shape of a tensor(demension)
    batch_size = tensor.shape[0]
    out = vals.gather(-1, tensor.cpu())
    return out.reshape(batch_size, *((1,)*(len(x_shape)-1))).to(tensor.device)

def forward_diffusion_sample(x_0, tensor, device="cpu"):
    """
    Input an image and a timestep, and return the noisy version
    """
    #forward
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, tensor, x_0.shape)
    srqtOneMinusalphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, tensor, x_0.shape)

    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) + srqtOneMinusalphas_cumprod_t.to(device) * noise.to(device), noise.to(device)

def load_transformed_dataset():
    data_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(), # randomly flip and rotate
        transforms.ToTensor(), # transform it into a torch tensor -> [0, 1]
        transforms.Lambda(lambda t: (t*2) - 1) # [-1, 1]
    ]
    data_transform = transforms.Compose(data_transforms)

    train = torchvision.datasets.StanfordCars(root=".", download=True, transform=data_transform)
    test = torchvision.datasets.StanfordCars(root=".", download=True, transform=data_transform, split='test')

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


"""
initialize parameters (beta, alpha, etc.)
"""

"""
pre-calculate values used in the forward pass
"""
# assign a T value -> timesteps
betas = linear_beta_schedule(timesteps=T)

alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
alphas_cumprod_prev = functional.pad(alphas_cumprod[:-1], (1,0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0/alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
posterior_variance = betas*(1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

data = load_transformed_dataset()
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
image = next(iter(dataloader))[0]

plt.figure(figsize=(15, 15))
num_images = 10
stepsize = T // num_images
for i in range(0, T, stepsize):
    t = torch.Tensor([i]).long()
    plt.subplot(1, num_images+1, int(i/stepsize)+1)
    image, noise = forward_diffusion_sample(image, t)
    show_tensor_image(image)

pylab.show()