from forward import forward_diffusion_sample,  show_tensor_image, get_index_from_list, T, betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance
from Unet import SimpleUNet
from load_dataset import IMG_SIZE
import torch.nn.functional as functional
import matplotlib.pyplot as plt
import torch
from model_init import device, model

"""
loss -> L1 loss between noise and noise prediction, calculate noise by calling forward_diffusion_sample(image + timestep->new image and noise) every time
"""
def get_loss(model, x_0, t):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device) # forward
    noise_pred = model(x_noisy, t) # predict noise
    return functional.l1_loss(noise, noise_pred) # why l1 loss

@torch.no_grad()
def sample_timestep(x, t):
    """
    Calls the model to predict the noise in the image and returns 
    the denoised image. 
    Applies noise to this image, if we are not in the last step yet.
    """
    global betas
    global sqrt_one_minus_alphas_cumprod
    global sqrt_recip_alphas
    global posterior_variance

    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)
    
    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)
    
    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise 

@torch.no_grad()
def sample_plot_image():
    # Sample noise
    img_size = IMG_SIZE
    img = torch.randn((1, 3, img_size, img_size), device=device)
    plt.figure(figsize=(32,32))
    plt.axis('off')
    num_images = 10
    stepsize = int(T/num_images)

    for i in range(0,T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, t)
        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i/stepsize)+1)
            show_tensor_image(img.detach().cpu())
    plt.show()           