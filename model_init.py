# model_init.py
from Unet import SimpleUNet
import torch

# Check if there are multiple GPUs
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(SimpleUNet())
else:
    model = SimpleUNet()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
