from torch.optim import Adam
import torch
from lossSampling import get_loss, model, sample_plot_image
from forward import T, BATCH_SIZE, dataloader

device = "cuda"
model.to(device)
optimizer = Adam(model.parameters(), lr=0.001)
epochs = 5 # Try more!


for epoch in range(epochs):
    for step, batch in enumerate(dataloader):
      optimizer.zero_grad()

      t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
      loss = get_loss(model, batch[0], t)
      loss.backward()
      optimizer.step()

      # if epoch % 2 == 0 and step == 0:
      if True:
        print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
      if step == 0:
        sample_plot_image()