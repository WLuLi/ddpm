from torch.optim import Adam
import torch
import matplotlib.pyplot as plt
from lossSampling import get_loss, model, sample_plot_image
from forward import T, BATCH_SIZE, dataloader
from model_init import device, model

optimizer = Adam(model.parameters(), lr=0.001)
epochs = 51 # Try more!

# Check if saved model and losses exist, if so, load them
try:
    model.load_state_dict(torch.load("trained_model/model_epoch.pth"))
    losses = torch.load("trained_model/losses_epoch.pth")
except:
    losses = []

SAVE_EPOCHS = [10, 20, 30, 40, 50, 60, 70, 80, 90]  # List of epochs at which to save the model

for epoch in range(epochs):
    epoch_loss = 0.0  # To accumulate loss over the epoch
    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()

        t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
        loss = get_loss(model, batch[0], t)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")

    # Average the loss over the epoch and append to losses list
    losses.append(epoch_loss / len(dataloader))

    # Save the model and losses at specific epochs
    if epoch in SAVE_EPOCHS:
        torch.save(model.state_dict(), "trained_model/model_epoch.pth")
        torch.save(losses, "trained_model/losses_epoch.pth")

        sample_plot_image()
        
        # Plot the loss curve
        plt.figure(figsize=(10, 5))
        plt.plot(losses, label='Training Loss')
        plt.title('Loss over time')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
