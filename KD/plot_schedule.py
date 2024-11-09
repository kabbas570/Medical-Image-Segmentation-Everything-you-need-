def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    

Max_Epochs = 500
def scheduler_cnn(epoch,initial_lr,final_lr):
    lr = initial_lr + (final_lr - initial_lr) * (epoch / Max_Epochs)
    lr = max(lr, final_lr)
    return lr

def scheduler_ViT(epoch, initial_lr, peak_lr, final_lr, warmup_epochs=30, total_epochs=500):
    if epoch < warmup_epochs:
        lr = initial_lr + (peak_lr - initial_lr) * (epoch / warmup_epochs)
    else:
        lr = peak_lr - (peak_lr - final_lr) * ((epoch - warmup_epochs) / (total_epochs - warmup_epochs))
    lr = max(lr, final_lr) if epoch >= warmup_epochs else max(lr, initial_lr)
    return lr

import torch
from torch import autograd, optim

x = autograd.Variable(torch.FloatTensor([1.0]), requires_grad=True)
optimizer = optim.SGD([x], lr=1)

values = []

for epoch in range(500):
    #cnn_learning_rate = scheduler_cnn(epoch,initial_lr=0.01,final_lr=0.0001)
    cnn_learning_rate = scheduler_ViT(epoch,initial_lr=0.00001,peak_lr=0.001,final_lr=0.00001)
    values.append(cnn_learning_rate)

import matplotlib.pyplot as plt

# Create the plot
plt.figure(figsize=(10, 6))  # Set the figure size

x = list(range(len(values)))


# Plot a continuous black line
plt.plot(x, values, color='black', linewidth=4)

# Customize the plot
plt.title('Linear-Decay Schedule')
plt.xlabel('Learning Rate')
plt.ylabel('Epoch')

# Add grid
plt.grid(True, linestyle='--', alpha=0.7)

# Adjust the plot layout
plt.tight_layout()

# Display the plot
plt.show()
