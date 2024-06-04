import torch
import torch.nn as nn

# Define different levels of sparsity
sparseness_list = [32, 64, 128, 256, 512, 1024]

# Set temperature for Gumbel-Softmax
temperature = 0.8

# Initialize thetas as learnable parameters
thetas = nn.Parameter(torch.Tensor([1.0 / len(sparseness_list) for _ in sparseness_list]))

print("Initial Thetas:")
print(thetas)

# Apply Gumbel-Softmax to thetas
soft_mask_variables = nn.functional.gumbel_softmax(thetas, temperature)
sparseness  = sum(m * sp for m, sp in zip(soft_mask_variables, sparseness_list))
print(sparseness)
print("Soft Mask Variables (Soft Masks):")
print(soft_mask_variables)
