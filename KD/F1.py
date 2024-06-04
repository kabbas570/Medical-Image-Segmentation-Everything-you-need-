import torch
import torch.nn as nn
import torch.optim as optim

class DynamicConvNet(nn.Module):
    def __init__(self, in_1 =None, out_1 = None, in_2=None):
        super(DynamicConvNet, self).__init__()
        self.in_1 = in_1
        self.in_2 = in_2
        self.out_1 = out_1

        self.conv = nn.Conv2d(in_1, out_1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv(x))
        return x

    def grow_filters(self, in_2):
        
        print(in_2)
        
        if self.out_1 is not None:

            new_conv = nn.Conv2d(in_2, in_2*2, kernel_size=3, padding=1)
            print(self.conv.weight.data[:in_2*2, :in_2, :, :].shape)
            print(new_conv.weight.data.shape)
            
            print(new_conv.bias.data[:self.in_1].shape)
            print(self.conv.bias.data[:self.in_1].shape)
                     
            new_conv.weight.data[:self.in_1, :, :, :] = self.conv.weight.data[:in_2*2, :in_2, :, :] 
            new_conv.bias.data[:self.in_1] = self.conv.bias.data[:self.in_1//2]
            self.conv = new_conv


# Initialize the model
model = DynamicConvNet(in_1 = 16, out_1 = 32, in_2=None)

# Example data
input_data = torch.randn(1, 16, 32, 32)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)


def create_indexed_dict(*value_pairs):
    return {i: list(pair) for i, pair in enumerate(value_pairs)}

# Example usage:
value_pairs = ([4], [8], [10])
my_dict = create_indexed_dict(*value_pairs)

# Training loop
for epoch in range(1):
    optimizer.zero_grad()
    output = model(input_data)
    loss = output.mean()  # Example loss
    loss.backward()
    optimizer.step()
    
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters:", num_parameters)
    
    # Grow filters after every epoch
    model.grow_filters(in_2=my_dict[epoch][0])
    
    #model = DynamicConvNet()
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters:", num_parameters)


# in_1 = 64 
# out_1 = 128


# a = []

# for epoch in range(1,5):
    
#     print(epoch*16,'   ',epoch*2*16)
    
    
    
# def create_indexed_dict(*value_pairs):
#     return {i: list(pair) for i, pair in enumerate(value_pairs)}

# # Example usage:
# value_pairs = ([5, 16], [8, 9], [7, 6])
# my_dict = create_indexed_dict(*value_pairs)

# # Accessing values using indices
# print(my_dict[0])  # Output: [5, 16]
# print(my_dict[1])  # Output: [8, 9]
# print(my_dict[2])  # Output: [7, 6]

    
# for epoch in range(3):
    
#     print(my_dict[epoch][1])
    
    
    
    
    
    
    
