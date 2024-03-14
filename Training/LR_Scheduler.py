def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    

import torch
from torch import autograd, optim

x = autograd.Variable(torch.FloatTensor([1.0]), requires_grad=True)
optimizer = optim.SGD([x], lr=1)

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,50,80,100], gamma=0.1)
for epoch in range(100):
    scheduler.step()
    optimizer.step()
    print(epoch, '    ',scheduler.get_last_lr(), '   ',get_lr(optimizer))
