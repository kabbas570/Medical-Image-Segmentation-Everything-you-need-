import torch.nn as nn
import torch
from mamba_ssm import Mamba
patch_size = 32

class Double_SSM_Block_Custom_Channel(nn.Module):
    def __init__(self, n_channels,out_channels):
        super(Double_SSM_Block_Custom_Channel, self).__init__()
        self.n_channels = n_channels
        self.out_channels = out_channels

        self.ssm1 = Mamba(
          d_model = self.n_channels,
          out_c = self.out_channels,
          d_state=16,  
          #d_conv=4,   
          expand=2,
      )
        
        self.ssm2 = Mamba(
          d_model = self.out_channels,
          out_c = self.out_channels,
          d_state=16,  
          #d_conv=4,   
          expand=2,
      )
        self.ln1 = nn.LayerNorm(normalized_shape=self.out_channels)
        self.ln2 = nn.LayerNorm(normalized_shape=self.out_channels)

    def forward(self, x1):
        x1 = self.ssm1(x1)
        x1 = self.ln1(x1)
        x1 = self.ssm2(x1)
        x1 = self.ln2(x1)
        return x1
    
class Linear_Layer(nn.Module):
    def __init__(self, n_channels,out_channels):
        super(Linear_Layer, self).__init__()
        self.n_channels = n_channels
        self.out_channels = out_channels
        self.linear1 = nn.Linear(self.n_channels,self.out_channels)
        self.ln1 = nn.LayerNorm(normalized_shape=self.out_channels)

    def forward(self, x1):  
        x1 = self.linear1(x1)
        x1 = self.ln1(x1)
        return x1
    
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            Double_SSM_Block_Custom_Channel(in_channels, out_channels),
        )
        self.res = Linear_Layer(in_channels, out_channels)

        
    def forward(self, x):
        x1 = self.res(x)
        x2 = self.double_conv(x)
        y = x1+x2
        return  y

class Conv_free_Emb(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.out_channel = out_channels        
        self.SSM = DoubleConv(in_channels,out_channels)
        
    def forward(self, x1):
        num_patches_h = x1.shape[2] // patch_size
        num_patches_w = x1.shape[3] // patch_size
        
        patches = x1.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.contiguous().view(x1.shape[0], num_patches_h, num_patches_w, x1.shape[1], patch_size, patch_size)
        patches = patches.permute(0,3,1,2,4,5)
        patches = torch.flatten(patches, start_dim=2,end_dim=3) # [2,1,16,64,64]
        b1,c1,n1,h1,w1 = patches.shape
        patches = torch.flatten(patches, start_dim=2,end_dim=4)
        patches = patches.permute(0,2,1)
        
        ### SSM Here ###
        ## after SSM ####
        
        patches = self.SSM(patches)
                
        patches_back = patches.permute(0,2,1)
        patches_back = patches_back.reshape(x1.shape[0],self.out_channel,n1,h1,w1) 
    
        return patches_back
    
class SSM_Block_Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.avg_pool = nn.AvgPool3d((1,2,2), stride=(1,2,2))        
        self.out_channel = out_channels        
        self.SSM = DoubleConv(in_channels,out_channels)
        
    def forward(self, patches):
        b1,c1,n1,h1,w1 = patches.shape
        patches = torch.flatten(patches, start_dim=2,end_dim=4)
        patches = patches.permute(0,2,1)
        patches = self.SSM(patches)
                
        ### SSM Here ###
        ## after SSM ####
        
        patches_back = patches.permute(0,2,1)
        patches_back = patches_back.reshape(b1,self.out_channel,n1,h1,w1) 
        patches_back = self.avg_pool(patches_back)
        return patches_back


class SSM_Block_Up(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.upsample = nn.Upsample(scale_factor=(1,2,2),mode='trilinear',align_corners=True)
        
        self.SSM = DoubleConv(in_channels+in_channels//2,in_channels//2)
        
    def forward(self, patches1,patches2):
        patches1 = self.upsample(patches1)
        patches = torch.cat([patches1, patches2], dim=1)
        
        b1,c1,n1,h1,w1 = patches.shape
        patches = torch.flatten(patches, start_dim=2,end_dim=4)
        patches = patches.permute(0,2,1)
        
        ### SSM Here ###
        ## after SSM ####
        
        patches = self.SSM(patches)
        
        patches = patches.permute(0,2,1)
        patches = patches.reshape(b1,self.in_channels//2,n1,h1,w1) 

        return patches
    
class SSM_Block_Last(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.in_channels = in_channels  
        self.out_channels = out_channels
        self.SSM = DoubleConv(self.in_channels,self.out_channels)
        
    def forward(self, patches):        
        b1,c1,n1,h1,w1 = patches.shape
        patches = torch.flatten(patches, start_dim=2,end_dim=4)
        patches = patches.permute(0,2,1)
        
        ### SSM Here ###
        ## after SSM ####
        
        patches = self.SSM(patches)
        
        patches = patches.permute(0,2,1)
        patches = patches.reshape(b1,self.out_channels,n1,h1,w1) 

        return patches
    
    
    
def reshape_to_original(patches_back):
    B, ch, num_patches_h, _, _ = patches_back.shape
    
    H, W = 256,256

    patches_back_reshaped = patches_back.view(B,ch, -1, patch_size, patch_size)
    
    # Calculate the number of patches in the height and width dimensions
    num_patches_w = W // patch_size
    num_patches_h = H // patch_size

    x1_reshaped = patches_back_reshaped.view(B,ch, num_patches_h, num_patches_w, patch_size, patch_size)
    x1_reshaped = x1_reshaped.permute(0,1,2,4,3,5).contiguous().view(B,4,256,256)
    
    return x1_reshaped

    
class UNet(nn.Module):
    def __init__(self, n_channels=1):
        super(UNet, self).__init__()
        self.n_channels = n_channels

        self.embd = Conv_free_Emb(1,64)
        self.down1 =  SSM_Block_Down(64,128)
        self.down2 =  SSM_Block_Down(128,256)
        self.down3 =  SSM_Block_Down(256,512)
        self.down4 =  SSM_Block_Down(512,1024)
        self.down5 =  SSM_Block_Down(1024,2048)
        
        
        self.up1 = SSM_Block_Up(2048)
        self.up2 = SSM_Block_Up(1024)
        self.up3 = SSM_Block_Up(512)
        self.up4 = SSM_Block_Up(256)
        self.up5 = SSM_Block_Up(128)
        
        self.out = SSM_Block_Last(64,4)
        
        
    def forward(self, inp):
        
        x0 = self.embd(inp)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
                
        y = self.up1(x5,x4)
        y = self.up2(y,x3)
        y = self.up3(y,x2)
        y = self.up4(y,x1)
        y = self.up5(y,x0)
        y = self.out(y)
        y = reshape_to_original(y)
        return y

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
def model() -> UNet:
    model = UNet()
    model.to(device=DEVICE,dtype=torch.float)
    return model
from torchsummary import summary
model = model()
summary(model, [(1, 256,256)])
