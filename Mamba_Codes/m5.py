import torch 
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from mamba_ssm import Mamba

class Double_SSM_Block_Encoder(nn.Module):
    def __init__(self, n_channels):
        super(Double_SSM_Block_Encoder, self).__init__()
        self.n_channels = n_channels
       
        self.ln1 = nn.LayerNorm(normalized_shape=n_channels)
        self.ln2 = nn.LayerNorm(normalized_shape=2*n_channels)
    
        self.ssm1 = Mamba(
          d_model = self.n_channels, # Model dimension d_model
          out_c = self.n_channels,
          d_state=16,  # SSM state expansion factor
          d_conv=4,    # Local convolution width
          expand=2,    # Block expansion factor
      )
        self.ssm2 = Mamba(
          d_model = self.n_channels, # Model dimension d_model
          out_c = 2*self.n_channels,
          d_state=16,  # SSM state expansion factor
          d_conv=4,    # Local convolution width
          expand=2,    # Block expansion factor
      )
      
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
      
    def forward(self, x):
        b,c,h,w = x.shape
        x = x.permute(0,2,3,1) ##  [B,H,W,C]
        x = torch.flatten(x, start_dim=1,end_dim=2) ##  [B,H*W,C]
        
        x = self.ssm1(x)
        x = self.ln1(x)
        x = self.act1(x)
        x = self.ssm2(x)
        x = self.ln2(x)
        x = self.act2(x)
        
        x = x.reshape(b,h,w,2*c) 
        x = x.permute(0,3,2,1)
        return x

class Double_SSM_Block_Custom_Channel(nn.Module):
    def __init__(self, n_channels,out_channels):
        super(Double_SSM_Block_Custom_Channel, self).__init__()
        self.n_channels = n_channels
        self.out_channels = out_channels
       
        self.ln1 = nn.LayerNorm(normalized_shape=out_channels)
        self.ln2 = nn.LayerNorm(normalized_shape=out_channels)
    
        self.ssm1 = Mamba(
          d_model = self.n_channels, # Model dimension d_model
          out_c = self.out_channels,
          d_state=16,  # SSM state expansion factor
          d_conv=4,    # Local convolution width
          expand=2,    # Block expansion factor
      )
        self.ssm2 = Mamba(
          d_model = self.out_channels, # Model dimension d_model
          out_c = self.out_channels,
          d_state=16,  # SSM state expansion factor
          d_conv=4,    # Local convolution width
          expand=2,    # Block expansion factor
      )
      
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
      
    def forward(self, x):
        b,c,h,w = x.shape
        x = x.permute(0,2,3,1) ##  [B,H,W,C]
        x = torch.flatten(x, start_dim=1,end_dim=2) ##  [B,H*W,C]
        
        x = self.ssm1(x)
        x = self.ln1(x)
        x = self.act1(x)
        x = self.ssm2(x)
        x = self.ln2(x)
        x = self.act2(x)
        
        x = x.reshape(b,h,w,self.out_channels) 
        x = x.permute(0,3,2,1)
        return x

class Down(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.ssm_maxpool = nn.Sequential(
            Double_SSM_Block_Encoder(in_channels),
            nn.MaxPool2d(2)
        )
    def forward(self, x):
        x = self.ssm_maxpool(x)
        return x
        
class Double_SSM_Block_Decoder(nn.Module):
    def __init__(self, n_channels):
        super(Double_SSM_Block_Decoder, self).__init__()
        self.n_channels = n_channels
       
        self.ln1 = nn.LayerNorm(normalized_shape=n_channels)
        self.ln2 = nn.LayerNorm(normalized_shape=n_channels//3)
    
        self.ssm1 = Mamba(
          d_model = self.n_channels, # Model dimension d_model
          out_c = self.n_channels,
          d_state=16,  # SSM state expansion factor
          d_conv=4,    # Local convolution width
          expand=2,    # Block expansion factor
      )
        self.ssm2 = Mamba(
          d_model = self.n_channels, # Model dimension d_model
          out_c = self.n_channels//3,
          d_state=16,  # SSM state expansion factor
          d_conv=4,    # Local convolution width
          expand=2,    # Block expansion factor
      )
      
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
    
    def forward(self, x):
        b,c,h,w = x.shape
        x = x.permute(0,2,3,1) ##  [B,H,W,C]
        x = torch.flatten(x, start_dim=1,end_dim=2) ##  [B,H*W,C]
        
        x = self.ssm1(x)
        x = self.ln1(x)
        x = self.act1(x)
        x = self.ssm2(x)
        x = self.ln2(x)
        x = self.act2(x)
        
        x = x.reshape(b,h,w,c//3) 
        x = x.permute(0,3,2,1)
        return x

class Up(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.ssm_block = Double_SSM_Block_Decoder(in_channels+in_channels//2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def forward(self, x1,x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)  # 1024 + 512 
        x = self.ssm_block(x)
        return x  
    
class Double_SSM_Block_out(nn.Module):
    def __init__(self, n_channels,out_channels):
        super(Double_SSM_Block_out, self).__init__()
        self.n_channels = n_channels
        self.out_channels = out_channels
       
    
        self.ssm1 = Mamba(
          d_model = self.n_channels, # Model dimension d_model
          out_c = self.out_channels,
          d_state=16,  # SSM state expansion factor
          d_conv=4,    # Local convolution width
          expand=2,    # Block expansion factor
      )

      
    def forward(self, x):
        b,c,h,w = x.shape
        x = x.permute(0,2,3,1) ##  [B,H,W,C]
        x = torch.flatten(x, start_dim=1,end_dim=2) ##  [B,H*W,C]
        
        x = self.ssm1(x)
        
        x = x.reshape(b,h,w,self.out_channels) 
        x = x.permute(0,3,2,1)
        return x
    
    
class UNet(nn.Module):
    def __init__(self, n_channels=1,):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        
        self.inc = Double_SSM_Block_Custom_Channel(1,32)
        self.down0 = Down(32)
        self.down1 = Down(64)
        self.down2 = Down(128)
        self.down3 = Down(256)
        self.down4 = Down(512)
        
        self.up1 = Up(1024)
        self.up2 = Up(512)
        self.up3 = Up(256)
        self.up4 = Up(128)
        self.up5 = Up(64)
        
        self.outc =Double_SSM_Block_out(32,4)

    def forward(self, inp):        
        x0 = self.inc(inp)        
        x1 = self.down0(x0)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        z = self.up1(x5,x4)
        z = self.up2(z,x3)
        z = self.up3(z,x2)
        z = self.up4(z,x1)
        z = self.up5(z,x0)
        z = self.outc(z)
        return z


#DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#def model() -> UNet:
#    model = UNet()
#    model.to(device=DEVICE,dtype=torch.float)
#    return model
#from torchsummary import summary
#model = model()
#summary(model, [(1,256,256)])

