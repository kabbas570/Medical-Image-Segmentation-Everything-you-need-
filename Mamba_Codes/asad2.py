import torch 
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

  
class Linear_Layer_Ch(nn.Module):
    def __init__(self, n_channels,out_channels):
        super(Linear_Layer_Ch, self).__init__()
        self.n_channels = n_channels
        self.out_channels = out_channels

        self.linear1 = nn.Linear(self.n_channels,self.out_channels)
        self.drop = nn.Dropout(p=0.1)

    def forward(self, x1):
    
        b,c,h,w = x1.shape
        x1 = x1.permute(0,2,3,1) 
        x1 = torch.flatten(x1, start_dim=1,end_dim=2) ##  [B,H*W,C]
        
        x1 = self.linear1(x1)
        x1 = self.drop(x1)
         
        x1 = x1.reshape(b,h,w,self.out_channels) 
        x1 = x1.permute(0,3,1,2)
        return x1
    

class Linear_Layer_Spa(nn.Module):
    def __init__(self, size,in_channels,out_channels, sp_dim1 = None ,sp_dim2= None):
        super(Linear_Layer_Spa, self).__init__()
        self.sp_dim1 = sp_dim1
        self.sp_dim2 = sp_dim2
        self.size = size

        self.linear1 = nn.Linear(self.size*self.size,self.size*self.size)
        self.linear_ch = Linear_Layer_Ch(in_channels, out_channels) ## This will increase the no.channels to be added with other branches 
        self.drop = nn.Dropout(p=0.1)

    def forward(self, x1):
        b,c,h,w = x1.shape
        
        x1 = F.interpolate(x1, size = (self.size,self.size), mode='bilinear', align_corners=True)
    
        x1 = torch.flatten(x1, start_dim=2,end_dim=3)
        
        x1 = self.linear1(x1)
        x1 = self.drop(x1)
         
        x1 = x1.reshape(b,c,self.size,self.size)   
        x1 = F.interpolate(x1, size = (h,w), mode='bilinear', align_corners=True)
        x1 = self.linear_ch(x1)
        return x1
    
class SSM_spa(nn.Module):
    def __init__(self,size, in_channels, out_channels, sp_dim1 = None ,sp_dim2= None):
        super(SSM_spa, self).__init__()
        
        self.sp_dim2 = sp_dim2
        self.size = size
        
        self.ssm2 = Mamba(
          d_model = self.size*self.size,
          out_c = self.size*self.size,
          d_state=16,  
          expand=2)
        self.drop = nn.Dropout(p=0.1)
        self.linear_ch = Linear_Layer_Ch(in_channels, out_channels)
        self.norm = nn.LayerNorm(normalized_shape=self.size*self.size)
        
    def forward(self, x1):
        
        b,c,h,w = x1.shape
        
        x1 = F.interpolate(x1, size = (self.size,self.size), mode='bilinear', align_corners=True)
        
        x1 = torch.flatten(x1, start_dim=2,end_dim=3) ##  [B,H*W,C]
                
        x1 = self.ssm2(x1)
        x1 = self.norm(x1)
        x1 = self.drop(x1)
         
        x1 = x1.reshape(b,c,self.size,self.size) 
        x1 = F.interpolate(x1, size = (h,w), mode='bilinear', align_corners=True)
        x1 = self.linear_ch(x1)
        return x1
    
class SSM_cha(nn.Module):
    def __init__(self, n_channels,out_channels):
        super(SSM_cha, self).__init__()
        self.n_channels = n_channels
        self.out_channels = out_channels

        self.ssm1 = Mamba(
          d_model = self.n_channels,
          out_c = self.out_channels,
          d_state=16,  
          expand=2,
      )
       
        self.ssm2 = Mamba(
          d_model = self.out_channels,
          out_c = self.out_channels,
          d_state=16,  
          expand=2,
      )
        self.drop1 = nn.Dropout(p=0.1)
        self.drop2 = nn.Dropout(p=0.1)
        self.norm1 = nn.LayerNorm(normalized_shape=out_channels)
        self.norm2 = nn.LayerNorm(normalized_shape=out_channels)

    def forward(self, x1):
    
        b,c,h,w = x1.shape #  (0,1,2,3)
        x1 = x1.permute(0,2,3,1) ##  [B,H,W,C]
        x1 = torch.flatten(x1, start_dim=1,end_dim=2) ##  [B,H*W,C]
        
        x1 = self.ssm1(x1)
        x1 = self.norm1(x1)
        x1 = self.drop1(x1)
        
        x1 = self.ssm2(x1)
        x1 = self.norm2(x1)
        x1 = self.drop2(x1)
        
         
        x1 = x1.reshape(b,h,w,self.out_channels) 
        x1 = x1.permute(0,3,1,2)
        return x1  
    

# input_tensor = torch.randn(1, 128, 64, 64)
# reducer = Resize_Spa(size=(128, 128)) 
# output_tensor = reducer(input_tensor)

class Branch1234(nn.Module):
    def __init__(self, size , in_channels, out_channels):
        super().__init__()
        self.linear_ch = Linear_Layer_Ch(in_channels, out_channels)
        self.ssm_cha = SSM_cha(in_channels, out_channels)
        self.ssm_spa = SSM_spa(size , in_channels, out_channels)
        self.linear_spa = Linear_Layer_Spa(size , in_channels, out_channels)
        
        self.drop = nn.Dropout(p=0.1)
           
    def forward(self, x):
        x1 = self.linear_ch(x)
        x2 = self.ssm_cha(x)
        x3 = self.ssm_spa(x)
        x4 = self.linear_spa(x)
        return self.drop(x1+x2+x3+x4) 
    
    
class Down(nn.Module):
    def __init__(self, size , in_channels, out_channels):
        super().__init__()
        self.avgpool_br1234 = nn.Sequential(
            nn.AvgPool2d(2),
            Branch1234(size , in_channels, out_channels)
        )
    def forward(self, x):
        return self.avgpool_br1234(x)

class Up(nn.Module):
    def __init__(self,size, in_channels, out_channels):
        super().__init__()
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.br1234 = Branch1234(size , in_channels+in_channels//2, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.br1234(x)
    
class UNet(nn.Module):
    def __init__(self, n_channels=1):
        super(UNet, self).__init__()
        self.n_channels = n_channels

        self.inc = Branch1234(40, 1,64)   # size, in_channels, out_channels
        self.down1 = Down(40, 64,128)
        self.down2 = Down(20, 128,256)
        self.down3 = Down(10, 256,512)
        self.down4 = Down(10, 512,1024)
        self.up1 = Up(10, 1024,512)
        self.up2 = Up(10, 512,256)
        self.up3 = Up(20, 256,128)
        self.up4 = Up(40, 128,64)
        self.outc = Branch1234(40, 64, 5)
       
        
    def forward(self, inp):

        x1 = self.inc(inp)        
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5,x4)
        x = self.up2(x,x3)
        x = self.up3(x,x2)
        x = self.up4(x,x1)
        x = self.outc(x)
        return x
        
        

#DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#def model() -> UNet:
#    model = UNet()
#    model.to(device=DEVICE,dtype=torch.float)
#    return model
#from torchsummary import summary
#model = model()
#summary(model, [(1, 160,160)])
