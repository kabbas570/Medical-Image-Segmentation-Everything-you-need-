import torch 
import torch.nn as nn
from mamba_ssm import Mamba

class Linear_Layer(nn.Module):
    def __init__(self, n_channels,out_channels):
        super(Linear_Layer, self).__init__()
        self.n_channels = n_channels
        self.out_channels = out_channels

        self.linear1 = nn.Linear(self.n_channels,self.out_channels)
        self.drop = nn.Dropout(p=0.1)

    def forward(self, x1):
    
        b,c,h,w = x1.shape
        x1 = x1.permute(0,2,3,1) ##  [B,H,W,C]
        x1 = torch.flatten(x1, start_dim=1,end_dim=2) ##  [B,H*W,C]
        
        x1 = self.linear1(x1)
        x1 = self.drop(x1)
         
        x1 = x1.reshape(b,h,w,self.out_channels) 
        x1 = x1.permute(0,3,1,2)
        return x1

class Linear_Layer_Last(nn.Module):
    def __init__(self, n_channels,out_channels):
        super(Linear_Layer_Last, self).__init__()
        self.n_channels = n_channels
        self.out_channels = out_channels

        self.linear1 = nn.Linear(self.n_channels,self.out_channels)

    def forward(self, x1):
    
        b,c,h,w = x1.shape
        x1 = x1.permute(0,2,3,1) ##  [B,H,W,C]
        x1 = torch.flatten(x1, start_dim=1,end_dim=2) ##  [B,H*W,C]
        
        x1 = self.linear1(x1)
         
        x1 = x1.reshape(b,h,w,self.out_channels) 
        x1 = x1.permute(0,3,1,2)
        return x1
    
    
class SSM_spa(nn.Module):
    def __init__(self, sp_dim1,sp_dim2):
        super(SSM_spa, self).__init__()
        
        self.sp_dim2 = sp_dim2
        self.ssm2 = Mamba(
          d_model = sp_dim1*sp_dim1,
          out_c = sp_dim2*sp_dim2,
          d_state=16,  
          expand=2)
        
        self.norm = nn.LayerNorm(normalized_shape=sp_dim2*sp_dim2)

    def forward(self, x1):
        
        b,c,h,w = x1.shape
        x1 = torch.flatten(x1, start_dim=2,end_dim=3) ##  [B,H*W,C]
                
        x1 = self.ssm2(x1)
        x1 = self.norm(x1) 
         
        x1 = x1.reshape(b,c,self.sp_dim2,self.sp_dim2) 
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

        self.norm = nn.LayerNorm(normalized_shape=out_channels)

    def forward(self, x1):
    
        b,c,h,w = x1.shape
        x1 = x1.permute(0,2,3,1) ##  [B,H,W,C]
        x1 = torch.flatten(x1, start_dim=1,end_dim=2) ##  [B,H*W,C]
        
        x1 = self.ssm1(x1)
        x1 = self.norm(x1) 
         
        x1 = x1.reshape(b,h,w,self.out_channels) 
        x1 = x1.permute(0,3,1,2)
        return x1


class SSM_cha_Last(nn.Module):
    def __init__(self, n_channels,out_channels):
        super(SSM_cha_Last, self).__init__()
        self.n_channels = n_channels
        self.out_channels = out_channels

        self.ssm1 = Mamba(
          d_model = self.n_channels,
          out_c = self.out_channels,
          d_state=16,  
          expand=2,
      )

    def forward(self, x1):
    
        b,c,h,w = x1.shape
        x1 = x1.permute(0,2,3,1) ##  [B,H,W,C]
        x1 = torch.flatten(x1, start_dim=1,end_dim=2) ##  [B,H*W,C]
        
        x1 = self.ssm1(x1)
         
        x1 = x1.reshape(b,h,w,self.out_channels) 
        x1 = x1.permute(0,3,1,2)
        return x1
    
    
class Branch_3(nn.Module):
    def __init__(self,in_channels, out_channels,sp_dim1,sp_dim2):
        super().__init__()
        self.branch3 = nn.Sequential(
            SSM_spa(sp_dim1,sp_dim2),
            Linear_Layer(in_channels, out_channels),
        )
    def forward(self, x):
        return  self.branch3(x)
    
    
class Branch_2(nn.Module):
    def __init__(self,in_channels, out_channels):
        super().__init__()
        self.branch2 = nn.Sequential(
            SSM_cha(in_channels, out_channels),
            Linear_Layer(out_channels, out_channels),
            SSM_cha(out_channels, out_channels), 
            nn.Dropout(p=0.1),
        )
    def forward(self, x):
        return  self.branch2(x)
    
class Branch_2_Last(nn.Module):
    def __init__(self,in_channels, out_channels):
        super().__init__()
        self.branch2 = nn.Sequential(
            SSM_cha_Last(in_channels, out_channels),
            Linear_Layer_Last(out_channels, out_channels),
            SSM_cha_Last(out_channels, out_channels), 
        )
    def forward(self, x):
        return  self.branch2(x)
    
class Branch12_Last(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = Linear_Layer_Last(in_channels, out_channels)
        self.br2 = Branch_2_Last(in_channels, out_channels)
           
    def forward(self, x):
        x1 = self.linear(x)
        x2 = self.br2(x)
        return (x1+x2)
    
    
class Branch12(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = Linear_Layer(in_channels, out_channels)
        self.br2 = Branch_2(in_channels, out_channels)
        self.drop = nn.Dropout(p=0.1)
           
    def forward(self, x):
        x1 = self.linear(x)
        x2 = self.br2(x)
        return self.drop(x1+x2)
    
class Branch123(nn.Module):
    def __init__(self, in_channels, out_channels, sp_dim1,sp_dim2):
        super().__init__()
        self.linear = Linear_Layer(in_channels, out_channels)
        self.br2 = Branch_2(in_channels, out_channels)
        self.br3 = Branch_3(in_channels, out_channels, sp_dim1,sp_dim2)
        
        self.drop = nn.Dropout(p=0.1)
           
    def forward(self, x):
        x1 = self.linear(x)
        x2 = self.br2(x)
        x3 = self.br3(x)
        return self.drop(x1+x2+x3)   
    
     
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_br12 = nn.Sequential(
            nn.AvgPool2d(2),
            Branch12(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_br12(x)
    
class Down_1(nn.Module):
    def __init__(self, in_channels, out_channels, sp_dim1,sp_dim2):
        super().__init__()
        self.maxpool_br123 = nn.Sequential(
            nn.AvgPool2d(2),
            Branch123(in_channels, out_channels, sp_dim1,sp_dim2)
        )
    def forward(self, x):
        return self.maxpool_br123(x)
    
class Up_1(nn.Module):
    def __init__(self, in_channels, out_channels,sp_dim1,sp_dim2):
        super().__init__()
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.br123 = Branch123(in_channels+in_channels//2, out_channels,sp_dim1,sp_dim2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.br123(x)
    
class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.br12 = Branch12(in_channels+in_channels//2, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.br12(x)
    

class UNet(nn.Module):
    def __init__(self, n_channels=1):
        super(UNet, self).__init__()
        self.n_channels = n_channels

        self.inc = Branch12(1,64)
        self.down1 = Down(64,128)
        self.down2 = Down(128,256)
        self.down3 = Down_1(256,512,20,20)
        self.down4 = Down_1(512,1024,10,10)
        self.up1 = Up_1(1024,512,20,20)
        self.up2 = Up(512,256)
        self.up3 = Up(256,128)
        self.up4 = Up(128,64)
        self.outc = Branch12_Last(64, 5)
       
        
    def forward(self, inp):

        x1 = self.inc(inp)        
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
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
#
#model = UNet()
#num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
#print("Number of trainable parameters:", num_parameters)

## 38 Linear+norm
## 40 Linear only 
##  617 wiht AdamW and only linear 
## 627 linear between two SSM blocks and adamW + 638 wiht BS = 16
# 799 SGD with high LR and 1000 epochs
## 812 AdamW with conv2d as alsy layer

## 471 --> BS = 32 and AdamW
## 473,  --> BS = 16 with double param expand and state 
## 478 --> BS = 16, single param but conv1d in mamba
