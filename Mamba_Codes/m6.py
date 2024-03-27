import torch 
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from timm.models.layers import to_2tuple


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=256, patch_size=2, stride=2, in_chans=1, embed_dim=64, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = ((img_size[0] - patch_size[0]) // stride + 1, (img_size[1] - patch_size[1]) // stride + 1)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x
    
class Double_SSM_Block_Custom_Channel(nn.Module):
    def __init__(self, n_channels,out_channels):
        super(Double_SSM_Block_Custom_Channel, self).__init__()
        self.n_channels = n_channels
        self.out_channels = out_channels

        self.ssm1 = Mamba(
          d_model = self.n_channels,
          out_c = self.out_channels,
          d_state=16,  
          d_conv=4,   
          expand=2,
      )

        self.ln2 = nn.LayerNorm(normalized_shape=self.out_channels)

    def forward(self, x1):
    
        b,c,h,w = x1.shape
        x1 = x1.permute(0,2,3,1) ##  [B,H,W,C]
        x1 = torch.flatten(x1, start_dim=1,end_dim=2) ##  [B,H*W,C]
        
        x1 = self.ssm1(x1)
        x1 = self.ln2(x1)
         
        x1 = x1.reshape(b,h,w,self.out_channels) 
        x1 = x1.permute(0,3,1,2)
        return x1
    
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            Double_SSM_Block_Custom_Channel(in_channels, out_channels),
            Double_SSM_Block_Custom_Channel(out_channels, out_channels)
        )
        

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels+in_channels//2, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = Double_SSM_Block_Custom_Channel(in_channels,out_channels)

    def forward(self, x):
        return self.conv(x)
    
    
def Reshape_1(x):
    b,L,C = x.shape
    x = x.reshape(b,128,128,C) 
    x = x.permute(0,3,1,2)
    return x 

class UNet(nn.Module):
    def __init__(self, n_channels=1):
        super(UNet, self).__init__()
        self.n_channels = n_channels

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512 )
        self.up2 = Up(512, 256 )
        self.up3 = Up(256, 128 )
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, 4)
        
        self.PatchEmbed_1 = PatchEmbed()        
        self.pos_embed = nn.Parameter(torch.zeros(1, 128*128, 64))
        self.pos_drop = nn.Dropout(p=0.1)

    def forward(self, inp):
        inp = self.PatchEmbed_1(inp)
        inp = inp + self.pos_embed 
        inp = self.pos_drop(inp)
        inp = Reshape_1(inp)

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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
def model() -> UNet:
    model = UNet()
    model.to(device=DEVICE,dtype=torch.float)
    return model
from torchsummary import summary
model = model()
summary(model, [(1, 256,256)])
